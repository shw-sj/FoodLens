"""Train ResNet-18+GRU or ViT+Transformer on Food-500 Cap.

Run from the project root::

    python -m src.task1_caption.train --config configs/task1_resnet18_gru.yaml
    python -m src.task1_caption.train --config configs/task1_vit_transformer.yaml

Useful overrides for a first smoke test::

    python -m src.task1_caption.train \
        --config configs/task1_resnet18_gru.yaml \
        --limit-train 32 --limit-val 16 --epochs 1 \
        --set model.pretrained=false --set train.amp=false

Checkpoint selection uses the validation split (``train.monitor``, default
CIDEr-D) and ``best.pt`` / ``last.pt`` are written under
``train.save_dir``.  A CSV with one row per epoch is written next to them.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from src.common.io import (
    apply_overrides,
    ensure_dir,
    load_yaml,
    resolve_path,
    save_json,
    save_yaml,
)
from src.common.logging import MetricLogger, get_logger, log_config
from src.common.metrics import compute_all
from src.common.seed import set_seed
from src.task1_caption.beam_search import greedy_decode
from src.task1_caption.datasets import (
    EOS_ID,
    PAD_ID,
    SOS_ID,
    Vocab,
    load_split,
    make_dataloader,
)
from src.task1_caption.evaluate import load_protocol, metric_kwargs_from_protocol
from src.task1_caption.models import build_model


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(resolve_path(".")),
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _amp_components(device: torch.device, enabled: bool):
    """Return (autocast_factory, scaler) with a torch-version-safe scaler."""
    enabled = bool(enabled) and device.type == "cuda"

    def autocast():
        return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)

    if enabled:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        try:
            scaler = torch.amp.GradScaler("cpu", enabled=False)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=False)
    return autocast, scaler


def _configure_freeze(
    model: torch.nn.Module,
    epoch: int,
    freeze_epochs: int,
    mode: str,
) -> None:
    """Set requires_grad for the encoder before each epoch."""
    if freeze_epochs <= 0:
        return
    if epoch < freeze_epochs:
        model.set_encoder_trainable(False)
    elif mode == "all":
        model.set_encoder_trainable(True)
    elif mode == "last_block":
        model.set_encoder_trainable(False)
        model.unfreeze_last_encoder_block()
    else:  # "none", "frozen"
        model.set_encoder_trainable(False)


def _set_train_modes(
    model: torch.nn.Module,
    epoch: int,
    freeze_epochs: int,
    mode: str,
) -> None:
    """Keep frozen BN / dropout in eval mode (ResNet BN matters here)."""
    model.train()
    if freeze_epochs <= 0:
        return
    if mode == "all" and epoch >= freeze_epochs:
        return
    model.encoder_module().eval()
    if mode == "last_block" and epoch >= freeze_epochs:
        model.last_encoder_block().train()


def _train_epoch(
    model: torch.nn.Module,
    loader,
    optimizer,
    scheduler,
    scaler,
    autocast,
    device: torch.device,
    grad_accum: int,
    grad_clip: float,
    label_smoothing: float,
    logger,
) -> Tuple[float, int]:
    # The caller already put the model in train mode and re-applied the
    # frozen-encoder eval() flags; do not call model.train() here.
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc="train", leave=False)
    for step, batch in enumerate(progress):
        images = batch["images"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        target_ids = batch["target_ids"].to(device, non_blocking=True)

        with autocast():
            logits = model(images, input_ids)
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=PAD_ID,
                label_smoothing=float(label_smoothing),
            )

        scaler.scale(loss / grad_accum).backward()
        last_batch = step == len(loader) - 1
        if (step + 1) % grad_accum == 0 or last_batch:
            scaler.unscale_(optimizer)
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += float(loss.item())
        n_batches += 1
        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )
    progress.close()
    return total_loss / max(1, n_batches), n_batches


@torch.no_grad()
def _validate_loss(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    autocast,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in tqdm(loader, desc="val-loss", leave=False):
        images = batch["images"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        target_ids = batch["target_ids"].to(device, non_blocking=True)
        with autocast():
            logits = model(images, input_ids)
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=PAD_ID,
            )
        batch_n = images.size(0)
        total += float(loss.item()) * batch_n
        count += batch_n
    return total / max(1, count)


@torch.no_grad()
def _generate_val(
    model: torch.nn.Module,
    loader,
    vocab: Vocab,
    max_len: int,
    device: torch.device,
    autocast,
) -> Tuple[List[str], List[List[str]]]:
    model.eval()
    if hasattr(model, "max_len"):
        max_len = min(max_len, int(model.max_len))
    predictions: List[str] = []
    references: List[List[str]] = []
    for batch in tqdm(loader, desc="val-gen", leave=False):
        images = batch["images"].to(device, non_blocking=True)
        with autocast():
            token_matrix = greedy_decode(
                model, images, max_len=max_len, sos_id=SOS_ID, eos_id=EOS_ID
            )
        predictions.extend(
            vocab.decode(row) for row in token_matrix.detach().cpu().tolist()
        )
        references.extend([list(refs) for refs in batch["references"]])
    return predictions, references


def _checkpoint_payload(
    model,
    vocab,
    cfg,
    epoch,
    metrics,
    monitor,
    monitor_value,
    best_epoch,
    optimizer,
    scheduler,
    seed,
    deterministic,
) -> Dict[str, Any]:
    return {
        "format_version": 1,
        "model_name": cfg["model"]["name"],
        "model_config": cfg["model"],
        "data_config": cfg["data"],
        "train_config": cfg["train"],
        "config": cfg,
        "vocab": vocab.to_dict(),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "val_metrics": metrics,
        "monitor": monitor,
        "monitor_value": monitor_value,
        "best_epoch": int(best_epoch),
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "git_commit": _git_commit(),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }


def _save_checkpoint(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train a task-1 caption model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    cfg = apply_overrides(load_yaml(args.config), args.set)
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg.setdefault("train", {})["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        cfg.setdefault("data", {})["num_workers"] = int(args.num_workers)

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    deterministic = bool(cfg.get("deterministic", False))
    set_seed(seed, deterministic=deterministic)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    experiment = str(cfg.get("experiment", model_cfg.get("name", "task1")))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = resolve_path(train_cfg.get("log_dir", "logs/task1"))
    assert log_dir is not None
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(
        f"task1.{experiment}",
        log_file=str(log_dir / f"{experiment}_{timestamp}.log"),
    )
    log_config(logger, "config", cfg)
    logger.info("device=%s | seed=%d | deterministic=%s", device, seed, deterministic)

    # -- data -------------------------------------------------------------
    vocab = Vocab.from_file(data_cfg["vocab"])
    logger.info("vocab: %d tokens (%s)", len(vocab), vocab.meta.get("tokenizer"))
    max_len = int(data_cfg.get("max_len", 40))
    image_size = int(data_cfg.get("image_size", 224))
    num_workers = int(data_cfg.get("num_workers", 4))

    train_ds = load_split(
        manifest=data_cfg["manifest"],
        ids_file=data_cfg["train_ids"],
        vocab=vocab,
        image_root=data_cfg.get("image_root"),
        image_size=image_size,
        max_len=max_len,
        train=True,
        augment=bool(data_cfg.get("augment", True)),
        limit=args.limit_train,
    )
    val_ds = load_split(
        manifest=data_cfg["manifest"],
        ids_file=data_cfg["val_ids"],
        vocab=vocab,
        image_root=data_cfg.get("image_root"),
        image_size=image_size,
        max_len=max_len,
        train=False,
        augment=False,
        limit=args.limit_val,
    )
    if len(train_ds) == 0:
        raise ValueError("training split is empty; check data.train_ids and --limit-train")
    if len(val_ds) == 0:
        raise ValueError("validation split is empty; check data.val_ids")

    batch_size = int(train_cfg.get("batch_size", 32))
    grad_accum = int(train_cfg.get("grad_accum_steps", 1))
    train_loader = make_dataloader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = make_dataloader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    logger.info(
        "train=%d images (%d batches) | val=%d images (%d batches) | "
        "batch=%d x accum=%d | max_len=%d",
        len(train_ds), len(train_loader), len(val_ds), len(val_loader),
        batch_size, grad_accum, max_len,
    )

    # -- model / optimiser ------------------------------------------------
    model = build_model(cfg, len(vocab), max_len)
    model.to(device)

    freeze_epochs = int(model_cfg.get("freeze_encoder_epochs", 0))
    unfreeze_mode = str(model_cfg.get("unfreeze_mode", "last_block")).lower()
    if freeze_epochs > 0:
        model.set_encoder_trainable(False)
        logger.info(
            "encoder frozen for %d epoch(s); afterwards mode=%s",
            freeze_epochs, unfreeze_mode,
        )

    total_params, trainable_params = model.trainable_parameter_counts()
    logger.info(
        "model=%s | parameters=%.2fM (initially trainable %.2fM)",
        model_cfg["name"], total_params / 1e6, trainable_params / 1e6,
    )

    epochs = int(train_cfg.get("epochs", 30))
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-2))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, grad_accum)))
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = max(
        0,
        int(round(float(train_cfg.get("warmup_epochs", 0)) * steps_per_epoch)),
    )

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    autocast, scaler = _amp_components(
        device, bool(train_cfg.get("amp", True))
    )

    save_dir = resolve_path(train_cfg.get("save_dir", f"checkpoints/task1/{model_cfg['name']}"))
    assert save_dir is not None
    ensure_dir(str(save_dir))
    save_yaml(cfg, str(save_dir / "config.yaml"))
    metric_logger = MetricLogger(str(save_dir / "metrics.csv"))

    monitor = str(train_cfg.get("monitor", "cider_d"))
    monitor_mode = str(train_cfg.get("monitor_mode", "min" if monitor == "val_loss" else "max")).lower()
    patience = int(train_cfg.get("patience", 0))
    eval_every = max(1, int(train_cfg.get("eval_every", 1)))
    protocol = load_protocol(cfg.get("eval", {}).get("protocol", "configs/eval_protocol.yaml"))
    metric_kwargs = metric_kwargs_from_protocol(protocol)

    start_epoch = 0
    best_value = float("inf") if monitor_mode == "min" else float("-inf")
    best_epoch = -1
    if args.resume:
        resume_path = resolve_path(args.resume)
        assert resume_path is not None
        try:
            resumed = torch.load(resume_path, map_location="cpu", weights_only=False)
        except TypeError:
            resumed = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(resumed["state_dict"])
        if "optimizer" in resumed:
            optimizer.load_state_dict(resumed["optimizer"])
        if "scheduler" in resumed:
            scheduler.load_state_dict(resumed["scheduler"])
        start_epoch = int(resumed.get("epoch", -1)) + 1
        best_value = float(resumed.get("monitor_value", best_value))
        best_epoch = int(resumed.get("best_epoch", -1))
        logger.info("resumed from %s at epoch %d", resume_path, start_epoch)

    logger.info(
        "monitor=%s (%s) | eval_every=%d | patience=%d | save_dir=%s",
        monitor, monitor_mode, eval_every, patience, save_dir,
    )

    if args.dry_run:
        logger.info("dry run: data and model are ready, exiting without training")
        return 0

    epochs_without_improvement = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        _configure_freeze(model, epoch, freeze_epochs, unfreeze_mode)
        _set_train_modes(model, epoch, freeze_epochs, unfreeze_mode)
        if epoch == freeze_epochs and freeze_epochs > 0:
            _total, _trainable = model.trainable_parameter_counts()
            logger.info(
                "epoch %d: encoder mode=%s, trainable parameters=%.2fM",
                epoch + 1, unfreeze_mode, _trainable / 1e6,
            )

        train_loss, n_batches = _train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            autocast=autocast,
            device=device,
            grad_accum=grad_accum,
            grad_clip=float(train_cfg.get("grad_clip", 5.0)),
            label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
            logger=logger,
        )
        val_loss = _validate_loss(model, val_loader, device, autocast)

        metrics: Dict[str, float] = {"val_loss": float(val_loss)}
        did_eval = ((epoch + 1) % eval_every == 0) or (epoch == epochs - 1)
        if did_eval:
            predictions, references = _generate_val(
                model, val_loader, vocab, max_len, device, autocast
            )
            metrics.update(compute_all(predictions, references, **metric_kwargs))

        row = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": round(time.time() - epoch_start, 1),
            **{key: round(value, 6) for key, value in metrics.items()},
        }
        metric_logger.log(row)
        history.append(row)

        if did_eval:
            if monitor not in metrics:
                raise KeyError(
                    f"monitor={monitor!r} is not in validation metrics: "
                    f"{sorted(metrics)}"
                )
            value = float(metrics[monitor])
            better = value < best_value if monitor_mode == "min" else value > best_value
            if better:
                best_value = value
                best_epoch = epoch
                epochs_without_improvement = 0
                payload = _checkpoint_payload(
                    model=model, vocab=vocab, cfg=cfg, epoch=epoch, metrics=metrics,
                    monitor=monitor, monitor_value=value, best_epoch=best_epoch,
                    optimizer=optimizer, scheduler=scheduler, seed=seed,
                    deterministic=deterministic,
                )
                _save_checkpoint(payload, save_dir / "best.pt")
                logger.info(
                    "epoch %d | train %.4f | val %.4f | BLEU-4 %.2f | ROUGE-L %.2f | "
                    "CIDEr-D %.4f | %s=%.4f -> new best",
                    epoch + 1, train_loss, val_loss, metrics.get("bleu4", float("nan")),
                    metrics.get("rouge_l", float("nan")), metrics.get("cider_d", float("nan")),
                    monitor, value,
                )
            else:
                epochs_without_improvement += 1
                logger.info(
                    "epoch %d | train %.4f | val %.4f | BLEU-4 %.2f | ROUGE-L %.2f | "
                    "CIDEr-D %.4f | %s=%.4f (best %.4f @ epoch %d)",
                    epoch + 1, train_loss, val_loss, metrics.get("bleu4", float("nan")),
                    metrics.get("rouge_l", float("nan")), metrics.get("cider_d", float("nan")),
                    monitor, value, best_value, best_epoch + 1,
                )
        else:
            logger.info(
                "epoch %d | train %.4f | val %.4f (generation skipped)",
                epoch + 1, train_loss, val_loss,
            )

        payload = _checkpoint_payload(
            model=model, vocab=vocab, cfg=cfg, epoch=epoch, metrics=metrics,
            monitor=monitor, monitor_value=best_value, best_epoch=best_epoch,
            optimizer=optimizer, scheduler=scheduler, seed=seed,
            deterministic=deterministic,
        )
        _save_checkpoint(payload, save_dir / "last.pt")

        if patience > 0 and epochs_without_improvement >= patience:
            logger.info(
                "early stopping after %d epochs without improvement (best epoch %d)",
                epochs_without_improvement, best_epoch + 1,
            )
            break

    summary = {
        "experiment": experiment,
        "config_path": str(resolve_path(args.config)),
        "model": model_cfg["name"],
        "seed": seed,
        "deterministic": deterministic,
        "device": str(device),
        "monitor": monitor,
        "monitor_mode": monitor_mode,
        "best_epoch": best_epoch + 1,
        "best_value": best_value,
        "epochs_ran": len(history),
        "checkpoint_best": str(save_dir / "best.pt"),
        "checkpoint_last": str(save_dir / "last.pt"),
        "metrics_csv": str(save_dir / "metrics.csv"),
        "log_file": str(log_dir / f"{experiment}_{timestamp}.log"),
        "history": history,
    }
    save_json(summary, str(save_dir / "summary.json"))
    logger.info("best epoch=%d | %s=%.6f", best_epoch + 1, monitor, best_value)
    logger.info("best checkpoint: %s", save_dir / "best.pt")
    logger.info("summary: %s", save_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
