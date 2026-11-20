"""Generate captions for a checkpoint and write the frozen prediction CSV.

Prediction contract (also used by the SCST hand-off)::

    image_id,prediction,reference

``reference`` is the concatenation of all references for that image with the
separator ``" ||| "``.  The file therefore contains everything ``evaluate.py``
needs, and downstream owners can score their own outputs with the same
script.

Examples::

    python -m src.task1_caption.infer \
        --checkpoint checkpoints/task1/vit_transformer/best.pt \
        --split test --method greedy

    python -m src.task1_caption.infer \
        --checkpoint checkpoints/task1/vit_transformer/best.pt \
        --split test --method beam --beam-size 5 --length-penalty 1.0
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from tqdm import tqdm

from src.common.io import ensure_dir, load_ids, resolve_path
from src.common.metrics import average_length, repetition_rate
from src.task1_caption.beam_search import beam_search_decode, greedy_decode
from src.task1_caption.datasets import (
    EOS_ID,
    SOS_ID,
    CaptionDataset,
    Vocab,
    make_dataloader,
)
from src.task1_caption.models import build_model


def resolve_device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _torch_load(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch
        return torch.load(path, map_location="cpu")


def load_bundle(checkpoint: str, device: Optional[str] = None) -> Dict[str, Any]:
    """Load checkpoint, vocabulary and model into one dict."""
    checkpoint_path = resolve_path(checkpoint)
    assert checkpoint_path is not None
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    ckpt = _torch_load(str(checkpoint_path))
    if "state_dict" not in ckpt or "model_config" not in ckpt:
        raise ValueError(
            f"{checkpoint_path} is not a task-1 checkpoint (missing state_dict/model_config)"
        )
    config = ckpt.get("config", {})
    data_config = ckpt.get("data_config") or config.get("data", {})
    model_config = ckpt["model_config"]
    max_len = int(data_config.get("max_len", 40))

    if "vocab" in ckpt:
        vocab = Vocab.from_dict(ckpt["vocab"])
    elif data_config.get("vocab"):
        vocab = Vocab.from_file(data_config["vocab"])
    else:
        raise ValueError("checkpoint has no vocabulary and data_config.vocab is missing")

    model = build_model({"model": model_config}, len(vocab), max_len)
    model.load_state_dict(ckpt["state_dict"])
    dev = resolve_device(device)
    model.to(dev)
    model.eval()

    return {
        "checkpoint": str(checkpoint_path),
        "ckpt": ckpt,
        "model": model,
        "vocab": vocab,
        "model_name": model_config.get("name", "model"),
        "model_config": model_config,
        "data_config": data_config,
        "config": config,
        "device": dev,
        "max_len": max_len,
    }


def build_split_dataset(
    bundle: Dict[str, Any],
    split: str = "test",
    manifest: Optional[str] = None,
    image_root: Optional[str] = None,
    ids_file: Optional[str] = None,
    limit: Optional[int] = None,
) -> CaptionDataset:
    data_cfg = bundle["data_config"]
    split = "val" if split in ("valid", "validation") else split
    if ids_file is None:
        ids_path = data_cfg.get(f"{split}_ids")
        if not ids_path:
            raise KeyError(
                f"data_config has no '{split}_ids'; available keys: "
                + ", ".join(k for k in data_cfg if k.endswith("_ids"))
            )
    else:
        ids_path = ids_file

    ids = load_ids(ids_path)
    if limit is not None and int(limit) > 0:
        ids = ids[: int(limit)]

    manifest_path = manifest or data_cfg.get("manifest")
    if not manifest_path:
        raise KeyError("data_config.manifest is missing; pass --manifest")
    image_root_path = image_root or data_cfg.get("image_root")

    return CaptionDataset(
        manifest=manifest_path,
        ids=ids,
        vocab=bundle["vocab"],
        image_root=image_root_path,
        image_size=int(data_cfg.get("image_size", 224)),
        max_len=int(data_cfg.get("max_len", 40)),
        train=False,
        augment=False,
    )


@torch.no_grad()
def run_inference(
    bundle: Dict[str, Any],
    dataset: CaptionDataset,
    method: str = "greedy",
    max_len: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    beam_size: int = 5,
    length_penalty: float = 1.0,
    min_len: int = 0,
) -> Dict[str, Any]:
    """Run greedy or beam decoding and return decoded strings + timing."""
    model = bundle["model"]
    vocab = bundle["vocab"]
    device = bundle["device"]
    max_len = int(max_len or bundle["max_len"])
    if hasattr(model, "max_len"):
        max_len = min(max_len, int(model.max_len))

    method = method.lower()
    if method not in ("greedy", "beam"):
        raise ValueError(f"unknown method {method!r}; expected greedy or beam")

    loader_batch = 1 if method == "beam" else int(batch_size)
    loader = make_dataloader(
        dataset,
        batch_size=loader_batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model.eval()
    predictions: List[str] = []
    references: List[List[str]] = []
    image_ids: List[str] = []
    generation_seconds = 0.0
    total_start = time.perf_counter()

    desc = f"{method} {'beam=' + str(beam_size) if method == 'beam' else ''}".strip()
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["images"].to(device, non_blocking=True)
        start = time.perf_counter()
        if method == "greedy":
            token_matrix = greedy_decode(
                model, images, max_len=max_len, sos_id=SOS_ID, eos_id=EOS_ID
            )
            token_rows = token_matrix.detach().cpu().tolist()
        else:
            if images.size(0) != 1:
                raise RuntimeError("beam inference expects batch_size=1")
            tokens, _score = beam_search_decode(
                model,
                images,
                beam_size=int(beam_size),
                max_len=max_len,
                sos_id=SOS_ID,
                eos_id=EOS_ID,
                length_penalty=float(length_penalty),
                min_len=int(min_len),
            )
            token_rows = [tokens]
        generation_seconds += time.perf_counter() - start

        predictions.extend(vocab.decode(row) for row in token_rows)
        references.extend([list(refs) for refs in batch["references"]])
        image_ids.extend(list(batch["image_ids"]))

    total_seconds = time.perf_counter() - total_start
    n = len(predictions)
    return {
        "method": method,
        "beam_size": int(beam_size),
        "length_penalty": float(length_penalty),
        "min_len": int(min_len),
        "max_len": max_len,
        "image_ids": image_ids,
        "predictions": predictions,
        "references": references,
        "generation_seconds": generation_seconds,
        "total_seconds": total_seconds,
        "ms_per_image": (generation_seconds / n * 1000.0) if n else 0.0,
        "avg_len": average_length(predictions),
        "rep4": repetition_rate(predictions),
    }


def save_predictions(
    image_ids: Sequence[str],
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
    output: str,
) -> Path:
    """Write the frozen ``image_id,prediction,reference`` CSV."""
    if not (len(image_ids) == len(predictions) == len(references)):
        raise ValueError("image_ids / predictions / references length mismatch")
    output_path = resolve_path(output)
    assert output_path is not None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "prediction", "reference"])
        for image_id, prediction, refs in zip(image_ids, predictions, references):
            clean_prediction = " ".join(str(prediction).split())
            if isinstance(refs, str):
                refs = [refs]
            clean_refs = [" ".join(str(r).split()) for r in refs]
            writer.writerow([image_id, clean_prediction, " ||| ".join(clean_refs)])
    return output_path


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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate task-1 captions")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--method", default="greedy", choices=["greedy", "beam"])
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--min-len", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--ids-file", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    bundle = load_bundle(args.checkpoint, device=args.device)
    dataset = build_split_dataset(
        bundle,
        split=args.split,
        manifest=args.manifest,
        image_root=args.image_root,
        ids_file=args.ids_file,
        limit=args.limit,
    )
    run = run_inference(
        bundle,
        dataset,
        method=args.method,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        min_len=args.min_len,
    )

    if args.output:
        output = args.output
    else:
        suffix = f"beam{args.beam_size}_lp{args.length_penalty:g}" if args.method == "beam" else "greedy"
        output = (
            f"outputs/task1/predictions/"
            f"{bundle['model_name']}_{args.split}_{suffix}.csv"
        )
    output_path = save_predictions(
        run["image_ids"], run["predictions"], run["references"], output
    )

    from src.common.io import save_json

    meta = {
        "checkpoint": bundle["checkpoint"],
        "git_commit": _git_commit(),
        "model": bundle["model_name"],
        "model_config": bundle["model_config"],
        "data_config": bundle["data_config"],
        "split": args.split,
        "method": args.method,
        "beam_size": int(args.beam_size) if args.method == "beam" else None,
        "length_penalty": float(args.length_penalty) if args.method == "beam" else None,
        "min_len": int(args.min_len) if args.method == "beam" else None,
        "max_len": run["max_len"],
        "num_images": len(run["predictions"]),
        "generation_seconds": run["generation_seconds"],
        "total_seconds": run["total_seconds"],
        "ms_per_image": run["ms_per_image"],
        "avg_len": run["avg_len"],
        "rep4": run["rep4"],
        "predictions_file": str(output_path),
    }
    meta_path = output_path.with_name(output_path.stem + "_meta.json")
    save_json(meta, str(meta_path))

    print(f"[infer] wrote {output_path}")
    print(
        f"[infer] {len(run['predictions'])} images | "
        f"{run['ms_per_image']:.1f} ms/img | avg_len={run['avg_len']:.1f} | "
        f"rep4={run['rep4']:.3f}"
    )
    print(f"[infer] wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
