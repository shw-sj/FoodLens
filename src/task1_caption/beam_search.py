"""Greedy / beam decoding plus the beam-width and length-penalty sweep.

Both decoders use the model's small state interface (see
``models/__init__.py``), so they work unchanged for ResNet-18+GRU and
ViT+Transformer.

Beam search details
-------------------
* Cumulative log-probability is used to prune candidates.
* A finished sentence is scored with the Google-NMT length normalisation
  ``logprob / (((5 + length) / 6) ** alpha)`` where ``length`` is the number
  of generated words (without SOS/EOS).  ``alpha=0`` disables the penalty.
* ``min_len`` delays ``<eos>`` until at least that many words were produced.
* Width and alpha are tuned on the validation split; the test split is only
  used with the selected setting.

Command-line usage (tuning + optional final test run)::

    python -m src.task1_caption.beam_search \
        --checkpoint checkpoints/task1/vit_transformer/best.pt \
        --split val --beam-sizes 3 5 7 \
        --length-penalties 0.6 0.8 1.0 1.2 1.4 --limit 500

or through ``configs/task1_beam_search.yaml``.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.common.io import ensure_dir, load_yaml, resolve_path
from src.common.metrics import compute_all

NEG_INF = float("-inf")


def _length_norm(length: int, alpha: float) -> float:
    if alpha == 0:
        return 1.0
    return ((5.0 + float(length)) / 6.0) ** float(alpha)


def greedy_decode(
    model: torch.nn.Module,
    images: torch.Tensor,
    max_len: int,
    sos_id: int = 1,
    eos_id: int = 2,
) -> torch.Tensor:
    """Batched greedy decoding.

    Returns ``[N, T]`` token ids (SOS excluded, EOS included when emitted).
    """
    if max_len <= 0:
        return images.new_zeros((images.size(0), 0), dtype=torch.long)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            state = model.init_decode_state(images)
            batch = images.size(0)
            prev = torch.full(
                (batch,), int(sos_id), dtype=torch.long, device=images.device
            )
            outputs: List[torch.Tensor] = []
            finished = torch.zeros(batch, dtype=torch.bool, device=images.device)
            for _ in range(int(max_len)):
                logits, state = model.decode_step(state, prev)
                next_token = logits.argmax(dim=-1)
                outputs.append(next_token)
                finished = finished | next_token.eq(int(eos_id))
                prev = next_token
                if bool(finished.all()):
                    break
            if not outputs:
                return images.new_zeros((batch, 0), dtype=torch.long)
            return torch.stack(outputs, dim=1)
    finally:
        if was_training:
            model.train()


def beam_search_decode(
    model: torch.nn.Module,
    image: torch.Tensor,
    beam_size: int = 5,
    max_len: int = 40,
    sos_id: int = 1,
    eos_id: int = 2,
    length_penalty: float = 1.0,
    min_len: int = 0,
) -> Tuple[List[int], float]:
    """Beam search for a single image (``image`` has batch size 1).

    Returns ``(tokens, normalised_score)`` with SOS/EOS removed.
    """
    if image.size(0) != 1:
        raise ValueError("beam_search_decode expects a single image")
    if beam_size < 1:
        raise ValueError("beam_size must be >= 1")
    if max_len <= 0:
        return [], 0.0

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            state = model.init_decode_state(image)
            state = model.repeat_decode_state(state, beam_size)
            device = image.device

            tokens: List[List[int]] = [[int(sos_id)] for _ in range(beam_size)]
            scores: List[float] = [NEG_INF] * beam_size
            scores[0] = 0.0
            finished: List[Tuple[List[int], float]] = []

            for _step in range(int(max_len)):
                if not any(math.isfinite(s) for s in scores):
                    break
                prev = torch.tensor(
                    [tokens[b][-1] for b in range(beam_size)],
                    dtype=torch.long,
                    device=device,
                )
                logits, state = model.decode_step(state, prev)
                log_probs = F.log_softmax(logits.float(), dim=-1)
                vocab_size = log_probs.size(-1)
                top_k = min(beam_size, vocab_size)
                top_values, top_indices = torch.topk(log_probs, k=top_k, dim=-1)

                candidates: List[Tuple[float, int, int]] = []
                for b in range(beam_size):
                    if not math.isfinite(scores[b]):
                        continue
                    for rank in range(top_k):
                        value = float(top_values[b, rank].item())
                        token = int(top_indices[b, rank].item())
                        candidates.append((scores[b] + value, b, token))
                if not candidates:
                    break
                candidates.sort(key=lambda item: item[0], reverse=True)
                candidates = candidates[:beam_size]

                parents = [candidate[1] for candidate in candidates]
                if len(parents) < beam_size:
                    parents += [parents[-1]] * (beam_size - len(parents))
                parent_tensor = torch.tensor(parents, dtype=torch.long, device=device)
                state = model.select_decode_state(state, parent_tensor)

                new_tokens: List[List[int]] = []
                new_scores: List[float] = []
                for score, parent, token in candidates:
                    sequence = tokens[parent] + [token]
                    generated = len(sequence) - 1  # without SOS
                    if token == int(eos_id) and generated >= int(min_len):
                        word_tokens = sequence[1:-1]
                        norm_score = score / _length_norm(max(1, len(word_tokens)), length_penalty)
                        finished.append((word_tokens, norm_score))
                        new_tokens.append([int(sos_id)])
                        new_scores.append(NEG_INF)
                    else:
                        new_tokens.append(sequence)
                        new_scores.append(score)
                while len(new_tokens) < beam_size:
                    new_tokens.append([int(sos_id)])
                    new_scores.append(NEG_INF)
                tokens, scores = new_tokens, new_scores

            if finished:
                best_tokens, best_score = max(finished, key=lambda item: item[1])
                return list(best_tokens), float(best_score)

            best_index = max(
                range(beam_size), key=lambda i: scores[i] if math.isfinite(scores[i]) else NEG_INF
            )
            if not math.isfinite(scores[best_index]):
                return [], NEG_INF
            word_tokens = tokens[best_index][1:]
            norm_score = scores[best_index] / _length_norm(max(1, len(word_tokens)), length_penalty)
            return list(word_tokens), float(norm_score)
    finally:
        if was_training:
            model.train()


# ---------------------------------------------------------------------------
# Sweep / comparison CLI
# ---------------------------------------------------------------------------
def _read_protocol(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = resolve_path(path)
    if p is None or not p.exists():
        return {}
    return load_yaml(path)


def _metric_kwargs(protocol: Dict[str, Any]) -> Dict[str, Any]:
    metrics_cfg = protocol.get("metrics", {}) if protocol else {}
    cider_cfg = metrics_cfg.get("cider_d", {})
    bleu_cfg = metrics_cfg.get("bleu", {})
    rouge_cfg = metrics_cfg.get("rouge_l", {})
    return {
        "bleu_max_order": int(bleu_cfg.get("max_order", 4)),
        "rouge_beta": float(rouge_cfg.get("beta", 1.2)),
        "cider_max_order": int(cider_cfg.get("max_order", 4)),
        "cider_sigma": float(cider_cfg.get("sigma", 6.0)),
        "cider_clip": int(cider_cfg.get("clip", 0)),
        "bleu_scale": float(bleu_cfg.get("scale", 100.0)),
        "rouge_scale": float(rouge_cfg.get("scale", 100.0)),
        "cider_scale": float(cider_cfg.get("scale", 10.0)),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Beam-width / length-penalty sweep for a task-1 checkpoint"
    )
    parser.add_argument("--config", default=None, help="optional YAML config")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--final-split", default=None,
                        help="e.g. test; runs greedy and the best beam setting")
    parser.add_argument("--beam-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--length-penalties", nargs="+", type=float, default=None)
    parser.add_argument("--min-len", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of validation images during tuning")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--protocol", default=None)
    args = parser.parse_args(argv)

    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = load_yaml(args.config)
    checkpoint = args.checkpoint or cfg.get("checkpoint")
    if not checkpoint:
        parser.error("--checkpoint is required (or set checkpoint: in the config)")
    split = args.split or cfg.get("split", "val")
    beam_sizes = args.beam_sizes or cfg.get("beam_sizes", [3, 5, 7])
    length_penalties = args.length_penalties or cfg.get("length_penalties", [0.6, 0.8, 1.0, 1.2, 1.4])
    max_len = args.max_len or cfg.get("max_len")
    min_len = args.min_len if args.min_len is not None else int(cfg.get("min_len", 0))
    limit = args.limit if args.limit is not None else cfg.get("limit")
    batch_size = args.batch_size or int(cfg.get("batch_size", 32))
    num_workers = args.num_workers if args.num_workers is not None else int(cfg.get("num_workers", 4))
    out_dir = ensure_dir(args.out_dir or cfg.get("output_dir", "outputs/task1/beam_search"))
    protocol_path = args.protocol or cfg.get("protocol", "configs/eval_protocol.yaml")
    protocol = _read_protocol(protocol_path)
    metric_kwargs = _metric_kwargs(protocol)

    # Lazy import avoids a module cycle (infer imports the decoders above).
    from . import infer as infer_mod

    bundle = infer_mod.load_bundle(checkpoint, device=args.device)
    model = bundle["model"]
    data_cfg = bundle["data_config"]
    max_len = int(max_len or data_cfg.get("max_len", 40))
    max_len = min(max_len, int(getattr(model, "max_len", max_len)))

    dataset = infer_mod.build_split_dataset(
        bundle,
        split=split,
        manifest=args.manifest,
        image_root=args.image_root,
        limit=limit,
    )

    # Greedy baseline.
    print(f"[beam] greedy baseline on {split} ({len(dataset)} images)")
    greedy = infer_mod.run_inference(
        bundle, dataset, method="greedy", max_len=max_len, batch_size=batch_size,
        num_workers=num_workers,
    )
    greedy_metrics = compute_all(greedy["predictions"], greedy["references"], **metric_kwargs)
    infer_mod.save_predictions(
        greedy["image_ids"], greedy["predictions"], greedy["references"],
        str(Path(out_dir) / f"{split}_greedy.csv"),
    )
    print("[beam] greedy:", _format_metrics(greedy_metrics))

    results: List[Dict[str, Any]] = []
    for beam_size in beam_sizes:
        for alpha in length_penalties:
            name = f"{split}_bs{beam_size}_lp{alpha:g}"
            print(f"[beam] {name} ({len(dataset)} images)")
            run = infer_mod.run_inference(
                bundle, dataset, method="beam", max_len=max_len,
                batch_size=1, num_workers=num_workers,
                beam_size=int(beam_size), length_penalty=float(alpha),
                min_len=int(min_len),
            )
            metrics = compute_all(run["predictions"], run["references"], **metric_kwargs)
            save_path = str(Path(out_dir) / f"{name}.csv")
            infer_mod.save_predictions(
                run["image_ids"], run["predictions"], run["references"], save_path
            )
            entry = {
                "name": name,
                "beam_size": int(beam_size),
                "length_penalty": float(alpha),
                "min_len": int(min_len),
                "split": split,
                "predictions": save_path,
                "generation_seconds": run["generation_seconds"],
                "ms_per_image": run["ms_per_image"],
                **metrics,
            }
            results.append(entry)
            print(f"[beam]   -> {_format_metrics(metrics)}  ({run['ms_per_image']:.1f} ms/img)")

    results.sort(key=lambda item: (item["cider_d"], item["bleu4"]), reverse=True)
    best = results[0] if results else None
    summary = {
        "checkpoint": str(resolve_path(checkpoint)),
        "split": split,
        "limit": limit,
        "protocol": protocol_path,
        "greedy": {
            "predictions": str(Path(out_dir) / f"{split}_greedy.csv"),
            "generation_seconds": greedy["generation_seconds"],
            "ms_per_image": greedy["ms_per_image"],
            **greedy_metrics,
        },
        "beam_results": results,
        "best": best,
    }
    from src.common.io import save_json

    save_json(summary, str(Path(out_dir) / "tuning_results.json"))
    _write_markdown(summary, str(Path(out_dir) / "tuning_results.md"))
    if best is None:
        print("[beam] no beam settings were evaluated")
        return 0
    print(f"[beam] best: {best['name']}  {_format_metrics(best)}")
    print(f"[beam] wrote {Path(out_dir) / 'tuning_results.json'}")

    if args.final_split and best is not None:
        final_dir = ensure_dir(str(Path(out_dir) / "final"))
        print(f"[beam] final test run on {args.final_split}")
        final_dataset = infer_mod.build_split_dataset(
            bundle,
            split=args.final_split,
            manifest=args.manifest,
            image_root=args.image_root,
            limit=None,
        )
        final_greedy = infer_mod.run_inference(
            bundle, final_dataset, method="greedy", max_len=max_len,
            batch_size=batch_size, num_workers=num_workers,
        )
        final_beam = infer_mod.run_inference(
            bundle, final_dataset, method="beam", max_len=max_len,
            batch_size=1, num_workers=num_workers,
            beam_size=best["beam_size"], length_penalty=best["length_penalty"],
            min_len=best["min_len"],
        )
        final_metrics = {
            "greedy": {
                **compute_all(final_greedy["predictions"], final_greedy["references"], **metric_kwargs),
                "generation_seconds": final_greedy["generation_seconds"],
                "ms_per_image": final_greedy["ms_per_image"],
            },
            "beam": {
                "beam_size": best["beam_size"],
                "length_penalty": best["length_penalty"],
                **compute_all(final_beam["predictions"], final_beam["references"], **metric_kwargs),
                "generation_seconds": final_beam["generation_seconds"],
                "ms_per_image": final_beam["ms_per_image"],
            },
            "split": args.final_split,
            "checkpoint": str(resolve_path(checkpoint)),
        }
        infer_mod.save_predictions(
            final_greedy["image_ids"], final_greedy["predictions"], final_greedy["references"],
            str(final_dir / f"{args.final_split}_greedy.csv"),
        )
        infer_mod.save_predictions(
            final_beam["image_ids"], final_beam["predictions"], final_beam["references"],
            str(final_dir / f"{args.final_split}_beam_bs{best['beam_size']}_lp{best['length_penalty']:g}.csv"),
        )
        save_json(final_metrics, str(final_dir / "metrics.json"))
        _write_final_markdown(final_metrics, str(final_dir / "metrics.md"))
        print("[beam] final greedy:", _format_metrics(final_metrics["greedy"]))
        print("[beam] final beam:  ", _format_metrics(final_metrics["beam"]))
        print(f"[beam] wrote {final_dir / 'metrics.json'}")

    return 0


def _format_metrics(metrics: Dict[str, Any]) -> str:
    return (
        f"BLEU-4={metrics['bleu4']:.2f} "
        f"ROUGE-L={metrics['rouge_l']:.2f} "
        f"CIDEr-D={metrics['cider_d']:.3f} "
        f"rep4={metrics['rep4']:.3f} len={metrics['avg_len']:.1f}"
    )


def _write_markdown(summary: Dict[str, Any], path: str) -> None:
    lines = [
        "# Beam search sweep",
        "",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- split: `{summary['split']}`",
        f"- limit: {summary['limit']}",
        "",
        "| setting | BLEU-4 | ROUGE-L | CIDEr-D | rep4 | avg len | ms/img |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    g = summary["greedy"]
    lines.append(
        f"| greedy | {g['bleu4']:.2f} | {g['rouge_l']:.2f} | {g['cider_d']:.3f} | "
        f"{g['rep4']:.3f} | {g['avg_len']:.1f} | {g.get('ms_per_image', 0):.1f} |"
    )
    for row in summary["beam_results"]:
        lines.append(
            f"| beam={row['beam_size']} lp={row['length_penalty']:g} | "
            f"{row['bleu4']:.2f} | {row['rouge_l']:.2f} | {row['cider_d']:.3f} | "
            f"{row['rep4']:.3f} | {row['avg_len']:.1f} | {row['ms_per_image']:.1f} |"
        )
    best = summary.get("best")
    if best:
        lines += ["", f"Selected on validation: `{best['name']}` (highest CIDEr-D, BLEU-4 tie-break)."]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def _write_final_markdown(final: Dict[str, Any], path: str) -> None:
    lines = [
        f"# Final test comparison ({final['split']})",
        "",
        f"- checkpoint: `{final['checkpoint']}`",
        "",
        "| setting | BLEU-4 | ROUGE-L | CIDEr-D | rep4 | avg len | ms/img |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    g = final["greedy"]
    b = final["beam"]
    lines.append(
        f"| greedy | {g['bleu4']:.2f} | {g['rouge_l']:.2f} | {g['cider_d']:.3f} | "
        f"{g['rep4']:.3f} | {g['avg_len']:.1f} | {g['ms_per_image']:.1f} |"
    )
    lines.append(
        f"| beam={b['beam_size']} lp={b['length_penalty']:g} | {b['bleu4']:.2f} | "
        f"{b['rouge_l']:.2f} | {b['cider_d']:.3f} | {b['rep4']:.3f} | "
        f"{b['avg_len']:.1f} | {b['ms_per_image']:.1f} |"
    )
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
