"""Score one or more prediction CSVs with the frozen task-1 protocol.

The input format is the same one ``infer.py`` writes::

    image_id,prediction,reference

where ``reference`` may contain multiple references separated by
``" ||| "``.  By default all runs must cover the same image IDs and carry the
same references; use ``--allow-id-mismatch`` only for diagnostics.

Examples::

    python -m src.task1_caption.evaluate \
        --pred resnet18_gru=outputs/task1/predictions/resnet18_gru_test_greedy.csv \
        --pred vit_greedy=outputs/task1/predictions/vit_transformer_test_greedy.csv \
        --markdown outputs/task1/metrics/model_comparison.md
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.common.io import ensure_dir, load_yaml, resolve_path, save_json
from src.common.metrics import (
    METRIC_VERSION,
    TOKENIZER_VERSION,
    compute_all,
)


def load_protocol(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = resolve_path(path)
    if p is None or not p.exists():
        return {}
    return load_yaml(path)


def metric_kwargs_from_protocol(protocol: Dict[str, Any]) -> Dict[str, Any]:
    metrics_cfg = protocol.get("metrics", {}) if protocol else {}
    bleu_cfg = metrics_cfg.get("bleu", {})
    rouge_cfg = metrics_cfg.get("rouge_l", {})
    cider_cfg = metrics_cfg.get("cider_d", {})
    return {
        "bleu_max_order": int(bleu_cfg.get("max_order", 4)),
        "bleu_scale": float(bleu_cfg.get("scale", 100.0)),
        "rouge_beta": float(rouge_cfg.get("beta", 1.2)),
        "rouge_scale": float(rouge_cfg.get("scale", 100.0)),
        "cider_max_order": int(cider_cfg.get("max_order", 4)),
        "cider_sigma": float(cider_cfg.get("sigma", 6.0)),
        "cider_clip": int(cider_cfg.get("clip", 0)),
        "cider_scale": float(cider_cfg.get("scale", 10.0)),
    }


def reference_separator(protocol: Dict[str, Any]) -> str:
    return str(protocol.get("references", {}).get("multi_ref_sep", " ||| "))


def read_prediction_csv(
    path: str, separator: str = " ||| "
) -> Tuple[List[str], List[str], List[List[str]]]:
    p = resolve_path(path)
    assert p is not None
    if not p.exists():
        raise FileNotFoundError(f"prediction file not found: {p}")

    ids: List[str] = []
    predictions: List[str] = []
    references: List[List[str]] = []
    seen = set()
    with open(p, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image_id", "prediction", "reference"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"{p} must contain columns image_id,prediction,reference; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            image_id = str(row["image_id"]).strip()
            if not image_id:
                continue
            if image_id in seen:
                raise ValueError(f"duplicate image_id {image_id!r} in {p}")
            seen.add(image_id)
            refs = [part.strip() for part in str(row["reference"]).split(separator) if part.strip()]
            if not refs:
                raise ValueError(f"empty reference for image {image_id} in {p}")
            ids.append(image_id)
            predictions.append(str(row["prediction"]))
            references.append(refs)
    if not ids:
        raise ValueError(f"prediction file is empty: {p}")
    return ids, predictions, references


def _parse_run_args(items: Sequence[str]) -> List[Tuple[str, str]]:
    runs: List[Tuple[str, str]] = []
    labels = set()
    for item in items:
        if "=" in item:
            label, path = item.split("=", 1)
        else:
            path = item
            label = Path(item).stem
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"bad --pred value {item!r}; expected LABEL=PATH")
        if label in labels:
            raise ValueError(f"duplicate run label {label!r}")
        labels.add(label)
        runs.append((label, path))
    sort_index = {label: i for i, (label, _) in enumerate(runs)}
    runs.sort(key=lambda item: sort_index[item[0]])
    return runs


def compare_runs(
    runs: Sequence[Tuple[str, str]],
    protocol: Dict[str, Any],
    allow_id_mismatch: bool = False,
) -> Dict[str, Any]:
    separator = reference_separator(protocol)
    parsed: Dict[str, Tuple[List[str], List[str], List[List[str]]]] = {}
    for label, path in runs:
        parsed[label] = read_prediction_csv(path, separator=separator)

    first_label = runs[0][0]
    base_ids = parsed[first_label][0]
    base_refs = {image_id: refs for image_id, refs in zip(base_ids, parsed[first_label][2])}
    base_set = set(base_ids)

    common_ids = set(base_ids)
    id_mismatch: Dict[str, Any] = {}
    for label, _ in runs[1:]:
        ids = set(parsed[label][0])
        if ids != base_set:
            id_mismatch[label] = {
                "missing_vs_first": sorted(base_set - ids)[:20],
                "extra_vs_first": sorted(ids - base_set)[:20],
                "missing_count": len(base_set - ids),
                "extra_count": len(ids - base_set),
            }
            common_ids &= ids
    if id_mismatch and not allow_id_mismatch:
        raise ValueError(
            "prediction files cover different image IDs; pass "
            "--allow-id-mismatch to score the intersection. Details: "
            + json.dumps(id_mismatch, ensure_ascii=False)
        )

    ordered_ids = sorted(common_ids)
    metric_kwargs = metric_kwargs_from_protocol(protocol)
    results: Dict[str, Any] = {}
    for label, path in runs:
        ids, preds, refs = parsed[label]
        lookup = {image_id: (pred, ref) for image_id, pred, ref in zip(ids, preds, refs)}
        run_preds: List[str] = []
        run_refs: List[List[str]] = []
        for image_id in ordered_ids:
            if image_id not in lookup:
                continue
            pred, ref_group = lookup[image_id]
            if ref_group != base_refs[image_id]:
                raise ValueError(
                    f"references for image {image_id} differ between runs; "
                    "the prediction files cannot be compared"
                )
            run_preds.append(pred)
            run_refs.append(ref_group)
        metrics = compute_all(run_preds, run_refs, **metric_kwargs)
        results[label] = {
            "path": str(resolve_path(path)),
            **metrics,
        }

    ranking = sorted(
        results.items(), key=lambda item: (item[1]["cider_d"], item[1]["bleu4"]), reverse=True
    )
    return {
        "protocol_version": protocol.get("protocol_version", "unknown"),
        "metric_version": METRIC_VERSION,
        "tokenizer_version": TOKENIZER_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "num_images": len(ordered_ids),
        "id_mismatch": id_mismatch,
        "runs": results,
        "ranking": [label for label, _ in ranking],
    }


def _format_table(report: Dict[str, Any]) -> str:
    lines = [
        "| run | BLEU-4 | ROUGE-L | CIDEr-D | rep4 | avg len | n |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in report["ranking"]:
        row = report["runs"][label]
        lines.append(
            f"| {label} | {row['bleu4']:.2f} | {row['rouge_l']:.2f} | "
            f"{row['cider_d']:.4f} | {row['rep4']:.3f} | {row['avg_len']:.1f} | "
            f"{int(row['n'])} |"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate task-1 prediction CSVs")
    parser.add_argument(
        "--pred",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="prediction CSV; repeat for multiple models",
    )
    parser.add_argument("--protocol", default="configs/eval_protocol.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--markdown", default=None)
    parser.add_argument("--allow-id-mismatch", action="store_true")
    parser.add_argument("--split", default=None, help="only recorded in the report")
    args = parser.parse_args(argv)

    protocol = load_protocol(args.protocol)
    runs = _parse_run_args(args.pred)
    report = compare_runs(runs, protocol, allow_id_mismatch=args.allow_id_mismatch)
    report["split"] = args.split
    report["protocol_path"] = str(resolve_path(args.protocol)) if args.protocol else None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out or f"outputs/task1/metrics/eval_{timestamp}.json"
    markdown = args.markdown or str(Path(out).with_suffix(".md"))
    out_path = resolve_path(out)
    assert out_path is not None
    ensure_dir(str(out_path.parent))
    save_json(report, str(out_path))

    table = _format_table(report)
    md_path = resolve_path(markdown)
    assert md_path is not None
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# Task-1 evaluation\n\n")
        f.write(f"- protocol: `{report['protocol_version']}`\n")
        f.write(f"- metric version: `{METRIC_VERSION}` / `{TOKENIZER_VERSION}`\n")
        f.write(f"- images: {report['num_images']}\n\n")
        f.write(table + "\n")

    print(table)
    print(f"\n[evaluate] wrote {out_path}")
    print(f"[evaluate] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
