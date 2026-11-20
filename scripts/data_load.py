#!/usr/bin/env python3
"""Load Food-500 Cap from HuggingFace and materialise it for task 1.

Why this script
---------------
The rest of the task-1 pipeline expects a local raw directory::

    data/raw/food500cap/
    ├── annotations.json        # generated here
    └── images/
        ├── train/<class>/000000.jpg
        └── test/<class>/000000.jpg

If you already have the course-provided images + annotations, skip this
script and run ``prepare_data_task1.py`` directly.  If you do not have the
data yet, this script can pull ``advancedcv/Food500Cap`` (or another
HuggingFace dataset), inspect its columns, save the images, and write an
``annotations.json`` that ``prepare_data_task1.py`` parses.

Quick start
-----------
::

    pip install datasets
    python scripts/data_load.py --out-dir data/raw/food500cap --inspect
    python scripts/data_load.py --out-dir data/raw/food500cap

    # then the normal preparation step
    python scripts/prepare_data_task1.py --raw-dir data/raw/food500cap

In mainland China the HuggingFace mirror is usually faster::

    HF_ENDPOINT=https://hf-mirror.com python scripts/data_load.py ...

If the dataset is gated, log in first::

    huggingface-cli login

or pass ``--token hf_xxx``.

Column auto-detection
---------------------
The script looks for common names:

* image:  ``image``, ``img``, ``picture``, ``photo``
* caption:``caption``, ``captions``, ``text``, ``description``,
  ``food_caption``, ``sentence``, ``target_text``
* class:  ``class``, ``category``, ``label``, ``food_type``, ``dish``,
  ``cuisine``
* split:  ``split``, ``subset``, ``phase``

If detection is wrong, pass ``--image-column`` / ``--caption-column`` /
``--class-column`` / ``--split-column`` explicitly.  ``--inspect`` prints the
actual columns without saving anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.common.io import ensure_dir, resolve_path  # noqa: E402

IMAGE_COLUMNS = ["image", "img", "picture", "photo", "image_path"]
CAPTION_COLUMNS = [
    "caption", "captions", "text", "description", "food_caption",
    "sentence", "sentences", "target_text", "raw",
]
CLASS_COLUMNS = ["class", "category", "label", "food_type", "dish", "cuisine", "food"]
SPLIT_COLUMNS = ["split", "subset", "phase", "set"]
SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize_split(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in ("train", "training"):
        return "train"
    if text in ("val", "valid", "validation", "dev"):
        return "val"
    if text in ("test", "testing"):
        return "test"
    return text


def _slug(value: str, fallback: str = "unlabeled") -> str:
    value = " ".join(str(value or "").split())
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return value[:80] or fallback


def _caption_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            out.extend(_caption_list(item))
        return _dedupe(out)
    if isinstance(value, dict):
        for key in CAPTION_COLUMNS:
            if key in value:
                return _caption_list(value[key])
    return []


def _dedupe(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        item = " ".join(str(item).split())
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _class_name(dataset, row: Dict[str, Any], column: Optional[str]) -> str:
    if not column:
        return ""
    value = row.get(column)
    feature = dataset.features.get(column) if hasattr(dataset, "features") else None
    if feature is not None and type(feature).__name__ == "ClassLabel":
        try:
            return str(feature.names[int(value)])
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value) if value is not None else ""


def _image_from_value(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            import io

            return Image.open(io.BytesIO(value["bytes"]))
        if value.get("path"):
            return Image.open(value["path"])
    if isinstance(value, (bytes, bytearray)):
        import io

        return Image.open(io.BytesIO(bytes(value)))
    if isinstance(value, str) and Path(value).exists():
        return Image.open(value)
    raise ValueError(
        "cannot interpret image value; expected PIL.Image, dict with bytes/path, "
        f"bytes or an existing path, got {type(value)!r}"
    )


def _image_hash(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.md5()
    digest.update(str(rgb.size).encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Dataset loading / column detection
# ---------------------------------------------------------------------------
def _load_datasets(args) -> Any:
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "the 'datasets' package is required for scripts/data_load.py.\n"
            "Install it with:  pip install datasets"
        ) from exc

    if args.token:
        # Works across datasets/huggingface_hub versions and avoids the
        # token=/use_auth_token= signature difference.
        os.environ["HF_TOKEN"] = args.token

    kwargs: Dict[str, Any] = {}
    if args.revision:
        kwargs["revision"] = args.revision
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True

    if args.config:
        kwargs["name"] = args.config

    if args.splits:
        from datasets import DatasetDict

        loaded = {}
        for split in args.splits:
            loaded[split] = load_dataset(args.hf_dataset, split=split, **kwargs)
        return DatasetDict(loaded)

    return load_dataset(args.hf_dataset, **kwargs)


def _detect_columns(dataset, args) -> Tuple[str, str, Optional[str], Optional[str]]:
    columns = list(dataset.column_names)
    lower = {name.lower(): name for name in columns}

    image_col = args.image_column
    if image_col is None:
        image_col = next((lower[name] for name in IMAGE_COLUMNS if name in lower), None)
    if image_col is None:
        for name in columns:
            feature = dataset.features.get(name)
            if feature is not None and type(feature).__name__ == "Image":
                image_col = name
                break
    if image_col is None:
        raise SystemExit(
            "could not find an image column. Columns are: "
            f"{columns}. Pass --image-column."
        )

    caption_col = args.caption_column
    if caption_col is None:
        caption_col = next((lower[name] for name in CAPTION_COLUMNS if name in lower), None)
    if caption_col is None:
        # Last resort: a string column with reasonably long text.
        sample_size = min(20, len(dataset))
        best, best_len = None, 0.0
        for name in columns:
            if name == image_col:
                continue
            feature = dataset.features.get(name)
            if feature is None or type(feature).__name__ != "Value":
                continue
            if getattr(feature, "dtype", "") != "string":
                continue
            lengths = []
            for i in range(sample_size):
                value = dataset[i].get(name)
                if isinstance(value, str):
                    lengths.append(len(value.split()))
            avg = sum(lengths) / len(lengths) if lengths else 0.0
            if avg > best_len and avg >= 5:
                best, best_len = name, avg
        caption_col = best
    if caption_col is None:
        raise SystemExit(
            "could not find a caption column. Columns are: "
            f"{columns}. Pass --caption-column."
        )

    class_col = args.class_column
    if class_col is None:
        class_col = next((lower[name] for name in CLASS_COLUMNS if name in lower), None)
    if class_col is None:
        for name in columns:
            feature = dataset.features.get(name)
            if feature is not None and type(feature).__name__ == "ClassLabel":
                class_col = name
                break

    split_col = args.split_column
    if split_col is None:
        split_col = next((lower[name] for name in SPLIT_COLUMNS if name in lower), None)

    return image_col, caption_col, class_col, split_col


def _inspect(dataset, args) -> None:
    print(f"[data_load] dataset: {args.hf_dataset}")
    print(f"[data_load] type:     {type(dataset).__name__}")
    if hasattr(dataset, "keys"):
        print(f"[data_load] splits:   {list(dataset.keys())}")
        for split_name, split_ds in dataset.items():
            print(f"\n[{split_name}] rows={len(split_ds)} columns={split_ds.column_names}")
            print(f"  features: {split_ds.features}")
            if len(split_ds):
                row = split_ds[0]
                preview = {}
                for key, value in row.items():
                    if isinstance(value, Image.Image):
                        preview[key] = f"<PIL {value.size} {value.mode}>"
                    else:
                        text = str(value)
                        preview[key] = text[:200] + ("..." if len(text) > 200 else "")
                print(f"  first row: {json.dumps(preview, ensure_ascii=False, indent=2)}")
    else:
        print(f"[data_load] rows={len(dataset)} columns={dataset.column_names}")
        print(f"  features: {dataset.features}")
    image_col, caption_col, class_col, split_col = _detect_columns(
        next(iter(dataset.values())) if hasattr(dataset, "values") else dataset, args
    )
    print(
        f"\n[data_load] detected columns: image={image_col!r} "
        f"caption={caption_col!r} class={class_col!r} split={split_col!r}"
    )
    print("[data_load] inspect only; nothing written")


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def _save_split(
    dataset,
    split_name: str,
    args,
    out_dir: Path,
    records_by_hash: Dict[str, Dict[str, Any]],
    report: Dict[str, Any],
) -> int:
    image_col, caption_col, class_col, split_col = _detect_columns(dataset, args)
    limit = int(args.limit) if args.limit and int(args.limit) > 0 else None
    count = len(dataset) if limit is None else min(limit, len(dataset))
    saved = 0
    merged = 0
    skipped = 0
    counters: Counter = Counter()

    iterator = dataset.select(range(count)) if count < len(dataset) else dataset
    for index, row in enumerate(tqdm(iterator, total=count, desc=f"load {split_name}")):
        captions = _caption_list(row.get(caption_col))
        if not captions:
            skipped += 1
            continue
        record_split = _normalize_split(row.get(split_col)) if split_col else ""
        split = record_split or _normalize_split(split_name) or "train"
        if split not in ("train", "val", "test"):
            split = "train"
        class_name = _class_name(dataset, row, class_col)
        class_slug = _slug(class_name)

        try:
            image = _image_from_value(row.get(image_col)).convert("RGB")
        except Exception as exc:
            print(f"[data_load] skip row {index} in {split_name}: {exc}")
            skipped += 1
            continue

        image_hash = _image_hash(image)
        if image_hash in records_by_hash:
            existing = records_by_hash[image_hash]
            existing["captions"] = _dedupe(existing["captions"] + captions)
            if not existing.get("class") and class_name:
                existing["class"] = class_name
            previous_split = existing.get("split", "")
            if SPLIT_PRIORITY.get(split, 3) < SPLIT_PRIORITY.get(previous_split, 3):
                existing["split"] = split
            merged += 1
            continue

        counters[(split, class_slug)] += 1
        target_dir = out_dir / "images" / split / class_slug
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{counters[(split, class_slug)] - 1:06d}.jpg"
        image.save(target, format="JPEG", quality=int(args.jpeg_quality))

        relative = target.relative_to(out_dir).as_posix()
        records_by_hash[image_hash] = {
            "image": relative,
            "captions": _dedupe(captions),
            "class": class_name,
            "split": split,
            "md5": image_hash,
        }
        saved += 1

    report["splits"][split_name] = {
        "rows": count,
        "saved_images": saved,
        "merged_duplicate_images": merged,
        "skipped_rows": skipped,
    }
    return saved


def _write_annotations(
    records_by_hash: Dict[str, Dict[str, Any]],
    out_dir: Path,
    overwrite: bool,
) -> Path:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records_by_hash.values():
        grouped.setdefault(record["split"], []).append(record)
    for records in grouped.values():
        records.sort(key=lambda item: item["image"])
    target = out_dir / "annotations.json"
    if target.exists() and not overwrite:
        target = out_dir / "annotations_hf.json"
        print(
            f"[data_load] {out_dir / 'annotations.json'} already exists; "
            f"writing {target.name} instead. Pass --overwrite to replace it."
        )
    with open(target, "w", encoding="utf-8") as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2)
    return target


def _run_prepare(out_dir: Path, args, annotations_path: Path) -> None:
    import subprocess

    prepare_script = ROOT / "scripts" / "prepare_data_task1.py"
    cmd = [sys.executable, str(prepare_script), "--raw-dir", str(out_dir)]
    if annotations_path.name != "annotations.json":
        cmd += ["--annotations", str(annotations_path)]
    if args.images_dir:
        cmd += ["--images-dir", str(args.images_dir)]
    if args.limit and int(args.limit) > 0:
        cmd += ["--limit", str(args.limit)]
    print("[data_load] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download Food-500 Cap from HuggingFace and prepare raw files"
    )
    parser.add_argument("--hf-dataset", default="advancedcv/Food500Cap")
    parser.add_argument("--config", default=None, help="HuggingFace dataset config name")
    parser.add_argument("--splits", nargs="+", default=None,
                        help="splits to load, e.g. train test; default = all")
    parser.add_argument("--out-dir", default="data/raw/food500cap")
    parser.add_argument("--images-dir", default=None,
                        help="only used when forwarding to prepare_data_task1.py")
    parser.add_argument("--image-column", default=None)
    parser.add_argument("--caption-column", default=None)
    parser.add_argument("--class-column", default=None)
    parser.add_argument("--split-column", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="max rows per split (quick test)")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite an existing annotations.json")
    parser.add_argument("--inspect", action="store_true",
                        help="print dataset columns/splits without saving")
    parser.add_argument("--prepare", action="store_true",
                        help="run scripts/prepare_data_task1.py afterwards")
    parser.add_argument("--hf-endpoint", default=None,
                        help="e.g. https://hf-mirror.com")
    parser.add_argument("--token", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(argv)

    out_dir = resolve_path(args.out_dir)
    assert out_dir is not None

    dataset = _load_datasets(args)
    if args.inspect:
        _inspect(dataset, args)
        return 0

    ensure_dir(str(out_dir))
    records_by_hash: Dict[str, Dict[str, Any]] = {}
    report: Dict[str, Any] = {
        "hf_dataset": args.hf_dataset,
        "config": args.config,
        "out_dir": str(out_dir),
        "splits": {},
        "images": 0,
        "captions": 0,
    }

    if hasattr(dataset, "items"):
        for split_name, split_ds in dataset.items():
            _save_split(split_ds, split_name, args, out_dir, records_by_hash, report)
    else:
        _save_split(dataset, "train", args, out_dir, records_by_hash, report)

    report["images"] = len(records_by_hash)
    report["captions"] = sum(len(record["captions"]) for record in records_by_hash.values())
    report["split_counts"] = {
        split: sum(1 for record in records_by_hash.values() if record["split"] == split)
        for split in sorted({record["split"] for record in records_by_hash.values()})
    }
    annotations_path = _write_annotations(records_by_hash, out_dir, args.overwrite)
    with open(out_dir / "data_load_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("")
    print(f"[data_load] wrote {annotations_path}")
    print(f"[data_load] images: {report['images']} | captions: {report['captions']}")
    print(f"[data_load] split counts: {report['split_counts']}")
    print(f"[data_load] report: {out_dir / 'data_load_report.json'}")
    print("")
    print("Next step:")
    annotation_arg = "" if annotations_path.name == "annotations.json" else (
        f" --annotations {annotations_path}"
    )
    print(f"  python scripts/prepare_data_task1.py --raw-dir {args.out_dir}{annotation_arg}")
    print("If the official split is missing, prepare_data_task1.py will hold out "
          "val/test from the training pool and print a warning.")

    if args.prepare:
        _run_prepare(out_dir, args, annotations_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
