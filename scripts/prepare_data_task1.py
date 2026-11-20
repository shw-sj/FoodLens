#!/usr/bin/env python3
"""Prepare Food-500 Cap for task 1.

What it does
------------
1. Auto-detects annotation files (JSON / JSONL / CSV / TSV) and the image
   root under ``data/raw/food500cap`` (or a directory you pass explicitly).
2. Parses several common formats:

   * ``{image_name: [caption, ...]}``
   * ``{image_name: {"captions": [...], "class": "..."}}``
   * a list of records with ``image`` + ``caption(s)`` keys
   * COCO-style ``{"images": [...], "annotations": [...]}``
   * top-level ``{"train": ..., "val": ..., "test": ...}``
   * CSV/TSV rows (one caption per row is fine; rows are grouped by image)

3. Checks that images exist, verifies them with Pillow, removes exact
   duplicate files by MD5 (including cross-split leakage), drops empty
   captions and merges duplicate image records.
4. Preserves an official test split when present.  Otherwise it holds out
   ``--test-ratio``.  Validation is the official val split when present,
   otherwise ``--val-ratio`` of the remaining training pool (stratified by
   class, seed 42 by default).
5. Writes:

   * ``data/processed/food500cap_clean.jsonl`` (gitignored)
   * ``data/manifests/food500cap_{train,val,test}_ids.txt`` (committed)
   * ``data/vocab/shared_vocab_5000.json`` (committed)
   * ``data/processed/food500cap_prepare_report.json``

Run::

    python scripts/prepare_data_task1.py --raw-dir data/raw/food500cap

If the course package keeps train/test annotations in separate files::

    python scripts/prepare_data_task1.py \
        --raw-dir data/raw/food500cap \
        --annotations data/raw/food500cap/train.json \
        --test-annotations data/raw/food500cap/test.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from src.common.io import (  # noqa: E402
    ensure_dir,
    resolve_path,
    save_ids,
    save_json,
)
from src.common.json_utils import write_jsonl  # noqa: E402
from src.common.metrics import tokenize  # noqa: E402
from src.task1_caption.datasets import Vocab  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CAPTION_KEYS = [
    "captions", "caption", "target_text", "text", "sentence", "sentences",
    "description", "food_caption", "desc", "annotation", "raw",
    "caption_text", "food_description",
]
IMAGE_KEYS = [
    "image", "image_name", "image_path", "img", "file_name", "filename",
    "file", "path", "img_name", "image_file", "image_fn", "filepath",
]
CLASS_KEYS = ["class", "category", "label", "food_type", "food_name", "dish", "food"]
SPLIT_KEYS = ["split", "subset", "set", "partition", "phase"]
SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2, "": 3}


# ---------------------------------------------------------------------------
# Generic parsing helpers
# ---------------------------------------------------------------------------
def _lower_map(mapping: Dict[str, Any]) -> Dict[str, str]:
    return {str(key).lower(): str(key) for key in mapping}


def _first_value(mapping: Dict[str, Any], keys: Sequence[str]) -> Any:
    lower = _lower_map(mapping)
    for key in keys:
        if key in lower:
            value = mapping[lower[key]]
            if value is not None and str(value).strip() != "":
                return value
    return None


def _normalize_split(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in ("train", "training"):
        return "train"
    if text in ("val", "valid", "validation", "dev"):
        return "val"
    if text in ("test", "testing"):
        return "test"
    return ""


def _dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        item = " ".join(str(item).split())
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _caption_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return _dedupe([value])
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            out.extend(_caption_list(item))
        return _dedupe(out)
    if isinstance(value, dict):
        nested = _first_value(value, CAPTION_KEYS)
        return _caption_list(nested)
    return []


def _parse_record(
    item: Any,
    default_split: Optional[str],
    source: str,
    image_override: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        image = str(item[0]).strip()
        captions = _caption_list(item[1])
        class_name = str(item[2]).strip() if len(item) > 2 and item[2] is not None else ""
        split = _normalize_split(item[3]) if len(item) > 3 else ""
        if image and captions:
            return {
                "image": image,
                "captions": captions,
                "class": class_name,
                "split": split or (default_split or ""),
                "source": source,
            }
        return None

    if not isinstance(item, dict):
        return None

    explicit_image = _first_value(item, IMAGE_KEYS)
    image_value = explicit_image if explicit_image is not None else image_override
    if isinstance(image_value, dict):
        image_value = _first_value(image_value, IMAGE_KEYS)
    image = str(image_value).strip() if image_value is not None else ""
    if not image:
        return None

    captions: List[str] = []
    for key in CAPTION_KEYS:
        value = _first_value(item, [key])
        if value is not None:
            captions = _caption_list(value)
            if captions:
                break
    if not captions:
        # COCO-like annotation records nested inside a record.
        nested = _first_value(item, ["annotations"])
        if isinstance(nested, list):
            for entry in nested:
                if isinstance(entry, dict):
                    captions.extend(_caption_list(_first_value(entry, CAPTION_KEYS)))
            captions = _dedupe(captions)
    if not captions:
        return None

    class_name = _first_value(item, CLASS_KEYS)
    split_value = _first_value(item, SPLIT_KEYS)
    split = _normalize_split(split_value) or (default_split or "")
    return {
        "image": image,
        "captions": captions,
        "class": str(class_name).strip() if class_name is not None else "",
        "split": split,
        "source": source,
    }


def _parse_mapping(
    mapping: Dict[str, Any], default_split: Optional[str], source: str
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for key, value in mapping.items():
        lower_key = str(key).lower()
        if lower_key in ("annotations", "data", "records", "items") and isinstance(value, (list, dict)):
            is_single_record = isinstance(value, dict) and (
                _first_value(value, IMAGE_KEYS) is not None
                or _first_value(value, CAPTION_KEYS) is not None
            )
            if not is_single_record:
                if isinstance(value, list):
                    records.extend(_parse_list(value, default_split, source))
                else:
                    records.extend(_parse_mapping(value, default_split, source))
                continue
        if isinstance(value, dict) and (
            _first_value(value, IMAGE_KEYS) is not None
            or _first_value(value, CAPTION_KEYS) is not None
        ):
            parsed = _parse_record(value, default_split, source, image_override=str(key))
            if parsed:
                records.append(parsed)
            continue
        # Plain mapping image_name -> captions.
        parsed = {
            "image": str(key),
            "captions": _caption_list(value),
            "class": "",
            "split": default_split or "",
            "source": source,
        }
        if parsed["captions"]:
            records.append(parsed)
    return records


def _parse_list(
    items: Sequence[Any], default_split: Optional[str], source: str
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for item in items:
        parsed = _parse_record(item, default_split, source)
        if parsed:
            records.append(parsed)
    return records


def _parse_coco(data: Dict[str, Any], default_split: Optional[str], source: str) -> List[Dict[str, Any]]:
    images = {
        image.get("id"): image
        for image in data.get("images", [])
        if isinstance(image, dict)
    }
    grouped: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for annotation in data.get("annotations", []):
        if not isinstance(annotation, dict):
            continue
        image_info = images.get(annotation.get("image_id"))
        if not image_info:
            continue
        file_name = image_info.get("file_name") or image_info.get("file")
        if not file_name:
            continue
        caption = annotation.get("caption", annotation.get("text"))
        if not caption:
            continue
        entry = grouped.setdefault(
            str(file_name),
            {
                "image": str(file_name),
                "captions": [],
                "class": str(
                    image_info.get("category", image_info.get("class", ""))
                ),
                "split": _normalize_split(
                    image_info.get("split", annotation.get("split"))
                ) or (default_split or ""),
                "source": source,
            },
        )
        entry["captions"].append(str(caption))
    for entry in grouped.values():
        entry["captions"] = _dedupe(entry["captions"])
    return list(grouped.values())


def _parse_csv_file(path: Path, default_split: Optional[str]) -> List[Dict[str, Any]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = reader.fieldnames or []
        lower = {name.lower(): name for name in fieldnames}
        image_key = next((lower[k] for k in IMAGE_KEYS if k in lower), None)
        if image_key is None:
            raise ValueError(
                f"{path}: CSV needs an image column among {IMAGE_KEYS}; found {fieldnames}"
            )
        caption_keys = [lower[k] for k in CAPTION_KEYS if k in lower]
        if not caption_keys:
            raise ValueError(
                f"{path}: CSV needs a caption column among {CAPTION_KEYS}; found {fieldnames}"
            )
        class_key = next((lower[k] for k in CLASS_KEYS if k in lower), None)
        split_key = next((lower[k] for k in SPLIT_KEYS if k in lower), None)

        grouped: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        for row in reader:
            image = str(row.get(image_key, "")).strip()
            if not image:
                continue
            captions = [str(row.get(key, "")) for key in caption_keys]
            captions = [c for c in captions if c.strip()]
            if not captions:
                continue
            entry = grouped.setdefault(
                image,
                {
                    "image": image,
                    "captions": [],
                    "class": "",
                    "split": "",
                    "source": str(path),
                },
            )
            entry["captions"].extend(captions)
            if class_key and not entry["class"]:
                entry["class"] = str(row.get(class_key, "")).strip()
            if split_key and not entry["split"]:
                entry["split"] = _normalize_split(row.get(split_key))
        for entry in grouped.values():
            entry["captions"] = _dedupe(entry["captions"])
            entry["split"] = entry["split"] or (default_split or "")
        return list(grouped.values())


def parse_annotation_file(path: Path, default_split: Optional[str]) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        return _parse_csv_file(path, default_split)
    if suffix == ".jsonl":
        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.extend(_parse_list([json.loads(line)], default_split, str(path)))
        return records
    if suffix != ".json":
        raise ValueError(f"unsupported annotation format: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return _parse_list(data, default_split, str(path))
    if not isinstance(data, dict):
        return []

    lower = _lower_map(data)
    if "images" in lower and "annotations" in lower:
        return _parse_coco(data, default_split, str(path))

    # Top-level split keys.
    split_records: List[Dict[str, Any]] = []
    found_split = False
    for key, value in data.items():
        split_name = _normalize_split(key)
        if not split_name:
            continue
        found_split = True
        if isinstance(value, list):
            split_records.extend(_parse_list(value, split_name, str(path)))
        elif isinstance(value, dict):
            lower_value = _lower_map(value)
            if "annotations" in lower_value and isinstance(value[lower_value["annotations"]], list):
                split_records.extend(
                    _parse_list(value[lower_value["annotations"]], split_name, str(path))
                )
            else:
                split_records.extend(_parse_mapping(value, split_name, str(path)))
    if found_split:
        return split_records

    for key in ("annotations", "data", "records", "items"):
        if key in lower and isinstance(data[lower[key]], list):
            return _parse_list(data[lower[key]], default_split, str(path))
    return _parse_mapping(data, default_split, str(path))


# ---------------------------------------------------------------------------
# File / image discovery
# ---------------------------------------------------------------------------
def _split_hint(path: Path) -> str:
    stem = path.stem.lower()
    if "train" in stem:
        return "train"
    if "test" in stem:
        return "test"
    if "val" in stem or "valid" in stem:
        return "val"
    return ""


def discover_annotation_files(
    raw_dir: Path,
    annotations: Optional[str],
    test_annotations: Optional[str],
) -> List[Tuple[str, Path]]:
    files: List[Tuple[str, Path]] = []
    if annotations:
        explicit = resolve_path(annotations)
        assert explicit is not None
        if explicit.is_dir():
            for path in sorted(explicit.rglob("*")):
                if path.suffix.lower() in (".json", ".jsonl", ".csv", ".tsv"):
                    files.append((_split_hint(path), path))
        elif explicit.is_file():
            files.append((_split_hint(explicit), explicit))
        else:
            raise FileNotFoundError(f"--annotations not found: {explicit}")
    else:
        for name in ("annotations.json", "captions.json", "data.json"):
            candidate = raw_dir / name
            if candidate.exists():
                files.append(("", candidate))
        annotation_dir = raw_dir / "annotations"
        if annotation_dir.is_dir():
            for path in sorted(annotation_dir.rglob("*")):
                if path.suffix.lower() in (".json", ".jsonl", ".csv", ".tsv"):
                    files.append((_split_hint(path), path))
        if not files:
            for suffix in ("*.json", "*.jsonl", "*.csv", "*.tsv"):
                files.extend((_split_hint(path), path) for path in sorted(raw_dir.glob(suffix)))

    if test_annotations:
        test_path = resolve_path(test_annotations)
        assert test_path is not None
        if not test_path.exists():
            raise FileNotFoundError(f"--test-annotations not found: {test_path}")
        test_resolved = test_path.resolve()
        files = [(h, p) for h, p in files if p.resolve() != test_resolved]
        files.append(("test", test_path))

    # De-duplicate paths.
    unique: "OrderedDict[Path, str]" = OrderedDict()
    for hint, path in files:
        resolved = path.resolve()
        if resolved not in unique:
            unique[resolved] = hint
        elif hint and not unique[resolved]:
            unique[resolved] = hint
    if not unique:
        raise FileNotFoundError(
            f"no annotation files found under {raw_dir}. Put annotations.json "
            "there, pass --annotations, or extract annotations.zip first."
        )
    all_files = [(hint, path) for path, hint in unique.items()]
    if annotations or test_annotations:
        return all_files
    hinted = [(hint, path) for hint, path in all_files if hint]
    if hinted:
        return hinted
    return all_files


def build_image_index(manifest_root: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    for path in manifest_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(path.name, []).append(path)
    return index


def resolve_image(
    name: str,
    split: str,
    manifest_root: Path,
    raw_dir: Path,
    index: Dict[str, List[Path]],
) -> Optional[Path]:
    name_path = Path(str(name).replace("\\", "/"))
    if name_path.is_absolute():
        return name_path if name_path.exists() else None
    candidates = [
        manifest_root / name_path,
        raw_dir / name_path,
        manifest_root / split / name_path if split else None,
        raw_dir / split / name_path if split else None,
        manifest_root / "images" / name_path,
        raw_dir / "images" / name_path,
        raw_dir / "images" / split / name_path if split else None,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    matches = index.get(name_path.name, [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        preferred = [p for p in matches if split and split in p.parts]
        if len(preferred) == 1:
            return preferred[0]
    return None


def infer_class_from_path(relative_image: str) -> str:
    """Use a class-named parent folder when annotations omit ``class``."""
    ignored = {
        "images", "image", "imgs", "train", "training", "test", "testing",
        "val", "valid", "validation", "data", "raw",
    }
    parts = list(Path(relative_image).parts[:-1])
    for part in reversed(parts):
        lowered = part.lower()
        if lowered in ignored or part.isdigit():
            continue
        return part
    return ""


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Cleaning / splitting
# ---------------------------------------------------------------------------
def stratified_split(
    records: Sequence[Dict[str, Any]],
    ratio: float,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if ratio <= 0 or not records:
        return [], list(records)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record.get("class") or "__unknown__", []).append(record)
    selected: List[Dict[str, Any]] = []
    remaining: List[Dict[str, Any]] = []
    for _key, group in groups.items():
        group = list(group)
        rng.shuffle(group)
        n_selected = int(round(len(group) * float(ratio)))
        if len(group) >= 2:
            n_selected = max(1, min(len(group) - 1, n_selected))
        else:
            n_selected = 0
        selected.extend(group[:n_selected])
        remaining.extend(group[n_selected:])
    rng.shuffle(selected)
    rng.shuffle(remaining)
    return selected, remaining


def _load_split_list(path: Optional[str]) -> set:
    if not path:
        return set()
    p = resolve_path(path)
    assert p is not None
    if not p.exists():
        raise FileNotFoundError(f"split list not found: {p}")
    values = set()
    with open(p, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values.add(line.replace("\\", "/").lstrip("./"))
    return values


def _apply_split_lists(
    records: Sequence[Dict[str, Any]],
    split_lists: Dict[str, Optional[str]],
) -> int:
    """Assign splits from train/val/test image-list files (escape hatch).

    Full relative paths take precedence.  Basenames are only used when they
    appear in exactly one split list, so ``train/0001.jpg`` and
    ``test/0001.jpg`` do not contaminate each other.
    """
    full_sets = {name: _load_split_list(path) for name, path in split_lists.items()}
    if not any(full_sets.values()):
        return 0
    basename_sets = {
        name: {Path(value).name for value in values}
        for name, values in full_sets.items()
    }
    priority = ("test", "val", "train")
    assigned = 0
    for record in records:
        key = str(record["image"]).replace("\\", "/").lstrip("./")
        matches = [name for name in priority if key in full_sets.get(name, set())]
        if matches:
            record["split"] = matches[0]
            assigned += 1
            continue
        basename = Path(key).name
        matches = [
            name for name in priority if basename in basename_sets.get(name, set())
        ]
        if len(matches) == 1:
            record["split"] = matches[0]
            assigned += 1
    return assigned


def _merge_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for record in records:
        key = record["image_id"]
        if key not in merged:
            merged[key] = {
                "image_id": key,
                "image": record["image"],
                "captions": [],
                "class": record.get("class", ""),
                "split": record.get("split", ""),
                "abs_path": record["abs_path"],
                "sources": [],
            }
        entry = merged[key]
        entry["captions"].extend(record["captions"])
        if record.get("class") and not entry["class"]:
            entry["class"] = record["class"]
        previous = entry.get("split", "")
        if SPLIT_PRIORITY.get(record.get("split", ""), 3) < SPLIT_PRIORITY.get(previous, 3):
            entry["split"] = record["split"]
        if record.get("source"):
            entry["sources"].append(record["source"])
    for entry in merged.values():
        entry["captions"] = _dedupe(entry["captions"])
    return list(merged.values())


def split_stats(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    classes = Counter(record.get("class") or "__unknown__" for record in records)
    n_captions = sum(len(record["captions"]) for record in records)
    n_tokens = sum(
        len(tokenize(caption))
        for record in records
        for caption in record["captions"]
    )
    return {
        "images": len(records),
        "captions": n_captions,
        "captions_per_image": round(n_captions / max(1, len(records)), 3),
        "avg_tokens_per_caption": round(n_tokens / max(1, n_captions), 3),
        "classes": len(classes),
        "top_classes": classes.most_common(20),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    raw_dir = resolve_path(args.raw_dir)
    assert raw_dir is not None
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw data directory not found: {raw_dir}")

    manifest_root = resolve_path(args.images_dir) if args.images_dir else raw_dir
    assert manifest_root is not None
    if not manifest_root.exists():
        raise FileNotFoundError(f"images directory not found: {manifest_root}")

    annotation_files = discover_annotation_files(
        raw_dir, args.annotations, args.test_annotations
    )
    parsed: List[Dict[str, Any]] = []
    for split_hint, path in annotation_files:
        records = parse_annotation_file(path, split_hint or None)
        parsed.extend(records)
        print(f"[prepare] {path}: {len(records)} records (split hint={split_hint or '-'})")
    assigned = _apply_split_lists(
        parsed,
        {"train": args.train_list, "val": args.val_list, "test": args.test_list},
    )
    if assigned:
        print(f"[prepare] assigned {assigned} records from explicit split lists")
    if args.limit:
        parsed = parsed[: int(args.limit)]
    if not parsed:
        raise ValueError(
            "no annotated records were parsed. Check the annotation keys; see "
            "the docstring of scripts/prepare_data_task1.py for supported formats."
        )

    print(f"[prepare] indexing images under {manifest_root} ...")
    image_index = build_image_index(manifest_root)
    print(f"[prepare] indexed {sum(len(v) for v in image_index.values())} image files")

    resolved: List[Dict[str, Any]] = []
    missing: List[Dict[str, str]] = []
    for i, record in enumerate(parsed):
        path = resolve_image(
            record["image"], record.get("split", ""), manifest_root, raw_dir, image_index
        )
        if path is None:
            missing.append({"image": record["image"], "class": record.get("class", "")})
            continue
        try:
            rel = path.resolve().relative_to(manifest_root.resolve())
        except ValueError:
            missing.append({
                "image": record["image"],
                "class": record.get("class", ""),
                "reason": f"resolved outside --images-dir/--raw-dir: {path}",
            })
            continue
        entry = {
            **record,
            "image_id": rel.as_posix(),
            "image": rel.as_posix(),
            "abs_path": path,
        }
        if not entry.get("class"):
            entry["class"] = infer_class_from_path(entry["image_id"])
        resolved.append(entry)
        if (i + 1) % 2000 == 0:
            print(f"[prepare] resolved {i + 1}/{len(parsed)} records ...")
    if not resolved:
        raise ValueError(
            f"none of the annotated images could be found under {manifest_root}; "
            "check --images-dir or the annotation image paths"
        )

    # Verify and drop broken images.
    dropped_corrupt: List[str] = []
    if not args.skip_integrity_check:
        for i, record in enumerate(resolved):
            if not verify_image(record["abs_path"]):
                dropped_corrupt.append(record["image_id"])
            if (i + 1) % 2000 == 0:
                print(f"[prepare] verified {i + 1}/{len(resolved)} images ...")
        bad = set(dropped_corrupt)
        resolved = [record for record in resolved if record["image_id"] not in bad]

    # Drop empty / tokenless captions.
    dropped_empty: List[str] = []
    for record in resolved:
        captions = _dedupe(record["captions"])
        captions = [c for c in captions if tokenize(c)]
        if not captions:
            dropped_empty.append(record["image_id"])
        record["captions"] = captions
    empty_set = set(dropped_empty)
    resolved = [record for record in resolved if record["image_id"] not in empty_set]

    # Merge duplicates by image ID (same image reached from two files).
    merged = _merge_records(resolved)

    # Exact duplicate files: drop all but the highest-priority copy.
    duplicates: List[Dict[str, Any]] = []
    by_hash: Dict[str, List[Dict[str, Any]]] = {}
    for i, record in enumerate(merged):
        hash_value = md5sum(record["abs_path"])
        record["md5"] = hash_value
        by_hash.setdefault(hash_value, []).append(record)
        if (i + 1) % 2000 == 0:
            print(f"[prepare] hashed {i + 1}/{len(merged)} images ...")
    kept: List[Dict[str, Any]] = []
    for hash_value, group in by_hash.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        group = sorted(
            group,
            key=lambda r: (SPLIT_PRIORITY.get(r.get("split", ""), 3), r["image_id"]),
        )
        winner = group[0]
        kept.append(winner)
        for other in group[1:]:
            duplicates.append(
                {
                    "kept": winner["image_id"],
                    "dropped": other["image_id"],
                    "kept_split": winner.get("split", ""),
                    "dropped_split": other.get("split", ""),
                    "md5": hash_value,
                }
            )
    kept.sort(key=lambda r: r["image_id"])

    # Freeze split lists.
    train_records = [r for r in kept if r.get("split") == "train"]
    val_records = [r for r in kept if r.get("split") == "val"]
    test_records = [r for r in kept if r.get("split") == "test"]
    unassigned = [r for r in kept if r.get("split") not in ("train", "val", "test")]
    pool = train_records + unassigned

    rng = random.Random(int(args.seed))
    warnings: List[str] = []
    if kept and all(not record.get("class") for record in kept):
        warnings.append(
            "no class labels were found; the validation/test hold-out is random "
            "rather than class-stratified. Add a 'class' field or folder name "
            "if the split must be stratified."
        )
    if not test_records:
        test_records, pool = stratified_split(pool, float(args.test_ratio), rng)
        warnings.append(
            f"no official test split found; held out {args.test_ratio:.0%} as test"
        )
    if not val_records:
        val_records, pool = stratified_split(pool, float(args.val_ratio), rng)
        warnings.append(
            f"no official validation split found; held out {args.val_ratio:.0%} of "
            "the remaining training pool as validation"
        )
    train_records = pool

    for name, records in (("train", train_records), ("val", val_records), ("test", test_records)):
        if not records:
            warnings.append(f"{name} split is empty")
    if not train_records:
        raise ValueError(
            "the training split is empty. If your annotations only contain a test "
            "split, pass --test-annotations for it and --annotations for the rest, "
            "or provide all splits in one file."
        )

    # Deterministic ordering.
    for records in (train_records, val_records, test_records):
        records.sort(key=lambda r: r["image_id"])

    all_records = train_records + val_records + test_records
    for record in all_records:
        record.pop("abs_path", None)
        record.pop("sources", None)

    # Vocabulary from training captions only.
    train_captions = [
        caption for record in train_records for caption in record["captions"]
    ]
    vocab = Vocab.build(
        train_captions, max_size=int(args.vocab_size), min_freq=int(args.min_freq)
    )
    oov_tokens = 0
    total_tokens = 0
    for caption in train_captions:
        tokens = tokenize(caption)
        total_tokens += len(tokens)
        oov_tokens += sum(1 for token in tokens if token not in vocab.stoi)
    vocab.meta["oov_rate_train"] = round(oov_tokens / max(1, total_tokens), 6)
    vocab.meta["split_source"] = "train split only"

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "raw_dir": str(raw_dir),
        "manifest_root": str(manifest_root),
        "annotation_files": [str(path) for _hint, path in annotation_files],
        "counts": {
            "parsed_records": len(parsed),
            "resolved_records": len(resolved),
            "after_merge": len(merged),
            "after_dedupe": len(kept),
            "train": split_stats(train_records),
            "val": split_stats(val_records),
            "test": split_stats(test_records),
            "total": split_stats(all_records),
        },
        "dropped": {
            "missing_images": missing[:200],
            "missing_images_count": len(missing),
            "corrupt_images": dropped_corrupt[:200],
            "corrupt_images_count": len(dropped_corrupt),
            "empty_captions": dropped_empty[:200],
            "empty_captions_count": len(dropped_empty),
            "exact_duplicates": duplicates[:500],
            "exact_duplicates_count": len(duplicates),
        },
        "vocab": {
            "path": str(resolve_path(args.vocab)),
            "size": len(vocab),
            "oov_rate_train": vocab.meta.get("oov_rate_train"),
            "meta": vocab.meta,
        },
        "warnings": warnings,
    }

    if args.dry_run:
        print(json.dumps(report["counts"], ensure_ascii=False, indent=2, default=str))
        print("[prepare] dry run: nothing written")
        return report

    manifest_path = resolve_path(args.manifest)
    assert manifest_path is not None
    ensure_dir(str(manifest_path.parent))
    write_jsonl(all_records, str(manifest_path))

    manifests_dir = resolve_path(args.manifests_dir)
    assert manifests_dir is not None
    ensure_dir(str(manifests_dir))
    for name, records in (("train", train_records), ("val", val_records), ("test", test_records)):
        save_ids(
            [record["image_id"] for record in records],
            str(manifests_dir / f"{args.prefix}_{name}_ids.txt"),
        )

    vocab_path = resolve_path(args.vocab)
    assert vocab_path is not None
    vocab.save(str(vocab_path))

    report_path = resolve_path(args.report) if args.report else manifest_path.with_name(
        f"{manifest_path.stem.replace('_clean', '')}_prepare_report.json"
    )
    assert report_path is not None
    save_json(report, str(report_path))

    print("")
    print("[prepare] done")
    for name, records in (("train", train_records), ("val", val_records), ("test", test_records)):
        stats = split_stats(records)
        print(
            f"  {name:5s}: {stats['images']:6d} images | {stats['captions']:6d} captions | "
            f"{stats['classes']:4d} classes | {stats['captions_per_image']:.2f} caps/img"
        )
    print(
        f"  dropped: missing={len(missing)} corrupt={len(dropped_corrupt)} "
        f"empty={len(dropped_empty)} exact_duplicates={len(duplicates)}"
    )
    print(f"  vocab: {len(vocab)} tokens | train OOV={vocab.meta.get('oov_rate_train')}")
    print(f"  manifest: {manifest_path}")
    print(f"  IDs:      {manifests_dir}/{args.prefix}_{{train,val,test}}_ids.txt")
    print(f"  vocab:    {vocab_path}")
    print(f"  report:   {report_path}")
    for warning in warnings:
        print(f"  WARNING: {warning}")
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Food-500 Cap annotations")
    parser.add_argument("--raw-dir", default="data/raw/food500cap")
    parser.add_argument("--images-dir", default=None,
                        help="root containing the image files; defaults to --raw-dir")
    parser.add_argument("--annotations", default=None,
                        help="annotation file or directory; auto-detected by default")
    parser.add_argument("--test-annotations", default=None,
                        help="separate official test annotation file")
    parser.add_argument("--train-list", default=None,
                        help="optional text file with training image names")
    parser.add_argument("--val-list", default=None,
                        help="optional text file with validation image names")
    parser.add_argument("--test-list", default=None,
                        help="optional text file with test image names")
    parser.add_argument("--manifest", default="data/processed/food500cap_clean.jsonl")
    parser.add_argument("--manifests-dir", default="data/manifests")
    parser.add_argument("--vocab", default="data/vocab/shared_vocab_5000.json")
    parser.add_argument("--report", default=None)
    parser.add_argument("--prefix", default="food500cap")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab-size", type=int, default=5000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap parsed records (smoke tests only)")
    parser.add_argument("--skip-integrity-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    prepare(parse_args())
