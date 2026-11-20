"""Four-way error flagging for task-1 captions.

The proposal asks for four description-error categories, with the original
image, references, both model predictions and a manual label preserved:

1. ``ingredient_misidentification`` -- the prediction names a food term that
   is absent from the references while a reference food term is missing.
2. ``cooking_method_error`` -- the cooking-method set differs from the
   references (e.g. "fried" vs "grilled").
3. ``missing_ingredient`` -- a reference food term is not mentioned.
4. ``unsupported_addition`` -- content appears in the prediction without
   support in any reference.

A sample can carry several flags; ``primary_<label>`` applies a fixed
priority for tables.  The CSV also contains blank ``manual_category_*`` and
``notes_*`` columns so a human can confirm or override the automatic flags.

Usage::

    python -m src.task1_caption.error_analysis \
        --manifest data/processed/food500cap_clean.jsonl \
        --image-root data/raw/food500cap \
        --pred resnet18_gru=outputs/task1/predictions/resnet18_gru_test_greedy.csv \
        --pred vit=outputs/task1/predictions/vit_transformer_test_greedy.csv \
        --out-dir outputs/task1/error_cases \
        --copy-images
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.common.io import ensure_dir, load_yaml, resolve_path, save_json, write_csv
from src.common.json_utils import load_jsonl
from src.common.metrics import tokenize
from src.task1_caption.evaluate import read_prediction_csv, reference_separator

CATEGORY_KEYS = [
    "ingredient_misidentification",
    "cooking_method_error",
    "missing_ingredient",
    "unsupported_addition",
]

STOPWORDS: Set[str] = {
    "a", "an", "the", "of", "with", "and", "or", "on", "in", "at", "is",
    "are", "was", "were", "this", "that", "these", "those", "some", "any",
    "to", "from", "for", "as", "by", "into", "over", "under", "near",
    "image", "photo", "picture", "food", "meal", "dish", "plate", "bowl",
    "platter", "table", "background", "portion", "serving", "cuisine",
    "style", "close", "closeup", "top", "view", "side", "front", "made",
    "cooked", "prepared", "served", "topped", "covered", "filled", "mixed",
    "containing", "contains", "has", "have", "includes", "including",
    "variety", "assortment", "delicious", "tasty", "yummy", "healthy",
    "homemade", "restaurant", "wooden", "metal", "glass", "paper",
    "napkin", "fork", "spoon", "knife", "fresh", "green", "red", "yellow",
    "white", "brown", "black", "small", "large", "big", "thin", "thick",
    "sliced", "chopped", "diced", "minced", "whole", "half", "piece",
    "pieces", "hot", "cold", "warm", "sweet", "savory", "spicy", "crispy",
    "creamy", "colorful", "beautiful", "appetizing", "looks", "looking",
}

COOKING_TERMS: Set[str] = {
    "fried", "fry", "deep-fried", "pan-fried", "stir-fried", "grilled",
    "grill", "baked", "bake", "roasted", "roast", "steamed", "steam",
    "boiled", "boil", "sauteed", "sautéed", "saute", "sauté", "simmered",
    "poached", "braised", "smoked", "toasted", "marinated", "pickled",
    "fermented", "dried", "uncooked", "raw", "stewed", "chargrilled",
    "char-grilled", "barbecued", "bbq", "broiled", "pan-seared", "seared",
}

DEFAULT_FOOD_TERMS: Set[str] = {
    "rice", "noodle", "noodles", "pasta", "bread", "toast", "bun", "cake",
    "cookie", "pie", "pizza", "burger", "sandwich", "taco", "burrito",
    "sushi", "soup", "salad", "stew", "curry", "chicken", "beef", "pork",
    "lamb", "fish", "salmon", "tuna", "shrimp", "prawn", "crab", "tofu",
    "egg", "eggs", "cheese", "milk", "yogurt", "butter", "cream", "sauce",
    "tomato", "potato", "onion", "garlic", "carrot", "broccoli", "spinach",
    "lettuce", "cabbage", "mushroom", "pepper", "cucumber", "corn", "bean",
    "beans", "pea", "peas", "lentil", "avocado", "lemon", "lime", "orange",
    "apple", "banana", "berry", "strawberry", "grape", "mango", "pineapple",
    "watermelon", "olive", "nut", "nuts", "almond", "peanut", "seed",
    "seeds", "herb", "basil", "parsley", "cilantro", "ginger", "chili",
    "chilli", "spice", "oil", "honey", "sugar", "flour", "oat", "oats",
    "ham", "bacon", "sausage", "meatball", "dumpling", "wrap", "roll",
    "pancake", "waffle", "fries", "chips", "nugget", "fillet", "steak",
}


def _parse_run_args(items: Sequence[str]) -> List[Tuple[str, str]]:
    runs: List[Tuple[str, str]] = []
    labels = set()
    for item in items:
        if "=" in item:
            label, path = item.split("=", 1)
        else:
            path = item
            label = Path(item).stem
        label, path = label.strip(), path.strip()
        if not label or not path:
            raise ValueError(f"bad --pred value {item!r}, expected LABEL=PATH")
        if label in labels:
            raise ValueError(f"duplicate label {label!r}")
        labels.add(label)
        runs.append((label, path))
    return runs


def _safe_label(label: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in label).strip("_") or "model"


def _load_lexicon(
    manifest_records: Dict[str, Dict[str, Any]],
    references_by_id: Dict[str, List[str]],
    lexicon_path: Optional[str],
    min_count: int,
) -> Set[str]:
    lexicon: Set[str] = set(DEFAULT_FOOD_TERMS)

    # Food-500 Cap category names are the most reliable food-term source.
    for record in manifest_records.values():
        class_name = record.get("class")
        if class_name:
            lexicon.update(tokenize(str(class_name).replace("_", " ").replace("-", " ")))

    # Optional curated file: one term (possibly multi-word) per line.
    if lexicon_path:
        p = resolve_path(lexicon_path)
        if p is not None and p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        lexicon.update(tokenize(line))

    # Mine frequent content words from the references as a fallback.
    counts: Counter = Counter()
    for refs in references_by_id.values():
        seen: Set[str] = set()
        for ref in refs:
            seen.update(tokenize(ref))
        counts.update(seen)
    for token, count in counts.items():
        if count < int(min_count):
            continue
        if token in STOPWORDS or token in COOKING_TERMS:
            continue
        lexicon.add(token)

    lexicon -= STOPWORDS
    lexicon -= COOKING_TERMS
    return lexicon


def _analyze_one(
    prediction: str,
    references: Sequence[str],
    food_terms: Set[str],
) -> Dict[str, Any]:
    pred_tokens = set(tokenize(prediction)) - STOPWORDS
    ref_tokens: Set[str] = set()
    for ref in references:
        ref_tokens.update(tokenize(ref))
    ref_tokens -= STOPWORDS

    pred_food = pred_tokens & food_terms
    ref_food = ref_tokens & food_terms
    pred_cooking = set(tokenize(prediction)) & COOKING_TERMS
    ref_cooking = set(tokenize(" ".join(references))) & COOKING_TERMS

    missing_food = ref_food - pred_food
    added_food = pred_food - ref_food
    pred_other = pred_tokens - food_terms - COOKING_TERMS
    ref_other = ref_tokens - food_terms - COOKING_TERMS
    added_other = pred_other - ref_other

    flags: List[str] = []
    if missing_food:
        flags.append("missing_ingredient")
    if added_food or added_other:
        flags.append("unsupported_addition")
    if missing_food and added_food:
        flags.append("ingredient_misidentification")
    if pred_cooking != ref_cooking and (pred_cooking or ref_cooking):
        flags.append("cooking_method_error")

    if missing_food and added_food:
        primary = "ingredient_misidentification"
    elif pred_cooking != ref_cooking and (pred_cooking or ref_cooking):
        primary = "cooking_method_error"
    elif missing_food:
        primary = "missing_ingredient"
    elif added_food or added_other:
        primary = "unsupported_addition"
    else:
        primary = "none"

    return {
        "flags": flags,
        "primary": primary,
        "missing_food": sorted(missing_food),
        "added_food": sorted(added_food),
        "added_other": sorted(added_other),
        "pred_cooking": sorted(pred_cooking),
        "ref_cooking": sorted(ref_cooking),
    }


def _join_tokens(tokens: Sequence[str], limit: int = 12) -> str:
    return ";".join(tokens[:limit])


def _resolve_image_path(record: Dict[str, Any], image_root: Optional[str]) -> str:
    if not record:
        return ""
    value = str(record.get("image", "")).strip()
    if not value:
        return ""
    image_path = Path(value)
    if image_path.is_absolute():
        return str(image_path)
    candidates: List[Path] = []
    root = resolve_path(image_root) if image_root else None
    if root is not None:
        candidates.append(root / image_path)
    fallback = resolve_path(value)
    if fallback is not None:
        candidates.append(fallback)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0]) if candidates else value


def analyze(
    manifest_path: str,
    image_root: Optional[str],
    runs: Sequence[Tuple[str, str]],
    protocol: Dict[str, Any],
    out_dir: str,
    lexicon_path: Optional[str] = None,
    min_lexicon_count: int = 3,
    max_cases: int = 200,
    copy_images: bool = False,
) -> Dict[str, Any]:
    separator = reference_separator(protocol)
    parsed = {label: read_prediction_csv(path, separator=separator) for label, path in runs}

    manifest_records: Dict[str, Dict[str, Any]] = {}
    for record in load_jsonl(manifest_path):
        manifest_records[str(record["image_id"])] = record

    all_ids: Set[str] = set()
    references_by_id: Dict[str, List[str]] = {}
    for label, (ids, _preds, refs) in parsed.items():
        all_ids.update(ids)
        for image_id, ref_group in zip(ids, refs):
            references_by_id.setdefault(image_id, list(ref_group))

    food_terms = _load_lexicon(
        manifest_records, references_by_id, lexicon_path, min_lexicon_count
    )

    per_model_data: Dict[str, Tuple[List[str], List[str], List[List[str]]]] = {}
    for label, (ids, preds, refs) in parsed.items():
        per_model_data[label] = (ids, preds, refs)

    ordered_ids = sorted(all_ids)
    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "manifest": str(resolve_path(manifest_path)),
        "image_root": str(resolve_path(image_root)) if image_root else None,
        "num_images": len(ordered_ids),
        "food_lexicon_size": len(food_terms),
        "runs": {},
        "missing_from_manifest": sorted(set(ordered_ids) - set(manifest_records))[:50],
        "category_keys": CATEGORY_KEYS,
    }

    flag_counts: Dict[str, Counter] = {label: Counter() for label, _ in runs}
    primary_counts: Dict[str, Counter] = {label: Counter() for label, _ in runs}
    detail_counts: Dict[str, Counter] = {label: Counter() for label, _ in runs}

    for image_id in ordered_ids:
        record = manifest_records.get(image_id, {})
        refs = references_by_id.get(image_id, [])
        row: Dict[str, Any] = {
            "image_id": image_id,
            "class": record.get("class", ""),
            "image_path": _resolve_image_path(record, image_root),
            "references": " ||| ".join(refs),
        }
        any_flag = False
        flag_total = 0
        for label, _path in runs:
            ids, preds, refs_list = per_model_data[label]
            lookup = {iid: (pred, ref) for iid, pred, ref in zip(ids, preds, refs_list)}
            if image_id not in lookup:
                continue
            prediction, ref_group = lookup[image_id]
            analysis = _analyze_one(prediction, ref_group, food_terms)
            safe = _safe_label(label)
            flags = analysis["flags"]
            primary = analysis["primary"]
            flag_counts[label].update(flags)
            primary_counts[label][primary] += 1
            detail_counts[label]["missing_food_total"] += len(analysis["missing_food"])
            detail_counts[label]["added_food_total"] += len(analysis["added_food"])
            detail_counts[label]["added_other_total"] += len(analysis["added_other"])

            row[f"prediction_{safe}"] = prediction
            row[f"flags_{safe}"] = ";".join(flags)
            row[f"primary_{safe}"] = primary
            row[f"missing_food_{safe}"] = _join_tokens(analysis["missing_food"])
            row[f"added_food_{safe}"] = _join_tokens(analysis["added_food"])
            row[f"added_other_{safe}"] = _join_tokens(analysis["added_other"])
            row[f"cooking_{safe}"] = (
                f"pred={','.join(analysis['pred_cooking'])}|ref={','.join(analysis['ref_cooking'])}"
            )
            row[f"manual_category_{safe}"] = ""
            row[f"notes_{safe}"] = ""

            if flags:
                any_flag = True
                flag_total += len(flags)

        if any_flag or max_cases <= 0:
            row["_flag_total"] = flag_total
            rows.append(row)

    # Summary is over all images, not only the saved cases.
    for label, _path in runs:
        n = len(per_model_data[label][0])
        summary["runs"][label] = {
            "predictions": str(resolve_path(_path)),
            "n": n,
            "flags": {key: flag_counts[label][key] for key in CATEGORY_KEYS},
            "flags_pct": {
                key: (100.0 * flag_counts[label][key] / n if n else 0.0)
                for key in CATEGORY_KEYS
            },
            "primary": dict(primary_counts[label]),
            "details": dict(detail_counts[label]),
        }

    rows.sort(key=lambda item: (-int(item.get("_flag_total", 0)), item["image_id"]))
    if max_cases and max_cases > 0:
        rows = rows[: int(max_cases)]

    fieldnames: List[str] = ["image_id", "class", "image_path", "references"]
    for label, _ in runs:
        safe = _safe_label(label)
        fieldnames += [
            f"prediction_{safe}",
            f"flags_{safe}",
            f"primary_{safe}",
            f"missing_food_{safe}",
            f"added_food_{safe}",
            f"added_other_{safe}",
            f"cooking_{safe}",
            f"manual_category_{safe}",
            f"notes_{safe}",
        ]

    out_path_dir = ensure_dir(out_dir)
    write_csv(rows, str(out_path_dir / "error_cases.csv"), fieldnames=fieldnames)
    save_json(summary, str(out_path_dir / "summary.json"))
    _write_markdown(summary, str(out_path_dir / "summary.md"))

    if copy_images:
        image_dir = ensure_dir(str(out_path_dir / "images"))
        for row in rows:
            image_path = Path(str(row.get("image_path", "")))
            if image_path.exists():
                target = image_dir / (str(row["image_id"]).replace("/", "_").replace("\\", "_"))
                shutil.copy2(image_path, target)

    return summary


def _write_markdown(summary: Dict[str, Any], path: str) -> None:
    lines = [
        "# Task-1 error analysis",
        "",
        f"- manifest: `{summary['manifest']}`",
        f"- images: {summary['num_images']}",
        f"- food lexicon size: {summary['food_lexicon_size']}",
        "- manual categories (fill `manual_category_*` in `error_cases.csv`): "
        + ", ".join(f"`{key}`" for key in CATEGORY_KEYS),
        "",
        "| model | n | ingredient mis-id | cooking method | missing ingredient | unsupported addition |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, run in summary["runs"].items():
        pct = run["flags_pct"]
        lines.append(
            f"| {label} | {run['n']} | {pct['ingredient_misidentification']:.1f}% | "
            f"{pct['cooking_method_error']:.1f}% | {pct['missing_ingredient']:.1f}% | "
            f"{pct['unsupported_addition']:.1f}% |"
        )
    lines += [
        "",
        "Flags may overlap; the CSV has a `primary_<model>` column with the "
        "priority ingredient-misidentification > cooking-method > missing > "
        "unsupported addition.",
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Flag task-1 description errors")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--pred", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--protocol", default="configs/eval_protocol.yaml")
    parser.add_argument("--food-lexicon", default="data/vocab/food_lexicon.txt")
    parser.add_argument("--min-lexicon-count", type=int, default=3)
    parser.add_argument("--out-dir", default="outputs/task1/error_cases")
    parser.add_argument("--max-cases", type=int, default=200,
                        help="0 keeps every flagged case")
    parser.add_argument("--copy-images", action="store_true")
    args = parser.parse_args(argv)

    protocol_path = resolve_path(args.protocol)
    protocol = load_yaml(args.protocol) if protocol_path and protocol_path.exists() else {}
    runs = _parse_run_args(args.pred)
    summary = analyze(
        manifest_path=args.manifest,
        image_root=args.image_root,
        runs=runs,
        protocol=protocol,
        out_dir=args.out_dir,
        lexicon_path=args.food_lexicon,
        min_lexicon_count=args.min_lexicon_count,
        max_cases=args.max_cases,
        copy_images=args.copy_images,
    )
    for label, run in summary["runs"].items():
        print(
            f"[error] {label}: n={run['n']} "
            f"mis-id={run['flags_pct']['ingredient_misidentification']:.1f}% "
            f"cooking={run['flags_pct']['cooking_method_error']:.1f}% "
            f"missing={run['flags_pct']['missing_ingredient']:.1f}% "
            f"addition={run['flags_pct']['unsupported_addition']:.1f}%"
        )
    print(f"[error] wrote {resolve_path(args.out_dir)}/error_cases.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
