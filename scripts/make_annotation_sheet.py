# -*- coding: utf-8 -*-
"""Build a 60-case manual-annotation worksheet (15 per error category).

For each of the four description-error categories pick the 15 clearest cases
(short, clean food terms) from outputs/task1/error_cases/error_cases.csv,
de-duplicating images across categories.  A human then fills manual_category
and notes; the automatic flags are only a rough pre-filter.

Outputs:
  reports/error_cases/manual_annotation_sheet.csv
  reports/error_cases/_annotation_images.txt
"""
import csv
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE_DIR = os.path.join(ROOT, "reports", "error_cases")
os.makedirs(CASE_DIR, exist_ok=True)
SRC = os.path.join(ROOT, "outputs", "task1", "error_cases", "error_cases.csv")

rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig")))

NONFOOD = {
    "slices", "slice", "surface", "color", "moist", "oven", "reddish", "adding", "full",
    "lightly", "oily", "seasoned", "various", "lot", "thin", "dried", "brown", "white",
    "plate", "bowl", "top", "side", "small", "large", "thick", "crispy", "dark", "red",
    "yellow", "green", "black", "skin", "texture", "inside", "bottom", "pieces", "piece",
    "hot", "cold", "warm", "sweet", "savory", "spicy", "creamy", "fresh", "raw", "whole",
    "half", "cooked", "prepared", "served", "topped", "covered", "filled", "mixed",
    "three", "two", "four", "five", "some", "many", "much", "little", "big", "round",
    "square", "golden", "light", "deep", "over", "under", "near", "close", "view",
    "combination", "up", "down", "out", "on", "in", "at", "of", "with", "and", "or",
    "is", "are", "was", "were", "this", "that", "these", "those", "it", "its", "be",
    "crust", "sauce", "juice", "liquid", "garnish", "decoration",
}

CATS = ["ingredient_misidentification", "cooking_method_error",
        "missing_ingredient", "unsupported_addition"]
CN = {"ingredient_misidentification": "食材误识别", "cooking_method_error": "烹饪方式错误",
      "missing_ingredient": "食材漏述", "unsupported_addition": "无依据添加"}
PER_CAT = 15


def tk(s):
    return [t for t in (s or "").split(";") if t]


def clean(ts):
    return [t for t in ts if t not in NONFOOD and len(t) > 1]


def score(cat, row, m):
    flags = set((row[f"flags_{m}"] or "").split(";"))
    miss = clean(tk(row[f"missing_food_{m}"]))
    add = clean(tk(row[f"added_food_{m}"]))
    cook = (row[f"cooking_{m}"] or "").strip()
    pred = cook.split("|")[0][5:] if cook.startswith("pred=") else ""
    if cat == "cooking_method_error":
        if "cooking_method_error" not in flags or not cook:
            return None
        return 100 + (30 if pred else 0) - len(miss) * 2
    if cat == "ingredient_misidentification":
        if "ingredient_misidentification" not in flags or not add or not miss:
            return None
        return 120 - len(add) * 6 - len(miss) * 3
    if cat == "missing_ingredient":
        if "missing_ingredient" not in flags or not (1 <= len(miss) <= 3):
            return None
        return 100 - len(miss) * 10 - len(add) * 6 + (25 if not add else 0)
    if cat == "unsupported_addition":
        if "unsupported_addition" not in flags or not (1 <= len(add) <= 3):
            return None
        return 100 - len(add) * 10 - len(miss) * 6 + (25 if len(miss) <= 1 else 0)
    return None


cands = defaultdict(list)
for row in rows:
    for m in ["resnet18_gru", "vit_transformer"]:
        for cat in CATS:
            s = score(cat, row, m)
            if s is not None:
                cands[cat].append((s, row["image_id"], row))

used = set()
picked = []
for cat in CATS:
    n = 0
    for s, img, row in sorted(cands[cat], key=lambda x: -x[0]):
        if img in used:
            continue
        used.add(img)
        picked.append((cat, row))
        n += 1
        if n >= PER_CAT:
            break
    print(f"{cat}: picked {n} (from {len(cands[cat])} candidates)")

OUT = ["image_id", "class", "image_local", "reference",
       "pred_resnet18_gru", "pred_vit_transformer", "suggested_category",
       "auto_flags_resnet18_gru", "auto_flags_vit_transformer",
       "manual_category", "notes"]

rel_images = []
with open(os.path.join(CASE_DIR, "manual_annotation_sheet.csv"), "w",
          encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=OUT)
    w.writeheader()
    for cat, r in picked:
        img_id = r["image_id"]
        rel = img_id.split("data/raw/food500cap/")[-1]
        local = "reports/figures/annotation/" + img_id.replace("/", "__")
        if rel not in rel_images:
            rel_images.append(rel)
        w.writerow({
            "image_id": img_id,
            "class": r["class"],
            "image_local": local,
            "reference": r["references"],
            "pred_resnet18_gru": r["prediction_resnet18_gru"],
            "pred_vit_transformer": r["prediction_vit_transformer"],
            "suggested_category": CN[cat] + f" ({cat})",
            "auto_flags_resnet18_gru": r["flags_resnet18_gru"],
            "auto_flags_vit_transformer": r["flags_vit_transformer"],
            "manual_category": "",
            "notes": "",
        })

with open(os.path.join(CASE_DIR, "_annotation_images.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(rel_images) + "\n")

print(f"manual_annotation_sheet.csv: {len(picked)} cases, {len(rel_images)} images")
