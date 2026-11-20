# -*- coding: utf-8 -*-
"""Analyse the completed manual annotation sheet.

- reads manual_annotation_sheet.csv with a robust encoding (Excel may save GBK)
- re-saves it as UTF-8-with-BOM (keeps a .gbk.bak copy)
- writes reports/error_cases/manual_annotation_summary.md (+ .csv)
"""
import csv
import os
import re
import shutil
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE_DIR = os.path.join(ROOT, "reports", "error_cases")
CSV_PATH = os.path.join(CASE_DIR, "manual_annotation_sheet.csv")

KEYS = ["ingredient_misidentification", "cooking_method_error",
        "missing_ingredient", "unsupported_addition"]
CN = {"ingredient_misidentification": "食材误识别", "cooking_method_error": "烹饪方式错误",
      "missing_ingredient": "食材漏述", "unsupported_addition": "无依据添加"}


def read_rows(path):
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try:
            with open(path, encoding=enc, newline="") as f:
                return list(csv.DictReader(f)), enc
        except UnicodeDecodeError:
            continue
    raise RuntimeError("cannot decode csv")


def parse_suggested(s):
    m = re.search(r"\(([a-z_]+)\)", s or "")
    if m:
        return m.group(1)
    for k in KEYS:
        if k in (s or ""):
            return k
    return (s or "").strip()


rows, enc = read_rows(CSV_PATH)
print(f"read {len(rows)} rows with encoding={enc}")

man = Counter()
sug = Counter()
conf = defaultdict(Counter)          # suggested -> manual
agree = 0
per_model_exhibit = {"resnet18_gru": 0, "vit_transformer": 0}
for r in rows:
    mk = (r.get("manual_category") or "").strip()
    sk = parse_suggested(r.get("suggested_category"))
    man[mk] += 1
    sug[sk] += 1
    conf[sk][mk] += 1
    if mk == sk:
        agree += 1
    for m in ["resnet18_gru", "vit_transformer"]:
        flags = set((r.get(f"auto_flags_{m}") or "").split(";"))
        if mk in flags:
            per_model_exhibit[m] += 1

n = len(rows)
print("manual distribution:", dict(man))
print("suggested distribution:", dict(sug))
print(f"agreement (manual==suggested): {agree}/{n} = {agree/n*100:.1f}%")
for m in per_model_exhibit:
    print(f"{m}: manual-flag present in auto flags {per_model_exhibit[m]}/{n}")

# ---- re-save as UTF-8-sig if needed ----
if enc != "utf-8-sig":
    shutil.copy2(CSV_PATH, CSV_PATH + ".gbk.bak")
    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("re-saved as utf-8-sig; backup ->", os.path.basename(CSV_PATH) + ".gbk.bak")

# ---- markdown summary ----
md = []
md.append("# 任务一 错误分析 人工标注结果\n")
md.append(f"- 样本：{n} 例（四类各 15 例分层抽取）")
md.append(f"- 编码：原始 Excel 存为 GBK，已统一转为 UTF-8-sig（备份 `manual_annotation_sheet.csv.gbk.bak`）")
md.append(f"- 人工标注工具：`reports/error_cases/manual_annotation_review.html`\n")

md.append("## 1 人工判定的主错误分布\n")
md.append("| 错误类别 | 例数 | 占比 |")
md.append("|---|---:|---:|")
for k in KEYS:
    md.append(f"| {CN[k]}（`{k}`） | {man.get(k, 0)} | {man.get(k, 0)/n*100:.1f}% |")
md.append(f"| 合计 | {sum(man.values())} | 100% |\n")

md.append("## 2 自动建议 vs 人工判定（一致率）\n")
md.append(f"- 一致：**{agree}/{n} = {agree/n*100:.1f}%**")
md.append("\n混淆矩阵（行=自动建议，列=人工判定）：\n")
hdr = "| 自动＼人工 | " + " | ".join(CN[k] for k in KEYS) + " |"
md.append(hdr)
md.append("|" + "---|" * (len(KEYS) + 1))
for sk in KEYS:
    cells = " | ".join(str(conf[sk].get(mk, 0)) for mk in KEYS)
    md.append(f"| {CN[sk]} | {cells} |")
md.append("")

md.append("## 3 各模型的自动标志对人工主错误的复现率\n")
md.append("（该图人工判定的主错误，是否也出现在对应模型的自动标志里；自动标志偏宽，故接近 100%）\n")
md.append("| 模型 | 命中/总数 | 比例 |")
md.append("|---|---:|---:|")
for m in per_model_exhibit:
    md.append(f"| {m} | {per_model_exhibit[m]}/{n} | {per_model_exhibit[m]/n*100:.1f}% |")
md.append("")
md.append("## 4 结论\n")
md.append(f"1. 人工判定的主错误以**食材误识别**为主（{man.get('ingredient_misidentification',0)}/{n}，"
          f"{man.get('ingredient_misidentification',0)/n*100:.0f}%），与 Food-500 Cap 细粒度长尾特点一致。")
md.append(f"2. 自动建议与人工判定一致率 {agree/n*100:.1f}%，说明启发式初筛**只能粗筛**，"
          "最终类别必须以人工判定为准（支撑报告中“自动标注偏宽、需人工复核”的结论）。")
md.append("3. 人工判定的主错误几乎都能在自动标志中被复现（召回高），即自动标志适合“召回优先”的初筛，"
          "不适合直接当作精确类别统计。")
md.append("")

with open(os.path.join(CASE_DIR, "manual_annotation_summary.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")

# ---- csv summary ----
with open(os.path.join(CASE_DIR, "manual_annotation_summary.csv"), "w",
          encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["type", "category_en", "category_cn", "count", "pct"])
    for k in KEYS:
        w.writerow(["manual", k, CN[k], man.get(k, 0), f"{man.get(k,0)/n*100:.1f}"])
    w.writerow(["agreement", "", "", f"{agree}/{n}", f"{agree/n*100:.1f}"])

print("wrote manual_annotation_summary.md / .csv")
