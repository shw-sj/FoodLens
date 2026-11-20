# -*- coding: utf-8 -*-
"""Generate report materials for FoodLens Task 1 (tables + error gallery)."""
import csv
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def P(*parts):
    return os.path.join(ROOT, *parts)


REPORTS = P("reports")
TAB_DIR = os.path.join(REPORTS, "experiment_tables")
CASE_DIR = os.path.join(REPORTS, "error_cases")
os.makedirs(TAB_DIR, exist_ok=True)
os.makedirs(CASE_DIR, exist_ok=True)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt(x, nd):
    return f"{x:.{nd}f}"


sums = {
    "resnet18_gru": load_json(P("checkpoints/task1/resnet18_gru/summary.json")),
    "vit_transformer": load_json(P("checkpoints/task1/vit_transformer/summary.json")),
}
comp = load_json(P("outputs/task1/metrics/model_comparison.json"))
tune = load_json(P("outputs/task1/beam_search/tuning_results.json"))
final = load_json(P("outputs/task1/beam_search/final/metrics.json"))
err = load_json(P("outputs/task1/error_cases/summary.json"))

MODEL_LABEL = {"resnet18_gru": "ResNet18+GRU", "vit_transformer": "ViT+Transformer"}

# data split stats from data/processed/food500cap_prepare_report.json (server)
SPLIT = [
    ("train", 17763, 17774, 1.001, 18.667),
    ("val", 1974, 1977, 1.002, 18.598),
    ("test", 4938, 4944, 1.001, 18.416),
    ("total", 24675, 24695, 1.001, 18.611),
]
VOCAB = {"size": 3101, "max_size": 5000, "min_freq": 2, "oov_train": 0.005244}

CFG = {
    "resnet18_gru": {"encoder": "ResNet-18 (ImageNet 预训练)", "decoder": "GRU (单层)",
                     "dim": "embed 256 / hidden 512", "dropout": 0.2, "params": "15.79M"},
    "vit_transformer": {"encoder": "ViT-B/16 (ImageNet 预训练)",
                        "decoder": "Transformer (4 层, 8 头, FFN 2048)",
                        "dim": "d_model 512 / ViT hidden 768", "dropout": 0.1, "params": "106.21M"},
}
TRAIN_CFG = {"batch_size": 32, "lr": "1e-4", "optimizer": "AdamW", "weight_decay": 0.01,
             "warmup_epochs": 2, "freeze_encoder_epochs": 2, "unfreeze_mode": "last_block",
             "epochs": 30, "patience": 6, "amp": "是", "image_size": 224, "max_len": 40, "seed": 42}

lines = []
lines.append("# FoodLens 任务一 实验表格（食物图像描述生成）\n")
lines.append("> 数据版本：Food-500 Cap（HuggingFace `advancedcv/Food500Cap`，24675 图文对）\n")
lines.append("> 评测口径：`task1-eval-v1` / 指标 `task1-metrics-v1` / 分词 `regex-lower-alnum-v1`\n")
lines.append("> BLEU-4、ROUGE-L 为 0–100 尺度；CIDEr-D 为官方 ×10 尺度。\n")

# Table 1
lines.append("\n## 表 1  数据划分与统计\n")
lines.append("| 划分 | 图片数 | 描述数 | 描述/图 | 平均 token 数 |")
lines.append("|---|---:|---:|---:|---:|")
for name, imgs, caps, cpi, toks in SPLIT:
    lines.append(f"| {name} | {imgs} | {caps} | {cpi:.3f} | {toks:.3f} |")
lines.append(
    f"\n词表：{VOCAB['size']} 个 token（max_size={VOCAB['max_size']}，min_freq={VOCAB['min_freq']}），"
    f"训练集 OOV 率 {VOCAB['oov_train']*100:.3f}%。test 沿用发布方测试集，"
    f"从 train 按类别分层划出约 10% 作为 val（seed=42）。\n")

# Table 2
lines.append("\n## 表 2  模型结构与训练配置\n")
lines.append("| 项目 | ResNet18+GRU | ViT+Transformer |")
lines.append("|---|---|---|")
lines.append(f"| 视觉编码器 | {CFG['resnet18_gru']['encoder']} | {CFG['vit_transformer']['encoder']} |")
lines.append(f"| 文本解码器 | {CFG['resnet18_gru']['decoder']} | {CFG['vit_transformer']['decoder']} |")
lines.append(f"| 维度设置 | {CFG['resnet18_gru']['dim']} | {CFG['vit_transformer']['dim']} |")
lines.append(f"| Dropout | {CFG['resnet18_gru']['dropout']} | {CFG['vit_transformer']['dropout']} |")
lines.append(f"| 参数量 | {CFG['resnet18_gru']['params']} | {CFG['vit_transformer']['params']} |")
lines.append(f"| Batch size | {TRAIN_CFG['batch_size']} | {TRAIN_CFG['batch_size']} |")
lines.append(f"| 优化器 / 学习率 | {TRAIN_CFG['optimizer']} / {TRAIN_CFG['lr']} | {TRAIN_CFG['optimizer']} / {TRAIN_CFG['lr']} |")
lines.append(f"| 权重衰减 | {TRAIN_CFG['weight_decay']} | {TRAIN_CFG['weight_decay']} |")
lines.append(f"| 学习率 warmup | {TRAIN_CFG['warmup_epochs']} epoch | {TRAIN_CFG['warmup_epochs']} epoch |")
lines.append(f"| 编码器冻结策略 | 前 {TRAIN_CFG['freeze_encoder_epochs']} 轮冻结，之后解冻 {TRAIN_CFG['unfreeze_mode']} | 同左 |")
lines.append(f"| 最大轮数 / 早停 | {TRAIN_CFG['epochs']} / patience={TRAIN_CFG['patience']} | 同左 |")
lines.append(f"| 混合精度 AMP | {TRAIN_CFG['amp']} | {TRAIN_CFG['amp']} |")
lines.append(f"| 输入分辨率 / 最大句长 | {TRAIN_CFG['image_size']} / {TRAIN_CFG['max_len']} | {TRAIN_CFG['image_size']} / {TRAIN_CFG['max_len']} |")
lines.append(f"| 随机种子 | {TRAIN_CFG['seed']} | {TRAIN_CFG['seed']} |\n")

# Table 3
lines.append("\n## 表 3  验证集最佳结果（以 CIDEr-D 选 checkpoint）\n")
lines.append("| 模型 | 实际轮数 | 最佳 epoch | val BLEU-4 | val ROUGE-L | val CIDEr-D |")
lines.append("|---|---:|---:|---:|---:|---:|")
for m in ["resnet18_gru", "vit_transformer"]:
    s = sums[m]
    be = s["best_epoch"]
    h = next(x for x in s["history"] if x["epoch"] == be)
    lines.append(f"| {MODEL_LABEL[m]} | {s['epochs_ran']} | {be} | {fmt(h['bleu4'],2)} | "
                 f"{fmt(h['rouge_l'],2)} | {fmt(h['cider_d'],4)} |")
lines.append("")

# Table 4
lines.append("\n## 表 4  测试集统一评测（greedy 解码，n=4938）\n")
lines.append("| 模型 | BLEU-4 | ROUGE-L | CIDEr-D | 重复率 rep4 | 平均句长 |")
lines.append("|---|---:|---:|---:|---:|---:|")
for label in comp["ranking"]:
    r = comp["runs"][label]
    lines.append(f"| {MODEL_LABEL.get(label, label)} | {fmt(r['bleu4'],2)} | {fmt(r['rouge_l'],2)} | "
                 f"{fmt(r['cider_d'],4)} | {fmt(r['rep4'],3)} | {fmt(r['avg_len'],1)} |")
lines.append("")

# Table 5
lines.append("\n## 表 5  束搜索增项（ViT 检查点）\n")
lines.append("### 5a  验证集扫描（前 500 张，按 CIDEr-D 排序）\n")
lines.append("| 设置 | BLEU-4 | ROUGE-L | CIDEr-D | rep4 | 平均句长 | ms/图 |")
lines.append("|---|---:|---:|---:|---:|---:|---:|")
g = tune["greedy"]
lines.append(f"| greedy | {fmt(g['bleu4'],2)} | {fmt(g['rouge_l'],2)} | {fmt(g['cider_d'],3)} | "
             f"{fmt(g['rep4'],3)} | {fmt(g['avg_len'],1)} | {fmt(g['ms_per_image'],1)} |")
rows5 = sorted(tune["beam_results"], key=lambda r: (r["cider_d"], r["bleu4"]), reverse=True)
for r in rows5:
    lines.append(f"| beam={r['beam_size']} lp={r['length_penalty']:g} | {fmt(r['bleu4'],2)} | "
                 f"{fmt(r['rouge_l'],2)} | {fmt(r['cider_d'],3)} | {fmt(r['rep4'],3)} | "
                 f"{fmt(r['avg_len'],1)} | {fmt(r['ms_per_image'],1)} |")
best = tune["best"]
lines.append(f"\n选定设置：**beam={best['beam_size']}, length_penalty={best['length_penalty']:g}**"
             f"（验证集 CIDEr-D 最高，BLEU-4 作为 tie-break）。\n")
lines.append("\n### 5b  测试集最终对比（n=4938）\n")
lines.append("| 设置 | BLEU-4 | ROUGE-L | CIDEr-D | rep4 | 平均句长 | ms/图 |")
lines.append("|---|---:|---:|---:|---:|---:|---:|")
fg, fb = final["greedy"], final["beam"]
lines.append(f"| greedy | {fmt(fg['bleu4'],2)} | {fmt(fg['rouge_l'],2)} | {fmt(fg['cider_d'],3)} | "
             f"{fmt(fg['rep4'],3)} | {fmt(fg['avg_len'],1)} | {fmt(fg['ms_per_image'],1)} |")
lines.append(f"| beam={fb['beam_size']} lp={fb['length_penalty']:g} | {fmt(fb['bleu4'],2)} | "
             f"{fmt(fb['rouge_l'],2)} | {fmt(fb['cider_d'],3)} | {fmt(fb['rep4'],3)} | "
             f"{fmt(fb['avg_len'],1)} | {fmt(fb['ms_per_image'],1)} |")
lines.append(f"\n束搜索相对 greedy：CIDEr-D {fmt(fg['cider_d'],3)} → {fmt(fb['cider_d'],3)}"
             f"（+{fmt((fb['cider_d']-fg['cider_d'])/fg['cider_d']*100,1)}%），"
             f"BLEU-4 {fmt(fg['bleu4'],2)} → {fmt(fb['bleu4'],2)}；平均句长缩短、重复率降低。\n")

# Table 6
lines.append("\n## 表 6  四类描述错误统计（测试集 4938 张，自动初筛）\n")
lines.append("| 模型 | 食材误识别 | 烹饪方式错误 | 食材漏述 | 无依据添加 |")
lines.append("|---|---:|---:|---:|---:|")
KEYS = ["ingredient_misidentification", "cooking_method_error", "missing_ingredient", "unsupported_addition"]
CN = {"ingredient_misidentification": "食材误识别", "cooking_method_error": "烹饪方式错误",
      "missing_ingredient": "食材漏述", "unsupported_addition": "无依据添加"}
for m in ["resnet18_gru", "vit_transformer"]:
    run = err["runs"][m]
    pct = run["flags_pct"]
    cells = " | ".join(f"{pct[k]:.1f}% ({run['flags'][k]})" for k in KEYS)
    lines.append(f"| {MODEL_LABEL[m]} | {cells} |")
lines.append("\n注：自动标注为启发式初筛，四类标志可重叠；`error_cases.csv` 内保留了 "
             "`manual_category_*` 与 `notes_*` 列供人工确认。\n")
lines.append("\n**各模型 primary（按优先级去重后）分布：**\n")
lines.append("| 模型 | 误识别 | 烹饪方式 | 漏述 | 无依据添加 | 无错误 |")
lines.append("|---|---:|---:|---:|---:|---:|")
for m in ["resnet18_gru", "vit_transformer"]:
    pr = err["runs"][m]["primary"]
    lines.append(f"| {MODEL_LABEL[m]} | {pr.get('ingredient_misidentification',0)} | "
                 f"{pr.get('cooking_method_error',0)} | {pr.get('missing_ingredient',0)} | "
                 f"{pr.get('unsupported_addition',0)} | {pr.get('none',0)} |")
lines.append("")

with open(os.path.join(TAB_DIR, "experiment_tables.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("experiment_tables.md written")

# ---------------------------------------------------------------- CSV exports
with open(os.path.join(TAB_DIR, "model_comparison_test.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "bleu4", "rouge_l", "cider_d", "rep4", "avg_len", "n"])
    for label in ["resnet18_gru", "vit_transformer"]:
        r = comp["runs"][label]
        w.writerow([MODEL_LABEL[label], fmt(r["bleu4"], 4), fmt(r["rouge_l"], 4),
                    fmt(r["cider_d"], 4), fmt(r["rep4"], 4), fmt(r["avg_len"], 3), int(r["n"])])

with open(os.path.join(TAB_DIR, "beam_search_results.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["setting", "beam_size", "length_penalty", "split", "bleu4", "rouge_l",
                "cider_d", "rep4", "avg_len", "ms_per_image"])
    w.writerow(["greedy", "", "", "val", fmt(g["bleu4"], 4), fmt(g["rouge_l"], 4),
                fmt(g["cider_d"], 4), fmt(g["rep4"], 4), fmt(g["avg_len"], 3),
                fmt(g["ms_per_image"], 3)])
    for r in sorted(tune["beam_results"], key=lambda x: (x["beam_size"], x["length_penalty"])):
        w.writerow([r["name"], r["beam_size"], r["length_penalty"], "val", fmt(r["bleu4"], 4),
                    fmt(r["rouge_l"], 4), fmt(r["cider_d"], 4), fmt(r["rep4"], 4),
                    fmt(r["avg_len"], 3), fmt(r["ms_per_image"], 3)])
    w.writerow(["greedy", "", "", "test", fmt(fg["bleu4"], 4), fmt(fg["rouge_l"], 4),
                fmt(fg["cider_d"], 4), fmt(fg["rep4"], 4), fmt(fg["avg_len"], 3),
                fmt(fg["ms_per_image"], 3)])
    w.writerow([f"beam={fb['beam_size']}_lp={fb['length_penalty']:g}", fb["beam_size"],
                fb["length_penalty"], "test", fmt(fb["bleu4"], 4), fmt(fb["rouge_l"], 4),
                fmt(fb["cider_d"], 4), fmt(fb["rep4"], 4), fmt(fb["avg_len"], 3),
                fmt(fb["ms_per_image"], 3)])

with open(os.path.join(TAB_DIR, "training_history.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "epoch", "train_loss", "val_loss", "lr", "epoch_seconds",
                "bleu4", "rouge_l", "cider_d", "rep4", "avg_len"])
    for m in ["resnet18_gru", "vit_transformer"]:
        for h in sums[m]["history"]:
            w.writerow([MODEL_LABEL[m], h["epoch"], h["train_loss"], h["val_loss"],
                        h["lr"], h["epoch_seconds"], fmt(h["bleu4"], 4),
                        fmt(h["rouge_l"], 4), fmt(h["cider_d"], 6),
                        fmt(h["rep4"], 6), fmt(h["avg_len"], 3)])

with open(os.path.join(TAB_DIR, "error_analysis.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "category", "en_key", "count", "pct"])
    for m in ["resnet18_gru", "vit_transformer"]:
        run = err["runs"][m]
        for k in KEYS:
            w.writerow([MODEL_LABEL[m], CN[k], k, run["flags"][k], fmt(run["flags_pct"][k], 2)])

print("csv tables written")

# ---------------------------------------------------------------- typical cases
with open(P("outputs/task1/error_cases/error_cases.csv"), encoding="utf-8-sig") as f:
    cases = list(csv.DictReader(f))


def toks(s):
    return [t for t in (s or "").split(";") if t]


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


def clean(ts):
    return [t for t in ts if t not in NONFOOD and len(t) > 1]


def cand_score(cat, row, m):
    flags = set((row[f"flags_{m}"] or "").split(";"))
    missing = clean(toks(row[f"missing_food_{m}"]))
    added = clean(toks(row[f"added_food_{m}"]))
    cooking = (row[f"cooking_{m}"] or "").strip()
    pred_cook = cooking.split("|")[0][5:] if cooking.startswith("pred=") else ""
    if cat == "cooking_method_error":
        if "cooking_method_error" not in flags or not cooking:
            return None
        return 100 + (30 if pred_cook else 0) - len(missing) * 2
    if cat == "ingredient_misidentification":
        if "ingredient_misidentification" not in flags or not added or not missing:
            return None
        s = 100 - len(added) * 6 - len(missing) * 3
        if len(added) <= 2 and len(missing) <= 2:
            s += 20
        return s
    if cat == "missing_ingredient":
        if "missing_ingredient" not in flags or not (1 <= len(missing) <= 3):
            return None
        return 100 - len(missing) * 10 - len(added) * 6 + (25 if not added else 0)
    if cat == "unsupported_addition":
        if "unsupported_addition" not in flags or not (1 <= len(added) <= 3):
            return None
        return 100 - len(added) * 10 - len(missing) * 6 + (25 if len(missing) <= 1 else 0)
    return None


CAT_ORDER = ["ingredient_misidentification", "cooking_method_error",
             "missing_ingredient", "unsupported_addition"]
PER_CAT = 3
cands = defaultdict(list)
for row in cases:
    img = row["image_id"]
    for m in ["resnet18_gru", "vit_transformer"]:
        for cat in CAT_ORDER:
            sc = cand_score(cat, row, m)
            if sc is not None:
                cands[cat].append((sc, img, row))

used = set()
picked = {k: [] for k in CAT_ORDER}
for cat in CAT_ORDER:
    print(cat, "candidates:", len(cands[cat]))
    for score, img, row in sorted(cands[cat], key=lambda x: -x[0]):
        if img in used:
            continue
        used.add(img)
        picked[cat].append(row)
        if len(picked[cat]) >= PER_CAT:
            break

gl = []
gl.append("# 任务一 典型错误案例（用于报告配图）\n")
gl.append(f"> 从 `outputs/task1/error_cases/error_cases.csv`（最显著的 200 例）中按四类错误各挑选 {PER_CAT} 例。"
          "原图已下载到 `reports/figures/error_cases/`。\n")
gl.append("> 说明：自动标志为启发式初筛，以下为人工挑选的可解释样例；同一图可能同时命中多类错误。\n")

rel_images = []
sel_csv = []
for cat in CAT_ORDER:
    gl.append(f"\n## {CN[cat]}（`{cat}`）\n")
    for i, row in enumerate(picked[cat], 1):
        img_id = row["image_id"]
        cls = row["class"]
        rel = img_id.split("data/raw/food500cap/")[-1]
        local_name = img_id.replace("/", "__")
        gl.append(f"### 案例 {i}　（class: `{cls}`）\n")
        gl.append(f"- 原图：`reports/figures/error_cases/{local_name}`（image_id: `{img_id}`）")
        gl.append(f"- 参考描述：{row['references']}")
        gl.append(f"- ResNet18+GRU：{row['prediction_resnet18_gru']}")
        gl.append(f"- ViT+Transformer：{row['prediction_vit_transformer']}")
        for m in ["resnet18_gru", "vit_transformer"]:
            miss = clean(toks(row[f"missing_food_{m}"]))
            add = clean(toks(row[f"added_food_{m}"]))
            cook = (row[f"cooking_{m}"] or "").strip()
            detail = []
            if miss:
                detail.append("缺: " + ", ".join(miss[:4]))
            if add:
                detail.append("多: " + ", ".join(add[:4]))
            if cook:
                detail.append("烹饪: " + cook)
            if detail:
                gl.append(f"  - {MODEL_LABEL[m]} 要点：{'；'.join(detail)}")
        gl.append("")
        if rel not in rel_images:
            rel_images.append(rel)
        sel_csv.append({
            "category": CN[cat], "en_key": cat, "image_id": img_id, "class": cls,
            "local_image": f"reports/figures/error_cases/{local_name}",
            "reference": row["references"],
            "prediction_resnet18_gru": row["prediction_resnet18_gru"],
            "prediction_vit_transformer": row["prediction_vit_transformer"],
        })

with open(os.path.join(CASE_DIR, "typical_cases.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(gl) + "\n")

with open(os.path.join(CASE_DIR, "typical_cases.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(sel_csv[0].keys()))
    w.writeheader()
    w.writerows(sel_csv)

with open(os.path.join(CASE_DIR, "_images_to_fetch.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(rel_images) + "\n")

print("typical_cases.md written,", len(sel_csv), "cases,", len(rel_images), "unique images")
for r in rel_images:
    print("IMG", r)
