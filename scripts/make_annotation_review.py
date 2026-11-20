# -*- coding: utf-8 -*-
"""Render a self-contained HTML review page for manual annotation.

Reads reports/error_cases/manual_annotation_sheet.csv and writes
reports/error_cases/manual_annotation_review.html with each case's image
(relative path into reports/figures/annotation/) next to its reference and
both model predictions.  Open the HTML in a browser to review; record results
back into the CSV's manual_category / notes columns.
"""
import csv
import html
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE_DIR = os.path.join(ROOT, "reports", "error_cases")
rows = list(csv.DictReader(open(os.path.join(CASE_DIR, "manual_annotation_sheet.csv"),
                               encoding="utf-8-sig")))

p = []
p.append("<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>")
p.append("<title>任务一 描述错误 人工标注审阅表</title>")
p.append("<style>")
p.append("body{font-family:'Microsoft YaHei',Arial,sans-serif;margin:24px;background:#f4f5f7;color:#222;}")
p.append("h1{font-size:20px;} .intro{background:#fff;border-left:4px solid #4a7;padding:12px 16px;border-radius:6px;margin-bottom:18px;line-height:1.7;}")
p.append(".case{display:flex;gap:16px;background:#fff;border:1px solid #e0e0e0;border-radius:10px;padding:12px;margin-bottom:14px;box-shadow:0 1px 2px rgba(0,0,0,.04);}")
p.append(".img img{width:300px;height:auto;border-radius:8px;display:block;}")
p.append(".txt{flex:1;font-size:14px;line-height:1.65;}")
p.append(".badge{display:inline-block;background:#e8f0ff;color:#245;padding:2px 9px;border-radius:11px;font-size:12px;margin-left:6px;}")
p.append(".ref{color:#333;} .rn{color:#b03;} .vt{color:#227;} .sug{color:#078;font-weight:bold;margin-top:6px;}")
p.append("code{background:#f0f2f4;padding:1px 5px;border-radius:3px;font-size:12px;}")
p.append("</style></head><body>")
p.append("<h1>任务一 描述错误 人工标注审阅表（60 例，每类 15 例）</h1>")
p.append("<div class='intro'>")
p.append("<b>怎么用：</b>对照左侧原图与参考描述，判断该图<b>最主要的描述错误类别</b>：")
p.append("食材误识别 / 烹饪方式错误 / 食材漏述 / 无依据添加。<br>")
p.append("把判断结果填入 <code>manual_annotation_sheet.csv</code> 的 <code>manual_category</code> 列，")
p.append("理由填 <code>notes</code>（若同意自动建议，写 <code>agree</code> 即可）。")
p.append("</div>")
for i, r in enumerate(rows, 1):
    img_rel = "../figures/annotation/" + r["image_id"].replace("/", "__")
    p.append("<div class='case'>")
    p.append(f"<div class='img'><img loading='lazy' src='{html.escape(img_rel)}' "
             f"alt='{html.escape(r['image_id'])}'></div>")
    p.append("<div class='txt'>")
    p.append(f"<div><b>#{i:02d}</b><span class='badge'>{html.escape(r['suggested_category'])}</span>"
             f" <code>{html.escape(r['image_id'])}</code></div>")
    p.append(f"<p class='ref'><b>参考：</b>{html.escape(r['reference'])}</p>")
    p.append(f"<p class='rn'><b>ResNet18+GRU：</b>{html.escape(r['pred_resnet18_gru'])}</p>")
    p.append(f"<p class='vt'><b>ViT+Transformer：</b>{html.escape(r['pred_vit_transformer'])}</p>")
    p.append(f"<p class='sug'>建议类别：{html.escape(r['suggested_category'])}</p>")
    p.append("</div></div>")
p.append("</body></html>")

out = os.path.join(CASE_DIR, "manual_annotation_review.html")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(p))
print("written", out, "|", len(rows), "cases")
