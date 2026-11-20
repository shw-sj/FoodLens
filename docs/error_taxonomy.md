# 错误分类标准（Error Taxonomy）

本文件固定 FoodLens 的错误判定口径，供任务一（描述错误）和任务二（JSON 错误）共用。
自动脚本 `src/task1_caption/error_analysis.py` 只做**启发式初筛**，最终以人工在
`reports/error_cases/manual_annotation_sheet.csv` 中填写的 `manual_category` 为准。

---

## 一、任务一：四类描述错误（Captions）

一条预测可以同时命中多类错误；统计时分别计数（`flags`），并给出按优先级去重后的
`primary`（优先级：误识别 > 烹饪方式 > 漏述 > 无依据添加）。

### 1. 食材误识别（`ingredient_misidentification`）
- **定义**：预测中出现了参考描述里**不存在**的食材词，同时参考里的关键食材**没被提到**。
- **判定信号**：`added_food ⊄ ref` 且 `missing_food ≠ ∅`。
- **示例**：参考 "Baked apple strudel …" → 预测 "a piece of **pumpkin** cake with brown **bread**"
  （把苹果卷说成南瓜蛋糕，且漏了 apple/almond）。
- **典型成因**：细粒度食材（aebleflaesk、ayam goreng、banh bo 等罕见菜名）混淆。

### 2. 烹饪方式错误（`cooking_method_error`）
- **定义**：预测的烹饪方式与参考不一致，或参考有烹饪词而预测完全缺失。
- **判定信号**：`cooking_pred ≠ cooking_ref`（字段格式 `pred=X|ref=Y`，X 为空表示漏说）。
- **示例**：参考 "**baked** almond biscuits" → 预测 "**fried** corn tortillas"（烤→炸）。
- **注意**：`pred=`（空）表示"漏说烹饪方式"，`pred=fried|ref=baked` 表示"说错"。

### 3. 食材漏述（`missing_ingredient`）
- **定义**：参考里存在的食材在预测中未被提及。
- **判定信号**：`missing_food ≠ ∅`（人工标注时只计**高显著度**食材，忽略 surface/color 等非食材）。
- **示例**：参考 "**rice** adding bistek …" → 预测只写 bistek/beef，漏了 rice。

### 4. 无依据添加（`unsupported_addition` / 幻觉）
- **定义**：预测出现参考中完全没有的内容（食材或明显无关名词）。
- **判定信号**：`added_food ≠ ∅`（人工确认该词确属"无中生有"）。
- **示例**：参考 "baked beans" → 预测多出 "**meat** sauce"。

> 说明：自动脚本的食物词典来自训练描述（min_count=3，已去停用词），仍会混入
> slices/surface/color 等非食材词，因此自动标志率偏高（接近 100%）；人工复核时
> 只需判断**主错误类别**，不要求逐一枚举所有 token。

---

## 二、任务二：三类 JSON 错误（Structured records）

供任务二（梁佳庆）使用，任务一不涉及。

| 类别 | 定义 | 校验点 |
|---|---|---|
| 格式不合规（`schema_invalid`） | JSON 无法解析、缺必填字段、类型错误 | `validate_json.py`：语法、字段名、类型 |
| 食材漏检 / 误检（`ingredient_error`） | 食材集合与标注不一致 | 集合比对 |
| 重量 / 营养偏差（`value_error`） | 重量、总重、热量、宏量营养素估计偏差 | 分字段 MSE |

---

## 三、人工标注流程

1. 打开 `reports/error_cases/manual_annotation_sheet.csv`（60 例，每类 15 例，`suggested_category` 为预填建议）。
2. 对照 `image_local` 指向的原图（在 `reports/figures/annotation/`），查看 `reference` 与两模型预测。
3. 在 `manual_category` 填入主错误类别（取值见上表英文键或中文名），`notes` 简述原因。
4. 若自动建议正确，直接复制 `suggested_category` 或在 `notes` 写 "agree"。
