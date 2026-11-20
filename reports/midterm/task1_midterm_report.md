# FoodLens 课程设计 —— 任务一 中期/结题报告（草稿）

> 题目：食物图像描述生成（Food-500 Cap 细粒度描述）
> 负责人：单文杰（组长）　数据：Food-500 Cap（HuggingFace `advancedcv/Food500Cap`）
> 评测口径：`task1-eval-v1`　指标实现：`task1-metrics-v1` / `regex-lower-alnum-v1`
> 说明：本文件为报告草稿，数字均取自 `outputs/task1/**/metrics.json` 与 `reports/experiment_tables/`。

## 摘要

本任务在 Food-500 Cap 上实现两种图像描述模型：ResNet-18＋GRU（Show-and-Tell 范式的强基线）
与 ViT-B/16＋Transformer 解码器。两模型共享同一数据划分、词表与交叉熵目标。在测试集 4938 张
图片上，ViT 模型在 BLEU-4 / ROUGE-L / CIDEr-D 三项指标均优于 ResNet-18＋GRU
（CIDEr-D 0.4620 vs 0.3523）。进一步对 ViT 做束搜索增项：在验证集上扫描束宽与长度惩罚，
选定 beam=3、length_penalty=0.8，测试集 CIDEr-D 由 0.462 提升到 **0.527**（+14.0%），
BLEU-4 由 5.35 提升到 **6.41**。最后按食材误识别、烹饪方式错误、食材漏述、无依据添加
四类对样例做错误统计与典型样例分析，为后续 SCST 与结构化记录任务提供可复现基线。

## 1 任务背景与目标

食物图像描述需指出图中可辨识的**食材、形态与烹饪方式**，属细粒度图像描述任务。与通用
图像描述相比，食物类别长尾、同类菜名差异细微（如 ayam goreng / ayam bakar），对模型
的视觉细粒度判别与语言建模都提出更高要求。

本任务目标：
1. 复现两类编码器—解码器结构，得到可复现的强基线；
2. 在统一评测口径下比较两模型，并给出细粒度错误统计；
3. 对最优模型增加束搜索对照，考察解码策略对流畅度与描述差异的影响；
4. 为任务二（结构化记录，复用 ResNet18+GRU）与任务三（ViT + SCST）交付冻结检查点与接口。

## 2 数据与预处理

- **来源**：Food-500 Cap（发布方 ACM MM'2023）。本次实验使用其 HuggingFace 版本
  `advancedcv/Food500Cap`（24675 图文对），与官方 `finetune/evaluation` 发布版存在约 145 条差异，
  已在 `data/README.md` 注明；方法在数据上保持内部一致。
- **划分**：沿用发布方测试集（4938 张），从训练部分按类别分层划出约 10% 作为验证集
  （seed=42）。最终 train 17763 / val 1974 / test 4938。
- **清洗**：核查缺失、损坏、空描述与精确重复，本次数据 0 条丢弃；保存三份 ID 清单
  （`data/manifests/food500cap_{train,val,test}_ids.txt`）与清洗报告。
- **词表**：用训练描述构建共享词表（max_size=5000、min_freq=2），实际 3101 token，
  训练集 OOV 率 0.52%。分词器固定为 `regex-lower-alnum-v1`。

| 划分 | 图片数 | 描述数 | 描述/图 | 平均 token 数 |
|---|---:|---:|---:|---:|
| train | 17763 | 17774 | 1.001 | 18.667 |
| val | 1974 | 1977 | 1.002 | 18.598 |
| test | 4938 | 4944 | 1.001 | 18.416 |

## 3 方法

### 3.1 ResNet-18＋GRU
ImageNet 预训练 ResNet-18 提取整图特征，经全局平均池化与线性投影 + tanh 得到视觉向量 v；
v 用于初始化 GRU 隐状态，并在每一步与词嵌入拼接输入 GRU，使解码器持续看到图像条件；
最后线性层映射到词表 logits。训练先用 teacher forcing（以参考前缀预测下一词，屏蔽 pad）。

### 3.2 ViT-B/16＋Transformer 解码器
ViT-B/16 保留 196 个 patch token 与 class token（输入 224×224）；解码器为 4 层 pre-LN
Transformer，前缀自注意力带因果掩码，并通过**交叉注意力**读取视觉 token 序列；实现带
KV cache，贪心/束搜索可增量解码。

两模型共享划分、词表与交叉熵目标；均先冻结预训练编码器 2 个 epoch，再解冻末段
（`last_block`），是否解冻由验证集表现确定（记录于 `configs/`）。

## 4 实验设置

| 项目 | ResNet18+GRU | ViT+Transformer |
|---|---|---|
| 参数量 | 15.79M | 106.21M |
| Batch / 优化器 | 32 / AdamW 1e-4 | 32 / AdamW 1e-4 |
| weight decay / warmup | 0.01 / 2 ep | 0.01 / 2 ep |
| 最大轮数 / 早停 | 30 / patience=6 | 30 / patience=6 |
| 冻结策略 | 前 2 轮冻结→last_block | 同左 |
| 混合精度 | 是 | 是 |

**评测口径**（冻结）：分词 `regex-lower-alnum-v1`；BLEU-4 多参考+最近参考长度+COCO 平滑；
ROUGE-L beta=1.2；CIDEr-D 1–4 gram tf-idf、sigma=6、不截断计数、官方 ×10 尺度。
实现经 `tests/test_metrics.py` 与 `pycocoevalcap` 逐位对齐。checkpoint 按验证集 CIDEr-D 选择。

## 5 结果与分析

### 5.1 验证集最佳结果
| 模型 | 实际轮数 | 最佳 epoch | val BLEU-4 | val ROUGE-L | val CIDEr-D |
|---|---:|---:|---:|---:|---:|
| ResNet18+GRU | 30 | 30 | 5.02 | 26.67 | 0.3508 |
| ViT+Transformer | 21（早停） | 15 | 5.69 | 27.46 | **0.4867** |

### 5.2 测试集统一评测（greedy，n=4938）
| 模型 | BLEU-4 | ROUGE-L | CIDEr-D | 重复率 rep4 | 平均句长 |
|---|---:|---:|---:|---:|---:|
| ResNet18+GRU | 4.87 | 26.92 | 0.3523 | 0.018 | 17.3 |
| ViT+Transformer | **5.35** | **27.66** | **0.4620** | 0.005 | 19.9 |

**分析**：ViT 模型三项指标全面领先，CIDEr-D 相对提升约 31%，重复率更低（0.005 vs 0.018），
句子略长但与参考更贴合。这与 ViT 的细粒度 patch 特征和交叉注意力对食材局部区域的建模能力一致。
ResNet18+GRU 训练到 30 轮仍在缓慢提升，说明其容量/条件方式（整图向量）在细粒度长尾上受限。

## 6 束搜索增项（ViT）

对同一 ViT 检查点，在验证集（前 500 张）扫描 beam∈{3,5,7} × length_penalty∈{0.6,0.8,1.0,1.2,1.4}，
以验证集 CIDEr-D（BLEU-4 作 tie-break）选择设置，再在测试集跑最终结果。

| 设置 | 验证集 BLEU-4 | 验证集 CIDEr-D | 测试集 BLEU-4 | 测试集 CIDEr-D | 平均句长 | ms/图 |
|---|---:|---:|---:|---:|---:|---:|
| greedy | 5.36 | 0.461 | 5.35 | 0.462 | 19.9 | 2.7 |
| **beam=3 lp=0.8** | **6.81** | **0.554** | **6.41** | **0.527** | 16.0 | 76.5 |

**分析**：束搜索相对贪心，测试集 CIDEr-D +14.0%、BLEU-4 +19.8%，重复率下降、平均句长缩短
（19.9→16.0）。更大的束宽（5/7）在验证集未带来更好 CIDEr-D，且推理耗时约 2–3 倍，故
选定 beam=3。代价是单图耗时由 2.7ms 升到 ~76ms（逐图搜索），但仍在可接受范围。

## 7 错误分析

### 7.1 自动初筛统计

按四类描述错误对测试集自动初筛（`error_analysis.py`，结果见 `outputs/task1/error_cases/`），
并人工挑选典型样例（`reports/error_cases/typical_cases.md`，含 12 张原图配图）。

| 模型 | 食材误识别 | 烹饪方式错误 | 食材漏述 | 无依据添加 |
|---|---:|---:|---:|---:|
| ResNet18+GRU | 97.2% | 30.7% | 99.9% | 97.3% |
| ViT+Transformer | 99.4% | 33.6% | 99.8% | 99.6% |

> 说明：自动标志为启发式初筛（食物词典含少量非食材词，标志偏宽），四类可重叠；
> 最终以人工在 `reports/error_cases/manual_annotation_sheet.csv` 中填写为准。

### 7.2 人工复核结果（60 例，每类 15）

为校准自动标志，对 60 例做了人工判定（工具 `reports/error_cases/manual_annotation_review.html`，
每个案例并排展示原图、参考与两模型预测）。结果如下：

**人工判定的主错误分布：**

| 错误类别 | 例数 | 占比 |
|---|---:|---:|
| 食材误识别 | 43 | 71.7% |
| 食材漏述 | 8 | 13.3% |
| 无依据添加 | 6 | 10.0% |
| 烹饪方式错误 | 3 | 5.0% |
| 合计 | 60 | 100% |

**自动建议 vs 人工判定（一致率 24/60 = 40%），混淆矩阵（行=自动建议，列=人工判定）：**

| 自动＼人工 | 食材误识别 | 烹饪方式错误 | 食材漏述 | 无依据添加 |
|---|---:|---:|---:|---:|
| 食材误识别 | 11 | 1 | 1 | 2 |
| 烹饪方式错误 | 13 | 2 | 0 | 0 |
| 食材漏述 | 8 | 0 | 7 | 0 |
| 无依据添加 | 11 | 0 | 0 | 4 |

**关键发现**：
1. 人工判定的**主错误压倒性地是"食材误识别"（72%）**：即使某图同时存在烹饪方式或添加错误，
   人工认为其主导错误仍是认错食材——这与 Food-500 Cap 细粒度长尾（罕见菜名）高度一致。
2. 自动建议（按挑选类别）与人工一致率仅 **40%**，且被挑为"烹饪方式/无依据添加"的案例中
   多数（13/15、11/15）被人工判为误识别，说明**启发式初筛只能做"召回优先"的粗筛，不能直接当精确类别**。
3. 人工判定的主错误在两个模型的自动标志中复现率均为 100%（自动标志偏宽），
   印证了"自动标志召回高、精确度低，必须人工确认"的口径。

**定性结论**（来自典型样例与人工复核）：
1. **食材误识别**：罕见菜名（aebleflaesk / ayam goreng / banh bo）易被替换为常见菜（pumpkin cake、
   svickova），两模型都会犯，ViT 相对更贴近原菜名——这是最主要瓶颈。
2. **烹饪方式错误**：典型如"烤杏仁饼干"被判成"炸玉米饼"（baked→fried）；多为**说错**或**漏说**烹饪词，但常被误识别掩盖。
3. **食材漏述**：配菜/点缀（rice、almond、mashed potatoes）易被忽略。
4. **无依据添加**：模型倾向补充"pork belly / sauce"等高先验成分，出现无中生有（幻觉）。

## 8 结论与局限

**结论**
1. 两模型在统一口径下完成训练与评测，ViT 全面优于 ResNet18+GRU（CIDEr-D 0.462 vs 0.352）。
2. 束搜索为 ViT 带来一致提升（CIDEr-D 0.462→0.527），验证了"A1 常规 SCST"对照所需的解码基线。
3. 四类错误分析与典型样例已归档，暴露细粒度食材与烹饪方式仍是主要瓶颈。

**局限**
1. 使用 HF 版数据（24675 条），与官方发布版存在少量差异。
2. 自动错误标注偏宽；已完成 60 例人工复核（主错误 72% 为食材误识别），但全量 4938 例未逐一人工标注。
3. 未做大模型评测/人工盲评（属任务三范围）。

## 9 复现命令

```bash
# 训练（两个模型）
python -m src.task1_caption.train --config configs/task1_resnet18_gru.yaml
python -m src.task1_caption.train --config configs/task1_vit_transformer.yaml

# 测试集贪心推理 + 统一评测
python -m src.task1_caption.infer --checkpoint checkpoints/task1/vit_transformer/best.pt \
    --split test --method greedy
python -m src.task1_caption.evaluate \
    --pred resnet18_gru=outputs/task1/predictions/resnet18_gru_test_greedy.csv \
    --pred vit_transformer=outputs/task1/predictions/vit_transformer_test_greedy.csv --split test

# 束搜索（验证集调参 + 测试集最终结果）
python -m src.task1_caption.beam_search --config configs/task1_beam_search.yaml --final-split test

# 错误分析 + 报告材料
python -m src.task1_caption.error_analysis --manifest data/processed/food500cap_clean.jsonl \
    --image-root data/raw/food500cap \
    --pred resnet18_gru=outputs/task1/predictions/resnet18_gru_test_greedy.csv \
    --pred vit_transformer=outputs/task1/predictions/vit_transformer_test_greedy.csv \
    --out-dir outputs/task1/error_cases
python scripts/make_report_materials.py
python scripts/make_annotation_sheet.py
```

## 附录：产物清单

- 结果表格与 CSV：`reports/experiment_tables/`（6 张表 + 4 个 CSV）
- 典型错误案例：`reports/error_cases/typical_cases.md` + `reports/figures/error_cases/`（12 图）
- 人工标注：`reports/error_cases/manual_annotation_sheet.csv`（已填 60 例）+ `manual_annotation_review.html`（审阅页）+ `manual_annotation_summary.md/.csv`（统计）+ `reports/figures/annotation/`（60 图）
- 冻结权重与清单：`checkpoints/task1/*/{best,last}.pt`、`checkpoints/manifest.csv`
- 接口与错误口径：`docs/interface_contract.md`、`docs/error_taxonomy.md`
- 生成脚本：`scripts/make_report_materials.py`、`make_annotation_sheet.py`、`make_annotation_review.py`、`analyze_manual_annotation.py`


