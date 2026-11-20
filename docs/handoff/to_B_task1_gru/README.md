# 交接给 梁佳庆 —— 任务二：ResNet18+GRU 基线复用

> 接收人：梁佳庆（任务二：食物图像 → 结构化膳食记录）
> 交付人：单文杰（任务一）
> 日期：2026-11-20

## 一、可复用的东西

| 资源 | 路径 | 说明 |
|---|---|---|
| ResNet-18 编码器 + GRU 解码器权重 | `checkpoints/task1/resnet18_gru/best.pt` | task1 基线，val CIDEr-D 0.3508 |
| 模型定义 | `src/task1_caption/models/resnet18_gru.py` | `ResNet18GRU` |
| 加载工具 | `src/task1_caption/infer.load_bundle()` | 重建模型 + 词表 |
| 共享词表 | `data/vocab/shared_vocab_5000.json` | 3101 token |
| 数据清单 | `data/manifests/food500cap_*.txt` | task1 划分 |

## 二、任务二的改法（对照开题报告 4.2.2）

1. **扩充词表**：在共享词表基础上，追加 JSON 字段名、`{ } [ ] : , "`、数字与全部食材名，
   得到 `data/vocab/task2_extended_vocab.json`。
2. **复用权重**：`load_bundle` 加载 task1 checkpoint 后，调用
   ```python
   model.resize_token_embeddings(len(new_vocab))
   ```
   旧行自动保留、新行随机初始化；**随后必须重建 optimizer**（参数张量被替换）。
3. **改训练目标**：输入仍是图片，监督目标换成 `target_text`（整条 JSON 字符串），
   用交叉熵训练；先小样本验证"图片→完整记录"能生成。
4. **评测**：调用 `src/task2_diet/validate_json.py` 做语法/字段/类型/非负校验，
   对比 `target`；错误分三类（见 `docs/error_taxonomy.md` 第二节）。

## 三、接口约定（务必遵守）

- 预测文件格式、checkpoint 结构、词表扩展、指标口径见 **`docs/interface_contract.md`**（冻结 v1）。
- 不要修改 `src/common/metrics.py` 的公式；如需新指标，在 task2 内另写。
- 不要用 task1 的 test 划分调参（沿用同一 test_ids 保证可比）。

## 四、验收标准

- [ ] 能在 task1 checkpoint 基础上完成扩词表并训练，loss 正常下降。
- [ ] 生成的 JSON 通过 `validate_json.py`，并在验证集选定 checkpoint。
- [ ] 输出 `outputs/task2/predictions/`、`outputs/task2/metrics/`，字段符合接口约定。
- [ ] 错误分三类归档，保留原始输出与失败原因。

## 五、常见坑

- **ViT 检查点不能直接给 GRU 用**：两者 encoder 结构不同，任务二基线请用 `resnet18_gru/best.pt`。
- **扩词表后忘记重建 optimizer** → 参数更新错乱。
- **task1 用的 HF 数据（24675 条）**，任务二数据是课程 `annotations.zip`/`images.zip`（nutrition5k-student-v1），两者独立。
