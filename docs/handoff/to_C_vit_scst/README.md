# 交接给 李春鹏 —— 任务三：ViT 检查点与 SCST 起点

> 接收人：李春鹏（任务三：SCST + 三层奖励 + 盲评）
> 交付人：单文杰（任务一）
> 日期：2026-11-20

## 一、可复用的东西

| 资源 | 路径 | 说明 |
|---|---|---|
| ViT-B/16 + Transformer 解码器权重 | `checkpoints/task1/vit_transformer/best.pt` | task1 最强基线，val CIDEr-D 0.4867（best epoch 15） |
| 模型定义 | `src/task1_caption/models/vit_transformer.py` | `ViTTransformer` |
| 可微打分接口 | `src/task1_caption/models/sequence_log_prob` | SCST 采样句打分 |
| 贪心解码 | `src/task1_caption/beam_search.greedy_decode` | SCST 自批判基线 |
| 束搜索 | `src/task1_caption/beam_search.beam_search_decode` | 推理/对比 |
| 测试集预测（greedy / beam） | `outputs/task1/predictions/`、`outputs/task1/beam_search/final/` | 可作为 SCST 前基线 |

task1 测试集结果（供你对齐基线）：

| 设置 | BLEU-4 | ROUGE-L | CIDEr-D |
|---|---|---|---|
| greedy | 5.35 | 27.66 | 0.4620 |
| beam=3 lp=0.8 | 6.41 | 27.83 | 0.5270 |

## 二、SCST 怎么接（对照开题报告 5.1）

1. **起点**：以 `vit_transformer/best.pt`（交叉熵检查点）初始化，不要重新训练编码器。
2. **采样**：对同一批图片，用 `model` 采样一句（`torch.multinomial` over softmax），
   再用 `greedy_decode` 得贪心句作为基线。
3. **优势**：`advantage = reward(采样句) − reward(贪心句)`；
   采样句的 `sequence_log_prob` 保留梯度，贪心项 detach。
4. **奖励**：先只上 CIDEr-D（A1 常规 SCST），再逐层加语义/约束/均衡（A2–A4）。
5. **学习率**：取交叉熵阶段的十分之一（≈1e-5），每 500 步在验证集贪心评测。

## 三、接口约定（务必遵守）

- `sequence_log_prob(model, images, token_ids)` 返回 `[N]`，**可微**，不做 `no_grad`。
- `token_ids` 形如 `[N, L]`，含 `<sos>`…`<eos>`，`<pad>` 位置自动屏蔽。
- 指标统一用 `src.common.metrics.compute_all`（冻结口径）；详见 `docs/interface_contract.md`。
- 盲评：从 task1 测试集按类别固定抽 100 张（seed=42），候选描述匿名随机排序。

## 四、验收标准

- [ ] 从 ViT 交叉熵检查点出发，SCST 训练可运行，reward 曲线正常上升。
- [ ] 五组消融（A0–A4）统一数据/词表/解码/预算，报告三项主指标 + 最差组 CIDEr-D。
- [ ] 输出 `outputs/scst/{A0..A4}/`，保留每步验证集贪心指标。
- [ ] 盲评材料（100 张 ID + 原始结构化评分）存入 `outputs/blind_eval/`。

## 五、常见坑

- **采样句退化**：温度过低导致样本≈贪心，advantage≈0；可适当提高温度或加熵正则。
- **奖励尺度**：CIDEr-D 用 ×10 尺度时数值较小，advantage 做 batch 内标准化更稳。
- **不要改 metrics.py**：`tests/test_metrics.py` 会失败；如需新奖励分量，在 `src/task3_scst/rewards/` 内实现。
- ViT 输入必须 224×224（196 patch + class token），改分辨率会触发位置编码尺寸错误。
