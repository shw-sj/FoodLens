# 交接接口约定（Interface Contract，冻结 v1）

本文件冻结任务一对外交付的接口，任务二（梁佳庆）与任务三（李春鹏）在此之上开发，
**不要修改**已冻结的字段名与文件格式；若必须变更，需 bump 版本并在本文件记录。

标注 `task1-eval-v1` 的评测口径与 `task1-metrics-v1` 的指标实现一经冻结，所有对比实验
必须复用，不得各自复制一份。

---

## 1. 预测文件格式（CSV）

所有推理输出统一为三列（UTF-8，带 BOM 亦可）：

```
image_id,prediction,reference
```

- `image_id`：形如 `images/test/unlabeled/000170.jpg` 的相对路径，唯一。
- `prediction`：模型生成的描述字符串。
- `reference`：多参考描述用 ` ||| ` 连接（默认分隔符见 `configs/eval_protocol.yaml`）。

由 `src/task1_caption/infer.py` 写出，`evaluate.py` / `error_analysis.py` 读取。
`*_meta.json` 姊妹文件记录 checkpoint、方法、耗时等元信息。

## 2. Checkpoint 结构（.pt）

`torch.save` 的 dict，至少包含：

```
{
  "state_dict":  <model.state_dict()>,
  "model_config": {"name": "resnet18_gru" | "vit_transformer", ...},
  "vocab":        <vocab dict, 可用 Vocab.from_dict 还原>,
  "data_config":  {"max_len": 40, "vocab": "...", ...},
  "config":       <完整 YAML 配置>
}
```

`src.task1_caption.infer.load_bundle()` 负责加载并重建模型 + 词表，下游请复用它。

当前冻结权重（见 `checkpoints/manifest.csv`）：

| 模型 | 路径 | sha256 | val CIDEr-D |
|---|---|---|---|
| ResNet18+GRU | `checkpoints/task1/resnet18_gru/best.pt` | `a920d213…d2fed7a` | 0.3508 |
| ViT+Transformer | `checkpoints/task1/vit_transformer/best.pt` | `fe3ba76a…481318` | 0.4867 |

## 3. 词表扩展接口（任务二用）

`ResNet18GRU.resize_token_embeddings(new_vocab_size)` 与
`ViTTransformer.resize_token_embeddings(new_vocab_size)`：

- 保留旧行，新增行按 N(0, 0.02) 初始化，`pad` 行清零。
- **调用后必须重建优化器**（参数张量被替换）。
- 词表文件：`data/vocab/shared_vocab_5000.json`（任务一，3101 token）；
  任务二在其基础上追加字段名、JSON 标点、数字与食材名 → `task2_extended_vocab.json`。

## 4. SCST 交接接口（任务三用）

`src/task1_caption/models/__init__.py` 与 `beam_search.py` 暴露：

```python
from src.task1_caption.models import build_model, sequence_log_prob
from src.task1_caption.beam_search import greedy_decode, beam_search_decode

# 可微的整句对数概率 [N]，用于 SCST 对采样句打分
logp = sequence_log_prob(model, images, token_ids)   # token_ids: [N, L]，含 <sos>…<eos>

# 贪心解码，作为 SCST 自批判基线
tokens = greedy_decode(model, images, max_len=40)
```

两种模型实现同一解码状态接口（`init_decode_state / decode_step /
select_decode_state / repeat_decode_state`），`greedy_decode` / `beam_search_decode` 无需分支。

## 5. 指标实现

- 统一入口：`src.common.metrics.compute_all(preds, refs, **kwargs)` → dict
  （`bleu4`, `rouge_l`, `cider_d`, `rep4`, `avg_len`, `n`）。
- 版本：`METRIC_VERSION="task1-metrics-v1"`，`TOKENIZER_VERSION="regex-lower-alnum-v1"`。
- 尺度：BLEU-4 / ROUGE-L = 0–100；CIDEr-D = ×10（`configs/eval_protocol.yaml` 控制）。
- `tests/test_metrics.py` 锁定与 `pycocoevalcap` 的一致性，改动公式会使测试失败（有意为之）。

## 6. 数据清单

- `data/manifests/food500cap_{train,val,test}_ids.txt`：每行一个 image_id。
- `data/processed/food500cap_clean.jsonl`：每行 `{image_id, captions, class, split}`（不入库）。
- 评测以 manifest 中的 test_ids 为准，禁止用 test 调参。

## 7. 随机性

- 全局种子 42（`src/common/seed.set_seed`）。
- README/报告中引用的数字须来自 `outputs/task1/**/metrics.json`，不手抄。
