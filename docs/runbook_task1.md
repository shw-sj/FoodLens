# 任务一运行与交接说明

本说明对应当前仓库里的任务一代码。任务二、任务三的目录没有创建，避免和其他组员的工作冲突。

任务一覆盖开题报告中的以下内容：

| 报告条目 | 本仓库实现 |
| --- | --- |
| 4.1.1 数据准备与流程 | `scripts/prepare_data_task1.py`、`data/manifests/`、`data/vocab/shared_vocab_5000.json` |
| 4.1.2 ResNet-18＋GRU | `src/task1_caption/models/resnet18_gru.py` |
| 4.1.2 ViT＋Transformer 解码器 | `src/task1_caption/models/vit_transformer.py` |
| 4.1.3 训练、对照、组内交接 | `src/task1_caption/train.py`、`infer.py`、`evaluate.py` |
| 统一评测（BLEU-4 / ROUGE-L / CIDEr-D） | `src/common/metrics.py`、`configs/eval_protocol.yaml` |
| 束搜索增项 | `src/task1_caption/beam_search.py`、`configs/task1_beam_search.yaml` |
| 四类错误统计 | `src/task1_caption/error_analysis.py` |
| 给梁佳庆的基线交接 | ResNet-18＋GRU 权重、配置、词表、推理命令、扩词表接口 |
| 给李春鹏的 SCST 交接 | ViT 交叉熵检查点、`sequence_log_prob`、`greedy_decode`、数据清单、指标脚本 |

---

## 0. 最短路径

假设已经解压数据、装好依赖，在仓库根目录执行：

```bash
# 0. 如果还没有本地原始数据（可选）：从 HuggingFace 拉取并落盘
python scripts/data_load.py --out-dir data/raw/food500cap

# 1. 数据 -> manifest + ID 清单 + 共享词表
python scripts/prepare_data_task1.py --raw-dir data/raw/food500cap

# 2. 两个模型（单卡依次跑；两张卡可以并行）
bash scripts/run_task1_train.sh resnet18_gru
bash scripts/run_task1_train.sh vit_transformer

# 3. 贪心 + 束搜索推理和统一评测（以 ViT 为例）
bash scripts/run_task1_eval.sh checkpoints/task1/vit_transformer/best.pt test 5 1.0

# 4. 两模型贪心对照
python -m src.task1_caption.evaluate \
  --pred "resnet18_gru=outputs/task1/predictions/resnet18_gru_test_greedy.csv" \
  --pred "vit_transformer=outputs/task1/predictions/vit_transformer_test_greedy.csv" \
  --split test \
  --out outputs/task1/metrics/task1_models_test.json \
  --markdown outputs/task1/metrics/task1_models_test.md

# 5. 错误分析
python -m src.task1_caption.error_analysis \
  --manifest data/processed/food500cap_clean.jsonl \
  --image-root data/raw/food500cap \
  --pred "resnet18_gru=outputs/task1/predictions/resnet18_gru_test_greedy.csv" \
  --pred "vit_transformer=outputs/task1/predictions/vit_transformer_test_greedy.csv" \
  --out-dir outputs/task1/error_cases --copy-images
```

---

## 1. 服务器环境

建议单独建 conda 环境。PyTorch 的 CUDA 版本要和服务器驱动匹配，下面以 CUDA 12.1 为例：

```bash
conda create -n foodlens python=3.10 -y
conda activate foodlens

# 二选一：按服务器 CUDA 版本选择 index-url
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r environment/requirements-task1.txt

python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

如果服务器已经装好带 CUDA 的 torch，`requirements-task1.txt` 里的 `torch>=2.1` 不会强行覆盖，直接装其余包即可。

### 预训练权重

两个模型默认 `pretrained: true`，会从 torchvision 下载：

- ResNet-18：约 45 MB
- ViT-B/16：约 330 MB

服务器不能联网时，先在一台能联网的机器上下载，再把缓存拷到服务器：

```bash
# 联网机器
python -c "from torchvision.models import resnet18, ResNet18_Weights; resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)"
python -c "from torchvision.models import vit_b_16, ViT_B_16_Weights; vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)"
# 缓存默认在 ~/.cache/torch/hub/checkpoints
```

把 `~/.cache/torch/hub/checkpoints/*.pth` 拷到服务器的同一目录，或设置 `TORCH_HOME` 指向拷贝位置。也可以显式指定本地权重：

```bash
--set model.pretrained=false --set model.pretrained_path=/abs/path/resnet18-....pth
```

---

## 2. 数据准备

### 2.1 放原始数据

```bash
mkdir -p data/raw/food500cap
unzip /path/to/images.zip      -d data/raw/food500cap/images
unzip /path/to/annotations.zip -d data/raw/food500cap
```

脚本会自动在 `data/raw/food500cap` 下找标注文件（`annotations.json`、`train.json`/`test.json`、`annotations/*.json`、`*.csv` 等）和图片目录。

### 2.2 没有本地数据：从 HuggingFace 加载（可选）

如果课程数据还没下载，可以用 `scripts/data_load.py` 拉取 `advancedcv/Food500Cap`，落盘成 prepare 能读的目录：

```bash
pip install datasets          # requirements-task1.txt 已包含
python scripts/data_load.py --out-dir data/raw/food500cap --inspect
python scripts/data_load.py --out-dir data/raw/food500cap
```

`--inspect` 只打印 split、列名和第一条记录，不保存图片；先确认列名，再正式落盘。脚本会自动识别
`image` / `caption` / `category` / `label` / `split` 等常见列；如果识别错，显式指定：

```bash
python scripts/data_load.py \
  --out-dir data/raw/food500cap \
  --image-column image --caption-column text --class-column category
```

国内服务器建议用镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com python scripts/data_load.py --out-dir data/raw/food500cap
```

需要登录或接受条款的数据集：

```bash
huggingface-cli login
# 或
python scripts/data_load.py ... --token hf_xxx
```

只拉 200 条做链路验证：

```bash
python scripts/data_load.py --out-dir data/raw/food500cap_demo --limit 200
python scripts/prepare_data_task1.py --raw-dir data/raw/food500cap_demo
```

生成：

```text
data/raw/food500cap/annotations.json
data/raw/food500cap/images/<split>/<class>/<idx>.jpg
data/raw/food500cap/data_load_report.json
```

然后照常跑 2.3 的 prepare；也可以加 `--prepare` 让 data_load 直接调用 prepare。

两个注意点：

- HF 上的 `advancedcv/Food500Cap` 不一定带发布方的官方 train/test 划分。如果它只有一个 `train` split，`prepare_data_task1.py` 会按比例随机留出 val/test，数量和开题的 19,760 / 4,940 官方划分不一致。正式实验优先用课程提供的官方划分；如果用 HF 数据，请在报告里注明实际的划分方式和数量。
- 脚本会把图片统一转成 JPEG（默认质量 95），和原始文件不是逐字节相同。如果课程对原图文件有要求，直接用课程包，不要用这个脚本转码。

### 2.3 运行

```bash
python scripts/prepare_data_task1.py --raw-dir data/raw/food500cap
```

默认行为（与开题一致）：

- 保留发布方测试集；
- 没有官方验证集时，从训练部分**按类别分层**划 10% 作为验证集，随机种子 42；
- 用清洗后的训练集描述构建两模型共享的约 5,000 词词表；
- `image_id` 使用相对 `data/raw/food500cap` 的路径，例如 `train/000123.jpg`。

输出：

```text
data/processed/food500cap_clean.jsonl
data/processed/food500cap_prepare_report.json
data/manifests/food500cap_train_ids.txt
data/manifests/food500cap_val_ids.txt
data/manifests/food500cap_test_ids.txt
data/vocab/shared_vocab_5000.json
```

跑完先看控制台和报告：

```bash
python -m json.tool data/processed/food500cap_prepare_report.json | less
```

重点核对：

- `counts.train/val/test` 的图片数和描述数；
- `dropped.missing_images_count`：标注里有、图片找不到的条数；
- `dropped.corrupt_images_count`：Pillow 打不开的图片；
- `dropped.exact_duplicates_count`：跨划分/重复文件，脚本会保留 test > val > train 的副本；
- `vocab.size` 和 `vocab.oov_rate_train`。

> 开题里的 19,760 / 4,940 是**图文对**数量。脚本同时统计图片数和描述数；如果一张图有多条描述，图片数会小于图文对数。

### 2.4 标注格式不匹配时

脚本支持以下常见格式，详细解析逻辑在 `scripts/prepare_data_task1.py` 顶部：

```json
// A. 图片 -> 描述列表
{"train/0001.jpg": ["a plate of ...", "..."]}

// B. 图片 -> 记录
{"train/0001.jpg": {"captions": ["..."], "class": "pizza"}}

// C. 记录列表
[{"image": "train/0001.jpg", "captions": ["..."], "class": "pizza", "split": "train"}]

// D. COCO 风格
{"images": [{"id": 1, "file_name": "train/0001.jpg"}],
 "annotations": [{"image_id": 1, "caption": "..."}]}

// E. 顶层 train/val/test
{"train": [...], "val": [...], "test": [...]}
```

CSV/TSV 也支持，一张图多行描述会按图片合并。

如果是训练/测试分开的两个文件：

```bash
python scripts/prepare_data_task1.py \
  --raw-dir data/raw/food500cap \
  --annotations /path/to/train_annotations.json \
  --test-annotations /path/to/test_annotations.json
```

如果标注文件不区分 split，但发布方额外给了 `train.txt` / `test.txt` 图片清单：

```bash
python scripts/prepare_data_task1.py \
  --raw-dir data/raw/food500cap \
  --annotations /path/to/all_captions.json \
  --train-list /path/to/train.txt \
  --test-list /path/to/test.txt
```

脚本先按完整相对路径匹配；只有路径匹配不到时才用文件名，且文件名必须只出现在一个 split 里，避免 `train/0001.jpg` 和 `test/0001.jpg` 互相污染。

如果图片不在 `data/raw/food500cap` 下：

```bash
python scripts/prepare_data_task1.py \
  --raw-dir /path/to/annotations_root \
  --images-dir /path/to/images_root
```

这时要把训练配置里的 `data.image_root` 也改成同一个 `images_root`，否则推理/训练会报 `image not found`。可以在训练命令里用 `--set data.image_root=...` 传入，训练保存的 checkpoint 会记录这个路径，后续 `infer.py` 默认直接使用。

如果实在解析不出来，先看一下标注结构：

```bash
python - <<'PY'
import json
p = 'data/raw/food500cap/annotations.json'
d = json.load(open(p, encoding='utf-8'))
print(type(d))
print(list(d)[:10] if isinstance(d, dict) else d[:2])
PY
```

把前几条的结构发出来即可再适配解析器；现有解析器覆盖了上面 A–E 和 CSV。

---

## 3. 冒烟测试（先确认整条链路）

冒烟测试**不要写进正式 checkpoint 目录**，用 `--set train.save_dir=...` 指到 `smoke_*`。模型是随机初始化的，指标接近 0 是正常的，目的只是验证数据、模型、日志、推理、评测都能跑通。

### 3.1 ResNet-18＋GRU

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -m src.task1_caption.train \
  --config configs/task1_resnet18_gru.yaml \
  --limit-train 32 --limit-val 16 --epochs 1 \
  --set model.pretrained=false \
  --set data.num_workers=0 \
  --set train.save_dir=checkpoints/task1/smoke_resnet \
  --set train.log_dir=logs/task1/smoke
```

### 3.2 ViT＋Transformer

为了在 CPU 或小显存上快速跑通，把 decoder 调小：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -m src.task1_caption.train \
  --config configs/task1_vit_transformer.yaml \
  --limit-train 8 --limit-val 8 --epochs 1 \
  --set model.pretrained=false \
  --set model.d_model=64 --set model.nhead=4 \
  --set model.num_decoder_layers=1 \
  --set train.batch_size=2 \
  --set data.num_workers=0 \
  --set train.save_dir=checkpoints/task1/smoke_vit \
  --set train.log_dir=logs/task1/smoke
```

纯 CPU 服务器再加 `--set train.amp=false --device cpu`。

### 3.3 冒烟推理和评测

```bash
python -m src.task1_caption.infer \
  --checkpoint checkpoints/task1/smoke_resnet/best.pt \
  --split test --method greedy --limit 20 \
  --device cuda --num-workers 0 \
  --output outputs/task1/predictions/smoke_resnet_test_greedy.csv

python -m src.task1_caption.evaluate \
  --pred "smoke=outputs/task1/predictions/smoke_resnet_test_greedy.csv" \
  --split test \
  --out outputs/task1/metrics/smoke_eval.json
```

确认无误后删掉冒烟目录（**只删 smoke 目录**）：

```bash
rm -rf checkpoints/task1/smoke_resnet checkpoints/task1/smoke_vit logs/task1/smoke
```

---

## 4. 正式训练

### 4.1 单卡依次跑

```bash
bash scripts/run_task1_train.sh resnet18_gru
bash scripts/run_task1_train.sh vit_transformer
```

### 4.2 两张卡并行

```bash
mkdir -p logs/task1
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_task1_train.sh resnet18_gru \
  > logs/task1/resnet18_gru.nohup 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_task1_train.sh vit_transformer \
  > logs/task1/vit_transformer.nohup 2>&1 &
```

查看：

```bash
tail -f logs/task1/task1_resnet18_gru_*.log
tail -f logs/task1/task1_vit_transformer_*.log
watch -n 60 nvidia-smi
```

### 4.3 配置要点

两个配置在 `configs/`：

- `task1_resnet18_gru.yaml`
- `task1_vit_transformer.yaml`

与开题一致的初始设置：

- AdamW，lr 1e-4，weight decay 1e-2；
- 前 2 轮 warmup，之后 cosine；
- 有效批大小目标 32；
- 最多 30 轮；
- 前 2 轮冻结视觉编码器，之后解冻末段（`layer4` / ViT 最后一个 encoder block）；
- 用验证集 **CIDEr-D** 选 checkpoint，patience 6。

显存不够时，保持有效批大小 32：

```bash
bash scripts/run_task1_train.sh vit_transformer \
  --set train.batch_size=16 --set train.grad_accum_steps=2
```

多卡机器上不建议直接开 DDP；本代码是单进程训练。两张卡并行跑两个模型更简单，也符合报告的对照实验。

### 4.4 输出

```text
checkpoints/task1/<model>/
├── best.pt          # 验证集 monitor 最优
├── last.pt          # 每轮覆盖，含 optimizer/scheduler，可 resume
├── metrics.csv      # 每轮指标
├── summary.json     # 最佳 epoch、历史、路径
└── config.yaml      # 实际使用的配置快照

logs/task1/<experiment>_<timestamp>.log
```

`best.pt` 里包含：`model_config`、`data_config`、完整 `config`、`vocab`、
`state_dict`、`optimizer`、`scheduler`、`epoch`、`val_metrics`、`monitor`、
`monitor_value`、`seed`、`git_commit`。

### 4.5 断点续训

```bash
bash scripts/run_task1_train.sh vit_transformer \
  --resume checkpoints/task1/vit_transformer/last.pt
```

注意：续训时 `train.epochs` 仍表示**总轮数**，不是“再跑多少轮”。例如从第 10 轮恢复到总共 30 轮，仍然写 `epochs: 30`。

### 4.6 换共享词表

任务一的词表由 `prepare_data_task1.py` 用训练集描述构建，两个模型共用。如果重新生成了词表，**重新训练**；不要在已有 checkpoint 上直接换词表（除非按第 8 节的扩词表接口处理）。

---

## 5. 推理与统一评测

### 5.1 单模型推理

```bash
# 贪心
python -m src.task1_caption.infer \
  --checkpoint checkpoints/task1/vit_transformer/best.pt \
  --split test --method greedy \
  --device cuda --num-workers 8 \
  --output outputs/task1/predictions/vit_transformer_test_greedy.csv

# 束搜索
python -m src.task1_caption.infer \
  --checkpoint checkpoints/task1/vit_transformer/best.pt \
  --split test --method beam \
  --beam-size 5 --length-penalty 1.0 \
  --device cuda --num-workers 8 \
  --output outputs/task1/predictions/vit_transformer_test_beam5_lp1.0.csv
```

预测文件永远是三列：

```csv
image_id,prediction,reference
train/000123.jpg,a plate of fried chicken ...,a plate of fried chicken with rice ||| fried chicken and rice
```

`reference` 用 ` ||| ` 分隔多条参考描述。`<pad>/<sos>/<eos>/<unk>` 不会出现在 `prediction` 里。

### 5.2 统一评测

```bash
python -m src.task1_caption.evaluate \
  --pred "resnet18_gru=outputs/task1/predictions/resnet18_gru_test_greedy.csv" \
  --pred "vit_greedy=outputs/task1/predictions/vit_transformer_test_greedy.csv" \
  --pred "vit_beam=outputs/task1/predictions/vit_transformer_test_beam5_lp1.0.csv" \
  --split test \
  --out outputs/task1/metrics/task1_models_test.json \
  --markdown outputs/task1/metrics/task1_models_test.md
```

所有预测文件必须覆盖同一组 `image_id`，且同一张图的 reference 要一致；否则脚本会报错。只做诊断时可以用 `--allow-id-mismatch` 取交集。

### 5.3 一条命令跑一个 checkpoint 的贪心 vs 束搜索

```bash
bash scripts/run_task1_eval.sh checkpoints/task1/vit_transformer/best.pt test 5 1.0
```

参数依次是：checkpoint、split、beam size、length penalty。环境变量：

```bash
DEVICE=cpu NUM_WORKERS=0 bash scripts/run_task1_eval.sh ...
PYTHON=python3 bash scripts/run_task1_eval.sh ...
```

---

## 6. 束搜索对照

### 6.1 在验证集上调参

```bash
python -m src.task1_caption.beam_search \
  --config configs/task1_beam_search.yaml \
  --split val \
  --limit 500
```

默认扫：

- beam size：3、5、7
- length penalty：0.6、0.8、1.0、1.2、1.4

结果：

```text
outputs/task1/beam_search/tuning_results.json
outputs/task1/beam_search/tuning_results.md
outputs/task1/beam_search/val_bs*_lp*.csv
```

选择规则：验证集 CIDEr-D 最高，BLEU-4 作为 tie-break。**测试集不参与选择**。

全量验证集（约 2,000 张）跑全部 15 组会比较久；推荐先用 `--limit 500` 粗筛，再对前 2–3 组用全量验证集复核：

```bash
python -m src.task1_caption.beam_search \
  --config configs/task1_beam_search.yaml \
  --split val --limit 0 \
  --beam-sizes 5 --length-penalties 1.0 1.2
```

`--limit 0` 表示不截断。

### 6.2 用选定设置跑测试集

```bash
python -m src.task1_caption.beam_search \
  --config configs/task1_beam_search.yaml \
  --final-split test
```

这会：

1. 按配置在 val（默认 `limit: 500`）上重跑一遍 sweep，选出最佳；
2. 在 test 上跑一次 greedy 和最佳 beam；
3. 写 `outputs/task1/beam_search/final/metrics.json` 和 `.md`。

如果已经确定了设置，直接：

```bash
bash scripts/run_task1_eval.sh checkpoints/task1/vit_transformer/best.pt test 5 1.0
```

报告里“束宽与长度设置在验证集确定”这句话，对应的就是这一步。

---

## 7. 错误分析

```bash
python -m src.task1_caption.error_analysis \
  --manifest data/processed/food500cap_clean.jsonl \
  --image-root data/raw/food500cap \
  --pred "resnet18_gru=outputs/task1/predictions/resnet18_gru_test_greedy.csv" \
  --pred "vit_transformer=outputs/task1/predictions/vit_transformer_test_greedy.csv" \
  --out-dir outputs/task1/error_cases \
  --max-cases 200 \
  --copy-images
```

输出：

```text
outputs/task1/error_cases/
├── error_cases.csv       # image_id、class、原图路径、参考、两模型预测、自动 flags、人工列
├── summary.json          # 每个模型的四类错误计数/百分比
├── summary.md            # 直接可贴报告的表
└── images/               # --copy-images 时复制原图，便于人工复核
```

四类自动标记：

1. `ingredient_misidentification`：参考里有食材 A、预测里出现食材 B，同时缺 A；
2. `cooking_method_error`：烹饪方式集合与参考不一致；
3. `missing_ingredient`：参考食材在预测里没出现；
4. `unsupported_addition`：预测里出现参考中完全没有依据的内容。

一个样本可能同时命中多个 flag；`primary_*` 按
`ingredient_misidentification > cooking_method_error > missing_ingredient > unsupported_addition`
给出主类别。自动标记只是初筛，报告里要人工确认：

- 在 `error_cases.csv` 里填写 `manual_category_<model>`；
- 建议值就是上面四个 key 或 `none`；
- `notes_<model>` 写简短理由（例如“把 chicken 看成 pork”）。

食材词表来源：

1. manifest 里的 `class`（Food-500 类别名）；
2. 可选文件 `data/vocab/food_lexicon.txt`（一行一个词/短语）；
3. 参考描述中高频、非停用词、非烹饪方式的词。

如果某类错误明显偏多/偏少，先检查词表；必要时手写 `data/vocab/food_lexicon.txt`，
用 `--food-lexicon` 指过去。

---

## 8. 交接

### 8.1 给梁佳庆：ResNet-18＋GRU 基线

交付物：

```text
checkpoints/task1/resnet18_gru/best.pt
checkpoints/task1/resnet18_gru/config.yaml
configs/task1_resnet18_gru.yaml
data/vocab/shared_vocab_5000.json
data/manifests/food500cap_train_ids.txt
data/manifests/food500cap_val_ids.txt
data/manifests/food500cap_test_ids.txt
src/task1_caption/models/resnet18_gru.py
src/task1_caption/datasets.py
src/task1_caption/infer.py
```

复现基线推理：

```bash
python -m src.task1_caption.infer \
  --checkpoint checkpoints/task1/resnet18_gru/best.pt \
  --split test --method greedy \
  --output outputs/task1/predictions/handoff_resnet_test_greedy.csv
```

加载方式（直接复用 checkpoint 里的词表和预处理配置）：

```python
from src.task1_caption.infer import load_bundle, build_split_dataset

bundle = load_bundle("checkpoints/task1/resnet18_gru/best.pt", device="cuda")
model, vocab = bundle["model"], bundle["vocab"]
dataset = build_split_dataset(bundle, split="test")
```

任务二需要扩词表时的推荐做法（旧 token 的 id 不变）：

```python
from src.task1_caption.datasets import Vocab

vocab = Vocab.from_file("data/vocab/shared_vocab_5000.json")
added = vocab.extend(
    new_captions,                 # 任务二的 target_text / JSON 序列
    max_size=8000,                # 目标总大小，可按需调整
    min_freq=1,
)
model.resize_token_embeddings(len(vocab))
# 重新创建 optimizer（embedding / 输出层参数张量已经换了）
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

`resize_token_embeddings` 会保留旧行，新增行随机初始化。词表文件要另存为
`data/vocab/task2_extended_vocab.json`，不要覆盖任务一的共享词表。

验收：

- 用同一个 checkpoint、同一批图片、同一个 `max_len` 跑两次 greedy，输出逐字一致；
- 用脚本扩展词表后，模型能前向，loss 能下降；
- 基线预测文件能被 `src/task1_caption/evaluate.py` 正常评分。

### 8.2 给李春鹏：ViT 交叉熵检查点与 SCST 接口

交付物：

```text
checkpoints/task1/vit_transformer/best.pt     # 验证集 CIDEr-D 最高
configs/task1_vit_transformer.yaml
configs/eval_protocol.yaml
data/manifests/*.txt
data/vocab/shared_vocab_5000.json
src/task1_caption/models/vit_transformer.py
src/task1_caption/beam_search.py
src/task1_caption/infer.py
src/task1_caption/evaluate.py
src/common/metrics.py
```

加载和接口：

```python
import torch
from src.task1_caption.infer import load_bundle
from src.task1_caption.models import sequence_log_prob
from src.task1_caption.beam_search import greedy_decode

bundle = load_bundle("checkpoints/task1/vit_transformer/best.pt", device="cuda")
model, vocab = bundle["model"], bundle["vocab"]

# images: [B, 3, 224, 224]
# token_ids: [B, L]，第 0 位必须是 <sos>(id=1)，通常以 <eos>(id=2) 结尾；
#            之后用 <pad>(id=0) 补齐。pad 位置不计入 log-prob。
model.train()   # SCST 训练时必须让 dropout 生效、梯度可回传
log_prob = sequence_log_prob(model, images, token_ids)   # [B]，可反向传播

# 自批评基线：同一模型、同一图像的贪心输出
greedy_ids = greedy_decode(model, images, max_len=40, sos_id=1, eos_id=2)  # [B, T]
greedy_text = vocab.decode(greedy_ids[0].tolist())
```

约束：

- `sequence_log_prob` **没有**加 `@torch.no_grad()`，梯度会回传；
- 它内部使用 teacher-forcing 的 `forward(images, input_ids)`，和手工
  `log_softmax(logits).gather(target)` 完全一致；
- `token_ids` 里的 `<pad>` 只用于补齐，不参与损失；
- `greedy_decode` 是确定性 argmax，不采样；采样由 SCST 侧自己用 logits 完成。

验收（对应开题“同输入贪心输出一致 / 指标误差 < 0.001 / 对数概率抽查误差 < 0.0001”）：

```bash
# 1. 接口自测：log-prob 对照 + greedy 确定性 + beam 长度
python -m unittest tests.test_interface -v

# 2. 指标自测：与 pycocoevalcap 的参考值对照
python -m unittest tests.test_metrics -v
```

另外可以用同一份预测 CSV 跑：

```bash
python -m src.task1_caption.evaluate --pred "scst=你的预测.csv" --split test
```

李春鹏的 SCST 预测也应按三列格式落盘，这样指标口径完全一致。

---

## 9. 指标口径（冻结）

实现位于 `src/common/metrics.py`，参数位于 `configs/eval_protocol.yaml`。所有任务一实验必须使用同一套。

| 项 | 默认值/说明 |
| --- | --- |
| 分词 | 小写；保留字母数字、内部单引号和连字符（`stir-fried` 是一个 token）；丢标点 |
| BLEU-4 | 语料级、多参考最大计数、closest reference length、COCO 的 `1e-15/1e-9` 平滑；报告为 0–100 |
| ROUGE-L | `pycocoevalcap` 口径：beta=1.2，参考集上分别取最大 P / 最大 R；报告为 0–100 |
| CIDEr-D | `pycocoevalcap` 口径：1–4 gram tf-idf，按图片统计 document frequency，`N=图片数`，Gaussian 长度惩罚 sigma=6，`min(tf_hyp, tf_ref)` 相似度；默认 **不截断计数**；官方 ×10 尺度 |
| 重复率 `rep4` | 每条预测中重复 4-gram 的比例，再对句子取平均；0–1 |
| 平均句长 | 分词后的 token 数 |

两个容易混淆的点：

1. **CIDEr-D 的数值尺度**。默认沿用 COCO 官方实现的 ×10 尺度（完美单图 = 10，常见模型约 0.5–2）。如果报告表格想统一成 0–100，把 `eval_protocol.yaml` 的
   `metrics.cider_d.scale` 改成 `100`，然后**重新跑所有评测**；不要混用两种尺度的数字。
2. **CIDEr-D 的截断**。官方 COCO 代码不做 count clipping；CIDEr-D 论文版本会把计数截到 10。默认 `clip: 0` 与官方实现一致；如果要报告论文口径，改成 `clip: 10`，并在报告里注明“启用计数截断”。两种口径不能混着比较。

`tests/test_metrics.py` 里锁定了与 `pycocoevalcap` 对照的 BLEU-4 / ROUGE-L / CIDEr-D 数值。改 `metrics.py` 任何公式，这个测试会失败；这是有意的。

---

## 10. 产物清单与检查点对应

| 开题检查点 | 产物 |
| --- | --- |
| 检查点 1：数据划分、技术路线、评测方案 | `data/manifests/`、`data/vocab/`、`configs/eval_protocol.yaml`、本说明 |
| 检查点 2：两模型代码、日志、初版接口 | `src/task1_caption/`、`tests/`、`logs/task1/`、冒烟日志 |
| 检查点 3：三项主指标、束搜索、错误案例 | `outputs/task1/metrics/`、`outputs/task1/beam_search/`、`outputs/task1/error_cases/` |
| 检查点 4：权重、配置、推理/评测脚本 | `checkpoints/task1/*/best.pt`、`configs/task1_*.yaml`、`scripts/run_task1_*.sh` |

建议在 `checkpoints/manifest.csv` 里登记正式权重：

```csv
model,path,sha256,val_bleu4,val_rouge_l,val_cider_d,owner,date,notes
resnet18_gru,checkpoints/task1/resnet18_gru/best.pt,<sha256>,...,...,...,单文杰,2026-..,task1 baseline
vit_transformer,checkpoints/task1/vit_transformer/best.pt,<sha256>,...,...,...,单文杰,2026-..,selected by val CIDEr-D
```

`sha256sum checkpoints/task1/.../best.pt` 生成哈希；指标从
`checkpoints/task1/.../summary.json` 或 `metrics.csv` 的最佳 epoch 抄。

---

## 11. 常见问题

**`FileNotFoundError: image not found ...`**
`data.image_root` 和 manifest 里的相对路径不一致。默认应是 `data/raw/food500cap`。用 `--set data.image_root=...` 或修改 YAML；推理时也可以 `--image-root ...` 覆盖。

**`no annotation files found`**
标注不在 `raw-dir` 顶层，或文件名不匹配。用 `--annotations` 显式指文件/目录；如果标注还在 zip 里，先解压。

**`KeyError: data_config has no 'test_ids'`**
没有生成测试 ID 清单，先跑 `prepare_data_task1.py`；或者用 `--ids-file` 指定自己的清单。

**ViT 报位置编码尺寸不匹配**
`ViT-B/16` 的输入必须是 224×224（196 个 patch + 1 个 class token）。不要把 `data.image_size` 改成其他值，除非同时改模型的位置编码。

**CUDA out of memory**
先降 `train.batch_size` 并用 `train.grad_accum_steps` 保持有效批大小；ViT 还可以降 `data.num_workers`。例如：

```bash
--set train.batch_size=16 --set train.grad_accum_steps=2
```

**预训练权重下载失败**
见第 1 节；也可以先用 `--set model.pretrained=false` 冒烟，但正式实验不建议完全从零训练视觉编码器（和报告“先冻结预训练编码器”的设定不符）。

**DataLoader worker 报错 / 卡住**
先用 `--set data.num_workers=0` 排除多进程问题，再逐步调大。

**束搜索很慢**
束搜索是逐图循环；GPU 上 5k 测试集 × beam 5 仍在可接受范围。先用 `--limit 500` 调参，正式测试只跑选定设置。CPU 上会很慢。

**验证集只有 1 张图时 CIDEr-D 为 0**
`pycocoevalcap` 的 CIDEr 用 `log(N_images)` 做 idf，`N=1` 时所有 idf 为 0，分数为 0。这是参考实现的特性；用完整验证集即可。

**指标和论文数字差很多**
先确认 CIDEr 尺度（×10 vs ×100）和是否启用了 count clipping；再确认分词器。我们的实现和 `pycocoevalcap` 在相同 tokenized 字符串上逐位一致（`tests/test_metrics.py`），差异通常来自 tokenizer 或尺度，而不是公式。

---

## 12. 和开题报告的对应关系（写报告时可直接引用）

- 4.1.1：`prepare_data_task1.py` 的清洗报告对应“核查损坏图像、错误配对与跨划分重复”；`data/manifests` 是三个 ID 清单；`shared_vocab_5000.json` 是两模型共享词表。
- 4.1.2：`resnet18_gru.py` 用 ResNet-18 全局特征初始化 GRU 隐状态；`vit_transformer.py` 保留 patch token，用因果自注意力 + 交叉注意力解码。
- 4.1.3：`train.py` 的 `freeze_encoder_epochs` / `unfreeze_mode` 对应“先冻结编码器，是否解冻末段由验证集确定”；`metrics.py` 对应统一评测；`beam_search.py` 对应束搜索增项；`error_analysis.py` 对应四类错误统计。
- 组内交接：第 8 节分别对应表 4 中给梁佳庆、李春鹏的材料与验收标准。

如果后续要补任务二/任务三，只需要新增对应目录；本说明中的词表扩展接口和指标脚本可以直接复用。
