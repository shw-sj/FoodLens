# data/ 目录说明（任务一）

本目录只提交**ID 清单**、**词表**和说明文件，原始图片与处理后的缓存不入 Git。

## 放置原始数据

把课程提供的 Food-500 Cap 数据解压到 `data/raw/food500cap/`。常见布局均可：

```text
data/raw/food500cap/
├── annotations.json          # 或 train.json / test.json / annotations/*.json
├── images/                   # 或直接 train/、test/ 图片子目录
│   ├── ...
└── ...
```

如果压缩包是 `annotations.zip` / `images.zip`，先解压：

```bash
unzip data/raw/annotations.zip -d data/raw/food500cap
unzip data/raw/images.zip      -d data/raw/food500cap/images
```

如果还没有下载数据，可以用 `scripts/data_load.py` 从 HuggingFace 拉取
`advancedcv/Food500Cap` 并落盘（需要 `pip install datasets`）：

```bash
python scripts/data_load.py --out-dir data/raw/food500cap --inspect
python scripts/data_load.py --out-dir data/raw/food500cap
```

`--inspect` 只查看列名和 split；正式下载会生成 `annotations.json` 和
`images/<split>/<class>/`。国内可用 `HF_ENDPOINT=https://hf-mirror.com`。

## 生成文件

```bash
python scripts/prepare_data_task1.py --raw-dir data/raw/food500cap
```

脚本会读取/生成：

| 路径 | 是否入库 | 内容 |
| --- | --- | --- |
| `data/raw/` | 否 | 原始图片与标注 |
| `data/processed/food500cap_clean.jsonl` | 否 | 清洗后的清单（图片相对路径、参考描述、类别、划分、MD5） |
| `data/processed/food500cap_prepare_report.json` | 否 | 清洗/去重/划分统计报告 |
| `data/manifests/food500cap_train_ids.txt` | 是 | 训练集 image_id 清单 |
| `data/manifests/food500cap_val_ids.txt` | 是 | 验证集 image_id 清单 |
| `data/manifests/food500cap_test_ids.txt` | 是 | 测试集 image_id 清单（优先保留发布方官方测试集） |
| `data/vocab/shared_vocab_5000.json` | 是 | 两模型共享、仅用训练集描述构建的词表 |
| `data/vocab/food_lexicon.txt` | 可选 | 错误分析用的食材词表（一行一个词/短语） |

> `image_id` 使用相对于 `data.image_root`（默认 `data/raw/food500cap`）的 POSIX
> 路径，例如 `train/000123.jpg`。预测 CSV 中的 `image_id` 与之完全一致。
