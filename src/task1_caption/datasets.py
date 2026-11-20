"""Food-500 Cap dataset, vocabulary and data loading for task 1.

The raw annotations are converted once by ``scripts/prepare_data_task1.py``
into a JSONL manifest::

    {"image_id": "train/000123.jpg",
     "image": "train/000123.jpg",
     "captions": ["a plate of ...", "..."],
     "class": "pizza",
     "split": "train",
     "md5": "..."}

``image`` is relative to the configured ``data.image_root``.  Training,
inference and evaluation all read the same manifest and the same ID lists, so
the split is frozen and easy to hand off.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.common.io import resolve_path
from src.common.json_utils import load_jsonl
from src.common.metrics import TOKENIZER_VERSION, tokenize
from src.common.seed import worker_init_fn

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
PAD_ID, SOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Vocab:
    """Token <-> id mapping shared by both task-1 models."""

    def __init__(
        self,
        itos: Sequence[str],
        freqs: Optional[Dict[str, int]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.itos: List[str] = list(itos)
        self.stoi: Dict[str, int] = {token: i for i, token in enumerate(self.itos)}
        self.freqs: Dict[str, int] = dict(freqs or {})
        self.meta: Dict[str, Any] = dict(meta or {})
        if self.itos[: len(SPECIAL_TOKENS)] != SPECIAL_TOKENS:
            raise ValueError(
                "vocabulary must start with the special tokens "
                f"{SPECIAL_TOKENS}, got {self.itos[:4]}"
            )

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def sos_id(self) -> int:
        return SOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    def encode_tokens(self, tokens: Sequence[str]) -> List[int]:
        return [self.stoi.get(token, UNK_ID) for token in tokens]

    def encode(
        self,
        tokens: Sequence[str],
        add_sos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        ids = self.encode_tokens(tokens)
        if add_sos:
            ids = [SOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids

    def decode(self, ids: Iterable[int], strip_special: bool = True) -> str:
        words: List[str] = []
        for idx in ids:
            idx = int(idx)
            if idx < 0 or idx >= len(self.itos):
                continue
            token = self.itos[idx]
            if strip_special and token in SPECIAL_TOKENS:
                continue
            words.append(token)
        return " ".join(words)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "itos": self.itos,
            "stoi": self.stoi,
            "freqs": self.freqs,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Vocab":
        return cls(
            itos=data["itos"],
            freqs=data.get("freqs"),
            meta=data.get("meta"),
        )

    @classmethod
    def from_file(cls, path: str) -> "Vocab":
        from src.common.io import load_json

        data = load_json(path)
        if data is None:
            raise FileNotFoundError(f"vocabulary not found: {path}")
        return cls.from_dict(data)

    def save(self, path: str) -> None:
        from src.common.io import save_json

        save_json(self.to_dict(), path)

    def extend(
        self,
        captions: Iterable[str],
        max_size: Optional[int] = None,
        min_freq: int = 1,
    ) -> int:
        """Append unseen tokens from new captions, keeping old ids stable.

        This is the task-2 hand-off helper: 梁佳庆 can extend the task-1
        vocabulary with JSON field names / numbers / new food names without
        remapping the existing embedding rows.  Call
        ``model.resize_token_embeddings(len(vocab))`` afterwards.

        Returns the number of tokens added.
        """
        counter: Counter = Counter()
        for caption in captions:
            counter.update(tokenize(caption))
        existing = set(self.stoi)
        new_tokens = [
            (token, count)
            for token, count in counter.items()
            if token not in existing and count >= int(min_freq)
        ]
        new_tokens.sort(key=lambda item: (-item[1], item[0]))
        if max_size is not None:
            room = max(0, int(max_size) - len(self.itos))
            new_tokens = new_tokens[:room]
        for token, count in new_tokens:
            index = len(self.itos)
            self.itos.append(token)
            self.stoi[token] = index
            self.freqs[token] = int(count)
        self.meta["extended"] = int(self.meta.get("extended", 0)) + len(new_tokens)
        self.meta["size_after_extend"] = len(self.itos)
        return len(new_tokens)

    @classmethod
    def build(
        cls,
        captions: Iterable[str],
        max_size: int = 5000,
        min_freq: int = 2,
    ) -> "Vocab":
        """Build a vocabulary from training captions only.

        ``max_size`` counts the four special tokens.  Ties are broken
        lexicographically so the vocabulary is deterministic.
        """
        counter: Counter = Counter()
        n_captions = 0
        n_tokens = 0
        for caption in captions:
            n_captions += 1
            tokens = tokenize(caption)
            n_tokens += len(tokens)
            counter.update(tokens)

        max_words = max(0, int(max_size) - len(SPECIAL_TOKENS))
        kept = [
            (token, count)
            for token, count in counter.items()
            if count >= int(min_freq)
        ]
        kept.sort(key=lambda item: (-item[1], item[0]))
        kept = kept[:max_words]

        itos = list(SPECIAL_TOKENS) + [token for token, _ in kept]
        freqs = {token: count for token, count in kept}
        for token in SPECIAL_TOKENS:
            freqs[token] = 0
        meta = {
            "tokenizer": TOKENIZER_VERSION,
            "max_size": int(max_size),
            "min_freq": int(min_freq),
            "num_captions": n_captions,
            "num_tokens": n_tokens,
            "num_unique_tokens": len(counter),
            "built_from": "train split captions",
        }
        return cls(itos=itos, freqs=freqs, meta=meta)


def build_transform(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size, scale=(0.7, 1.0), ratio=(0.8, 1.25)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    resize = int(round(image_size * 256 / 224))
    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class CaptionDataset(Dataset):
    """One image -> one caption per item during training; all refs at eval."""

    def __init__(
        self,
        manifest: str,
        ids: Optional[Sequence[str]] = None,
        vocab: Optional[Vocab] = None,
        image_root: Optional[str] = None,
        image_size: int = 224,
        max_len: int = 40,
        train: bool = False,
        augment: bool = True,
    ):
        if max_len < 2:
            raise ValueError("max_len must be at least 2 (SOS + EOS)")
        records = load_jsonl(manifest)
        by_id: Dict[str, Dict[str, Any]] = {}
        for record in records:
            image_id = str(record.get("image_id", "")).strip()
            if not image_id:
                raise ValueError(f"manifest record without image_id: {record}")
            if image_id in by_id:
                raise ValueError(f"duplicate image_id in manifest: {image_id}")
            by_id[image_id] = record

        if ids is None:
            self.image_ids: List[str] = list(by_id.keys())
        else:
            missing = [str(i) for i in ids if str(i) not in by_id]
            if missing:
                preview = ", ".join(missing[:5])
                raise KeyError(
                    f"{len(missing)} IDs from the split list are missing from "
                    f"the manifest (e.g. {preview})"
                )
            self.image_ids = [str(i) for i in ids]

        self.records = [by_id[i] for i in self.image_ids]
        self.vocab = vocab
        self.image_root = resolve_path(image_root) if image_root else None
        self.image_size = int(image_size)
        self.max_len = int(max_len)
        self.train = train
        self.transform = build_transform(self.image_size, train=train and augment)

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_image(self, record: Dict[str, Any]) -> Path:
        image_value = str(record.get("image", "")).strip()
        if not image_value:
            raise ValueError(f"manifest record without image path: {record}")
        image_path = Path(image_value)
        candidates: List[Path] = []
        if image_path.is_absolute():
            candidates.append(image_path)
        else:
            if self.image_root is not None:
                candidates.append(self.image_root / image_path)
            candidates.append(resolve_path(image_value))  # project root fallback
        for candidate in candidates:
            if candidate and candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"image not found for {record.get('image_id')}: tried "
            + ", ".join(str(c) for c in candidates if c)
            + ". Set data.image_root (or --image-root) to the folder that "
            "contains the manifest's relative image paths."
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        captions = [str(c).strip() for c in record.get("captions", []) if str(c).strip()]
        if not captions:
            raise ValueError(f"no captions for image {record.get('image_id')}")

        if self.train and len(captions) > 1:
            caption = random.choice(captions)
        else:
            caption = captions[0]

        # Reserve two slots for SOS / EOS.
        tokens = tokenize(caption)[: self.max_len - 2]
        token_ids = self.vocab.encode(tokens, add_sos=True, add_eos=True) if self.vocab else []
        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]

        image = Image.open(self._resolve_image(record)).convert("RGB")
        image = self.transform(image)

        return {
            "image_id": record["image_id"],
            "image": image,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "references": captions,
            "caption": caption,
            "class": record.get("class"),
        }


def collate_batch(batch: Sequence[Dict[str, Any]], pad_id: int = PAD_ID) -> Dict[str, Any]:
    """Pad variable-length sequences and stack images."""
    max_len = max(int(item["input_ids"].numel()) for item in batch)
    images = torch.stack([item["image"] for item in batch], dim=0)

    input_ids = torch.full(
        (len(batch), max_len), pad_id, dtype=torch.long
    )
    target_ids = torch.full(
        (len(batch), max_len), pad_id, dtype=torch.long
    )
    for row, item in enumerate(batch):
        length = int(item["input_ids"].numel())
        input_ids[row, :length] = item["input_ids"]
        target_ids[row, :length] = item["target_ids"]

    return {
        "image_ids": [item["image_id"] for item in batch],
        "images": images,
        "input_ids": input_ids,
        "target_ids": target_ids,
        "references": [item["references"] for item in batch],
        "classes": [item.get("class") for item in batch],
    }


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    kwargs: Dict[str, Any] = dict(
        dataset=dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=bool(drop_last),
        collate_fn=collate_batch,
        worker_init_fn=worker_init_fn,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = False
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def load_split(
    manifest: str,
    ids_file: str,
    vocab: Vocab,
    image_root: Optional[str],
    image_size: int,
    max_len: int,
    train: bool,
    augment: bool = True,
    limit: Optional[int] = None,
) -> CaptionDataset:
    from src.common.io import load_ids

    ids = load_ids(ids_file)
    if limit is not None and int(limit) > 0:
        ids = ids[: int(limit)]
    return CaptionDataset(
        manifest=manifest,
        ids=ids,
        vocab=vocab,
        image_root=image_root,
        image_size=image_size,
        max_len=max_len,
        train=train,
        augment=augment,
    )
