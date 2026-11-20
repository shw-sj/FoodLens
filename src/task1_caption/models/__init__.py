"""Task-1 caption models and the SCST hand-off interface.

Public API used by ``train.py`` / ``infer.py`` / ``beam_search.py``:

* ``build_model(config, vocab_size, max_len)``
* ``sequence_log_prob(model, images, token_ids)`` -- differentiable
  log-probability of a full (sampled) sentence, used by the SCST owner.
* ``greedy_decode`` lives in ``src.task1_caption.beam_search`` and is shared
  by training-time validation and inference.

Both models implement the same small decoding interface::

    state = model.init_decode_state(images)          # images: [N, 3, H, W]
    logits, state = model.decode_step(state, prev)   # prev: [N]
    state = model.select_decode_state(state, idx)
    state = model.repeat_decode_state(state, repeats)

so ``beam_search.py`` can run on either architecture without special cases.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "ResNet18GRU",
    "ViTTransformer",
    "build_model",
    "sequence_log_prob",
    "load_state_dict_file",
]


def load_state_dict_file(module: torch.nn.Module, path: str) -> Tuple[int, int]:
    """Load a plain state dict or a training checkpoint into ``module``.

    Returns ``(missing, unexpected)`` key counts.  ``module.`` prefixes from
    DataParallel checkpoints are stripped.
    """
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 1.13
        obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise ValueError(f"cannot interpret checkpoint weights: {path}")

    cleaned: Dict[str, Any] = {}
    for key, value in obj.items():
        cleaned[key[len("module.") :] if key.startswith("module.") else key] = value

    incompatible = module.load_state_dict(cleaned, strict=False)
    return len(incompatible.missing_keys), len(incompatible.unexpected_keys)


def build_model(config: Dict[str, Any], vocab_size: int, max_len: int):
    from .resnet18_gru import ResNet18GRU
    from .vit_transformer import ViTTransformer

    model_cfg = config.get("model", config)
    name = str(model_cfg.get("name", "")).lower()

    common = dict(
        vocab_size=int(vocab_size),
        pad_id=int(model_cfg.get("pad_id", 0)),
        pretrained=bool(model_cfg.get("pretrained", True)),
        pretrained_path=model_cfg.get("pretrained_path"),
    )

    if name in ("resnet18_gru", "resnet18-gru", "resnet_gru"):
        return ResNet18GRU(
            embed_dim=int(model_cfg.get("embed_dim", 256)),
            hidden_dim=int(model_cfg.get("hidden_dim", 512)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            **common,
        )
    if name in ("vit_transformer", "vit-transformer", "vit"):
        return ViTTransformer(
            d_model=int(model_cfg.get("d_model", 512)),
            nhead=int(model_cfg.get("nhead", 8)),
            num_decoder_layers=int(model_cfg.get("num_decoder_layers", 4)),
            dim_feedforward=int(model_cfg.get("dim_feedforward", 2048)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            max_len=int(max_len),
            **common,
        )
    raise ValueError(
        f"unknown model.name={name!r}; expected resnet18_gru or vit_transformer"
    )


def sequence_log_prob(
    model: torch.nn.Module,
    images: torch.Tensor,
    token_ids: torch.Tensor,
    pad_id: int = 0,
) -> torch.Tensor:
    """Differentiable sum of log p(token_t | image, prefix_t) per sequence.

    Parameters
    ----------
    model:
        A task-1 model (``forward(images, input_ids) -> logits``).
    images:
        ``[N, 3, H, W]``.
    token_ids:
        ``[N, L]`` full sequences starting with ``<sos>`` and normally ending
        with ``<eos>``.  Positions after the first ``<pad>`` are ignored.

    Returns
    -------
    ``[N]`` tensor.  Gradients flow through the model; this is the interface
    used by SCST to score sampled sentences.

    Notes
    -----
    No ``@torch.no_grad()`` is applied on purpose.  The caller controls the
    train/eval mode and the optimizer.  The model's teacher-forced forward
    pass is used, so the result is identical to manually summing
    ``log_softmax`` of the logits for the same prefix.
    """
    if token_ids.dim() != 2 or token_ids.size(1) < 2:
        raise ValueError("token_ids must be [N, L] with L >= 2")
    input_ids = token_ids[:, :-1]
    targets = token_ids[:, 1:]
    logits = model(images, input_ids)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    mask = targets.ne(int(pad_id)).to(gathered.dtype)
    return (gathered * mask).sum(dim=-1)
