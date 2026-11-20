"""ViT-B/16 patch encoder + causal Transformer caption decoder.

The encoder keeps all 196 patch tokens (plus the class token) and the decoder
reads them through cross-attention.  The decoder is a small pre-LN
Transformer (4 layers by default) written with an explicit key/value cache so
greedy / beam-search decoding is incremental instead of recomputing the whole
prefix at every step.

The cache is also what makes the SCRST hand-off efficient: the SCST owner can
call ``model(images, token_ids)`` (or ``sequence_log_prob``) for training and
``greedy_decode`` for the self-critical baseline without touching internals.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ViT_B_16_Weights, vit_b_16

from . import load_state_dict_file


def _split_heads(x: torch.Tensor, nhead: int) -> torch.Tensor:
    batch, length, dim = x.shape
    return x.view(batch, length, nhead, dim // nhead).transpose(1, 2).contiguous()


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, nhead, length, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch, length, nhead * head_dim)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.nhead = int(nhead)
        self.head_dim = d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, length, dim = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.nhead, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3,B,H,L,Dh]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(causal, float("-inf"))
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :].to(torch.bool), float("-inf")
            )
        attention = self.dropout(torch.softmax(scores, dim=-1))
        output = torch.matmul(attention, v)
        return self.out(_merge_heads(output))

    def step(
        self, x: torch.Tensor, cache: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """One-token update; ``x`` is ``[B, 1, D]`` and ``cache`` holds K/V."""
        batch = x.size(0)
        qkv = self.qkv(x).view(batch, 1, 3, self.nhead, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3,B,H,1,Dh]
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = self.dropout(torch.softmax(scores, dim=-1))
        output = torch.matmul(attention, v)
        return self.out(_merge_heads(output)), (k, v)


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.nhead = int(nhead)
        self.head_dim = d_model // nhead
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def project(self, memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cache K/V for a fixed image memory ``[B, S, D]``."""
        return _split_heads(self.k(memory), self.nhead), _split_heads(self.v(memory), self.nhead)

    def attend(
        self, x: torch.Tensor, key_value: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        k, v = key_value
        q = _split_heads(self.q(x), self.nhead)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = self.dropout(torch.softmax(scores, dim=-1))
        output = torch.matmul(attention, v)
        return self.out(_merge_heads(output))


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class DecoderLayer(nn.Module):
    """Pre-LN decoder layer with self-attention cache and cross-attention."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = CausalSelfAttention(d_model, nhead, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = CrossAttention(d_model, nhead, dropout)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, dim_feedforward, dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory_kv: Tuple[torch.Tensor, torch.Tensor],
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.ln1(x), key_padding_mask)
        x = x + self.cross_attn.attend(self.ln2(x), memory_kv)
        x = x + self.ffn(self.ln3(x))
        return x

    def step(
        self,
        x: torch.Tensor,
        self_cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
        cross_kv: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        self_out, new_cache = self.self_attn.step(self.ln1(x), self_cache)
        x = x + self_out
        x = x + self.cross_attn.attend(self.ln2(x), cross_kv)
        x = x + self.ffn(self.ln3(x))
        return x, new_cache


class ViTTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 40,
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        pad_id: int = 0,
    ):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if (pretrained and not pretrained_path) else None
        self.vit = vit_b_16(weights=weights)
        if pretrained_path:
            missing, unexpected = load_state_dict_file(self.vit, str(pretrained_path))
            if missing or unexpected:
                print(
                    f"[vit_b_16] loaded encoder weights with "
                    f"{missing} missing / {unexpected} unexpected keys"
                )
        self.vit.heads = nn.Identity()
        self.encoder_dim = int(self.vit.hidden_dim)

        self.pad_id = int(pad_id)
        self.vocab_size = int(vocab_size)
        self.max_len = int(max_len)
        self.d_model = int(d_model)

        self.vis_proj = nn.Linear(self.encoder_dim, self.d_model)
        self.embed = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_id)
        self.pos = nn.Embedding(self.max_len, self.d_model)
        self.dropout = nn.Dropout(float(dropout))
        self.layers = nn.ModuleList(
            [
                DecoderLayer(self.d_model, nhead, dim_feedforward, dropout)
                for _ in range(int(num_decoder_layers))
            ]
        )
        self.ln_f = nn.LayerNorm(self.d_model)
        self.fc = nn.Linear(self.d_model, self.vocab_size)

        self._init_new_parameters()

    def _init_new_parameters(self) -> None:
        for module in [self.vis_proj, self.fc]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embed.weight[self.pad_id].fill_(0.0)
        nn.init.normal_(self.pos.weight, mean=0.0, std=0.02)
        for layer in self.layers:
            for submodule in [layer.self_attn.qkv, layer.self_attn.out,
                              layer.cross_attn.q, layer.cross_attn.k,
                              layer.cross_attn.v, layer.cross_attn.out]:
                nn.init.xavier_uniform_(submodule.weight)
                nn.init.zeros_(submodule.bias)
            nn.init.xavier_uniform_(layer.ffn.linear1.weight)
            nn.init.zeros_(layer.ffn.linear1.bias)
            nn.init.xavier_uniform_(layer.ffn.linear2.weight)
            nn.init.zeros_(layer.ffn.linear2.bias)

    def resize_token_embeddings(self, new_vocab_size: int) -> None:
        """Grow/shrink the embedding and output layer, keeping old rows.

        Used when the task-2 owner extends the vocabulary.  Re-create the
        optimizer after calling this (the parameter tensors are replaced).
        """
        new_vocab_size = int(new_vocab_size)
        old_vocab, d_model = self.embed.weight.shape
        if new_vocab_size == old_vocab:
            return
        device = self.embed.weight.device
        dtype = self.embed.weight.dtype

        new_embed = nn.Embedding(
            new_vocab_size, d_model, padding_idx=self.pad_id
        ).to(device=device, dtype=dtype)
        with torch.no_grad():
            keep = min(old_vocab, new_vocab_size)
            new_embed.weight[:keep].copy_(self.embed.weight[:keep])
            if new_vocab_size > old_vocab:
                nn.init.normal_(new_embed.weight[old_vocab:], mean=0.0, std=0.02)
            new_embed.weight[self.pad_id].zero_()
        self.embed = new_embed

        old_out, hidden_dim = self.fc.weight.shape
        new_fc = nn.Linear(hidden_dim, new_vocab_size).to(device=device, dtype=dtype)
        with torch.no_grad():
            keep = min(old_out, new_vocab_size)
            new_fc.weight[:keep].copy_(self.fc.weight[:keep])
            new_fc.bias[:keep].copy_(self.fc.bias[:keep])
            if new_vocab_size > old_out:
                nn.init.xavier_uniform_(new_fc.weight[old_out:])
                nn.init.zeros_(new_fc.bias[old_out:])
        self.fc = new_fc
        self.vocab_size = new_vocab_size

    # -- encoding ---------------------------------------------------------
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """ViT patch memory ``[N, 1 + num_patches, d_model]``."""
        if hasattr(self.vit, "_process_input"):
            tokens = self.vit._process_input(images)
        else:  # pragma: no cover - fallback for unusual torchvision builds
            tokens = self.vit.conv_proj(images)
            tokens = tokens.flatten(2).transpose(1, 2)
        batch = tokens.size(0)
        class_token = self.vit.class_token.expand(batch, -1, -1)
        tokens = torch.cat([class_token, tokens], dim=1)
        expected = int(self.vit.encoder.pos_embedding.size(1))
        if tokens.size(1) != expected:
            raise ValueError(
                "ViT-B/16 expects 224x224 inputs (196 patches + class token). "
                f"Got {tokens.size(1)} tokens, but the pretrained position "
                f"embedding has {expected}. Set data.image_size: 224."
            )
        tokens = self.vit.encoder(tokens)
        return self.vis_proj(tokens)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Teacher-forced logits ``[N, L, vocab_size]``."""
        memory = self.encode(images)
        memory_kvs = [layer.cross_attn.project(memory) for layer in self.layers]

        length = input_ids.size(1)
        if length > self.max_len:
            raise ValueError(
                f"sequence length {length} exceeds model max_len {self.max_len}"
            )
        positions = torch.arange(length, device=input_ids.device)
        x = self.embed(input_ids) + self.pos(positions).unsqueeze(0)
        x = self.dropout(x)
        key_padding_mask = input_ids.eq(self.pad_id)

        for layer, memory_kv in zip(self.layers, memory_kvs):
            x = layer(x, memory_kv, key_padding_mask)
        return self.fc(self.ln_f(x))

    # -- decoding interface ----------------------------------------------
    def init_decode_state(self, images: torch.Tensor) -> Dict[str, Any]:
        memory = self.encode(images)
        cross = [layer.cross_attn.project(memory) for layer in self.layers]
        return {
            "cross": cross,
            "self_caches": [None for _ in self.layers],
            "pos": 0,
        }

    def decode_step(
        self, state: Dict[str, Any], prev_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        position = int(state["pos"])
        if position >= self.max_len:
            raise RuntimeError(f"decoder reached max_len={self.max_len}")
        position_ids = torch.full(
            (prev_tokens.size(0),), position, dtype=torch.long, device=prev_tokens.device
        )
        x = (self.embed(prev_tokens) + self.pos(position_ids)).unsqueeze(1)
        x = self.dropout(x)

        new_caches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        for layer, cross_kv, cache in zip(
            self.layers, state["cross"], state["self_caches"]
        ):
            x, new_cache = layer.step(x, cache, cross_kv)
            new_caches.append(new_cache)
        logits = self.fc(self.ln_f(x).squeeze(1))
        return logits, {
            "cross": state["cross"],
            "self_caches": new_caches,
            "pos": position + 1,
        }

    def select_decode_state(
        self, state: Dict[str, Any], indices: torch.Tensor
    ) -> Dict[str, Any]:
        index = indices.to(state["cross"][0][0].device)
        cross = [
            (k.index_select(0, index), v.index_select(0, index))
            for k, v in state["cross"]
        ]
        caches = []
        for cache in state["self_caches"]:
            if cache is None:
                caches.append(None)
            else:
                caches.append(
                    (
                        cache[0].index_select(0, index),
                        cache[1].index_select(0, index),
                    )
                )
        return {"cross": cross, "self_caches": caches, "pos": state["pos"]}

    def repeat_decode_state(self, state: Dict[str, Any], repeats: int) -> Dict[str, Any]:
        repeats = int(repeats)
        cross = [
            (k.repeat_interleave(repeats, dim=0), v.repeat_interleave(repeats, dim=0))
            for k, v in state["cross"]
        ]
        caches = []
        for cache in state["self_caches"]:
            if cache is None:
                caches.append(None)
            else:
                caches.append(
                    (
                        cache[0].repeat_interleave(repeats, dim=0),
                        cache[1].repeat_interleave(repeats, dim=0),
                    )
                )
        return {"cross": cross, "self_caches": caches, "pos": state["pos"]}

    # -- freezing ---------------------------------------------------------
    def encoder_module(self) -> nn.Module:
        return self.vit

    def last_encoder_block(self) -> nn.Module:
        return self.vit.encoder.layers[-1]

    def set_encoder_trainable(self, trainable: bool) -> None:
        for parameter in self.vit.parameters():
            parameter.requires_grad = bool(trainable)

    def unfreeze_last_encoder_block(self) -> None:
        for parameter in self.vit.encoder.layers[-1].parameters():
            parameter.requires_grad = True

    def trainable_parameter_counts(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
