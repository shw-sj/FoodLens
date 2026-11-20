"""ResNet-18 image encoder + GRU caption decoder.

Architecture (matches the proposal):

* ImageNet-pretrained ResNet-18, global average pool -> 512-d feature.
* A linear projection + tanh produces the visual conditioning vector ``v``.
* ``v`` initialises the GRU hidden state and is also concatenated with every
  word embedding, so the decoder keeps seeing the image at each step.
* A linear layer maps GRU states to vocabulary logits.

The encoder can be frozen for the first ``freeze_encoder_epochs`` epochs and
then partially unfrozen (``layer4``) -- both options are set from YAML.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from . import load_state_dict_file


class ResNet18GRU(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        pad_id: int = 0,
    ):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if (pretrained and not pretrained_path) else None
        self.encoder = resnet18(weights=weights)
        if pretrained_path:
            missing, unexpected = load_state_dict_file(self.encoder, str(pretrained_path))
            if missing or unexpected:
                print(
                    f"[resnet18] loaded encoder weights with "
                    f"{missing} missing / {unexpected} unexpected keys"
                )

        feat_dim = int(self.encoder.fc.in_features)
        self.encoder.fc = nn.Identity()

        self.pad_id = int(pad_id)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.vocab_size = int(vocab_size)

        self.vis_proj = nn.Linear(feat_dim, self.hidden_dim)
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.pad_id)
        self.gru = nn.GRU(self.embed_dim + self.hidden_dim, self.hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(float(dropout))
        self.fc = nn.Linear(self.hidden_dim, self.vocab_size)

        self._init_new_parameters()

    def _init_new_parameters(self) -> None:
        nn.init.xavier_uniform_(self.vis_proj.weight)
        nn.init.zeros_(self.vis_proj.bias)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embed.weight[self.pad_id].fill_(0.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def resize_token_embeddings(self, new_vocab_size: int) -> None:
        """Grow/shrink the embedding and output layer, keeping old rows.

        Used when the task-2 owner extends the vocabulary.  Re-create the
        optimizer after calling this (the parameter tensors are replaced).
        """
        new_vocab_size = int(new_vocab_size)
        old_vocab, embed_dim = self.embed.weight.shape
        if new_vocab_size == old_vocab:
            return
        device = self.embed.weight.device
        dtype = self.embed.weight.dtype

        new_embed = nn.Embedding(
            new_vocab_size, embed_dim, padding_idx=self.pad_id
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
        """``[N,3,H,W] -> [N,hidden_dim]`` visual conditioning vector."""
        features = self.encoder(images)
        return torch.tanh(self.vis_proj(features))

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Teacher-forced logits ``[N, L, vocab_size]``."""
        visual = self.encode(images)  # [N, H]
        embeddings = self.embed(input_ids)  # [N, L, E]
        visual_seq = visual.unsqueeze(1).expand(-1, input_ids.size(1), -1)
        inputs = torch.cat([embeddings, visual_seq], dim=-1)
        hidden = visual.unsqueeze(0).contiguous()  # [1, N, H]
        outputs, _ = self.gru(inputs, hidden)
        return self.fc(self.dropout(outputs))

    # -- decoding interface ----------------------------------------------
    def init_decode_state(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        visual = self.encode(images)
        return {"visual": visual, "hidden": visual.unsqueeze(0).contiguous()}

    def decode_step(
        self, state: Dict[str, torch.Tensor], prev_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        visual = state["visual"]
        hidden = state["hidden"]
        embeddings = self.embed(prev_tokens)  # [N, E]
        step_input = torch.cat([embeddings, visual], dim=-1).unsqueeze(1)  # [N,1,E+H]
        output, hidden = self.gru(step_input, hidden)
        logits = self.fc(self.dropout(output.squeeze(1)))
        return logits, {"visual": visual, "hidden": hidden}

    def select_decode_state(
        self, state: Dict[str, torch.Tensor], indices: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        indices = indices.to(state["visual"].device)
        return {
            "visual": state["visual"].index_select(0, indices),
            "hidden": state["hidden"].index_select(1, indices),
        }

    def repeat_decode_state(
        self, state: Dict[str, torch.Tensor], repeats: int
    ) -> Dict[str, torch.Tensor]:
        return {
            "visual": state["visual"].repeat_interleave(int(repeats), dim=0),
            "hidden": state["hidden"].repeat_interleave(int(repeats), dim=1),
        }

    # -- freezing ---------------------------------------------------------
    def encoder_module(self) -> nn.Module:
        return self.encoder

    def last_encoder_block(self) -> nn.Module:
        return self.encoder.layer4

    def set_encoder_trainable(self, trainable: bool) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = bool(trainable)

    def unfreeze_last_encoder_block(self) -> None:
        for parameter in self.encoder.layer4.parameters():
            parameter.requires_grad = True

    def trainable_parameter_counts(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
