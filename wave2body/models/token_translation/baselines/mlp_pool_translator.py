# -*- coding: utf-8 -*-
"""Non-attention pooled-token translator for 054 architecture ablations."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MLPPoolTokenTranslator(nn.Module):
    """
    Translate sensor tokens to pose tokens through masked pooling and an MLP.

    This is a deliberately simple lower-bound architecture for the 054
    ablation suite. It keeps the same public forward interface as
    ``CrossAttentionTokenTranslator`` so the existing 052 trainer can use it
    without special cases.
    """

    def __init__(
        self,
        mmwave_token_dim: int = 128,
        pose_token_dim: int = 256,
        num_pose_tokens: int = 24,
        num_mmwave_tokens: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        codebook_size: Optional[int] = None,
        **_: object,
    ) -> None:
        super().__init__()
        self.mmwave_token_dim = int(mmwave_token_dim)
        self.pose_token_dim = int(pose_token_dim)
        self.num_pose_tokens = int(num_pose_tokens)
        self.num_mmwave_tokens = int(num_mmwave_tokens)
        self.hidden_dim = int(hidden_dim)
        self.codebook_size = codebook_size

        self.input_proj = nn.Sequential(
            nn.LayerNorm(self.mmwave_token_dim),
            nn.Linear(self.mmwave_token_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.pose_token_embed = nn.Parameter(torch.randn(1, self.num_pose_tokens, self.hidden_dim))
        nn.init.trunc_normal_(self.pose_token_embed, std=0.02)

        blocks: list[nn.Module] = []
        for _idx in range(max(1, int(num_layers))):
            blocks.extend(
                [
                    nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
        self.mlp = nn.Sequential(*blocks)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.pose_token_dim),
            nn.Dropout(float(dropout)),
        )
        if codebook_size is not None:
            self.token_classifier = nn.Sequential(
                nn.LayerNorm(self.pose_token_dim),
                nn.Linear(self.pose_token_dim, int(codebook_size)),
            )
        else:
            self.token_classifier = None

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return tokens.mean(dim=1)
        if mask.dim() == 3:
            mask = mask.any(dim=1)
        if mask.dim() != 2:
            raise ValueError(f"mask must be [B,N] or [B,T,N], got {tuple(mask.shape)}")
        weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (tokens * weights).sum(dim=1) / denom

    def forward(
        self,
        mmwave_tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = self.input_proj(mmwave_tokens)
        pooled = self._masked_mean(hidden, mask)
        pose_hidden = self.pose_token_embed.expand(mmwave_tokens.shape[0], -1, -1) + pooled.unsqueeze(1)
        pose_hidden = pose_hidden + self.mlp(pose_hidden)
        pose_tokens = self.output_proj(pose_hidden)
        if return_logits and self.token_classifier is not None:
            return pose_tokens, self.token_classifier(pose_tokens)
        return pose_tokens
