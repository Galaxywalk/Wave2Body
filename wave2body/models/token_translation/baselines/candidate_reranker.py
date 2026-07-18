# -*- coding: utf-8 -*-
"""Candidate reranker heads for part-token codebook selection."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


class PartTokenCandidateReranker(nn.Module):
    """Score learned-logit top-k code candidates with latent/codebook evidence.

    The head is intentionally lightweight: it keeps the original translator
    logits as candidate proposal scores, adds a codebook-distance score from
    ``pred_z_e``, and learns a small bilinear compatibility score between the
    predicted latent token and each candidate codebook vector.
    """

    def __init__(
        self,
        token_dim: int,
        rank_dim: int = 64,
        init_learned_weight: float = 1.0,
        init_distance_weight: float = 1.0,
        init_bilinear_weight: float = 0.1,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.rank_dim = int(rank_dim)
        if self.token_dim <= 0:
            raise ValueError(f"token_dim must be positive, got {token_dim}")
        if self.rank_dim <= 0:
            raise ValueError(f"rank_dim must be positive, got {rank_dim}")

        self.z_norm = nn.LayerNorm(self.token_dim) if use_layer_norm else nn.Identity()
        self.code_norm = nn.LayerNorm(self.token_dim) if use_layer_norm else nn.Identity()
        self.z_proj = nn.Linear(self.token_dim, self.rank_dim, bias=False)
        self.code_proj = nn.Linear(self.token_dim, self.rank_dim, bias=False)
        self.learned_weight = nn.Parameter(torch.tensor(float(init_learned_weight), dtype=torch.float32))
        self.distance_weight = nn.Parameter(torch.tensor(float(init_distance_weight), dtype=torch.float32))
        self.bilinear_weight = nn.Parameter(torch.tensor(float(init_bilinear_weight), dtype=torch.float32))

        nn.init.normal_(self.z_proj.weight, std=0.02)
        nn.init.normal_(self.code_proj.weight, std=0.02)

    def forward(
        self,
        *,
        vqvae: Any,
        pred_z_e: torch.Tensor,
        pred_logits: torch.Tensor,
        candidate_topk: int,
        distance_scale: float = 1.0,
    ) -> torch.Tensor:
        if pred_z_e.dim() != 3:
            raise ValueError(f"pred_z_e must be [B,T,D], got {tuple(pred_z_e.shape)}")
        if pred_logits.dim() != 3:
            raise ValueError(f"pred_logits must be [B,T,C], got {tuple(pred_logits.shape)}")
        if pred_z_e.shape[:2] != pred_logits.shape[:2]:
            raise ValueError(f"token shape mismatch: {tuple(pred_z_e.shape)} vs {tuple(pred_logits.shape)}")
        if int(pred_z_e.shape[-1]) != self.token_dim:
            raise ValueError(f"token dim mismatch: pred_z_e={int(pred_z_e.shape[-1])}, head={self.token_dim}")
        if candidate_topk <= 0:
            raise ValueError(f"candidate_topk must be positive, got {candidate_topk}")
        if distance_scale <= 0:
            raise ValueError(f"distance_scale must be > 0, got {distance_scale}")
        if not hasattr(vqvae, "layout") or not hasattr(vqvae, "quantizers"):
            raise RuntimeError("Expected a part-level VQ-VAE with layout and quantizers.")

        layout = vqvae.layout
        masked = torch.full_like(pred_logits.float(), float("-inf"))
        z_all = pred_z_e.float()
        logits_all = pred_logits.float()
        scale = math.sqrt(float(self.rank_dim))

        for part_name in layout.part_names:
            token_slice = layout.part_token_slices[part_name]
            offset = int(layout.part_codebook_offsets[part_name])
            codebook_size = int(layout.part_codebook_size[part_name])
            end = offset + codebook_size
            if offset < 0 or codebook_size <= 0 or end > int(pred_logits.shape[-1]):
                raise ValueError(
                    f"Invalid codebook slice for {part_name}: offset={offset}, size={codebook_size}, "
                    f"logits_dim={int(pred_logits.shape[-1])}"
                )

            learned_local = logits_all[:, token_slice, offset:end]
            effective_k = min(int(candidate_topk), codebook_size)
            top_idx = torch.topk(learned_local, k=effective_k, dim=-1).indices

            z = z_all[:, token_slice, :]
            codebook = vqvae.quantizers[part_name].codebook.to(device=pred_z_e.device, dtype=torch.float32)
            candidate_code = codebook[top_idx]
            z_exp = z.unsqueeze(2).expand_as(candidate_code)

            z_proj = self.z_proj(self.z_norm(z)).unsqueeze(2)
            code_proj = self.code_proj(self.code_norm(candidate_code))
            bilinear = (z_proj * code_proj).sum(dim=-1) / scale
            dist = -float(distance_scale) * ((z_exp - candidate_code) ** 2).sum(dim=-1)
            learned = learned_local.gather(-1, top_idx)
            scores = self.learned_weight * learned + self.distance_weight * dist + self.bilinear_weight * bilinear

            masked[:, token_slice, offset:end].scatter_(-1, top_idx, scores)

        return masked


__all__ = ["PartTokenCandidateReranker"]
