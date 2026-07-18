# -*- coding: utf-8 -*-
"""Small neural building blocks for mmWave sensor tokenizers."""

from __future__ import annotations

import torch
import torch.nn as nn


class PointPatchEncoder(nn.Module):
    """PointNet-style local patch encoder supporting radar feature channels."""

    def __init__(self, input_channels: int, latent_dim: int, hidden_dim: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.latent_dim = int(latent_dim)
        self.point_mlp = nn.Sequential(
            nn.Linear(self.input_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.dim() != 4:
            raise ValueError(f"Expected patches [B,G,K,C], got {tuple(patches.shape)}")
        bsz, groups, group_size, channels = patches.shape
        if channels != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, got {channels}")
        x = self.point_mlp(patches.reshape(bsz * groups, group_size, channels))
        pooled = torch.max(x, dim=1, keepdim=True).values
        x = torch.cat([x, pooled.expand_as(x)], dim=-1)
        out = self.global_mlp(torch.max(x, dim=1).values)
        return out.view(bsz, groups, self.latent_dim)


class CenterPositionalEmbedding(nn.Module):
    """MLP positional embedding from patch centers."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, centers: torch.Tensor) -> torch.Tensor:
        return self.net(centers)


class TokenTransformer(nn.Module):
    """Thin wrapper around ``nn.TransformerEncoder`` with optional disabled depth."""

    def __init__(
        self,
        latent_dim: int,
        *,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.depth = int(depth)
        if self.depth <= 0:
            self.encoder = nn.Identity()
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=int(latent_dim),
                nhead=int(num_heads),
                dim_feedforward=int(float(mlp_ratio) * int(latent_dim)),
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=self.depth)

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.depth <= 0:
            return tokens
        src_key_padding_mask = None
        if token_mask is not None:
            src_key_padding_mask = ~token_mask.bool()
        return self.encoder(tokens, src_key_padding_mask=src_key_padding_mask)


class PatchDecoder(nn.Module):
    """MLP patch decoder that predicts local xyz plus radar feature channels."""

    def __init__(self, latent_dim: int, group_size: int, output_channels: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.group_size = int(group_size)
        self.output_channels = int(output_channels)
        self.net = nn.Sequential(
            nn.Linear(int(latent_dim), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.group_size * self.output_channels),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz, groups, latent_dim = tokens.shape
        out = self.net(tokens.reshape(bsz * groups, latent_dim))
        return out.view(bsz, groups, self.group_size, self.output_channels)


def masked_token_mean(tokens: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
    if token_mask is None:
        return tokens.mean(dim=1)
    weights = token_mask.to(dtype=tokens.dtype).unsqueeze(-1)
    denom = torch.clamp(weights.sum(dim=1), min=1.0)
    return (tokens * weights).sum(dim=1) / denom


def masked_chamfer_l1(pred: torch.Tensor, target: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    """Local patch Chamfer L1 over valid groups."""
    valid = token_mask.reshape(-1)
    pred_flat = pred[..., :3].reshape(-1, pred.shape[-2], 3)
    target_flat = target[..., :3].reshape(-1, target.shape[-2], 3)
    pred_valid = pred_flat[valid]
    target_valid = target_flat[valid]
    if pred_valid.numel() == 0:
        return pred.sum() * 0.0
    dist = torch.cdist(pred_valid.float(), target_valid.float(), p=2)
    return 0.5 * (dist.min(dim=2).values.mean() + dist.min(dim=1).values.mean())


def masked_feature_mse(pred: torch.Tensor, target: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    """Direct radar-feature reconstruction loss for non-xyz channels."""
    if pred.shape[-1] <= 3:
        return pred.sum() * 0.0
    diff = (pred[..., 3:] - target[..., 3:]) ** 2
    per_group = diff.mean(dim=(2, 3))
    valid = token_mask.to(dtype=per_group.dtype)
    return (per_group * valid).sum() / torch.clamp(valid.sum(), min=1.0)


def make_random_mask(token_mask: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """Sample masked patch positions while respecting invalid padded groups."""
    bsz, groups = token_mask.shape
    ratio = max(0.0, min(1.0, float(mask_ratio)))
    valid_counts = token_mask.sum(dim=1)
    n_mask = torch.round(valid_counts.to(dtype=torch.float32) * ratio).to(dtype=torch.long)
    n_mask = torch.where(valid_counts > 0, n_mask.clamp_min(1), torch.zeros_like(n_mask))
    n_mask = torch.minimum(n_mask, valid_counts)
    out = torch.zeros_like(token_mask, dtype=torch.bool)
    if groups <= 0:
        return out

    scores = torch.rand(bsz, groups, device=token_mask.device)
    scores = scores.masked_fill(~token_mask.bool(), float("inf"))
    order = torch.topk(scores, k=groups, dim=1, largest=False, sorted=True).indices
    selected = torch.arange(groups, device=token_mask.device).unsqueeze(0) < n_mask.unsqueeze(1)
    out.scatter_(dim=1, index=order, src=selected)
    return out
