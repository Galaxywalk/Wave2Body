# -*- coding: utf-8 -*-
"""Production self-supervised mmWave sensor tokenizer model."""

from __future__ import annotations

import inspect
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .grouping import PatchBatch, group_point_cloud
from .modules import (
    CenterPositionalEmbedding,
    PatchDecoder,
    PointPatchEncoder,
    TokenTransformer,
    make_random_mask,
    masked_chamfer_l1,
    masked_feature_mse,
)


PRODUCTION_MMWAVE_TOKENIZER_METHOD = "mae"
PRODUCTION_MMWAVE_PATCH_POLICY = "spatial_fps_knn"


def _loss_dict(total: torch.Tensor, **items: torch.Tensor) -> dict[str, torch.Tensor]:
    out = {"total_loss": total}
    out.update(items)
    return out


def _zero_loss_like(value: torch.Tensor) -> torch.Tensor:
    return value.sum() * 0.0


def _expanded_loss_weight(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    while mask.dim() < like.dim():
        mask = mask.unsqueeze(-1)
    return mask.expand_as(like).to(dtype=torch.float32)


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = _expanded_loss_weight(mask, pred)
    diff = pred.float() - target.float()
    return (diff.square() * weight).sum() / weight.sum().clamp_min(1.0)


def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = _expanded_loss_weight(mask, pred)
    loss = F.smooth_l1_loss(pred.float(), target.float(), reduction="none")
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


class PointMaeTokenizer(nn.Module):
    """Point-MAE-style continuous masked patch reconstruction tokenizer."""

    method = PRODUCTION_MMWAVE_TOKENIZER_METHOD

    def __init__(
        self,
        *,
        input_channels: int = 4,
        num_groups: int = 96,
        group_size: int = 32,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        patch_policy: str = "spatial_fps_knn",
        mask_ratio: float = 0.6,
        transformer_depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        chamfer_weight: float = 1.0,
        feature_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.num_groups = int(num_groups)
        self.group_size = int(group_size)
        self.patch_policy = str(patch_policy)
        self.mask_ratio = float(mask_ratio)
        self.chamfer_weight = float(chamfer_weight)
        self.feature_weight = float(feature_weight)

        self.patch_encoder = PointPatchEncoder(self.input_channels, latent_dim, hidden_dim, dropout)
        self.pos_embed = CenterPositionalEmbedding(latent_dim, hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.context = TokenTransformer(latent_dim, depth=transformer_depth, num_heads=num_heads, dropout=dropout)
        self.decoder = PatchDecoder(latent_dim, self.group_size, self.input_channels, hidden_dim * 2)

    def _group(self, points: torch.Tensor, target_poses_absolute: torch.Tensor | None) -> PatchBatch:
        return group_point_cloud(
            points,
            num_groups=self.num_groups,
            group_size=self.group_size,
            policy=self.patch_policy,
            target_poses_absolute=target_poses_absolute,
        )

    def _encode_patch_batch(self, patch_batch: PatchBatch) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.patch_encoder(patch_batch.patches)
        tokens = self.context(encoded + self.pos_embed(patch_batch.centers), patch_batch.group_mask)
        return encoded, tokens

    def encode_tokens(self, points: torch.Tensor, target_poses_absolute: torch.Tensor | None = None) -> dict[str, Any]:
        """Deterministically encode point patches without random masking."""
        patch_batch = self._group(points, target_poses_absolute)
        encoded, tokens = self._encode_patch_batch(patch_batch)
        return {
            "token_ids": None,
            "token_embeddings": tokens,
            "continuous_embeddings": encoded,
            "centers": patch_batch.centers,
            "valid_mask": patch_batch.group_mask,
            "point_mask": patch_batch.point_mask,
            "support_counts": patch_batch.point_mask.sum(dim=-1).to(dtype=tokens.dtype),
            "patches": patch_batch.patches,
            "raw_patches": patch_batch.raw_patches,
            "patch_policy": patch_batch.policy,
        }

    def forward(self, points: torch.Tensor, target_poses_absolute: torch.Tensor | None = None) -> dict[str, Any]:
        patch_batch = self._group(points, target_poses_absolute)
        encoded = self.patch_encoder(patch_batch.patches)
        mask = make_random_mask(patch_batch.group_mask, self.mask_ratio)
        mask_token = self.mask_token.expand(encoded.shape[0], encoded.shape[1], -1)
        tokens = torch.where(mask.unsqueeze(-1), mask_token, encoded)
        tokens = self.context(tokens + self.pos_embed(patch_batch.centers), patch_batch.group_mask)
        recon = self.decoder(tokens)
        loss_mask = mask & patch_batch.group_mask
        chamfer_loss = masked_chamfer_l1(recon, patch_batch.patches, loss_mask)
        feature_loss = masked_feature_mse(recon, patch_batch.patches, loss_mask)
        total = self.chamfer_weight * chamfer_loss + self.feature_weight * feature_loss
        return {
            "losses": _loss_dict(total, chamfer_loss=chamfer_loss, feature_loss=feature_loss),
            "token_ids": None,
            "token_embeddings": tokens,
            "continuous_embeddings": encoded,
            "centers": patch_batch.centers,
            "valid_mask": patch_batch.group_mask,
            "point_mask": patch_batch.point_mask,
            "support_counts": patch_batch.point_mask.sum(dim=-1).to(dtype=tokens.dtype),
            "masked_positions": mask,
            "reconstruction": recon,
            "patches": patch_batch.patches,
            "raw_patches": patch_batch.raw_patches,
            "stats": {"mask_ratio": mask.float().mean()},
            "patch_policy": patch_batch.policy,
        }


def _constructor_kwargs(cls: type[nn.Module], cfg: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        name
        for name, param in inspect.signature(cls.__init__).parameters.items()
        if name != "self" and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    }
    return {key: value for key, value in cfg.items() if key in allowed}


def create_mmwave_tokenizer(config: dict[str, Any]) -> PointMaeTokenizer:
    """Instantiate the production mmWave tokenizer.

    Production Wave2Body only supports spatial Point-MAE sensor tokens. Historical
    alternatives live under ``wave2body.models.mmwave_tokenizer.baselines``.
    """
    cfg = dict(config)
    method = str(cfg.pop("method", PRODUCTION_MMWAVE_TOKENIZER_METHOD)).strip().lower()
    if method != PRODUCTION_MMWAVE_TOKENIZER_METHOD:
        raise ValueError(
            "Production mmWave tokenizer only supports method='mae'. "
            "Use wave2body.models.mmwave_tokenizer.baselines.create_mmwave_baseline_tokenizer "
            f"for historical method={method!r}."
        )
    patch_policy = str(cfg.get("patch_policy", PRODUCTION_MMWAVE_PATCH_POLICY)).strip().lower()
    if patch_policy != PRODUCTION_MMWAVE_PATCH_POLICY:
        raise ValueError(
            "Production mmWave tokenizer only supports patch_policy='spatial_fps_knn'. "
            "Body-part-assisted grouping is a historical baseline path."
        )
    return PointMaeTokenizer(**_constructor_kwargs(PointMaeTokenizer, cfg))
