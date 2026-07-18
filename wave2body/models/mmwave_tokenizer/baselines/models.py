# -*- coding: utf-8 -*-
"""Historical mmWave tokenizer baseline implementations."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..grouping import PatchBatch, group_point_cloud
from ..models import (
    PointMaeTokenizer,
    _constructor_kwargs,
    _loss_dict,
    _masked_mse,
    _masked_smooth_l1,
    _zero_loss_like,
)
from ..modules import (
    CenterPositionalEmbedding,
    PatchDecoder,
    PointPatchEncoder,
    TokenTransformer,
    make_random_mask,
    masked_chamfer_l1,
    masked_feature_mse,
    masked_token_mean,
)
from ..quantizer import EMAVectorQuantizer


class PartQueryDecoder(nn.Module):
    """Learned body-part queries that cross-attend to spatial point tokens."""

    def __init__(
        self,
        *,
        num_queries: int,
        latent_dim: int,
        depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_queries = int(num_queries)
        self.latent_dim = int(latent_dim)
        self.depth = int(depth)
        self.query_embed = nn.Parameter(torch.empty(1, self.num_queries, self.latent_dim))
        nn.init.normal_(self.query_embed, std=0.02)
        if self.depth <= 0:
            self.decoder = nn.Identity()
        else:
            layer = nn.TransformerDecoderLayer(
                d_model=self.latent_dim,
                nhead=int(num_heads),
                dim_feedforward=self.latent_dim * 2,
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(layer, num_layers=self.depth)

    def forward(self, memory: torch.Tensor, memory_mask: torch.Tensor | None = None) -> torch.Tensor:
        queries = self.query_embed.expand(memory.shape[0], -1, -1)
        if self.depth <= 0:
            return queries
        key_padding_mask = None
        if memory_mask is not None:
            key_padding_mask = ~memory_mask.bool()
        return self.decoder(queries, memory, memory_key_padding_mask=key_padding_mask)


class SpatialToPartQueryBackbone(nn.Module):
    """Spatial point-patch encoder followed by learned part queries."""

    def __init__(
        self,
        *,
        input_channels: int,
        num_parts: int,
        group_size: int,
        latent_dim: int,
        hidden_dim: int,
        num_context_groups: int,
        transformer_depth: int,
        query_depth: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.num_parts = int(num_parts)
        self.group_size = int(group_size)
        self.num_context_groups = int(num_context_groups)
        self.patch_encoder = PointPatchEncoder(self.input_channels, int(latent_dim), int(hidden_dim), float(dropout))
        self.pos_embed = CenterPositionalEmbedding(int(latent_dim), int(hidden_dim))
        self.context = TokenTransformer(
            int(latent_dim),
            depth=int(transformer_depth),
            num_heads=int(num_heads),
            dropout=float(dropout),
        )
        self.part_queries = PartQueryDecoder(
            num_queries=self.num_parts,
            latent_dim=int(latent_dim),
            depth=int(query_depth),
            num_heads=int(num_heads),
            dropout=float(dropout),
        )
        self.center_head = nn.Sequential(
            nn.LayerNorm(int(latent_dim)),
            nn.Linear(int(latent_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 3),
        )

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor | PatchBatch]:
        spatial_batch = group_point_cloud(
            points,
            num_groups=self.num_context_groups,
            group_size=self.group_size,
            policy="spatial_fps_knn",
            target_poses_absolute=None,
        )
        spatial_tokens = self.patch_encoder(spatial_batch.patches)
        spatial_tokens = self.context(spatial_tokens + self.pos_embed(spatial_batch.centers), spatial_batch.group_mask)
        part_tokens = self.part_queries(spatial_tokens, spatial_batch.group_mask)
        pred_centers = self.center_head(part_tokens)
        return {
            "spatial_batch": spatial_batch,
            "spatial_tokens": spatial_tokens,
            "part_tokens": part_tokens,
            "pred_centers": pred_centers,
        }


class PointBertDvaeTokenizer(nn.Module):
    """Point-BERT-style local patch dVAE/VQ tokenizer for radar points."""

    method = "dvae"

    def __init__(
        self,
        *,
        input_channels: int = 4,
        num_groups: int = 22,
        group_size: int = 32,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        codebook_size: int = 512,
        patch_policy: str = "spatial_fps_knn",
        transformer_depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        chamfer_weight: float = 1.0,
        feature_weight: float = 0.1,
        commitment_weight: float = 0.25,
        usage_weight: float = 0.01,
        ema_decay: float = 0.99,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.num_groups = int(num_groups)
        self.group_size = int(group_size)
        self.latent_dim = int(latent_dim)
        self.patch_policy = str(patch_policy)
        self.chamfer_weight = float(chamfer_weight)
        self.feature_weight = float(feature_weight)
        self.commitment_weight = float(commitment_weight)
        self.usage_weight = float(usage_weight)

        self.patch_encoder = PointPatchEncoder(self.input_channels, self.latent_dim, hidden_dim, dropout)
        self.pos_embed = CenterPositionalEmbedding(self.latent_dim, hidden_dim)
        self.context = TokenTransformer(
            self.latent_dim,
            depth=transformer_depth,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.quantizer = EMAVectorQuantizer(codebook_size, self.latent_dim, ema_decay=ema_decay)
        self.decoder = PatchDecoder(self.latent_dim, self.group_size, self.input_channels, hidden_dim * 2)

    def _group(self, points: torch.Tensor, target_poses_absolute: torch.Tensor | None) -> PatchBatch:
        return group_point_cloud(
            points,
            num_groups=self.num_groups,
            group_size=self.group_size,
            policy=self.patch_policy,
            target_poses_absolute=target_poses_absolute,
        )

    def forward(self, points: torch.Tensor, target_poses_absolute: torch.Tensor | None = None) -> dict[str, Any]:
        patch_batch = self._group(points, target_poses_absolute)
        z_e = self.patch_encoder(patch_batch.patches)
        z_e = self.context(z_e + self.pos_embed(patch_batch.centers), patch_batch.group_mask)
        token_ids, z_q, commit_loss, q_stats = self.quantizer(z_e, patch_batch.group_mask, train=self.training)
        recon = self.decoder(z_q)
        chamfer_loss = masked_chamfer_l1(recon, patch_batch.patches, patch_batch.group_mask)
        feature_loss = masked_feature_mse(recon, patch_batch.patches, patch_batch.group_mask)
        usage_loss = self.quantizer.usage_kl_to_uniform(z_e, patch_batch.group_mask)
        total = (
            self.chamfer_weight * chamfer_loss
            + self.feature_weight * feature_loss
            + self.commitment_weight * commit_loss
            + self.usage_weight * usage_loss
        )
        return {
            "losses": _loss_dict(
                total,
                chamfer_loss=chamfer_loss,
                feature_loss=feature_loss,
                commitment_loss=commit_loss,
                usage_loss=usage_loss,
            ),
            "token_ids": token_ids,
            "token_embeddings": z_q,
            "continuous_embeddings": z_e,
            "centers": patch_batch.centers,
            "valid_mask": patch_batch.group_mask,
            "reconstruction": recon,
            "patches": patch_batch.patches,
            "raw_patches": patch_batch.raw_patches,
            "stats": q_stats,
            "patch_policy": patch_batch.policy,
        }


class _PosEncoder(nn.Module):
    def __init__(self, input_channels: int, latent_dim: int, hidden_dim: int, depth: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.patch_encoder = PointPatchEncoder(input_channels, latent_dim, hidden_dim, dropout)
        self.pos_embed = CenterPositionalEmbedding(latent_dim, hidden_dim)
        self.context = TokenTransformer(latent_dim, depth=depth, num_heads=heads, dropout=dropout)

    def forward(self, patches: torch.Tensor, centers: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_encoder(patches) + self.pos_embed(centers)
        return self.context(tokens, token_mask)


class PosBertStyleTokenizer(nn.Module):
    """POS-BERT-inspired one-stage teacher-student masked patch baseline."""

    method = "posbert"

    def __init__(
        self,
        *,
        input_channels: int = 4,
        num_groups: int = 22,
        group_size: int = 32,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        patch_policy: str = "spatial_fps_knn",
        mask_ratio: float = 0.6,
        transformer_depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        teacher_momentum: float = 0.996,
        patch_feature_weight: float = 1.0,
        global_feature_weight: float = 1.0,
        reconstruction_weight: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.num_groups = int(num_groups)
        self.group_size = int(group_size)
        self.patch_policy = str(patch_policy)
        self.mask_ratio = float(mask_ratio)
        self.teacher_momentum = float(teacher_momentum)
        self.patch_feature_weight = float(patch_feature_weight)
        self.global_feature_weight = float(global_feature_weight)
        self.reconstruction_weight = float(reconstruction_weight)

        self.student = _PosEncoder(input_channels, latent_dim, hidden_dim, transformer_depth, num_heads, dropout)
        self.teacher = copy.deepcopy(self.student)
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.student_context = TokenTransformer(latent_dim, depth=transformer_depth, num_heads=num_heads, dropout=dropout)
        self.decoder = PatchDecoder(latent_dim, self.group_size, self.input_channels, hidden_dim * 2)

    @torch.no_grad()
    def update_teacher(self, momentum: float | None = None) -> None:
        m = self.teacher_momentum if momentum is None else float(momentum)
        for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters()):
            teacher_param.data.mul_(m).add_(student_param.data, alpha=1.0 - m)

    def forward(self, points: torch.Tensor, target_poses_absolute: torch.Tensor | None = None) -> dict[str, Any]:
        patch_batch = group_point_cloud(
            points,
            num_groups=self.num_groups,
            group_size=self.group_size,
            policy=self.patch_policy,
            target_poses_absolute=target_poses_absolute,
        )
        student_base = self.student.patch_encoder(patch_batch.patches)
        mask = make_random_mask(patch_batch.group_mask, self.mask_ratio)
        mask_token = self.mask_token.expand(student_base.shape[0], student_base.shape[1], -1)
        student_tokens = torch.where(mask.unsqueeze(-1), mask_token, student_base)
        student_tokens = student_tokens + self.student.pos_embed(patch_batch.centers)
        student_tokens = self.student_context(student_tokens, patch_batch.group_mask)

        with torch.no_grad():
            teacher_tokens = self.teacher(patch_batch.patches, patch_batch.centers, patch_batch.group_mask)

        loss_mask = mask & patch_batch.group_mask
        patch_feature_loss = _masked_mse(student_tokens, teacher_tokens.detach(), loss_mask)

        student_global = F.normalize(masked_token_mean(student_tokens, patch_batch.group_mask), dim=-1)
        teacher_global = F.normalize(masked_token_mean(teacher_tokens.detach(), patch_batch.group_mask), dim=-1)
        global_feature_loss = F.mse_loss(student_global, teacher_global)

        recon = self.decoder(student_tokens)
        recon_loss = masked_chamfer_l1(recon, patch_batch.patches, loss_mask)
        total = (
            self.patch_feature_weight * patch_feature_loss
            + self.global_feature_weight * global_feature_loss
            + self.reconstruction_weight * recon_loss
        )
        return {
            "losses": _loss_dict(
                total,
                patch_feature_loss=patch_feature_loss,
                global_feature_loss=global_feature_loss,
                reconstruction_loss=recon_loss,
            ),
            "token_ids": None,
            "token_embeddings": student_tokens,
            "teacher_embeddings": teacher_tokens,
            "centers": patch_batch.centers,
            "valid_mask": patch_batch.group_mask,
            "point_mask": patch_batch.point_mask,
            "support_counts": patch_batch.point_mask.sum(dim=-1).to(dtype=student_tokens.dtype),
            "masked_positions": mask,
            "reconstruction": recon,
            "patches": patch_batch.patches,
            "raw_patches": patch_batch.raw_patches,
            "stats": {"mask_ratio": mask.float().mean()},
            "patch_policy": patch_batch.policy,
        }


def _load_frozen_mae_teacher(checkpoint_path: str | Path) -> PointMaeTokenizer:
    ckpt = torch.load(Path(checkpoint_path).expanduser(), map_location="cpu", weights_only=False)
    cfg = {}
    if isinstance(ckpt.get("config"), dict):
        cfg = dict(ckpt["config"].get("mmwave_tokenizer", {}))
    cfg["method"] = "mae"
    teacher = PointMaeTokenizer(**_constructor_kwargs(PointMaeTokenizer, cfg))
    state = ckpt.get("model_state_dict")
    if not isinstance(state, dict):
        raise KeyError(f"Teacher checkpoint lacks model_state_dict: {checkpoint_path}")
    teacher.load_state_dict(state, strict=False)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


class MaeDistillPartTokenizer(nn.Module):
    """Deployable spatial student distilled from an oracle body-part MAE teacher."""

    method = "mae_distill"

    def __init__(
        self,
        *,
        input_channels: int = 4,
        num_groups: int = 22,
        group_size: int = 32,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        patch_policy: str = "spatial_fps_knn",
        num_context_groups: int | None = None,
        transformer_depth: int = 2,
        query_depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        teacher_checkpoint: str | None = None,
        distill_weight: float = 1.0,
        chamfer_weight: float = 1.0,
        feature_weight: float = 0.1,
        center_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.num_groups = int(num_groups)
        self.group_size = int(group_size)
        self.latent_dim = int(latent_dim)
        self.patch_policy = str(patch_policy)
        self.teacher_checkpoint = str(teacher_checkpoint) if teacher_checkpoint else None
        self.distill_weight = float(distill_weight)
        self.chamfer_weight = float(chamfer_weight)
        self.feature_weight = float(feature_weight)
        self.center_weight = float(center_weight)
        self.backbone = SpatialToPartQueryBackbone(
            input_channels=self.input_channels,
            num_parts=self.num_groups,
            group_size=self.group_size,
            latent_dim=self.latent_dim,
            hidden_dim=int(hidden_dim),
            num_context_groups=int(num_context_groups or self.num_groups),
            transformer_depth=int(transformer_depth),
            query_depth=int(query_depth),
            num_heads=int(num_heads),
            dropout=float(dropout),
        )
        self.decoder = PatchDecoder(self.latent_dim, self.group_size, self.input_channels, int(hidden_dim) * 2)
        self.teacher = _load_frozen_mae_teacher(self.teacher_checkpoint) if self.teacher_checkpoint else None

    def train(self, mode: bool = True):
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        return self

    def _supervised_losses(
        self,
        points: torch.Tensor,
        part_tokens: torch.Tensor,
        pred_centers: torch.Tensor,
        target_poses_absolute: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor], PatchBatch | None, torch.Tensor]:
        recon = self.decoder(part_tokens)
        if target_poses_absolute is None:
            zero = _zero_loss_like(part_tokens)
            return (
                {
                    "distill_loss": zero,
                    "chamfer_loss": zero,
                    "feature_loss": zero,
                    "center_loss": zero,
                },
                None,
                recon,
            )

        target_batch = group_point_cloud(
            points,
            num_groups=self.num_groups,
            group_size=self.group_size,
            policy="weak_body_part_knn",
            target_poses_absolute=target_poses_absolute,
        )
        mask = target_batch.group_mask
        chamfer_loss = masked_chamfer_l1(recon, target_batch.patches, mask)
        feature_loss = masked_feature_mse(recon, target_batch.patches, mask)
        center_loss = _masked_smooth_l1(pred_centers, target_batch.centers, mask)

        if self.teacher is None:
            distill_loss = _zero_loss_like(part_tokens)
        else:
            self.teacher.eval()
            with torch.no_grad():
                teacher_out = self.teacher.encode_tokens(points, target_poses_absolute=target_poses_absolute)
                teacher_tokens = teacher_out["token_embeddings"].detach()
                teacher_mask = teacher_out["valid_mask"].bool()
            distill_mask = mask & teacher_mask
            student_norm = F.normalize(part_tokens, dim=-1)
            teacher_norm = F.normalize(teacher_tokens, dim=-1)
            distill_loss = _masked_mse(student_norm, teacher_norm, distill_mask)

        return (
            {
                "distill_loss": distill_loss,
                "chamfer_loss": chamfer_loss,
                "feature_loss": feature_loss,
                "center_loss": center_loss,
            },
            target_batch,
            recon,
        )

    def forward(self, points: torch.Tensor, target_poses_absolute: torch.Tensor | None = None) -> dict[str, Any]:
        outputs = self.backbone(points)
        part_tokens = outputs["part_tokens"]
        pred_centers = outputs["pred_centers"]
        losses, target_batch, recon = self._supervised_losses(points, part_tokens, pred_centers, target_poses_absolute)
        total = (
            self.distill_weight * losses["distill_loss"]
            + self.chamfer_weight * losses["chamfer_loss"]
            + self.feature_weight * losses["feature_loss"]
            + self.center_weight * losses["center_loss"]
        )
        valid_mask = (
            target_batch.group_mask
            if target_batch is not None
            else torch.ones(part_tokens.shape[:2], device=part_tokens.device, dtype=torch.bool)
        )
        return {
            "losses": _loss_dict(total, **losses),
            "token_ids": None,
            "token_embeddings": part_tokens,
            "continuous_embeddings": outputs["spatial_tokens"],
            "centers": pred_centers,
            "valid_mask": valid_mask,
            "spatial_centers": outputs["spatial_batch"].centers,
            "spatial_valid_mask": outputs["spatial_batch"].group_mask,
            "reconstruction": recon,
            "patches": target_batch.patches if target_batch is not None else None,
            "stats": {"has_teacher": float(self.teacher is not None)},
            "patch_policy": "spatial_to_part_query_distill",
        }


class PartQueryMaeTokenizer(nn.Module):
    """Deployable part-query MAE trained with GT centers only as auxiliary supervision."""

    method = "part_query_mae"

    def __init__(
        self,
        *,
        input_channels: int = 4,
        num_groups: int = 22,
        group_size: int = 32,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        patch_policy: str = "spatial_fps_knn",
        num_context_groups: int | None = None,
        transformer_depth: int = 2,
        query_depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        chamfer_weight: float = 1.0,
        feature_weight: float = 0.1,
        center_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.num_groups = int(num_groups)
        self.group_size = int(group_size)
        self.latent_dim = int(latent_dim)
        self.patch_policy = str(patch_policy)
        self.chamfer_weight = float(chamfer_weight)
        self.feature_weight = float(feature_weight)
        self.center_weight = float(center_weight)
        self.backbone = SpatialToPartQueryBackbone(
            input_channels=self.input_channels,
            num_parts=self.num_groups,
            group_size=self.group_size,
            latent_dim=self.latent_dim,
            hidden_dim=int(hidden_dim),
            num_context_groups=int(num_context_groups or self.num_groups),
            transformer_depth=int(transformer_depth),
            query_depth=int(query_depth),
            num_heads=int(num_heads),
            dropout=float(dropout),
        )
        self.decoder = PatchDecoder(self.latent_dim, self.group_size, self.input_channels, int(hidden_dim) * 2)

    def forward(self, points: torch.Tensor, target_poses_absolute: torch.Tensor | None = None) -> dict[str, Any]:
        outputs = self.backbone(points)
        part_tokens = outputs["part_tokens"]
        pred_centers = outputs["pred_centers"]
        recon = self.decoder(part_tokens)
        if target_poses_absolute is None:
            mask = torch.ones(part_tokens.shape[:2], device=part_tokens.device, dtype=torch.bool)
            chamfer_loss = _zero_loss_like(part_tokens)
            feature_loss = _zero_loss_like(part_tokens)
            center_loss = _zero_loss_like(part_tokens)
            target_batch = None
        else:
            target_batch = group_point_cloud(
                points,
                num_groups=self.num_groups,
                group_size=self.group_size,
                policy="weak_body_part_knn",
                target_poses_absolute=target_poses_absolute,
            )
            mask = target_batch.group_mask
            chamfer_loss = masked_chamfer_l1(recon, target_batch.patches, mask)
            feature_loss = masked_feature_mse(recon, target_batch.patches, mask)
            center_loss = _masked_smooth_l1(pred_centers, target_batch.centers, mask)
        total = (
            self.chamfer_weight * chamfer_loss
            + self.feature_weight * feature_loss
            + self.center_weight * center_loss
        )
        return {
            "losses": _loss_dict(
                total,
                chamfer_loss=chamfer_loss,
                feature_loss=feature_loss,
                center_loss=center_loss,
            ),
            "token_ids": None,
            "token_embeddings": part_tokens,
            "continuous_embeddings": outputs["spatial_tokens"],
            "centers": pred_centers,
            "valid_mask": mask,
            "spatial_centers": outputs["spatial_batch"].centers,
            "spatial_valid_mask": outputs["spatial_batch"].group_mask,
            "reconstruction": recon,
            "patches": target_batch.patches if target_batch is not None else None,
            "stats": {},
            "patch_policy": "spatial_to_part_query",
        }


class RawPointMlpSensorEncoder(nn.Module):
    """Supervised raw-point sensor encoder used for no-tokenizer ablations."""

    method = "raw_point_mlp"

    def __init__(
        self,
        *,
        input_channels: int = 4,
        num_groups: int = 96,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        point_hidden_dim: int | None = None,
        num_point_layers: int = 2,
        num_self_attn_layers: int = 0,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.num_groups = int(num_groups)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.point_hidden_dim = int(point_hidden_dim or hidden_dim)
        self.num_point_layers = max(1, int(num_point_layers))
        self.num_self_attn_layers = max(0, int(num_self_attn_layers))
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        if self.latent_dim % self.num_heads != 0:
            raise ValueError(
                f"raw_point_mlp latent_dim={self.latent_dim} must be divisible by num_heads={self.num_heads}"
            )

        layers: list[nn.Module] = []
        dim = self.input_channels
        for _ in range(self.num_point_layers - 1):
            layers.extend([nn.Linear(dim, self.point_hidden_dim), nn.GELU(), nn.Dropout(self.dropout)])
            dim = self.point_hidden_dim
        layers.append(nn.Linear(dim, self.latent_dim))
        self.point_mlp = nn.Sequential(*layers)
        self.query_embed = nn.Parameter(torch.empty(1, self.num_groups, self.latent_dim))
        nn.init.trunc_normal_(self.query_embed, std=0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=max(self.latent_dim * 2, self.hidden_dim),
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = (
            nn.TransformerEncoder(encoder_layer, num_layers=self.num_self_attn_layers)
            if self.num_self_attn_layers > 0
            else nn.Identity()
        )
        self.norm = nn.LayerNorm(self.latent_dim)

    def _valid_point_mask(self, points: torch.Tensor) -> torch.Tensor:
        return torch.any(torch.abs(points[..., : self.input_channels]) > 0, dim=-1)

    def encode_tokens(self, points: torch.Tensor, target_poses_absolute: torch.Tensor | None = None) -> dict[str, Any]:
        del target_poses_absolute
        if points.dim() != 3 or int(points.shape[-1]) < self.input_channels:
            raise ValueError(
                f"Expected raw mmWave points [B,N,>={self.input_channels}], got {tuple(points.shape)}"
            )
        batch_size, n_points, _ = points.shape
        point_values = points[..., : self.input_channels].float()
        valid = self._valid_point_mask(points)
        valid_any = valid.any(dim=1, keepdim=True)

        fallback = torch.zeros_like(valid)
        if n_points > 0:
            fallback[:, 0] = True
        safe_valid = torch.where(valid_any, valid, fallback)
        point_features = self.point_mlp(point_values)
        point_features = point_features.masked_fill(~safe_valid.unsqueeze(-1), 0.0)

        queries = self.query_embed.expand(batch_size, -1, -1)
        tokens, weights = self.cross_attn(
            queries,
            point_features,
            point_features,
            key_padding_mask=~safe_valid,
            need_weights=True,
            average_attn_weights=True,
        )
        tokens = self.context(tokens)
        tokens = self.norm(tokens)

        token_valid = valid_any.expand(-1, self.num_groups)
        tokens = tokens.masked_fill(~token_valid.unsqueeze(-1), 0.0)

        weights = weights * safe_valid.unsqueeze(1).to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        centers = torch.bmm(weights, points[..., :3].float())
        centers = centers.masked_fill(~token_valid.unsqueeze(-1), 0.0)

        return {
            "token_ids": None,
            "token_embeddings": tokens,
            "continuous_embeddings": point_features,
            "centers": centers,
            "valid_mask": token_valid,
            "point_mask": valid,
            "support_counts": valid.sum(dim=1, keepdim=True).expand(-1, self.num_groups).to(dtype=tokens.dtype),
            "stats": {},
            "patch_policy": "raw_point_mlp",
        }

    def forward(self, points: torch.Tensor, target_poses_absolute: torch.Tensor | None = None) -> dict[str, Any]:
        encoded = self.encode_tokens(points, target_poses_absolute=target_poses_absolute)
        zero = _zero_loss_like(encoded["token_embeddings"])
        encoded["losses"] = _loss_dict(zero)
        return encoded


class RawPointNetSensorEncoder(nn.Module):
    """PointNet-style raw-point encoder for no-tokenizer ablations."""

    method = "raw_point_pointnet"

    def __init__(
        self,
        *,
        input_channels: int = 4,
        num_groups: int = 96,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        point_hidden_dim: int | None = None,
        num_point_layers: int = 3,
        num_token_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.num_groups = int(num_groups)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.point_hidden_dim = int(point_hidden_dim or hidden_dim)
        self.num_point_layers = max(1, int(num_point_layers))
        self.num_token_layers = max(1, int(num_token_layers))
        self.dropout = float(dropout)

        point_layers: list[nn.Module] = []
        dim = self.input_channels
        for _ in range(self.num_point_layers):
            point_layers.extend([nn.Linear(dim, self.point_hidden_dim), nn.GELU(), nn.Dropout(self.dropout)])
            dim = self.point_hidden_dim
        self.point_mlp = nn.Sequential(*point_layers)

        self.global_proj = nn.Sequential(
            nn.LayerNorm(self.point_hidden_dim * 2),
            nn.Linear(self.point_hidden_dim * 2, self.latent_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.query_embed = nn.Parameter(torch.empty(1, self.num_groups, self.latent_dim))
        nn.init.trunc_normal_(self.query_embed, std=0.02)

        token_layers: list[nn.Module] = []
        token_dim = self.latent_dim
        for _ in range(self.num_token_layers - 1):
            token_layers.extend([nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim), nn.GELU(), nn.Dropout(self.dropout)])
        token_layers.extend([nn.LayerNorm(token_dim), nn.Linear(token_dim, self.latent_dim)])
        self.token_mlp = nn.Sequential(*token_layers)
        self.center_head = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 3),
        )

    def _valid_point_mask(self, points: torch.Tensor) -> torch.Tensor:
        return torch.any(torch.abs(points[..., : self.input_channels]) > 0, dim=-1)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).to(dtype=values.dtype)
        return (values * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _masked_max(values: torch.Tensor, mask: torch.Tensor, valid_any: torch.Tensor) -> torch.Tensor:
        masked = values.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        pooled = masked.max(dim=1).values
        return torch.where(valid_any, pooled, torch.zeros_like(pooled))

    def encode_tokens(self, points: torch.Tensor, target_poses_absolute: torch.Tensor | None = None) -> dict[str, Any]:
        del target_poses_absolute
        if points.dim() != 3 or int(points.shape[-1]) < self.input_channels:
            raise ValueError(
                f"Expected raw mmWave points [B,N,>={self.input_channels}], got {tuple(points.shape)}"
            )
        batch_size, n_points, _ = points.shape
        point_values = points[..., : self.input_channels].float()
        valid = self._valid_point_mask(points)
        valid_any = valid.any(dim=1, keepdim=True)

        fallback = torch.zeros_like(valid)
        if n_points > 0:
            fallback[:, 0] = True
        safe_valid = torch.where(valid_any, valid, fallback)

        point_features = self.point_mlp(point_values).masked_fill(~safe_valid.unsqueeze(-1), 0.0)
        feature_mean = self._masked_mean(point_features, safe_valid)
        feature_max = self._masked_max(point_features, safe_valid, valid_any)
        global_feature = self.global_proj(torch.cat([feature_mean, feature_max], dim=-1))

        tokens = self.query_embed.expand(batch_size, -1, -1) + global_feature.unsqueeze(1)
        tokens = self.token_mlp(tokens)
        token_valid = valid_any.expand(-1, self.num_groups)
        tokens = tokens.masked_fill(~token_valid.unsqueeze(-1), 0.0)

        xyz_mean = self._masked_mean(points[..., :3].float(), safe_valid)
        centers = xyz_mean.unsqueeze(1) + self.center_head(tokens)
        centers = centers.masked_fill(~token_valid.unsqueeze(-1), 0.0)

        return {
            "token_ids": None,
            "token_embeddings": tokens,
            "continuous_embeddings": point_features,
            "centers": centers,
            "valid_mask": token_valid,
            "point_mask": valid,
            "support_counts": valid.sum(dim=1, keepdim=True).expand(-1, self.num_groups).to(dtype=tokens.dtype),
            "stats": {},
            "patch_policy": "raw_point_pointnet",
        }

    def forward(self, points: torch.Tensor, target_poses_absolute: torch.Tensor | None = None) -> dict[str, Any]:
        encoded = self.encode_tokens(points, target_poses_absolute=target_poses_absolute)
        zero = _zero_loss_like(encoded["token_embeddings"])
        encoded["losses"] = _loss_dict(zero)
        return encoded


def create_mmwave_baseline_tokenizer(config: dict[str, Any]) -> nn.Module:
    """Instantiate a historical mmWave tokenizer baseline from config."""
    cfg = dict(config)
    method = str(cfg.pop("method", cfg.pop("name", ""))).strip().lower()
    if method == "dvae":
        cls = PointBertDvaeTokenizer
    elif method == "posbert":
        cls = PosBertStyleTokenizer
    elif method == "mae_distill":
        cls = MaeDistillPartTokenizer
    elif method == "part_query_mae":
        cls = PartQueryMaeTokenizer
    elif method in {"raw_point_mlp", "raw_mlp", "raw_point"}:
        cls = RawPointMlpSensorEncoder
    elif method in {"raw_point_pointnet", "raw_pointnet", "pointnet"}:
        cls = RawPointNetSensorEncoder
    else:
        raise ValueError(f"Unknown mmWave baseline tokenizer method {method!r}")
    return cls(**_constructor_kwargs(cls, cfg))
