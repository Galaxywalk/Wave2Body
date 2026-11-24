# -*- coding: utf-8 -*-
"""Point Transformer pose baseline model and loss."""

from __future__ import annotations

import torch
import torch.nn as nn

from .point_transformer import PointTransformer
from ..hpe_token.absolute_root import AbsoluteRootHead, masked_point_root_stats


class PTPoseModel(nn.Module):
    """Point Transformer wrapper for direct joint-position regression."""

    def __init__(self, point_transformer_config: dict, device: str = "cuda") -> None:
        super().__init__()
        self.pt = PointTransformer(point_transformer_config, device=device)

    def forward(self, mmwave_points: torch.Tensor) -> dict[str, torch.Tensor]:
        pred_pose, joint_feat = self.pt(mmwave_points)
        return {
            "pred_pose": pred_pose,
            "joint_feat": joint_feat,
        }


class PoseMPJPELoss(nn.Module):
    """Mean per-joint Euclidean error in meters, used directly as training loss."""

    def forward(self, pred_pose: torch.Tensor, target_pose: torch.Tensor) -> dict[str, torch.Tensor]:
        if pred_pose.shape != target_pose.shape:
            raise ValueError(f"pred/target pose shape mismatch: {pred_pose.shape} vs {target_pose.shape}")

        per_joint_errors = torch.norm(pred_pose - target_pose, dim=-1)
        pose_loss = per_joint_errors.mean()
        return {
            "total_loss": pose_loss,
            "pose_loss": pose_loss,
            "per_joint_errors": per_joint_errors.detach(),
            "mean_joint_error_m": per_joint_errors.mean().detach(),
        }


class PTDecoupledRootPoseModel(nn.Module):
    """PointTransformer relative articulation head + Wave2Body-style residual root head."""

    def __init__(
        self,
        point_transformer_config: dict,
        root_head_config: dict | None = None,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.pt = PointTransformer(point_transformer_config, device=device)
        rcfg = dict(root_head_config or {})
        rcfg.pop("enabled", None)
        self.root_head = AbsoluteRootHead(
            sensor_token_dim=int(self.pt.feat_dim),
            hidden_dim=int(rcfg.get("hidden_dim", 256)),
            num_layers=int(rcfg.get("num_layers", 3)),
            dropout=float(rcfg.get("dropout", 0.1)),
            use_centers=False,
            point_stats_dim=int(rcfg.get("point_stats_dim", 19)),
            prior_mode=str(rcfg.get("prior_mode", "point_intensity_weighted_mean")),
            residual_scale=float(rcfg.get("residual_scale", 1.0)),
            architecture=str(rcfg.get("architecture", "query_attn")),
            attention_hidden_dim=rcfg.get("attention_hidden_dim", 256),
            attention_heads=int(rcfg.get("attention_heads", 4)),
            num_query_tokens=int(rcfg.get("num_query_tokens", 4)),
            use_point_stats=bool(rcfg.get("use_point_stats", True)),
        )

    def forward(self, mmwave_points: torch.Tensor) -> dict[str, torch.Tensor]:
        pred_pose_relative, joint_feat = self.pt(mmwave_points)
        stats = masked_point_root_stats(mmwave_points)
        pred_root, pred_root_prior, pred_root_delta = self.root_head.forward_with_components(
            joint_feat, None, None, stats, None
        )
        pred_pose_absolute = pred_pose_relative + pred_root.unsqueeze(1)
        return {
            "pred_pose": pred_pose_absolute,
            "pred_pose_absolute": pred_pose_absolute,
            "pred_pose_relative": pred_pose_relative,
            "pred_root": pred_root,
            "pred_root_prior": pred_root_prior,
            "pred_root_delta": pred_root_delta,
            "joint_feat": joint_feat,
        }


class PoseDecoupledRootLoss(nn.Module):
    """Decoupled relative + root + absolute MPJPE loss."""

    def __init__(
        self,
        rel_weight: float = 1.0,
        root_weight: float = 1.0,
        abs_weight: float = 0.25,
        root_index: int = 0,
    ) -> None:
        super().__init__()
        self.rel_weight = float(rel_weight)
        self.root_weight = float(root_weight)
        self.abs_weight = float(abs_weight)
        self.root_index = int(root_index)

    def forward(self, outputs: dict[str, torch.Tensor], target_pose: torch.Tensor) -> dict[str, torch.Tensor]:
        pred_abs = outputs["pred_pose"].float()
        pred_rel = outputs["pred_pose_relative"].float()
        pred_root = outputs["pred_root"].float()
        tgt_abs = target_pose.float()
        tgt_root = tgt_abs[:, self.root_index, :]
        tgt_rel = tgt_abs - tgt_root.unsqueeze(1)
        rel_loss = torch.norm(pred_rel - tgt_rel, dim=-1).mean()
        root_loss = torch.norm(pred_root - tgt_root, dim=-1).mean()
        abs_loss = torch.norm(pred_abs - tgt_abs, dim=-1).mean()
        total = self.rel_weight * rel_loss + self.root_weight * root_loss + self.abs_weight * abs_loss
        return {
            "total_loss": total,
            "pose_loss": abs_loss,
            "rel_loss": rel_loss,
            "root_loss": root_loss,
            "mean_joint_error_m": abs_loss.detach(),
        }


def compute_pt_pose_loss(
    criterion: nn.Module,
    outputs: dict[str, torch.Tensor],
    target_pose: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Dispatch PointTransformer loss uniformly across direct and decoupled-root criteria."""
    if isinstance(criterion, PoseDecoupledRootLoss):
        return criterion(outputs, target_pose.float())
    return criterion(outputs["pred_pose"].float(), target_pose.float())


__all__ = [
    "PTPoseModel",
    "PoseMPJPELoss",
    "PTDecoupledRootPoseModel",
    "PoseDecoupledRootLoss",
    "compute_pt_pose_loss",
]
