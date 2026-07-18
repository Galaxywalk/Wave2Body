# -*- coding: utf-8 -*-
"""Point Transformer pose baseline model and loss."""

from __future__ import annotations

import torch
import torch.nn as nn

from .point_transformer import PointTransformer


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


__all__ = ["PTPoseModel", "PoseMPJPELoss"]
