# -*- coding: utf-8 -*-
"""Config-driven constructors for pose-estimation baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .mmdiff import MMDiffPoseLoss, create_mmdiff_model
from .mmmesh import MMMeshLoss, create_mmmesh_model
from .point_transformer.pose_baseline import PTPoseModel, PoseMPJPELoss


@dataclass
class PoseBaselineBundle:
    """A model/loss pair built from a Wave2Body baseline config."""

    name: str
    model: nn.Module
    criterion: nn.Module


def _baseline_name(config: dict[str, Any]) -> str:
    baseline_cfg = config.get("baseline")
    if isinstance(baseline_cfg, dict) and baseline_cfg.get("name"):
        return str(baseline_cfg["name"]).strip().lower()
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict) and model_cfg.get("name"):
        return str(model_cfg["name"]).strip().lower()
    if "point_transformer" in config:
        return "point_transformer"
    raise KeyError("Pose baseline config must define baseline.name or model.name")


def _loss_cfg(config: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = config.get(key)
        if isinstance(value, dict):
            return dict(value)
    value = config.get("loss")
    return dict(value) if isinstance(value, dict) else {}


def build_pose_baseline_model(
    config: dict[str, Any],
    *,
    device: str = "cuda",
    feature_encoder: nn.Module | None = None,
) -> nn.Module:
    """Build one pose baseline model from config."""

    name = _baseline_name(config)
    if name in {"point_transformer", "pt", "pt_pose", "hpe_point_transformer"}:
        if "point_transformer" not in config:
            raise KeyError("PointTransformer baseline config requires a point_transformer section")
        return PTPoseModel(point_transformer_config=config["point_transformer"], device=device)

    if name in {"mmmesh", "mm_mesh"}:
        model_cfg = dict(config.get("mmmesh") or config.get("model") or {})
        model_cfg.pop("name", None)
        return create_mmmesh_model(model_cfg)

    if name in {"mmdiff", "mm_diff"}:
        model_cfg = dict(config.get("mmdiff") or config.get("model") or {})
        model_cfg.pop("name", None)
        return create_mmdiff_model(model_cfg, feature_encoder=feature_encoder)

    raise ValueError(f"Unsupported pose baseline: {name!r}")


def build_pose_baseline_loss(config: dict[str, Any]) -> nn.Module:
    """Build the criterion matching a pose baseline config."""

    name = _baseline_name(config)
    if name in {"point_transformer", "pt", "pt_pose", "hpe_point_transformer"}:
        return PoseMPJPELoss()
    if name in {"mmmesh", "mm_mesh"}:
        return MMMeshLoss(**_loss_cfg(config, "mmmesh_loss"))
    if name in {"mmdiff", "mm_diff"}:
        return MMDiffPoseLoss(**_loss_cfg(config, "mmdiff_loss"))
    raise ValueError(f"Unsupported pose baseline: {name!r}")


def build_pose_baseline(
    config: dict[str, Any],
    *,
    device: str = "cuda",
    feature_encoder: nn.Module | None = None,
) -> PoseBaselineBundle:
    """Build a model and criterion for a pose baseline config."""

    name = _baseline_name(config)
    return PoseBaselineBundle(
        name=name,
        model=build_pose_baseline_model(config, device=device, feature_encoder=feature_encoder),
        criterion=build_pose_baseline_loss(config),
    )


def _require_tensor(batch: dict[str, Any], key: str) -> torch.Tensor:
    value = batch.get(key)
    if not torch.is_tensor(value):
        raise KeyError(f"Batch is missing tensor field {key!r}")
    return value


def move_pose_baseline_batch(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    """Move tensor fields in a baseline batch to the target device."""

    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def forward_pose_baseline(bundle: PoseBaselineBundle, batch: dict[str, Any]) -> dict[str, Any]:
    """Run a baseline model against a standard radar-pose batch."""

    points = _require_tensor(batch, "mmwave_points").float()
    if bundle.name in {"mmdiff", "mm_diff"}:
        timesteps = batch.get("timesteps", batch.get("diffusion_timesteps", 0))
        outputs = bundle.model(points, timesteps=timesteps)
    elif bundle.name in {"mmmesh", "mm_mesh"}:
        gender = batch.get("target_gender", batch.get("gender"))
        if points.ndim == 3:
            points = points.unsqueeze(1)
        outputs = bundle.model(points, gender=gender)
    else:
        outputs = bundle.model(points)
    if not isinstance(outputs, dict):
        raise TypeError(f"Pose baseline {bundle.name!r} must return a dict, got {type(outputs)}")
    return outputs


def _match_mmmesh_time_dim(target: torch.Tensor | None, pred: torch.Tensor | None) -> torch.Tensor | None:
    """Broadcast per-sample mmMesh targets to the model's temporal output shape."""

    if target is None or pred is None:
        return target
    if not torch.is_tensor(target) or not torch.is_tensor(pred):
        return target
    if target.shape == pred.shape or pred.ndim < 3:
        return target

    batch_size, length_size = int(pred.shape[0]), int(pred.shape[1])
    if int(target.shape[0]) != batch_size:
        return target

    if target.ndim == 1 and pred.shape[-1] == 1:
        target = target.reshape(batch_size, 1)

    if target.ndim == pred.ndim - 1 and tuple(target.shape[1:]) == tuple(pred.shape[2:]):
        return target.unsqueeze(1).expand(batch_size, length_size, *target.shape[1:])

    if (
        target.ndim == pred.ndim
        and int(target.shape[1]) == 1
        and tuple(target.shape[2:]) == tuple(pred.shape[2:])
    ):
        return target.expand(batch_size, length_size, *target.shape[2:])

    return target


def _tensor_or_none(batch: dict[str, Any], key: str) -> torch.Tensor | None:
    value = batch.get(key)
    return value if torch.is_tensor(value) else None


def _root_xy_from_target_pose(batch: dict[str, Any]) -> torch.Tensor | None:
    target = _tensor_or_none(batch, "target_poses_absolute")
    if target is None:
        target = _tensor_or_none(batch, "target_poses")
    if target is None or target.ndim < 3 or target.shape[-1] < 2:
        return None
    if target.ndim == 3:
        return target[:, 0, :2]
    if target.ndim == 4:
        return target[:, :, 0, :2]
    return None


def _joints_from_target_pose(batch: dict[str, Any]) -> torch.Tensor | None:
    target = _tensor_or_none(batch, "target_poses_absolute")
    if target is None:
        target = _tensor_or_none(batch, "target_poses")
    if target is None or target.ndim < 3 or target.shape[-1] != 3:
        return None
    return target


def _translation_from_target_pose(batch: dict[str, Any]) -> torch.Tensor | None:
    target = _tensor_or_none(batch, "target_poses_absolute")
    if target is None:
        target = _tensor_or_none(batch, "target_poses")
    if target is None or target.ndim < 3 or target.shape[-1] != 3:
        return None
    if target.ndim == 3:
        return target[:, 0, :]
    if target.ndim == 4:
        return target[:, :, 0, :]
    return None


def compute_pose_baseline_loss(
    bundle: PoseBaselineBundle,
    outputs: dict[str, Any],
    batch: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Compute the right loss for one baseline output and data batch."""

    if isinstance(bundle.criterion, PoseMPJPELoss):
        target = _require_tensor(batch, "target_poses").float()
        pred = outputs.get("pred_pose")
        if not torch.is_tensor(pred):
            raise KeyError(f"Pose baseline {bundle.name!r} did not return tensor outputs['pred_pose']")
        return bundle.criterion(pred.float(), target)

    if isinstance(bundle.criterion, MMDiffPoseLoss):
        target = _require_tensor(batch, "target_poses").float()
        target_history = batch.get("target_history")
        target_limb_lengths = batch.get("target_limb_lengths")
        return bundle.criterion(
            outputs,
            target_pose=target,
            target_history=target_history if torch.is_tensor(target_history) else None,
            target_limb_lengths=target_limb_lengths if torch.is_tensor(target_limb_lengths) else None,
        )

    if isinstance(bundle.criterion, MMMeshLoss):
        pose_root_xy = _root_xy_from_target_pose(batch)
        pose_joints = _joints_from_target_pose(batch)
        pose_translation = _translation_from_target_pose(batch)
        missing = [
            key
            for key in (
                "target_translation",
                "target_root_xy",
                "target_betas",
                "target_gender",
                "target_rotmat",
                "target_joints",
                "target_vertices",
            )
            if (
                not torch.is_tensor(batch.get(key))
                and not (key == "target_root_xy" and pose_root_xy is not None)
                and not (key == "target_joints" and pose_joints is not None)
                and not (key == "target_translation" and pose_translation is not None)
            )
        ]
        if len(missing) == 7:
            raise KeyError(
                "mmMesh loss requires SMPL/body-parameter targets, root targets, or decoded joints/vertices. "
                "The standard radar-pose datasets only provide target_poses; add a dataset adapter "
                "that emits one of target_translation/target_betas/target_gender/target_rotmat/"
                "target_root_xy/target_joints/target_vertices."
            )
        body = outputs.get("body")
        pred_joints = body.get("joints") if isinstance(body, dict) and torch.is_tensor(body.get("joints")) else None
        if pred_joints is None and isinstance(body, dict) and torch.is_tensor(body.get("pred_joints")):
            pred_joints = body["pred_joints"]
        pred_vertices = body.get("vertices") if isinstance(body, dict) and torch.is_tensor(body.get("vertices")) else None
        if pred_vertices is None and isinstance(body, dict) and torch.is_tensor(body.get("pred_vertices")):
            pred_vertices = body["pred_vertices"]

        target_translation = _tensor_or_none(batch, "target_translation")
        if target_translation is None:
            target_translation = pose_translation
        target_translation = _match_mmmesh_time_dim(target_translation, outputs.get("pred_translation"))
        target_root_xy = _tensor_or_none(batch, "target_root_xy")
        if target_root_xy is None:
            target_root_xy = pose_root_xy
        target_root_xy = _match_mmmesh_time_dim(target_root_xy, outputs.get("pred_root_xy"))
        target_betas = _match_mmmesh_time_dim(_tensor_or_none(batch, "target_betas"), outputs.get("pred_betas"))
        target_gender = _match_mmmesh_time_dim(_tensor_or_none(batch, "target_gender"), outputs.get("pred_gender_logit"))
        target_rotmat = _match_mmmesh_time_dim(_tensor_or_none(batch, "target_rotmat"), outputs.get("pred_rotmat"))
        target_joints = _tensor_or_none(batch, "target_joints")
        if target_joints is None:
            target_joints = pose_joints
        target_joints = _match_mmmesh_time_dim(target_joints, pred_joints)
        target_vertices = _match_mmmesh_time_dim(_tensor_or_none(batch, "target_vertices"), pred_vertices)
        return bundle.criterion(
            outputs,
            target_rotmat=target_rotmat,
            target_translation=target_translation,
            target_root_xy=target_root_xy,
            target_betas=target_betas,
            target_gender=target_gender,
            target_joints=target_joints,
            target_vertices=target_vertices,
        )

    raise TypeError(f"Unsupported baseline criterion type: {type(bundle.criterion)}")


def pose_baseline_step(
    bundle: PoseBaselineBundle,
    batch: dict[str, Any],
    *,
    device: torch.device | str | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Run forward plus loss for one standard baseline batch."""

    if device is not None:
        batch = move_pose_baseline_batch(batch, device)
    outputs = forward_pose_baseline(bundle, batch)
    return outputs, compute_pose_baseline_loss(bundle, outputs, batch)


__all__ = [
    "PoseBaselineBundle",
    "build_pose_baseline",
    "build_pose_baseline_loss",
    "build_pose_baseline_model",
    "compute_pose_baseline_loss",
    "forward_pose_baseline",
    "move_pose_baseline_batch",
    "pose_baseline_step",
]
