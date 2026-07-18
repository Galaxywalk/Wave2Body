# -*- coding: utf-8 -*-
"""Sample-level pose and confidence export metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def resolve_path(base: Path, path_like: str | Path) -> Path:
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p
    cwd_candidate = p.resolve()
    base_candidate = (base / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    if base_candidate.exists():
        return base_candidate
    return cwd_candidate


def per_sample_mpjpe_mm(pred_pose: torch.Tensor, target_pose: torch.Tensor) -> np.ndarray:
    errors = torch.norm(pred_pose - target_pose, dim=-1)
    return (errors.mean(dim=-1) * 1000.0).detach().cpu().numpy()


def per_joint_error_mm(pred_pose: torch.Tensor, target_pose: torch.Tensor) -> np.ndarray:
    return (torch.norm(pred_pose - target_pose, dim=-1) * 1000.0).detach().cpu().numpy()


def per_sample_p_mpjpe_mm(pred_pose: torch.Tensor, target_pose: torch.Tensor) -> np.ndarray:
    pred = pred_pose.detach().cpu().numpy()
    gt = target_pose.detach().cpu().numpy()
    if pred.shape != gt.shape:
        raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {gt.shape}")

    mu_x = np.mean(gt, axis=1, keepdims=True)
    mu_y = np.mean(pred, axis=1, keepdims=True)
    x0 = gt - mu_x
    y0 = pred - mu_y

    norm_x = np.sqrt(np.sum(x0**2, axis=(1, 2), keepdims=True))
    norm_y = np.sqrt(np.sum(y0**2, axis=(1, 2), keepdims=True))

    eps = 1e-8
    x0 = x0 / np.maximum(norm_x, eps)
    y0 = y0 / np.maximum(norm_y, eps)

    h = np.matmul(x0.transpose(0, 2, 1), y0)
    u, s, vt = np.linalg.svd(h)
    v = vt.transpose(0, 2, 1)
    r = np.matmul(v, u.transpose(0, 2, 1))

    sign_det_r = np.sign(np.expand_dims(np.linalg.det(r), axis=1))
    v[:, :, -1] *= sign_det_r
    s[:, -1] *= sign_det_r.flatten()
    r = np.matmul(v, u.transpose(0, 2, 1))

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * norm_x / np.maximum(norm_y, eps)
    t = mu_x - a * np.matmul(mu_y, r)
    pred_aligned = a * np.matmul(pred, r) + t

    per_joint = np.linalg.norm(pred_aligned - gt, axis=-1)
    return per_joint.mean(axis=1) * 1000.0


def aggregate_part_confidence(part_conf: torch.Tensor, layout, mode: str) -> torch.Tensor:
    if mode == "mean":
        return part_conf.mean(dim=1)
    if mode == "min":
        return part_conf.min(dim=1).values
    if mode == "weighted_mean":
        weights = torch.tensor(
            [len(layout.part_joints[name]) for name in layout.part_names],
            dtype=part_conf.dtype,
            device=part_conf.device,
        )
        return (part_conf * weights.unsqueeze(0)).sum(dim=1) / torch.clamp(weights.sum(), min=1e-8)
    raise ValueError(f"Unsupported sample-confidence aggregation: {mode}")


def part_confidence_to_joint_confidence(part_conf: torch.Tensor, layout) -> torch.Tensor:
    batch_size = int(part_conf.shape[0])
    num_joints = int(layout.num_joints)
    out_sum = torch.zeros((batch_size, num_joints), dtype=part_conf.dtype, device=part_conf.device)
    out_cnt = torch.zeros((batch_size, num_joints), dtype=part_conf.dtype, device=part_conf.device)
    for pid, name in enumerate(layout.part_names):
        idx = torch.as_tensor(layout.part_joints[name], dtype=torch.long, device=part_conf.device)
        out_sum[:, idx] += part_conf[:, pid].unsqueeze(1)
        out_cnt[:, idx] += 1.0
    return out_sum / torch.clamp(out_cnt, min=1.0)


def bernoulli_entropy(prob: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    p = torch.clamp(prob, min=float(eps), max=1.0 - float(eps))
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
