# -*- coding: utf-8 -*-
"""Patch grouping utilities for mmWave sensor tokenizers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PatchBatch:
    """Fixed-shape local point patches.

    ``patches`` stores local coordinates: xyz is translated by the patch center,
    while non-xyz channels are kept in the original radar feature scale.
    """

    patches: torch.Tensor
    raw_patches: torch.Tensor
    centers: torch.Tensor
    group_mask: torch.Tensor
    point_mask: torch.Tensor
    indices: torch.Tensor
    policy: str


def valid_point_mask(points: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Return a mask for non-padding points using xyz channels only."""
    if points.dim() == 2:
        return torch.any(torch.abs(points[:, :3]) > float(eps), dim=-1)
    if points.dim() == 3:
        return torch.any(torch.abs(points[:, :, :3]) > float(eps), dim=-1)
    raise ValueError(f"Expected points [N,C] or [B,N,C], got {tuple(points.shape)}")


def _first_valid_indices(valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid_counts = valid_mask.sum(dim=1)
    first = torch.argmax(valid_mask.to(dtype=torch.long), dim=1)
    first = torch.where(valid_counts > 0, first, torch.zeros_like(first))
    return first, valid_counts


def _gather_points_batched(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    bsz, groups, group_size = indices.shape
    channels = int(points.shape[-1])
    source = points.unsqueeze(1).expand(bsz, groups, points.shape[1], channels)
    gather_idx = indices.unsqueeze(-1).expand(bsz, groups, group_size, channels)
    return torch.gather(source, dim=2, index=gather_idx)


def _gather_xyz_batched(xyz: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    bsz, groups = indices.shape
    gather_idx = indices.unsqueeze(-1).expand(bsz, groups, 3)
    return torch.gather(xyz, dim=1, index=gather_idx)


def _batched_farthest_point_indices(
    xyz: torch.Tensor,
    valid_mask: torch.Tensor,
    num_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched FPS over fixed padded point clouds."""
    bsz, n_points, _ = xyz.shape
    device = xyz.device
    first_valid, valid_counts = _first_valid_indices(valid_mask)

    weights = valid_mask.to(dtype=xyz.dtype).unsqueeze(-1)
    denom = valid_counts.clamp_min(1).to(dtype=xyz.dtype).unsqueeze(-1)
    centroid = (xyz * weights).sum(dim=1) / denom
    dist_to_centroid = torch.sum((xyz.float() - centroid.unsqueeze(1).float()) ** 2, dim=-1)
    dist_to_centroid = dist_to_centroid.masked_fill(~valid_mask, float("-inf"))
    farthest = torch.argmax(dist_to_centroid, dim=1)
    farthest = torch.where(valid_counts > 0, farthest, first_valid)

    selected = torch.zeros(bsz, int(num_groups), device=device, dtype=torch.long)
    min_dist = torch.full((bsz, n_points), float("inf"), device=device, dtype=torch.float32)
    for group_idx in range(int(num_groups)):
        selected[:, group_idx] = farthest
        center = torch.gather(xyz, dim=1, index=farthest.view(bsz, 1, 1).expand(bsz, 1, 3)).squeeze(1)
        dist = torch.sum((xyz.float() - center.unsqueeze(1).float()) ** 2, dim=-1)
        dist = dist.masked_fill(~valid_mask, float("inf"))
        min_dist = torch.minimum(min_dist, dist)
        candidate = min_dist.masked_fill(~valid_mask, float("-inf"))
        farthest = torch.argmax(candidate, dim=1)
        farthest = torch.where(valid_counts > 0, farthest, first_valid)

    group_ids = torch.arange(int(num_groups), device=device).unsqueeze(0)
    group_mask = group_ids < valid_counts.clamp(max=int(num_groups)).unsqueeze(1)
    return selected, group_mask


def _body_part_centers_batched(
    target_poses_absolute: torch.Tensor,
    *,
    valid_counts: torch.Tensor,
    num_groups: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    joints = target_poses_absolute[:, :, :3].to(dtype=dtype)
    bsz, n_joints, _ = joints.shape
    device = joints.device
    centers = torch.zeros(bsz, int(num_groups), 3, device=device, dtype=dtype)
    copy_count = min(int(n_joints), int(num_groups))
    if copy_count > 0:
        centers[:, :copy_count] = joints[:, :copy_count]
        if copy_count < int(num_groups):
            centers[:, copy_count:] = joints[:, copy_count - 1 : copy_count]
    group_ids = torch.arange(int(num_groups), device=device).unsqueeze(0)
    group_mask = (group_ids < copy_count) & (valid_counts.unsqueeze(1) > 0)
    return centers, group_mask


def _knn_indices_batched(
    xyz: torch.Tensor,
    centers: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    group_size: int,
    group_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, n_points, _ = xyz.shape
    group_size = int(group_size)
    k = min(group_size, int(n_points))
    dist = torch.cdist(centers.float(), xyz.float())
    dist = dist.masked_fill(~valid_mask.unsqueeze(1), float("inf"))
    knn_dist, idx = torch.topk(dist, k=k, dim=-1, largest=False, sorted=True)
    if k < group_size:
        pad_count = group_size - k
        pad_idx = idx[..., :1].expand(bsz, centers.shape[1], pad_count)
        pad_dist = torch.full(
            (bsz, centers.shape[1], pad_count),
            float("inf"),
            device=xyz.device,
            dtype=knn_dist.dtype,
        )
        idx = torch.cat([idx, pad_idx], dim=-1)
        knn_dist = torch.cat([knn_dist, pad_dist], dim=-1)

    first_valid, _valid_counts = _first_valid_indices(valid_mask)
    fill_idx = first_valid.view(bsz, 1, 1).expand_as(idx)
    finite = torch.isfinite(knn_dist)
    idx = torch.where(finite, idx, fill_idx)
    point_mask = finite & group_mask.unsqueeze(-1)
    return idx[:, :, :group_size], point_mask[:, :, :group_size]


def group_point_cloud(
    points: torch.Tensor,
    *,
    num_groups: int,
    group_size: int,
    policy: str = "spatial_fps_knn",
    target_poses_absolute: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> PatchBatch:
    """Group a fixed-size radar point cloud into local patches.

    Parameters
    ----------
    points:
        Tensor ``[B,N,C]`` where the first three channels are xyz.
    policy:
        ``spatial_fps_knn`` for pure self-supervised FPS centers, or
        ``weak_body_part_knn`` for GT-joint centers.
    target_poses_absolute:
        Required for the body-part policy. Used only to choose patch centers.
    """
    if points.dim() != 3 or points.shape[-1] < 3:
        raise ValueError(f"Expected points [B,N,C>=3], got {tuple(points.shape)}")
    key = str(policy).strip().lower()
    if key not in {"spatial_fps_knn", "weak_body_part_knn"}:
        raise ValueError(f"Unknown patch policy {policy!r}")
    if key == "weak_body_part_knn" and target_poses_absolute is None:
        raise ValueError("target_poses_absolute is required for weak_body_part_knn")

    bsz, _n_points, channels = points.shape
    device = points.device
    valid_mask = valid_point_mask(points, eps=eps)
    _first_valid, valid_counts = _first_valid_indices(valid_mask)
    xyz = points[:, :, :3]

    if key == "spatial_fps_knn":
        center_indices, group_mask_out = _batched_farthest_point_indices(xyz, valid_mask, int(num_groups))
        centers_out = _gather_xyz_batched(xyz, center_indices).to(dtype=points.dtype)
    else:
        assert target_poses_absolute is not None
        centers_out, group_mask_out = _body_part_centers_batched(
            target_poses_absolute.to(device=device),
            valid_counts=valid_counts,
            num_groups=int(num_groups),
            dtype=points.dtype,
        )

    indices_out, point_mask_out = _knn_indices_batched(
        xyz,
        centers_out,
        valid_mask,
        group_size=int(group_size),
        group_mask=group_mask_out,
    )
    raw_patches = _gather_points_batched(points, indices_out)
    patches = raw_patches.clone()
    patches[..., :3] = patches[..., :3] - centers_out.unsqueeze(2)

    return PatchBatch(
        patches=patches,
        raw_patches=raw_patches,
        centers=centers_out,
        group_mask=group_mask_out,
        point_mask=point_mask_out,
        indices=indices_out,
        policy=key,
    )
