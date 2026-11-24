# -*- coding: utf-8 -*-
"""Patch grouping utilities for mmWave sensor tokenizers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - optional GPU acceleration backend
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    HAS_TRITON = False


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


if HAS_TRITON:

    @triton.jit
    def _argmax_tie_lowest(val_a, idx_a, val_b, idx_b):
        take_a = (val_a > val_b) | ((val_a == val_b) & (idx_a < idx_b))
        return tl.where(take_a, val_a, val_b), tl.where(take_a, idx_a, idx_b)

    @triton.jit
    def _triton_valid_fps_loop_kernel(
        xyz_ptr,
        valid_ptr,
        init_ptr,
        first_ptr,
        vcnt_ptr,
        out_ptr,
        stride_xb,
        stride_xn,
        stride_vb,
        stride_ob,
        num_groups,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        b = tl.program_id(0)
        offs = tl.arange(0, BLOCK_N)
        in_bounds = offs < N

        valid_i8 = tl.load(valid_ptr + b * stride_vb + offs, mask=in_bounds, other=0)
        valid = (valid_i8 != 0) & in_bounds

        px = tl.load(xyz_ptr + b * stride_xb + offs * stride_xn + 0, mask=in_bounds, other=0.0).to(tl.float32)
        py = tl.load(xyz_ptr + b * stride_xb + offs * stride_xn + 1, mask=in_bounds, other=0.0).to(tl.float32)
        pz = tl.load(xyz_ptr + b * stride_xb + offs * stride_xn + 2, mask=in_bounds, other=0.0).to(tl.float32)

        vcnt = tl.load(vcnt_ptr + b)
        first_v = tl.load(first_ptr + b).to(tl.int32)
        farthest = tl.load(init_ptr + b).to(tl.int32)

        min_dist = tl.full([BLOCK_N], float("inf"), dtype=tl.float32)
        for g in range(num_groups):
            tl.store(out_ptr + b * stride_ob + g, farthest.to(tl.int64))
            cx = tl.load(xyz_ptr + b * stride_xb + farthest * stride_xn + 0).to(tl.float32)
            cy = tl.load(xyz_ptr + b * stride_xb + farthest * stride_xn + 1).to(tl.float32)
            cz = tl.load(xyz_ptr + b * stride_xb + farthest * stride_xn + 2).to(tl.float32)
            dx = px - cx
            dy = py - cy
            dz = pz - cz
            dist = tl.where(valid, dx * dx + dy * dy + dz * dz, float("inf"))
            min_dist = tl.minimum(min_dist, dist)
            cand = tl.where(valid, min_dist, -float("inf"))
            _, best_idx = tl.reduce((cand, offs), 0, _argmax_tie_lowest)
            farthest = tl.where(vcnt > 0, best_idx, first_v)

    @torch.library.custom_op("wave2body::triton_valid_fps_indices", mutates_args=())
    def _triton_valid_fps_indices_op(
        xyz: torch.Tensor,
        valid_mask: torch.Tensor,
        first_valid: torch.Tensor,
        valid_counts: torch.Tensor,
        init_farthest: torch.Tensor,
        num_groups: int,
    ) -> torch.Tensor:
        bsz, n_points, _ = xyz.shape
        xyz_c = xyz.contiguous()
        valid_u8 = valid_mask.to(dtype=torch.uint8).contiguous()
        out = torch.empty((bsz, int(num_groups)), device=xyz.device, dtype=torch.long)
        block_n = triton.next_power_of_2(int(n_points))
        num_warps = 4 if block_n <= 1024 else 8
        _triton_valid_fps_loop_kernel[(bsz,)](
            xyz_c,
            valid_u8,
            init_farthest.contiguous(),
            first_valid.contiguous(),
            valid_counts.contiguous(),
            out,
            xyz_c.stride(0),
            xyz_c.stride(1),
            valid_u8.stride(0),
            out.stride(0),
            int(num_groups),
            N=int(n_points),
            BLOCK_N=block_n,
            num_warps=num_warps,
        )
        return out

    @_triton_valid_fps_indices_op.register_fake
    def _triton_valid_fps_indices_fake(
        xyz: torch.Tensor,
        valid_mask: torch.Tensor,
        first_valid: torch.Tensor,
        valid_counts: torch.Tensor,
        init_farthest: torch.Tensor,
        num_groups: int,
    ) -> torch.Tensor:
        return torch.empty((xyz.shape[0], int(num_groups)), device=xyz.device, dtype=torch.long)


def triton_fps_available() -> bool:
    """Return True when the native Triton valid-only FPS backend can be used on CUDA."""
    return bool(HAS_TRITON and torch.cuda.is_available())


def triton_batched_farthest_point_indices(
    xyz: torch.Tensor,
    valid_mask: torch.Tensor,
    num_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU-native valid-only FPS mathematically identical to ``_batched_farthest_point_indices``."""
    if not HAS_TRITON:
        raise RuntimeError("Triton is not installed; cannot run triton_batched_farthest_point_indices.")
    if not xyz.is_cuda:
        raise ValueError("triton_batched_farthest_point_indices requires CUDA tensors.")
    first_valid, valid_counts = _first_valid_indices(valid_mask)
    weights = valid_mask.to(dtype=xyz.dtype).unsqueeze(-1)
    denom = valid_counts.clamp_min(1).to(dtype=xyz.dtype).unsqueeze(-1)
    centroid = (xyz * weights).sum(dim=1) / denom
    dist_to_centroid = torch.sum((xyz.float() - centroid.unsqueeze(1).float()) ** 2, dim=-1)
    dist_to_centroid = dist_to_centroid.masked_fill(~valid_mask, float("-inf"))
    farthest = torch.argmax(dist_to_centroid, dim=1)
    init_farthest = torch.where(valid_counts > 0, farthest, first_valid)

    selected = _triton_valid_fps_indices_op(
        xyz, valid_mask, first_valid, valid_counts, init_farthest, int(num_groups)
    )
    group_ids = torch.arange(int(num_groups), device=xyz.device).unsqueeze(0)
    group_mask = group_ids < valid_counts.clamp(max=int(num_groups)).unsqueeze(1)
    return selected, group_mask


def batched_farthest_point_indices(
    xyz: torch.Tensor,
    valid_mask: torch.Tensor,
    num_groups: int,
    *,
    fps_backend: str = "torch",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch valid-only batched FPS to ``torch`` or ``triton`` (or ``auto``)."""
    backend = str(fps_backend or "torch").strip().lower()
    if backend == "torch":
        return _batched_farthest_point_indices(xyz, valid_mask, int(num_groups))
    if backend == "triton":
        return triton_batched_farthest_point_indices(xyz, valid_mask, int(num_groups))
    if backend == "auto":
        if HAS_TRITON and xyz.is_cuda:
            return triton_batched_farthest_point_indices(xyz, valid_mask, int(num_groups))
        return _batched_farthest_point_indices(xyz, valid_mask, int(num_groups))
    raise ValueError(f"Unsupported fps_backend={fps_backend!r}; expected 'torch', 'triton', or 'auto'.")


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
    fps_backend: str = "torch",
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
        center_indices, group_mask_out = batched_farthest_point_indices(
            xyz,
            valid_mask,
            int(num_groups),
            fps_backend=fps_backend,
        )
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
