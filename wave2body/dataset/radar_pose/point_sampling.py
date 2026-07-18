# -*- coding: utf-8 -*-
"""Small point-cloud sampling helpers used by dataset adapters."""

from __future__ import annotations

import torch


VALID_POINT_SAMPLING_STRATEGIES = {
    "fps",
    "torch_cluster_fps",
    "random",
    "linspace",
    "top_abs_intensity",
    "top_velocity_abs",
    "near_origin",
    "hybrid_abs_intensity_fps",
}


def farthest_point_sample_indices(xyz: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Return deterministic farthest-point-sampling indices for one point set."""
    if xyz.dim() != 2 or xyz.shape[-1] < 3:
        raise ValueError(f"Expected xyz with shape [N, >=3], got {tuple(xyz.shape)}")
    n_points = int(xyz.shape[0])
    n_samples = int(n_samples)
    if n_samples <= 0:
        return torch.empty((0,), device=xyz.device, dtype=torch.long)
    if n_points <= 0:
        return torch.zeros((n_samples,), device=xyz.device, dtype=torch.long)

    selected = torch.zeros((n_samples,), device=xyz.device, dtype=torch.long)
    unique_count = min(n_points, n_samples)
    work_xyz = xyz[:, :3].float()

    centroid = work_xyz.mean(dim=0, keepdim=True)
    farthest = torch.argmax(torch.sum((work_xyz - centroid) ** 2, dim=-1))
    min_dist = torch.full((n_points,), float("inf"), device=xyz.device, dtype=work_xyz.dtype)
    for sample_idx in range(unique_count):
        selected[sample_idx] = farthest
        center = work_xyz[farthest : farthest + 1]
        dist = torch.sum((work_xyz - center) ** 2, dim=-1)
        min_dist = torch.minimum(min_dist, dist)
        farthest = torch.argmax(min_dist)

    if unique_count < n_samples:
        repeats = selected[:unique_count].repeat(
            (n_samples - unique_count + unique_count - 1) // unique_count
        )
        selected[unique_count:] = repeats[: n_samples - unique_count]
    return selected


def torch_cluster_farthest_point_sample_indices(
    xyz: torch.Tensor,
    n_samples: int,
    *,
    random_start: bool = False,
) -> torch.Tensor:
    """FPS via PyG's optional torch-cluster extension.

    ``torch_cluster`` ships prebuilt CPU/CUDA wheels for common PyTorch/CUDA
    combinations.  It is intentionally optional: the default code path remains
    pure PyTorch and reproducible without external C++/CUDA extensions.
    """
    if xyz.dim() != 2 or xyz.shape[-1] < 3:
        raise ValueError(f"Expected xyz with shape [N, >=3], got {tuple(xyz.shape)}")
    n_points = int(xyz.shape[0])
    n_samples = int(n_samples)
    if n_samples <= 0:
        return torch.empty((0,), device=xyz.device, dtype=torch.long)
    if n_points <= 0:
        return torch.zeros((n_samples,), device=xyz.device, dtype=torch.long)
    if n_samples >= n_points:
        return farthest_point_sample_indices(xyz, n_samples)

    try:
        from torch_cluster import fps as torch_cluster_fps
    except ImportError as exc:
        raise ImportError(
            "point_sampling='torch_cluster_fps' requires the optional torch-cluster package. "
            "For the locked Wave2Body Torch/CUDA stack, install it into the uv environment with: "
            "uv pip install --python .venv/bin/python torch-cluster "
            "-f https://data.pyg.org/whl/torch-2.5.0+cu124.html"
        ) from exc

    ratio = min(max(float(n_samples) / float(n_points), 1e-6), 1.0 - 1e-6)
    idx = torch_cluster_fps(
        xyz[:, :3].contiguous().float(),
        batch=None,
        ratio=ratio,
        random_start=bool(random_start),
    ).to(device=xyz.device, dtype=torch.long)

    if idx.numel() >= n_samples:
        return idx[:n_samples]

    fallback = farthest_point_sample_indices(xyz, n_samples)
    extras = fallback[~torch.isin(fallback, idx)]
    selected = torch.cat([idx, extras], dim=0)
    if selected.numel() >= n_samples:
        return selected[:n_samples]
    pad = fallback.repeat((n_samples - selected.numel() + fallback.numel() - 1) // fallback.numel())
    return torch.cat([selected, pad[: n_samples - selected.numel()]], dim=0)


def _top_score_indices(points: torch.Tensor, score: torch.Tensor, n_samples: int) -> torch.Tensor:
    if score.dim() != 1 or int(score.shape[0]) != int(points.shape[0]):
        raise ValueError(
            f"Expected score with shape [{int(points.shape[0])}], got {tuple(score.shape)}"
        )
    score = torch.nan_to_num(
        score.float(),
        nan=-float("inf"),
        posinf=float("inf"),
        neginf=-float("inf"),
    )
    return torch.topk(score, k=int(n_samples), largest=True, sorted=True).indices


def _required_feature(points: torch.Tensor, column: int, strategy: str) -> torch.Tensor:
    if int(points.shape[1]) <= int(column):
        raise ValueError(
            f"point_sampling={strategy!r} requires point channel {int(column)}, "
            f"got {int(points.shape[1])} channels"
        )
    return points[:, int(column)]


def sample_point_indices(
    points: torch.Tensor,
    n_samples: int,
    *,
    strategy: str = "fps",
) -> torch.Tensor:
    """Sample point indices with a small set of CPU/GPU-safe strategies."""
    if points.dim() != 2:
        raise ValueError(f"Expected points with shape [N, C], got {tuple(points.shape)}")
    n_points = int(points.shape[0])
    n_samples = int(n_samples)
    if n_samples <= 0:
        return torch.empty((0,), device=points.device, dtype=torch.long)
    if n_points <= 0:
        return torch.zeros((n_samples,), device=points.device, dtype=torch.long)

    strategy = str(strategy).strip().lower()
    if strategy == "fps":
        return farthest_point_sample_indices(points[:, :3], n_samples)
    if strategy == "torch_cluster_fps":
        return torch_cluster_farthest_point_sample_indices(points[:, :3], n_samples)
    if strategy == "random":
        return torch.randperm(n_points, device=points.device)[:n_samples]
    if strategy == "linspace":
        return torch.linspace(0, n_points - 1, steps=n_samples, device=points.device).round().long()
    if strategy == "top_abs_intensity":
        score = _required_feature(points, 3, strategy).abs()
        return _top_score_indices(points, score, n_samples)
    if strategy == "top_velocity_abs":
        score = _required_feature(points, 4, strategy).abs()
        return _top_score_indices(points, score, n_samples)
    if strategy == "near_origin":
        score = -torch.sum(points[:, :3].float() ** 2, dim=-1)
        return _top_score_indices(points, score, n_samples)
    if strategy == "hybrid_abs_intensity_fps":
        score = _required_feature(points, 3, strategy).abs()
        candidate_count = min(n_points, max(n_samples, n_samples * 4))
        candidate_idx = _top_score_indices(points, score, candidate_count)
        local_idx = farthest_point_sample_indices(points[candidate_idx, :3], n_samples)
        return candidate_idx[local_idx]
    raise ValueError(
        f"Unknown point sampling strategy {strategy!r}; "
        f"expected one of {sorted(VALID_POINT_SAMPLING_STRATEGIES)}"
    )


def pad_or_sample_points(
    points: torch.Tensor,
    *,
    n_points: int,
    input_channels: int,
    strategy: str = "fps",
) -> torch.Tensor:
    """Return a fixed-size point tensor without relying on custom CUDA ops."""
    if points.dim() != 2:
        raise ValueError(f"Expected points with shape [N, C], got {tuple(points.shape)}")
    n_points = int(n_points)
    input_channels = int(input_channels)
    if n_points <= 0:
        raise ValueError(f"n_points must be positive, got {n_points}")
    if input_channels <= 0:
        raise ValueError(f"input_channels must be positive, got {input_channels}")
    if int(points.shape[1]) != input_channels:
        raise ValueError(
            f"Point channel count mismatch: got {points.shape[1]}, expected {input_channels}"
        )

    n_current = int(points.shape[0])
    if n_current == 0:
        return points.new_zeros((n_points, input_channels))
    if n_current < n_points:
        pad = points.new_zeros((n_points - n_current, input_channels))
        return torch.cat([points, pad], dim=0)
    if n_current == n_points:
        return points

    idx = sample_point_indices(points, n_points, strategy=strategy)
    return points[idx]
