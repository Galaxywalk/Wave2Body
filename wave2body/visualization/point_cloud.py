# -*- coding: utf-8 -*-
"""Reusable mmWave point-cloud plotting utilities."""

from __future__ import annotations

from typing import Any

import numpy as np


def as_point_array(points: Any) -> np.ndarray:
    """Return a float point array with at least xyz columns."""

    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected point cloud [N,C>=3], got {arr.shape}")
    return arr


def valid_point_mask(points: Any) -> np.ndarray:
    """Mask padded or invalid radar points."""

    arr = as_point_array(points)
    finite = np.isfinite(arr).all(axis=1)
    nonzero_xyz = np.abs(arr[:, :3]).sum(axis=1) > 0.0
    return finite & nonzero_xyz


def normalize_intensity(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return robust [0,1] intensity values for valid points."""

    if points.shape[1] < 4 or not bool(mask.any()):
        return np.zeros(int(points.shape[0]), dtype=np.float32)
    intensity = np.asarray(points[:, 3], dtype=np.float32)
    valid_values = intensity[mask]
    low, high = np.percentile(valid_values, [5.0, 95.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(int(points.shape[0]), dtype=np.float32)
    return np.clip((intensity - low) / (high - low), 0.0, 1.0).astype(np.float32)


def point_bounds(
    points: Any,
    *,
    margin: float = 0.15,
    min_span: float = 1.0e-3,
    percentile: tuple[float, float] | None = (5.0, 95.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Compute padded xyz bounds for a radar point cloud."""

    arr = as_point_array(points)
    mask = valid_point_mask(arr)
    if not bool(mask.any()):
        return np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)
    xyz = arr[mask, :3]
    if percentile is None:
        low = xyz.min(axis=0)
        high = xyz.max(axis=0)
    else:
        low, high = np.percentile(xyz, [float(percentile[0]), float(percentile[1])], axis=0)
    span = np.maximum(high - low, float(min_span))
    pad = np.maximum(span * float(margin), float(min_span))
    return low - pad, high + pad


def plot_mmwave_points_3d(
    ax,
    points: Any,
    *,
    color: str = "#9ca3af",
    cmap: str | None = None,
    size: float = 2.2,
    alpha: float = 0.28,
    use_intensity: bool = False,
    rasterized: bool = True,
    label: str | None = None,
    zorder: float | None = None,
) -> None:
    """Plot valid mmWave points on a matplotlib 3D axis."""

    arr = as_point_array(points)
    mask = valid_point_mask(arr)
    if not bool(mask.any()):
        return
    valid = arr[mask]
    if use_intensity and arr.shape[1] >= 4 and cmap:
        intensity = normalize_intensity(arr, mask)[mask]
        ax.scatter(
            valid[:, 0],
            valid[:, 1],
            valid[:, 2],
            c=intensity,
            cmap=cmap,
            s=float(size),
            alpha=float(alpha),
            depthshade=False,
            linewidths=0.0,
            rasterized=bool(rasterized),
            label=label,
            zorder=zorder,
        )
    else:
        ax.scatter(
            valid[:, 0],
            valid[:, 1],
            valid[:, 2],
            color=color,
            s=float(size),
            alpha=float(alpha),
            depthshade=False,
            linewidths=0.0,
            rasterized=bool(rasterized),
            label=label,
            zorder=zorder,
        )
