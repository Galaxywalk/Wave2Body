# -*- coding: utf-8 -*-
"""Shared root-alignment options for radar-to-pose datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class RootAlignmentConfig:
    normalized: bool
    input_root_normalized: bool
    target_root_normalized: bool
    input_root_crop: bool
    input_root_crop_x_range: tuple[float, float]
    input_root_crop_y_range: tuple[float, float]
    input_root_crop_z_range: tuple[float, float]


def _range2(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    raw = default if value is None else value
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"Root-alignment crop range must be a two-item list/tuple, got {raw!r}")
    low, high = float(raw[0]), float(raw[1])
    if low > high:
        raise ValueError(f"Root-alignment crop range must satisfy low <= high, got {raw!r}")
    return low, high


def resolve_root_alignment_config(
    data_format: Mapping[str, Any],
    *,
    default_normalized: bool = True,
    default_input_root_normalized: bool | None = None,
    default_target_root_normalized: bool | None = None,
    default_input_root_crop: bool | None = None,
    default_crop_range: tuple[float, float] = (-1.0, 1.0),
) -> RootAlignmentConfig:
    """Resolve common input/target root-alignment switches from ``data_format``."""

    normalized = bool(data_format.get("normalized", default_normalized))
    input_default = normalized if default_input_root_normalized is None else bool(default_input_root_normalized)
    target_default = normalized if default_target_root_normalized is None else bool(default_target_root_normalized)
    input_root_normalized = bool(data_format.get("input_root_normalized", input_default))
    target_root_normalized = bool(data_format.get("target_root_normalized", target_default))
    crop_default = input_root_normalized if default_input_root_crop is None else bool(default_input_root_crop)
    input_root_crop = bool(data_format.get("input_root_crop", crop_default))
    return RootAlignmentConfig(
        normalized=normalized,
        input_root_normalized=input_root_normalized,
        target_root_normalized=target_root_normalized,
        input_root_crop=input_root_crop,
        input_root_crop_x_range=_range2(data_format.get("input_root_crop_x_range"), default_crop_range),
        input_root_crop_y_range=_range2(data_format.get("input_root_crop_y_range"), default_crop_range),
        input_root_crop_z_range=_range2(data_format.get("input_root_crop_z_range"), default_crop_range),
    )


def crop_point_cloud(
    points: np.ndarray,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    xyz_offset: np.ndarray | None = None,
) -> np.ndarray:
    """Crop by xyz ranges, optionally after subtracting an offset for the mask only."""

    if points.size == 0:
        return points
    local_xyz = points[:, :3]
    if xyz_offset is not None:
        local_xyz = local_xyz - np.asarray(xyz_offset[:3], dtype=np.float32).reshape(1, 3)
    mask = (
        (local_xyz[:, 0] >= x_range[0])
        & (local_xyz[:, 0] <= x_range[1])
        & (local_xyz[:, 1] >= y_range[0])
        & (local_xyz[:, 1] <= y_range[1])
        & (local_xyz[:, 2] >= z_range[0])
        & (local_xyz[:, 2] <= z_range[1])
    )
    return points[mask]


def apply_input_root_alignment(
    points: np.ndarray,
    *,
    root: np.ndarray | None,
    cfg: RootAlignmentConfig,
) -> np.ndarray:
    """Apply input root-normalization and optional human-ROI crop."""

    if points.size == 0:
        return points

    aligned = points
    root_offset = None if root is None else np.asarray(root[:3], dtype=np.float32).reshape(1, 3)
    crop_offset = root_offset
    if cfg.input_root_normalized and root_offset is not None:
        aligned = points.copy()
        aligned[:, :3] -= root_offset
        crop_offset = None

    if not cfg.input_root_crop:
        return aligned

    return crop_point_cloud(
        aligned,
        x_range=cfg.input_root_crop_x_range,
        y_range=cfg.input_root_crop_y_range,
        z_range=cfg.input_root_crop_z_range,
        xyz_offset=crop_offset,
    )
