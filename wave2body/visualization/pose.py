# -*- coding: utf-8 -*-
"""Reusable human-pose plotting utilities."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import matplotlib.patheffects as pe
import numpy as np


SMPL22_BONE_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (2, 5),
    (3, 6),
    (4, 7),
    (5, 8),
    (6, 9),
    (7, 10),
    (8, 11),
    (9, 12),
    (12, 13),
    (12, 14),
    (12, 15),
    (13, 16),
    (14, 17),
    (16, 18),
    (17, 19),
    (18, 20),
    (19, 21),
)


def as_pose_array(pose: Any, *, num_joints: int = 22) -> np.ndarray:
    """Return a finite ``[J,3]`` float array for plotting."""

    arr = np.asarray(pose, dtype=np.float32)
    if arr.shape != (int(num_joints), 3):
        raise ValueError(f"Expected pose shape ({int(num_joints)}, 3), got {arr.shape}")
    return arr


def pose_bounds(
    poses: Iterable[Any],
    *,
    margin: float = 0.15,
    min_span: float = 1.0e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute padded xyz bounds for one or more poses."""

    arrays = [np.asarray(pose, dtype=np.float32).reshape(-1, 3) for pose in poses]
    if not arrays:
        raise ValueError("pose_bounds requires at least one pose")
    points = np.concatenate(arrays, axis=0)
    finite = np.isfinite(points).all(axis=1)
    if not bool(finite.any()):
        return np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)
    valid = points[finite]
    low = valid.min(axis=0)
    high = valid.max(axis=0)
    span = np.maximum(high - low, float(min_span))
    pad = np.maximum(span * float(margin), float(min_span))
    return low - pad, high + pad


def set_axes_equal_3d(ax, *, bounds: tuple[np.ndarray, np.ndarray] | None = None) -> None:
    """Set equal xyz scaling on a matplotlib 3D axis."""

    if bounds is None:
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=np.float32)
        low = limits[:, 0]
        high = limits[:, 1]
    else:
        low, high = (np.asarray(bounds[0], dtype=np.float32), np.asarray(bounds[1], dtype=np.float32))
    center = (low + high) * 0.5
    radius = float(np.max(high - low) * 0.5)
    if radius <= 0.0 or not np.isfinite(radius):
        radius = 0.5
    ax.set_xlim3d(center[0] - radius, center[0] + radius)
    ax.set_ylim3d(center[1] - radius, center[1] + radius)
    ax.set_zlim3d(center[2] - radius, center[2] + radius)


def plot_skeleton_3d(
    ax,
    pose: Any,
    *,
    edges: Iterable[tuple[int, int]] = SMPL22_BONE_EDGES,
    color: str = "#1f77b4",
    linewidth: float = 1.8,
    linestyle: str = "-",
    joint_size: float = 12.0,
    joint_edge_color: str = "white",
    joint_edge_width: float = 0.45,
    line_outline: bool = True,
    alpha: float = 1.0,
    label: str | None = None,
    zorder: float | None = None,
) -> None:
    """Plot a 22-joint skeleton on a matplotlib 3D axis."""

    joints = as_pose_array(pose, num_joints=22)
    finite = np.isfinite(joints).all(axis=1)
    first_line = True
    for parent, child in edges:
        if parent >= joints.shape[0] or child >= joints.shape[0]:
            raise ValueError(f"Skeleton edge {(parent, child)} is out of range for {joints.shape[0]} joints")
        if not bool(finite[parent] and finite[child]):
            continue
        pts = joints[[parent, child]]
        line = ax.plot(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            color=color,
            linewidth=float(linewidth),
            linestyle=linestyle,
            alpha=float(alpha),
            label=label if first_line else None,
            zorder=zorder,
        )[0]
        if line_outline:
            line.set_path_effects(
                [
                    pe.Stroke(linewidth=float(linewidth) + 0.8, foreground="#282828", alpha=0.20),
                    pe.Normal(),
                ]
            )
        first_line = False
    if bool(finite.any()):
        valid = joints[finite]
        ax.scatter(
            valid[:, 0],
            valid[:, 1],
            valid[:, 2],
            s=float(joint_size),
            color=color,
            edgecolors=joint_edge_color,
            linewidths=float(joint_edge_width),
            alpha=float(alpha),
            depthshade=False,
            zorder=zorder,
        )
