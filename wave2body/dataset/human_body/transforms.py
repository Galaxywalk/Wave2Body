# -*- coding: utf-8 -*-
"""Pose-coordinate transforms for human-body joint-only datasets."""

from __future__ import annotations

import numpy as np


class PreprocessingUtils:
    """Common preprocessing utilities for pose data."""

    @staticmethod
    def pelvis_normalization(joints: np.ndarray) -> np.ndarray:
        """Normalize joints by subtracting pelvis position. Supports (T,J,3) or (J,3)."""
        if joints.ndim == 3:
            pelvis = joints[:, 0:1, :]
        elif joints.ndim == 2:
            pelvis = joints[0:1, :]
        else:
            raise ValueError(f"Unsupported joints shape: {joints.shape}")
        return joints - pelvis


__all__ = ["PreprocessingUtils"]
