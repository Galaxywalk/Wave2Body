# -*- coding: utf-8 -*-
"""AMASS joint-only dataset loader."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch

from ...utils.paths import resolve_dataset_uri
from .base import BaseJointsDataset
from .registry import register_dataset
from .transforms import PreprocessingUtils


@register_dataset("amass")
class AMASSJointsDataset(BaseJointsDataset):
    """AMASS dataset loader."""

    def __init__(self, root_path: str, num_joints: int, device: str, label: int, name: str, **kwargs):
        root = Path(resolve_dataset_uri(str(root_path)))
        packed_path = root if root.is_file() else Path(f"{root}_joints{num_joints}.pt")
        self._packed_tensor_path = packed_path if packed_path.is_file() else None
        super().__init__(str(self._packed_tensor_path or root), num_joints, device, label, name, **kwargs)

    def _load_with_cache(self) -> torch.Tensor:
        if self._packed_tensor_path is not None:
            payload = torch.load(self._packed_tensor_path, map_location="cpu", weights_only=False)
            joints = payload["joints"] if isinstance(payload, dict) and "joints" in payload else payload
            if not isinstance(joints, torch.Tensor):
                joints = torch.as_tensor(joints, dtype=torch.float32)
            joints = joints[:, : self.num_joints, :3].to(dtype=torch.float32, device=self.device)
            if self.normalized:
                joints = joints - joints[:, 0:1, :]
            print(f"[INFO] Loaded {len(joints)} frames from packed AMASS tensor: {self._packed_tensor_path}")
            return joints
        return super()._load_with_cache()

    def _get_file_paths(self) -> List[Path]:
        if not self.root_path.exists():
            print(f"[WARN] AMASS dir not found: {self.root_path}")
            return []
        return sorted(self.root_path.rglob(f"*joints{self.num_joints}.npz"))

    def preprocess(self, joints: np.ndarray) -> np.ndarray:
        if self.normalized:
            return PreprocessingUtils.pelvis_normalization(joints)
        return joints


__all__ = ["AMASSJointsDataset"]
