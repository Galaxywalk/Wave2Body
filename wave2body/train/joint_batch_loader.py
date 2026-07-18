# -*- coding: utf-8 -*-
"""Batched tensor loader for joint-only VQ-VAE training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import ConcatDataset, Dataset


@dataclass
class PackedJointsDataset:
    joints: torch.Tensor
    labels: torch.Tensor

    def __post_init__(self) -> None:
        if self.joints.ndim != 3:
            raise ValueError(f"Expected joints [N,J,3], got {tuple(self.joints.shape)}")
        if self.labels.ndim != 1 or self.labels.shape[0] != self.joints.shape[0]:
            raise ValueError("Packed labels must be [N] and match joints length")
        self.joints = self.joints.contiguous()
        self.labels = self.labels.contiguous().to(torch.long)

    def __len__(self) -> int:
        return int(self.joints.shape[0])

    def nbytes(self) -> int:
        return int(self.joints.numel() * self.joints.element_size() + self.labels.numel() * self.labels.element_size())

    def batch(self, indices: torch.Tensor, *, pin_memory: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        idx = indices.to(device=self.joints.device, dtype=torch.long, non_blocking=True)
        joints = self.joints.index_select(0, idx)
        labels = self.labels.index_select(0, idx)
        if pin_memory and joints.device.type == "cpu" and torch.cuda.is_available():
            joints = joints.pin_memory()
            labels = labels.pin_memory()
        return joints, labels


class JointTensorBatchLoader:
    """Yield joint batches by vectorized tensor indexing instead of per-sample collation."""

    def __init__(
        self,
        dataset: PackedJointsDataset,
        indices: torch.Tensor,
        *,
        batch_size: int,
        shuffle: bool,
        drop_last: bool = False,
        pin_memory: bool = False,
        generator: torch.Generator | None = None,
    ):
        self.dataset = dataset
        self.indices = indices.to(dtype=torch.long, device="cpu").contiguous()
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.pin_memory = bool(pin_memory)
        self.generator = generator
        self.num_workers = 0
        self.persistent_workers = False
        self.prefetch_factor = None

    def __len__(self) -> int:
        n = int(self.indices.shape[0])
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = int(self.indices.shape[0])
        if self.shuffle:
            order = torch.randperm(n, generator=self.generator)
            indices = self.indices.index_select(0, order)
        else:
            indices = self.indices

        limit = (n // self.batch_size) * self.batch_size if self.drop_last else n
        for start in range(0, limit, self.batch_size):
            end = min(start + self.batch_size, n)
            if self.drop_last and end - start < self.batch_size:
                break
            yield self.dataset.batch(indices[start:end], pin_memory=self.pin_memory)


def pack_joints_dataset(dataset: Dataset) -> PackedJointsDataset | None:
    """Pack a joint-only ConcatDataset into one tensor-backed dataset."""
    datasets = list(dataset.datasets) if isinstance(dataset, ConcatDataset) else [dataset]
    joints_tensors: list[torch.Tensor] = []
    label_tensors: list[torch.Tensor] = []
    for fallback_label, ds in enumerate(datasets):
        joints = getattr(ds, "joints_data", None)
        if not isinstance(joints, torch.Tensor) or joints.device.type != "cpu":
            return None
        joints = joints[:, :, :3].to(dtype=torch.float32)
        label = int(getattr(ds, "label", fallback_label))
        joints_tensors.append(joints)
        label_tensors.append(torch.full((int(joints.shape[0]),), label, dtype=torch.long))

    if not joints_tensors:
        return None
    if len(joints_tensors) == 1:
        joints = joints_tensors[0].contiguous()
        labels = label_tensors[0]
    else:
        joints = torch.cat(joints_tensors, dim=0).contiguous()
        labels = torch.cat(label_tensors, dim=0)
    return PackedJointsDataset(joints=joints, labels=labels)


__all__ = ["JointTensorBatchLoader", "PackedJointsDataset", "pack_joints_dataset"]
