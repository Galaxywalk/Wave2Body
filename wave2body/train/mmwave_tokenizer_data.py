# -*- coding: utf-8 -*-
"""Dataloader entry points for mmWave tokenizer training and diagnostics."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from .radar_pose_data import DataBundle, build_radar_pose_dataloaders, create_radar_pose_dataset


def create_mmwave_tokenizer_dataset(config: dict, split: str, device: str | None = "cpu") -> Dataset:
    """Create the radar-pose dataset used by mmWave tokenizer jobs."""
    return create_radar_pose_dataset(config=config, split=split, device=device)


def build_mmwave_tokenizer_dataloaders(
    config: dict,
    *,
    checkpoint_dir: str | Path | None = None,
    include_test: bool = False,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    max_test_batches: int | None = None,
) -> DataBundle:
    """Build train/val/test loaders for mmWave tokenizer jobs."""
    return build_radar_pose_dataloaders(
        config,
        checkpoint_dir=checkpoint_dir,
        include_test=include_test,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        max_test_batches=max_test_batches,
    )


__all__ = [
    "DataBundle",
    "build_mmwave_tokenizer_dataloaders",
    "create_mmwave_tokenizer_dataset",
]
