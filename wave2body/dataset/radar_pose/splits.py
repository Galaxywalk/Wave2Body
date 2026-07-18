# -*- coding: utf-8 -*-
"""Deterministic dataset split helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


def to_sequence_id(value) -> int:
    if isinstance(value, int):
        return value
    text = str(value)
    if text.startswith("sequence_"):
        return int(text.split("_")[-1])
    return int(text)


def list_train_sequence_ids(dataset_path: str | Path) -> list[int]:
    train_dir = Path(dataset_path) / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    seq_ids = [
        to_sequence_id(p.name)
        for p in sorted(train_dir.iterdir())
        if p.is_dir() and p.name.startswith("sequence_")
    ]
    if not seq_ids:
        raise RuntimeError(f"No training sequences found in {train_dir}")
    return sorted(seq_ids)


def split_sequence_ids(
    all_seq_ids: Iterable[int],
    val_ratio: float = 0.15,
    split_seed: int = 42,
    val_sequences: list | None = None,
) -> tuple[list[int], list[int]]:
    all_ids = sorted(set(int(x) for x in all_seq_ids))
    if val_sequences:
        val_ids = sorted(set(to_sequence_id(x) for x in val_sequences))
        missing = [x for x in val_ids if x not in all_ids]
        if missing:
            raise ValueError(f"Val sequences not found in train split: {missing}")
        train_ids = [x for x in all_ids if x not in val_ids]
    else:
        if not (0.0 < float(val_ratio) < 1.0):
            raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
        rng = np.random.RandomState(split_seed)
        shuffled = np.asarray(all_ids)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * float(val_ratio))))
        val_ids = sorted(int(x) for x in shuffled[:n_val].tolist())
        train_ids = sorted(int(x) for x in shuffled[n_val:].tolist())

    if not train_ids or not val_ids:
        raise RuntimeError(f"Invalid split: train={len(train_ids)}, val={len(val_ids)}")
    overlap = sorted(set(train_ids).intersection(val_ids))
    if overlap:
        raise RuntimeError(f"Train/val overlap detected: {overlap}")
    return train_ids, val_ids


class SequenceSubsetDataset(Dataset):
    """Subset wrapper based on integer indices from an existing dataset."""

    def __init__(self, base_dataset: Dataset, indices: list[int]):
        self.base_dataset = base_dataset
        self.indices = [int(x) for x in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]


def dataset_indices_from_sequences(base_dataset, sequence_ids: list[int]) -> list[int]:
    if not hasattr(base_dataset, "data_list"):
        raise AttributeError("base_dataset does not provide data_list for sequence filtering")
    data_list = base_dataset.data_list
    if "Sequence" not in data_list.columns:
        raise KeyError("base_dataset.data_list has no 'Sequence' column")
    seq_set = set(int(x) for x in sequence_ids)
    mask = data_list["Sequence"].astype(int).isin(seq_set)
    indices = data_list.index[mask].tolist()
    if not indices:
        raise RuntimeError(f"No samples found for sequence_ids={sorted(seq_set)}")
    return [int(x) for x in indices]


def build_train_val_manifest(
    dataset_path: str,
    train_seq_ids: list[int],
    val_seq_ids: list[int],
    train_num_samples: int,
    val_num_samples: int,
    split_seed: int,
    val_ratio: float,
) -> dict:
    return {
        "dataset_path": dataset_path,
        "train_sequences": [f"sequence_{x}" for x in sorted(train_seq_ids)],
        "val_sequences": [f"sequence_{x}" for x in sorted(val_seq_ids)],
        "train_num_samples": int(train_num_samples),
        "val_num_samples": int(val_num_samples),
        "split_seed": int(split_seed),
        "val_ratio": float(val_ratio),
        "note": "Validation is from wave2body.train sequences only; official test split reserved for final evaluation.",
    }


def save_manifest(manifest: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")


def make_worker_init_fn(base_seed: int):
    base_seed = int(base_seed)

    def _worker_init_fn(worker_id: int) -> None:
        worker_seed = (base_seed + int(worker_id)) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _worker_init_fn
