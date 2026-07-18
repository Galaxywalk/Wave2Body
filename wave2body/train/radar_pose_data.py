# -*- coding: utf-8 -*-
"""Shared radar-pose dataloader builders for training pipelines."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

try:
    from ..dataset.radar_pose import M4HumanLMDBPoseDataset, MMBodyDataset
    from ..dataset.radar_pose.splits import (
        SequenceSubsetDataset,
        build_train_val_manifest,
        dataset_indices_from_sequences,
        list_train_sequence_ids,
        make_worker_init_fn,
        save_manifest,
        split_sequence_ids,
    )
    from ..utils.paths import resolve_dataset_uri
    from ..utils.runtime_utils import dataset_cache_mode
except ImportError:  # Support scripts that put src/ directly on sys.path.
    from wave2body.dataset.radar_pose import M4HumanLMDBPoseDataset, MMBodyDataset
    from wave2body.dataset.radar_pose.splits import (
        SequenceSubsetDataset,
        build_train_val_manifest,
        dataset_indices_from_sequences,
        list_train_sequence_ids,
        make_worker_init_fn,
        save_manifest,
        split_sequence_ids,
    )
    from wave2body.utils.paths import resolve_dataset_uri
    from wave2body.utils.runtime_utils import dataset_cache_mode


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    train_count: int
    val_count: int
    split_manifest: dict[str, Any] = field(default_factory=dict)
    test_loader: DataLoader | None = None
    test_count: int = 0


class LimitedDataLoader:
    """Bound a dataloader for smoke runs while preserving normal loader attributes."""

    def __init__(self, loader: DataLoader, max_batches: int | None):
        self.loader = loader
        self.max_batches = int(max_batches) if max_batches is not None else None
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size

    def __iter__(self):
        iterator = iter(self.loader)
        if self.max_batches is None:
            return iterator
        return itertools.islice(iterator, max(0, self.max_batches))

    def __len__(self) -> int:
        if self.max_batches is None:
            return len(self.loader)
        return min(len(self.loader), max(0, self.max_batches))

    def __getattr__(self, name: str):
        return getattr(self.loader, name)


def dataset_name(config: dict) -> str:
    return str(config.get("dataset", {}).get("name", "mmbody")).strip().lower()


def dataset_class_from_config(config: dict):
    name = dataset_name(config).lower()
    dataset_classes = {
        "mmbody": MMBodyDataset,
        "m4human": M4HumanLMDBPoseDataset,
    }
    if name not in dataset_classes:
        raise ValueError("Unsupported dataset.name={!r}; expected 'mmbody' or 'm4human'".format(dataset_name(config)))
    return dataset_classes[name]


def create_radar_pose_dataset(config: dict, split: str, device: str | None = "cpu") -> Dataset:
    """Create a dataset that returns ``mmwave_points`` and ``target_poses``."""
    dataset_cls = dataset_class_from_config(config)
    if dataset_cls is M4HumanLMDBPoseDataset:
        return dataset_cls(config=config, split=split, device=device)
    return dataset_cls(config=config, split=split, vqvae_model=None, device=device)


def _train_val_split_config(config: dict) -> dict:
    training_cfg = config.get("training", {})
    if isinstance(training_cfg.get("train_val_split"), dict):
        return dict(training_cfg["train_val_split"])
    validation_cfg = config.get("validation", {})
    return dict(validation_cfg) if isinstance(validation_cfg, dict) else {}


def _build_loader(
    dataset: Dataset,
    *,
    training_cfg: dict,
    shuffle: bool,
    drop_last: bool,
    generator: torch.Generator | None,
    worker_init_fn,
) -> DataLoader:
    batch_size = int(training_cfg.get("batch_size", 32))
    num_workers = int(training_cfg.get("num_workers", 4))
    pin_memory = bool(training_cfg.get("pin_memory", True))
    persistent_workers = bool(training_cfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = int(training_cfg.get("prefetch_factor", 2))

    cache_mode = dataset_cache_mode(dataset)
    if cache_mode == "gpu_tensor":
        if num_workers > 0:
            print("[Runtime] data.cache.mode=gpu_tensor requires num_workers=0; forcing num_workers=0.")
        num_workers = 0
        pin_memory = False
        persistent_workers = False

    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
        "drop_last": drop_last,
        "worker_init_fn": worker_init_fn,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def _build_seed_state(training_cfg: dict) -> tuple[torch.Generator | None, Any]:
    seed = training_cfg.get("seed", training_cfg.get("dataloader_seed", None))
    if seed is None:
        return None, None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator, make_worker_init_fn(int(seed))


def _build_mmbody_train_val(config: dict, checkpoint_dir: str | Path | None) -> tuple[Dataset, Dataset, dict]:
    split_cfg = _train_val_split_config(config)
    base_train = create_radar_pose_dataset(config, split="train", device="cpu")
    dataset_path = resolve_dataset_uri(str(config.get("dataset_path", "")))
    if not dataset_path:
        raise ValueError("dataset_path must be set for MMBody held-out train/val splitting")

    all_seq_ids = list_train_sequence_ids(dataset_path)
    val_ratio = float(split_cfg.get("val_ratio", 0.15))
    split_seed = int(split_cfg.get("split_seed", 42))
    val_sequences = split_cfg.get("val_sequences", []) or []
    train_seq_ids, val_seq_ids = split_sequence_ids(
        all_seq_ids=all_seq_ids,
        val_ratio=val_ratio,
        split_seed=split_seed,
        val_sequences=val_sequences,
    )

    train_indices = dataset_indices_from_sequences(base_train, train_seq_ids)
    val_indices = dataset_indices_from_sequences(base_train, val_seq_ids)
    train_dataset = SequenceSubsetDataset(base_train, train_indices)
    val_dataset = SequenceSubsetDataset(base_train, val_indices)
    manifest = build_train_val_manifest(
        dataset_path=dataset_path,
        train_seq_ids=train_seq_ids,
        val_seq_ids=val_seq_ids,
        train_num_samples=len(train_dataset),
        val_num_samples=len(val_dataset),
        split_seed=split_seed,
        val_ratio=val_ratio,
    )
    manifest["protocol"] = split_cfg.get("protocol", "heldout_train_sequences")

    if checkpoint_dir is not None:
        save_manifest(manifest, Path(checkpoint_dir) / "split_manifest.json")
    return train_dataset, val_dataset, manifest


def _parse_string_list(value: Any, *, default: list[str]) -> list[str]:
    if value is None:
        items = list(default)
    elif isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        items = [part.strip() for part in str(value).split(",") if part.strip()]
    if not items:
        raise ValueError("Expected at least one non-empty item")
    return items


def _m4human_split_id(config: dict) -> str:
    split_cfg = config.get("m4human", {}).get("split", "s2")
    if isinstance(split_cfg, (list, tuple)) or "," in str(split_cfg):
        raise ValueError(
            "m4human.split is a split protocol id, not a data source list. "
            "Use one of s1/s2/s3, and use training.train_subsets to combine train/val/test."
        )
    split_id = str(split_cfg).strip()
    if split_id not in {"s1", "s2", "s3"}:
        raise ValueError(f"m4human.split must be one of s1/s2/s3, got {split_id!r}")
    return split_id


def _m4human_train_subsets(config: dict) -> list[str]:
    subsets = _parse_string_list(config.get("training", {}).get("train_subsets"), default=["train"])
    allowed = {"train", "val", "test"}
    invalid = [subset for subset in subsets if subset not in allowed]
    if invalid:
        raise ValueError(f"training.train_subsets must use only {sorted(allowed)}, got {invalid}")
    return subsets


def _build_m4human_train_dataset(config: dict) -> tuple[Dataset, dict[str, Any]]:
    _m4human_split_id(config)
    subsets = _m4human_train_subsets(config)
    datasets = [create_radar_pose_dataset(config, split=subset, device="cpu") for subset in subsets]
    counts_by_subset = {subset: len(dataset) for subset, dataset in zip(subsets, datasets)}
    if len(datasets) == 1:
        train_dataset: Dataset = datasets[0]
    else:
        train_dataset = ConcatDataset(datasets)
    return train_dataset, {
        "split": config.get("m4human", {}).get("split"),
        "train_subsets": subsets,
        "counts_by_subset": counts_by_subset,
        "train_num_samples": len(train_dataset),
    }


def build_radar_pose_dataloaders(
    config: dict,
    *,
    checkpoint_dir: str | Path | None = None,
    include_test: bool = False,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    max_test_batches: int | None = None,
) -> DataBundle:
    """Build train/val/test loaders for MMBody or M4Human."""
    training_cfg = config.get("training", {})
    split_cfg = _train_val_split_config(config)
    dataset_cls = dataset_class_from_config(config)
    split_enabled = bool(split_cfg.get("enabled", True))

    if dataset_cls is MMBodyDataset and split_enabled:
        train_dataset, val_dataset, manifest = _build_mmbody_train_val(config, checkpoint_dir)
    elif dataset_cls is M4HumanLMDBPoseDataset:
        val_split = str(split_cfg.get("val_split", "val"))
        train_dataset, train_manifest = _build_m4human_train_dataset(config)
        val_dataset = create_radar_pose_dataset(config, split=val_split, device="cpu")
        manifest = {
            "dataset_name": dataset_name(config),
            "dataset_path": str(config.get("dataset_path", "")),
            "train_split": train_manifest["train_subsets"],
            "val_split": val_split,
            "train_num_samples": len(train_dataset),
            "val_num_samples": len(val_dataset),
            "split": train_manifest["split"],
            "m4human_train": train_manifest,
        }
    else:
        val_split = str(split_cfg.get("val_split", "val"))
        train_dataset = create_radar_pose_dataset(config, split="train", device="cpu")
        val_dataset = create_radar_pose_dataset(config, split=val_split, device="cpu")
        manifest = {
            "dataset_name": dataset_name(config),
            "dataset_path": str(config.get("dataset_path", "")),
            "train_split": "train",
            "val_split": val_split,
            "train_num_samples": len(train_dataset),
            "val_num_samples": len(val_dataset),
        }
        if dataset_cls is M4HumanLMDBPoseDataset:
            manifest.update(
                {
                    "split": config.get("m4human", {}).get("split"),
                }
            )

    generator, worker_init_fn = _build_seed_state(training_cfg)
    train_loader = _build_loader(
        train_dataset,
        training_cfg=training_cfg,
        shuffle=True,
        drop_last=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    val_loader = _build_loader(
        val_dataset,
        training_cfg=training_cfg,
        shuffle=False,
        drop_last=False,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )

    test_loader = None
    test_count = 0
    if include_test:
        test_dataset = create_radar_pose_dataset(config, split="test", device="cpu")
        test_count = len(test_dataset)
        test_loader = _build_loader(
            test_dataset,
            training_cfg=training_cfg,
            shuffle=False,
            drop_last=False,
            generator=generator,
            worker_init_fn=worker_init_fn,
        )
        manifest["test_num_samples"] = int(test_count)

    if checkpoint_dir is not None:
        save_manifest(manifest, Path(checkpoint_dir) / "split_manifest.json")

    print(
        "[Runtime] "
        f"dataset={dataset_name(config)}, batch_size={int(training_cfg.get('batch_size', 32))}, "
        f"workers={int(training_cfg.get('num_workers', 4))}, "
        f"pin_memory={bool(training_cfg.get('pin_memory', True))}, "
        f"prefetch_factor={training_cfg.get('prefetch_factor', 2)}, "
        f"train={len(train_dataset)}, val={len(val_dataset)}, test={test_count}"
    )
    return DataBundle(
        train_loader=LimitedDataLoader(train_loader, max_train_batches),
        val_loader=LimitedDataLoader(val_loader, max_val_batches),
        test_loader=LimitedDataLoader(test_loader, max_test_batches) if test_loader is not None else None,
        train_count=len(train_dataset),
        val_count=len(val_dataset),
        test_count=test_count,
        split_manifest=manifest,
    )
