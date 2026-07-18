# -*- coding: utf-8 -*-
"""Factory helpers for human-body joint-only datasets."""

from __future__ import annotations

from typing import List, Tuple

from torch.utils.data import ConcatDataset

from . import amass as _amass  # noqa: F401
from . import m4human as _m4human  # noqa: F401
from . import mmbody as _mmbody  # noqa: F401
from .registry import DATASET_REGISTRY


def create_dataset(
    dataset_configs: List[dict],
    num_joints: int,
    device: str,
    use_cache: bool = True,
    cache_dir: str | None = None,
    cache_build_workers: int | None = None,
    cache_build_chunk_size: int | None = None,
) -> Tuple[ConcatDataset, dict]:
    """Create and concatenate multiple datasets from config."""
    datasets = []
    label_to_name = {}

    print("\n--- Creating Datasets ---")

    for i, config in enumerate(dataset_configs):
        dtype, name, path = config.get("type"), config.get("name"), config.get("path")
        if not all([dtype, name, path]) or dtype not in DATASET_REGISTRY:
            continue

        print(f"-> {name} ({dtype})")
        kwargs = dict(config.get("params", {}))
        kwargs["use_cache"] = use_cache
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if cache_build_workers is not None:
            kwargs.setdefault("cache_build_workers", cache_build_workers)
        if cache_build_chunk_size is not None:
            kwargs.setdefault("cache_build_chunk_size", cache_build_chunk_size)

        try:
            ds = DATASET_REGISTRY[dtype](root_path=path, num_joints=num_joints, device=device, label=i, name=name, **kwargs)
            if len(ds) > 0:
                datasets.append(ds)
                label_to_name[i] = name
            else:
                print(f"[WARN] {name} is empty, skipping.")
        except Exception as e:
            raise RuntimeError(f"Failed to load configured dataset {name!r} ({dtype}) from {path}") from e

    if not datasets:
        raise RuntimeError("No valid datasets loaded.")

    combined = ConcatDataset(datasets)
    print(f"\n--- Summary: {len(combined):,} total samples ---")
    for ds in datasets:
        print(f"  Label {ds.label}: {ds.name} ({len(ds):,})")

    return combined, {"label_to_name": label_to_name, "num_datasets": len(label_to_name)}


__all__ = ["create_dataset"]
