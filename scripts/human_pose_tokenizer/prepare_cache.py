#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Warm joint-only caches for the main human pose tokenizer configs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.human_pose_tokenizer._configs import print_configs, resolve_config  # noqa: E402
from wave2body.dataset.human_body import create_dataset  # noqa: E402
from wave2body.train.common import load_config  # noqa: E402
from wave2body.train.human_pose_tokenizer import resolve_enabled_datasets  # noqa: E402


def _filter_datasets(enabled_datasets: list[dict], names: list[str] | None) -> list[dict]:
    if not names:
        return enabled_datasets
    requested = set(names)
    selected = [ds for ds in enabled_datasets if ds.get("name") in requested]
    missing = sorted(requested - {ds.get("name") for ds in selected})
    if missing:
        raise ValueError(f"Requested dataset names not found in config: {', '.join(missing)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare joint tensor caches for human pose tokenizer training.")
    parser.add_argument("--config", default="pretrain", help="Config alias or YAML path.")
    parser.add_argument("--list-configs", action="store_true", help="List config aliases and exit.")
    parser.add_argument("--list-datasets", action="store_true", help="List resolved dataset names and exit.")
    parser.add_argument("--only", action="append", help="Prepare one dataset by name. Repeat for multiple datasets.")
    parser.add_argument("--dataset-device", choices=["cpu", "cuda"], help="Override joint tensor device.")
    parser.add_argument("--no-cache", action="store_true", help="Build datasets without writing/reading cache files.")
    args = parser.parse_args()

    if args.list_configs:
        print_configs()
        return

    config_path = resolve_config(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path)
    if args.dataset_device:
        config.setdefault("data", {})["dataset_device"] = args.dataset_device

    data_config = config.get("data", {})
    enabled_datasets = _filter_datasets(resolve_enabled_datasets(config), args.only)
    if not enabled_datasets:
        raise ValueError("No datasets selected. Check config or --only values.")

    if args.list_datasets:
        for ds in enabled_datasets:
            print(ds["name"])
        return

    dataset_device = data_config.get("dataset_device", "cpu")
    use_cache = (not args.no_cache) and bool(data_config.get("use_cache", True))
    cache_dir = data_config.get("cache_dir")

    print(f"Config: {config_path}")
    print(f"Cache dir: {cache_dir}")
    print(f"Dataset device: {dataset_device}")
    print(f"Use cache: {use_cache}")
    print("\n--- Cache Dataset List ---")
    for ds in enabled_datasets:
        print(f"  {ds['name']}: {ds['path']}")

    dataset, info = create_dataset(
        enabled_datasets,
        config["model"]["num_joints"],
        dataset_device,
        use_cache=use_cache,
        cache_dir=cache_dir,
        cache_build_workers=data_config.get("cache_build_workers"),
        cache_build_chunk_size=data_config.get("cache_build_chunk_size"),
    )
    print(f"\nPrepared {len(dataset)} samples across {len(enabled_datasets)} dataset(s).")
    if info:
        for name, count in info.items():
            print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
