#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train the fixed hist4 absolute PointTransformer HPE baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "wave2body" / "configs" / "hpe_point_transformer"
CONFIG_ALIASES = {
    "m4human_absolute_hist4": CONFIG_DIR / "m4human_absolute_hist4_s2_bf16.yaml",
    "m4human_absolute_hist4_no_rootnorm": CONFIG_DIR / "m4human_absolute_hist4_no_rootnorm_a100.yaml",
    "mmbody_noinput_rootnorm": CONFIG_DIR / "mmbody_pt_noinput_rootnorm.yaml",
    "mmbody_noinput_rootnorm_fastcache": CONFIG_DIR / "mmbody_pt_noinput_rootnorm_fastcache.yaml",
    "mmbody_noinput_absolute_fastcache": CONFIG_DIR / "mmbody_pt_noinput_abs_fastcache.yaml",
    "mmbody_noinput_absolute_fastcache_scratch": CONFIG_DIR / "mmbody_pt_noinput_abs_fastcache_scratch_a100.yaml",
}
VALID_SPLIT_IDS = ("s1", "s2", "s3")
EXPERIMENT_NAME = "hpe_point_transformer"
FIXED_FRAME_MODE = "hist4"
FIXED_TEMPORAL_OFFSETS = [-3, -2, -1, 0]
COORD_TRANSFORM = "m4human_radar_absolute_z_shifted_no_root_normalization"


def _preparse_gpus(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpus")
    args, _ = parser.parse_known_args(argv)
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)


_preparse_gpus(sys.argv[1:])

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wave2body.train.common import load_config  # noqa: E402
from wave2body.train.pipelines.pt_pose import run_pt_pose  # noqa: E402
from wave2body.utils.paths import artifact_root  # noqa: E402


def resolve_config(value: str) -> Path:
    if value not in CONFIG_ALIASES:
        known = ", ".join(sorted(CONFIG_ALIASES))
        raise ValueError(f"Unknown HPE PointTransformer config {value!r}; expected one of: {known}")
    return CONFIG_ALIASES[value]


def print_configs() -> None:
    for name, path in CONFIG_ALIASES.items():
        print(f"{name}: {path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="m4human_absolute_hist4", choices=sorted(CONFIG_ALIASES), help="Config alias.")
    parser.add_argument("--list-configs", action="store_true", help="List config aliases and exit.")
    parser.add_argument("--split-id", choices=VALID_SPLIT_IDS, help="M4Human split id. Defaults to config value.")
    parser.add_argument("--device", default="cuda", help="cuda/cpu")
    parser.add_argument("--gpus", help='CUDA_VISIBLE_DEVICES value, e.g. "0" or "0,1".')
    parser.add_argument("--resume", help="Resume a full HPE PointTransformer checkpoint.")
    parser.add_argument("--work-dir", help="Explicit output directory.")
    parser.add_argument("--run-name", help="Override run name.")
    parser.add_argument("--run-tag", default="", help="Optional suffix appended to generated run name.")
    parser.add_argument("--seed", type=int, help="Override training seed and append it to the run name.")
    parser.add_argument("--batch-size", type=int, help="Runtime batch-size override for memory-constrained GPUs.")
    parser.add_argument("--num-workers", type=int, help="Runtime-only dataloader worker override.")
    parser.add_argument("--smoke", action="store_true", help="One epoch and one train/val/test batch.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved run plan without training.")
    return parser


def _apply_experiment_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    training = config.setdefault("training", {})
    dataset_name = str(config.get("dataset", {}).get("name", "")).strip().lower()
    is_mmbody = dataset_name == "mmbody"
    m4human = config.setdefault("m4human", {}) if not is_mmbody else config.get("m4human", {})
    data_format = config.setdefault("data_format", {})
    evaluation = config.setdefault("evaluation", {})

    seed_override = getattr(args, "seed", None)
    if seed_override is not None:
        seed = int(seed_override)
        config["random_seed"] = seed
        training["seed"] = seed

    batch_size_override = getattr(args, "batch_size", None)
    if batch_size_override is not None:
        batch_size = int(batch_size_override)
        if batch_size <= 0:
            raise ValueError("--batch-size must be > 0")
        training["batch_size"] = batch_size

    if is_mmbody:
        if args.split_id:
            raise ValueError("--split-id is only valid for M4Human PointTransformer configs.")
        training["stage"] = "hpe_point_transformer"
        training["stage_display_name"] = "HPE PointTransformer"
        training["metric_namespace"] = "hpe_point_transformer"
        training.setdefault(
            "train_val_split",
            {"enabled": True, "protocol": "heldout_train_sequences", "val_ratio": 0.15, "split_seed": 42},
        )
        evaluation["mpjpe_only"] = True
        evaluation["m4human_action_groups"] = False
        evaluation.setdefault("primary_metric", "mpjpe")
        evaluation.setdefault("root_index", 0)
    else:
        split_id = args.split_id or str(m4human.get("split", "s2"))
        m4human["split"] = split_id
        m4human["frame_mode"] = FIXED_FRAME_MODE
        m4human["temporal_offsets"] = list(FIXED_TEMPORAL_OFFSETS)
        m4human["temporal_window"] = len(FIXED_TEMPORAL_OFFSETS)
        m4human["pose_source"] = "param_joints"
        m4human["normalize_z"] = True

        config["coord_transform"] = COORD_TRANSFORM
        data_format["merge_nframes"] = len(FIXED_TEMPORAL_OFFSETS)
        data_format["normalized"] = False
        evaluation["mpjpe_only"] = True
        evaluation["m4human_action_groups"] = True
        evaluation["primary_metric"] = "absolute_mpjpe"
        evaluation["diagnostic_metrics"] = ["absolute_mpjpe", "root_error", "root_aligned_mpjpe", "p_mpjpe"]
        evaluation["compute_p_mpjpe"] = True
        evaluation.setdefault("root_index", 0)

    if args.num_workers is not None:
        workers = int(args.num_workers)
        training["num_workers"] = workers
        if workers <= 0:
            training["pin_memory"] = False
            training["persistent_workers"] = False

    if args.smoke:
        training["epochs"] = 1
        training["batch_size"] = min(int(training.get("batch_size", 2)), 2)
        training["num_workers"] = 0
        training["pin_memory"] = False
        training["persistent_workers"] = False

    if args.run_name:
        training["run_name"] = args.run_name
    elif is_mmbody:
        tag = str(args.run_tag or "")
        if tag and not tag.startswith("_"):
            tag = f"_{tag}"
        smoke_tag = "_smoke" if args.smoke and "_smoke" not in tag else ""
        training["run_name"] = f"{training.get('run_name', 'mmbody_hpe_pt')}{tag}{smoke_tag}"
    else:
        precision = str(training.get("precision", "fp32"))
        batch_size = int(training.get("batch_size", 32))
        epochs = int(training.get("epochs", 25))
        tag = str(args.run_tag or "")
        if tag and not tag.startswith("_"):
            tag = f"_{tag}"
        smoke_tag = "_smoke" if args.smoke and "_smoke" not in tag else ""
        training["run_name"] = (
            f"m4human_hpe_pt_abs_{split_id}_{FIXED_FRAME_MODE}_"
            f"e{epochs}_{precision}_bs{batch_size}{tag}{smoke_tag}"
        )
    if seed_override is not None:
        seed_tag = f"_seed{int(seed_override)}"
        run_name = str(training["run_name"])
        if seed_tag not in run_name:
            training["run_name"] = f"{run_name}{seed_tag}"
    return config


def _batch_limits(args: argparse.Namespace) -> dict[str, int | None]:
    limit = 1 if args.smoke else None
    return {
        "train": limit,
        "val": limit,
        "test": limit,
    }


def _planned_output_root(config: dict[str, Any], work_dir: str | None) -> Path:
    if work_dir:
        return Path(work_dir).expanduser()
    run_name = str(config.get("training", {}).get("run_name", "m4human_hpe_pt_abs"))
    return artifact_root() / "experiments" / EXPERIMENT_NAME / "results" / run_name


def _dry_run_payload(
    config_path: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
    batch_limits: dict[str, int | None],
) -> dict[str, Any]:
    output_root = _planned_output_root(config, args.work_dir)
    return {
        "config": str(config_path),
        "run_name": config["training"]["run_name"],
        "dataset": config.get("dataset", {}).get("name"),
        "split_id": config.get("m4human", {}).get("split"),
        "frame_mode": config.get("m4human", {}).get("frame_mode"),
        "temporal_offsets": config.get("m4human", {}).get("temporal_offsets"),
        "data_format.normalized": config["data_format"]["normalized"],
        "input_root_normalized": config["data_format"].get("input_root_normalized"),
        "input_root_crop": config["data_format"].get("input_root_crop"),
        "input_root_crop_x_range": config["data_format"].get("input_root_crop_x_range"),
        "input_root_crop_y_range": config["data_format"].get("input_root_crop_y_range"),
        "input_root_crop_z_range": config["data_format"].get("input_root_crop_z_range"),
        "target_root_normalized": config["data_format"].get("target_root_normalized"),
        "m4human.normalize_z": config.get("m4human", {}).get("normalize_z"),
        "m4human.pose_source": config.get("m4human", {}).get("pose_source"),
        "coord_transform": config.get("coord_transform"),
        "output_root": str(output_root),
        "checkpoint_dir": str(output_root / "checkpoints"),
        "pretrained_point_transformer": config.get("pretrained", {}).get("point_transformer"),
        "max_batches": batch_limits,
    }


def main() -> None:
    args = _build_parser().parse_args()
    if args.list_configs:
        print_configs()
        return

    config_path = resolve_config(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = _apply_experiment_overrides(load_config(config_path), args)
    batch_limits = _batch_limits(args)
    if args.dry_run:
        print(json.dumps(_dry_run_payload(config_path, config, args, batch_limits), indent=2))
        return

    run_pt_pose(
        config,
        config_path=config_path,
        resume=args.resume,
        work_dir=args.work_dir,
        device=args.device,
        experiment_name=EXPERIMENT_NAME,
        max_train_batches=batch_limits["train"],
        max_val_batches=batch_limits["val"],
        max_test_batches=batch_limits["test"],
    )


if __name__ == "__main__":
    main()
