#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train config-driven HPE baseline ports such as mmDiff and mmMesh."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "wave2body" / "configs" / "hpe_baselines"
CONFIG_ALIASES = {
    "mmdiff_mmbody": CONFIG_DIR / "mmdiff_mmbody.yaml",
    "mmmesh_mmbody": CONFIG_DIR / "mmmesh_mmbody.yaml",
    "mmdiff_m4human_s2": CONFIG_DIR / "mmdiff_m4human_s2.yaml",
    "mmdiff_m4human_s3": CONFIG_DIR / "mmdiff_m4human_s3.yaml",
    "mmmesh_m4human_s2": CONFIG_DIR / "mmmesh_m4human_s2.yaml",
    "mmmesh_m4human_s3": CONFIG_DIR / "mmmesh_m4human_s3.yaml",
}
EXPERIMENT_NAME = "hpe_baselines"


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
from wave2body.train.pipelines.pose_baseline import run_pose_baseline  # noqa: E402
from wave2body.utils.paths import artifact_root  # noqa: E402


def resolve_config(value: str) -> Path:
    if value not in CONFIG_ALIASES:
        known = ", ".join(sorted(CONFIG_ALIASES))
        raise ValueError(f"Unknown HPE baseline config {value!r}; expected one of: {known}")
    return CONFIG_ALIASES[value]


def print_configs() -> None:
    for name, path in CONFIG_ALIASES.items():
        print(f"{name}: {path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="mmdiff_mmbody", choices=sorted(CONFIG_ALIASES), help="Config alias.")
    parser.add_argument("--list-configs", action="store_true", help="List config aliases and exit.")
    parser.add_argument("--device", default="cuda", help="cuda/cpu")
    parser.add_argument("--gpus", help='CUDA_VISIBLE_DEVICES value, e.g. "0" or "0,1".')
    parser.add_argument("--resume", help="Resume a full baseline checkpoint.")
    parser.add_argument("--work-dir", help="Explicit output directory.")
    parser.add_argument("--run-name", help="Override run name.")
    parser.add_argument("--run-tag", default="", help="Optional suffix appended to generated run name.")
    parser.add_argument("--seed", type=int, help="Override training seed and append it to the run name.")
    parser.add_argument("--num-workers", type=int, help="Runtime-only dataloader worker override.")
    parser.add_argument("--epochs", type=int, help="Override epoch count.")
    parser.add_argument("--batch-size", type=int, help="Override batch size.")
    parser.add_argument("--smoke", action="store_true", help="One epoch and one train/val/test batch.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved run plan without touching data.")
    return parser


def _baseline_name(config: dict[str, Any]) -> str:
    return str(config.get("baseline", {}).get("name", config.get("model", {}).get("name", "pose_baseline")))


def _assert_mmbody_absolute_uses_unrooted_io(config: dict[str, Any]) -> None:
    dataset = str(config.get("dataset", {}).get("name", "")).strip().lower()
    baseline = _baseline_name(config).strip().lower()
    if dataset != "mmbody" or baseline not in {"mmdiff", "mm_diff", "mmmesh", "mm_mesh"}:
        return

    fmt = config.get("data_format", {})
    if not isinstance(fmt, dict):
        raise TypeError("MMBody baseline config must define data_format as a mapping")

    required_false = {
        "normalized": fmt.get("normalized", True),
        "input_root_normalized": fmt.get("input_root_normalized", fmt.get("normalized", True)),
        "target_root_normalized": fmt.get("target_root_normalized", fmt.get("normalized", True)),
    }
    enabled = [key for key, value in required_false.items() if bool(value)]
    if enabled:
        raise ValueError(
            f"{baseline} MMBody baseline must use unrooted input/output coordinates; "
            f"set these data_format flags to false: {', '.join(enabled)}"
        )


def _apply_experiment_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    training = config.setdefault("training", {})
    if args.seed is not None:
        seed = int(args.seed)
        config["random_seed"] = seed
        training["seed"] = seed
    training["stage"] = "hpe_baseline"
    training["stage_display_name"] = "HPE Baseline"
    training["metric_namespace"] = _baseline_name(config)
    training.setdefault(
        "train_val_split",
        {"enabled": True, "protocol": "heldout_train_sequences", "val_ratio": 0.15, "split_seed": 42},
    )

    if args.epochs is not None:
        training["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        training["batch_size"] = int(args.batch_size)
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
    else:
        tag = str(args.run_tag or "")
        if tag and not tag.startswith("_"):
            tag = f"_{tag}"
        smoke_tag = "_smoke" if args.smoke and "_smoke" not in tag else ""
        baseline = _baseline_name(config)
        dataset = str(config.get("dataset", {}).get("name", "dataset")).lower()
        base = str(training.get("run_name", f"{dataset}_{baseline}_port"))
        training["run_name"] = f"{base}{tag}{smoke_tag}"
    if args.seed is not None:
        seed_tag = f"_seed{int(args.seed)}"
        run_name = str(training["run_name"])
        if seed_tag not in run_name:
            training["run_name"] = f"{run_name}{seed_tag}"
    _assert_mmbody_absolute_uses_unrooted_io(config)
    return config


def _batch_limits(args: argparse.Namespace) -> dict[str, int | None]:
    limit = 1 if args.smoke else None
    return {"train": limit, "val": limit, "test": limit}


def _planned_output_root(config: dict[str, Any], work_dir: str | None) -> Path:
    if work_dir:
        return Path(work_dir).expanduser()
    run_name = str(config.get("training", {}).get("run_name", "hpe_baseline"))
    return artifact_root() / "experiments" / EXPERIMENT_NAME / "results" / run_name


def _dry_run_payload(config_path: Path, config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    output_root = _planned_output_root(config, args.work_dir)
    return {
        "config": str(config_path),
        "baseline": _baseline_name(config),
        "dataset": config.get("dataset", {}).get("name"),
        "run_name": config.get("training", {}).get("run_name"),
        "stage": config.get("training", {}).get("stage"),
        "batch_size": config.get("training", {}).get("batch_size"),
        "epochs": config.get("training", {}).get("epochs"),
        "device": args.device,
        "gpus": args.gpus,
        "root_alignment": {
            "normalized": bool(config.get("data_format", {}).get("normalized", False)),
            "input_root_normalized": bool(config.get("data_format", {}).get("input_root_normalized", False)),
            "input_root_crop": bool(config.get("data_format", {}).get("input_root_crop", False)),
            "target_root_normalized": bool(config.get("data_format", {}).get("target_root_normalized", False)),
        },
        "output_root": str(output_root),
        "checkpoint_dir": str(output_root / "checkpoints"),
        "max_batches": _batch_limits(args),
        "notes": (
            "Dry-run does not instantiate datasets. MMBody can export optional "
            "SMPL/body-target fields for mmMesh when mesh npz files contain them; "
            "decoded joints/vertices losses still require local SMPL assets."
        ),
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
    if args.dry_run:
        print(json.dumps(_dry_run_payload(config_path, config, args), indent=2))
        return

    limits = _batch_limits(args)
    run_pose_baseline(
        config,
        config_path=config_path,
        resume=args.resume,
        work_dir=args.work_dir,
        device=args.device,
        experiment_name=EXPERIMENT_NAME,
        max_train_batches=limits["train"],
        max_val_batches=limits["val"],
        max_test_batches=limits["test"],
    )


if __name__ == "__main__":
    main()
