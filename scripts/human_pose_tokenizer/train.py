#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train the main human pose tokenizer recipe from package configs."""

from __future__ import annotations

import argparse
import shlex
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.human_pose_tokenizer._configs import (  # noqa: E402
    HUMAN_POSE_TOKENIZER_WORKSPACE,
    print_configs,
    resolve_config,
)
from wave2body.train.common import load_config, write_run_manifest  # noqa: E402
from wave2body.train.pipelines.human_pose_tokenizer import run_human_pose_tokenizer  # noqa: E402


def default_work_dir(config_path: Path, run_name: str | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name or f"{config_path.stem}_{timestamp}"
    return HUMAN_POSE_TOKENIZER_WORKSPACE / "checkpoints" / name


def build_extra_args(args: argparse.Namespace) -> list[str]:
    extra: list[str] = []
    optional_pairs = [
        ("--resume", args.resume),
        ("--gpus", args.gpus),
        ("--epochs", args.epochs),
        ("--batch-size", args.batch_size),
        ("--num-workers", args.num_workers),
        ("--log-interval", args.log_interval),
        ("--dataset-device", args.dataset_device),
        ("--batch-loader", args.batch_loader),
        ("--max-dataset-samples", args.max_dataset_samples),
        ("--max-train-batches", args.max_train_batches),
        ("--max-val-batches", args.max_val_batches),
    ]
    for flag, value in optional_pairs:
        if value is not None:
            extra.extend([flag, str(value)])
    if args.resume_model_only:
        extra.append("--resume-model-only")
    if args.deterministic:
        extra.append("--deterministic")
    return extra


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the main AMASS/MMBody/M4Human human pose tokenizer.")
    parser.add_argument("--config", default="pretrain", help="Config alias or YAML path.")
    parser.add_argument("--list-configs", action="store_true", help="List config aliases and exit.")
    parser.add_argument("--run-name", help="Name for the checkpoint output directory.")
    parser.add_argument("--work-dir", help="Explicit checkpoint output directory.")
    parser.add_argument("--resume", help="Checkpoint to resume/init from.")
    parser.add_argument("--resume-model-only", action="store_true", help="Load model weights and reset optimizer state.")
    parser.add_argument("--gpus", help='CUDA_VISIBLE_DEVICES value, e.g. "0" or "0,1".')
    parser.add_argument("--epochs", type=int, help="Override config training.epochs.")
    parser.add_argument("--batch-size", type=int, help="Override config training.batch_size.")
    parser.add_argument("--num-workers", type=int, help="Override config training.num_workers.")
    parser.add_argument("--log-interval", type=int, help="Override config training.log_interval.")
    parser.add_argument("--dataset-device", choices=["cpu", "cuda"], help="Override config data.dataset_device.")
    parser.add_argument("--batch-loader", choices=["dataloader", "tensor"], help="Override config data.batch_loader.")
    parser.add_argument("--max-dataset-samples", type=int, help="Limit samples per configured dataset before cache build.")
    parser.add_argument("--max-train-batches", type=int, help="Limit train batches.")
    parser.add_argument("--max-val-batches", type=int, help="Limit validation batches.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic cudnn.")
    parser.add_argument("--dry-run", action="store_true", help="Print equivalent trainer command and exit.")
    args = parser.parse_args()

    if args.list_configs:
        print_configs()
        return

    config_path = resolve_config(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    work_dir = Path(args.work_dir).expanduser() if args.work_dir else default_work_dir(config_path, args.run_name)
    extra_args = build_extra_args(args)
    command = [
        sys.executable,
        "-m",
        "wave2body.train.human_pose_tokenizer",
        "--config",
        str(config_path),
        "--work-dir",
        str(work_dir),
        *extra_args,
    ]
    if args.dry_run:
        print(" ".join(shlex.quote(part) for part in command))
        return

    result = run_human_pose_tokenizer(config_path=config_path, work_dir=work_dir, extra_args=extra_args)
    config = load_config(config_path)
    best_checkpoint = Path(result["best_checkpoint"])
    dataset_names = [item.get("name") for item in config.get("data", {}).get("dataset_configs", [])]
    write_run_manifest(
        work_dir,
        config_path=config_path,
        checkpoint_path=best_checkpoint if best_checkpoint.exists() else None,
        config={
            "dataset": {"name": "+".join(str(name) for name in dataset_names if name)},
            "dataset_path": "see data.dataset_configs",
        },
        dataset_split="combined_train_random_split",
        coord_transform="pelvis-centered/root-normalized joints",
        output_paths={
            "checkpoints": work_dir,
            "results": HUMAN_POSE_TOKENIZER_WORKSPACE / "results",
        },
        extra={
            "launcher": str(Path(__file__).resolve()),
            "trainer": "wave2body.train.human_pose_tokenizer",
            "trainer_result": result,
            "human_pose_tokenizer_config": {
                "model": config.get("model", {}),
                "data": config.get("data", {}),
            },
        },
    )
    print(f"Run complete. Work dir: {work_dir}")


if __name__ == "__main__":
    main()
