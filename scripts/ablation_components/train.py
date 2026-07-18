#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train controlled Wave2Body component ablations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


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
from wave2body.train.pipelines.ablation_components import (  # noqa: E402
    run_ablation_absolute_root,
    run_ablation_hpe_token,
    run_no_body_relative_root,
    run_no_body_token,
)
from wave2body.utils.paths import artifact_root  # noqa: E402


CONFIG_ROOT = REPO_ROOT / "wave2body" / "configs" / "ablations"


def _resolve_config(value: str) -> Path:
    path = Path(value).expanduser()
    candidates = []
    if path.suffix in {".yaml", ".yml"}:
        candidates.extend([path, CONFIG_ROOT / path])
    else:
        candidates.extend([CONFIG_ROOT / f"{value}.yaml", CONFIG_ROOT / value])
    for candidate in candidates:
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Config not found: {value}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Ablation config alias or YAML path.")
    parser.add_argument(
        "--stage",
        choices=("no_body", "no_body_relative_root", "hpe", "absolute"),
        help="Override training stage.",
    )
    parser.add_argument("--device", default="cuda", help="cuda/cpu")
    parser.add_argument("--gpus", help='CUDA_VISIBLE_DEVICES value, e.g. "0" or "0,1".')
    parser.add_argument("--split-id", choices=("s1", "s2", "s3"), help="Override M4Human split.")
    parser.add_argument("--work-dir", help="Explicit output directory.")
    parser.add_argument("--resume", help="Resume checkpoint.")
    parser.add_argument("--run-name", help="Override training.run_name.")
    parser.add_argument("--epochs", type=int, help="Override training epochs.")
    parser.add_argument("--batch-size", type=int, help="Override train/eval batch size.")
    parser.add_argument("--num-workers", type=int, help="Override dataloader workers.")
    parser.add_argument("--lr", type=float, help="Override base learning rate.")
    parser.add_argument("--min-lr", type=float, help="Override scheduler minimum learning rate.")
    parser.add_argument("--mmwave-tokenizer-checkpoint", help="Override pretrained.mmwave_tokenizer.")
    parser.add_argument("--body-vqvae-checkpoint", help="Override pretrained.vqvae.")
    parser.add_argument("--hpe-token-checkpoint", help="Override pretrained.hpe_token for absolute stage.")
    parser.add_argument("--max-train-batches", type=int, help="Limit train batches.")
    parser.add_argument("--max-val-batches", type=int, help="Limit val batches.")
    parser.add_argument("--max-test-batches", type=int, help="Limit test batches.")
    parser.add_argument("--smoke", action="store_true", help="One epoch and one train/val/test batch.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved run payload only.")
    return parser


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    training = config.setdefault("training", {})
    pretrained = config.setdefault("pretrained", {})
    if args.split_id:
        config.setdefault("m4human", {})["split"] = str(args.split_id)
    if args.run_name:
        training["run_name"] = str(args.run_name)
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
    if args.lr is not None:
        training["lr"] = float(args.lr)
    if args.min_lr is not None:
        training["min_lr"] = float(args.min_lr)
    if args.mmwave_tokenizer_checkpoint:
        pretrained["mmwave_tokenizer"] = str(Path(args.mmwave_tokenizer_checkpoint).expanduser())
    if args.body_vqvae_checkpoint:
        pretrained["vqvae"] = str(Path(args.body_vqvae_checkpoint).expanduser())
    if args.hpe_token_checkpoint:
        pretrained["hpe_token"] = str(Path(args.hpe_token_checkpoint).expanduser())
    if args.smoke:
        training["epochs"] = 1
        args.max_train_batches = args.max_train_batches or 1
        args.max_val_batches = args.max_val_batches or 1
        args.max_test_batches = args.max_test_batches or 1
    return config


def _stage_from_config(config: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    stage = str(config.get("training", {}).get("stage", "")).strip().lower()
    if stage in {"ablation_no_body_relative_root", "no_body_relative_root", "no_body_root"}:
        return "no_body_relative_root"
    if stage in {"ablation_no_body_token", "no_body", "no_body_token"}:
        return "no_body"
    if stage in {"absolute_root_head", "absolute"} or config.get("root_head"):
        return "absolute"
    return "hpe"


def _normalize_stage_labels(config: dict[str, Any], stage: str) -> None:
    training = config.setdefault("training", {})
    if stage == "no_body":
        training["stage"] = "ablation_no_body_token"
        training["stage_display_name"] = "No Body Token"
        training["metric_namespace"] = "ablation_no_body_token"
    elif stage == "no_body_relative_root":
        training["stage"] = "ablation_no_body_relative_root"
        training["stage_display_name"] = "No Body Token Relative Root"
        training["metric_namespace"] = "ablation_no_body_relative_root"
    elif stage == "absolute":
        training["stage"] = "absolute_root_head"
        training["stage_display_name"] = "Absolute Root Head Ablation"
        training["metric_namespace"] = "absolute_root_head"
    else:
        training["stage"] = "hpe_token"
        training["stage_display_name"] = "HPE Token Ablation"
        training["metric_namespace"] = "hpe_token"


def _planned_output_root(config: dict[str, Any], work_dir: str | None) -> Path:
    if work_dir:
        return Path(work_dir).expanduser()
    run_name = str(config.get("training", {}).get("run_name", "ablation_components"))
    return artifact_root() / "experiments" / "ablation_components_current" / run_name


def main() -> None:
    args = _build_parser().parse_args()
    config_path = _resolve_config(args.config)
    config = _apply_overrides(load_config(config_path), args)
    stage = _stage_from_config(config, args.stage)
    _normalize_stage_labels(config, stage)
    planned_root = _planned_output_root(config, args.work_dir)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": stage,
                    "config": str(config_path),
                    "work_dir": str(planned_root),
                    "split": config.get("m4human", {}).get("split"),
                    "training": config.get("training", {}),
                    "pretrained": config.get("pretrained", {}),
                    "max_batches": {
                        "train": args.max_train_batches,
                        "val": args.max_val_batches,
                        "test": args.max_test_batches,
                    },
                },
                indent=2,
            )
        )
        return

    if stage == "no_body":
        run_no_body_token(
            config,
            config_path=config_path,
            work_dir=args.work_dir,
            device=args.device,
            resume=args.resume,
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
            max_test_batches=args.max_test_batches,
        )
    elif stage == "no_body_relative_root":
        run_no_body_relative_root(
            config,
            config_path=config_path,
            work_dir=args.work_dir,
            device=args.device,
            resume=args.resume,
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
            max_test_batches=args.max_test_batches,
        )
    elif stage == "absolute":
        run_ablation_absolute_root(
            config,
            config_path=config_path,
            work_dir=args.work_dir,
            device=args.device,
            resume=args.resume,
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
            max_test_batches=args.max_test_batches,
        )
    else:
        run_ablation_hpe_token(
            config,
            config_path=config_path,
            work_dir=args.work_dir,
            device=args.device,
            resume=args.resume,
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
        )


if __name__ == "__main__":
    main()
