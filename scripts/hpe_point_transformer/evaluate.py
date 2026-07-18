#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate direct PointTransformer HPE checkpoints on the M4Human all-data route."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "wave2body" / "configs" / "hpe_point_transformer"
CONFIG_ALIASES = {
    "m4human_absolute_hist4": CONFIG_DIR / "m4human_absolute_hist4_s2_bf16.yaml",
    "mmbody_noinput_rootnorm": CONFIG_DIR / "mmbody_pt_noinput_rootnorm.yaml",
    "mmbody_noinput_rootnorm_fastcache": CONFIG_DIR / "mmbody_pt_noinput_rootnorm_fastcache.yaml",
    "mmbody_noinput_absolute_fastcache": CONFIG_DIR / "mmbody_pt_noinput_abs_fastcache.yaml",
}
VALID_SPLIT_IDS = ("s1", "s2", "s3")
VALID_EVAL_SPLITS = ("val", "test", "train")


def _preparse_gpus(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpus")
    args, _ = parser.parse_known_args(argv)
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)


_preparse_gpus(sys.argv[1:])

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wave2body.train.checkpoint import load_checkpoint  # noqa: E402
from wave2body.train.common import load_config  # noqa: E402
from wave2body.train.pipelines.pt_pose import run_pt_pose_evaluation  # noqa: E402


def resolve_config(value: str | None) -> Path | None:
    if value in (None, "", "checkpoint"):
        return None
    if value in CONFIG_ALIASES:
        return CONFIG_ALIASES[value]
    path = Path(str(value)).expanduser()
    if path.exists():
        return path
    known = ", ".join(["checkpoint", *sorted(CONFIG_ALIASES)])
    raise ValueError(f"Unknown HPE PointTransformer config {value!r}; expected alias/path in: {known}")


def print_configs() -> None:
    print("checkpoint: load config embedded in the checkpoint")
    for name, path in CONFIG_ALIASES.items():
        print(f"{name}: {path}")


def _csv_choices(value: str | None, *, choices: tuple[str, ...], default: list[str]) -> list[str]:
    if value in (None, "", "default"):
        items = list(default)
    elif str(value).lower() == "all":
        items = list(choices)
    else:
        items = [item.strip() for item in str(value).split(",") if item.strip()]
    bad = [item for item in items if item not in choices]
    if bad:
        raise ValueError(f"Invalid values {bad}; expected choices from {choices}")
    return items


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=False, help="HPE PointTransformer checkpoint to evaluate.")
    parser.add_argument("--config", default="checkpoint", help="Config alias/path, or 'checkpoint' to use checkpoint config.")
    parser.add_argument("--list-configs", action="store_true", help="List config aliases and exit.")
    parser.add_argument("--split-ids", default="default", help="Comma list, 'all', or default checkpoint/config split.")
    parser.add_argument("--eval-splits", default="val,test", help="Comma list, e.g. val,test.")
    parser.add_argument("--device", default="cuda", help="cuda/cpu")
    parser.add_argument("--gpus", help='CUDA_VISIBLE_DEVICES value, e.g. "0" or "0,1".')
    parser.add_argument("--batch-size", type=int, help="Eval batch-size override.")
    parser.add_argument("--num-workers", type=int, help="Eval dataloader worker override.")
    parser.add_argument("--max-batches", type=int, help="Limit eval batches for smoke checks.")
    parser.add_argument("--output-dir", help="Directory for per-combination metrics JSON files.")
    parser.add_argument("--summary-output", help="Output summary JSON path.")
    parser.add_argument("--no-action-groups", action="store_true", help="Disable M4Human action-group metrics.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved eval plan without running.")
    return parser


def _load_base_config(args: argparse.Namespace) -> tuple[dict[str, Any], str | None]:
    config_path = resolve_config(args.config)
    if config_path is not None:
        return load_config(config_path), str(config_path)
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required when --config=checkpoint")
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    config = ckpt.get("config")
    if not isinstance(config, dict):
        raise KeyError(f"Checkpoint {args.checkpoint} does not contain an embedded config")
    return copy.deepcopy(config), None


def _apply_eval_overrides(
    config: dict[str, Any],
    *,
    split_id: str,
    batch_size: int | None,
    num_workers: int | None,
    include_action_groups: bool,
) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    training = cfg.setdefault("training", {})
    m4human = cfg.setdefault("m4human", {})
    data_format = cfg.setdefault("data_format", {})
    evaluation = cfg.setdefault("evaluation", {})

    m4human.pop("scale", None)
    m4human["split"] = split_id
    m4human.setdefault("frame_mode", "hist4")
    m4human.setdefault("temporal_offsets", [-3, -2, -1, 0])
    m4human["temporal_window"] = len(m4human["temporal_offsets"])
    m4human.setdefault("pose_source", "param_joints")
    m4human.setdefault("normalize_z", True)
    data_format["merge_nframes"] = len(m4human["temporal_offsets"])

    evaluation["mpjpe_only"] = True
    evaluation["m4human_action_groups"] = bool(include_action_groups)
    if not bool(data_format.get("normalized", True)):
        evaluation.setdefault("primary_metric", "absolute_mpjpe")
        diagnostics = evaluation.get("diagnostic_metrics", [])
        if isinstance(diagnostics, str):
            diagnostics = [diagnostics]
        for name in ("absolute_mpjpe", "root_error", "root_aligned_mpjpe", "p_mpjpe"):
            if name not in diagnostics:
                diagnostics.append(name)
        evaluation["diagnostic_metrics"] = diagnostics
        evaluation["compute_p_mpjpe"] = True
    else:
        evaluation.setdefault("primary_metric", "mpjpe")

    if batch_size is not None:
        training["batch_size"] = int(batch_size)
    if num_workers is not None:
        workers = int(num_workers)
        training["num_workers"] = workers
        if workers <= 0:
            training["pin_memory"] = False
            training["persistent_workers"] = False
    return cfg


def _default_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser()
    return Path(args.checkpoint).expanduser().parent.parent


def _plan_payload(
    *,
    config_source: str | None,
    base_config: dict[str, Any],
    args: argparse.Namespace,
    split_ids: list[str],
    eval_splits: list[str],
) -> dict[str, Any]:
    return {
        "checkpoint": args.checkpoint,
        "config_source": config_source or "checkpoint",
        "split_ids": split_ids,
        "eval_splits": eval_splits,
        "target_normalized": base_config.get("data_format", {}).get("normalized"),
        "coord_transform": base_config.get("coord_transform"),
        "batch_size": args.batch_size or base_config.get("training", {}).get("batch_size"),
        "num_workers": args.num_workers if args.num_workers is not None else base_config.get("training", {}).get("num_workers"),
        "max_batches": args.max_batches,
        "action_groups": not bool(args.no_action_groups),
        "output_dir": str(_default_output_dir(args)) if args.checkpoint else args.output_dir,
    }


def main() -> None:
    args = _build_parser().parse_args()
    if args.list_configs:
        print_configs()
        return
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --list-configs is used")

    base_config, config_source = _load_base_config(args)
    default_split = [str(base_config.get("m4human", {}).get("split", "s2"))]
    split_ids = _csv_choices(args.split_ids, choices=VALID_SPLIT_IDS, default=default_split)
    eval_splits = _csv_choices(args.eval_splits, choices=VALID_EVAL_SPLITS, default=["val", "test"])
    if args.dry_run:
        print(
            json.dumps(
                _plan_payload(
                    config_source=config_source,
                    base_config=base_config,
                    args=args,
                    split_ids=split_ids,
                    eval_splits=eval_splits,
                ),
                indent=2,
            )
        )
        return

    output_dir = _default_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for split_id in split_ids:
        for eval_split in eval_splits:
            cfg = _apply_eval_overrides(
                base_config,
                split_id=split_id,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                include_action_groups=not bool(args.no_action_groups),
            )
            output_path = output_dir / f"metrics_{eval_split}_{split_id}.json"
            results.append(
                run_pt_pose_evaluation(
                    cfg,
                    checkpoint=args.checkpoint,
                    config_path=config_source,
                    eval_split=eval_split,
                    device=args.device,
                    output_path=output_path,
                    max_eval_batches=args.max_batches,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    include_action_groups=not bool(args.no_action_groups),
                )
            )

    summary = {
        "stage": "hpe_point_transformer",
        "checkpoint": args.checkpoint,
        "config_source": config_source or "checkpoint",
        "split_ids": split_ids,
        "eval_splits": eval_splits,
        "results": results,
    }
    summary_path = Path(args.summary_output).expanduser() if args.summary_output else output_dir / "eval_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[PT Eval] Wrote summary {summary_path}")


if __name__ == "__main__":
    main()
