#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Select low-error M4Human absolute-pose samples for qualitative figures."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
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

from scripts.hpe_token._configs import resolve_config  # noqa: E402
from wave2body.train.common import load_config  # noqa: E402
from wave2body.train.pipelines.absolute_root_head import run_absolute_root_head_evaluation  # noqa: E402
from wave2body.utils.paths import artifact_root, resolve_runtime_uri  # noqa: E402
from wave2body.visualization.sample_selection import (  # noqa: E402
    DEFAULT_QUALITATIVE_GROUPS,
    ALL_M4HUMAN_ACTION_IDS,
    LowErrorPerActionSelector,
    LowErrorSampleSelector,
)


def _parse_groups(value: str) -> tuple[str, ...]:
    groups = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not groups:
        raise ValueError("Expected at least one action group")
    return groups


def _parse_action_ids(value: str) -> tuple[int, ...]:
    raw = str(value).strip()
    if raw.lower() in {"all", "1_50", "1-50"}:
        return ALL_M4HUMAN_ACTION_IDS
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(part))
    if not out:
        raise ValueError("Expected at least one action id")
    return tuple(dict.fromkeys(out))


def _default_checkpoint(config: dict[str, Any]) -> Path:
    run_name = str(config.get("training", {}).get("run_name", "")).strip()
    if not run_name:
        raise ValueError("--checkpoint is required because config.training.run_name is empty")
    return artifact_root() / "experiments" / "hpe_token" / "results" / run_name / "checkpoints" / "best.pth"


def _default_output_dir(config: dict[str, Any]) -> Path:
    split_id = str(config.get("m4human", {}).get("split", "s1"))
    return artifact_root() / "experiments" / "visualizations" / f"m4human_abs_qualitative_{split_id}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="m4human_abs_s2",
        help="Absolute HPE-token config alias, YAML stem, or YAML path.",
    )
    parser.add_argument("--checkpoint", help="Absolute root-head checkpoint. Defaults to config run_name best.pth.")
    parser.add_argument("--eval-split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpus", help='CUDA_VISIBLE_DEVICES value, e.g. "0".')
    parser.add_argument("--batch-size", type=int, help="Eval batch-size override.")
    parser.add_argument("--num-workers", type=int, help="Eval dataloader worker override.")
    parser.add_argument("--max-batches", type=int, help="Limit eval batches for smoke checks.")
    parser.add_argument("--top-k", type=int, default=3, help="Samples to keep per action group.")
    parser.add_argument("--per-action", action="store_true", help="Select top-k independently for each action id.")
    parser.add_argument("--action-ids", default="all", help="Comma/range action ids for --per-action, e.g. all or 1-50.")
    parser.add_argument(
        "--action-groups",
        default=",".join(DEFAULT_QUALITATIVE_GROUPS),
        help="Comma-separated semantic M4Human action groups.",
    )
    parser.add_argument("--max-per-action", type=int, default=1)
    parser.add_argument("--min-frame-gap", type=int, default=30)
    parser.add_argument("--pool-size", type=int, help="Candidate pool retained per action group.")
    parser.add_argument("--output-dir", help="Directory for manifest and sample NPZ files.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved plan without running eval.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config_path = resolve_config(args.config)
    config = load_config(config_path)
    checkpoint = Path(resolve_runtime_uri(args.checkpoint)).expanduser() if args.checkpoint else _default_checkpoint(config)
    output_dir = (
        Path(resolve_runtime_uri(args.output_dir)).expanduser()
        if args.output_dir
        else _default_output_dir(config)
    )
    action_groups = _parse_groups(args.action_groups)
    action_ids = _parse_action_ids(args.action_ids)
    payload = {
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "split_id": config.get("m4human", {}).get("split"),
        "eval_split": args.eval_split,
        "top_k": int(args.top_k),
        "action_groups": list(action_groups),
        "per_action": bool(args.per_action),
        "action_ids": list(action_ids),
        "max_per_action": int(args.max_per_action),
        "min_frame_gap": int(args.min_frame_gap),
        "pool_size": args.pool_size,
        "max_batches": args.max_batches,
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.per_action:
        selector = LowErrorPerActionSelector(
            top_k=int(args.top_k),
            action_ids=action_ids,
            min_frame_gap=int(args.min_frame_gap),
            pool_size=args.pool_size,
            split_id=str(config.get("m4human", {}).get("split", "")),
            eval_split=str(args.eval_split),
            config=str(config_path),
            checkpoint=str(checkpoint),
        )
    else:
        selector = LowErrorSampleSelector(
            top_k=int(args.top_k),
            action_groups=action_groups,
            max_per_action=int(args.max_per_action),
            min_frame_gap=int(args.min_frame_gap),
            pool_size=args.pool_size,
            split_id=str(config.get("m4human", {}).get("split", "")),
            eval_split=str(args.eval_split),
            config=str(config_path),
            checkpoint=str(checkpoint),
        )
    eval_payload = run_absolute_root_head_evaluation(
        config,
        checkpoint=checkpoint,
        config_path=config_path,
        eval_split=args.eval_split,
        device=args.device,
        output_path=output_dir / "eval_metrics.json",
        max_eval_batches=args.max_batches,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        include_per_action=True,
        sample_collector=selector,
    )

    records: list[dict[str, Any]] = []
    samples_dir = output_dir / "samples"
    selected_items = selector.selected().items()
    for group_or_action, candidates in selected_items:
        for rank, candidate in enumerate(candidates, start=1):
            name = (
                f"action_{candidate.action_id:02d}_rank{rank:02d}_"
                f"s{candidate.subject_id:02d}_a{candidate.action_id:02d}_f{candidate.frame_id:06d}.npz"
            )
            npz_path = samples_dir / name
            candidate.save_npz(npz_path)
            records.append(candidate.record(npz_path=str(npz_path.relative_to(output_dir))))

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
        "output_dir": str(output_dir),
        "eval_metrics_path": "eval_metrics.json",
        "num_seen_samples": int(selector.seen_count),
        "num_selected_samples": int(len(records)),
        "samples": records,
        "eval_summary": {
            "absolute_mpjpe_mm": eval_payload.get("metrics", {}).get("absolute_mpjpe_mm"),
            "num_samples": eval_payload.get("metrics", {}).get("num_samples"),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[Qualitative Samples] Wrote {manifest_path}")
    print(f"[Qualitative Samples] Selected {len(records)} samples from {selector.seen_count} seen samples")


if __name__ == "__main__":
    main()
