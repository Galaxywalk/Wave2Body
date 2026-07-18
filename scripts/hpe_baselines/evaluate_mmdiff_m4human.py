#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate M4Human mmDiff checkpoints with semantic action-protocol metrics."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
VALID_EVAL_SPLITS = ("val", "test")


def _preparse_gpus(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpus")
    args, _ = parser.parse_known_args(argv)
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)


_preparse_gpus(sys.argv[1:])

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from wave2body.evaluate.m4human_action_groups import (  # noqa: E402
    M4HumanActionGroupAccumulator,
    m4human_action_ids_from_indicator,
)
from wave2body.models.baselines import build_pose_baseline, forward_pose_baseline, move_pose_baseline_batch  # noqa: E402
from wave2body.train.checkpoint import load_checkpoint, load_state_dict_shape_safe  # noqa: E402
from wave2body.train.radar_pose_data import create_radar_pose_dataset  # noqa: E402
from wave2body.utils.metrics import compute_per_sample_p_mpjpe  # noqa: E402
from wave2body.utils.runtime_utils import autocast_context, configure_runtime  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, help="Completed mmDiff run directory.")
    parser.add_argument("--checkpoint-name", default="best.pth", help="Checkpoint filename under run_dir/checkpoints.")
    parser.add_argument("--eval-split", default="test", choices=VALID_EVAL_SPLITS)
    parser.add_argument("--device", default="cuda", help="cuda/cpu")
    parser.add_argument("--gpus", help='CUDA_VISIBLE_DEVICES value, e.g. "0".')
    parser.add_argument("--batch-size", type=int, default=256, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--max-batches", type=int, help="Limit batches for smoke checks.")
    parser.add_argument("--output-dir", required=True, help="Directory for metrics JSON files.")
    parser.add_argument("--summary-output", help="Explicit summary JSON path.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _eval_config(base_config: dict[str, Any], *, batch_size: int, num_workers: int) -> dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    m4human = cfg.setdefault("m4human", {})
    training = cfg.setdefault("training", {})
    m4human.pop("scale", None)
    m4human.setdefault("frame_mode", "hist4")
    m4human.setdefault("temporal_offsets", [-3, -2, -1, 0])
    m4human["temporal_window"] = len(m4human["temporal_offsets"])
    m4human.setdefault("pose_source", "param_joints")
    m4human.setdefault("normalize_z", True)
    cfg.setdefault("data_format", {})["merge_nframes"] = len(m4human["temporal_offsets"])
    training["batch_size"] = int(batch_size)
    training["num_workers"] = int(num_workers)
    training["pin_memory"] = True
    training["persistent_workers"] = False
    return cfg


def _load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    config = ckpt.get("config")
    if not isinstance(config, dict):
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain embedded config")
    bundle = build_pose_baseline(config, device=str(device))
    bundle.model.to(device)

    # Materialize LazyLinear layers before shape-safe loading.
    fmt = config.get("data_format", {})
    n_points = int(fmt.get("n_points", 256))
    radar_input_c = int(fmt.get("radar_input_c", config.get("mmdiff", {}).get("radar_input_dim", 4)))
    moved = {
        "mmwave_points": torch.zeros((1, n_points, radar_input_c), device=device, dtype=torch.float32),
        "timesteps": torch.zeros((1,), device=device, dtype=torch.long),
    }
    with torch.inference_mode():
        forward_pose_baseline(bundle, moved)

    state = ckpt.get("model_state_dict", ckpt)
    if not isinstance(state, dict):
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain model_state_dict")
    load_state_dict_shape_safe(bundle.model, state, module_name="MMDiffBaseline", strict=False)
    bundle.model.to(device).eval()
    return bundle.model, config, ckpt


def _target_joints(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    target = batch.get("target_poses")
    if not torch.is_tensor(target):
        target = batch.get("target_poses_absolute")
    if not torch.is_tensor(target):
        raise KeyError("M4Human batch must contain target_poses or target_poses_absolute")
    return target.to(device, non_blocking=True).float()


def _root_aligned(pose: torch.Tensor, root_index: int) -> torch.Tensor:
    return pose - pose[:, root_index : root_index + 1]


def _fmt_metric(value: Any) -> str:
    return "--" if value is None else f"{float(value):.2f}"


def evaluate_one(
    *,
    model: torch.nn.Module,
    base_config: dict[str, Any],
    checkpoint_path: Path,
    eval_split: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_batches: int | None,
) -> dict[str, Any]:
    cfg = _eval_config(base_config, batch_size=batch_size, num_workers=num_workers)
    split_id = str(cfg.get("m4human", {}).get("split", "unknown"))
    dataset = create_radar_pose_dataset(cfg, split=eval_split, device="cpu")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        drop_last=False,
    )
    runtime = configure_runtime(cfg.get("training", {}), device, default_amp=True, log_prefix="[mmDiff Eval Runtime]")
    action_acc = M4HumanActionGroupAccumulator(
        ("mpjpe", "root_error", "root_aligned_mpjpe", "p_mpjpe"),
        device=device,
    )
    total_mpjpe = 0.0
    total_root = 0.0
    total_root_aligned = 0.0
    total_p = 0.0
    total_samples = 0
    root_index = int(cfg.get("evaluation", {}).get("root_index", 0))

    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"mmDiff {split_id} {eval_split}")):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            moved = move_pose_baseline_batch(batch, device)
            target = _target_joints(moved, device)
            with autocast_context(runtime):
                outputs = model(moved["mmwave_points"].float(), timesteps=moved.get("timesteps", moved.get("diffusion_timesteps", 0)))
            pred = outputs["pred_pose"].float()
            if pred.shape != target.shape:
                raise ValueError(f"Pose shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
            per_sample = torch.linalg.vector_norm(pred - target, dim=-1).mean(dim=-1)
            root_error = torch.linalg.vector_norm(pred[:, root_index] - target[:, root_index], dim=-1)
            root_aligned = torch.linalg.vector_norm(
                _root_aligned(pred, root_index) - _root_aligned(target, root_index),
                dim=-1,
            ).mean(dim=-1)
            p_mpjpe = torch.as_tensor(compute_per_sample_p_mpjpe(pred, target), device=device, dtype=torch.float32)
            total_mpjpe += float(per_sample.double().sum().item())
            total_root += float(root_error.double().sum().item())
            total_root_aligned += float(root_aligned.double().sum().item())
            total_p += float(p_mpjpe.double().sum().item())
            total_samples += int(per_sample.numel())
            action_ids = m4human_action_ids_from_indicator(moved["indicator"], device=device)
            action_acc.update("mpjpe", action_ids, per_sample)
            action_acc.update("root_error", action_ids, root_error)
            action_acc.update("root_aligned_mpjpe", action_ids, root_aligned)
            action_acc.update("p_mpjpe", action_ids, p_mpjpe)

    if total_samples <= 0:
        raise RuntimeError(f"No samples evaluated for {split_id}/{eval_split}")
    return {
        "checkpoint": str(checkpoint_path),
        "split": split_id,
        "eval_split": eval_split,
        "num_samples": total_samples,
        "mpjpe_m": total_mpjpe / total_samples,
        "mpjpe_mm": total_mpjpe / total_samples * 1000.0,
        "root_error_m": total_root / total_samples,
        "root_error_mm": total_root / total_samples * 1000.0,
        "root_aligned_mpjpe_m": total_root_aligned / total_samples,
        "root_aligned_mpjpe_mm": total_root_aligned / total_samples * 1000.0,
        "p_mpjpe_m": total_p / total_samples,
        "p_mpjpe_mm": total_p / total_samples * 1000.0,
        "m4human_action_groups_mm": action_acc.summary(scale=1000.0, include_per_action=True),
    }


def main() -> None:
    args = _build_parser().parse_args()
    run_dirs = [Path(value).expanduser() for value in args.run_dir]
    output_dir = Path(args.output_dir).expanduser()
    summary_path = Path(args.summary_output).expanduser() if args.summary_output else output_dir / "summary.json"
    if args.dry_run:
        print(json.dumps({"run_dirs": [str(path) for path in run_dirs], "output_dir": str(output_dir)}, indent=2))
        return

    requested_device = str(args.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(requested_device)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        checkpoint_path = run_dir / "checkpoints" / args.checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        model, config, _ = _load_model_from_checkpoint(checkpoint_path, device)
        result = evaluate_one(
            model=model,
            base_config=config,
            checkpoint_path=checkpoint_path,
            eval_split=args.eval_split,
            device=device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            max_batches=args.max_batches,
        )
        out_path = output_dir / f"metrics_{args.eval_split}_{result['split']}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        groups = result["m4human_action_groups_mm"]["mpjpe"]["by_group"]
        print(
            f"[mmDiff Eval] split={result['split']} mpjpe={result['mpjpe_mm']:.2f}mm "
            f"1_30={_fmt_metric(groups.get('1_30'))} 31_35={_fmt_metric(groups.get('31_35'))} "
            f"36_50={_fmt_metric(groups.get('36_50'))} 1_50={_fmt_metric(groups.get('1_50'))}"
        )
        results.append(result)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "stage": "hpe_baseline_mmdiff",
        "eval_split": args.eval_split,
        "checkpoint_name": args.checkpoint_name,
        "metric_note": "M4Human mmDiff absolute MPJPE; action groups are sample-micro averages by 1-based action id.",
        "results": sorted(results, key=lambda item: item["split"]),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[mmDiff Eval] Wrote {summary_path}")


if __name__ == "__main__":
    main()
