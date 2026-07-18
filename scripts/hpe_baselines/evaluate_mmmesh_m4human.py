#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate M4Human mmMesh checkpoints with semantic action-protocol metrics."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_NAMES = (
    "m4human_mmmesh_s1_hist4_absmpjpe_e25_bf16_bs1024",
    "m4human_mmmesh_s2_hist4_absmpjpe_e25_bf16_bs1024",
    "m4human_mmmesh_s3_hist4_absmpjpe_e25_bf16_bs1024",
)
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

from wave2body.dataset.radar_pose import M4HumanLMDBPoseDataset  # noqa: E402
from wave2body.dataset.radar_pose.m4human_protocols import (  # noqa: E402
    M4HUMAN_ACTION_GROUP_ORDER,
    M4HUMAN_ACTION_GROUP_TAGS,
)
from wave2body.evaluate.m4human_action_groups import (  # noqa: E402
    M4HumanActionGroupAccumulator,
    m4human_action_ids_from_indicator,
)
from wave2body.models.baselines import build_pose_baseline  # noqa: E402
from wave2body.train.checkpoint import load_checkpoint, load_state_dict_shape_safe  # noqa: E402
from wave2body.utils.paths import artifact_root  # noqa: E402


def _default_result_root() -> Path:
    return artifact_root() / "experiments" / "hpe_baselines" / "results"


def _default_run_dirs() -> list[Path]:
    root = _default_result_root()
    return [root / name for name in DEFAULT_RUN_NAMES]


def _run_dirs_from_args(values: list[str] | None) -> list[Path]:
    if not values:
        return _default_run_dirs()
    return [Path(value).expanduser() for value in values]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        action="append",
        help="Completed mmMesh run directory. Repeat for s1/s2/s3. Defaults to current M4Human mmMesh runs.",
    )
    parser.add_argument("--checkpoint-name", default="best.pth", help="Checkpoint filename under run_dir/checkpoints.")
    parser.add_argument("--eval-split", default="test", choices=VALID_EVAL_SPLITS)
    parser.add_argument("--device", default="cuda", help="cuda/cpu")
    parser.add_argument("--gpus", help='CUDA_VISIBLE_DEVICES value, e.g. "0".')
    parser.add_argument("--batch-size", type=int, default=512, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--max-batches", type=int, help="Limit batches for smoke checks.")
    parser.add_argument("--output-dir", help="Directory for metrics JSON/Markdown files.")
    parser.add_argument("--summary-output", help="Explicit summary JSON path.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    config = ckpt.get("config")
    if not isinstance(config, dict):
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain embedded config")
    bundle = build_pose_baseline(config, device=str(device))
    state = ckpt.get("model_state_dict", ckpt)
    if not isinstance(state, dict):
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain model_state_dict")
    load_state_dict_shape_safe(bundle.model, state, module_name="MMMeshBaseline", strict=False)
    bundle.model.to(device).eval()
    return bundle.model, config, ckpt


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


def _target_joints(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    target = batch.get("target_poses_absolute")
    if not torch.is_tensor(target):
        target = batch.get("target_poses")
    if not torch.is_tensor(target):
        raise KeyError("M4Human batch must contain target_poses_absolute or target_poses")
    return target.to(device, non_blocking=True).float()


def _pred_joints(outputs: dict[str, Any]) -> torch.Tensor:
    pred = outputs.get("pred_pose")
    if torch.is_tensor(pred):
        return pred.float()
    body = outputs.get("body")
    if isinstance(body, dict):
        pred = body.get("joints")
        if not torch.is_tensor(pred):
            pred = body.get("pred_joints")
    if not torch.is_tensor(pred):
        raise KeyError("mmMesh output must contain decoded joints in pred_pose or body['joints']")
    if pred.ndim == 4:
        return pred[:, -1].float()
    if pred.ndim == 3:
        return pred.float()
    raise ValueError(f"Unexpected decoded joint shape: {tuple(pred.shape)}")


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "--"
    return f"{float(value):.2f}"


def _autocast_enabled(config: dict[str, Any], device: torch.device) -> tuple[bool, torch.dtype]:
    training = config.get("training", {})
    if device.type != "cuda" or not bool(training.get("amp", True)):
        return False, torch.float32
    precision = str(training.get("precision", "bf16")).lower()
    return True, torch.bfloat16 if precision == "bf16" else torch.float16


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
    dataset = M4HumanLMDBPoseDataset(config=cfg, split=eval_split, device="cpu")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        drop_last=False,
    )

    action_acc = M4HumanActionGroupAccumulator(("mpjpe", "root_error", "root_aligned_mpjpe"), device=device)
    total_mpjpe = 0.0
    total_root = 0.0
    total_root_aligned = 0.0
    total_samples = 0
    autocast, autocast_dtype = _autocast_enabled(base_config, device)
    desc = f"mmMesh {split_id} {eval_split}"

    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(loader, desc=desc)):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            points = batch["mmwave_points"].to(device, non_blocking=True).float()
            target = _target_joints(batch, device)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast):
                outputs = model(points)
            pred = _pred_joints(outputs)
            if pred.shape != target.shape:
                raise ValueError(f"Decoded joint shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
            joint_error = torch.linalg.vector_norm(pred - target, dim=-1)
            mpjpe_sample = joint_error.mean(dim=-1)
            root_error_sample = torch.linalg.vector_norm(pred[:, 0] - target[:, 0], dim=-1)
            pred_rel = pred - pred[:, :1]
            target_rel = target - target[:, :1]
            root_aligned_sample = torch.linalg.vector_norm(pred_rel - target_rel, dim=-1).mean(dim=-1)
            num = int(mpjpe_sample.numel())
            total_mpjpe += float(mpjpe_sample.double().sum().item())
            total_root += float(root_error_sample.double().sum().item())
            total_root_aligned += float(root_aligned_sample.double().sum().item())
            total_samples += num

            action_ids = m4human_action_ids_from_indicator(batch["indicator"], device=device)
            action_acc.update("mpjpe", action_ids, mpjpe_sample)
            action_acc.update("root_error", action_ids, root_error_sample)
            action_acc.update("root_aligned_mpjpe", action_ids, root_aligned_sample)

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
        "m4human_action_groups_mm": action_acc.summary(scale=1000.0, include_per_action=True),
    }


def _write_markdown(results: list[dict[str, Any]], path: Path) -> None:
    rows = sorted(results, key=lambda item: item["split"])
    lines = [
        "# M4Human mmMesh Semantic Action-Protocol Test Evaluation",
        "",
        "Best checkpoints are evaluated on M4Human test subsets. Protocol columns use the semantic action split: IP 1-30, SIP 31-35, NIP 36-50.",
        "",
        "| Split | P1/IP 1-30 | P2/SIP 31-35 | P3/NIP 36-50 | ALL 1-50 | N |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in rows:
        groups = item["m4human_action_groups_mm"]["mpjpe"]["by_group"]
        counts = item["m4human_action_groups_mm"]["mpjpe"]["num_samples_by_group"]
        lines.append(
            f"| `{item['split']}` | "
            f"`{_fmt_metric(groups.get('1_30'))}` | "
            f"`{_fmt_metric(groups.get('31_35'))}` | "
            f"`{_fmt_metric(groups.get('36_50'))}` | "
            f"`{_fmt_metric(groups.get('1_50'))}` | "
            f"`{counts.get('1_50', item['num_samples'])}` |"
        )
    lines.extend(
        [
            "",
            "## Overall Diagnostics",
            "",
            "| Split | Absolute MPJPE | Root Error | Root-Aligned MPJPE | N |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in rows:
        lines.append(
            f"| `{item['split']}` | "
            f"`{item['mpjpe_mm']:.2f}` | "
            f"`{item['root_error_mm']:.2f}` | "
            f"`{item['root_aligned_mpjpe_mm']:.2f}` | "
            f"`{item['num_samples']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _build_parser().parse_args()
    run_dirs = _run_dirs_from_args(args.run_dir)
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else _default_result_root() / "m4human_mmmesh_action_group_eval"
    summary_path = Path(args.summary_output).expanduser() if args.summary_output else output_dir / "metrics_test_matrix.json"
    markdown_path = output_dir / "metrics_test_matrix.md"

    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_dirs": [str(path) for path in run_dirs],
                    "checkpoint_name": args.checkpoint_name,
                    "action_groups": {
                        name: M4HUMAN_ACTION_GROUP_TAGS[name]
                        for name in M4HUMAN_ACTION_GROUP_ORDER
                    },
                    "eval_split": args.eval_split,
                    "device": args.device,
                    "gpus": args.gpus,
                    "batch_size": args.batch_size,
                    "num_workers": args.num_workers,
                    "max_batches": args.max_batches,
                    "summary_output": str(summary_path),
                    "markdown_output": str(markdown_path),
                },
                indent=2,
            )
        )
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
        model, config, ckpt = _load_model_from_checkpoint(checkpoint_path, device)
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
        group_text = " ".join(
            f"{M4HUMAN_ACTION_GROUP_TAGS[name]}={_fmt_metric(groups.get(name))}mm"
            for name in M4HUMAN_ACTION_GROUP_ORDER
        )
        print(
            f"[mmMesh Eval] split={result['split']} "
            f"mpjpe={result['mpjpe_mm']:.2f}mm "
            f"root={result['root_error_mm']:.2f}mm "
            f"root_aligned={result['root_aligned_mpjpe_mm']:.2f}mm "
            f"{group_text} N={result['num_samples']}"
        )
        results.append(result)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "stage": "hpe_baseline_mmmesh",
        "eval_split": args.eval_split,
        "checkpoint_name": args.checkpoint_name,
        "action_group_note": "M4Human semantic paper protocols: P1/IP=1-30, P2/SIP=31-35, P3/NIP=36-50, ALL=1-50.",
        "action_group_tags": {
            name: M4HUMAN_ACTION_GROUP_TAGS[name]
            for name in M4HUMAN_ACTION_GROUP_ORDER
        },
        "metric_note": "M4Human mmMesh absolute MPJPE from decoded 22-joint outputs; action groups are sample-micro averages by 1-based action id.",
        "results": sorted(results, key=lambda item: item["split"]),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(summary["results"], markdown_path)
    print(f"[mmMesh Eval] Wrote {summary_path}")
    print(f"[mmMesh Eval] Wrote {markdown_path}")


if __name__ == "__main__":
    main()
