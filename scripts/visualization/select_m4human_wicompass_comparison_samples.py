#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Select same-frame M4Human Wave2Body vs WiCompass/PT comparison samples."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PT_CONFIG = REPO_ROOT / "wave2body" / "configs" / "hpe_point_transformer" / "m4human_absolute_hist4_no_rootnorm_a100.yaml"
DEFAULT_SPLITS = ("s2", "s3")
DEFAULT_ACTION_GROUPS = ("31_35", "36_50")
DEFAULT_PT_RUNS = {
    "s1": "m4human_hpe_pt_abs_p1_s1_hist4_norootnorm_e25_bf16_bs64_cdist_nocompile_20260621",
    "s2": "m4human_hpe_pt_abs_p1_s2_hist4_norootnorm_e25_bf16_bs64_cdist_nocompile_20260620",
    "s3": "m4human_hpe_pt_abs_p1_s3_hist4_norootnorm_e25_bf16_bs64_cdist_nocompile_20260620",
}


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
from tqdm import tqdm  # noqa: E402

from scripts.hpe_token._configs import resolve_config as resolve_wave2body_config  # noqa: E402
from wave2body.models.hpe_token.absolute_root import AbsoluteRootHeadEstimatorModel  # noqa: E402
from wave2body.models.point_transformer.pose_baseline import PTPoseModel  # noqa: E402
from wave2body.train.checkpoint import load_checkpoint, load_state_dict_shape_safe  # noqa: E402
from wave2body.train.common import load_config  # noqa: E402
from wave2body.train.pt_pose_utils import materialize_lazy_modules  # noqa: E402
from wave2body.train.radar_pose_data import create_radar_pose_dataset  # noqa: E402
from wave2body.train.pipelines.absolute_root_head import _load_absolute_component_states  # noqa: E402
from wave2body.utils.paths import artifact_root, resolve_runtime_uri  # noqa: E402
from wave2body.utils.runtime_utils import autocast_context, configure_runtime  # noqa: E402
from wave2body.visualization.sample_selection import PairedAdvantageSampleSelector  # noqa: E402


def _parse_csv(value: str) -> tuple[str, ...]:
    items = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not items:
        raise ValueError("Expected at least one comma-separated value")
    return items


def _split_config_ref(pattern: str, split_id: str) -> str:
    return str(pattern).format(split=split_id)


def _default_wave_checkpoint(config: dict[str, Any]) -> Path:
    run_name = str(config.get("training", {}).get("run_name", "")).strip()
    if not run_name:
        raise ValueError("Wave2Body config has no training.run_name; pass --wave2body-checkpoint.")
    return artifact_root() / "experiments" / "hpe_token" / "results" / run_name / "checkpoints" / "best.pth"


def _default_wicompass_checkpoint(split_id: str) -> Path:
    return (
        artifact_root()
        / "experiments"
        / "hpe_point_transformer"
        / "results"
        / DEFAULT_PT_RUNS[split_id]
        / "checkpoints"
        / "best.pth"
    )


def _default_output_dir(split_ids: tuple[str, ...]) -> Path:
    split_tag = "_".join(split_ids)
    return artifact_root() / "experiments" / "visualizations" / f"m4human_wicompass_comparison_{split_tag}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-ids", default=",".join(DEFAULT_SPLITS), help="Comma-separated M4Human splits.")
    parser.add_argument("--eval-split", default="test", choices=("train", "val", "test"))
    parser.add_argument(
        "--action-groups",
        default=",".join(DEFAULT_ACTION_GROUPS),
        help="Comma-separated M4Human semantic groups.",
    )
    parser.add_argument(
        "--wave2body-config",
        default="m4human_abs_{split}",
        help="Wave2Body config alias/path. May contain {split}.",
    )
    parser.add_argument("--wave2body-checkpoint", help="Wave2Body checkpoint path. May contain {split}.")
    parser.add_argument("--wicompass-config", default=str(PT_CONFIG), help="WiCompass/PT config path.")
    parser.add_argument("--wicompass-checkpoint", help="WiCompass/PT checkpoint path. May contain {split}.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpus", help='CUDA_VISIBLE_DEVICES value, e.g. "0".')
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, help="Limit eval batches for smoke checks.")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--min-wicompass-mpjpe-mm", type=float, default=100.0)
    parser.add_argument("--min-delta-mpjpe-mm", type=float, default=5.0)
    parser.add_argument("--max-per-action", type=int, default=1)
    parser.add_argument("--max-per-subject", type=int, default=1)
    parser.add_argument("--min-frame-gap", type=int, default=30)
    parser.add_argument("--pool-size", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _prepare_wave2body_config(config: dict[str, Any], *, split_id: str, batch_size: int, num_workers: int) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    training = cfg.setdefault("training", {})
    m4human = cfg.setdefault("m4human", {})
    data_format = cfg.setdefault("data_format", {})
    evaluation = cfg.setdefault("evaluation", {})
    m4human["split"] = split_id
    m4human["temporal_offsets"] = [-3, -2, -1, 0]
    m4human["temporal_window"] = 4
    m4human["pose_source"] = "param_joints"
    m4human["normalize_z"] = True
    data_format["merge_nframes"] = 4
    data_format["input_root_normalized"] = False
    data_format["input_root_crop"] = False
    training["stage"] = "absolute_root_head"
    training["freeze_mmwave_tokenizer"] = True
    training["freeze_token_translator"] = True
    training["freeze_vqvae"] = True
    training["batch_size"] = int(batch_size)
    training["num_workers"] = int(num_workers)
    training["pin_memory"] = bool(num_workers > 0)
    training["persistent_workers"] = bool(num_workers > 0)
    evaluation["primary_metric"] = "absolute_mpjpe"
    evaluation["diagnostic_metrics"] = ["absolute_mpjpe", "root_error", "root_aligned_mpjpe"]
    evaluation["root_index"] = 0
    return cfg


def _prepare_wicompass_config(config: dict[str, Any], *, split_id: str, batch_size: int, num_workers: int) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    training = cfg.setdefault("training", {})
    m4human = cfg.setdefault("m4human", {})
    data_format = cfg.setdefault("data_format", {})
    evaluation = cfg.setdefault("evaluation", {})
    m4human["split"] = split_id
    m4human["frame_mode"] = "hist4"
    m4human["temporal_offsets"] = [-3, -2, -1, 0]
    m4human["temporal_window"] = 4
    m4human["pose_source"] = "param_joints"
    m4human["normalize_z"] = True
    data_format["merge_nframes"] = 4
    data_format["normalized"] = False
    data_format["input_root_normalized"] = False
    data_format["input_root_crop"] = False
    data_format["target_root_normalized"] = False
    training["batch_size"] = int(batch_size)
    training["num_workers"] = int(num_workers)
    training["pin_memory"] = bool(num_workers > 0)
    training["persistent_workers"] = bool(num_workers > 0)
    evaluation["mpjpe_only"] = True
    evaluation["primary_metric"] = "absolute_mpjpe"
    evaluation["diagnostic_metrics"] = ["absolute_mpjpe", "root_error", "root_aligned_mpjpe"]
    evaluation["root_index"] = 0
    return cfg


def _validate_matched_inputs(wave_cfg: dict[str, Any], pt_cfg: dict[str, Any]) -> None:
    checks = (
        ("data_format.n_points", wave_cfg["data_format"].get("n_points"), pt_cfg["data_format"].get("n_points")),
        ("data_format.merge_nframes", wave_cfg["data_format"].get("merge_nframes"), pt_cfg["data_format"].get("merge_nframes")),
        ("data_format.radar_input_c", wave_cfg["data_format"].get("radar_input_c"), pt_cfg["data_format"].get("radar_input_c")),
        (
            "data_format.input_root_normalized",
            wave_cfg["data_format"].get("input_root_normalized"),
            pt_cfg["data_format"].get("input_root_normalized"),
        ),
        ("data_format.input_root_crop", wave_cfg["data_format"].get("input_root_crop"), pt_cfg["data_format"].get("input_root_crop")),
        ("m4human.temporal_offsets", wave_cfg["m4human"].get("temporal_offsets"), pt_cfg["m4human"].get("temporal_offsets")),
        ("m4human.pose_source", wave_cfg["m4human"].get("pose_source"), pt_cfg["m4human"].get("pose_source")),
        ("m4human.normalize_z", wave_cfg["m4human"].get("normalize_z"), pt_cfg["m4human"].get("normalize_z")),
    )
    mismatches = [f"{name}: Wave2Body={left!r}, WiCompass/PT={right!r}" for name, left, right in checks if left != right]
    if mismatches:
        raise ValueError("Wave2Body and WiCompass/PT input configs are not matched:\n" + "\n".join(mismatches))


def _build_loader(config: dict[str, Any], *, eval_split: str, batch_size: int, num_workers: int) -> DataLoader:
    dataset = create_radar_pose_dataset(config, split=eval_split, device="cpu")
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": bool(num_workers > 0),
        "persistent_workers": bool(num_workers > 0),
        "drop_last": False,
    }
    if int(num_workers) > 0:
        kwargs["prefetch_factor"] = int(config.get("training", {}).get("prefetch_factor", 2))
    return DataLoader(**kwargs)


def _load_wave2body_model(config: dict[str, Any], *, checkpoint: Path, device: torch.device) -> AbsoluteRootHeadEstimatorModel:
    model = AbsoluteRootHeadEstimatorModel(
        mmwave_tokenizer_config=config["mmwave_tokenizer"],
        sensor_token_config=config.get("sensor_tokens", {}),
        token_translator_config=config["token_translator"],
        vqvae_config=config["vqvae"],
        root_head_config=config.get("root_head", {}),
        device=str(device),
        freeze_mmwave_tokenizer=True,
        freeze_token_translator=True,
        freeze_vqvae=True,
    )
    pretrained = config.get("pretrained", {})
    model.load_pretrained_weights(
        mmwave_tokenizer_path=pretrained.get("mmwave_tokenizer"),
        vqvae_path=pretrained.get("vqvae"),
        hpe_token_path=pretrained.get("hpe_token"),
        device=str(device),
    )
    ckpt = load_checkpoint(checkpoint, map_location=device)
    _load_absolute_component_states(model, ckpt, prefix="[PairedComparison Wave2Body]")
    return model.to(device).eval()


def _load_wicompass_model(config: dict[str, Any], *, checkpoint: Path, device: torch.device, runtime) -> PTPoseModel:
    model = PTPoseModel(point_transformer_config=config["point_transformer"], device=str(device)).to(device)
    materialize_lazy_modules(model, config=config, device=device, runtime=runtime)
    ckpt = load_checkpoint(checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    if not isinstance(state, dict):
        raise KeyError(f"Checkpoint {checkpoint} does not contain model_state_dict")
    load_state_dict_shape_safe(model, state, module_name="WiCompassPT Eval", strict=False)
    return model.to(device).eval()


def _per_sample_mpjpe(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(pred.float() - target.float(), dim=-1).mean(dim=-1)


def _root_aligned(pose: torch.Tensor, root_index: int) -> torch.Tensor:
    return pose - pose[:, root_index : root_index + 1]


def _metric_values(pred: torch.Tensor, target: torch.Tensor, *, root_index: int = 0) -> dict[str, torch.Tensor]:
    return {
        "mpjpe": _per_sample_mpjpe(pred, target),
        "root_error": torch.linalg.vector_norm(pred[:, root_index].float() - target[:, root_index].float(), dim=-1),
        "root_aligned_mpjpe": _per_sample_mpjpe(_root_aligned(pred, root_index), _root_aligned(target, root_index)),
    }


def _format_sample_name(candidate, rank: int) -> str:
    split = candidate.split_id or "split"
    return (
        f"{split}_{candidate.action_group}_rank{rank:02d}_"
        f"s{candidate.subject_id:02d}_a{candidate.action_id:02d}_f{candidate.frame_id:06d}.npz"
    )


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        path.write_text("", encoding="utf-8")
        return
    keys = list(records[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def _run_split(
    *,
    split_id: str,
    args: argparse.Namespace,
    action_groups: tuple[str, ...],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    wave_config_ref = _split_config_ref(args.wave2body_config, split_id)
    wave_config_path = resolve_wave2body_config(wave_config_ref)
    raw_wave_config = load_config(wave_config_path)
    wave_config = _prepare_wave2body_config(
        raw_wave_config,
        split_id=split_id,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
    wave_checkpoint = (
        Path(resolve_runtime_uri(_split_config_ref(args.wave2body_checkpoint, split_id))).expanduser()
        if args.wave2body_checkpoint
        else _default_wave_checkpoint(wave_config)
    )

    pt_config_path = Path(args.wicompass_config).expanduser()
    pt_config = _prepare_wicompass_config(
        load_config(pt_config_path),
        split_id=split_id,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
    pt_checkpoint = (
        Path(resolve_runtime_uri(_split_config_ref(args.wicompass_checkpoint, split_id))).expanduser()
        if args.wicompass_checkpoint
        else _default_wicompass_checkpoint(split_id)
    )
    _validate_matched_inputs(wave_config, pt_config)
    for path in (wave_checkpoint, pt_checkpoint):
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")
    wave_runtime = configure_runtime(wave_config.get("training", {}), device, default_amp=True, log_prefix="[Paired Wave2Body Runtime]")
    pt_runtime = configure_runtime(pt_config.get("training", {}), device, default_amp=True, log_prefix="[Paired WiCompass Runtime]")
    loader = _build_loader(wave_config, eval_split=args.eval_split, batch_size=int(args.batch_size), num_workers=int(args.num_workers))
    wave_model = _load_wave2body_model(wave_config, checkpoint=wave_checkpoint, device=device)
    pt_model = _load_wicompass_model(pt_config, checkpoint=pt_checkpoint, device=device, runtime=pt_runtime)
    selector = PairedAdvantageSampleSelector(
        top_k=int(args.top_k),
        action_groups=action_groups,
        min_wicompass_mpjpe_mm=float(args.min_wicompass_mpjpe_mm),
        min_delta_mpjpe_mm=float(args.min_delta_mpjpe_mm),
        max_per_action=int(args.max_per_action),
        max_per_subject=int(args.max_per_subject),
        min_frame_gap=int(args.min_frame_gap),
        pool_size=args.pool_size,
        split_id=split_id,
        eval_split=args.eval_split,
        wave2body_config=str(wave_config_path),
        wave2body_checkpoint=str(wave_checkpoint),
        wicompass_config=str(pt_config_path),
        wicompass_checkpoint=str(pt_checkpoint),
    )

    totals = {
        "wave2body_mpjpe": 0.0,
        "wicompass_mpjpe": 0.0,
        "wave2body_root_error": 0.0,
        "wicompass_root_error": 0.0,
        "wave2body_root_aligned_mpjpe": 0.0,
        "wicompass_root_aligned_mpjpe": 0.0,
    }
    sample_count = 0
    t0 = time.perf_counter()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Paired comparison {split_id}/{args.eval_split}")):
            if args.max_batches is not None and batch_idx >= int(args.max_batches):
                break
            mmwave_points = batch["mmwave_points"].to(device, non_blocking=True).float()
            target_absolute = batch["target_poses_absolute"].to(device, non_blocking=True).float()
            target_relative = None
            with autocast_context(wave_runtime):
                wave_outputs = wave_model(mmwave_points, target_relative)
            with autocast_context(pt_runtime):
                pt_outputs = pt_model(mmwave_points)
            wave_pred = wave_outputs["pred_pose_absolute"].float()
            pt_pred = pt_outputs["pred_pose"].float()
            if wave_pred.shape != target_absolute.shape or pt_pred.shape != target_absolute.shape:
                raise ValueError(
                    "Prediction/target shape mismatch: "
                    f"wave={tuple(wave_pred.shape)}, pt={tuple(pt_pred.shape)}, target={tuple(target_absolute.shape)}"
                )
            wave_metrics = _metric_values(wave_pred, target_absolute)
            pt_metrics = _metric_values(pt_pred, target_absolute)
            bsz = int(target_absolute.shape[0])
            for key, values in (
                ("wave2body_mpjpe", wave_metrics["mpjpe"]),
                ("wicompass_mpjpe", pt_metrics["mpjpe"]),
                ("wave2body_root_error", wave_metrics["root_error"]),
                ("wicompass_root_error", pt_metrics["root_error"]),
                ("wave2body_root_aligned_mpjpe", wave_metrics["root_aligned_mpjpe"]),
                ("wicompass_root_aligned_mpjpe", pt_metrics["root_aligned_mpjpe"]),
            ):
                totals[key] += float(values.double().sum().item())
            sample_count += bsz
            selector.update_from_batch(
                batch=batch,
                wave2body_pred_pose=wave_pred.detach(),
                wicompass_pred_pose=pt_pred.detach(),
                gt_pose=target_absolute.detach(),
                mmwave_points=mmwave_points.detach(),
                wave2body_mpjpe=wave_metrics["mpjpe"].detach(),
                wicompass_mpjpe=pt_metrics["mpjpe"].detach(),
                wave2body_root_error=wave_metrics["root_error"].detach(),
                wicompass_root_error=pt_metrics["root_error"].detach(),
                wave2body_root_aligned_mpjpe=wave_metrics["root_aligned_mpjpe"].detach(),
                wicompass_root_aligned_mpjpe=pt_metrics["root_aligned_mpjpe"].detach(),
            )

    records: list[dict[str, Any]] = []
    samples_dir = output_dir / "samples"
    for group, candidates in selector.selected().items():
        for rank, candidate in enumerate(candidates, start=1):
            npz_path = samples_dir / _format_sample_name(candidate, rank)
            candidate.save_npz(npz_path)
            records.append(candidate.record(npz_path=str(npz_path.relative_to(output_dir))))

    denom = float(max(1, sample_count))
    summary = {
        "split_id": split_id,
        "eval_split": args.eval_split,
        "num_samples": int(sample_count),
        "num_seen_samples": int(selector.seen_count),
        "num_eligible_samples": int(selector.eligible_count),
        "num_selected_samples": int(len(records)),
        "samples_per_sec": float(sample_count / max(time.perf_counter() - t0, 1e-6)),
        "wave2body_mpjpe_mm": float(totals["wave2body_mpjpe"] / denom * 1000.0),
        "wicompass_mpjpe_mm": float(totals["wicompass_mpjpe"] / denom * 1000.0),
        "delta_mpjpe_mm": float((totals["wicompass_mpjpe"] - totals["wave2body_mpjpe"]) / denom * 1000.0),
        "wave2body_config": str(wave_config_path),
        "wave2body_checkpoint": str(wave_checkpoint),
        "wicompass_config": str(pt_config_path),
        "wicompass_checkpoint": str(pt_checkpoint),
    }
    return records, summary


def main() -> None:
    args = _build_parser().parse_args()
    split_ids = _parse_csv(args.split_ids)
    action_groups = _parse_csv(args.action_groups)
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else _default_output_dir(split_ids)
    payload = {
        "split_ids": list(split_ids),
        "eval_split": args.eval_split,
        "action_groups": list(action_groups),
        "wave2body_config": args.wave2body_config,
        "wave2body_checkpoint": args.wave2body_checkpoint,
        "wicompass_config": args.wicompass_config,
        "wicompass_checkpoint": args.wicompass_checkpoint,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "max_batches": args.max_batches,
        "top_k": int(args.top_k),
        "min_wicompass_mpjpe_mm": float(args.min_wicompass_mpjpe_mm),
        "min_delta_mpjpe_mm": float(args.min_delta_mpjpe_mm),
        "max_per_action": int(args.max_per_action),
        "max_per_subject": int(args.max_per_subject),
        "min_frame_gap": int(args.min_frame_gap),
        "pool_size": args.pool_size,
        "output_dir": str(output_dir),
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for split_id in split_ids:
        split_records, split_summary = _run_split(
            split_id=split_id,
            args=args,
            action_groups=action_groups,
            output_dir=output_dir,
        )
        records.extend(split_records)
        summaries.append(split_summary)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
        "num_selected_samples": int(len(records)),
        "split_summaries": summaries,
        "samples": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_csv(output_dir / "selected_samples.csv", records)
    print(f"[Paired Comparison] Wrote {manifest_path}")
    print(f"[Paired Comparison] Selected {len(records)} samples")


if __name__ == "__main__":
    main()
