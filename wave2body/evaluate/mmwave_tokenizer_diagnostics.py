#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compute mmWave tokenizer coverage and diversity diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


from ..models.mmwave_tokenizer import create_mmwave_tokenizer, valid_point_mask
from ..train.checkpoint import load_checkpoint
from ..train.common import load_config, write_run_manifest
from ..train.radar_pose_data import dataset_name
from ..train.mmwave_tokenizer_data import create_mmwave_tokenizer_dataset
from ..utils.paths import artifact_root
from ..utils.runtime_utils import configure_runtime


DIAGNOSTICS_STAGE = "mmwave_tokenizer_diagnostics"
DEFAULT_OUTPUT_DIR = artifact_root() / "experiments" / DIAGNOSTICS_STAGE


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="Single mmWave tokenizer checkpoint")
    parser.add_argument("--checkpoint-map", default=None, help="sweep_results.json to evaluate all ok rows")
    parser.add_argument("--config", default=None, help="Optional config override for a single checkpoint")
    parser.add_argument("--label", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=None)
    parser.add_argument("--body-thresholds-m", default="0.15,0.30,0.50")
    parser.add_argument("--duplicate-center-thresholds-m", default="0.02,0.05")
    parser.add_argument("--embedding-duplicate-cos", type=float, default=0.98)
    return parser


def _load_sweep_rows(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path).expanduser()
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get("results") or data.get("records") or data.get("trials")
    else:
        rows = None
    if not isinstance(rows, list):
        raise TypeError(f"Unsupported checkpoint-map JSON shape: {p}")
    out = []
    for row in rows:
        if str(row.get("status")) not in {"ok", "skipped_existing"}:
            continue
        ckpt = row.get("best_ckpt") or row.get("checkpoint")
        if not ckpt:
            continue
        out.append(dict(row))
    return out


def _load_eval_config(config_path: str | None, checkpoint: dict[str, Any]) -> dict[str, Any]:
    if config_path:
        config = load_config(config_path)
    else:
        config = checkpoint.get("config")
        if not isinstance(config, dict):
            raise KeyError("Checkpoint does not contain config; pass --config.")
    if str(config.get("dataset", {}).get("name", "")).lower() == "m4human":
        m4 = config.setdefault("m4human", {})
        m4["load_pose"] = True
        m4.setdefault("pose_source", "param_joints")
    training = config.setdefault("training", {})
    training["batch_size"] = int(training.get("batch_size", 64))
    training["num_workers"] = int(training.get("num_workers", 0))
    return config


def _build_loader(config: dict[str, Any], split: str, batch_size: int, num_workers: int) -> DataLoader:
    dataset = create_mmwave_tokenizer_dataset(config=config, split=split, device="cpu")
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=int(num_workers) > 0,
        persistent_workers=int(num_workers) > 0,
        drop_last=False,
    )


def _parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def _safe_percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _sample_stats(
    *,
    points: torch.Tensor,
    centers: torch.Tensor,
    embeddings: torch.Tensor,
    center_mask: torch.Tensor,
    target_pose: torch.Tensor | None,
    body_thresholds: list[float],
    duplicate_thresholds: list[float],
    embedding_duplicate_cos: float,
) -> dict[str, Any] | None:
    point_mask = valid_point_mask(points.unsqueeze(0)).squeeze(0)
    xyz = points[point_mask, :3].float()
    cmask = center_mask.bool()
    c = centers[cmask, :3].float()
    if xyz.numel() == 0 or c.numel() == 0:
        return None
    k_valid = int(c.shape[0])
    k_total = int(centers.shape[0])
    dist_pc = torch.cdist(xyz.unsqueeze(0), c.unsqueeze(0)).squeeze(0)
    nearest_point_dist, assignments = torch.min(dist_pc, dim=1)
    counts = torch.bincount(assignments, minlength=k_valid).float()
    probs = counts / counts.sum().clamp_min(1.0)
    entropy = -(probs[probs > 0] * torch.log(probs[probs > 0])).sum()
    entropy_norm = entropy / torch.log(torch.tensor(float(max(k_valid, 2)), device=entropy.device))
    empty_assignment_frac = float((counts == 0).float().mean().item())

    center_dup: dict[str, float] = {}
    if k_valid > 1:
        dist_cc = torch.cdist(c.unsqueeze(0), c.unsqueeze(0)).squeeze(0)
        dist_cc.fill_diagonal_(float("inf"))
        nearest_center = dist_cc.min(dim=1).values
        for threshold in duplicate_thresholds:
            key = f"duplicate_center_frac_{int(round(threshold * 100)):02d}cm"
            center_dup[key] = float((nearest_center <= float(threshold)).float().mean().item())
    else:
        for threshold in duplicate_thresholds:
            key = f"duplicate_center_frac_{int(round(threshold * 100)):02d}cm"
            center_dup[key] = 0.0

    e = embeddings[cmask].float()
    if k_valid > 1 and e.numel() > 0:
        e = torch.nn.functional.normalize(e, dim=-1)
        sim = e @ e.t()
        sim.fill_diagonal_(-float("inf"))
        emb_dup = float((sim.max(dim=1).values >= float(embedding_duplicate_cos)).float().mean().item())
    else:
        emb_dup = 0.0

    body_stats: dict[str, float] = {}
    if target_pose is not None and target_pose.numel() > 0:
        joints = target_pose[:, :3].float()
        body_dist = torch.cdist(c.unsqueeze(0), joints.unsqueeze(0)).squeeze(0).min(dim=1).values
        for threshold in body_thresholds:
            key = f"body_near_frac_{int(round(threshold * 100)):02d}cm"
            body_stats[key] = float((body_dist <= float(threshold)).float().mean().item())
    else:
        for threshold in body_thresholds:
            key = f"body_near_frac_{int(round(threshold * 100)):02d}cm"
            body_stats[key] = float("nan")

    return {
        "valid_token_frac": float(k_valid / max(1, k_total)),
        "empty_assignment_frac": empty_assignment_frac,
        "assignment_entropy_norm": float(entropy_norm.item()),
        "coverage_values": nearest_point_dist.detach().cpu().float().tolist(),
        "coverage_mean": float(nearest_point_dist.mean().item()),
        **center_dup,
        "embedding_duplicate_frac_cos098": emb_dup,
        **body_stats,
    }


@torch.no_grad()
def _evaluate_one(
    *,
    checkpoint_path: Path,
    config_path: str | None,
    label: str | None,
    split: str,
    batch_size: int,
    num_workers: int,
    max_batches: int | None,
    device_arg: str,
    body_thresholds: list[float],
    duplicate_thresholds: list[float],
    embedding_duplicate_cos: float,
) -> dict[str, Any]:
    requested = torch.device(device_arg)
    device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    config = _load_eval_config(config_path, ckpt)
    runtime_cfg = dict(config.get("training", {}))
    runtime_cfg["compile"] = {"enabled": False}
    configure_runtime(runtime_cfg, device, default_amp=False, log_prefix="[MMWaveTokenizer Diagnostics]")
    model = create_mmwave_tokenizer(config["mmwave_tokenizer"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    loader = _build_loader(config, split, batch_size, num_workers)

    accum: dict[str, list[float]] = {}
    coverage_values: list[float] = []
    samples = 0
    for batch_idx, batch in enumerate(tqdm(loader, desc=f"MMWaveTokenizerDiagnostics({label or checkpoint_path.parent.name})")):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        points = batch["mmwave_points"].to(device, non_blocking=True).float()
        target_abs = batch.get("target_poses_absolute")
        target_abs = target_abs.to(device, non_blocking=True).float() if torch.is_tensor(target_abs) else None
        if hasattr(model, "encode_tokens"):
            outputs = model.encode_tokens(points, target_poses_absolute=target_abs)
        else:
            outputs = model(points, target_poses_absolute=target_abs)
        centers = outputs["centers"].detach()
        embeddings = outputs["token_embeddings"].detach()
        valid_mask = outputs["valid_mask"].detach()
        for i in range(points.shape[0]):
            row = _sample_stats(
                points=points[i],
                centers=centers[i],
                embeddings=embeddings[i],
                center_mask=valid_mask[i],
                target_pose=target_abs[i] if target_abs is not None else None,
                body_thresholds=body_thresholds,
                duplicate_thresholds=duplicate_thresholds,
                embedding_duplicate_cos=embedding_duplicate_cos,
            )
            if row is None:
                continue
            samples += 1
            coverage_values.extend(row.pop("coverage_values"))
            for key, value in row.items():
                if np.isfinite(value):
                    accum.setdefault(key, []).append(float(value))

    if samples <= 0:
        raise RuntimeError("No diagnostic samples were evaluated.")

    metrics = {key: float(np.mean(values)) for key, values in sorted(accum.items()) if values}
    metrics.update(
        {
            "coverage_mean": float(np.mean(coverage_values)) if coverage_values else float("nan"),
            "coverage_p50": _safe_percentile(coverage_values, 50),
            "coverage_p90": _safe_percentile(coverage_values, 90),
            "coverage_p95": _safe_percentile(coverage_values, 95),
        }
    )
    pt_cfg = config.get("mmwave_tokenizer", {})
    return {
        "label": label or checkpoint_path.parent.parent.name,
        "k": int(pt_cfg.get("num_groups", -1)),
        "method": str(pt_cfg.get("method", "")),
        "patch_policy": str(pt_cfg.get("patch_policy", "")),
        "dataset": dataset_name(config),
        "split": split,
        "num_samples": int(samples),
        "checkpoint": str(checkpoint_path.resolve()),
        "config": str(Path(config_path).resolve()) if config_path else str(ckpt.get("config_path", "")),
        "metrics": metrics,
    }


def _write_outputs(records: list[dict[str, Any]], out_path: Path, args: argparse.Namespace) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records if len(records) != 1 else records[0], indent=2), encoding="utf-8")
    csv_path = out_path.with_suffix(".csv")
    fieldnames = [
        "label",
        "k",
        "method",
        "patch_policy",
        "dataset",
        "split",
        "num_samples",
        "checkpoint",
    ]
    metric_keys = sorted({key for row in records for key in row.get("metrics", {})})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[*fieldnames, *metric_keys])
        writer.writeheader()
        for row in records:
            flat = {key: row.get(key) for key in fieldnames}
            flat.update(row.get("metrics", {}))
            writer.writerow(flat)
    lines = [
        "# mmWave Tokenizer Diagnostics",
        "",
        "| label | K | samples | valid_token_frac | empty_assignment_frac | assignment_entropy_norm | coverage_p90 | duplicate_center_frac_05cm | embedding_duplicate_frac_cos098 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        m = row.get("metrics", {})
        lines.append(
            "| {label} | {k} | {samples} | {valid:.3f} | {empty:.3f} | {entropy:.3f} | {cov:.4f} | {dup:.3f} | {emb:.3f} |".format(
                label=row.get("label"),
                k=int(row.get("k", -1)),
                samples=int(row.get("num_samples", 0)),
                valid=float(m.get("valid_token_frac", float("nan"))),
                empty=float(m.get("empty_assignment_frac", float("nan"))),
                entropy=float(m.get("assignment_entropy_norm", float("nan"))),
                cov=float(m.get("coverage_p90", float("nan"))),
                dup=float(m.get("duplicate_center_frac_05cm", float("nan"))),
                emb=float(m.get("embedding_duplicate_frac_cos098", float("nan"))),
            )
        )
    out_path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_run_manifest(
        out_path.parent,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        config={"stage": DIAGNOSTICS_STAGE},
        dataset_split=args.split,
        coord_transform=None,
        output_paths={"json": out_path, "csv": csv_path, "markdown": out_path.with_suffix(".md")},
        extra={"max_batches": args.max_batches, "checkpoint_map": args.checkpoint_map},
        filename=f"{out_path.stem}.manifest.json",
    )


def main() -> None:
    args = _build_parser().parse_args()
    if not args.checkpoint and not args.checkpoint_map:
        raise ValueError("Pass --checkpoint or --checkpoint-map.")
    body_thresholds = _parse_float_list(args.body_thresholds_m)
    duplicate_thresholds = _parse_float_list(args.duplicate_center_thresholds_m)
    records: list[dict[str, Any]] = []
    if args.checkpoint_map:
        for row in _load_sweep_rows(args.checkpoint_map):
            records.append(
                _evaluate_one(
                    checkpoint_path=Path(str(row.get("best_ckpt") or row.get("checkpoint"))).expanduser(),
                    config_path=None,
                    label=str(row.get("trial_id", "")),
                    split=args.split,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    max_batches=args.max_batches,
                    device_arg=args.device,
                    body_thresholds=body_thresholds,
                    duplicate_thresholds=duplicate_thresholds,
                    embedding_duplicate_cos=args.embedding_duplicate_cos,
                )
            )
    else:
        records.append(
            _evaluate_one(
                checkpoint_path=Path(str(args.checkpoint)).expanduser(),
                config_path=args.config,
                label=args.label,
                split=args.split,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_batches=args.max_batches,
                device_arg=args.device,
                body_thresholds=body_thresholds,
                duplicate_thresholds=duplicate_thresholds,
                embedding_duplicate_cos=args.embedding_duplicate_cos,
            )
        )
    out_path = Path(args.out).expanduser() if args.out else DEFAULT_OUTPUT_DIR / "diagnostics_results.json"
    _write_outputs(records, out_path, args)
    print(json.dumps(records if len(records) != 1 else records[0], indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
