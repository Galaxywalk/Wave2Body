#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Select and render Wave2Body qualitative samples across error buckets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

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
from scripts.hpe_token.evaluate import (  # noqa: E402
    _apply_eval_overrides,
    _is_absolute_root_config,
    _validate_production_hpe_token_config,
)
from wave2body.train.common import load_config  # noqa: E402
from wave2body.train.pipelines.absolute_root_head import run_absolute_root_head_evaluation  # noqa: E402
from wave2body.utils.paths import artifact_root, resolve_runtime_uri  # noqa: E402
from wave2body.visualization.rendering import load_sample_npz, render_sample_axis, sample_bounds  # noqa: E402
from wave2body.visualization.sample_selection import (  # noqa: E402
    ErrorBucketExportCollector,
    ErrorBucketMetricCollector,
    ErrorBucketRecord,
)


DATASET_SPECS = {
    "m4human_s2": {
        "display": "M4Human",
        "config_default": "m4human_abs_s2",
        "checkpoint_arg": "m4human_checkpoint",
        "split_id": "s2",
    },
    "mmbody": {
        "display": "mmBody",
        "config_default": "mmbody_abs",
        "checkpoint_arg": "mmbody_checkpoint",
        "split_id": None,
    },
}

CSV_FIELDS = [
    "dataset",
    "dataset_display",
    "bin_index",
    "global_index",
    "selection_metric",
    "bin_start_error_mm",
    "bin_end_error_mm",
    "bin_center_error_mm",
    "absolute_mpjpe_mm",
    "root_error_mm",
    "root_aligned_mpjpe_mm",
    "point_count",
    "rank",
    "fallback_low_points",
    "npz_path",
    "metadata_json",
]


@dataclass
class DatasetRun:
    key: str
    display: str
    config_name: str
    config_path: Path
    config: dict[str, Any]
    checkpoint: Path
    output_dir: Path


def _parse_datasets(value: str) -> list[str]:
    raw = [part.strip() for part in str(value).split(",") if part.strip()]
    if not raw or "all" in raw:
        return ["m4human_s2", "mmbody"]
    invalid = [item for item in raw if item not in DATASET_SPECS]
    if invalid:
        raise ValueError(f"Unknown datasets {invalid}; expected all,{','.join(DATASET_SPECS)}")
    return raw


def _default_checkpoint(config: dict[str, Any]) -> Path:
    run_name = str(config.get("training", {}).get("run_name", "")).strip()
    if not run_name:
        raise ValueError("Checkpoint is required because config.training.run_name is empty")
    return artifact_root() / "experiments" / "hpe_token" / "results" / run_name / "checkpoints" / "best.pth"


def _metric_name(value: str) -> str:
    value = str(value).strip().lower()
    aliases = {
        "mpjpe": "absolute_mpjpe_mm",
        "absolute_mpjpe": "absolute_mpjpe_mm",
        "absolute_mpjpe_mm": "absolute_mpjpe_mm",
        "root": "root_error_mm",
        "root_error": "root_error_mm",
        "root_error_mm": "root_error_mm",
        "root_aligned": "root_aligned_mpjpe_mm",
        "root_aligned_mpjpe": "root_aligned_mpjpe_mm",
        "root_aligned_mpjpe_mm": "root_aligned_mpjpe_mm",
    }
    if value not in aliases:
        raise ValueError(f"Unknown metric {value!r}; expected one of {sorted(aliases)}")
    return aliases[value]


def _metric_label(metric: str) -> str:
    if metric == "root_error_mm":
        return "Root"
    if metric == "root_aligned_mpjpe_mm":
        return "Root-aligned MPJPE"
    return "MPJPE"


def _eval_args_for(run_key: str, args: argparse.Namespace) -> argparse.Namespace:
    spec = DATASET_SPECS[run_key]
    return argparse.Namespace(
        checkpoint=None,
        split_id=spec["split_id"],
        eval_split=args.eval_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_points=args.n_points,
        merge_nframes=args.merge_nframes,
        point_sampling=args.point_sampling,
        mmwave_tokenizer_checkpoint=args.mmwave_tokenizer_checkpoint,
        body_vqvae_checkpoint=args.body_vqvae_checkpoint,
        point_ensemble_n_points=None,
        point_ensemble_weights=None,
        checkpoint_ensemble=None,
        checkpoint_ensemble_weights=None,
        logit_decode_diagnostics=False,
        logit_decode_temperatures="0.5,1.0",
        logit_blend_weights=None,
        token_topk=None,
        token_topk_no_slots=False,
        token_topk_oracle=False,
        no_action_groups=True,
        per_action=False,
        output=None,
        max_batches=args.max_batches,
    )


def _prepare_run(run_key: str, args: argparse.Namespace, output_root: Path) -> DatasetRun:
    spec = DATASET_SPECS[run_key]
    config_name = getattr(args, f"{run_key}_config") or spec["config_default"]
    config_path = resolve_config(config_name)
    config = _apply_eval_overrides(load_config(config_path), _eval_args_for(run_key, args))
    _validate_production_hpe_token_config(config)
    if not _is_absolute_root_config(config):
        raise ValueError(f"{run_key} config must be an absolute root-head Wave2Body config: {config_path}")
    checkpoint_raw = getattr(args, spec["checkpoint_arg"]) or args.checkpoint
    checkpoint = Path(resolve_runtime_uri(checkpoint_raw)).expanduser() if checkpoint_raw else _default_checkpoint(config)
    return DatasetRun(
        key=run_key,
        display=str(spec["display"]),
        config_name=str(config_name),
        config_path=config_path,
        config=config,
        checkpoint=checkpoint,
        output_dir=output_root / run_key,
    )


def _record_to_row(record: ErrorBucketRecord, base_dir: Path | None = None) -> dict[str, Any]:
    payload = record.record()
    npz_path = payload.get("npz_path", "")
    if npz_path and base_dir is not None:
        p = Path(str(npz_path))
        try:
            npz_path = str(p.relative_to(base_dir)) if p.is_absolute() else str(p)
        except ValueError:
            npz_path = str(p)
    return {
        "dataset": payload["dataset"],
        "dataset_display": payload["dataset_display"],
        "bin_index": payload.get("bin_index"),
        "global_index": payload["global_index"],
        "selection_metric": payload["selection_metric"],
        "bin_start_error_mm": payload.get("bin_start_error_mm"),
        "bin_end_error_mm": payload.get("bin_end_error_mm"),
        "bin_center_error_mm": payload.get("bin_center_error_mm"),
        "absolute_mpjpe_mm": payload["absolute_mpjpe_mm"],
        "root_error_mm": payload["root_error_mm"],
        "root_aligned_mpjpe_mm": payload["root_aligned_mpjpe_mm"],
        "point_count": payload["point_count"],
        "rank": payload.get("rank"),
        "fallback_low_points": payload.get("fallback_low_points"),
        "npz_path": npz_path,
        "metadata_json": json.dumps(payload.get("metadata", {}), sort_keys=True),
    }


def _write_csv(path: Path, records: list[ErrorBucketRecord], *, base_dir: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(_record_to_row(record, base_dir=base_dir))


def _run_eval(run: DatasetRun, *, args: argparse.Namespace, output_path: Path, collector: Any) -> dict[str, Any]:
    if not run.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found for {run.display}: {run.checkpoint}")
    return run_absolute_root_head_evaluation(
        run.config,
        checkpoint=run.checkpoint,
        config_path=run.config_path,
        eval_split=args.eval_split,
        device=args.device,
        output_path=output_path,
        max_eval_batches=args.max_batches,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        include_per_action=False,
        sample_collector=collector,
    )


def _select_run(run: DatasetRun, args: argparse.Namespace) -> tuple[list[ErrorBucketRecord], list[ErrorBucketRecord], dict[str, Any]]:
    run.output_dir.mkdir(parents=True, exist_ok=True)
    metric = _metric_name(args.metric)
    collector = ErrorBucketMetricCollector(
        dataset=run.key,
        dataset_display=run.display,
        bins=args.bins,
        metric=metric,
        min_points=args.min_points,
        top_k_candidates=args.top_k_candidates,
    )
    payload = _run_eval(run, args=args, output_path=run.output_dir / "eval_metrics.json", collector=collector)
    selected, candidates = collector.selected()
    _write_csv(run.output_dir / "selected_samples.csv", selected)
    _write_csv(run.output_dir / "candidate_samples.csv", candidates)
    return selected, candidates, payload


def _export_run(run: DatasetRun, selected: list[ErrorBucketRecord], args: argparse.Namespace) -> list[ErrorBucketRecord]:
    samples_dir = run.output_dir / "samples"
    collector = ErrorBucketExportCollector(selected=selected, output_dir=samples_dir)
    _run_eval(run, args=args, output_path=run.output_dir / "eval_metrics_export_pass.json", collector=collector)
    exported = sorted(collector.exported, key=lambda item: (item.bin_index or 0, item.global_index))
    if len(exported) != len(selected):
        missing = sorted(set(item.global_index for item in selected) - set(item.global_index for item in exported))
        raise RuntimeError(f"Exported {len(exported)}/{len(selected)} selected samples for {run.display}; missing={missing[:10]}")
    _write_csv(run.output_dir / "selected_samples_exported.csv", exported, base_dir=run.output_dir)
    return exported


def _resolve_npz_path(path_value: str, root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else root / path


def _gt_root_crop_bounds(sample: dict[str, Any], *, half_extent: float) -> tuple[np.ndarray, np.ndarray]:
    gt_pose = np.asarray(sample["gt_pose"], dtype=np.float32)
    if gt_pose.ndim != 2 or gt_pose.shape[0] == 0 or gt_pose.shape[1] != 3:
        return sample_bounds(sample)
    center = gt_pose[0]
    if not bool(np.isfinite(center).all()):
        return sample_bounds(sample)
    radius = max(float(half_extent), 1.0e-3)
    pad = np.full(3, radius, dtype=np.float32)
    return center - pad, center + pad


def _add_panel_frame(fig, ax, *, linewidth: float = 0.55) -> None:
    frame_ax = fig.add_axes(ax.get_position(), frameon=False)
    frame_ax.set_axis_off()
    frame = Rectangle(
        (0.0, 0.0),
        1.0,
        1.0,
        transform=frame_ax.transAxes,
        fill=False,
        edgecolor="#000000",
        linewidth=float(linewidth),
        clip_on=False,
    )
    frame_ax.add_patch(frame)


def _parse_drop_columns(value: str | None) -> set[int]:
    if value is None or not str(value).strip():
        return set()
    columns: set[int] = set()
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx <= 0:
            raise ValueError("drop columns are 1-based and must be positive")
        columns.add(idx)
    return columns


def _render_groups(rows: list[dict[str, Any]], *, drop_columns: set[int] | None = None) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["dataset"]), []).append(row)
    ordered: list[tuple[str, list[dict[str, Any]]]] = []
    dropped = drop_columns or set()
    for dataset in sorted(groups):
        dataset_rows = sorted(groups[dataset], key=lambda row: int(row["bin_index"]))
        if dropped:
            dataset_rows = [row for pos, row in enumerate(dataset_rows, start=1) if pos not in dropped]
        if dataset_rows:
            ordered.append((dataset, dataset_rows))
    return ordered


def _draw_error_bar(ax, rows: list[dict[str, Any]], *, metric: str, args: argparse.Namespace) -> None:
    errors = [float(row[metric]) for row in rows]
    if not errors:
        ax.set_axis_off()
        return
    n = len(errors)
    x = list(range(n))
    min_error = min(errors)
    max_error = max(errors)
    span = max(max_error - min_error, 1e-6)
    heights = [0.18 + 0.76 * ((value - min_error) / span) for value in errors]
    ax.bar(x, heights, width=0.46, color="#343434", alpha=0.74, linewidth=0)
    ax.plot(x, heights, color="#343434", linewidth=0.45, alpha=0.85)
    ax.scatter(x, heights, s=4.0, color="#343434", linewidths=0)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0.0, 1.08)
    ax.set_axis_off()
    ax.text(
        -0.72,
        0.50,
        f"{_metric_label(metric)}",
        ha="right",
        va="center",
        fontsize=args.error_label_size,
        color="#282828",
        transform=ax.transData,
    )
    ax.text(0, -0.08, f"{errors[0]:.0f}", ha="center", va="top", fontsize=args.error_label_size, color="#282828")
    ax.text(
        n - 1,
        -0.08,
        f"{errors[-1]:.0f} mm",
        ha="center",
        va="top",
        fontsize=args.error_label_size,
        color="#282828",
    )


def _render_rows(rows: list[dict[str, Any]], output_prefix: Path, *, title: str, args: argparse.Namespace) -> list[Path]:
    rows = sorted(rows, key=lambda row: (str(row["dataset"]), int(row["bin_index"])))
    if not rows:
        return []
    groups = _render_groups(rows, drop_columns=_parse_drop_columns(args.drop_columns))
    max_cols = max(len(group_rows) for _, group_rows in groups)
    metric = _metric_name(args.metric)
    rows_per_group = 1
    height_ratios = [1.0 for _dataset, _group_rows in groups]
    fig_height = max(1.45, len(groups) * args.row_height + 0.18)
    fig = plt.figure(figsize=(args.figure_width, fig_height), facecolor="white")
    grid = fig.add_gridspec(
        len(height_ratios),
        max_cols,
        height_ratios=height_ratios,
        left=0.060,
        right=0.995,
        top=0.990,
        bottom=0.025,
        wspace=args.sample_wspace,
        hspace=args.modality_hspace,
    )
    for group_idx, (_dataset, group_rows) in enumerate(groups):
        base_row = group_idx * rows_per_group
        display = str(group_rows[0].get("dataset_display", _dataset))
        if display == "M4Human S2":
            display = "M4Human"
        samples = [load_sample_npz(row["npz_abs_path"]) for row in group_rows]
        bounds_list = [_gt_root_crop_bounds(sample, half_extent=args.crop_half_extent) for sample in samples]
        for col_idx, (row, sample, bounds) in enumerate(zip(group_rows, samples, bounds_list, strict=True)):
            ax = fig.add_subplot(grid[base_row, col_idx], projection="3d")
            render_sample_axis(
                ax,
                sample,
                title=None,
                bounds=bounds,
                show_points=True,
                show_gt=True,
                show_pred=True,
                point_size=args.point_size,
                point_alpha=args.point_alpha,
                gt_linewidth=args.gt_linewidth,
                gt_joint_size=args.gt_joint_size,
                gt_joint_edge_width=args.gt_joint_edge_width,
                pred_linewidth=args.pred_linewidth,
                pred_joint_size=args.pred_joint_size,
                pred_joint_edge_width=args.pred_joint_edge_width,
                skeleton_outline=args.skeleton_outline,
            )
            try:
                ax.set_box_aspect((1.0, 1.0, 1.0), zoom=float(args.axis_zoom))
            except TypeError:
                ax.set_box_aspect((1.0, 1.0, 1.0))
            metric_value = float(row[metric])
            ax.text2D(
                0.045,
                0.935,
                f"{metric_value:.0f} mm",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=args.metric_value_size,
                color="#282828",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.74, "pad": 0.8},
            )
            if col_idx == 0:
                ax.text2D(
                    -0.105,
                    0.50,
                    display,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=args.title_size,
                    fontweight="semibold",
                    color="#282828",
                )
            _add_panel_frame(fig, ax, linewidth=args.panel_frame_width)
        for col_idx in range(len(group_rows), max_cols):
            ax = fig.add_subplot(grid[base_row, col_idx], projection="3d")
            ax.set_axis_off()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in args.formats:
        out = output_prefix.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight", pad_inches=args.pad_inches, facecolor="white")
        paths.append(out)
    plt.close(fig)
    return paths

def _render_manifest(manifest_path: Path, args: argparse.Namespace) -> list[Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    rows = []
    for sample in manifest.get("samples", []):
        if not sample.get("npz_path"):
            continue
        row = dict(sample)
        npz = _resolve_npz_path(str(sample["npz_path"]), root)
        row["npz_abs_path"] = str(npz)
        rows.append(row)
    outputs: list[Path] = []
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)
    for dataset, dataset_rows in by_dataset.items():
        display = str(dataset_rows[0].get("dataset_display", dataset))
        outputs.extend(_render_rows(dataset_rows, root / f"{dataset}_error_buckets", title=f"{display} Error Buckets", args=args))
    if len(by_dataset) > 1:
        outputs.extend(_render_rows(rows, root / "combined_error_buckets", title="Wave2Body Error Bucket Samples", args=args))
    return outputs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("all", "select", "render"), default="all")
    parser.add_argument("--datasets", default="all", help="Comma-separated: all,m4human_s2,mmbody")
    parser.add_argument("--m4human-s2-config", dest="m4human_s2_config", default="m4human_abs_s2")
    parser.add_argument("--mmbody-config", default="mmbody_abs")
    parser.add_argument("--checkpoint", help="Common Wave2Body absolute checkpoint override.")
    parser.add_argument("--m4human-checkpoint", help="M4Human S2 checkpoint override.")
    parser.add_argument("--mmbody-checkpoint", help="mmBody checkpoint override.")
    parser.add_argument("--eval-split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpus", help='CUDA_VISIBLE_DEVICES value, e.g. "0".')
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-batches", type=int, help="Smoke-test limit for each eval pass.")
    parser.add_argument("--n-points", type=int)
    parser.add_argument("--merge-nframes", type=int)
    parser.add_argument("--point-sampling")
    parser.add_argument("--mmwave-tokenizer-checkpoint")
    parser.add_argument("--body-vqvae-checkpoint")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--metric", default="absolute_mpjpe", help="absolute_mpjpe/mpjpe, root_error, or root_aligned_mpjpe.")
    parser.add_argument("--min-points", type=int, default=10)
    parser.add_argument("--top-k-candidates", type=int, default=10)
    parser.add_argument("--output-dir", help="Default: artifact visualizations/error_bucket_pose_figure_<timestamp>.")
    parser.add_argument("--manifest", help="Manifest to render when command=render.")
    parser.add_argument("--formats", default="png,pdf", help="Comma-separated output formats.")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--pad-inches", type=float, default=0.04)
    parser.add_argument("--figure-width", type=float, default=10.8)
    parser.add_argument("--row-height", type=float, default=1.28, help="Height in inches for each overlaid sample row.")
    parser.add_argument("--error-bar-height", type=float, default=0.16, help="Deprecated; kept for CLI compatibility.")
    parser.add_argument("--sample-wspace", type=float, default=-0.10)
    parser.add_argument("--modality-hspace", type=float, default=-0.18)
    parser.add_argument("--title-size", type=float, default=8.6)
    parser.add_argument("--row-label-size", type=float, default=6.2)
    parser.add_argument("--error-label-size", type=float, default=5.4)
    parser.add_argument("--metric-value-size", type=float, default=7.4)
    parser.add_argument("--crop-half-extent", type=float, default=0.95, help="Half side length in meters for GT-root-centered crop boxes.")
    parser.add_argument("--axis-zoom", type=float, default=1.25, help="Uniform 3D axis zoom applied inside each sample panel.")
    parser.add_argument("--panel-frame-width", type=float, default=0.85, help="Black frame linewidth around each sample panel.")
    parser.add_argument("--drop-columns", default="2,7", help="1-based rendered sample columns to drop, comma-separated.")
    parser.add_argument("--point-size", type=float, default=1.90)
    parser.add_argument("--point-alpha", type=float, default=0.24)
    parser.add_argument("--gt-linewidth", type=float, default=2.05)
    parser.add_argument("--gt-joint-size", type=float, default=10.5)
    parser.add_argument("--gt-joint-edge-width", type=float, default=0.30)
    parser.add_argument("--pred-linewidth", type=float, default=1.78)
    parser.add_argument("--pred-joint-size", type=float, default=8.2)
    parser.add_argument("--pred-joint-edge-width", type=float, default=0.28)
    parser.add_argument("--skeleton-outline", action="store_true", help="Use dark skeleton outlines in compact overlays.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    args.formats = tuple(part.strip().lstrip(".") for part in str(args.formats).split(",") if part.strip())
    if args.command == "render":
        if not args.manifest:
            raise SystemExit("--manifest is required for render")
        for path in _render_manifest(Path(args.manifest).expanduser(), args):
            print(f"[Error Bucket Render] Wrote {path}")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(resolve_runtime_uri(args.output_dir)).expanduser()
        if args.output_dir
        else artifact_root() / "experiments" / "visualizations" / f"error_bucket_pose_figure_{timestamp}"
    )
    dataset_keys = _parse_datasets(args.datasets)
    runs = [_prepare_run(key, args, output_root) for key in dataset_keys]
    plan = {
        "output_dir": str(output_root),
        "datasets": [
            {
                "key": run.key,
                "display": run.display,
                "config": str(run.config_path),
                "checkpoint": str(run.checkpoint),
                "eval_split": args.eval_split,
            }
            for run in runs
        ],
        "bins": int(args.bins),
        "metric": _metric_name(args.metric),
        "min_points": int(args.min_points),
        "top_k_candidates": int(args.top_k_candidates),
        "max_batches": args.max_batches,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    all_exported: list[ErrorBucketRecord] = []
    manifest_runs: list[dict[str, Any]] = []
    for run in runs:
        print(f"[Error Bucket] Selecting {run.display} samples...")
        selected, candidates, payload = _select_run(run, args)
        manifest_runs.append(
            {
                "key": run.key,
                "display": run.display,
                "config": str(run.config_path),
                "checkpoint": str(run.checkpoint),
                "eval_metrics_path": str((run.output_dir / "eval_metrics.json").relative_to(output_root)),
                "num_seen_samples": int(payload.get("metrics", {}).get("num_samples", 0)),
                "num_selected_samples": len(selected),
                "num_candidate_samples": len(candidates),
            }
        )
        if args.command == "all":
            print(f"[Error Bucket] Exporting {run.display} selected sample arrays...")
            exported = _export_run(run, selected, args)
            all_exported.extend(exported)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **plan,
        "runs": manifest_runs,
        "samples": [_record_to_row(record, base_dir=output_root) for record in all_exported],
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[Error Bucket] Wrote {manifest_path}")
    if args.command == "all":
        _write_csv(output_root / "selected_samples_exported.csv", all_exported, base_dir=output_root)
        for path in _render_manifest(manifest_path, args):
            print(f"[Error Bucket Render] Wrote {path}")


if __name__ == "__main__":
    main()
