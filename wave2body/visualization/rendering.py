# -*- coding: utf-8 -*-
"""Matplotlib renderers for qualitative Wave2Body samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .point_cloud import plot_mmwave_points_3d, point_bounds
from .pose import plot_skeleton_3d, pose_bounds, set_axes_equal_3d


PRED_COLOR = "#1e90ff"
WAVE2BODY_COLOR = "#1e90ff"
WICOMPASS_COLOR = "#7c3aed"
WICOMPASS_OVERLAY_COLOR = "#6B7280"
GT_COLOR = "#ff3c10"
POINT_COLOR = "#8A96A8"
AXIS_COLOR = "#282828"
RENDER_VARIANTS: tuple[str, ...] = ("pointcloud", "gt", "pointcloud_gt", "pointcloud_gt_pred")


def configure_publication_style() -> None:
    """Configure matplotlib for paper-quality static exports."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "axes.linewidth": 0.45,
            "axes.edgecolor": AXIS_COLOR,
            "text.color": "#282828",
            "axes.labelcolor": AXIS_COLOR,
            "xtick.color": AXIS_COLOR,
            "ytick.color": AXIS_COLOR,
        }
    )


configure_publication_style()


def _scalar_str(value: Any) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def load_sample_npz(path: str | Path) -> dict[str, Any]:
    """Load one exported qualitative sample."""

    with np.load(Path(path).expanduser(), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def sample_bounds(sample: dict[str, Any], *, margin: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Compute shared bounds from points, GT, and prediction."""

    point_low, point_high = point_bounds(sample["mmwave_points"], margin=margin, percentile=(5.0, 95.0))
    pose_low, pose_high = pose_bounds([sample["gt_pose"], sample["pred_pose"]], margin=0.35)
    return np.minimum(point_low, pose_low), np.maximum(point_high, pose_high)


def paired_sample_bounds(sample: dict[str, Any], *, margin: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Compute shared bounds for a paired Wave2Body/WiCompass sample."""

    point_low, point_high = point_bounds(sample["mmwave_points"], margin=margin, percentile=(5.0, 95.0))
    pose_low, pose_high = pose_bounds(
        [sample["gt_pose"], sample["wave2body_pred_pose"], sample["wicompass_pred_pose"]],
        margin=0.35,
    )
    return np.minimum(point_low, pose_low), np.maximum(point_high, pose_high)


def style_3d_axis(ax, *, bounds: tuple[np.ndarray, np.ndarray], show_axes: bool = False) -> None:
    """Apply compact paper-figure 3D axis styling."""

    set_axes_equal_3d(ax, bounds=bounds)
    ax.view_init(elev=18, azim=-65)
    if not show_axes:
        ax.set_axis_off()
        ax.grid(False)
        return
    ax.set_xlabel("x", labelpad=-10, fontsize=6)
    ax.set_ylabel("y", labelpad=-10, fontsize=6)
    ax.set_zlabel("z", labelpad=-10, fontsize=6)
    ax.tick_params(labelsize=5, pad=-4, length=2.0, width=0.35)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.0)
        axis.line.set_linewidth(0.35)
        axis.line.set_color(AXIS_COLOR)


def render_sample_axis(
    ax,
    sample: dict[str, Any],
    *,
    title: str | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    show_legend: bool = False,
    show_axes: bool = False,
    show_points: bool = True,
    show_gt: bool = True,
    show_pred: bool = True,
    point_size: float = 3.2,
    point_alpha: float = 0.30,
    gt_linewidth: float = 2.65,
    gt_joint_size: float = 24.0,
    gt_joint_edge_width: float = 0.55,
    pred_linewidth: float = 2.15,
    pred_joint_size: float = 17.0,
    pred_joint_edge_width: float = 0.45,
    skeleton_outline: bool = True,
) -> None:
    """Render one qualitative sample into an existing 3D axis."""

    if show_points:
        plot_mmwave_points_3d(
            ax,
            sample["mmwave_points"],
            color=POINT_COLOR,
            cmap=None,
            size=float(point_size),
            alpha=float(point_alpha),
            use_intensity=False,
            label="mmWave",
            zorder=1,
        )
    if show_gt:
        plot_skeleton_3d(
            ax,
            sample["gt_pose"],
            color=GT_COLOR,
            linewidth=float(gt_linewidth),
            linestyle="-",
            joint_size=float(gt_joint_size),
            joint_edge_width=float(gt_joint_edge_width),
            line_outline=bool(skeleton_outline),
            label="GT",
            zorder=2,
        )
    if show_pred:
        plot_skeleton_3d(
            ax,
            sample["pred_pose"],
            color=PRED_COLOR,
            linewidth=float(pred_linewidth),
            linestyle="-",
            joint_size=float(pred_joint_size),
            joint_edge_width=float(pred_joint_edge_width),
            line_outline=bool(skeleton_outline),
            alpha=0.98,
            label="Pred",
            zorder=3,
        )
    if bounds is None:
        bounds = sample_bounds(sample)
    style_3d_axis(ax, bounds=bounds, show_axes=show_axes)
    if title:
        ax.set_title(title, fontsize=8.5, pad=2, fontweight="regular")
    if show_legend:
        ax.legend(
            loc="upper left",
            fontsize=7,
            frameon=False,
            borderpad=0.1,
            handlelength=1.8,
            markerscale=1.2,
        )


def render_paired_comparison_axis(
    ax,
    sample: dict[str, Any],
    *,
    panel: str,
    title: str | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    show_axes: bool = False,
) -> None:
    """Render one panel from a paired Wave2Body/WiCompass comparison."""

    if panel not in {"gt", "wave2body", "wicompass"}:
        raise ValueError("panel must be one of {'gt', 'wave2body', 'wicompass'}")
    plot_mmwave_points_3d(
        ax,
        sample["mmwave_points"],
        color=POINT_COLOR,
        cmap=None,
        size=3.2,
        alpha=0.30,
        use_intensity=False,
        label="mmWave",
    )
    plot_skeleton_3d(
        ax,
        sample["gt_pose"],
        color=GT_COLOR,
        linewidth=2.65,
        linestyle="-",
        joint_size=24.0,
        joint_edge_width=0.55,
        label="GT",
        zorder=10,
    )
    if panel == "wave2body":
        plot_skeleton_3d(
            ax,
            sample["wave2body_pred_pose"],
            color=WAVE2BODY_COLOR,
            linewidth=2.15,
            linestyle="-",
            joint_size=17.0,
            joint_edge_width=0.45,
            alpha=0.98,
            label="Wave2Body",
            zorder=20,
        )
    elif panel == "wicompass":
        plot_skeleton_3d(
            ax,
            sample["wicompass_pred_pose"],
            color=WICOMPASS_COLOR,
            linewidth=2.15,
            linestyle="-",
            joint_size=17.0,
            joint_edge_width=0.45,
            alpha=0.98,
            label="WiCompass/PT",
            zorder=20,
        )
    if bounds is None:
        bounds = paired_sample_bounds(sample)
    style_3d_axis(ax, bounds=bounds, show_axes=show_axes)
    if title:
        ax.set_title(title, fontsize=8.0, pad=1.5, fontweight="regular")


def render_paired_pointcloud_axis(
    ax,
    sample: dict[str, Any],
    *,
    title: str | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    show_axes: bool = False,
) -> None:
    """Render only the mmWave point cloud for one paired comparison sample."""

    plot_mmwave_points_3d(
        ax,
        sample["mmwave_points"],
        color=POINT_COLOR,
        cmap=None,
        size=3.2,
        alpha=0.30,
        use_intensity=False,
        label="mmWave",
    )
    if bounds is None:
        bounds = paired_sample_bounds(sample)
    style_3d_axis(ax, bounds=bounds, show_axes=show_axes)
    if title:
        ax.set_title(title, fontsize=8.0, pad=1.5, fontweight="regular")


def render_paired_overlay_axis(
    ax,
    sample: dict[str, Any],
    *,
    title: str | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    show_axes: bool = False,
) -> None:
    """Render mmWave points with GT, Wave2Body, and grey WiCompass poses."""

    plot_mmwave_points_3d(
        ax,
        sample["mmwave_points"],
        color=POINT_COLOR,
        cmap=None,
        size=3.2,
        alpha=0.26,
        use_intensity=False,
        label="mmWave",
    )
    plot_skeleton_3d(
        ax,
        sample["wicompass_pred_pose"],
        color=WICOMPASS_OVERLAY_COLOR,
        linewidth=2.05,
        linestyle="-",
        joint_size=16.0,
        joint_edge_width=0.42,
        alpha=0.88,
        label="WiCompass/PT",
        zorder=10,
    )
    plot_skeleton_3d(
        ax,
        sample["gt_pose"],
        color=GT_COLOR,
        linewidth=2.65,
        linestyle="-",
        joint_size=24.0,
        joint_edge_width=0.55,
        label="GT",
        zorder=20,
    )
    plot_skeleton_3d(
        ax,
        sample["wave2body_pred_pose"],
        color=WAVE2BODY_COLOR,
        linewidth=2.15,
        linestyle="-",
        joint_size=17.0,
        joint_edge_width=0.45,
        alpha=0.98,
        label="Wave2Body",
        zorder=30,
    )
    if bounds is None:
        bounds = paired_sample_bounds(sample)
    style_3d_axis(ax, bounds=bounds, show_axes=show_axes)
    if title:
        ax.set_title(title, fontsize=8.0, pad=1.5, fontweight="regular")


def default_sample_title(record: dict[str, Any], sample: dict[str, Any] | None = None) -> str:
    """Return a compact title for paper grids."""

    group = str(record.get("action_group", _scalar_str(sample["action_group"]) if sample else ""))
    action_id = int(record.get("action_id", np.asarray(sample["indicator"])[1] if sample else -1))
    frame_id = int(record.get("frame_id", np.asarray(sample["indicator"])[2] if sample else -1))
    mpjpe = float(record.get("absolute_mpjpe_mm", np.asarray(sample["absolute_mpjpe_mm"]).item() if sample else 0.0))
    return f"{group} / A{action_id} / F{frame_id}\nAbs {mpjpe:.1f} mm"


def render_sample_file(
    npz_path: str | Path,
    output_path: str | Path,
    *,
    title: str | None = None,
    dpi: int = 600,
    pad_inches: float = 0.04,
    variant: str = "pointcloud_gt_pred",
) -> Path:
    """Render one exported sample to a PNG/PDF path."""

    if variant not in RENDER_VARIANTS:
        raise ValueError(f"Unknown render variant {variant!r}; expected one of {RENDER_VARIANTS}")
    sample = load_sample_npz(npz_path)
    fig = plt.figure(figsize=(3.15, 3.15), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    render_sample_axis(
        ax,
        sample,
        title=title,
        show_legend=False,
        show_points=variant != "gt",
        show_gt=variant in {"gt", "pointcloud_gt", "pointcloud_gt_pred"},
        show_pred=variant == "pointcloud_gt_pred",
    )
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(dpi), bbox_inches="tight", pad_inches=float(pad_inches), facecolor="white")
    plt.close(fig)
    return out


def render_sample_variant_files(
    npz_path: str | Path,
    output_dir: str | Path,
    *,
    action_id: int,
    frame_id: int,
    sample_index: int,
    variants: tuple[str, ...] = RENDER_VARIANTS,
    dpi: int = 600,
    pad_inches: float = 0.04,
) -> list[Path]:
    """Render one sample into multiple variant PDF files in one directory."""

    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for variant in variants:
        if variant not in RENDER_VARIANTS:
            raise ValueError(f"Unknown render variant {variant!r}; expected one of {RENDER_VARIANTS}")
        output = out_dir / f"{variant}_action_{int(action_id):02d}_sample_{int(sample_index):02d}_frame_{int(frame_id):06d}.pdf"
        out_paths.append(
            render_sample_file(
                npz_path,
                output,
                dpi=dpi,
                pad_inches=pad_inches,
                variant=variant,
            )
        )
    return out_paths


def render_paired_comparison_file(
    npz_path: str | Path,
    output_path: str | Path,
    *,
    dpi: int = 600,
    pad_inches: float = 0.04,
) -> Path:
    """Render one paired sample as a three-panel comparison figure."""

    sample = load_sample_npz(npz_path)
    bounds = paired_sample_bounds(sample)
    fig = plt.figure(figsize=(9.15, 3.0), facecolor="white")
    for idx, (panel, title) in enumerate(
        (
            ("gt", "mmWave + GT"),
            ("wave2body", "Wave2Body"),
            ("wicompass", "WiCompass/PT"),
        ),
        start=1,
    ):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        render_paired_comparison_axis(ax, sample, panel=panel, title=title, bounds=bounds)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.93, wspace=0.0, hspace=0.0)
    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(dpi), bbox_inches="tight", pad_inches=float(pad_inches), facecolor="white")
    plt.close(fig)
    return out


def render_paired_comparison_panel_file(
    npz_path: str | Path,
    output_path: str | Path,
    *,
    panel: str,
    title: str | None = None,
    dpi: int = 600,
    pad_inches: float = 0.0,
) -> Path:
    """Render one paired comparison panel as an individual tightly cropped file."""

    sample = load_sample_npz(npz_path)
    bounds = paired_sample_bounds(sample)
    fig = plt.figure(figsize=(2.85, 2.85), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    render_paired_comparison_axis(ax, sample, panel=panel, title=title, bounds=bounds)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)
    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(dpi), bbox_inches="tight", pad_inches=float(pad_inches), facecolor="white")
    plt.close(fig)
    return out


def render_paired_case_variant_file(
    npz_path: str | Path,
    output_path: str | Path,
    *,
    variant: str,
    title: str | None = None,
    dpi: int = 600,
    pad_inches: float = 0.0,
) -> Path:
    """Render one paired comparison sample as either point cloud only or overlay."""

    if variant not in {"pointcloud", "overlay"}:
        raise ValueError("variant must be one of {'pointcloud', 'overlay'}")
    sample = load_sample_npz(npz_path)
    bounds = paired_sample_bounds(sample)
    fig = plt.figure(figsize=(2.85, 2.85), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    if variant == "pointcloud":
        render_paired_pointcloud_axis(ax, sample, title=title, bounds=bounds)
    else:
        render_paired_overlay_axis(ax, sample, title=title, bounds=bounds)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)
    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(dpi), bbox_inches="tight", pad_inches=float(pad_inches), facecolor="white")
    plt.close(fig)
    return out


def render_paired_case_variant_files(
    npz_path: str | Path,
    output_dir: str | Path,
    *,
    record: dict[str, Any],
    sample_index: int,
    dpi: int = 600,
    pad_inches: float = 0.0,
    file_format: str = "pdf",
    include_titles: bool = False,
) -> list[Path]:
    """Render point-cloud-only and all-pose overlay PDFs for one comparison case."""

    variants = (
        ("pointcloud", "pointcloud", "mmWave"),
        ("overlay", "pointcloud_wicompass_wave2body_gt", "mmWave + WiCompass/PT + Wave2Body + GT"),
    )
    suffix = f".{file_format.lstrip('.')}"
    stem = paired_panel_output_stem(record, sample_index=sample_index)
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for variant, variant_name, label in variants:
        out_paths.append(
            render_paired_case_variant_file(
                npz_path,
                out_dir / f"{stem}_{variant_name}{suffix}",
                variant=variant,
                title=label if include_titles else None,
                dpi=dpi,
                pad_inches=pad_inches,
            )
        )
    return out_paths


def paired_panel_output_stem(record: dict[str, Any], *, sample_index: int) -> str:
    """Return a collision-resistant file stem for one paired comparison case."""

    split = str(record.get("split_id", "split"))
    group = str(record.get("action_group", "group"))
    subject_id = int(record.get("subject_id", 0))
    action_id = int(record.get("action_id", 0))
    frame_id = int(record.get("frame_id", 0))
    clean_group = group.replace("/", "_")
    return f"{split}_{clean_group}_rank{int(sample_index):02d}_s{subject_id:02d}_a{action_id:02d}_f{frame_id:06d}"


def render_paired_comparison_panel_files(
    npz_path: str | Path,
    output_dir: str | Path,
    *,
    record: dict[str, Any],
    sample_index: int,
    dpi: int = 600,
    pad_inches: float = 0.0,
    file_format: str = "pdf",
    include_titles: bool = False,
) -> list[Path]:
    """Render one paired sample into separate mmWave/GT, Wave2Body, and WiCompass files."""

    panels = (
        ("gt", "mmwave_gt", "mmWave + GT"),
        ("wave2body", "wave2body", "Wave2Body"),
        ("wicompass", "wicompass_pt", "WiCompass/PT"),
    )
    suffix = f".{file_format.lstrip('.')}"
    stem = paired_panel_output_stem(record, sample_index=sample_index)
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for panel, panel_name, label in panels:
        out_paths.append(
            render_paired_comparison_panel_file(
                npz_path,
                out_dir / f"{stem}_{panel_name}{suffix}",
                panel=panel,
                title=label if include_titles else None,
                dpi=dpi,
                pad_inches=pad_inches,
            )
        )
    return out_paths


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _record_npz_path(manifest_path: Path, record: dict[str, Any]) -> Path:
    raw = Path(str(record["npz_path"]))
    if raw.is_absolute():
        return raw
    return manifest_path.parent / raw


def render_manifest_grid(
    manifest_path: str | Path,
    output_prefix: str | Path,
    *,
    columns: int = 3,
    dpi: int = 600,
    pad_inches: float = 0.05,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> list[Path]:
    """Render the selected samples from a manifest as one grid figure."""

    manifest_file = Path(manifest_path).expanduser()
    manifest = load_manifest(manifest_file)
    records = list(manifest.get("samples", []))
    if not records:
        raise ValueError(f"Manifest has no samples: {manifest_file}")
    samples = [load_sample_npz(_record_npz_path(manifest_file, record)) for record in records]
    bounds_list = [sample_bounds(sample) for sample in samples]
    low = np.minimum.reduce([item[0] for item in bounds_list])
    high = np.maximum.reduce([item[1] for item in bounds_list])
    shared_bounds = (low, high)

    cols = min(max(1, int(columns)), len(records))
    rows = int(np.ceil(len(records) / float(cols)))
    fig = plt.figure(figsize=(3.05 * cols, 2.95 * rows), facecolor="white")
    for idx, (record, sample) in enumerate(zip(records, samples), start=1):
        ax = fig.add_subplot(rows, cols, idx, projection="3d")
        render_sample_axis(
            ax,
            sample,
            title=None,
            bounds=shared_bounds,
            show_legend=False,
            show_axes=False,
        )
    for idx in range(len(records) + 1, rows * cols + 1):
        ax = fig.add_subplot(rows, cols, idx, projection="3d")
        ax.set_axis_off()
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.02)

    prefix = Path(output_prefix).expanduser()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for fmt in formats:
        out = prefix.with_suffix(f".{fmt.lstrip('.')}")
        fig.savefig(out, dpi=int(dpi), bbox_inches="tight", pad_inches=float(pad_inches), facecolor="white")
        out_paths.append(out)
    plt.close(fig)
    return out_paths


def render_paired_comparison_grid(
    manifest_path: str | Path,
    output_prefix: str | Path,
    *,
    dpi: int = 600,
    pad_inches: float = 0.05,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> list[Path]:
    """Render paired Wave2Body/WiCompass samples as a row-wise three-column grid."""

    manifest_file = Path(manifest_path).expanduser()
    manifest = load_manifest(manifest_file)
    records = list(manifest.get("samples", []))
    if not records:
        raise ValueError(f"Manifest has no samples: {manifest_file}")
    samples = [load_sample_npz(_record_npz_path(manifest_file, record)) for record in records]
    rows = len(samples)
    fig = plt.figure(figsize=(9.15, 2.85 * rows), facecolor="white")
    panels = (
        ("gt", "mmWave + GT"),
        ("wave2body", "Wave2Body"),
        ("wicompass", "WiCompass/PT"),
    )
    for row_idx, (record, sample) in enumerate(zip(records, samples), start=0):
        bounds = paired_sample_bounds(sample)
        for col_idx, (panel, label) in enumerate(panels, start=0):
            ax = fig.add_subplot(rows, 3, row_idx * 3 + col_idx + 1, projection="3d")
            title = label if row_idx == 0 else None
            render_paired_comparison_axis(ax, sample, panel=panel, title=title, bounds=bounds)
            if col_idx == 0:
                split = record.get("split_id", "")
                action = int(record.get("action_id", np.asarray(sample["indicator"])[1]))
                delta = float(record.get("delta_mpjpe_mm", np.asarray(sample["delta_mpjpe_mm"]).item()))
                ax.text2D(
                    0.02,
                    0.02,
                    f"{split} A{action:02d}  +{delta:.1f} mm",
                    transform=ax.transAxes,
                    fontsize=6.8,
                    color=AXIS_COLOR,
                )
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.98, wspace=0.0, hspace=0.0)

    prefix = Path(output_prefix).expanduser()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for fmt in formats:
        out = prefix.with_suffix(f".{fmt.lstrip('.')}")
        fig.savefig(out, dpi=int(dpi), bbox_inches="tight", pad_inches=float(pad_inches), facecolor="white")
        out_paths.append(out)
    plt.close(fig)
    return out_paths
