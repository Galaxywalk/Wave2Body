# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np

from wave2body.visualization.rendering import (
    render_manifest_grid,
    render_paired_case_variant_files,
    render_paired_comparison_file,
    render_paired_comparison_grid,
    render_paired_comparison_panel_files,
    render_sample_file,
    render_sample_variant_files,
)
from wave2body.visualization.sample_selection import PairedComparisonCandidate, SampleCandidate


def _candidate() -> SampleCandidate:
    points = np.zeros((32, 4), dtype=np.float32)
    points[:, 0] = np.linspace(-0.4, 0.4, 32)
    points[:, 1] = np.sin(np.linspace(0.0, 1.0, 32)) * 0.2
    points[:, 2] = np.linspace(0.2, 1.1, 32)
    points[:, 3] = np.linspace(0.0, 1.0, 32)
    pose = np.zeros((22, 3), dtype=np.float32)
    pose[:, 0] = np.linspace(-0.2, 0.2, 22)
    pose[:, 2] = np.linspace(0.0, 1.0, 22)
    pred = pose.copy()
    pred[:, 0] += 0.02
    return SampleCandidate(
        action_group="1_30",
        action_id=1,
        subject_id=1,
        frame_id=10,
        global_index=0,
        absolute_mpjpe_mm=12.3,
        root_error_mm=4.5,
        root_aligned_mpjpe_mm=11.0,
        mmwave_points=points,
        gt_pose=pose,
        pred_pose=pred,
        indicator=np.array([1, 1, 10], dtype=np.int64),
        temporal_indicators=np.array([[1, 1, 7], [1, 1, 8], [1, 1, 9], [1, 1, 10]], dtype=np.int64),
        split_id="s1",
        eval_split="test",
        config="cfg.yaml",
        checkpoint="best.pth",
    )


def _paired_candidate() -> PairedComparisonCandidate:
    points = np.zeros((32, 4), dtype=np.float32)
    points[:, 0] = np.linspace(-0.4, 0.4, 32)
    points[:, 1] = np.sin(np.linspace(0.0, 1.0, 32)) * 0.2
    points[:, 2] = np.linspace(0.2, 1.1, 32)
    points[:, 3] = np.linspace(0.0, 1.0, 32)
    pose = np.zeros((22, 3), dtype=np.float32)
    pose[:, 0] = np.linspace(-0.2, 0.2, 22)
    pose[:, 2] = np.linspace(0.0, 1.0, 22)
    wave = pose.copy()
    wave[:, 0] += 0.02
    pt = pose.copy()
    pt[:, 0] += 0.08
    return PairedComparisonCandidate(
        action_group="36_50",
        action_id=38,
        subject_id=2,
        frame_id=30,
        global_index=5,
        wave2body_mpjpe_mm=80.0,
        wicompass_mpjpe_mm=120.0,
        wave2body_root_error_mm=50.0,
        wicompass_root_error_mm=90.0,
        wave2body_root_aligned_mpjpe_mm=70.0,
        wicompass_root_aligned_mpjpe_mm=100.0,
        mmwave_points=points,
        gt_pose=pose,
        wave2body_pred_pose=wave,
        wicompass_pred_pose=pt,
        indicator=np.array([2, 38, 30], dtype=np.int64),
        temporal_indicators=np.array([[2, 38, 27], [2, 38, 28], [2, 38, 29], [2, 38, 30]], dtype=np.int64),
        split_id="s2",
        eval_split="test",
        wave2body_config="wave.yaml",
        wave2body_checkpoint="wave.pth",
        wicompass_config="pt.yaml",
        wicompass_checkpoint="pt.pth",
    )


def test_render_sample_and_grid_write_nonempty_files(tmp_path: Path):
    candidate = _candidate()
    sample_path = tmp_path / "samples" / "sample.npz"
    candidate.save_npz(sample_path)
    manifest = {
        "samples": [
            candidate.record(npz_path=str(sample_path.relative_to(tmp_path))),
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    single = render_sample_file(sample_path, tmp_path / "single.png", dpi=80)
    grid_paths = render_manifest_grid(manifest_path, tmp_path / "grid", dpi=80, formats=("png",))

    assert single.exists() and single.stat().st_size > 0
    assert len(grid_paths) == 1
    assert grid_paths[0].exists() and grid_paths[0].stat().st_size > 0


def test_render_sample_variants_write_prefixed_pdfs(tmp_path: Path):
    candidate = _candidate()
    sample_path = tmp_path / "sample.npz"
    candidate.save_npz(sample_path)

    paths = render_sample_variant_files(
        sample_path,
        tmp_path / "pdfs",
        action_id=1,
        frame_id=10,
        sample_index=1,
        variants=("pointcloud", "pointcloud_gt", "pointcloud_gt_pred"),
        dpi=80,
    )

    assert [path.name for path in paths] == [
        "pointcloud_action_01_sample_01_frame_000010.pdf",
        "pointcloud_gt_action_01_sample_01_frame_000010.pdf",
        "pointcloud_gt_pred_action_01_sample_01_frame_000010.pdf",
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)


def test_render_gt_only_variant_write_pdf(tmp_path: Path):
    candidate = _candidate()
    sample_path = tmp_path / "sample.npz"
    candidate.save_npz(sample_path)

    paths = render_sample_variant_files(
        sample_path,
        tmp_path / "pdfs",
        action_id=13,
        frame_id=561,
        sample_index=13,
        variants=("gt",),
        dpi=80,
    )

    assert [path.name for path in paths] == ["gt_action_13_sample_13_frame_000561.pdf"]
    assert paths[0].exists() and paths[0].stat().st_size > 0


def test_render_paired_comparison_write_nonempty_files(tmp_path: Path):
    candidate = _paired_candidate()
    sample_path = tmp_path / "samples" / "paired.npz"
    candidate.save_npz(sample_path)
    manifest = {
        "samples": [
            candidate.record(npz_path=str(sample_path.relative_to(tmp_path))),
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    single = render_paired_comparison_file(sample_path, tmp_path / "single.png", dpi=80)
    grid_paths = render_paired_comparison_grid(manifest_path, tmp_path / "comparison_grid", dpi=80, formats=("png",))

    assert single.exists() and single.stat().st_size > 0
    assert len(grid_paths) == 1
    assert grid_paths[0].exists() and grid_paths[0].stat().st_size > 0


def test_render_paired_comparison_panel_files_write_three_pdfs(tmp_path: Path):
    candidate = _paired_candidate()
    sample_path = tmp_path / "samples" / "paired.npz"
    candidate.save_npz(sample_path)
    record = candidate.record(npz_path=str(sample_path.relative_to(tmp_path)))

    paths = render_paired_comparison_panel_files(
        sample_path,
        tmp_path / "panels",
        record=record,
        sample_index=1,
        dpi=80,
        file_format="pdf",
    )

    assert [path.name for path in paths] == [
        "s2_36_50_rank01_s02_a38_f000030_mmwave_gt.pdf",
        "s2_36_50_rank01_s02_a38_f000030_wave2body.pdf",
        "s2_36_50_rank01_s02_a38_f000030_wicompass_pt.pdf",
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)


def test_render_paired_case_variant_files_write_pointcloud_and_overlay_pdfs(tmp_path: Path):
    candidate = _paired_candidate()
    sample_path = tmp_path / "samples" / "paired.npz"
    candidate.save_npz(sample_path)
    record = candidate.record(npz_path=str(sample_path.relative_to(tmp_path)))

    paths = render_paired_case_variant_files(
        sample_path,
        tmp_path / "variants",
        record=record,
        sample_index=1,
        dpi=80,
        file_format="pdf",
    )

    assert [path.name for path in paths] == [
        "s2_36_50_rank01_s02_a38_f000030_pointcloud.pdf",
        "s2_36_50_rank01_s02_a38_f000030_pointcloud_wicompass_wave2body_gt.pdf",
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
