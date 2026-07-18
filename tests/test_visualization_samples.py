# -*- coding: utf-8 -*-

import numpy as np
import pytest
import torch

from wave2body.visualization.point_cloud import valid_point_mask
from wave2body.visualization.pose import SMPL22_BONE_EDGES, as_pose_array
from wave2body.visualization.sample_selection import (
    LowErrorPerActionSelector,
    LowErrorSampleSelector,
    PairedAdvantageSampleSelector,
)


def _pose(offset: float) -> torch.Tensor:
    base = torch.zeros(22, 3)
    base[:, 0] = torch.linspace(0.0, 1.0, 22) + float(offset)
    return base


def test_low_error_selector_keeps_best_per_semantic_group_with_action_diversity():
    batch = {
        "indicator": torch.tensor(
            [
                [1, 1, 10],
                [1, 1, 20],
                [1, 2, 30],
                [1, 36, 40],
            ],
            dtype=torch.long,
        ),
        "temporal_indicators": torch.zeros(4, 4, 3, dtype=torch.long),
    }
    pred = torch.stack([_pose(i) for i in range(4)], dim=0)
    gt = pred.clone()
    points = torch.ones(4, 8, 4)
    selector = LowErrorSampleSelector(
        top_k=2,
        action_groups=("1_30", "36_50"),
        max_per_action=1,
        min_frame_gap=0,
        pool_size=10,
    )

    selector.update_from_batch(
        batch=batch,
        pred_pose=pred,
        gt_pose=gt,
        mmwave_points=points,
        absolute_mpjpe=torch.tensor([0.010, 0.005, 0.020, 0.030]),
        root_error=torch.tensor([0.001, 0.002, 0.003, 0.004]),
        root_aligned_mpjpe=torch.tensor([0.011, 0.012, 0.013, 0.014]),
    )

    selected = selector.selected()
    assert [item.action_id for item in selected["1_30"]] == [1, 2]
    assert [round(item.absolute_mpjpe_mm, 1) for item in selected["1_30"]] == [5.0, 20.0]
    assert [item.action_id for item in selected["36_50"]] == [36]
    assert selector.seen_count == 4


def test_point_cloud_mask_ignores_padded_zero_rows():
    points = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.5],
            [np.nan, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(valid_point_mask(points), np.array([False, True, False]))


def test_smpl22_pose_contract_and_edges():
    pose = np.zeros((22, 3), dtype=np.float32)
    assert as_pose_array(pose).shape == (22, 3)
    assert (0, 1) in SMPL22_BONE_EDGES
    assert (19, 21) in SMPL22_BONE_EDGES


def test_low_error_per_action_selector_keeps_each_action():
    batch = {
        "indicator": torch.tensor([[1, 1, 10], [1, 2, 20], [1, 2, 30]], dtype=torch.long),
        "temporal_indicators": torch.zeros(3, 4, 3, dtype=torch.long),
    }
    pose = torch.stack([_pose(i) for i in range(3)], dim=0)
    selector = LowErrorPerActionSelector(top_k=1, action_ids=(1, 2), min_frame_gap=0)

    selector.update_from_batch(
        batch=batch,
        pred_pose=pose,
        gt_pose=pose,
        mmwave_points=torch.ones(3, 8, 4),
        absolute_mpjpe=torch.tensor([0.030, 0.020, 0.010]),
        root_error=torch.tensor([0.003, 0.002, 0.001]),
        root_aligned_mpjpe=torch.tensor([0.013, 0.012, 0.011]),
    )

    selected = selector.selected()
    assert selected[1][0].frame_id == 10
    assert selected[2][0].frame_id == 30
    assert selected[2][0].absolute_mpjpe_mm == pytest.approx(10.0)


def test_paired_advantage_selector_filters_and_ranks_difficult_wave2body_wins():
    batch = {
        "indicator": torch.tensor(
            [
                [1, 31, 10],
                [2, 32, 20],
                [3, 36, 30],
                [4, 37, 40],
                [5, 38, 50],
            ],
            dtype=torch.long,
        ),
        "temporal_indicators": torch.zeros(5, 4, 3, dtype=torch.long),
    }
    gt = torch.stack([_pose(i) for i in range(5)], dim=0)
    wave = gt.clone()
    pt = gt.clone()
    selector = PairedAdvantageSampleSelector(
        top_k=2,
        action_groups=("31_35", "36_50"),
        min_wicompass_mpjpe_mm=100.0,
        min_delta_mpjpe_mm=5.0,
        min_frame_gap=0,
        split_id="s2",
        eval_split="test",
        wave2body_config="wave.yaml",
        wave2body_checkpoint="wave.pth",
        wicompass_config="pt.yaml",
        wicompass_checkpoint="pt.pth",
    )

    selector.update_from_batch(
        batch=batch,
        wave2body_pred_pose=wave,
        wicompass_pred_pose=pt,
        gt_pose=gt,
        mmwave_points=torch.ones(5, 8, 4),
        wave2body_mpjpe=torch.tensor([0.090, 0.080, 0.110, 0.100, 0.150]),
        wicompass_mpjpe=torch.tensor([0.130, 0.090, 0.116, 0.140, 0.170]),
        wave2body_root_error=torch.tensor([0.050, 0.040, 0.060, 0.070, 0.080]),
        wicompass_root_error=torch.tensor([0.090, 0.050, 0.080, 0.100, 0.110]),
        wave2body_root_aligned_mpjpe=torch.tensor([0.070, 0.060, 0.090, 0.080, 0.120]),
        wicompass_root_aligned_mpjpe=torch.tensor([0.100, 0.070, 0.096, 0.110, 0.140]),
    )

    selected = selector.selected()
    assert [item.action_id for item in selected["31_35"]] == [31]
    assert [item.action_id for item in selected["36_50"]] == [37, 38]
    assert selected["36_50"][0].delta_mpjpe_mm == pytest.approx(40.0)
    record = selected["36_50"][0].record(npz_path="sample.npz")
    assert record["split_id"] == "s2"
    assert record["delta_mpjpe_mm"] == pytest.approx(40.0)
    assert not any("joint" in key for key in record)
    assert selector.seen_count == 5
    assert selector.eligible_count == 4


def test_error_bucket_selector_supports_mmbody_metadata_without_indicator():
    from wave2body.visualization.sample_selection import ErrorBucketMetricCollector

    pose = torch.stack([_pose(i) for i in range(6)], dim=0)
    batch = {
        "mmbody_scenario": ["lab1"] * 6,
        "mmbody_sequence": ["sequence_0"] * 6,
        "mmbody_frame": torch.arange(6),
    }
    selector = ErrorBucketMetricCollector(
        dataset="mmbody",
        dataset_display="mmBody",
        bins=3,
        metric="absolute_mpjpe_mm",
        min_points=1,
        top_k_candidates=2,
    )

    selector.update_from_batch(
        batch=batch,
        pred_pose=pose,
        gt_pose=pose,
        mmwave_points=torch.ones(6, 8, 4),
        absolute_mpjpe=torch.tensor([0.060, 0.010, 0.030, 0.020, 0.050, 0.040]),
        root_error=torch.tensor([0.006, 0.001, 0.003, 0.002, 0.005, 0.004]),
        root_aligned_mpjpe=torch.tensor([0.016, 0.011, 0.013, 0.012, 0.015, 0.014]),
    )

    selected, candidates = selector.selected()
    assert len(selected) == 3
    assert len(candidates) == 6
    assert [round(item.absolute_mpjpe_mm, 1) for item in selected] == [10.0, 30.0, 50.0]
    assert selected[0].metadata["mmbody_frame"] == 1
    assert selected[0].point_count == 8


def test_error_bucket_selector_falls_back_when_bucket_has_too_few_points():
    from wave2body.visualization.sample_selection import ErrorBucketMetricCollector

    pose = torch.stack([_pose(i) for i in range(4)], dim=0)
    selector = ErrorBucketMetricCollector(
        dataset="m4human_s2",
        dataset_display="M4Human S2",
        bins=2,
        metric="absolute_mpjpe_mm",
        min_points=99,
        top_k_candidates=1,
    )
    selector.update_from_batch(
        batch={"indicator": torch.tensor([[1, 1, 10], [1, 2, 20], [1, 3, 30], [1, 4, 40]])},
        pred_pose=pose,
        gt_pose=pose,
        mmwave_points=torch.ones(4, 8, 4),
        absolute_mpjpe=torch.tensor([0.010, 0.020, 0.030, 0.040]),
        root_error=torch.tensor([0.001, 0.002, 0.003, 0.004]),
        root_aligned_mpjpe=torch.tensor([0.011, 0.012, 0.013, 0.014]),
    )

    selected, _ = selector.selected()
    assert len(selected) == 2
    assert all(item.fallback_low_points for item in selected)
    assert selected[0].metadata["action_id"] == 1
