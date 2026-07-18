# -*- coding: utf-8 -*-

import sys
import types

import numpy as np
import torch

from wave2body.dataset.radar_pose.base import BaseRadarPoseDataset
from wave2body.dataset.radar_pose.mmbody import MMBodyDataset
from wave2body.dataset.radar_pose.point_sampling import (
    farthest_point_sample_indices,
    pad_or_sample_points,
    sample_point_indices,
)
from wave2body.dataset.radar_pose.root_alignment import RootAlignmentConfig, apply_input_root_alignment


def test_builtin_fps_is_deterministic():
    xyz = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ]
    )

    idx = farthest_point_sample_indices(xyz, 3)

    torch.testing.assert_close(idx, torch.tensor([0, 4, 2]))


def test_pad_or_sample_points_preserves_dtype_and_padding():
    points = torch.tensor([[1.0, 2.0, 3.0, 0.5]], dtype=torch.float64)

    padded = pad_or_sample_points(points, n_points=3, input_channels=4)

    assert padded.dtype == torch.float64
    assert padded.shape == (3, 4)
    torch.testing.assert_close(padded[0], points[0])
    torch.testing.assert_close(padded[1:], torch.zeros((2, 4), dtype=torch.float64))


def test_base_dataset_uses_builtin_fps_without_pointnet_extension():
    ds = object.__new__(BaseRadarPoseDataset)
    ds.n_points = 3
    ds.input_channels = 4
    ds.point_sampling = "fps"
    points = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.1],
            [2.0, 0.0, 0.0, 0.2],
            [3.0, 0.0, 0.0, 0.3],
            [4.0, 0.0, 0.0, 0.4],
            [5.0, 0.0, 0.0, 0.5],
        ]
    )

    sampled = ds._process_points(points)

    torch.testing.assert_close(sampled[:, 0], torch.tensor([1.0, 5.0, 3.0]))


def test_torch_cluster_fps_strategy_is_optional(monkeypatch):
    fake = types.ModuleType("torch_cluster")

    def fake_fps(x, batch=None, ratio=0.5, random_start=True):
        assert batch is None
        assert 0.0 < ratio < 1.0
        assert random_start is False
        return torch.tensor([0, 4, 2, 1], device=x.device)

    fake.fps = fake_fps
    monkeypatch.setitem(sys.modules, "torch_cluster", fake)

    points = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.1],
            [2.0, 0.0, 0.0, 0.2],
            [3.0, 0.0, 0.0, 0.3],
            [4.0, 0.0, 0.0, 0.4],
            [5.0, 0.0, 0.0, 0.5],
        ]
    )

    idx = sample_point_indices(points, 3, strategy="torch_cluster_fps")

    torch.testing.assert_close(idx, torch.tensor([0, 4, 2]))


def test_feature_ranked_sampling_strategies():
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.1, 0.0],
            [1.0, 0.0, 0.0, -5.0, 0.2],
            [2.0, 0.0, 0.0, 2.0, -3.0],
            [3.0, 0.0, 0.0, 3.0, 0.4],
            [4.0, 0.0, 0.0, -4.0, 2.0],
        ]
    )

    intensity_idx = sample_point_indices(points, 3, strategy="top_abs_intensity")
    velocity_idx = sample_point_indices(points, 2, strategy="top_velocity_abs")

    torch.testing.assert_close(intensity_idx, torch.tensor([1, 4, 3]))
    torch.testing.assert_close(velocity_idx, torch.tensor([2, 4]))


def test_near_origin_sampling_uses_xyz_distance():
    points = torch.tensor(
        [
            [3.0, 4.0, 0.0, 0.1, 0.0],
            [0.1, 0.1, 0.1, 0.2, 0.0],
            [1.0, 1.0, 1.0, 0.3, 0.0],
            [0.0, 0.0, 0.0, 0.4, 0.0],
        ]
    )

    idx = sample_point_indices(points, 2, strategy="near_origin")

    torch.testing.assert_close(idx, torch.tensor([3, 1]))


def test_hybrid_abs_intensity_fps_filters_then_covers_space():
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 10.0, 0.0],
            [1.0, 0.0, 0.0, 9.0, 0.0],
            [2.0, 0.0, 0.0, 8.0, 0.0],
            [3.0, 0.0, 0.0, 7.0, 0.0],
            [4.0, 0.0, 0.0, 6.0, 0.0],
            [5.0, 0.0, 0.0, 5.0, 0.0],
            [6.0, 0.0, 0.0, 4.0, 0.0],
            [7.0, 0.0, 0.0, 3.0, 0.0],
            [8.0, 0.0, 0.0, 0.0, 0.0],
            [9.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    idx = sample_point_indices(points, 2, strategy="hybrid_abs_intensity_fps")

    torch.testing.assert_close(idx, torch.tensor([0, 7]))


def test_mmbody_cpu_point_processing_honors_configured_sampling():
    ds = object.__new__(MMBodyDataset)
    ds.n_points = 2
    ds.input_channels = 5
    ds.point_sampling = "top_abs_intensity"
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.1, 0.0],
            [1.0, 0.0, 0.0, 0.9, 0.0],
            [2.0, 0.0, 0.0, 0.5, 0.0],
        ]
    )

    sampled = ds._process_points_cpu(points)

    torch.testing.assert_close(sampled[:, 3], torch.tensor([0.9, 0.5]))


def test_mmbody_defaults_to_historical_random_sampling(monkeypatch):
    captured = {}

    def fake_base_init(self, config, split="train", vqvae_model=None, device=None):
        captured["config"] = config
        self.config = config

    monkeypatch.setattr(BaseRadarPoseDataset, "__init__", fake_base_init)

    MMBodyDataset({"data_format": {"n_points": 200}}, split="train")
    assert captured["config"]["data_format"]["point_sampling"] == "random"
    assert captured["config"]["data_format"]["input_root_crop"] is True

    MMBodyDataset({"data_format": {"point_sampling": "top_abs_intensity"}}, split="val")
    assert captured["config"]["data_format"]["point_sampling"] == "top_abs_intensity"


def test_mmbody_input_and_target_root_normalization_are_independent(monkeypatch):
    import pandas as pd

    ds = object.__new__(MMBodyDataset)
    ds.data_list = pd.DataFrame(
        [{"Sequence": 0, "SequenceName": "sequence_0", "Scenario": "lab1", "Frame": 0, "Radar": "", "Mesh": ""}]
    )
    ds.input_channels = 5
    ds.n_points = 2
    ds.point_sampling = "fps"
    ds.input_root_normalized = False
    ds.input_root_crop = False
    ds.target_root_normalized = True
    ds._cached_mmwave_points = None
    ds._cached_target_poses = None
    ds._cached_target_poses_absolute = None

    def fake_load_sample(_idx):
        radar = np.asarray([[10.0, 20.0, 30.0, 1.0, 0.0]], dtype=np.float32)
        joints = np.asarray([[10.0, 20.0, 30.0], [11.0, 20.0, 30.0]], dtype=np.float32)
        if ds.input_root_normalized:
            radar[:, :3] -= joints[0]
        return [radar], joints

    monkeypatch.setattr(ds, "_load_sample", fake_load_sample)

    item = ds[0]
    torch.testing.assert_close(item["mmwave_points"][0, :3], torch.tensor([10.0, 20.0, 30.0]))
    torch.testing.assert_close(item["target_poses"][0], torch.zeros(3))
    torch.testing.assert_close(item["target_poses"][1], torch.tensor([1.0, 0.0, 0.0]))

    ds.input_root_normalized = True
    ds.target_root_normalized = False
    item = ds[0]
    torch.testing.assert_close(item["mmwave_points"][0, :3], torch.zeros(3))
    torch.testing.assert_close(item["target_poses"][0], torch.tensor([10.0, 20.0, 30.0]))


def test_mmbody_root_crop_preserves_absolute_coordinates():
    radar = np.asarray(
        [
            [10.2, 20.1, 30.0, 1.0, 0.0],
            [12.5, 20.0, 30.0, 0.5, 0.0],
            [10.0, 21.5, 30.0, 0.2, 0.0],
        ],
        dtype=np.float32,
    )
    root = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
    cfg = RootAlignmentConfig(
        normalized=True,
        input_root_normalized=False,
        target_root_normalized=True,
        input_root_crop=True,
        input_root_crop_x_range=(-1.0, 1.0),
        input_root_crop_y_range=(-1.0, 1.0),
        input_root_crop_z_range=(-1.0, 1.0),
    )

    cropped = apply_input_root_alignment(radar, root=root, cfg=cfg)

    assert cropped.shape == (1, 5)
    np.testing.assert_allclose(cropped[0, :3], np.asarray([10.2, 20.1, 30.0], dtype=np.float32))


def test_mmbody_cached_item_preserves_temporal_metadata():
    import pandas as pd

    ds = object.__new__(MMBodyDataset)
    ds.data_list = pd.DataFrame(
        [
            {
                "Sequence": 3,
                "SequenceName": "sequence_3",
                "Scenario": "smoke",
                "Frame": 17,
                "Radar": "",
                "Mesh": "",
            }
        ]
    )
    ds._cached_mmwave_points = torch.zeros((1, 2, 5))
    ds._cached_target_poses = torch.zeros((1, 22, 3))
    ds._cached_target_poses_absolute = None
    ds.precompute_tokens = False
    ds.vqvae_model = None

    item = ds[0]

    assert item["mmbody_scenario"] == "smoke"
    assert item["mmbody_sequence"] == "sequence_3"
    assert item["mmbody_frame"] == 17
    torch.testing.assert_close(item["mmwave_points"], torch.zeros((2, 5)))
    torch.testing.assert_close(item["target_poses"], torch.zeros((22, 3)))
