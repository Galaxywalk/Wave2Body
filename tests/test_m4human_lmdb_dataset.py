# -*- coding: utf-8 -*-

import gzip
import pickle
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("lmdb")
pytest.importorskip("smplx")

from wave2body.dataset.radar_pose.m4human import (  # noqa: E402
    M4HumanLMDBPoseDataset,
    _decode_np,
    _parse_temporal_offsets,
    _unpack_ndarray,
)
from wave2body.dataset.radar_pose.root_alignment import (  # noqa: E402
    apply_input_root_alignment,
    resolve_root_alignment_config,
)


def _pack_float32_ndarray(arr: np.ndarray) -> bytes:
    arr = np.asarray(arr, dtype=np.float32)
    return (
        struct.pack("=I", arr.ndim)
        + struct.pack("=" + "I" * arr.ndim, *arr.shape)
        + arr.tobytes()
    )


def test_decode_np_restores_nested_msgpack_arrays():
    raw = np.arange(6, dtype=np.float32).reshape(2, 3)
    encoded = {"outer": [{"__nd__": True, "dtype": str(raw.dtype), "shape": raw.shape, "data": raw.tobytes()}]}

    decoded = _decode_np(encoded)

    np.testing.assert_array_equal(decoded["outer"][0], raw)


def test_unpack_ndarray_validates_header_and_shape():
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    got = _unpack_ndarray(_pack_float32_ndarray(arr))

    np.testing.assert_array_equal(got, arr)

    with pytest.raises(ValueError, match="too short"):
        _unpack_ndarray(b"\x01")

    with pytest.raises(ValueError, match="expected"):
        _unpack_ndarray(struct.pack("=I", 1) + struct.pack("=I", 4) + np.zeros(3, dtype=np.float32).tobytes())


def test_split_loader_filters_known_invalid_indicators(tmp_path: Path):
    split_payload = {
        "full": {
            "s1": {
                "train": [[1, 12, 58], [1, 12, 59]],
                "val": [],
                "test": [],
            }
        }
    }
    with gzip.open(tmp_path / "indeces.pkl.gz", "wb") as f:
        pickle.dump(split_payload, f)

    ds = object.__new__(M4HumanLMDBPoseDataset)
    ds.cache_path = tmp_path
    ds.split_id = "s1"
    ds.split = "train"

    assert ds._load_split_indices() == [[1, 12, 59]]


def test_temporal_offset_parser_defaults_to_history_window():
    mode, offsets = _parse_temporal_offsets({"merge_nframes": 4}, {})

    assert mode == "hist4"
    assert offsets == (-3, -2, -1, 0)


def test_temporal_offset_parser_accepts_named_oracle_window():
    mode, offsets = _parse_temporal_offsets({}, {"frame_mode": "center5_oracle"})

    assert mode == "center5_oracle"
    assert offsets == (-2, -1, 0, 1, 2)


def test_temporal_lookup_pads_inside_subject_action_sequence():
    ds = object.__new__(M4HumanLMDBPoseDataset)
    ds.data_list = [[1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 2, 0], [2, 1, 0]]
    ds.temporal_offsets = (-2, -1, 0, 1, 2)
    ds._read_radar_pc = lambda indicator: np.asarray(
        [[indicator[2], indicator[0], indicator[1], 1.0]],
        dtype=np.float32,
    )

    frames = ds._load_temporal_radar(1)

    assert [int(frame[0, 0]) for frame in frames] == [0, 0, 1, 2, 2]
    assert [int(frame[0, 1]) for frame in frames] == [1, 1, 1, 1, 1]
    assert [int(frame[0, 2]) for frame in frames] == [1, 1, 1, 1, 1]


def test_process_points_pads_and_uses_deterministic_eval_sampling():
    ds = object.__new__(M4HumanLMDBPoseDataset)
    ds.n_points = 3
    ds.input_channels = 4
    ds.split = "val"
    ds.random_sample_train = True

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
    torch.testing.assert_close(sampled[:, 0], torch.tensor([1.0, 3.0, 5.0]))

    padded = ds._process_points(points[:1])
    assert padded.shape == (3, 4)
    torch.testing.assert_close(padded[0], points[0])
    torch.testing.assert_close(padded[1:], torch.zeros(2, 4))


def test_shared_input_root_crop_preserves_raw_coordinates():
    cfg = resolve_root_alignment_config(
        {
            "normalized": True,
            "input_root_normalized": False,
            "input_root_crop": True,
            "input_root_crop_x_range": [-0.5, 0.5],
            "input_root_crop_y_range": [-0.5, 0.5],
            "input_root_crop_z_range": [-0.5, 0.5],
        }
    )
    points = np.asarray(
        [
            [10.0, 0.0, 0.0, 0.1],
            [10.4, 0.0, 0.0, 0.2],
            [11.0, 0.0, 0.0, 0.3],
        ],
        dtype=np.float32,
    )

    aligned = apply_input_root_alignment(points, root=np.asarray([10.0, 0.0, 0.0], dtype=np.float32), cfg=cfg)

    np.testing.assert_allclose(aligned[:, 0], np.asarray([10.0, 10.4], dtype=np.float32))


def test_getitem_can_use_shared_root_alignment_switches():
    ds = object.__new__(M4HumanLMDBPoseDataset)
    ds.data_list = [[1, 2, 3]]
    ds.n_points = 1
    ds.input_channels = 4
    ds.split = "val"
    ds.random_sample_train = False
    ds.point_sampling = ""
    ds.normalize_z = False
    ds.intensity_scale = 1.0
    ds.load_pose = True
    ds.root_alignment = resolve_root_alignment_config(
        {
            "normalized": True,
            "input_root_normalized": True,
            "input_root_crop": False,
            "target_root_normalized": True,
        },
        default_input_root_normalized=False,
        default_input_root_crop=False,
    )
    ds.target_root_normalized = True
    ds._load_temporal_radar = lambda idx: [np.asarray([[2.0, 0.0, 0.0, 0.5]], dtype=np.float32)]
    ds._load_joints = lambda indicator: np.asarray(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    item = ds[0]

    torch.testing.assert_close(item["mmwave_points"][0, :3], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(item["target_poses"][0], torch.zeros(3))
    torch.testing.assert_close(item["target_poses"][1], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(item["target_poses_absolute"][0], torch.tensor([1.0, 0.0, 0.0]))


def test_getitem_keeps_absolute_joints_in_same_z_frame_as_radar():
    ds = object.__new__(M4HumanLMDBPoseDataset)
    ds.data_list = [[1, 2, 3]]
    ds.n_points = 2
    ds.input_channels = 4
    ds.split = "val"
    ds.random_sample_train = False
    ds.normalize_z = True
    ds.normalized = True
    ds.intensity_scale = 1.0
    ds._load_temporal_radar = lambda idx: [np.asarray([[0.0, 0.0, 2.0, 0.5]], dtype=np.float32)]
    ds._load_joints = lambda indicator: np.asarray(
        [[0.0, 0.0, 0.5], [0.0, 0.0, 1.5]],
        dtype=np.float32,
    )

    item = ds[0]

    torch.testing.assert_close(item["mmwave_points"][0, 2], torch.tensor(0.5))
    torch.testing.assert_close(item["target_poses_absolute"][:, 2], torch.tensor([0.5, 1.5]))
    torch.testing.assert_close(item["target_poses"][:, 2], torch.tensor([0.0, 1.0]))
