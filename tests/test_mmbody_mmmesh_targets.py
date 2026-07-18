# -*- coding: utf-8 -*-

import numpy as np
import torch

from wave2body.dataset.radar_pose.mmbody import MMBodyDataset


def test_mmbody_can_export_optional_mmmesh_targets(tmp_path):
    seq_dir = tmp_path / "mmbody" / "train" / "sequence_0"
    radar_dir = seq_dir / "radar"
    mesh_dir = seq_dir / "mesh"
    radar_dir.mkdir(parents=True)
    mesh_dir.mkdir(parents=True)

    radar = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0e-38, 9.0, 10.0],
            [0.1, 0.0, 0.0, 2.0e-38, 8.0, 20.0],
            [0.2, 0.0, 0.0, 3.0e-38, 7.0, 30.0],
            [0.3, 0.0, 0.0, 4.0e-38, 6.0, 40.0],
        ],
        dtype=np.float32,
    )
    np.save(radar_dir / "frame_0.npy", radar)

    joints = np.arange(24 * 3, dtype=np.float32).reshape(24, 3)
    vertices = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)
    rotmat = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 24, axis=0)
    rotmat[16] = np.eye(3, dtype=np.float32) * 2.0
    np.savez(
        mesh_dir / "frame_0.npz",
        joints=joints,
        vertices=vertices,
        trans=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        betas=np.arange(10, dtype=np.float32),
        gender=np.asarray("male"),
        rotmat=rotmat,
    )

    ds = MMBodyDataset(
        {
            "dataset_path": str(tmp_path / "mmbody"),
            "data_format": {
                "n_points": 4,
                "merge_nframes": 1,
                "radar_input_c": 5,
                "num_joints": 24,
                "normalized": False,
                "input_root_crop": False,
                "target_root_normalized": False,
                "point_sampling": "linspace",
                "return_mmmesh_targets": True,
            },
        },
        split="train",
    )

    item = ds[0]

    assert len(ds) == 1
    assert item["mmwave_points"].shape == (4, 5)
    torch.testing.assert_close(item["target_translation"], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(item["target_root_xy"], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(item["target_betas"], torch.arange(10, dtype=torch.float32))
    torch.testing.assert_close(item["target_gender"], torch.tensor([1.0]))
    torch.testing.assert_close(item["target_joints"], torch.from_numpy(joints))
    torch.testing.assert_close(item["target_vertices"], torch.from_numpy(vertices))
    assert item["target_rotmat"].shape == (9, 3, 3)
    torch.testing.assert_close(item["target_rotmat"][5], torch.eye(3) * 2.0)
