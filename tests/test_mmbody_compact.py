from __future__ import annotations

import numpy as np
import torch

from wave2body.dataset.radar_pose.mmbody import MMBodyDataset
from wave2body.dataset.radar_pose.mmbody_compact import compact_mmbody_dataset
from wave2body.dataset.human_body.mmbody import MMBodyJointsDataset


def _write_sample(root, frame, radar, joints):
    sequence = root / "train" / "sequence_0"
    (sequence / "radar").mkdir(parents=True, exist_ok=True)
    (sequence / "mesh").mkdir(parents=True, exist_ok=True)
    np.save(sequence / "radar" / f"frame_{frame}.npy", radar)
    np.savez(sequence / "mesh" / f"frame_{frame}.npz", joints=joints, vertices=np.ones((10, 3)))


def _config(root):
    return {
        "dataset_path": str(root),
        "data_format": {
            "n_points": 8,
            "merge_nframes": 2,
            "radar_input_c": 5,
            "num_joints": 22,
            "normalized": False,
            "input_root_crop": False,
            "target_root_normalized": False,
            "point_sampling": "linspace",
        },
    }


def test_compact_mmbody_matches_raw_loader_and_deduplicates(tmp_path):
    raw_root = tmp_path / "raw"
    compact_root = tmp_path / "compact"
    radar = np.asarray(
        [[float(i), float(i + 1), float(i + 2), 1.0e-38 * (i + 1), 99.0, float(i * 10)] for i in range(8)],
        dtype=np.float64,
    )
    joints = np.arange(55 * 3, dtype=np.float64).reshape(55, 3) / 100.0
    _write_sample(raw_root, 0, radar, joints)
    _write_sample(raw_root, 1, radar, joints)

    manifest = compact_mmbody_dataset(raw_root, compact_root)
    assert manifest["sample_count"] == 2
    assert manifest["unique_radar_count"] == 1
    assert manifest["unique_pose_count"] == 1

    raw_dataset = MMBodyDataset(_config(raw_root), split="train")
    compact_dataset = MMBodyDataset(_config(compact_root), split="train")
    assert len(raw_dataset) == len(compact_dataset) == 2
    for index in range(2):
        raw_item = raw_dataset[index]
        compact_item = compact_dataset[index]
        torch.testing.assert_close(compact_item["mmwave_points"], raw_item["mmwave_points"], rtol=0, atol=0)
        torch.testing.assert_close(compact_item["target_poses"], raw_item["target_poses"], rtol=0, atol=0)
        torch.testing.assert_close(
            compact_item["target_poses_absolute"], raw_item["target_poses_absolute"], rtol=0, atol=0
        )
        assert compact_item["mmbody_scenario"] == raw_item["mmbody_scenario"]
        assert compact_item["mmbody_sequence"] == raw_item["mmbody_sequence"]
        assert compact_item["mmbody_frame"] == raw_item["mmbody_frame"]

    raw_dataset.close()
    compact_dataset.close()

    joints_dataset = MMBodyJointsDataset(
        root_path=str(compact_root),
        num_joints=22,
        device="cpu",
        label=0,
        name="MMBody",
        splits=["train"],
        normalized=False,
        use_cache=False,
    )
    assert len(joints_dataset) == 2
    compact_joints, label = joints_dataset[0]
    assert label == 0
    torch.testing.assert_close(compact_joints, torch.from_numpy(joints[:22].astype(np.float32)))
