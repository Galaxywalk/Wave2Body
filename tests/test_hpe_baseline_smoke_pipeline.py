# -*- coding: utf-8 -*-

import numpy as np

from wave2body.train.pipelines.pose_baseline import run_pose_baseline


def _write_sequence(root, split_parts, *, frames: int, num_joints: int, with_mmmesh_targets: bool) -> None:
    seq_dir = root.joinpath(*split_parts)
    radar_dir = seq_dir / "radar"
    mesh_dir = seq_dir / "mesh"
    radar_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    for frame in range(frames):
        radar = np.asarray(
            [
                [0.01 * frame, 0.00, 0.00, 1.0e-38, 9.0, 10.0],
                [0.10, 0.02 * frame, 0.00, 2.0e-38, 8.0, 20.0],
                [0.20, 0.00, 0.03 * frame, 3.0e-38, 7.0, 30.0],
                [0.30, 0.00, 0.00, 4.0e-38, 6.0, 40.0],
            ],
            dtype=np.float32,
        )
        np.save(radar_dir / f"frame_{frame}.npy", radar)

        joints = np.zeros((max(num_joints, 24), 3), dtype=np.float32)
        joints[:, 0] = np.linspace(0.0, 0.2, joints.shape[0], dtype=np.float32)
        joints[:, 1] = 0.01 * frame
        payload = {"joints": joints}
        if with_mmmesh_targets:
            payload.update(
                {
                    "vertices": np.zeros((8, 3), dtype=np.float32),
                    "trans": np.asarray([0.01 * frame, 0.0, 0.0], dtype=np.float32),
                    "betas": np.zeros(10, dtype=np.float32),
                    "gender": np.asarray([1.0], dtype=np.float32),
                    "rotmat": np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 24, axis=0),
                }
            )
        np.savez(mesh_dir / f"frame_{frame}.npz", **payload)


def _write_mmbody_root(tmp_path, *, num_joints: int, with_mmmesh_targets: bool):
    root = tmp_path / "mmbody"
    _write_sequence(root, ("train", "sequence_0"), frames=2, num_joints=num_joints, with_mmmesh_targets=with_mmmesh_targets)
    _write_sequence(
        root,
        ("test", "lab1", "sequence_0"),
        frames=1,
        num_joints=num_joints,
        with_mmmesh_targets=with_mmmesh_targets,
    )
    return root


def _training_cfg(*, batch_size: int = 1):
    return {
        "batch_size": int(batch_size),
        "epochs": 1,
        "lr": 1.0e-4,
        "optimizer": "adam",
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "amp": False,
        "precision": "fp32",
        "log_interval": 10,
        "train_val_split": {"enabled": False, "val_split": "test"},
    }


def test_mmdiff_pose_baseline_pipeline_smoke_on_synthetic_mmbody(tmp_path):
    dataset_root = _write_mmbody_root(tmp_path, num_joints=17, with_mmmesh_targets=False)
    config = {
        "baseline": {"name": "mmdiff"},
        "dataset": {"name": "mmbody"},
        "dataset_path": str(dataset_root),
        "data_format": {
            "n_points": 8,
            "merge_nframes": 1,
            "radar_input_c": 5,
            "num_joints": 17,
            "normalized": False,
            "input_root_crop": False,
            "target_root_normalized": False,
            "point_sampling": "linspace",
        },
        "mmdiff": {
            "num_joints": 17,
            "radar_input_dim": 5,
            "condition_dim": 8,
            "hidden_dim": 16,
            "history_frames": 2,
            "num_layers": 1,
            "num_heads": 2,
            "dropout": 0.0,
            "local_points": 4,
        },
        "mmdiff_loss": {"pose_weight": 1.0, "reduction": "mse"},
        "training": {**_training_cfg(batch_size=1), "run_name": "mmdiff_smoke"},
    }

    summary = run_pose_baseline(
        config,
        work_dir=tmp_path / "runs" / "mmdiff",
        device="cpu",
        max_train_batches=1,
        max_val_batches=1,
        max_test_batches=1,
    )

    assert summary["baseline"] == "mmdiff"
    assert summary["best_metric_name"] == "mpjpe_mm"
    assert (tmp_path / "runs" / "mmdiff" / "checkpoints" / "latest.pth").exists()
    assert (tmp_path / "runs" / "mmdiff" / "training_results.json").exists()


def test_mmmesh_pose_baseline_pipeline_smoke_on_synthetic_mmbody(tmp_path):
    dataset_root = _write_mmbody_root(tmp_path, num_joints=24, with_mmmesh_targets=True)
    config = {
        "baseline": {"name": "mmmesh"},
        "dataset": {"name": "mmbody"},
        "dataset_path": str(dataset_root),
        "data_format": {
            "n_points": 8,
            "merge_nframes": 1,
            "radar_input_c": 5,
            "num_joints": 24,
            "normalized": False,
            "input_root_crop": False,
            "target_root_normalized": False,
            "point_sampling": "linspace",
            "return_mmmesh_targets": True,
        },
        "mmmesh": {
            "input_dim": 5,
            "anchor_nsample": 4,
        },
        "mmmesh_loss": {
            "rotmat_weight": 0.1,
            "translation_weight": 1.0,
            "root_xy_weight": 1.0,
            "betas_weight": 0.1,
            "gender_weight": 1.0,
            "joints_weight": 0.0,
            "vertices_weight": 0.0,
        },
        "training": {**_training_cfg(batch_size=2), "run_name": "mmmesh_smoke"},
    }

    summary = run_pose_baseline(
        config,
        work_dir=tmp_path / "runs" / "mmmesh",
        device="cpu",
        max_train_batches=1,
        max_val_batches=1,
        max_test_batches=1,
    )

    assert summary["baseline"] == "mmmesh"
    assert summary["best_metric_name"] == "total_loss"
    assert (tmp_path / "runs" / "mmmesh" / "checkpoints" / "latest.pth").exists()
    assert (tmp_path / "runs" / "mmmesh" / "training_results.json").exists()
