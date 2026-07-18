# -*- coding: utf-8 -*-

import argparse

import torch

from scripts.hpe_point_transformer.evaluate import _apply_eval_overrides
from scripts.hpe_point_transformer.train import _apply_experiment_overrides, resolve_config
from wave2body.models.point_transformer.pose_baseline import PoseMPJPELoss
from wave2body.train.common import load_config
from wave2body.train.pipelines.pt_pose import evaluate_mpjpe_only
from wave2body.utils.runtime_utils import configure_runtime


def _args(**overrides):
    defaults = {
        "split_id": None,
        "run_name": None,
        "num_workers": None,
        "smoke": False,
        "run_tag": "",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_hpe_point_transformer_config_uses_radar_points_and_absolute_targets():
    config = _apply_experiment_overrides(
        load_config(resolve_config("m4human_absolute_hist4"), resolve_datasets=False),
        _args(split_id="s3", run_name="unit"),
    )

    assert config["data_format"]["normalized"] is False
    assert config["m4human"]["pose_source"] == "param_joints"
    assert config["m4human"]["normalize_z"] is True
    assert "scale" not in config["m4human"]
    assert config["m4human"]["frame_mode"] == "hist4"
    assert config["m4human"]["temporal_offsets"] == [-3, -2, -1, 0]
    assert config["coord_transform"] == "m4human_radar_absolute_z_shifted_no_root_normalization"
    assert config["evaluation"]["primary_metric"] == "absolute_mpjpe"
    assert config["evaluation"]["diagnostic_metrics"] == [
        "absolute_mpjpe",
        "root_error",
        "root_aligned_mpjpe",
        "p_mpjpe",
    ]
    assert config["training"]["batch_size"] == 224
    assert config["training"]["num_workers"] == 16


def test_hpe_point_transformer_eval_overrides_preserves_target_mode():
    config = load_config(resolve_config("m4human_absolute_hist4"), resolve_datasets=False)
    config["data_format"]["normalized"] = False
    config["evaluation"] = {
        "mpjpe_only": True,
        "primary_metric": "absolute_mpjpe",
        "diagnostic_metrics": ["root_error"],
    }

    eval_config = _apply_eval_overrides(
        config,
        split_id="s3",
        batch_size=64,
        num_workers=0,
        include_action_groups=True,
    )

    assert "scale" not in eval_config["m4human"]
    assert eval_config["m4human"]["split"] == "s3"
    assert eval_config["data_format"]["normalized"] is False
    assert eval_config["evaluation"]["primary_metric"] == "absolute_mpjpe"
    assert "root_aligned_mpjpe" in eval_config["evaluation"]["diagnostic_metrics"]
    assert eval_config["training"]["batch_size"] == 64
    assert eval_config["training"]["num_workers"] == 0
    assert eval_config["training"]["pin_memory"] is False


class _FixedPoseModel(torch.nn.Module):
    def __init__(self, pred_pose: torch.Tensor):
        super().__init__()
        self.register_buffer("pred_pose", pred_pose.float())

    def forward(self, mmwave_points: torch.Tensor):
        pred = self.pred_pose[: mmwave_points.shape[0]]
        return {"pred_pose": pred, "joint_feat": torch.empty(pred.shape[0], pred.shape[1], 0)}


def test_pt_m4human_pelvis_centered_eval_reports_mpjpe():
    target = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ]
    )
    pred = target + torch.tensor([1.0, 0.0, 0.0])
    batch = {
        "mmwave_points": torch.zeros(2, 8, 4),
        "target_poses": target,
        "indicator": torch.tensor([[1, 1, 0], [1, 2, 0]], dtype=torch.long),
    }
    runtime = configure_runtime({"precision": "fp32", "amp": False}, torch.device("cpu"), log_prefix="[test]")

    metrics = evaluate_mpjpe_only(
        model=_FixedPoseModel(pred),
        dataloader=[batch],
        criterion=PoseMPJPELoss(),
        device=torch.device("cpu"),
        runtime=runtime,
        desc="unit",
        eval_cfg={
            "primary_metric": "mpjpe",
            "diagnostic_metrics": ["p_mpjpe"],
            "compute_p_mpjpe": True,
            "m4human_action_groups": True,
        },
    )

    assert metrics["mpjpe_mm"] == 1000.0
    assert metrics["mpjpe_by_joint_mm"] == [1000.0, 1000.0, 1000.0]
    assert "absolute_mpjpe_mm" not in metrics
    assert "root_error_mm" not in metrics
    assert "root_aligned_mpjpe_by_joint_mm" not in metrics
    assert metrics["p_mpjpe_mm"] < 1.0e-3
    assert metrics["mpjpe_by_action_group_mm"]["1_50"] == 1000.0
    assert metrics["mpjpe_by_action_group_by_joint_mm"]["1_50"] == [1000.0, 1000.0, 1000.0]
