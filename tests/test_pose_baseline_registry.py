from pathlib import Path

import pytest
import torch

from wave2body.models import (
    PTPoseModel,
    PoseMPJPELoss,
)
from wave2body.models.baselines import (
    build_pose_baseline,
    pose_baseline_step,
)
from wave2body.models.mmdiff import MMDiffPoseLoss, MMDiffPoseModel
from wave2body.models.mmmesh import MMMeshLoss, MMMeshModel
from wave2body.train.common import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeEncoder(torch.nn.Module):
    def forward(self, radar):
        batch_size = radar.shape[0]
        return radar.new_zeros(batch_size, 17, 3), radar.new_ones(batch_size, 17, 8)


def test_build_point_transformer_baseline_from_existing_config_shape():
    bundle = build_pose_baseline(
        {
            "baseline": {"name": "point_transformer"},
            "point_transformer": {
                "data_format": {"radar_input_c": 4, "num_joints": 2},
                "model": {"scale": "zepto", "nneighbor": 2, "dropout": 0.0},
            },
        },
        device="cpu",
    )

    assert bundle.name == "point_transformer"
    assert isinstance(bundle.model, PTPoseModel)
    assert isinstance(bundle.criterion, PoseMPJPELoss)


def test_build_mmmesh_baseline_from_canonical_config():
    cfg = load_config(
        REPO_ROOT / "wave2body/configs/hpe_baselines/mmmesh_mmbody.yaml",
        resolve_datasets=False,
    )
    cfg["mmmesh"]["body_decoder_config"]["enabled"] = False

    bundle = build_pose_baseline(cfg, device="cpu")

    assert bundle.name == "mmmesh"
    assert isinstance(bundle.model, MMMeshModel)
    assert isinstance(bundle.criterion, MMMeshLoss)
    assert bundle.model.base.input_dim == 5
    assert bundle.model.anchor_module.nsample == 8


def test_build_mmdiff_baseline_from_canonical_config_with_encoder_override():
    cfg = load_config(
        REPO_ROOT / "wave2body/configs/hpe_baselines/mmdiff_mmbody.yaml",
        resolve_datasets=False,
    )
    cfg["mmdiff"].update(
        {
            "num_joints": 17,
            "condition_dim": 8,
            "hidden_dim": 16,
            "num_layers": 1,
            "num_heads": 2,
            "local_points": 4,
        }
    )

    bundle = build_pose_baseline(cfg, device="cpu", feature_encoder=_FakeEncoder())

    assert bundle.name == "mmdiff"
    assert isinstance(bundle.model, MMDiffPoseModel)
    assert isinstance(bundle.criterion, MMDiffPoseLoss)
    radar = torch.randn(1, 4, 8, 5)
    outputs = bundle.model(radar)
    assert outputs["pred_pose"].shape == (1, 17, 3)


def test_build_pose_baseline_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unsupported pose baseline"):
        build_pose_baseline({"baseline": {"name": "unknown"}}, device="cpu")


def test_pose_baseline_step_supports_point_transformer_contract():
    bundle = build_pose_baseline(
        {
            "baseline": {"name": "point_transformer"},
            "point_transformer": {
                "data_format": {"radar_input_c": 4, "num_joints": 2},
                "model": {"scale": "zepto", "nneighbor": 2, "dropout": 0.0},
            },
        },
        device="cpu",
    )
    batch = {
        "mmwave_points": torch.randn(2, 8, 4),
        "target_poses": torch.randn(2, 2, 3),
    }

    outputs, losses = pose_baseline_step(bundle, batch, device="cpu")

    assert outputs["pred_pose"].shape == (2, 2, 3)
    assert torch.isfinite(losses["total_loss"])


def test_pose_baseline_step_supports_mmdiff_contract():
    bundle = build_pose_baseline(
        {
            "baseline": {"name": "mmdiff"},
            "mmdiff": {
                "num_joints": 17,
                "radar_input_dim": 6,
                "condition_dim": 8,
                "hidden_dim": 16,
                "history_frames": 3,
                "num_layers": 1,
                "num_heads": 2,
                "dropout": 0.0,
                "local_points": 4,
            },
            "mmdiff_loss": {"pose_weight": 1.0, "reduction": "mse"},
        },
        device="cpu",
        feature_encoder=_FakeEncoder(),
    )
    batch = {
        "mmwave_points": torch.randn(2, 4, 8, 6),
        "target_poses": torch.randn(2, 17, 3),
        "timesteps": torch.tensor([0, 1]),
    }

    outputs, losses = pose_baseline_step(bundle, batch, device="cpu")

    assert outputs["pred_pose"].shape == (2, 17, 3)
    assert torch.isfinite(losses["total_loss"])


def test_pose_baseline_step_supports_mmmesh_single_frame_targets():
    bundle = build_pose_baseline(
        {
            "baseline": {"name": "mmmesh"},
            "mmmesh": {"input_dim": 5, "anchor_nsample": 4},
            "mmmesh_loss": {"vertices_weight": 0.0, "joints_weight": 0.0, "rotmat_weight": 0.1},
        },
        device="cpu",
    )
    batch = {
        "mmwave_points": torch.randn(2, 16, 5),
        "target_rotmat": torch.eye(3).view(1, 1, 3, 3).expand(2, 9, 3, 3),
        "target_translation": torch.zeros(2, 3),
        "target_betas": torch.zeros(2, 10),
        "target_gender": torch.ones(2, 1),
    }

    outputs, losses = pose_baseline_step(bundle, batch, device="cpu")

    assert outputs["pred_translation"].shape == (2, 1, 3)
    assert "rotmat_loss" in losses
    assert "translation_loss" in losses
    assert torch.isfinite(losses["total_loss"])


def test_pose_baseline_step_uses_pose_root_for_mmmesh_target_gap():
    bundle = build_pose_baseline(
        {
            "baseline": {"name": "mmmesh"},
            "mmmesh": {"input_dim": 6, "anchor_nsample": 4},
            "mmmesh_loss": {
                "rotmat_weight": 0.0,
                "translation_weight": 0.0,
                "root_xy_weight": 1.0,
                "betas_weight": 0.0,
                "gender_weight": 0.0,
                "vertices_weight": 0.0,
                "joints_weight": 0.0,
            },
        },
        device="cpu",
    )
    batch = {
        "mmwave_points": torch.randn(2, 3, 16, 6),
        "target_poses": torch.randn(2, 24, 3),
    }

    _, losses = pose_baseline_step(bundle, batch, device="cpu")

    assert "root_xy_loss" in losses
    assert torch.isfinite(losses["total_loss"])
