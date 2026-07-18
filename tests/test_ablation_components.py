# -*- coding: utf-8 -*-

import torch

from wave2body.models.ablations import NoBodyTokenPooledMLP, NoBodyTokenRelativeRootHeadMLP
from wave2body.models.ablations.token_estimators import create_ablation_sensor_tokenizer
from wave2body.train.common import load_config
from wave2body.train.pipelines.ablation_components import _relative_root_loss_from_batch


def test_no_body_token_pooled_mlp_forward_shape():
    config = {
        "data_format": {"num_joints": 22},
        "mmwave_tokenizer": {
            "method": "mae",
            "patch_policy": "spatial_fps_knn",
            "input_channels": 4,
            "num_groups": 4,
            "group_size": 4,
            "latent_dim": 8,
            "hidden_dim": 8,
            "transformer_depth": 1,
            "num_heads": 2,
            "dropout": 0.0,
            "mask_ratio": 0.5,
        },
        "sensor_tokens": {"mode": "token_center_concat", "zero_invalid": True},
        "direct_regressor": {"hidden_dim": 16, "num_layers": 2, "dropout": 0.0, "use_centers": True},
        "training": {"freeze_mmwave_tokenizer": True},
    }
    model = NoBodyTokenPooledMLP(config)
    points = torch.randn(2, 16, 4)
    out = model(points)

    assert out["pred_pose"].shape == (2, 22, 3)
    assert out["mmwave_tokens"].shape[:2] == (2, 4)
    assert out["mmwave_token_mask"].shape == (2, 4)


def test_no_body_relative_root_forward_composes_absolute_pose():
    config = {
        "data_format": {"num_joints": 22},
        "mmwave_tokenizer": {
            "method": "mae",
            "patch_policy": "spatial_fps_knn",
            "input_channels": 4,
            "num_groups": 4,
            "group_size": 4,
            "latent_dim": 8,
            "hidden_dim": 8,
            "transformer_depth": 1,
            "num_heads": 2,
            "dropout": 0.0,
            "mask_ratio": 0.5,
        },
        "sensor_tokens": {"mode": "token_center_concat", "zero_invalid": True},
        "direct_regressor": {"hidden_dim": 16, "num_layers": 2, "dropout": 0.0, "use_centers": True},
        "root_head": {
            "hidden_dim": 16,
            "attention_hidden_dim": 16,
            "attention_heads": 2,
            "num_layers": 2,
            "dropout": 0.0,
            "prior_mode": "point_intensity_weighted_mean",
        },
        "training": {"freeze_mmwave_tokenizer": True},
    }
    model = NoBodyTokenRelativeRootHeadMLP(config)
    points = torch.randn(2, 16, 4)
    out = model(points)

    assert model.regressor.sensor_token_dim == 11
    assert model.root_head.sensor_token_dim == 8
    assert out["pred_pose_relative"].shape == (2, 22, 3)
    assert out["pred_root"].shape == (2, 3)
    torch.testing.assert_close(out["pred_pose_absolute"], out["pred_pose_relative"] + out["pred_root"].unsqueeze(1))
    torch.testing.assert_close(out["pred_pose"], out["pred_pose_absolute"])


def test_no_body_relative_root_loss_uses_wave2body_weights():
    outputs = {
        "pred_pose_relative": torch.ones(2, 4, 3),
        "pred_pose_absolute": torch.ones(2, 4, 3) * 2.0,
        "pred_root": torch.ones(2, 3) * 3.0,
    }
    target_relative = torch.zeros(2, 4, 3)
    target_absolute = torch.zeros(2, 4, 3)
    loss = _relative_root_loss_from_batch(
        outputs=outputs,
        target_relative=target_relative,
        target_absolute=target_absolute,
        root_index=0,
        training_cfg={},
    )

    torch.testing.assert_close(loss["relative_pose_loss"], torch.tensor(1.0))
    torch.testing.assert_close(loss["absolute_pose_loss"], torch.tensor(4.0))
    torch.testing.assert_close(loss["root_loss"], torch.tensor(9.0))
    torch.testing.assert_close(loss["total_loss"], torch.tensor(11.0))


def test_ablation_sensor_tokenizer_allows_raw_point_pointnet():
    tokenizer = create_ablation_sensor_tokenizer(
        {
            "method": "raw_point_pointnet",
            "input_channels": 4,
            "num_groups": 4,
            "latent_dim": 8,
            "hidden_dim": 8,
            "point_hidden_dim": 8,
            "num_point_layers": 1,
            "num_token_layers": 1,
            "dropout": 0.0,
        }
    )
    out = tokenizer.encode_tokens(torch.randn(2, 12, 4))

    assert out["token_embeddings"].shape == (2, 4, 8)
    assert out["valid_mask"].shape == (2, 4)
    assert out["centers"].shape == (2, 4, 3)


def test_ablation_configs_resolve_current_structure():
    no_body = load_config("wave2body/configs/ablations/no_body_token_m4human_s2.yaml", resolve_datasets=False)
    no_body_relative_root = load_config(
        "wave2body/configs/ablations/no_body_token_relative_root_m4human_s2.yaml",
        resolve_datasets=False,
    )
    no_radar_abs = load_config("wave2body/configs/ablations/no_radar_tokenizer_absolute_m4human_s2.yaml", resolve_datasets=False)

    assert no_body["training"]["stage"] == "ablation_no_body_token"
    assert no_body["direct_regressor"]["architecture"] == "pooled_mlp"
    assert no_body_relative_root["training"]["stage"] == "ablation_no_body_relative_root"
    assert no_body_relative_root["data_format"]["target_root_normalized"] is True
    assert no_body_relative_root["root_head"]["architecture"] == "query_attn"
    assert no_body_relative_root["root_head"]["prior_mode"] == "point_intensity_weighted_mean"
    assert no_radar_abs["mmwave_tokenizer"]["method"] == "raw_point_pointnet"
    assert no_radar_abs["root_head"]["architecture"] == "query_attn"
    assert no_radar_abs["training"]["stage"] == "absolute_root_head"
