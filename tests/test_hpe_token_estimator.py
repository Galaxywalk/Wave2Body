# -*- coding: utf-8 -*-

import torch
import pytest

from wave2body.models.hpe_token.absolute_root import (
    ROOT_POINT_STATS_DIM,
    AbsoluteRootHeadEstimatorModel,
    masked_point_root_stats,
)
from wave2body.models.hpe_token.baselines.absolute_root import (
    AbsoluteRootHeadEstimatorModel as BaselineAbsoluteRootHeadEstimatorModel,
)
from wave2body.models.hpe_token.estimator import TokenHPEEstimatorModel
from wave2body.train.optim import build_hpe_token_optimizer


def _toy_points(batch_size: int = 2) -> torch.Tensor:
    base = torch.tensor(
        [
            [0.00, 0.00, 0.00, 1.0],
            [0.05, 0.00, 0.00, 0.2],
            [1.00, 0.00, 0.00, 0.4],
            [2.00, 0.00, 0.00, 0.6],
            [3.00, 0.00, 0.00, 0.8],
            [0.00, 0.00, 0.00, 0.0],
        ],
        dtype=torch.float32,
    )
    return base.unsqueeze(0).repeat(batch_size, 1, 1)


def _toy_poses(batch_size: int = 2) -> torch.Tensor:
    joints = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    return joints.unsqueeze(0).repeat(batch_size, 1, 1)


def _cfg(*, freeze_mmwave_tokenizer: bool) -> dict:
    return {
        "mmwave_tokenizer_config": {
            "method": "mae",
            "input_channels": 4,
            "num_groups": 3,
            "group_size": 4,
            "latent_dim": 8,
            "hidden_dim": 8,
            "patch_policy": "spatial_fps_knn",
            "mask_ratio": 0.5,
            "transformer_depth": 1,
            "num_heads": 2,
            "dropout": 0.0,
        },
        "sensor_token_config": {"mode": "token_center_concat"},
        "token_translator_config": {
            "mmwave_token_dim": 11,
            "pose_token_dim": 8,
            "num_pose_tokens": 2,
            "num_mmwave_tokens": 3,
            "hidden_dim": 16,
            "num_cross_attn_layers": 1,
            "num_self_attn_layers": 1,
            "heads": 2,
            "dim_head": 8,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "use_sdpa": False,
            "codebook_size": 4,
        },
        "vqvae_config": {
            "model_type": "part",
            "num_joints": 4,
            "input_dim": 3,
            "token_dim": 8,
            "part_hidden_dim": 16,
            "part_dropout": 0.0,
            "root_index": 0,
            "part_token_layout_order": ["Body"],
            "part_joints": {"Body": [0, 1, 2, 3]},
            "part_token_num": {"Body": 2},
            "part_codebook_size": {"Body": 4},
            "drop_rate": 0.0,
            "init_from_data": False,
        },
        "freeze_mmwave_tokenizer": freeze_mmwave_tokenizer,
        "freeze_vqvae": True,
    }


def test_hpe_token_forward_sensor_token_contract():
    model = TokenHPEEstimatorModel(**_cfg(freeze_mmwave_tokenizer=True))
    out = model(_toy_points(), _toy_poses())

    assert model.sensor_token_dim == 11
    assert out["mmwave_tokens"].shape == (2, 3, 11)
    assert out["mmwave_token_mask"].shape == (2, 3)
    assert out["mmwave_token_centers"].shape == (2, 3, 3)
    assert out["pred_z_e"].shape == (2, 2, 8)
    assert out["pred_pose"].shape == (2, 4, 3)
    assert out["target_indices"].shape == (2, 2)
    assert out["pred_logits"].shape == (2, 2, 4)


def test_hpe_token_mlp_pool_translator_is_baseline_only():
    cfg = _cfg(freeze_mmwave_tokenizer=True)
    cfg["sensor_token_config"] = {"mode": "token_only"}
    cfg["token_translator_config"].update(
        {
            "architecture": "mlp_pool",
            "mmwave_token_dim": 8,
            "hidden_dim": 16,
            "num_layers": 2,
        }
    )
    with pytest.raises(ValueError, match="Historical translator variants"):
        TokenHPEEstimatorModel(**cfg)


def test_hpe_token_candidate_reranker_is_baseline_only():
    cfg = _cfg(freeze_mmwave_tokenizer=True)
    cfg["token_translator_config"]["candidate_reranker"] = {"enabled": True}

    with pytest.raises(ValueError, match="candidate_reranker is a baseline-only"):
        TokenHPEEstimatorModel(**cfg)


def test_hpe_token_rejects_raw_point_mlp_sensor_encoder():
    cfg = _cfg(freeze_mmwave_tokenizer=False)
    cfg["mmwave_tokenizer_config"].update(
        {
            "method": "raw_point_mlp",
            "point_hidden_dim": 8,
            "num_point_layers": 2,
            "num_self_attn_layers": 1,
        }
    )
    with pytest.raises(ValueError, match="only supports method='mae'"):
        TokenHPEEstimatorModel(**cfg)


def test_hpe_token_rejects_raw_point_mlp_token_only_mode():
    cfg = _cfg(freeze_mmwave_tokenizer=False)
    cfg["sensor_token_config"] = {"mode": "token_only"}
    cfg["mmwave_tokenizer_config"].update(
        {
            "method": "raw_point_mlp",
            "point_hidden_dim": 8,
            "num_point_layers": 2,
        }
    )
    cfg["token_translator_config"]["mmwave_token_dim"] = 8
    with pytest.raises(ValueError, match="only supports method='mae'"):
        TokenHPEEstimatorModel(**cfg)


def test_hpe_token_rejects_raw_point_pointnet_sensor_encoder():
    cfg = _cfg(freeze_mmwave_tokenizer=False)
    cfg["sensor_token_config"] = {"mode": "token_center_concat"}
    cfg["mmwave_tokenizer_config"].update(
        {
            "method": "raw_point_pointnet",
            "point_hidden_dim": 8,
            "num_point_layers": 2,
            "num_token_layers": 2,
        }
    )
    with pytest.raises(ValueError, match="only supports method='mae'"):
        TokenHPEEstimatorModel(**cfg)


def test_hpe_token_frozen_tokenizer_blocks_gradients():
    model = TokenHPEEstimatorModel(**_cfg(freeze_mmwave_tokenizer=True))
    out = model(_toy_points(), _toy_poses())
    loss = out["pred_pose"].pow(2).mean() + out["pred_z_e"].pow(2).mean()
    loss.backward()

    assert all(param.grad is None for param in model.mmwave_tokenizer.parameters())
    translator_grad = sum(
        float(param.grad.abs().sum().item())
        for param in model.token_translator.parameters()
        if param.grad is not None
    )
    assert translator_grad > 0.0


def test_hpe_token_finetune_tokenizer_receives_gradients():
    model = TokenHPEEstimatorModel(**_cfg(freeze_mmwave_tokenizer=False))
    out = model(_toy_points(), _toy_poses())
    loss = out["pred_pose"].pow(2).mean() + out["pred_z_e"].pow(2).mean()
    loss.backward()

    tokenizer_grad = sum(
        float(param.grad.abs().sum().item())
        for param in model.mmwave_tokenizer.parameters()
        if param.grad is not None
    )
    assert tokenizer_grad > 0.0


def test_hpe_token_can_freeze_token_translator():
    cfg = _cfg(freeze_mmwave_tokenizer=True)
    cfg["freeze_token_translator"] = True
    model = TokenHPEEstimatorModel(**cfg)

    assert all(not param.requires_grad for param in model.token_translator.parameters())
    assert model.train().token_translator.training is False


def test_masked_point_root_stats_shape_and_padding_mask():
    stats = masked_point_root_stats(_toy_points(batch_size=1))

    assert stats.shape == (1, ROOT_POINT_STATS_DIM)
    torch.testing.assert_close(stats[0, :3], torch.tensor([1.21, 0.0, 0.0]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        stats[0, 16:19],
        torch.tensor([1.3366667, 0.0, 0.0]),
        atol=1e-5,
        rtol=1e-5,
    )


def test_absolute_root_head_predicts_and_composes_root():
    cfg = _cfg(freeze_mmwave_tokenizer=True)
    cfg["root_head_config"] = {"hidden_dim": 16, "num_layers": 2, "dropout": 0.0}
    model = AbsoluteRootHeadEstimatorModel(**cfg)
    out = model(_toy_points(), _toy_poses())

    assert out["pred_root"].shape == (2, 3)
    assert out["pred_pose_relative"].shape == (2, 4, 3)
    assert out["pred_pose_absolute"].shape == (2, 4, 3)
    torch.testing.assert_close(out["pred_pose"], out["pred_pose_absolute"])
    torch.testing.assert_close(out["pred_pose_absolute"], out["pred_pose_relative"] + out["pred_root"].unsqueeze(1))


def test_absolute_root_head_point_mean_prior_predicts_residual_from_point_mean():
    cfg = _cfg(freeze_mmwave_tokenizer=True)
    cfg["root_head_config"] = {
        "hidden_dim": 16,
        "num_layers": 2,
        "dropout": 0.0,
        "prior_mode": "point_mean",
        "residual_scale": 0.0,
    }
    model = AbsoluteRootHeadEstimatorModel(**cfg)
    points = _toy_points()
    out = model(points, _toy_poses())
    expected_prior = masked_point_root_stats(points)[:, :3]

    torch.testing.assert_close(out["pred_root"], expected_prior)
    torch.testing.assert_close(out["pred_root_prior"], expected_prior)
    torch.testing.assert_close(out["pred_pose_absolute"], out["pred_pose_relative"] + expected_prior.unsqueeze(1))


def test_absolute_root_head_intensity_weighted_prior_predicts_residual_from_weighted_mean():
    cfg = _cfg(freeze_mmwave_tokenizer=True)
    cfg["root_head_config"] = {
        "hidden_dim": 16,
        "num_layers": 2,
        "dropout": 0.0,
        "prior_mode": "point_intensity_weighted_mean",
        "point_stats_dim": ROOT_POINT_STATS_DIM,
        "residual_scale": 0.0,
    }
    model = AbsoluteRootHeadEstimatorModel(**cfg)
    points = _toy_points()
    out = model(points, _toy_poses())
    expected_prior = masked_point_root_stats(points)[:, 16:19]

    torch.testing.assert_close(out["pred_root"], expected_prior)
    torch.testing.assert_close(out["pred_root_prior"], expected_prior)


def test_absolute_root_head_attn_pool_accepts_legacy_point_stats_dim():
    cfg = _cfg(freeze_mmwave_tokenizer=True)
    cfg["root_head_config"] = {
        "architecture": "attn_pool",
        "hidden_dim": 16,
        "attention_hidden_dim": 12,
        "num_layers": 2,
        "dropout": 0.0,
        "prior_mode": "point_mean",
        "point_stats_dim": 16,
    }
    model = BaselineAbsoluteRootHeadEstimatorModel(**cfg)
    out = model(_toy_points(), _toy_poses(), point_root_stats=torch.randn(2, 16))

    assert out["pred_root"].shape == (2, 3)
    assert out["pred_root_delta"].shape == (2, 3)


def test_absolute_token_hpe_is_semantic_absolute_model():
    cfg = _cfg(freeze_mmwave_tokenizer=True)
    cfg["root_head_config"] = {"hidden_dim": 16, "num_layers": 1, "dropout": 0.0, "prior_mode": "point_mean"}
    model = AbsoluteRootHeadEstimatorModel(**cfg)
    out = model(_toy_points(), _toy_poses())

    assert out["pred_root"].shape == (2, 3)
    assert out["pred_pose_absolute"].shape == (2, 4, 3)


def test_hpe_token_optimizer_can_train_only_absolute_root_head():
    cfg = _cfg(freeze_mmwave_tokenizer=True)
    cfg["freeze_token_translator"] = True
    cfg["root_head_config"] = {"hidden_dim": 16, "num_layers": 2, "dropout": 0.0, "prior_mode": "point_mean"}
    model = AbsoluteRootHeadEstimatorModel(**cfg)
    opt = build_hpe_token_optimizer(
        model,
        {
            "training": {"lr": 1e-3, "weight_decay": 0.0},
        },
    )

    assert [group["name"] for group in opt.param_groups] == ["absolute_root_head"]
