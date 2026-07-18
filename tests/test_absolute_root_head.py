# -*- coding: utf-8 -*-

from pathlib import Path

import pytest
import torch

from wave2body.models.hpe_token.absolute_root import AbsoluteRootHeadEstimatorModel
from wave2body.models.hpe_token.baselines.absolute_root import AbsoluteRootHead, masked_patch_point_stats
from wave2body.train.pipelines.absolute_root_head import (
    evaluate_absolute_root_head,
    _load_absolute_component_states,
    _root_only_loss_from_batch,
    _use_root_only_loss,
)
from wave2body.utils.runtime_utils import RuntimeConfig


def _tiny_cfg() -> dict:
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
        "root_head_config": {"hidden_dim": 16, "num_layers": 2, "dropout": 0.0},
        "freeze_mmwave_tokenizer": True,
        "freeze_vqvae": True,
        "device": "cpu",
    }


def _filled_state(module: torch.nn.Module, value: float) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for key, tensor in module.state_dict().items():
        if torch.is_floating_point(tensor):
            state[key] = torch.full_like(tensor, float(value))
        else:
            state[key] = torch.zeros_like(tensor)
    return state


def _first_floating_key(module: torch.nn.Module) -> str:
    for key, tensor in module.state_dict().items():
        if torch.is_floating_point(tensor) and tensor.numel() > 0:
            return key
    raise AssertionError("module has no floating state")


def test_hpe_token_warmstart_does_not_get_overwritten_by_standalone_tokenizer(tmp_path: Path):
    model = AbsoluteRootHeadEstimatorModel(**_tiny_cfg())
    hpe_token_tokenizer_state = _filled_state(model.mmwave_tokenizer, 1.0)
    standalone_tokenizer_state = _filled_state(model.mmwave_tokenizer, 2.0)
    hpe_token_path = tmp_path / "hpe_token.pth"
    standalone_path = tmp_path / "mmwave_tokenizer.pth"
    torch.save({"mmwave_tokenizer_state_dict": hpe_token_tokenizer_state}, hpe_token_path)
    torch.save({"mmwave_tokenizer_state_dict": standalone_tokenizer_state}, standalone_path)

    model.load_pretrained_weights(
        hpe_token_path=str(hpe_token_path),
        mmwave_tokenizer_path=str(standalone_path),
        vqvae_path=None,
        device="cpu",
    )

    key = _first_floating_key(model.mmwave_tokenizer)
    torch.testing.assert_close(model.mmwave_tokenizer.state_dict()[key], hpe_token_tokenizer_state[key])


def test_best_reload_loads_compiled_token_translator_state():
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile unavailable")
    model = AbsoluteRootHeadEstimatorModel(**_tiny_cfg())
    target_state = _filled_state(model.token_translator, 0.25)
    model.token_translator = torch.compile(model.token_translator, mode="reduce-overhead", fullgraph=False)

    _load_absolute_component_states(
        model,
        {"token_translator_state_dict": target_state},
        prefix="[test]",
    )

    loaded_module = getattr(model.token_translator, "_orig_mod", model.token_translator)
    key = _first_floating_key(loaded_module)
    torch.testing.assert_close(loaded_module.state_dict()[key], target_state[key])


def test_root_only_loss_skips_frozen_token_targets():
    target_absolute = torch.zeros(2, 4, 3)
    outputs = {
        "pred_pose_absolute": torch.ones(2, 4, 3),
        "pred_root": torch.ones(2, 3),
    }
    training_cfg = {
        "freeze_token_translator": True,
        "absolute_pose_loss_weight": 0.25,
        "root_loss_weight": 1.0,
    }

    assert _use_root_only_loss(training_cfg) is True
    loss = _root_only_loss_from_batch(
        outputs=outputs,
        target_absolute=target_absolute,
        root_index=0,
        training_cfg=training_cfg,
        device=torch.device("cpu"),
    )

    assert set(loss) == {
        "total_loss",
        "relative_total_loss",
        "latent_loss",
        "token_loss",
        "pose_loss",
        "absolute_pose_loss",
        "root_loss",
    }
    torch.testing.assert_close(loss["relative_total_loss"], torch.tensor(0.0))
    torch.testing.assert_close(loss["total_loss"], torch.tensor(1.25))


def test_absolute_root_head_eval_reports_per_joint_action_groups():
    class DummyAbsoluteModel(torch.nn.Module):
        root_index = 0

        def forward(self, mmwave_points, target_relative):
            batch_size = int(mmwave_points.shape[0])
            pred_pose = torch.tensor(
                [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]],
                device=mmwave_points.device,
            ).expand(batch_size, -1, -1)
            pred_root = pred_pose[:, 0]
            return {
                "pred_pose_absolute": pred_pose,
                "pred_pose_relative": pred_pose - pred_root[:, None],
                "pred_root": pred_root,
            }

    runtime = RuntimeConfig(
        device=torch.device("cpu"),
        precision="fp32",
        amp_enabled=False,
        autocast_dtype=None,
        use_grad_scaler=False,
        deterministic=False,
        tf32=False,
        matmul_precision="high",
        compile_enabled=False,
        compile_mode=None,
        compile_dynamic=False,
        compile_fullgraph=False,
    )
    batch = {
        "mmwave_points": torch.zeros(2, 4, 5),
        "target_poses_absolute": torch.zeros(2, 4, 3),
        "indicator": torch.tensor([[1, 30, 0], [1, 31, 0]], dtype=torch.long),
    }

    metrics = evaluate_absolute_root_head(
        model=DummyAbsoluteModel(),
        dataloader=[batch],
        criterion=None,
        device=torch.device("cpu"),
        epoch=0,
        writer=None,
        config={
            "training": {"freeze_token_translator": True},
            "evaluation": {"m4human_action_groups": True, "compute_p_mpjpe": False},
        },
        runtime=runtime,
        split_name="val",
    )

    assert metrics["mpjpe_by_joint_mm"] == [1000.0, 2000.0, 3000.0, 4000.0]
    assert metrics["absolute_mpjpe_by_joint_mm"] == [1000.0, 2000.0, 3000.0, 4000.0]
    assert metrics["root_aligned_mpjpe_by_joint_mm"] == [0.0, 1000.0, 2000.0, 3000.0]
    assert metrics["mpjpe_by_action_group_by_joint_mm"]["1_50"] == [1000.0, 2000.0, 3000.0, 4000.0]
    assert metrics["root_aligned_mpjpe_by_action_group_by_joint_mm"]["1_50"] == [0.0, 1000.0, 2000.0, 3000.0]


def test_pointnext_geo_root_head_uses_point_mean_prior_and_local_stats():
    torch.manual_seed(7)
    head = AbsoluteRootHead(
        sensor_token_dim=8,
        point_stats_dim=19,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        use_centers=True,
        prior_mode="point_mean",
        architecture="pointnext_geo",
        pointnext_k=2,
        local_point_stats_dim=19,
        fourier_bands=2,
    )
    tokens = torch.randn(2, 4, 8)
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    centers = torch.randn(2, 4, 3)
    point_stats = torch.randn(2, 19)
    point_stats[:, :3] = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
    patches = torch.randn(2, 4, 3, 4)
    point_mask = torch.tensor(
        [
            [[True, True, False], [True, False, False], [True, True, True], [False, False, False]],
            [[True, True, True], [True, False, False], [False, False, False], [False, False, False]],
        ]
    )
    local_stats = masked_patch_point_stats(patches, point_mask)

    pred_root, prior, delta = head.forward_with_components(tokens, mask, centers, point_stats, local_stats)

    assert pred_root.shape == (2, 3)
    assert prior.shape == (2, 3)
    assert delta.shape == (2, 3)
    torch.testing.assert_close(prior, point_stats[:, :3])
    assert torch.isfinite(pred_root).all()
