# -*- coding: utf-8 -*-

import torch
import pytest

from wave2body.models.mmwave_tokenizer import PointMaeTokenizer, create_mmwave_tokenizer, group_point_cloud
from wave2body.models.mmwave_tokenizer.baselines import (
    MaeDistillPartTokenizer,
    PartQueryMaeTokenizer,
    PointBertDvaeTokenizer,
    PosBertStyleTokenizer,
    create_mmwave_baseline_tokenizer,
)
from wave2body.models.mmwave_tokenizer.modules import make_random_mask


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


def _toy_joints(batch_size: int = 2, num_joints: int = 5) -> torch.Tensor:
    joints = torch.stack(
        [
            torch.linspace(0.1, 0.5, steps=num_joints),
            torch.zeros(num_joints),
            torch.zeros(num_joints),
        ],
        dim=1,
    )
    return joints.unsqueeze(0).repeat(batch_size, 1, 1).float()


def _small_cfg(**overrides):
    cfg = {
        "input_channels": 4,
        "num_groups": 3,
        "group_size": 4,
        "latent_dim": 16,
        "hidden_dim": 16,
        "codebook_size": 8,
        "transformer_depth": 1,
        "num_heads": 2,
        "dropout": 0.0,
        "patch_policy": "spatial_fps_knn",
    }
    cfg.update(overrides)
    return cfg


def test_spatial_grouping_ignores_zero_padding_indices():
    grouped = group_point_cloud(_toy_points(1), num_groups=3, group_size=3, policy="spatial_fps_knn")

    assert grouped.patches.shape == (1, 3, 3, 4)
    assert grouped.group_mask.tolist() == [[True, True, True]]
    assert set(grouped.indices.reshape(-1).tolist()).issubset({1, 2, 3, 4})


def test_body_part_grouping_uses_absolute_joint_centers():
    joints = _toy_joints(1, 5)
    grouped = group_point_cloud(
        _toy_points(1),
        num_groups=3,
        group_size=2,
        policy="weak_body_part_knn",
        target_poses_absolute=joints,
    )

    assert torch.allclose(grouped.centers[0], joints[0, :3])
    assert grouped.group_mask.tolist() == [[True, True, True]]


def test_dvae_tokenizer_forward_backward_contract():
    model = PointBertDvaeTokenizer(**_small_cfg())
    out = model(_toy_points())

    assert out["token_ids"].shape == (2, 3)
    assert out["token_embeddings"].shape == (2, 3, 16)
    assert out["centers"].shape == (2, 3, 3)
    assert torch.isfinite(out["losses"]["total_loss"])
    out["losses"]["total_loss"].backward()


def test_mae_tokenizer_spatial_forward_contract():
    cfg = _small_cfg(patch_policy="spatial_fps_knn", mask_ratio=0.5)
    cfg.pop("codebook_size")
    model = PointMaeTokenizer(**cfg)
    out = model(_toy_points())

    assert out["token_ids"] is None
    assert out["token_embeddings"].shape == (2, 3, 16)
    assert out["masked_positions"].shape == (2, 3)
    assert torch.isfinite(out["losses"]["total_loss"])


def test_random_mask_preserves_per_sample_valid_counts():
    token_mask = torch.tensor(
        [
            [True, True, True, True, False],
            [True, True, False, False, False],
            [False, False, False, False, False],
        ]
    )

    mask = make_random_mask(token_mask, 0.5)

    assert mask.shape == token_mask.shape
    assert bool((mask & ~token_mask).any()) is False
    assert mask.sum(dim=1).tolist() == [2, 1, 0]


def test_mae_encode_tokens_is_deterministic():
    cfg = _small_cfg(patch_policy="weak_body_part_knn", mask_ratio=0.5)
    cfg.pop("codebook_size")
    model = PointMaeTokenizer(**cfg)
    model.train()

    out_a = model.encode_tokens(_toy_points(), target_poses_absolute=_toy_joints())
    out_b = model.encode_tokens(_toy_points(), target_poses_absolute=_toy_joints())

    torch.testing.assert_close(out_a["token_embeddings"], out_b["token_embeddings"])
    assert out_a["token_embeddings"].shape == (2, 3, 16)


def test_mae_distill_part_tokenizer_forward_backward_and_no_pose_inference(tmp_path):
    teacher_cfg = _small_cfg(patch_policy="weak_body_part_knn", mask_ratio=0.5)
    teacher_cfg.pop("codebook_size")
    teacher = PointMaeTokenizer(**teacher_cfg)
    teacher_ckpt = tmp_path / "teacher.pt"
    torch.save(
        {
            "config": {"mmwave_tokenizer": {"method": "mae", **teacher_cfg}},
            "model_state_dict": teacher.state_dict(),
        },
        teacher_ckpt,
    )
    student_cfg = _small_cfg(
        teacher_checkpoint=str(teacher_ckpt),
        patch_policy="spatial_fps_knn",
        query_depth=1,
    )
    student_cfg.pop("codebook_size")
    model = MaeDistillPartTokenizer(**student_cfg)

    out = model(_toy_points(), target_poses_absolute=_toy_joints())
    assert out["token_embeddings"].shape == (2, 3, 16)
    assert out["centers"].shape == (2, 3, 3)
    assert torch.isfinite(out["losses"]["total_loss"])
    assert all(not param.requires_grad for param in model.teacher.parameters())
    out["losses"]["total_loss"].backward()
    assert all(param.grad is None for param in model.teacher.parameters())

    no_pose = model(_toy_points(), target_poses_absolute=None)
    assert no_pose["token_embeddings"].shape == (2, 3, 16)
    assert no_pose["centers"].shape == (2, 3, 3)
    assert torch.isfinite(no_pose["losses"]["total_loss"])


def test_part_query_mae_forward_backward_and_no_pose_inference():
    cfg = _small_cfg(query_depth=1)
    cfg.pop("codebook_size")
    model = PartQueryMaeTokenizer(**cfg)

    out = model(_toy_points(), target_poses_absolute=_toy_joints())
    assert out["token_ids"] is None
    assert out["token_embeddings"].shape == (2, 3, 16)
    assert out["centers"].shape == (2, 3, 3)
    assert torch.isfinite(out["losses"]["total_loss"])
    out["losses"]["total_loss"].backward()

    no_pose = model(_toy_points(), target_poses_absolute=None)
    assert no_pose["token_embeddings"].shape == (2, 3, 16)
    assert no_pose["valid_mask"].tolist() == [[True, True, True], [True, True, True]]


def test_posbert_tokenizer_forward_and_teacher_update_contract():
    cfg = _small_cfg(mask_ratio=0.5)
    cfg.pop("codebook_size")
    model = PosBertStyleTokenizer(**cfg)
    before = next(model.teacher.parameters()).detach().clone()
    out = model(_toy_points())
    out["losses"]["total_loss"].backward()
    model.update_teacher(momentum=0.9)
    after = next(model.teacher.parameters()).detach()

    assert out["token_embeddings"].shape == (2, 3, 16)
    assert out["teacher_embeddings"].shape == (2, 3, 16)
    assert torch.isfinite(out["losses"]["total_loss"])
    assert before.shape == after.shape


def test_production_factory_rejects_historical_methods_and_policies():
    with pytest.raises(ValueError, match="only supports method='mae'"):
        create_mmwave_tokenizer({"method": "dvae"})
    with pytest.raises(ValueError, match="patch_policy='spatial_fps_knn'"):
        create_mmwave_tokenizer({"method": "mae", "patch_policy": "weak_body_part_knn"})


def test_baseline_factory_constructs_historical_method():
    model = create_mmwave_baseline_tokenizer({"method": "dvae", **_small_cfg()})
    assert isinstance(model, PointBertDvaeTokenizer)
