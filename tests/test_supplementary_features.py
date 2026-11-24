# -*- coding: utf-8 -*-
"""Unit tests for merged supplementary ablations, deployment optimizations, and diagnostics."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from scripts.supplementary.generate_low_label_subsets import build_stratified_subset
from wave2body.evaluate.diagnostics import (
    apply_temporal_filter,
    compute_anatomical_pose_metrics,
    compute_same_prev_mask,
    compute_temporal_smoothness_metrics,
)
from wave2body.models.hpe_token.absolute_root import AbsoluteRootHead, AbsoluteRootHeadEstimatorModel
from wave2body.models.hpe_token.deploy import (
    BatchedDeployablePartVQVAE,
    prepare_deployable_model,
    prune_body_encoder,
)
from wave2body.models.human_body_tokenizer.model_part_vqvae import PartJointTokenizer, PartLinearDecoder
from wave2body.models.human_body_tokenizer.model_vqvae import JointTokenizer
from wave2body.models.mmwave_tokenizer.grouping import (
    _batched_farthest_point_indices,
    batched_farthest_point_indices,
    triton_fps_available,
)
from wave2body.models.point_transformer.pose_baseline import (
    PTDecoupledRootPoseModel,
    PoseDecoupledRootLoss,
    PoseMPJPELoss,
    compute_pt_pose_loss,
)


def _tiny_five_part_cfg() -> dict:
    return {
        "mmwave_tokenizer_config": {
            "method": "mae",
            "input_channels": 4,
            "num_groups": 6,
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
            "num_pose_tokens": 24,
            "num_mmwave_tokens": 6,
            "hidden_dim": 16,
            "num_cross_attn_layers": 1,
            "num_self_attn_layers": 1,
            "heads": 2,
            "dim_head": 8,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "use_sdpa": False,
            "codebook_size": 40,
        },
        "vqvae_config": {
            "model_type": "part",
            "num_joints": 22,
            "input_dim": 3,
            "token_dim": 8,
            "part_hidden_dim": 16,
            "part_dropout": 0.0,
            "root_index": 0,
            "part_token_num": {
                "L_Leg": 5,
                "R_Leg": 5,
                "SpineCore": 4,
                "L_Arm": 5,
                "R_Arm": 5,
            },
            "part_codebook_size": 8,
            "drop_rate": 0.0,
            "init_from_data": False,
        },
        "root_head_config": {
            "hidden_dim": 16,
            "num_layers": 2,
            "dropout": 0.0,
            "prior_mode": "point_intensity_weighted_mean",
        },
        "freeze_mmwave_tokenizer": True,
        "freeze_vqvae": True,
        "device": "cpu",
    }


def test_root_head_mlp_no_point_stats_and_all_invalid_mask():
    torch.manual_seed(0)
    head_mlp = AbsoluteRootHead(
        sensor_token_dim=8,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        architecture="mlp",
        use_point_stats=False,
    )
    tokens = torch.randn(2, 4, 8)
    mask = torch.tensor([[True, True, False, False], [False, False, False, False]])
    centers = torch.randn(2, 4, 3)
    out_mlp = head_mlp(tokens, mask, centers, None)
    assert out_mlp.shape == (2, 3)
    assert torch.isfinite(out_mlp).all()

    head_attn = AbsoluteRootHead(
        sensor_token_dim=8,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        architecture="query_attn",
        use_point_stats=False,
    )
    out_attn = head_attn(tokens, mask, centers, None)
    assert out_attn.shape == (2, 3)
    assert torch.isfinite(out_attn).all()


def test_part_linear_decoder_and_bypass_vq_quantization():
    torch.manual_seed(1)
    cfg = _tiny_five_part_cfg()
    cfg["vqvae_config"]["decoder_type"] = "linear"
    model = AbsoluteRootHeadEstimatorModel(**cfg, bypass_vq_quantization=True).eval()
    assert all(isinstance(dec, PartLinearDecoder) for dec in model.vqvae.decoders.values())

    pts = torch.randn(2, 16, 4)
    out = model(pts)
    assert out["pred_pose_absolute"].shape == (2, 22, 3)
    torch.testing.assert_close(out["pred_z_q"], out["pred_z_e"])


def test_holistic_joint_tokenizer_both_ema_and_ema_reset():
    torch.manual_seed(2)
    joints = torch.randn(3, 22, 3)

    # Legacy EMA mode (must not collide with register_buffer('codebook', ...))
    legacy = JointTokenizer(
        num_joints=22,
        enc_num_blocks=1,
        enc_hidden_dim=16,
        enc_token_inter_dim=16,
        enc_hidden_inter_dim=32,
        token_num=8,
        token_class_num=16,
        token_dim=12,
        dec_num_blocks=1,
        dec_hidden_dim=16,
        dec_token_inter_dim=16,
        dec_hidden_inter_dim=32,
        quantizer_type="ema",
        init_from_data=False,
    )
    assert "codebook" in legacy.state_dict()
    rec_l, idx_l, loss_l = legacy(joints, train=True)
    assert rec_l.shape == (3, 22, 3)
    assert idx_l.shape == (3, 8)
    assert torch.isfinite(loss_l)
    zq_l, qidx_l = legacy.quantize_tokens(torch.randn(3, 8, 12))
    assert zq_l.shape == (3, 8, 12) and qidx_l.shape == (3, 8)

    # EMA reset mode (for holistic 24-slot VQ-VAE ablation)
    reset_tok = JointTokenizer(
        num_joints=22,
        enc_num_blocks=1,
        enc_hidden_dim=16,
        enc_token_inter_dim=16,
        enc_hidden_inter_dim=32,
        token_num=8,
        token_class_num=16,
        token_dim=12,
        dec_num_blocks=1,
        dec_hidden_dim=16,
        dec_token_inter_dim=16,
        dec_hidden_inter_dim=32,
        quantizer_type="ema_reset",
        init_from_data=True,
    )
    assert "quantizer.codebook" in reset_tok.state_dict()
    assert reset_tok.codebook.shape == (16, 12)
    rec_r, idx_r, loss_r = reset_tok(joints, train=True)
    assert rec_r.shape == (3, 22, 3)
    assert idx_r.shape == (3, 8)
    assert torch.isfinite(loss_r)
    assert reset_tok.get_quantizer_metrics()


@pytest.mark.parametrize("fuse_mlp", [True, False])
def test_deployable_wrapper_matches_reference_numerically(fuse_mlp: bool):
    torch.manual_seed(3)
    cfg = _tiny_five_part_cfg()
    model_ref = AbsoluteRootHeadEstimatorModel(**cfg).eval()
    model_opt = copy.deepcopy(model_ref).eval()

    full_params = sum(p.numel() for p in model_ref.parameters())
    assert prune_body_encoder(model_ref) is True
    pruned_params = sum(p.numel() for p in model_ref.parameters())
    assert pruned_params < full_params

    wrapper = prepare_deployable_model(
        model_opt,
        prune_encoder=True,
        batch_part_vq=True,
        fuse_mlp_decoder=fuse_mlp,
        use_triton_fps=False,
    )
    assert isinstance(model_opt.vqvae, BatchedDeployablePartVQVAE)
    assert sum(p.numel() for p in model_opt.parameters()) == pruned_params

    pts = torch.randn(4, 24, 4)
    pts[3, 12:] = 0.0  # padded sample
    with torch.no_grad():
        ref_out = model_ref(pts)
        opt_pose = wrapper(pts)
        _, opt_idx = model_opt.quantize_tokens(ref_out["pred_z_e"])

    assert torch.equal(ref_out["pred_indices"], opt_idx)
    torch.testing.assert_close(opt_pose, ref_out["pred_pose_absolute"], atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not triton_fps_available(), reason="Triton CUDA FPS backend unavailable")
def test_triton_valid_fps_matches_torch_reference_on_cuda():
    torch.manual_seed(4)
    device = torch.device("cuda")
    xyz = torch.randn(4, 128, 3, device=device)
    valid = torch.ones(4, 128, device=device, dtype=torch.bool)
    valid[1, 40:] = False
    valid[2, :] = False  # all-invalid edge case

    ref_idx, ref_mask = _batched_farthest_point_indices(xyz, valid, 32)
    tri_idx, tri_mask = batched_farthest_point_indices(xyz, valid, 32, fps_backend="triton")
    assert torch.equal(ref_idx, tri_idx)
    assert torch.equal(ref_mask, tri_mask)


def test_pt_decoupled_root_model_and_loss_dispatch():
    torch.manual_seed(5)
    pt_cfg = {
        "data_format": {"radar_input_c": 4, "num_joints": 4},
        "model": {"scale": "zepto", "nneighbor": 2, "dropout": 0.0},
    }
    model = PTDecoupledRootPoseModel(pt_cfg, root_head_config={"hidden_dim": 16, "num_layers": 2}, device="cpu")
    pts = torch.randn(2, 8, 4)
    tgt = torch.randn(2, 4, 3)
    out = model(pts)
    assert out["pred_pose"].shape == (2, 4, 3)
    assert out["pred_root"].shape == (2, 3)

    dec_loss = compute_pt_pose_loss(PoseDecoupledRootLoss(), out, tgt)
    assert {"total_loss", "pose_loss", "rel_loss", "root_loss"} <= set(dec_loss)

    direct_loss = compute_pt_pose_loss(PoseMPJPELoss(), out, tgt)
    assert {"total_loss", "pose_loss"} <= set(direct_loss)


def test_anatomical_and_causal_temporal_diagnostics():
    torch.manual_seed(6)
    gt = torch.randn(6, 22, 3)
    # Uniformly contract predictions by 10% around pelvis -> negative distal bias
    pelvis = gt[:, :1, :]
    pred = pelvis + 0.9 * (gt - pelvis)
    vecs = compute_anatomical_pose_metrics(pred, gt, root_index=0)
    assert vecs["distal_bias_mm"].mean() < 0.0
    assert pytest.approx(vecs["distal_bias_rel_pct"].mean(), abs=1e-3) == -10.0

    # Two sequences: frames [0,1,2] of (subj=1,act=1) and frames [0,1,2] of (subj=1,act=2)
    seq_meta = {
        "kind": "m4human",
        "subject_ids": np.array([1, 1, 1, 1, 1, 1]),
        "action_ids": np.array([1, 1, 1, 2, 2, 2]),
        "frame_ids": np.array([0, 1, 2, 0, 1, 2]),
    }
    same_prev = compute_same_prev_mask(seq_meta)
    assert same_prev.tolist() == [False, True, True, False, True, True]

    # Causal EMA resets at index 3 (sequence boundary) and never reads future frames
    seq_pred = torch.arange(6, dtype=torch.float32).view(6, 1, 1).expand(6, 22, 3).clone()
    ema = apply_temporal_filter(seq_pred, seq_meta, "causal_ema_0.7")
    torch.testing.assert_close(ema[0], seq_pred[0])
    torch.testing.assert_close(ema[1], 0.7 * seq_pred[1] + 0.3 * seq_pred[0])
    torch.testing.assert_close(ema[3], seq_pred[3])  # boundary reset!

    # Causal median-3 only filters interior frame t>=2 within same sequence
    med = apply_temporal_filter(seq_pred, seq_meta, "causal_median3")
    torch.testing.assert_close(med[0], seq_pred[0])
    torch.testing.assert_close(med[1], seq_pred[1])
    torch.testing.assert_close(med[2], seq_pred[1])  # median({0, 1, 2}) == 1
    torch.testing.assert_close(med[3], seq_pred[3])  # boundary reset!
    torch.testing.assert_close(med[5], seq_pred[4])  # median({3, 4, 5}) == 4

    temp = compute_temporal_smoothness_metrics(seq_pred, seq_pred, seq_meta)
    assert temp["n_vel_pairs"] == 4
    assert temp["n_acc_triples"] == 2
    assert temp["mpjve_abs_mm_per_frame"] == 0.0
    assert temp["accel_err_abs_mm_per_frame2"] == 0.0


def test_stratified_low_label_subset_builder():
    rows = []
    for subj in (1, 2):
        for act in (1, 2, 3):
            for frame in range(20):
                rows.append((subj, act, frame))
    data = np.asarray(rows, dtype=np.int64)
    chosen_a, stats_a = build_stratified_subset(data, fraction=0.25, seed=42)
    chosen_b, _ = build_stratified_subset(data, fraction=0.25, seed=42)
    assert chosen_a == chosen_b
    assert len(chosen_a) == 30
    assert len(set(chosen_a)) == 30
    assert stats_a["subjects_covered"] == 2
    assert stats_a["actions_covered"] == 3
