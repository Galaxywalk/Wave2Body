# -*- coding: utf-8 -*-

import torch

from scripts.human_pose_tokenizer.train import build_extra_args
from wave2body.dataset.human_body import AMASSJointsDataset, DATASET_REGISTRY, create_dataset
from wave2body.dataset.human_body.mmbody import MMBodyJointsDataset
from wave2body.models.human_body_tokenizer import PartJointTokenizer, create_joint_tokenizer


PART_NAMES = ["L_Leg", "R_Leg", "SpineCore", "L_Arm", "R_Arm"]
PART_JOINTS = {
    "L_Leg": [1, 4, 7, 10],
    "R_Leg": [2, 5, 8, 11],
    "SpineCore": [3, 6, 9, 12, 15],
    "L_Arm": [9, 13, 16, 18, 20],
    "R_Arm": [9, 14, 17, 19, 21],
}
PART_TOKEN_NUM = {name: 1 for name in PART_NAMES}


def _small_part_vqvae_config() -> dict:
    return {
        "model_type": "part",
        "num_joints": 22,
        "input_dim": 3,
        "token_dim": 8,
        "part_hidden_dim": 16,
        "part_dropout": 0.0,
        "quantizer_type": "ema_reset",
        "ema_decay": 0.9,
        "reset_threshold": 0.0,
        "reset_noise_std": 0.0,
        "reset_min_active_frac": 0.0,
        "init_from_data": True,
        "root_index": 0,
        "part_token_layout_order": PART_NAMES,
        "part_joints": PART_JOINTS,
        "part_token_num": PART_TOKEN_NUM,
        "part_codebook_size": 4,
        "drop_rate": 0.0,
    }


def test_human_body_dataset_registry_exports_current_body_sources():
    assert create_dataset is not None
    assert DATASET_REGISTRY["amass"] is AMASSJointsDataset
    assert sorted(DATASET_REGISTRY) == ["amass", "m4human", "mmbody"]


def test_mmbody_joint_cache_key_includes_smoke_sample_limit(tmp_path):
    ds = MMBodyJointsDataset.__new__(MMBodyJointsDataset)
    ds.cache_dir = tmp_path
    ds.name = "MMBody_train"
    ds.splits = ["train"]
    ds.normalized = True
    ds.max_samples = 8
    ds.num_joints = 22

    assert ds._get_cache_path().name == "MMBody_train_train_norm_max8_joints22.pt"


def test_human_pose_tokenizer_launcher_forwards_dataset_sample_limit():
    class Args:
        resume = None
        gpus = None
        epochs = None
        batch_size = None
        num_workers = None
        log_interval = None
        dataset_device = None
        batch_loader = None
        max_dataset_samples = 8
        max_train_batches = None
        max_val_batches = None
        resume_model_only = False
        deterministic = False

    assert build_extra_args(Args()) == ["--max-dataset-samples", "8"]


def test_part_joint_tokenizer_forward_contract():
    model = create_joint_tokenizer(_small_part_vqvae_config())
    joints = torch.randn(2, 22, 3)
    joints[:, 0] = 0.0

    recovered, indices, e_loss = model(joints=joints, train=True)
    intermediates = model.encode_with_intermediates(joints)

    assert isinstance(model, PartJointTokenizer)
    assert recovered.shape == joints.shape
    assert indices.shape == (2, 5)
    assert e_loss is not None
    assert intermediates["z_e"].shape == (2, 5, 8)
    assert intermediates["part_ids"].tolist() == [0, 1, 2, 3, 4]
