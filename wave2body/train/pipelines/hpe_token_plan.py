# -*- coding: utf-8 -*-
"""Run-plan helpers for HPE-token launchers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...utils.paths import artifact_root, resolve_runtime_uri


EXPERIMENT_NAME = "hpe_token"


def is_absolute_root_config(config: dict[str, Any]) -> bool:
    stage = str(config.get("training", {}).get("stage", "")).strip().lower()
    return "root_head" in config or stage.startswith("absolute")


def planned_output_root(config: dict[str, Any], work_dir: str | None) -> Path:
    if work_dir:
        return Path(work_dir).expanduser()
    run_name = str(config.get("training", {}).get("run_name", "m4human_hpe_token"))
    return artifact_root() / "experiments" / EXPERIMENT_NAME / "results" / run_name


def dry_run_payload(
    *,
    config_path: Path,
    config: dict[str, Any],
    work_dir: str | None,
    resume: str | None,
    reset_epoch: bool,
    batch_limits: dict[str, int | None],
) -> dict[str, Any]:
    output_root = planned_output_root(config, work_dir)
    absolute_root = is_absolute_root_config(config)
    pretrained = config.get("pretrained", {})
    training = config.get("training", {})
    if absolute_root:
        if bool(training.get("freeze_token_translator", False)):
            recipe = "stage2_absolute_root_frozen_translator"
        else:
            recipe = "stage2_absolute_query_root_trainable_translator"
    elif not bool(training.get("freeze_mmwave_tokenizer", True)):
        recipe = "stage1_relative_translator_finetune_mmwave_tokenizer"
    else:
        recipe = "stage1_relative_translator"
    checkpoint_paths = {
        "mmwave_tokenizer": pretrained.get("mmwave_tokenizer"),
        "body_vqvae": pretrained.get("vqvae"),
        "hpe_token": pretrained.get("hpe_token"),
        "resume": resume,
    }
    checkpoint_status = {
        name: (None if path in (None, "", "null") else Path(resolve_runtime_uri(str(path))).expanduser().is_file())
        for name, path in checkpoint_paths.items()
    }
    return {
        "config": str(config_path),
        "run_name": training["run_name"],
        "recipe": recipe,
        "dataset": config.get("dataset", {}).get("name"),
        "absolute_root": absolute_root,
        "root_prior": config.get("root_head", {}).get("prior_mode") if absolute_root else None,
        "root_head_architecture": config.get("root_head", {}).get("architecture") if absolute_root else None,
        "split_id": config.get("m4human", {}).get("split"),
        "sensor_tokens": config["mmwave_tokenizer"]["num_groups"],
        "sensor_token_mode": config.get("sensor_tokens", {}).get("mode"),
        "sensor_token_dim": config["token_translator"]["mmwave_token_dim"],
        "human_tokens": config["token_translator"]["num_pose_tokens"],
        "freeze_mmwave_tokenizer": training["freeze_mmwave_tokenizer"],
        "mmwave_tokenizer_lr_mult": training.get("mmwave_tokenizer_lr_mult"),
        "freeze_hpe_token_translator": bool(training.get("freeze_token_translator", False)),
        "freeze_hpe_token_translator_except_classifier": bool(
            training.get("freeze_token_translator_except_classifier", False)
        ),
        "freeze_body_vqvae": training["freeze_vqvae"],
        "frozen_feature_cache": training.get("frozen_feature_cache"),
        "n_points": config.get("data_format", {}).get("n_points"),
        "merge_nframes": config.get("data_format", {}).get("merge_nframes"),
        "point_sampling": config.get("data_format", {}).get("point_sampling"),
        "input_root_normalized": config.get("data_format", {}).get("input_root_normalized"),
        "input_root_crop": config.get("data_format", {}).get("input_root_crop"),
        "input_root_crop_x_range": config.get("data_format", {}).get("input_root_crop_x_range"),
        "input_root_crop_y_range": config.get("data_format", {}).get("input_root_crop_y_range"),
        "input_root_crop_z_range": config.get("data_format", {}).get("input_root_crop_z_range"),
        "target_root_normalized": config.get("data_format", {}).get("target_root_normalized"),
        "latent_loss_weight": training.get("latent_loss_weight"),
        "token_loss_weight": training.get("token_loss_weight"),
        "pose_loss_weight": training.get("pose_loss_weight"),
        "lr": training.get("lr"),
        "min_lr": training.get("min_lr"),
        "soft_token_ce": training.get("soft_token_ce"),
        "point_augmentation": training.get("point_augmentation"),
        "pose_decode": training.get("pose_decode"),
        "output_root": str(output_root),
        "checkpoint_dir": str(output_root / "checkpoints"),
        "mmwave_tokenizer_checkpoint": pretrained.get("mmwave_tokenizer"),
        "body_vqvae_checkpoint": pretrained.get("vqvae"),
        "hpe_token_checkpoint": pretrained.get("hpe_token"),
        "checkpoint_status": checkpoint_status,
        "resume": resume,
        "reset_epoch": bool(reset_epoch),
        "max_batches": batch_limits,
    }
