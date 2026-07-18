# -*- coding: utf-8 -*-
"""Config path helpers for human pose tokenizer launchers."""

from __future__ import annotations

import os
from pathlib import Path

from wave2body.utils.paths import artifact_root


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "wave2body" / "configs" / "human_pose_tokenizer"


def _human_pose_tokenizer_workspace() -> Path:
    value = os.environ.get("WAVE2BODY_VQVAE_ROOT")
    if value not in (None, ""):
        return Path(os.path.expanduser(os.path.expandvars(str(value))))
    return artifact_root().parent / "vqvae"


HUMAN_POSE_TOKENIZER_WORKSPACE = _human_pose_tokenizer_workspace()


CONFIG_ALIASES = {
    "pretrain": CONFIG_DIR / "stage_a_part_k96_amass_mmbody_m4human.yaml",
    "adapt": CONFIG_DIR / "stage_b_part_k96_mmbody_m4human.yaml",
    "mmbody_train_pretrain": CONFIG_DIR / "stage_a_part_k96_amass_mmbody_train_joints.yaml",
    "mmbody_train_finetune": CONFIG_DIR / "stage_b_part_k96_mmbody_trainonly_from_amass_mmbody_train.yaml",
}


def resolve_config(value: str) -> Path:
    if value in CONFIG_ALIASES:
        return CONFIG_ALIASES[value]
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def print_configs() -> None:
    for name, path in CONFIG_ALIASES.items():
        print(f"{name}: {path}")


__all__ = [
    "CONFIG_ALIASES",
    "CONFIG_DIR",
    "HUMAN_POSE_TOKENIZER_WORKSPACE",
    "resolve_config",
    "print_configs",
]
