# -*- coding: utf-8 -*-
"""Reusable staged training pipelines."""

__all__ = [
    "run_pt_pose",
    "run_mmwave_tokenizer_training",
    "export_mmwave_tokens",
    "run_hpe_token",
    "run_absolute_root_head",
    "run_human_pose_tokenizer",
    "run_pose_baseline",
]


def __getattr__(name: str):
    if name == "run_pt_pose":
        from .pt_pose import run_pt_pose

        return run_pt_pose
    if name == "run_mmwave_tokenizer_training":
        from .mmwave_tokenizer import run_mmwave_tokenizer_training

        return run_mmwave_tokenizer_training
    if name == "export_mmwave_tokens":
        from .mmwave_tokenizer import export_mmwave_tokens

        return export_mmwave_tokens
    if name == "run_hpe_token":
        from .hpe_token import run_hpe_token

        return run_hpe_token
    if name == "run_absolute_root_head":
        from .absolute_root_head import run_absolute_root_head

        return run_absolute_root_head
    if name == "run_human_pose_tokenizer":
        from .human_pose_tokenizer import run_human_pose_tokenizer

        return run_human_pose_tokenizer
    if name == "run_pose_baseline":
        from .pose_baseline import run_pose_baseline

        return run_pose_baseline
    raise AttributeError(name)
