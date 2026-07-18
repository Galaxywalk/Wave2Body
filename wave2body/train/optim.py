# -*- coding: utf-8 -*-
"""Optimizer and trainability helpers for shared training pipelines."""

from __future__ import annotations

from typing import Any

import torch

try:
    from .common import safe_param_count
except ImportError:  # Support scripts that put src/ directly on sys.path.
    from wave2body.train.common import safe_param_count


def set_requires_grad(module: torch.nn.Module, flag: bool) -> None:
    for param in module.parameters():
        param.requires_grad = bool(flag)


def count_trainable(module_or_params: Any) -> int:
    return safe_param_count(module_or_params, trainable_only=True)


def build_hpe_token_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """Build optimizer groups for token-based HPE.

    The default recipe freezes the mmWave tokenizer and body VQ-VAE, so this
    usually optimizes only the human-token translator. A lower LR group is kept
    for explicit tokenizer-unfreeze experiments without changing the entrypoint.
    """
    training_cfg = config.get("training", {})
    base_lr = float(training_cfg.get("lr", 1e-4))
    weight_decay = float(training_cfg.get("weight_decay", 0.01))
    tokenizer_lr_mult = float(
        training_cfg.get("mmwave_tokenizer_lr_mult", training_cfg.get("sensor_encoder_lr_mult", 0.1))
    )

    translator_params = [p for p in model.token_translator.parameters() if p.requires_grad]
    root_head_module = getattr(model, "root_head", None)
    root_head_params = [] if root_head_module is None else [p for p in root_head_module.parameters() if p.requires_grad]
    tokenizer_module = getattr(model, "mmwave_tokenizer", None)
    tokenizer_params = [] if tokenizer_module is None else [p for p in tokenizer_module.parameters() if p.requires_grad]

    param_groups: list[dict[str, Any]] = []
    if translator_params:
        param_groups.append({"params": translator_params, "lr": base_lr, "name": "human_token_translator"})
    if root_head_params:
        param_groups.append({"params": root_head_params, "lr": base_lr, "name": "absolute_root_head"})
    if tokenizer_params:
        param_groups.append({"params": tokenizer_params, "lr": base_lr * tokenizer_lr_mult, "name": "mmwave_tokenizer"})
    if not param_groups:
        raise RuntimeError("No trainable parameters found for HPE token optimizer.")

    print(
        "[HPE Token] Optimizer param groups | "
        f"human_token_translator={safe_param_count(translator_params):,}, "
        f"absolute_root_head={safe_param_count(root_head_params):,}, "
        f"mmwave_tokenizer={safe_param_count(tokenizer_params):,}, "
        f"mmwave_tokenizer_lr_mult={tokenizer_lr_mult:.3f}"
    )
    return torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)
