# -*- coding: utf-8 -*-
"""Shared utilities for the current HPE-token model path."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..token_translation import CrossAttentionTokenTranslator


def infer_num_parts(vqvae_config: dict) -> int:
    if isinstance(vqvae_config.get("part_token_layout_order"), list):
        return len(vqvae_config["part_token_layout_order"])
    if isinstance(vqvae_config.get("part_token_num"), dict):
        return len(vqvae_config["part_token_num"])
    if isinstance(vqvae_config.get("part_codebook_size"), dict):
        return len(vqvae_config["part_codebook_size"])
    return 5


def sum_part_value(value, num_parts: int) -> int:
    if isinstance(value, int):
        return int(value) * int(num_parts)
    if isinstance(value, dict):
        return int(sum(int(v) for v in value.values()))
    raise TypeError(f"Unsupported part value type: {type(value)}")


def infer_vqvae_sizes(vqvae_config: dict) -> tuple[int, int, int]:
    is_part = (
        str(vqvae_config.get("model_type", "joint")) in ("part", "part_vqvae")
        or "part_token_num" in vqvae_config
        or "part_codebook_size" in vqvae_config
    )
    if is_part:
        num_parts = infer_num_parts(vqvae_config)
        token_num = sum_part_value(vqvae_config["part_token_num"], num_parts)
        codebook_size = sum_part_value(vqvae_config["part_codebook_size"], num_parts)
    else:
        token_num = int(vqvae_config["token_num"])
        codebook_size = int(vqvae_config["token_class_num"])
    token_dim = int(vqvae_config["token_dim"])
    return token_num, token_dim, codebook_size


def build_token_translator(translator_cfg: dict) -> nn.Module:
    cfg = dict(translator_cfg)
    if cfg.get("candidate_reranker") is not None:
        raise ValueError("token_translator.candidate_reranker is a baseline-only HPE-token option.")
    architecture = str(cfg.pop("architecture", cfg.pop("type", "cross_attention"))).strip().lower()
    if architecture not in {"cross_attention", "cross_attention_token_translator", "default"}:
        raise ValueError(
            f"Unsupported production HPE-token translator architecture: {architecture!r}. "
            "Historical translator variants live under wave2body.models.hpe_token.baselines."
        )
    return CrossAttentionTokenTranslator(**cfg)


def load_state_dict_shape_safe(module: nn.Module, state_dict: dict, module_name: str) -> None:
    model_state = module.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        try:
            dst_shape = tuple(model_state[key].shape)
        except RuntimeError:
            dst_shape = None
        if dst_shape is not None and dst_shape != tuple(value.shape):
            skipped.append((key, tuple(value.shape), dst_shape))
            continue
        filtered[key] = value

    missing_keys, unexpected_keys = module.load_state_dict(filtered, strict=False)
    print(
        f"[{module_name}] loaded={len(filtered)} | "
        f"missing={len(missing_keys)} | unexpected={len(unexpected_keys)} | "
        f"shape_skipped={len(skipped)}"
    )
    for key, src_shape, dst_shape in skipped[:8]:
        print(f"[{module_name}] skip shape mismatch: {key} ckpt{src_shape} -> model{dst_shape}")
    if len(skipped) > 8:
        print(f"[{module_name}] ... and {len(skipped) - 8} more mismatched keys")
