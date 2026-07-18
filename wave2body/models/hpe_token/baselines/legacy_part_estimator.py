# -*- coding: utf-8 -*-
"""Reusable radar-to-body-token estimator model."""

from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from ...point_transformer import PointTransformer
from ...token_translation import CrossAttentionTokenTranslator
from ...token_translation.baselines import (
    ContextEvidenceResidualGatedTranslator,
    EvidenceGroundedAdapterTranslator,
    MLPPoolTokenTranslator,
    PartTokenCandidateReranker,
)
from ...human_body_tokenizer import create_joint_tokenizer, load_pretrained_vqvae


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


def _build_token_translator(translator_cfg: dict, vqvae_config: dict) -> nn.Module:
    cfg = dict(translator_cfg)
    cfg.pop("candidate_reranker", None)
    architecture = str(cfg.pop("architecture", cfg.pop("type", "cross_attention"))).strip().lower()
    if architecture in {"cross_attention", "cross_attention_token_translator", "default"}:
        return CrossAttentionTokenTranslator(**cfg)
    if architecture in {"mlp_pool", "pooled_mlp", "pool_mlp"}:
        return MLPPoolTokenTranslator(**cfg)
    if architecture in {
        "context_evidence_residual_gated",
        "context_evidence_gated",
        "evidence_grounded",
    }:
        cfg.setdefault("part_names", vqvae_config.get("part_token_layout_order"))
        cfg.setdefault("part_joints", vqvae_config.get("part_joints"))
        cfg.setdefault("part_token_num", vqvae_config.get("part_token_num"))
        cfg.setdefault("part_codebook_size", vqvae_config.get("part_codebook_size"))
        cfg.setdefault("root_index", vqvae_config.get("root_index", 0))
        return ContextEvidenceResidualGatedTranslator(**cfg)
    if architecture in {
        "evidence_grounded_adapter",
        "evidence_adapter",
        "from033_evidence_adapter",
    }:
        cfg.setdefault("part_names", vqvae_config.get("part_token_layout_order"))
        cfg.setdefault("part_joints", vqvae_config.get("part_joints"))
        cfg.setdefault("part_token_num", vqvae_config.get("part_token_num"))
        cfg.setdefault("part_codebook_size", vqvae_config.get("part_codebook_size"))
        cfg.setdefault("root_index", vqvae_config.get("root_index", 0))
        return EvidenceGroundedAdapterTranslator(**cfg)
    raise ValueError(f"Unsupported token translator architecture: {architecture}")


def _load_state_dict_shape_safe(module: nn.Module, state_dict: dict, module_name: str) -> None:
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


class PartTokenEstimatorModel(nn.Module):
    """
    Radar-to-body-token estimator:
    mmWave points -> Point Transformer -> Token Translator -> VQ-VAE decode.
    """

    def __init__(
        self,
        point_transformer_config: dict,
        token_translator_config: dict,
        vqvae_config: dict,
        device: str = "cuda",
        freeze_point_transformer: bool = True,
        freeze_vqvae: bool = True,
    ):
        super().__init__()
        self.device = device
        self.freeze_point_transformer = bool(freeze_point_transformer)
        self.freeze_vqvae = bool(freeze_vqvae)
        self.vqvae_config = dict(vqvae_config)

        vq_token_num, vq_token_dim, vq_codebook_size = infer_vqvae_sizes(vqvae_config)

        self.point_transformer = PointTransformer(point_transformer_config, device=device)
        if self.freeze_point_transformer:
            for param in self.point_transformer.parameters():
                param.requires_grad = False
            self.point_transformer.eval()

        translator_cfg = dict(token_translator_config)
        translator_cfg.setdefault("num_pose_tokens", vq_token_num)
        translator_cfg.setdefault("pose_token_dim", vq_token_dim)
        translator_cfg.setdefault("codebook_size", vq_codebook_size)
        if int(translator_cfg["num_pose_tokens"]) != vq_token_num:
            raise ValueError(
                f"num_pose_tokens mismatch: token_translator={translator_cfg['num_pose_tokens']} vs vqvae={vq_token_num}"
            )
        if int(translator_cfg["pose_token_dim"]) != vq_token_dim:
            raise ValueError(
                f"pose_token_dim mismatch: token_translator={translator_cfg['pose_token_dim']} vs vqvae={vq_token_dim}"
            )
        if int(translator_cfg["codebook_size"]) != vq_codebook_size:
            raise ValueError(
                f"codebook_size mismatch: token_translator={translator_cfg['codebook_size']} vs vqvae={vq_codebook_size}"
            )
        self.token_translator = _build_token_translator(translator_cfg, vqvae_config)

        self.vqvae = create_joint_tokenizer(vqvae_config)
        if self.freeze_vqvae:
            for param in self.vqvae.parameters():
                param.requires_grad = False
            self.vqvae.eval()
        reranker_cfg = dict(translator_cfg.get("candidate_reranker") or {})
        if bool(reranker_cfg.pop("enabled", False)):
            reranker_cfg.setdefault("token_dim", vq_token_dim)
            self.candidate_reranker = PartTokenCandidateReranker(**reranker_cfg)
        else:
            self.candidate_reranker = None

    def forward(
        self,
        mmwave_points: Optional[torch.Tensor] = None,
        target_poses: Optional[torch.Tensor] = None,
        *,
        mmwave_tokens: Optional[torch.Tensor] = None,
        target_z_e: Optional[torch.Tensor] = None,
        target_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if mmwave_tokens is None:
            if mmwave_points is None:
                raise ValueError("Either mmwave_points or mmwave_tokens must be provided.")
            if self.freeze_point_transformer:
                with torch.no_grad():
                    _, mmwave_tokens = self.point_transformer(mmwave_points)
            else:
                _, mmwave_tokens = self.point_transformer(mmwave_points)

        raw_translator = getattr(self.token_translator, "_orig_mod", self.token_translator)
        supports_logits = bool(getattr(raw_translator, "supports_logits", False)) or (
            getattr(raw_translator, "token_classifier", None) is not None
        )
        if supports_logits:
            pred_z_e, pred_logits = self.token_translator(mmwave_tokens, return_logits=True)
        else:
            pred_z_e = self.token_translator(mmwave_tokens)
            pred_logits = None

        pred_z_q, pred_indices = self.quantize_tokens(pred_z_e)
        pred_pose = self.vqvae.decode(pred_z_q)

        out = {
            "pred_z_e": pred_z_e,
            "pred_z_q": pred_z_q,
            "pred_logits": pred_logits,
            "pred_indices": pred_indices,
            "pred_pose": pred_pose,
            "mmwave_tokens": mmwave_tokens,
        }

        if target_z_e is not None:
            if target_indices is None:
                raise ValueError("target_indices must be provided when target_z_e is provided.")
            out["target_z_e"] = target_z_e
            out["target_indices"] = target_indices
        elif target_poses is not None:
            with torch.no_grad():
                target_inter = self.vqvae.encode_with_intermediates(target_poses)
            out["target_z_e"] = target_inter["z_e"]
            out["target_z_q"] = target_inter["z_q"]
            out["target_indices"] = target_inter["indices"]
            if "indices_local" in target_inter:
                out["target_indices_local"] = target_inter["indices_local"]

        return out

    def quantize_tokens(self, continuous_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.vqvae, "quantize_tokens"):
            return self.vqvae.quantize_tokens(continuous_tokens, apply_straight_through=True)

        batch, n_tokens, token_dim = continuous_tokens.shape
        flat = continuous_tokens.reshape(-1, token_dim)
        dist = (
            torch.sum(flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.vqvae.codebook ** 2, dim=1)
            - 2 * torch.matmul(flat, self.vqvae.codebook.t())
        )
        idx = torch.argmin(dist, dim=1).view(batch, n_tokens)
        zq = self.vqvae.codebook[idx.reshape(-1)].view(batch, n_tokens, token_dim)
        zq = continuous_tokens + (zq - continuous_tokens).detach()
        return zq, idx

    def candidate_rerank_logits(
        self,
        *,
        pred_z_e: torch.Tensor,
        pred_logits: torch.Tensor,
        candidate_topk: int,
        distance_scale: float = 1.0,
    ) -> torch.Tensor:
        if self.candidate_reranker is None:
            raise RuntimeError("candidate_reranker is not enabled for this model.")
        return self.candidate_reranker(
            vqvae=self.vqvae,
            pred_z_e=pred_z_e,
            pred_logits=pred_logits,
            candidate_topk=int(candidate_topk),
            distance_scale=float(distance_scale),
        )

    def load_pretrained_weights(
        self,
        point_transformer_path: Optional[str] = None,
        vqvae_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = self.device

        if point_transformer_path and os.path.exists(point_transformer_path):
            print(f"Loading Point Transformer from {point_transformer_path}")
            ckpt = torch.load(point_transformer_path, map_location=device, weights_only=False)
            state_dict = ckpt.get("point_transformer_state_dict", ckpt.get("model_state_dict", ckpt))
            _load_state_dict_shape_safe(self.point_transformer, state_dict, "PointTransformer")
        elif point_transformer_path:
            print(f"Warning: Point Transformer checkpoint not found: {point_transformer_path}")

        if vqvae_path and os.path.exists(vqvae_path):
            print(f"Loading VQ-VAE from {vqvae_path}")
            vqvae_loaded = load_pretrained_vqvae(vqvae_path, config=self.vqvae_config, device=device)
            self.vqvae.load_state_dict(vqvae_loaded.state_dict())
        elif vqvae_path:
            print(f"Warning: VQ-VAE checkpoint not found: {vqvae_path}")
