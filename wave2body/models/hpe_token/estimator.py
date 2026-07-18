# -*- coding: utf-8 -*-
"""Current HPE-token estimator using the K96 mmWave tokenizer as encoder."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .common import build_token_translator, infer_vqvae_sizes, load_state_dict_shape_safe
from ..mmwave_tokenizer import create_mmwave_tokenizer
from ..human_body_tokenizer import create_joint_tokenizer, load_pretrained_vqvae
from ...utils.paths import resolve_runtime_uri


def _extract_mmwave_tokenizer_state_dict(ckpt: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Extract a mmWave tokenizer state dict from 051 or composed checkpoints."""
    for key in ("mmwave_tokenizer_state_dict", "sensor_encoder_state_dict", "model_state_dict"):
        state = ckpt.get(key)
        if isinstance(state, dict) and state:
            return dict(state)

    if ckpt and all(torch.is_tensor(value) for value in ckpt.values()):
        return dict(ckpt)

    raise KeyError("Checkpoint does not contain mmWave tokenizer weights.")


def _resolve_checkpoint_file(path: Optional[str]) -> Path | None:
    if path in (None, "", "null"):
        return None
    return Path(resolve_runtime_uri(str(path))).expanduser()


class TokenHPEEstimatorModel(nn.Module):
    """
    End-to-end body-token estimator:

    mmWave points -> mmWave tokenizer sensor tokens -> token translator ->
    VQ-VAE quantize/decode -> pose coordinates.
    """

    def __init__(
        self,
        mmwave_tokenizer_config: dict,
        token_translator_config: dict,
        vqvae_config: dict,
        sensor_token_config: Optional[dict] = None,
        device: str = "cuda",
        freeze_mmwave_tokenizer: bool = True,
        freeze_token_translator: bool = False,
        freeze_vqvae: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.freeze_mmwave_tokenizer = bool(freeze_mmwave_tokenizer)
        self.freeze_token_translator = bool(freeze_token_translator)
        self.freeze_vqvae = bool(freeze_vqvae)
        self.vqvae_config = dict(vqvae_config)
        self.mmwave_tokenizer_config = dict(mmwave_tokenizer_config)
        self.sensor_token_config = dict(sensor_token_config or {})
        self.sensor_token_mode = str(self.sensor_token_config.get("mode", "token_center_concat")).strip().lower()
        self.center_scale = float(self.sensor_token_config.get("center_scale", 1.0))
        self.zero_invalid_tokens = bool(self.sensor_token_config.get("zero_invalid", True))

        self.mmwave_tokenizer = create_mmwave_tokenizer(self.mmwave_tokenizer_config)
        if self.freeze_mmwave_tokenizer:
            for param in self.mmwave_tokenizer.parameters():
                param.requires_grad = False
            self.mmwave_tokenizer.eval()

        vq_token_num, vq_token_dim, vq_codebook_size = infer_vqvae_sizes(vqvae_config)
        sensor_token_dim = self._infer_sensor_token_dim(self.mmwave_tokenizer_config, self.sensor_token_mode)
        sensor_token_num = int(self.mmwave_tokenizer_config.get("num_groups", 22))

        translator_cfg = dict(token_translator_config)
        translator_cfg.setdefault("num_mmwave_tokens", sensor_token_num)
        translator_cfg.setdefault("mmwave_token_dim", sensor_token_dim)
        translator_cfg.setdefault("num_pose_tokens", vq_token_num)
        translator_cfg.setdefault("pose_token_dim", vq_token_dim)
        translator_cfg.setdefault("codebook_size", vq_codebook_size)

        if int(translator_cfg["mmwave_token_dim"]) != sensor_token_dim:
            raise ValueError(
                "mmwave_token_dim mismatch: "
                f"token_translator={translator_cfg['mmwave_token_dim']} vs "
                f"mmwave_tokenizer sensor dim={sensor_token_dim}"
            )
        if int(translator_cfg["num_mmwave_tokens"]) != sensor_token_num:
            raise ValueError(
                "num_mmwave_tokens mismatch: "
                f"token_translator={translator_cfg['num_mmwave_tokens']} vs "
                f"mmwave_tokenizer num_groups={sensor_token_num}"
            )
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

        self.sensor_token_dim = sensor_token_dim
        self.sensor_token_num = sensor_token_num
        self.token_translator = build_token_translator(translator_cfg)
        if self.freeze_token_translator:
            for param in self.token_translator.parameters():
                param.requires_grad = False
            self.token_translator.eval()
        self.vqvae = create_joint_tokenizer(vqvae_config)
        if self.freeze_vqvae:
            for param in self.vqvae.parameters():
                param.requires_grad = False
            self.vqvae.eval()
    @staticmethod
    def _infer_sensor_token_dim(mmwave_tokenizer_config: dict, mode: str) -> int:
        latent_dim = int(mmwave_tokenizer_config.get("latent_dim", 128))
        if mode in {"token", "embedding", "token_only"}:
            return latent_dim
        if mode in {"token_center_concat", "token+center", "embedding_center_concat"}:
            return latent_dim + 3
        raise ValueError(f"Unsupported sensor token mode: {mode!r}")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_mmwave_tokenizer:
            self.mmwave_tokenizer.eval()
        if self.freeze_token_translator:
            self.token_translator.eval()
        if getattr(self, "freeze_token_translator_except_classifier", False):
            self.token_translator.eval()
            raw_translator = getattr(self.token_translator, "_orig_mod", self.token_translator)
            token_classifier = getattr(raw_translator, "token_classifier", None)
            if token_classifier is not None and mode:
                token_classifier.train()
        if self.freeze_vqvae:
            self.vqvae.eval()
        return self

    def encode_mmwave_tokens(self, mmwave_points: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode deployable sensor tokens without using GT pose."""
        if hasattr(self.mmwave_tokenizer, "encode_tokens"):
            tokenizer_out = self.mmwave_tokenizer.encode_tokens(mmwave_points, target_poses_absolute=None)
        else:
            tokenizer_out = self.mmwave_tokenizer(mmwave_points, target_poses_absolute=None)

        embeddings = tokenizer_out["token_embeddings"]
        valid_mask = tokenizer_out.get("valid_mask")
        if valid_mask is None:
            valid_mask = torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.bool)
        else:
            valid_mask = valid_mask.bool()

        if self.sensor_token_mode in {"token", "embedding", "token_only"}:
            sensor_tokens = embeddings
        elif self.sensor_token_mode in {"token_center_concat", "token+center", "embedding_center_concat"}:
            centers = tokenizer_out["centers"].to(device=embeddings.device, dtype=embeddings.dtype)
            sensor_tokens = torch.cat([embeddings, centers * self.center_scale], dim=-1)
        else:
            raise ValueError(f"Unsupported sensor token mode: {self.sensor_token_mode!r}")

        if self.zero_invalid_tokens:
            sensor_tokens = sensor_tokens.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        out = {"tokens": sensor_tokens, "mask": valid_mask, "token_embeddings": embeddings}
        if "centers" in tokenizer_out:
            out["centers"] = tokenizer_out["centers"].to(device=embeddings.device, dtype=embeddings.dtype)
        for key in ("patches", "raw_patches", "point_mask", "support_counts"):
            if key in tokenizer_out:
                value = tokenizer_out[key]
                out[key] = value.to(device=embeddings.device) if torch.is_tensor(value) else value
        return out

    def _translate(
        self,
        mmwave_tokens: torch.Tensor,
        mmwave_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        raw_translator = getattr(self.token_translator, "_orig_mod", self.token_translator)
        supports_logits = bool(getattr(raw_translator, "supports_logits", False)) or (
            getattr(raw_translator, "token_classifier", None) is not None
        )
        if supports_logits:
            return self.token_translator(mmwave_tokens, mask=mmwave_mask, return_logits=True)
        return self.token_translator(mmwave_tokens, mask=mmwave_mask), None

    def forward(
        self,
        mmwave_points: Optional[torch.Tensor] = None,
        target_poses: Optional[torch.Tensor] = None,
        *,
        mmwave_tokens: Optional[torch.Tensor] = None,
        target_z_e: Optional[torch.Tensor] = None,
        target_indices: Optional[torch.Tensor] = None,
        mmwave_token_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        mmwave_token_centers = None
        if mmwave_tokens is None:
            if mmwave_points is None:
                raise ValueError("Either mmwave_points or mmwave_tokens must be provided.")
            grad_ctx = torch.no_grad() if self.freeze_mmwave_tokenizer else nullcontext()
            with grad_ctx:
                encoded = self.encode_mmwave_tokens(mmwave_points)
            mmwave_tokens = encoded["tokens"]
            mmwave_token_mask = encoded["mask"]
            mmwave_token_centers = encoded.get("centers")

        pred_z_e, pred_logits = self._translate(mmwave_tokens, mmwave_token_mask)
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
        if mmwave_token_mask is not None:
            out["mmwave_token_mask"] = mmwave_token_mask
        if mmwave_token_centers is not None:
            out["mmwave_token_centers"] = mmwave_token_centers

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
            torch.sum(flat**2, dim=1, keepdim=True)
            + torch.sum(self.vqvae.codebook**2, dim=1)
            - 2 * torch.matmul(flat, self.vqvae.codebook.t())
        )
        idx = torch.argmin(dist, dim=1).view(batch, n_tokens)
        zq = self.vqvae.codebook[idx.reshape(-1)].view(batch, n_tokens, token_dim)
        zq = continuous_tokens + (zq - continuous_tokens).detach()
        return zq, idx

    def load_pretrained_weights(
        self,
        mmwave_tokenizer_path: Optional[str] = None,
        vqvae_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = self.device

        mmwave_tokenizer_file = _resolve_checkpoint_file(mmwave_tokenizer_path)
        vqvae_file = _resolve_checkpoint_file(vqvae_path)

        if mmwave_tokenizer_file and mmwave_tokenizer_file.exists():
            print(f"Loading mmWave tokenizer from {mmwave_tokenizer_file}")
            ckpt = torch.load(mmwave_tokenizer_file, map_location=device, weights_only=False)
            state_dict = _extract_mmwave_tokenizer_state_dict(ckpt)
            load_state_dict_shape_safe(self.mmwave_tokenizer, state_dict, "MMWaveTokenizer")
        elif mmwave_tokenizer_path:
            print(f"Warning: mmWave tokenizer checkpoint not found: {mmwave_tokenizer_path} -> {mmwave_tokenizer_file}")

        if vqvae_file and vqvae_file.exists():
            print(f"Loading VQ-VAE from {vqvae_file}")
            vqvae_loaded = load_pretrained_vqvae(str(vqvae_file), config=self.vqvae_config, device=device)
            self.vqvae.load_state_dict(vqvae_loaded.state_dict())
        elif vqvae_path:
            print(f"Warning: VQ-VAE checkpoint not found: {vqvae_path} -> {vqvae_file}")
