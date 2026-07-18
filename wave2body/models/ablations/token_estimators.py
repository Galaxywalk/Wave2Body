# -*- coding: utf-8 -*-
"""HPE-token estimator variants for controlled ablation studies."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..hpe_token.absolute_root import AbsoluteRootHead, masked_point_root_stats
from ..hpe_token.common import build_token_translator, infer_vqvae_sizes, load_state_dict_shape_safe
from ..hpe_token.estimator import TokenHPEEstimatorModel, _extract_mmwave_tokenizer_state_dict
from ..human_body_tokenizer import create_joint_tokenizer, load_pretrained_vqvae
from ..mmwave_tokenizer import create_mmwave_tokenizer
from ..mmwave_tokenizer.baselines import create_mmwave_baseline_tokenizer
from ...utils.paths import resolve_runtime_uri


def create_ablation_sensor_tokenizer(config: dict[str, Any]) -> nn.Module:
    """Create a sensor tokenizer for production or ablation-only methods."""

    method = str(config.get("method", "mae")).strip().lower()
    patch_policy = str(config.get("patch_policy", "spatial_fps_knn")).strip().lower()
    if method == "mae" and patch_policy == "spatial_fps_knn":
        return create_mmwave_tokenizer(config)
    return create_mmwave_baseline_tokenizer(config)


def _resolve_checkpoint_file(path: Optional[str]) -> Path | None:
    if path in (None, "", "null"):
        return None
    return Path(resolve_runtime_uri(str(path))).expanduser()


class AblationTokenHPEEstimatorModel(TokenHPEEstimatorModel):
    """Token HPE estimator that permits ablation-only sensor tokenizers."""

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
        nn.Module.__init__(self)
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

        self.mmwave_tokenizer = create_ablation_sensor_tokenizer(self.mmwave_tokenizer_config)
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
                f"token_translator={translator_cfg['mmwave_token_dim']} vs sensor dim={sensor_token_dim}"
            )
        if int(translator_cfg["num_mmwave_tokens"]) != sensor_token_num:
            raise ValueError(
                "num_mmwave_tokens mismatch: "
                f"token_translator={translator_cfg['num_mmwave_tokens']} vs num_groups={sensor_token_num}"
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


class AblationAbsoluteRootHeadEstimatorModel(AblationTokenHPEEstimatorModel):
    """Absolute-root HPE-token estimator that permits ablation sensor tokenizers."""

    def __init__(
        self,
        mmwave_tokenizer_config: dict,
        token_translator_config: dict,
        vqvae_config: dict,
        sensor_token_config: Optional[dict] = None,
        root_head_config: Optional[dict] = None,
        device: str = "cuda",
        freeze_mmwave_tokenizer: bool = True,
        freeze_token_translator: bool = False,
        freeze_vqvae: bool = True,
    ) -> None:
        super().__init__(
            mmwave_tokenizer_config=mmwave_tokenizer_config,
            token_translator_config=token_translator_config,
            vqvae_config=vqvae_config,
            sensor_token_config=sensor_token_config,
            device=device,
            freeze_mmwave_tokenizer=freeze_mmwave_tokenizer,
            freeze_token_translator=freeze_token_translator,
            freeze_vqvae=freeze_vqvae,
        )
        root_cfg = dict(root_head_config or {})
        self.root_index = int(vqvae_config.get("root_index", root_cfg.pop("root_index", 0)))
        attention_hidden_dim = root_cfg.pop("attention_hidden_dim", None)
        self.root_head = AbsoluteRootHead(
            sensor_token_dim=self.sensor_token_dim,
            hidden_dim=int(root_cfg.pop("hidden_dim", 256)),
            num_layers=int(root_cfg.pop("num_layers", 3)),
            dropout=float(root_cfg.pop("dropout", 0.1)),
            use_centers=bool(root_cfg.pop("use_centers", True)),
            point_stats_dim=int(root_cfg.pop("point_stats_dim", 19)),
            prior_mode=str(root_cfg.pop("prior_mode", "none")),
            residual_scale=float(root_cfg.pop("residual_scale", 1.0)),
            architecture=str(root_cfg.pop("architecture", "query_attn")),
            attention_hidden_dim=None if attention_hidden_dim is None else int(attention_hidden_dim),
            attention_heads=int(root_cfg.pop("attention_heads", 4)),
            num_query_tokens=int(root_cfg.pop("num_query_tokens", 4)),
        )
        if root_cfg:
            raise ValueError(f"Unused root_head config keys: {sorted(root_cfg)}")

    def forward(
        self,
        mmwave_points: Optional[torch.Tensor] = None,
        target_poses: Optional[torch.Tensor] = None,
        *,
        mmwave_tokens: Optional[torch.Tensor] = None,
        target_z_e: Optional[torch.Tensor] = None,
        target_indices: Optional[torch.Tensor] = None,
        mmwave_token_mask: Optional[torch.Tensor] = None,
        mmwave_token_centers: Optional[torch.Tensor] = None,
        point_root_stats: Optional[torch.Tensor] = None,
        mmwave_local_point_stats: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        centers = mmwave_token_centers
        if mmwave_tokens is None:
            if mmwave_points is None:
                raise ValueError("Either mmwave_points or mmwave_tokens must be provided.")
            grad_ctx = torch.no_grad() if self.freeze_mmwave_tokenizer else nullcontext()
            with grad_ctx:
                encoded = self.encode_mmwave_tokens(mmwave_points)
            mmwave_tokens = encoded["tokens"]
            mmwave_token_mask = encoded["mask"]
            centers = encoded.get("centers")

        pred_z_e, pred_logits = self._translate(mmwave_tokens, mmwave_token_mask)
        pred_z_q, pred_indices = self.quantize_tokens(pred_z_e)
        pred_pose_relative = self.vqvae.decode(pred_z_q)

        if point_root_stats is None:
            point_root_stats = masked_point_root_stats(mmwave_points)
        pred_root, pred_root_prior, pred_root_delta = self.root_head.forward_with_components(
            mmwave_tokens,
            mmwave_token_mask,
            centers,
            point_root_stats,
            mmwave_local_point_stats,
        )
        pred_pose_absolute = pred_pose_relative + pred_root.unsqueeze(1)

        out = {
            "pred_z_e": pred_z_e,
            "pred_z_q": pred_z_q,
            "pred_logits": pred_logits,
            "pred_indices": pred_indices,
            "pred_pose": pred_pose_absolute,
            "pred_pose_absolute": pred_pose_absolute,
            "pred_pose_relative": pred_pose_relative,
            "pred_root": pred_root,
            "pred_root_prior": pred_root_prior,
            "pred_root_delta": pred_root_delta,
            "mmwave_tokens": mmwave_tokens,
        }
        if mmwave_token_mask is not None:
            out["mmwave_token_mask"] = mmwave_token_mask
        if centers is not None:
            out["mmwave_token_centers"] = centers
        if point_root_stats is not None:
            out["point_root_stats"] = point_root_stats
        if mmwave_local_point_stats is not None:
            out["mmwave_local_point_stats"] = mmwave_local_point_stats

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

    def load_pretrained_weights(
        self,
        mmwave_tokenizer_path: Optional[str] = None,
        vqvae_path: Optional[str] = None,
        hpe_token_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = self.device

        mmwave_tokenizer_file = _resolve_checkpoint_file(mmwave_tokenizer_path)
        vqvae_file = _resolve_checkpoint_file(vqvae_path)
        hpe_token_file = _resolve_checkpoint_file(hpe_token_path)

        loaded_mmwave_tokenizer_from_hpe_token = False
        if hpe_token_file and hpe_token_file.exists():
            print(f"Loading HPE token checkpoint from {hpe_token_file}")
            ckpt = torch.load(hpe_token_file, map_location=device, weights_only=False)
            if "mmwave_tokenizer_state_dict" in ckpt:
                load_state_dict_shape_safe(self.mmwave_tokenizer, ckpt["mmwave_tokenizer_state_dict"], "MMWaveTokenizer")
                loaded_mmwave_tokenizer_from_hpe_token = True
            elif mmwave_tokenizer_path is None:
                try:
                    load_state_dict_shape_safe(
                        self.mmwave_tokenizer,
                        _extract_mmwave_tokenizer_state_dict(ckpt),
                        "MMWaveTokenizer",
                    )
                    loaded_mmwave_tokenizer_from_hpe_token = True
                except KeyError:
                    pass
            if "token_translator_state_dict" in ckpt:
                load_state_dict_shape_safe(self.token_translator, ckpt["token_translator_state_dict"], "TokenTranslator")
            if "vqvae_state_dict" in ckpt:
                load_state_dict_shape_safe(self.vqvae, ckpt["vqvae_state_dict"], "VQ-VAE")
            if "root_head_state_dict" in ckpt:
                load_state_dict_shape_safe(self.root_head, ckpt["root_head_state_dict"], "RootHead")
        elif hpe_token_path:
            print(f"Warning: HPE token checkpoint not found: {hpe_token_path} -> {hpe_token_file}")

        if mmwave_tokenizer_file and loaded_mmwave_tokenizer_from_hpe_token:
            print("Skipping standalone mmWave tokenizer because HPE checkpoint already provided tokenizer weights.")
        elif mmwave_tokenizer_file and mmwave_tokenizer_file.exists():
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
