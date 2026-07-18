# -*- coding: utf-8 -*-
"""Absolute root-head extension of the deployable HPE-token estimator."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from .common import load_state_dict_shape_safe
from .estimator import TokenHPEEstimatorModel, _extract_mmwave_tokenizer_state_dict
from ..human_body_tokenizer import load_pretrained_vqvae
from ...utils.paths import resolve_runtime_uri


ROOT_POINT_STATS_BASE_DIM = 16
ROOT_POINT_STATS_DIM = 19


def _masked_reduce(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    reduce: str,
    fill_value: float,
) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).to(dtype=values.dtype)
    if reduce == "mean":
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return (values * mask_f).sum(dim=1) / denom
    masked = values.masked_fill(~mask.unsqueeze(-1), fill_value)
    if reduce == "max":
        out = masked.max(dim=1).values
    elif reduce == "min":
        out = masked.min(dim=1).values
    else:
        raise ValueError(f"Unsupported masked reduce: {reduce}")
    valid_any = mask.any(dim=1, keepdim=True)
    return torch.where(valid_any, out, torch.zeros_like(out))


def masked_point_root_stats(mmwave_points: torch.Tensor | None) -> torch.Tensor | None:
    """Return compact absolute-location statistics from padded radar points."""

    if mmwave_points is None:
        return None
    if mmwave_points.dim() != 3 or int(mmwave_points.shape[-1]) < 4:
        raise ValueError(f"Expected mmwave_points [B,N,>=4], got {tuple(mmwave_points.shape)}")
    points = mmwave_points[..., :4].float()
    valid = torch.any(torch.abs(points) > 0, dim=-1)
    xyz = points[..., :3]
    intensity = points[..., 3:4]

    mean_xyz = _masked_reduce(xyz, valid, reduce="mean", fill_value=0.0)
    centered = (xyz - mean_xyz.unsqueeze(1)).masked_fill(~valid.unsqueeze(-1), 0.0)
    denom = valid.unsqueeze(-1).to(dtype=xyz.dtype).sum(dim=1).clamp_min(1.0)
    std_xyz = torch.sqrt((centered.square().sum(dim=1) / denom).clamp_min(0.0) + 1e-8)
    min_xyz = _masked_reduce(xyz, valid, reduce="min", fill_value=float("inf"))
    max_xyz = _masked_reduce(xyz, valid, reduce="max", fill_value=float("-inf"))

    mean_i = _masked_reduce(intensity, valid, reduce="mean", fill_value=0.0)
    centered_i = (intensity - mean_i.unsqueeze(1)).masked_fill(~valid.unsqueeze(-1), 0.0)
    std_i = torch.sqrt((centered_i.square().sum(dim=1) / denom[:, :1]).clamp_min(0.0) + 1e-8)
    min_i = _masked_reduce(intensity, valid, reduce="min", fill_value=float("inf"))
    max_i = _masked_reduce(intensity, valid, reduce="max", fill_value=float("-inf"))

    weights = intensity.abs().masked_fill(~valid.unsqueeze(-1), 0.0)
    weight_sum = weights.sum(dim=1).clamp_min(1e-6)
    weighted_mean_xyz = (xyz * weights).sum(dim=1) / weight_sum
    valid_any = valid.any(dim=1, keepdim=True)
    weighted_mean_xyz = torch.where(valid_any, weighted_mean_xyz, mean_xyz)

    return torch.cat([mean_xyz, std_xyz, min_xyz, max_xyz, mean_i, std_i, min_i, max_i, weighted_mean_xyz], dim=-1)


def _build_residual_mlp(input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> nn.Sequential:
    depth = max(1, int(num_layers))
    hidden = int(hidden_dim)
    layers: list[nn.Module] = []
    dim = int(input_dim)
    for _ in range(depth - 1):
        layers.extend([nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(float(dropout))])
        dim = hidden
    layers.append(nn.Linear(dim, 3))
    return nn.Sequential(*layers)


def _build_hidden_mlp(input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> nn.Sequential:
    depth = max(1, int(num_layers))
    hidden = int(hidden_dim)
    layers: list[nn.Module] = []
    dim = int(input_dim)
    for _ in range(depth - 1):
        layers.extend([nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(float(dropout))])
        dim = hidden
    layers.append(nn.Linear(dim, hidden))
    return nn.Sequential(*layers)


def _resolve_checkpoint_file(path: Optional[str]) -> Path | None:
    if path in (None, "", "null"):
        return None
    return Path(resolve_runtime_uri(str(path))).expanduser()


class AbsoluteRootHead(nn.Module):
    """Predict radar-frame pelvis/root coordinates from deployable sensor evidence."""

    def __init__(
        self,
        *,
        sensor_token_dim: int,
        point_stats_dim: int = ROOT_POINT_STATS_DIM,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_centers: bool = True,
        prior_mode: str = "none",
        residual_scale: float = 1.0,
        architecture: str = "query_attn",
        attention_hidden_dim: int | None = None,
        attention_heads: int = 4,
        num_query_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.sensor_token_dim = int(sensor_token_dim)
        self.point_stats_dim = int(point_stats_dim)
        if self.point_stats_dim <= 0:
            raise ValueError(f"root_head.point_stats_dim must be positive, got {self.point_stats_dim}")
        self.use_centers = bool(use_centers)
        self.prior_mode = str(prior_mode or "none").strip().lower()
        self.residual_scale = float(residual_scale)
        self.architecture = str(architecture or "query_attn").strip().lower()
        if self.prior_mode not in {
            "none",
            "point_mean",
            "point_bbox_midpoint",
            "point_intensity_weighted_mean",
            "point_iw_mean",
        }:
            raise ValueError(
                "Unsupported root_head.prior_mode="
                f"{self.prior_mode!r}; expected one of "
                "['none', 'point_mean', 'point_bbox_midpoint', 'point_intensity_weighted_mean']"
            )
        if self.architecture != "query_attn":
            raise ValueError(
                f"Unsupported root_head.architecture={self.architecture!r}; "
                "production HPE-token uses 'query_attn'. Historical root-head "
                "architectures live under wave2body.models.hpe_token.baselines."
            )
        center_dim = 9 if self.use_centers else 0
        attn_hidden = int(attention_hidden_dim or hidden_dim)
        heads = max(1, int(attention_heads))
        if attn_hidden % heads != 0:
            raise ValueError(f"root_head.attention_hidden_dim={attn_hidden} must be divisible by attention_heads={heads}")
        self.num_query_tokens = max(1, int(num_query_tokens))
        token_feature_dim = self.sensor_token_dim + (6 if self.use_centers else 0)
        context_dim = self.point_stats_dim + center_dim
        self.token_proj = nn.Sequential(
            nn.Linear(token_feature_dim, attn_hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(attn_hidden, attn_hidden),
        )
        self.context_proj = _build_hidden_mlp(context_dim, attn_hidden, 2, float(dropout))
        self.query_tokens = nn.Parameter(torch.empty(1, self.num_query_tokens, attn_hidden))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=attn_hidden,
            num_heads=heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(attn_hidden)
        self.token_norm = nn.LayerNorm(attn_hidden)
        self.query_ffn = nn.Sequential(
            nn.LayerNorm(attn_hidden),
            nn.Linear(attn_hidden, attn_hidden * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(attn_hidden * 2, attn_hidden),
            nn.Dropout(float(dropout)),
        )
        input_dim = self.sensor_token_dim * 2 + attn_hidden + self.point_stats_dim + center_dim
        self.net = _build_residual_mlp(input_dim, int(hidden_dim), int(num_layers), float(dropout))

    def _root_prior(self, point_stats: torch.Tensor | None, *, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.prior_mode == "none":
            return torch.zeros(batch_size, 3, device=device, dtype=torch.float32)
        if point_stats is None:
            return torch.zeros(batch_size, 3, device=device, dtype=torch.float32)
        point_stats = point_stats.to(device=device, dtype=torch.float32)
        if self.prior_mode == "point_mean":
            min_dim = 3
        elif self.prior_mode == "point_bbox_midpoint":
            min_dim = 12
        else:
            min_dim = ROOT_POINT_STATS_DIM
        if point_stats.shape[-1] < min_dim:
            raise ValueError(
                f"root_head.prior_mode={self.prior_mode!r} requires point stats dim >= {min_dim}, "
                f"got {point_stats.shape[-1]}"
            )
        if self.prior_mode == "point_mean":
            return point_stats[..., :3]
        if self.prior_mode == "point_bbox_midpoint":
            return 0.5 * (point_stats[..., 6:9] + point_stats[..., 9:12])
        return point_stats[..., ROOT_POINT_STATS_BASE_DIM:ROOT_POINT_STATS_DIM]

    def _point_stats_features(
        self,
        point_stats: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if point_stats is None:
            return torch.zeros(batch_size, self.point_stats_dim, device=device, dtype=torch.float32)
        point_stats = point_stats.to(device=device, dtype=torch.float32)
        if point_stats.shape[-1] == self.point_stats_dim:
            return point_stats
        if point_stats.shape[-1] > self.point_stats_dim:
            return point_stats[..., : self.point_stats_dim]
        pad = torch.zeros(
            *point_stats.shape[:-1],
            self.point_stats_dim - int(point_stats.shape[-1]),
            device=device,
            dtype=torch.float32,
        )
        return torch.cat([point_stats, pad], dim=-1)

    def _center_features(
        self,
        centers: torch.Tensor | None,
        mmwave_mask: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if not self.use_centers:
            return None, torch.zeros(batch_size, 0, device=device, dtype=torch.float32)
        if centers is None:
            center_tokens = torch.zeros(
                batch_size,
                int(mmwave_mask.shape[1]),
                3,
                device=device,
                dtype=torch.float32,
            )
        else:
            center_tokens = centers.to(device=device, dtype=torch.float32)
        center_summary = torch.cat(
            [
                _masked_reduce(center_tokens, mmwave_mask, reduce="mean", fill_value=0.0),
                _masked_reduce(center_tokens, mmwave_mask, reduce="min", fill_value=float("inf")),
                _masked_reduce(center_tokens, mmwave_mask, reduce="max", fill_value=float("-inf")),
            ],
            dim=-1,
        )
        return center_tokens, center_summary

    def forward_with_components(
        self,
        mmwave_tokens: torch.Tensor,
        mmwave_mask: torch.Tensor | None = None,
        centers: torch.Tensor | None = None,
        point_stats: torch.Tensor | None = None,
        local_point_stats: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = self._residual(mmwave_tokens, mmwave_mask, centers, point_stats, local_point_stats)
        prior = self._root_prior(point_stats, batch_size=int(mmwave_tokens.shape[0]), device=mmwave_tokens.device)
        pred_root = prior + self.residual_scale * residual
        return pred_root.float(), prior.float(), residual.float()

    def forward(
        self,
        mmwave_tokens: torch.Tensor,
        mmwave_mask: torch.Tensor | None = None,
        centers: torch.Tensor | None = None,
        point_stats: torch.Tensor | None = None,
        local_point_stats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred_root, _, _ = self.forward_with_components(mmwave_tokens, mmwave_mask, centers, point_stats, local_point_stats)
        return pred_root

    def _residual(
        self,
        mmwave_tokens: torch.Tensor,
        mmwave_mask: torch.Tensor | None = None,
        centers: torch.Tensor | None = None,
        point_stats: torch.Tensor | None = None,
        local_point_stats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mmwave_mask is None:
            mmwave_mask = torch.ones(mmwave_tokens.shape[:2], device=mmwave_tokens.device, dtype=torch.bool)
        else:
            mmwave_mask = mmwave_mask.to(device=mmwave_tokens.device, dtype=torch.bool)

        batch_size = int(mmwave_tokens.shape[0])
        device = mmwave_tokens.device
        token_mean = _masked_reduce(mmwave_tokens.float(), mmwave_mask, reduce="mean", fill_value=0.0)
        token_max = _masked_reduce(mmwave_tokens.float(), mmwave_mask, reduce="max", fill_value=float("-inf"))
        center_tokens, center_summary = self._center_features(
            centers,
            mmwave_mask,
            batch_size=batch_size,
            device=device,
        )
        point_features = self._point_stats_features(point_stats, batch_size=batch_size, device=device)
        features = [token_mean, token_max]
        token_inputs = [mmwave_tokens.float()]
        if self.use_centers:
            assert center_tokens is not None
            prior = self._root_prior(point_stats, batch_size=batch_size, device=device)
            token_inputs.extend([center_tokens, center_tokens - prior.unsqueeze(1)])
        token_features = torch.cat(token_inputs, dim=-1)
        token_repr = self.token_norm(self.token_proj(token_features))
        context_features = torch.cat([point_features, center_summary], dim=-1)
        query_context = self.context_proj(context_features).unsqueeze(1)
        queries = self.query_tokens.expand(batch_size, -1, -1) + query_context
        key_padding_mask = ~mmwave_mask
        all_invalid = key_padding_mask.all(dim=1)
        if all_invalid.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_invalid] = False
        attn_out, _ = self.cross_attn(
            self.query_norm(queries),
            token_repr,
            token_repr,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        query_tokens = queries + attn_out
        query_tokens = query_tokens + self.query_ffn(query_tokens)
        features.append(query_tokens.mean(dim=1))
        if self.use_centers:
            features.append(center_summary)
        features.append(point_features)
        return self.net(torch.cat(features, dim=-1)).float()


class AbsoluteRootHeadEstimatorModel(TokenHPEEstimatorModel):
    """
    HPE-token estimator with an explicit absolute-root head.

    The inherited tokenizer/translator/VQ-VAE path predicts pelvis-centered
    pose tokens.  This class predicts the radar-frame root and composes
    ``absolute_pose = relative_pose + root`` without changing the locked K96
    body-token decoder.
    """

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
            point_stats_dim=int(root_cfg.pop("point_stats_dim", ROOT_POINT_STATS_DIM)),
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
            print(
                "Skipping standalone mmWave tokenizer because the HPE token "
                "checkpoint already provided tokenizer weights."
            )
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
