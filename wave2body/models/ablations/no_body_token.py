# -*- coding: utf-8 -*-
"""No-body-token ablation models."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn

from ..hpe_token.absolute_root import AbsoluteRootHead, ROOT_POINT_STATS_DIM, masked_point_root_stats
from ..hpe_token.estimator import _extract_mmwave_tokenizer_state_dict
from ..hpe_token.common import load_state_dict_shape_safe
from ..mmwave_tokenizer import create_mmwave_tokenizer


def _infer_sensor_token_dim(mmwave_tokenizer_config: dict[str, Any], mode: str) -> int:
    latent_dim = int(mmwave_tokenizer_config.get("latent_dim", 128))
    if mode in {"token", "embedding", "token_only"}:
        return latent_dim
    if mode in {"token_center_concat", "token+center", "embedding_center_concat"}:
        return latent_dim + 3
    raise ValueError(f"Unsupported sensor token mode: {mode!r}")


def _masked_reduce(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    reduce: str,
    fill_value: float,
) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=torch.bool)
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
        raise ValueError(f"Unsupported masked reduce: {reduce!r}")
    valid_any = mask.any(dim=1, keepdim=True)
    return torch.where(valid_any, out, torch.zeros_like(out))


def _build_mlp(input_dim: int, hidden_dim: int, num_layers: int, dropout: float, output_dim: int) -> nn.Sequential:
    depth = max(1, int(num_layers))
    layers: list[nn.Module] = []
    dim = int(input_dim)
    hidden = int(hidden_dim)
    for _ in range(depth - 1):
        layers.extend([nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(float(dropout))])
        dim = hidden
    layers.append(nn.Linear(dim, int(output_dim)))
    return nn.Sequential(*layers)


class PooledMLPRegressor(nn.Module):
    """Direct pose regressor over pooled sensor-token and point statistics."""

    def __init__(
        self,
        *,
        sensor_token_dim: int,
        num_joints: int = 22,
        hidden_dim: int = 512,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_centers: bool = True,
        point_stats_dim: int = ROOT_POINT_STATS_DIM,
    ) -> None:
        super().__init__()
        self.sensor_token_dim = int(sensor_token_dim)
        self.num_joints = int(num_joints)
        self.use_centers = bool(use_centers)
        self.point_stats_dim = int(point_stats_dim)
        center_dim = 9 if self.use_centers else 0
        input_dim = self.sensor_token_dim * 2 + center_dim + self.point_stats_dim
        self.net = _build_mlp(
            input_dim=input_dim,
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
            output_dim=self.num_joints * 3,
        )

    def _center_summary(
        self,
        centers: torch.Tensor | None,
        mask: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not self.use_centers:
            return torch.zeros(batch_size, 0, device=device, dtype=torch.float32)
        if centers is None:
            centers = torch.zeros(batch_size, int(mask.shape[1]), 3, device=device, dtype=torch.float32)
        centers = centers.to(device=device, dtype=torch.float32)
        return torch.cat(
            [
                _masked_reduce(centers, mask, reduce="mean", fill_value=0.0),
                _masked_reduce(centers, mask, reduce="min", fill_value=float("inf")),
                _masked_reduce(centers, mask, reduce="max", fill_value=float("-inf")),
            ],
            dim=-1,
        )

    def _point_features(
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

    def forward(
        self,
        sensor_tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        centers: torch.Tensor | None = None,
        point_stats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(sensor_tokens.shape[:2], device=sensor_tokens.device, dtype=torch.bool)
        else:
            mask = mask.to(device=sensor_tokens.device, dtype=torch.bool)
        batch_size = int(sensor_tokens.shape[0])
        token_mean = _masked_reduce(sensor_tokens.float(), mask, reduce="mean", fill_value=0.0)
        token_max = _masked_reduce(sensor_tokens.float(), mask, reduce="max", fill_value=float("-inf"))
        features = torch.cat(
            [
                token_mean,
                token_max,
                self._center_summary(centers, mask, batch_size=batch_size, device=sensor_tokens.device),
                self._point_features(point_stats, batch_size=batch_size, device=sensor_tokens.device),
            ],
            dim=-1,
        )
        return self.net(features).view(batch_size, self.num_joints, 3).float()


class NoBodyTokenPooledMLP(nn.Module):
    """Frozen mmWave tokenizer plus pooled MLP direct absolute-pose regressor."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        training_cfg = config.get("training", {})
        self.freeze_mmwave_tokenizer = bool(training_cfg.get("freeze_mmwave_tokenizer", True))
        self.sensor_token_config = dict(config.get("sensor_tokens", {}))
        self.sensor_token_mode = str(self.sensor_token_config.get("mode", "token_center_concat")).strip().lower()
        self.center_scale = float(self.sensor_token_config.get("center_scale", 1.0))
        self.zero_invalid_tokens = bool(self.sensor_token_config.get("zero_invalid", True))
        self.mmwave_tokenizer = create_mmwave_tokenizer(config["mmwave_tokenizer"])
        if self.freeze_mmwave_tokenizer:
            for param in self.mmwave_tokenizer.parameters():
                param.requires_grad = False
            self.mmwave_tokenizer.eval()

        sensor_dim = _infer_sensor_token_dim(config["mmwave_tokenizer"], self.sensor_token_mode)
        regressor_cfg = dict(config.get("direct_regressor", config.get("pooled_mlp", {})))
        configured_dim = int(regressor_cfg.get("sensor_token_dim", sensor_dim))
        if configured_dim != sensor_dim:
            raise ValueError(f"sensor_token_dim={configured_dim} does not match inferred sensor dim={sensor_dim}")
        self.regressor = PooledMLPRegressor(
            sensor_token_dim=sensor_dim,
            num_joints=int(config.get("data_format", {}).get("num_joints", regressor_cfg.get("num_joints", 22))),
            hidden_dim=int(regressor_cfg.get("hidden_dim", 512)),
            num_layers=int(regressor_cfg.get("num_layers", 4)),
            dropout=float(regressor_cfg.get("dropout", 0.1)),
            use_centers=bool(regressor_cfg.get("use_centers", True)),
            point_stats_dim=int(regressor_cfg.get("point_stats_dim", ROOT_POINT_STATS_DIM)),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_mmwave_tokenizer:
            self.mmwave_tokenizer.eval()
        return self

    def load_pretrained_weights(self, mmwave_tokenizer_path: str | None = None, *, device: str | torch.device = "cpu") -> None:
        if not mmwave_tokenizer_path:
            return
        from pathlib import Path

        ckpt_path = Path(mmwave_tokenizer_path).expanduser()
        if not ckpt_path.exists():
            print(f"Warning: mmWave tokenizer checkpoint not found: {mmwave_tokenizer_path}")
            return
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = _extract_mmwave_tokenizer_state_dict(ckpt)
        load_state_dict_shape_safe(self.mmwave_tokenizer, state, module_name="NoBodyToken mmWave tokenizer")

    def encode_mmwave_tokens(self, mmwave_points: torch.Tensor) -> dict[str, torch.Tensor]:
        grad_ctx = torch.no_grad() if self.freeze_mmwave_tokenizer else nullcontext()
        with grad_ctx:
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

        centers = tokenizer_out.get("centers")
        if self.sensor_token_mode in {"token", "embedding", "token_only"}:
            sensor_tokens = embeddings
        elif self.sensor_token_mode in {"token_center_concat", "token+center", "embedding_center_concat"}:
            if centers is None:
                raise KeyError("sensor_tokens.mode='token_center_concat' requires tokenizer output centers")
            centers = centers.to(device=embeddings.device, dtype=embeddings.dtype)
            sensor_tokens = torch.cat([embeddings, centers * self.center_scale], dim=-1)
        else:
            raise ValueError(f"Unsupported sensor token mode: {self.sensor_token_mode!r}")
        if self.zero_invalid_tokens:
            sensor_tokens = sensor_tokens.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        out = {"tokens": sensor_tokens, "mask": valid_mask, "token_embeddings": embeddings}
        if centers is not None:
            out["centers"] = centers.to(device=embeddings.device, dtype=embeddings.dtype)
        return out

    def forward(self, mmwave_points: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encode_mmwave_tokens(mmwave_points)
        point_stats = masked_point_root_stats(mmwave_points)
        pred_pose = self.regressor(
            encoded["tokens"],
            encoded["mask"],
            centers=encoded.get("centers"),
            point_stats=point_stats,
        )
        return {
            "pred_pose": pred_pose,
            "pred_pose_absolute": pred_pose,
            "mmwave_tokens": encoded["tokens"],
            "mmwave_token_mask": encoded["mask"],
            "point_root_stats": point_stats,
        }


class NoBodyTokenRelativeRootHeadMLP(NoBodyTokenPooledMLP):
    """No-body-token regressor that predicts relative pose plus Wave2Body root."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        root_cfg = dict(config.get("root_head", {}))
        self.root_index = int(config.get("data_format", {}).get("root_index", root_cfg.pop("root_index", 0)))
        self.root_sensor_token_dim = int(config["mmwave_tokenizer"].get("latent_dim", 128))
        attention_hidden_dim = root_cfg.pop("attention_hidden_dim", None)
        self.root_head = AbsoluteRootHead(
            sensor_token_dim=self.root_sensor_token_dim,
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

    def forward(self, mmwave_points: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encode_mmwave_tokens(mmwave_points)
        point_stats = masked_point_root_stats(mmwave_points)
        pred_pose_relative = self.regressor(
            encoded["tokens"],
            encoded["mask"],
            centers=encoded.get("centers"),
            point_stats=point_stats,
        )
        pred_root, pred_root_prior, pred_root_delta = self.root_head.forward_with_components(
            encoded["token_embeddings"].masked_fill(~encoded["mask"].unsqueeze(-1), 0.0),
            encoded["mask"],
            encoded.get("centers"),
            point_stats,
        )
        pred_pose_absolute = pred_pose_relative + pred_root.unsqueeze(1)
        return {
            "pred_pose": pred_pose_absolute,
            "pred_pose_absolute": pred_pose_absolute,
            "pred_pose_relative": pred_pose_relative,
            "pred_root": pred_root,
            "pred_root_prior": pred_root_prior,
            "pred_root_delta": pred_root_delta,
            "mmwave_tokens": encoded["tokens"],
            "mmwave_token_mask": encoded["mask"],
            "point_root_stats": point_stats,
        }
