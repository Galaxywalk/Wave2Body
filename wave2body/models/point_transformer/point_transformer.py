# -*- coding: utf-8 -*-
"""
Point Transformer for mmWave Radar-based Human Pose Estimation.

This module provides scalable Point Transformer models with depth-first compound scaling.
Supports multiple model sizes: zepto, atto, femto, pico, nano, micro, tiny, small, base, large, xl

Usage:
    config = {
        "data_format": {"radar_input_c": 5, "num_joints": 17},
        "model": {"scale": "base"}
    }
    model = PointTransformer(config, device='cuda')
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .transformer_helper import Backbone, Transformer


# Utility functions
def _count_params(module: nn.Module):
    """Count parameters, handling LazyModules safely."""
    total = 0
    trainable = 0
    for param in module.parameters():
        try:
            if hasattr(param, 'is_lazy') and param.is_lazy:
                continue
            if hasattr(param, '_is_uninitialized') and param._is_uninitialized:
                continue
            param_count = param.numel()
            total += param_count
            if param.requires_grad:
                trainable += param_count
        except (RuntimeError, ValueError):
            continue
    return total, trainable


def _bytes_mb(n_params: int, bytes_per_param: int = 4) -> float:
    """Convert parameter count to megabytes."""
    return n_params * bytes_per_param / (1024 ** 2)


def _sinusoidal_posemb(seq_len: int, dim: int, device=None, dtype=torch.float32):
    """Generate sinusoidal positional embeddings [1, seq_len, dim]."""
    position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


def _estimate_transformer_flops_per_forward(L: int, d: int, mlp_dim: int, depth: int):
    """
    Estimate MACs (multiply-adds) for a Transformer encoder stack.
    Returns GMACs (giga multiply-accumulate operations).
    """
    macs_per_layer = 4 * L * (d ** 2) + (L ** 2) * d + 2 * L * d * mlp_dim
    total_macs = depth * macs_per_layer
    gmacs = total_macs / 1e9
    return gmacs


class RMSNorm(nn.Module):
    """RMSNorm (no bias), stable for deeper networks."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms
        return x * self.weight


# Scaling presets (depth-first compound scaling)
_SCALE_PRESETS = {
    # nblocks: backbone blocks, depth: transformer depth, heads/dim_head: attention config
    # bb_dim: backbone internal channel, feat_dim: output feature dimension
    "zepto": dict(nblocks=1, depth=1, heads=1, dim_head=16, bb_dim=32, feat_dim=4, mlp_ratio=2.0),
    "atto":  dict(nblocks=1, depth=1, heads=1, dim_head=32, bb_dim=32, feat_dim=6, mlp_ratio=2.0),
    "femto": dict(nblocks=1, depth=1, heads=2, dim_head=32, bb_dim=48, feat_dim=8),
    "pico":  dict(nblocks=2, depth=2, heads=2, dim_head=64, bb_dim=64, feat_dim=8),
    "nano":  dict(nblocks=2, depth=2, heads=3, dim_head=64, bb_dim=64, feat_dim=12),
    "micro": dict(nblocks=3, depth=2, heads=4, dim_head=64, bb_dim=96, feat_dim=12),
    "tiny":  dict(nblocks=3, depth=3, heads=6, dim_head=64, bb_dim=96, feat_dim=16),
    "small": dict(nblocks=4, depth=5, heads=8, dim_head=64, bb_dim=128, feat_dim=24),
    "base":  dict(nblocks=5, depth=8, heads=8, dim_head=64, bb_dim=128, feat_dim=32),
    "large": dict(nblocks=6, depth=12, heads=10, dim_head=64, bb_dim=160, feat_dim=48),
    "xl":    dict(nblocks=8, depth=16, heads=12, dim_head=64, bb_dim=192, feat_dim=64),
}


class PointTransformer(nn.Module):
    """
    Scalable Point Transformer for mmWave radar-based human pose estimation.

    This model takes point cloud data from mmWave radar and predicts 3D human joint positions.

    Args:
        config (dict): Configuration dictionary containing:
            - data_format.radar_input_c (int): Input channel dimension (default: 5 for x,y,z,doppler,intensity)
            - data_format.num_joints (int): Number of output joints (default: 17)
            - model.scale (str): Model scale preset (default: "base")
            - model.depth (int, optional): Override transformer depth
            - model.heads (int, optional): Override attention heads
            - model.dim_head (int, optional): Override head dimension
            - model.dropout (float, optional): Dropout rate (default: 0.1)
            - model.nneighbor (int, optional): K for k-NN attention (default: 16)
        device: Torch device for model placement
    """

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

        # Data parameters
        input_dim = int(config["data_format"]["radar_input_c"])
        self.n_p = int(config["data_format"]["num_joints"])

        # Model configuration
        model_cfg_user = config.get("model", {})
        scale = model_cfg_user.get("scale", "base")
        base = dict(_SCALE_PRESETS.get("base"))

        if isinstance(scale, str) and scale in _SCALE_PRESETS:
            base.update(_SCALE_PRESETS[scale])

        # User overrides
        for k in ["nblocks", "depth", "heads", "dim_head", "bb_dim", "feat_dim"]:
            if k in model_cfg_user:
                base[k] = model_cfg_user[k]

        print(f"Using model size: {scale}")

        self.nblocks = int(base["nblocks"])
        depth = int(base["depth"])
        heads = int(base["heads"])
        dim_head = int(base["dim_head"])
        bb_dim = int(base["bb_dim"])
        self.feat_dim = int(base["feat_dim"])

        # Dimension calculations
        dim = int(heads * dim_head)
        mlp_ratio = float(model_cfg_user.get("mlp_ratio", 4.0))
        mlp_dim = max(128, int(dim * mlp_ratio))
        nneighbor = int(model_cfg_user.get("nneighbor", 16))
        dropout = float(model_cfg_user.get("dropout", 0.1))
        self.report_flops = bool(model_cfg_user.get("report_flops", False))

        # Backbone
        self.backbone = Backbone(
            nblocks=self.nblocks,
            nneighbor=nneighbor,
            input_dim=input_dim,
            transformer_dim=bb_dim,
        )
        # Adaptive projection (LazyLinear resolves on first forward)
        self.backbone_proj = nn.LazyLinear(dim)

        # Positional embeddings for joint tokens
        posemb_init = _sinusoidal_posemb(self.n_p, dim, device=device).to(torch.float32)
        self.joint_posembeds_vector = nn.Parameter(posemb_init, requires_grad=True)

        # Pre/Post normalization
        self.in_norm = RMSNorm(dim)
        self.out_norm = RMSNorm(dim)
        self.in_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

        # Transformer encoder stack
        self.transformer = Transformer(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout
        )

        # Regressor head
        fc2_hidden = max(128, dim // 2)
        self.fc2 = nn.Sequential(
            nn.Linear(dim, fc2_hidden), nn.Dropout(dropout),
            nn.ReLU(), nn.Linear(fc2_hidden, self.feat_dim),
        )
        fc3_mid = max(64, self.feat_dim * 2)
        self.fc3 = nn.Sequential(
            nn.ReLU(), nn.Linear(self.feat_dim, fc3_mid), nn.Dropout(dropout),
            nn.ReLU(), nn.Linear(fc3_mid, 3),
        )

        # Model info
        self.model_size_report = {
            "scale": scale if isinstance(scale, str) else "custom",
            "dim": dim, "depth": depth, "heads": heads, "dim_head": dim_head,
            "backbone_blocks": self.nblocks, "backbone_dim": bb_dim,
            "mlp_dim": mlp_dim, "feat_dim": self.feat_dim,
            "params_total": None, "params_trainable": None,
            "param_size_MB_at_fp32": None,
        }
        self._params_counted = False
        self._cached_dim = dim
        self._cached_depth = depth
        self._cached_mlp_dim = mlp_dim

        print(
            f"[PointTransformer:init:{self.model_size_report['scale']}] "
            f"dim={dim}, depth={depth}, heads={heads}, dim_head={dim_head}, "
            f"bb(nblocks={self.nblocks}, dim={bb_dim}), mlp_dim={mlp_dim}, "
            f"dropout={dropout}"
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input point cloud [B, N, C] or [B, T, N, C] (if temporal dimension present)

        Returns:
            pts: Predicted joint positions [B, num_joints, 3]
            feat: Joint features [B, num_joints, feat_dim]
        """
        if len(x.shape) == 4:
            b, t, n, c = x.shape
            x = x.view(b, t * n, c)
        else:
            b, n, c = x.shape

        # Backbone feature extraction
        points, _ = self.backbone(x)
        points = self.backbone_proj(points)

        # Build tokens: [joint_tokens | point_tokens]
        joint_embedding = self.joint_posembeds_vector.expand(b, -1, -1)
        embedding = torch.cat([joint_embedding, points], dim=1)

        # Pre-norm & dropout
        embedding = self.in_norm(embedding)
        embedding = self.in_drop(embedding)

        # Transformer
        output = self.transformer(embedding)[:, :self.n_p, :]

        # Post-norm & dropout
        output = self.out_norm(output)
        output = self.out_drop(output)

        # Heads
        feat = self.fc2(output)
        pts = self.fc3(feat)

        # Count parameters after first forward (LazyLinear now initialized)
        if not self._params_counted:
            total, trainable = _count_params(self)
            size_mb = _bytes_mb(total, 4)
            self.model_size_report.update({
                "params_total": total,
                "params_trainable": trainable,
                "param_size_MB_at_fp32": round(size_mb, 2),
            })
            print(
                f"[PointTransformer] params(train/total)={trainable:,}/{total:,} "
                f"({size_mb:.2f} MB @fp32)"
            )
            if self.report_flops:
                L = embedding.shape[1]
                d = self._cached_dim
                gmacs = _estimate_transformer_flops_per_forward(
                    L, d, self._cached_mlp_dim, self._cached_depth
                )
                print(f"[PointTransformer:FLOPs] ~{gmacs:.3f} GMACs / forward")
            self._params_counted = True

        return pts, feat

    def get_feature_embeddings(self, dataset, batch_size=128, embed_full=False):
        """
        Extract feature embeddings from a dataset.

        Args:
            dataset: PyTorch dataset
            batch_size: Batch size for inference
            embed_full: If True, flatten all joint features; if False, average across joints

        Returns:
            Feature embeddings [N_samples, feat_dim] or [N_samples, num_joints * feat_dim]
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        feature_embeddings = []
        self.eval()
        with torch.no_grad():
            for x, _ in tqdm(dataloader):
                x = x.type(torch.FloatTensor).to(self.device)
                _, feat = self.forward(x)
                if embed_full:
                    feat = feat.reshape(x.shape[0], -1)
                else:
                    feat = feat.mean(dim=-2)
                feature_embeddings.append(feat)
        feature_embeddings = torch.cat(feature_embeddings, dim=0)
        return feature_embeddings
