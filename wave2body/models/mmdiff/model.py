# -*- coding: utf-8 -*-
"""Wave2Body port of mmDiff's graph-conditioned diffusion refiner."""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_HPE17_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (8, 14),
    (14, 15),
    (15, 16),
)


def make_skeleton_adjacency(
    num_joints: int = 17,
    edges: tuple[tuple[int, int], ...] | list[tuple[int, int]] | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the row-normalized undirected skeleton adjacency used by mmDiff."""

    edges = DEFAULT_HPE17_EDGES if edges is None else tuple(edges)
    adj = torch.eye(num_joints, device=device, dtype=dtype)
    for src, dst in edges:
        if src >= num_joints or dst >= num_joints:
            continue
        adj[src, dst] = 1.0
        adj[dst, src] = 1.0
    rowsum = adj.sum(dim=1, keepdim=True).clamp_min(1.0)
    return adj / rowsum


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Sinusoidal diffusion timestep embedding from the upstream mmDiff code."""

    if timesteps.ndim != 1:
        raise ValueError("timesteps must be a rank-1 tensor")
    half_dim = embedding_dim // 2
    scale = math.log(10000.0) / max(half_dim - 1, 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -scale)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def _swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class ChebConv(nn.Module):
    """Chebyshev graph convolution matching mmDiff's GraphFormer blocks."""

    def __init__(self, in_channels: int, out_channels: int, order: int = 2, bias: bool = True) -> None:
        super().__init__()
        self.order = int(order) + 1
        self.weight = nn.Parameter(torch.empty(self.order, 1, in_channels, out_channels))
        nn.init.xavier_normal_(self.weight)
        self.bias = nn.Parameter(torch.zeros(1, 1, out_channels)) if bias else None
        self._cached_polynomials_key: tuple[torch.device, torch.dtype, int, int] | None = None
        self._cached_polynomials: torch.Tensor | None = None

    def forward(self, inputs: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        polynomials = self._cheb_polynomial_from_graph(graph).unsqueeze(1)
        result = torch.matmul(polynomials, inputs)
        result = torch.matmul(result, self.weight).sum(dim=0)
        if self.bias is not None:
            result = result + self.bias
        return result

    def _cheb_polynomial_from_graph(self, graph: torch.Tensor) -> torch.Tensor:
        if graph.requires_grad:
            return self.cheb_polynomial(self.get_laplacian(graph))

        key = (
            graph.device,
            graph.dtype,
            graph.data_ptr(),
            int(getattr(graph, "_version", 0)),
        )
        cached = self._cached_polynomials
        if cached is not None and self._cached_polynomials_key == key:
            return cached

        polynomials = self.cheb_polynomial(self.get_laplacian(graph))
        self._cached_polynomials_key = key
        self._cached_polynomials = polynomials
        return polynomials

    def cheb_polynomial(self, laplacian: torch.Tensor) -> torch.Tensor:
        num_nodes = laplacian.shape[0]
        polynomials = torch.zeros(
            (self.order, num_nodes, num_nodes),
            device=laplacian.device,
            dtype=laplacian.dtype,
        )
        polynomials[0] = torch.eye(num_nodes, device=laplacian.device, dtype=laplacian.dtype)
        if self.order > 1:
            polynomials[1] = laplacian
        for idx in range(2, self.order):
            polynomials[idx] = 2.0 * torch.mm(laplacian, polynomials[idx - 1]) - polynomials[idx - 2]
        return polynomials

    @staticmethod
    def get_laplacian(graph: torch.Tensor) -> torch.Tensor:
        degree = torch.sum(graph, dim=-1).clamp_min(1.0e-6).pow(-0.5)
        degree_mat = torch.diag(degree)
        eye = torch.eye(graph.shape[0], device=graph.device, dtype=graph.dtype)
        return eye - degree_mat @ graph @ degree_mat


class _GraphConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.gconv = ChebConv(input_dim, output_dim, order=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.dropout(F.relu(self.gconv(x, adj)))


class _ResChebGCDiff(nn.Module):
    def __init__(self, adj: torch.Tensor, hidden_dim: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.register_buffer("adj", adj)
        self.gconv1 = _GraphConv(hidden_dim, hidden_dim, dropout)
        self.gconv2 = _GraphConv(hidden_dim, hidden_dim, dropout)
        self.temb_proj = nn.Linear(embedding_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        out = self.gconv1(x, self.adj)
        out = out + self.temb_proj(_swish(temb)).unsqueeze(1)
        if condition is not None:
            out = out + condition
        out = self.gconv2(out, self.adj)
        return residual + out


def _clones(module: nn.Module, count: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(count)])


class _LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + self.eps) + self.bias


class _SublayerConnection(nn.Module):
    def __init__(self, size: int, dropout: float) -> None:
        super().__init__()
        self.norm = _LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


class _MultiHeadedAttention(nn.Module):
    def __init__(self, heads: int, dim_model: int, dropout: float) -> None:
        super().__init__()
        if dim_model % heads != 0:
            raise ValueError("dim_model must be divisible by heads")
        self.heads = int(heads)
        self.head_dim = dim_model // heads
        self.linears = _clones(nn.Linear(dim_model, dim_model), 4)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            mask = mask.unsqueeze(1)
            if bool(mask.all()):
                mask = None
        batch_size = query.shape[0]
        query, key, value = [
            layer(x).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
            for layer, x in zip(self.linears, (query, key, value))
        ]
        if mask is None and hasattr(F, "scaled_dot_product_attention"):
            x = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        else:
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1.0e9)
            attn = self.dropout(torch.softmax(scores, dim=-1))
            x = torch.matmul(attn, value)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.heads * self.head_dim)
        return self.linears[-1](x)


class _LAMGraphConv(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: nn.Module | None = nn.ReLU(inplace=True),
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.activation = activation

    @staticmethod
    def laplacian_batch(adj: torch.Tensor) -> torch.Tensor:
        degree = (torch.sum(adj, dim=1) + 1.0e-5).pow(-0.5)
        return degree.unsqueeze(2) * adj * degree.unsqueeze(1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
        x = self.fc(torch.bmm(self.laplacian_batch(adj), x))
        return x if self.activation is None else self.activation(x)


class _GraphNet(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_joints: int) -> None:
        super().__init__()
        self.adj = nn.Parameter(torch.eye(num_joints), requires_grad=True)
        self.gconv1 = _LAMGraphConv(in_features, in_features * 2)
        self.gconv2 = _LAMGraphConv(in_features * 2, out_features, activation=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gconv2(self.gconv1(x, self.adj), self.adj)


class _GraphAttentionLayer(nn.Module):
    def __init__(self, size: int, heads: int, num_joints: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = _MultiHeadedAttention(heads, size, dropout)
        self.feed_forward = _GraphNet(size, size, num_joints)
        self.sublayer = _clones(_SublayerConnection(size, dropout), 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        x = self.sublayer[0](x, lambda y: self.self_attn(y, y, y, mask))
        return self.sublayer[1](x, self.feed_forward)


class MMDiffDenoiser(nn.Module):
    """Graph-conditioned mmDiff denoising block.

    ``pose_history`` uses shape ``[B, J, 3 * history_frames]`` where the final
    triplet is the current coarse/noisy pose.  This mirrors the effective tensor
    contract used by the upstream runner while avoiding hard-coded CUDA calls.
    """

    def __init__(
        self,
        *,
        num_joints: int = 17,
        hidden_dim: int = 96,
        condition_dim: int = 64,
        history_frames: int = 6,
        radar_input_dim: int = 6,
        num_layers: int = 5,
        num_heads: int = 4,
        dropout: float = 0.25,
        local_points: int = 50,
        local_radius_sq: float = 0.04,
        use_local: bool = True,
        use_temporal: bool = True,
        use_limb: bool = True,
        edges: tuple[tuple[int, int], ...] | list[tuple[int, int]] | None = None,
    ) -> None:
        super().__init__()
        self.num_joints = int(num_joints)
        self.hidden_dim = int(hidden_dim)
        self.condition_dim = int(condition_dim)
        self.history_frames = int(history_frames)
        self.radar_input_dim = int(radar_input_dim)
        self.local_points = int(local_points)
        self.local_radius_sq = float(local_radius_sq)
        self.use_local = bool(use_local)
        self.use_temporal = bool(use_temporal)
        self.use_limb = bool(use_limb)

        adj = make_skeleton_adjacency(self.num_joints, edges=edges)
        self.register_buffer("adj", adj)
        self.register_buffer("src_mask", torch.ones(1, 1, self.num_joints, dtype=torch.bool))

        self.time_mlp = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim, self.hidden_dim * 4),
                nn.Linear(self.hidden_dim * 4, self.hidden_dim * 4),
            ]
        )
        self.condition_gconv = ChebConv(self.condition_dim, self.hidden_dim, order=2)
        self.current_pose_gconv = ChebConv(3, self.hidden_dim, order=2)
        self.output_gconv = ChebConv(self.hidden_dim, 3 * self.history_frames, order=2)

        if self.use_local:
            self.local_mlp = nn.ModuleList(
                [
                    nn.Linear(self.radar_input_dim, self.hidden_dim // 2),
                    nn.Linear(self.hidden_dim // 2, self.hidden_dim),
                ]
            )
            local_attn = _MultiHeadedAttention(num_heads, self.hidden_dim, dropout)
            self.local_attention = _clones(local_attn, 5)

        if self.use_temporal:
            self.temporal_gconv = ChebConv(3, self.hidden_dim, order=2)
            conv_dim = self.hidden_dim * self.num_joints
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(conv_dim, conv_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(conv_dim, conv_dim, kernel_size=3, padding=1),
            )

        if self.use_limb:
            limb_count = max(self.num_joints - 1, 1)
            self.limb_head = nn.Sequential(
                nn.LayerNorm(self.num_joints * self.condition_dim),
                nn.Linear(self.num_joints * self.condition_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, limb_count),
            )
            self.limb_mlp = nn.ModuleList(
                [
                    nn.Linear(limb_count, self.hidden_dim * 4),
                    nn.Linear(self.hidden_dim * 4, self.hidden_dim * 4),
                ]
            )

        self.attention_layers = nn.ModuleList(
            [_GraphAttentionLayer(self.hidden_dim, num_heads, self.num_joints, dropout) for _ in range(num_layers)]
        )
        self.gconv_layers = nn.ModuleList(
            [_ResChebGCDiff(self.adj, self.hidden_dim, self.hidden_dim * 4, dropout) for _ in range(num_layers)]
        )

    def _local_embedding(self, pose_history: torch.Tensor, radar: torch.Tensor) -> torch.Tensor:
        if radar.ndim == 4:
            batch_size, time_size, point_size, channels = radar.shape
            point_cloud = radar.reshape(batch_size, time_size * point_size, channels)
        elif radar.ndim == 3:
            batch_size = radar.shape[0]
            point_cloud = radar
        else:
            raise ValueError("radar must have shape [B, T, N, C] or [B, N, C]")

        if point_cloud.shape[-1] != self.radar_input_dim:
            raise ValueError(f"expected radar_input_dim={self.radar_input_dim}, got {point_cloud.shape[-1]}")

        anchors = pose_history[:, :, -3:]
        k = min(self.local_points, point_cloud.shape[1])
        distances = torch.sum((point_cloud[:, None, :, :3] - anchors[:, :, None, :]).square(), dim=-1)
        indices = torch.topk(distances, k=k, dim=-1, largest=False, sorted=True).indices
        selected = torch.gather(
            point_cloud[:, None, :, :].expand(-1, self.num_joints, -1, -1),
            2,
            indices.unsqueeze(-1).expand(-1, -1, -1, point_cloud.shape[-1]),
        )
        dist_selected = torch.gather(distances, 2, indices)
        local_score = dist_selected.lt(self.local_radius_sq).float().mean(dim=-1, keepdim=True)

        emb = self.local_mlp[0](selected)
        emb = _swish(emb)
        emb = self.local_mlp[1](emb)
        emb = emb.reshape(batch_size * self.num_joints, k, self.hidden_dim)
        for attn in self.local_attention:
            emb = emb + attn(emb, emb, emb)
        emb = emb.reshape(batch_size, self.num_joints, k, self.hidden_dim)
        return emb.mean(dim=2) * local_score

    def _temporal_embedding(self, pose_history: torch.Tensor) -> torch.Tensor:
        batch_size = pose_history.shape[0]
        frame_pose = pose_history.reshape(batch_size, self.num_joints, self.history_frames, 3)
        frame_pose = frame_pose.permute(0, 2, 1, 3).reshape(batch_size * self.history_frames, self.num_joints, 3)
        stacked = self.temporal_gconv(frame_pose, self.adj)
        stacked = stacked.reshape(batch_size, self.history_frames, self.num_joints, self.hidden_dim)
        temporal = stacked.reshape(batch_size, self.history_frames, self.num_joints * self.hidden_dim).transpose(1, 2)
        temporal = self.temporal_conv(temporal).mean(dim=-1)
        return temporal.view(batch_size, self.num_joints, self.hidden_dim)

    def forward(
        self,
        pose_history: torch.Tensor,
        timesteps: torch.Tensor | int,
        condition: torch.Tensor,
        radar: torch.Tensor | None = None,
        limb_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if pose_history.shape[1:] != (self.num_joints, 3 * self.history_frames):
            raise ValueError(
                "pose_history must have shape "
                f"[B, {self.num_joints}, {3 * self.history_frames}], got {tuple(pose_history.shape)}"
            )
        if condition.shape[:2] != (pose_history.shape[0], self.num_joints):
            raise ValueError("condition must have shape [B, J, C]")

        batch_size = pose_history.shape[0]
        if isinstance(timesteps, int):
            timesteps = torch.full((batch_size,), timesteps, device=pose_history.device, dtype=torch.long)
        elif timesteps.ndim == 0:
            timesteps = timesteps.expand(batch_size).to(device=pose_history.device)
        else:
            timesteps = timesteps.to(device=pose_history.device)

        temb = get_timestep_embedding(timesteps, self.hidden_dim)
        temb = self.time_mlp[0](temb)
        temb = _swish(temb)
        temb = self.time_mlp[1](temb)

        condition_embedding = self.condition_gconv(condition, self.adj)
        if self.use_local and radar is not None:
            condition_embedding = condition_embedding + self._local_embedding(pose_history, radar)
        if self.use_temporal:
            condition_embedding = condition_embedding + self._temporal_embedding(pose_history)

        limb_pred = pose_history.new_zeros(batch_size, max(self.num_joints - 1, 1))
        if self.use_limb:
            limb_pred = self.limb_head(condition.reshape(batch_size, self.num_joints * self.condition_dim))
            limb_in = limb_pred if limb_lengths is None else limb_lengths
            limb_emb = self.limb_mlp[0](limb_in)
            limb_emb = _swish(limb_emb)
            temb = temb + self.limb_mlp[1](limb_emb)

        out = self.current_pose_gconv(pose_history[:, :, -3:], self.adj) + condition_embedding
        for attention, graph_layer in zip(self.attention_layers, self.gconv_layers):
            out = attention(out, None)
            out = graph_layer(out, temb, condition=condition_embedding)
        denoised_history = self.output_gconv(out, self.adj)
        return {
            "pred_pose": denoised_history[:, :, -3:],
            "denoised_history": denoised_history,
            "limb_lengths": limb_pred,
        }


class MMDiffPoseModel(nn.Module):
    """End-to-end mmDiff-style pose model for Wave2Body.

    The feature encoder must return either ``(coarse_pose, joint_features)`` or a
    mapping with ``pred_pose`` and ``joint_feat`` keys.
    """

    def __init__(
        self,
        *,
        feature_encoder: nn.Module | None = None,
        num_joints: int = 17,
        radar_input_dim: int = 6,
        condition_dim: int = 64,
        hidden_dim: int = 96,
        history_frames: int = 6,
        num_layers: int = 5,
        num_heads: int = 4,
        dropout: float = 0.25,
        local_points: int = 50,
        edges: tuple[tuple[int, int], ...] | list[tuple[int, int]] | None = None,
    ) -> None:
        super().__init__()
        self.num_joints = int(num_joints)
        self.history_frames = int(history_frames)
        if feature_encoder is None:
            from wave2body.models.point_transformer import PointTransformer

            feature_encoder = PointTransformer(
                {
                    "data_format": {"radar_input_c": radar_input_dim, "num_joints": num_joints},
                    "model": {
                        "scale": "femto",
                        "feat_dim": condition_dim,
                        "dropout": dropout,
                    },
                },
                device="cpu",
            )
        self.feature_encoder = feature_encoder
        self.condition_proj = nn.LazyLinear(condition_dim)
        self.denoiser = MMDiffDenoiser(
            num_joints=num_joints,
            hidden_dim=hidden_dim,
            condition_dim=condition_dim,
            history_frames=history_frames,
            radar_input_dim=radar_input_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            local_points=local_points,
            edges=edges,
        )

    @staticmethod
    def _parse_encoder_output(output: object) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(output, dict):
            pose = output["pred_pose"]
            feat = output["joint_feat"]
            return pose, feat
        if isinstance(output, tuple) and len(output) >= 2:
            return output[0], output[1]
        raise TypeError("feature_encoder must return a dict or tuple containing pose and joint features")

    def forward(
        self,
        radar: torch.Tensor,
        timesteps: torch.Tensor | int = 0,
        *,
        x_curr: torch.Tensor | None = None,
        x_history: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        coarse_pose: torch.Tensor | None = None
        joint_feat: torch.Tensor | None = condition
        if x_curr is None or condition is None:
            coarse_pose, encoder_feat = self._parse_encoder_output(self.feature_encoder(radar))
            if x_curr is None:
                x_curr = coarse_pose
            if condition is None:
                joint_feat = encoder_feat

        if x_curr.shape[1:] != (self.num_joints, 3):
            raise ValueError(f"x_curr must have shape [B, {self.num_joints}, 3]")
        if joint_feat is None:
            raise ValueError("condition is required when feature_encoder is not used")

        condition_proj = self.condition_proj(joint_feat)
        if x_history is None:
            x_history = torch.cat([x_curr] * self.history_frames, dim=-1)
        else:
            expected = (self.num_joints, 3 * self.history_frames)
            if x_history.shape[1:] != expected:
                raise ValueError(f"x_history must have shape [B, {expected[0]}, {expected[1]}]")
            x_history = torch.cat([x_history[:, :, 3:], x_curr], dim=-1)

        denoised = self.denoiser(
            pose_history=x_history,
            timesteps=timesteps,
            condition=condition_proj,
            radar=radar,
        )
        denoised.update(
            {
                "coarse_pose": x_curr if coarse_pose is None else coarse_pose,
                "pose_history": x_history,
                "joint_feat": condition_proj,
            }
        )
        return denoised


class MMDiffPoseLoss(nn.Module):
    """Loss for mmDiff-style coarse pose and denoised pose outputs."""

    def __init__(
        self,
        *,
        pose_weight: float = 1.0,
        coarse_weight: float = 0.0,
        history_weight: float = 0.0,
        limb_weight: float = 0.0,
        reduction: str = "mse",
    ) -> None:
        super().__init__()
        self.pose_weight = float(pose_weight)
        self.coarse_weight = float(coarse_weight)
        self.history_weight = float(history_weight)
        self.limb_weight = float(limb_weight)
        self.reduction = str(reduction).strip().lower()

    def _pose_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"pose shape mismatch: {pred.shape} vs {target.shape}")
        pred = pred.float()
        target = target.detach().float()
        if self.reduction == "mse":
            return F.mse_loss(pred, target)
        if self.reduction == "l1":
            return F.l1_loss(pred, target)
        if self.reduction == "mpjpe":
            return torch.norm(pred - target, dim=-1).mean()
        raise ValueError(f"Unsupported reduction: {self.reduction!r}")

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        target_pose: torch.Tensor,
        *,
        target_history: torch.Tensor | None = None,
        target_limb_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if "pred_pose" not in outputs:
            raise KeyError("MMDiffPoseLoss expects outputs['pred_pose']")

        losses: dict[str, torch.Tensor] = {}
        total = outputs["pred_pose"].new_zeros(())

        pose_loss = self._pose_loss(outputs["pred_pose"], target_pose)
        losses["pose_loss"] = pose_loss
        total = total + self.pose_weight * pose_loss

        if self.coarse_weight > 0 and "coarse_pose" in outputs:
            coarse_loss = self._pose_loss(outputs["coarse_pose"], target_pose)
            losses["coarse_pose_loss"] = coarse_loss
            total = total + self.coarse_weight * coarse_loss

        if self.history_weight > 0 and target_history is not None and "denoised_history" in outputs:
            history_loss = self._pose_loss(outputs["denoised_history"], target_history)
            losses["history_loss"] = history_loss
            total = total + self.history_weight * history_loss

        if self.limb_weight > 0 and target_limb_lengths is not None and "limb_lengths" in outputs:
            pred_limb = outputs["limb_lengths"].float()
            target_limb = target_limb_lengths.detach().to(device=pred_limb.device, dtype=pred_limb.dtype)
            if pred_limb.shape != target_limb.shape:
                raise ValueError(f"limb length shape mismatch: {pred_limb.shape} vs {target_limb.shape}")
            limb_loss = F.mse_loss(pred_limb, target_limb)
            losses["limb_loss"] = limb_loss
            total = total + self.limb_weight * limb_loss

        with torch.no_grad():
            losses["mean_joint_error_m"] = torch.norm(outputs["pred_pose"].float() - target_pose.float(), dim=-1).mean()
        losses["total_loss"] = total
        return losses


def create_mmdiff_model(
    config: dict[str, Any] | None = None,
    *,
    feature_encoder: nn.Module | None = None,
    **overrides: Any,
) -> MMDiffPoseModel:
    """Instantiate :class:`MMDiffPoseModel` from a config dictionary."""

    cfg: dict[str, Any] = dict(config or {})
    cfg.update(overrides)
    allowed = {
        "num_joints",
        "radar_input_dim",
        "condition_dim",
        "hidden_dim",
        "history_frames",
        "num_layers",
        "num_heads",
        "dropout",
        "local_points",
        "edges",
    }
    kwargs = {key: value for key, value in cfg.items() if key in allowed}
    return MMDiffPoseModel(feature_encoder=feature_encoder, **kwargs)
