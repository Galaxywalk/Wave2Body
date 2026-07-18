# -*- coding: utf-8 -*-
"""Wave2Body port of the upstream mmMesh model architecture.

The upstream repository keeps SMPL model files and the network in the same
script directory.  This port keeps the neural architecture reusable and accepts
an optional body decoder callable for projects that have licensed SMPL assets
available locally.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distance.

    Args:
        src: Source points with shape ``[B, N, C]``.
        dst: Target points with shape ``[B, M, C]``.

    Returns:
        Distances with shape ``[B, N, M]``.
    """

    return (
        torch.sum(src.square(), dim=-1, keepdim=True)
        + torch.sum(dst.square(), dim=-1).unsqueeze(1)
        - 2.0 * torch.matmul(src, dst.transpose(1, 2))
    )


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather point features using batched indices."""

    raw_shape = idx.shape
    flat_idx = idx.reshape(raw_shape[0], -1)
    gathered = torch.gather(
        points,
        dim=1,
        index=flat_idx.unsqueeze(-1).expand(-1, -1, points.shape[-1]),
    )
    return gathered.reshape(*raw_shape, points.shape[-1])


def point_ball_set(nsample: int, xyz: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Return nearest-neighbor indices for each anchor.

    The upstream function sorts all distances and keeps the first ``nsample``
    points.  ``topk`` is equivalent here and handles small synthetic tests by
    capping ``k`` at the available point count.
    """

    if xyz.ndim != 3 or anchors.ndim != 3:
        raise ValueError("xyz and anchors must both have shape [B, N, 3]/[B, S, 3]")
    k = min(int(nsample), xyz.shape[1])
    if k <= 0:
        raise ValueError("nsample and xyz point count must be positive")
    distances = square_distance(anchors, xyz)
    return torch.topk(distances, k=k, dim=-1, largest=False, sorted=True).indices


def anchor_init(
    x_min: float = -0.3,
    x_max: float = 0.3,
    x_interval: float = 0.3,
    y_min: float = -0.3,
    y_max: float = 0.3,
    y_interval: float = 0.3,
    z_min: float = -0.3,
    z_max: float = 2.1,
    z_interval: float = 0.3,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create the fixed 3D anchor grid used by mmMesh."""

    xs = torch.arange(x_min, x_max + 0.5 * x_interval, x_interval, device=device, dtype=dtype)
    ys = torch.arange(y_min, y_max + 0.5 * y_interval, y_interval, device=device, dtype=dtype)
    zs = torch.arange(z_min, z_max + 0.5 * z_interval, z_interval, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    return torch.stack((xx, yy, zz), dim=-1)


def anchor_grouping(
    anchors: torch.Tensor,
    nsample: int,
    xyz: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    """Group point features around anchor locations.

    Returns ``[B, S, K, 6 + D]`` containing anchor coordinates, anchor-relative
    xyz offsets, and the gathered point features.
    """

    idx = point_ball_set(nsample, xyz, anchors)
    grouped_xyz = index_points(xyz, idx)
    grouped_points = index_points(points, idx)
    grouped_anchors = anchors.unsqueeze(2).expand(-1, -1, grouped_xyz.shape[2], -1)
    grouped_xyz_norm = grouped_xyz - grouped_anchors
    return torch.cat([grouped_anchors, grouped_xyz_norm, grouped_points], dim=-1)


class BasePointNet(nn.Module):
    """Per-point feature stem from mmMesh."""

    def __init__(self, input_dim: int = 6, keep_raw_dims: int = 4, feature_dim: int = 24) -> None:
        super().__init__()
        if input_dim < 3:
            raise ValueError("mmMesh expects at least xyz point coordinates")
        self.input_dim = int(input_dim)
        self.keep_raw_dims = min(int(keep_raw_dims), self.input_dim)
        self.feature_dim = int(feature_dim)
        self.output_dim = self.keep_raw_dims + self.feature_dim

        self.net = nn.Sequential(
            nn.Conv1d(self.input_dim, 8, kernel_size=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, self.feature_dim, kernel_size=1),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3:
            raise ValueError("points must have shape [B, N, C]")
        features = self.net(points.transpose(1, 2)).transpose(1, 2)
        return torch.cat((points[:, :, : self.keep_raw_dims], features), dim=-1)


class _AttentionPointNet(nn.Module):
    def __init__(self, in_channels: int, hidden: tuple[int, int, int]) -> None:
        super().__init__()
        h1, h2, h3 = hidden
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, h1, kernel_size=1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Conv1d(h1, h2, kernel_size=1),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Conv1d(h2, h3, kernel_size=1),
            nn.BatchNorm1d(h3),
            nn.ReLU(),
        )
        self.attn = nn.Linear(h3, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.net(x.transpose(1, 2)).transpose(1, 2)
        weights = torch.softmax(self.attn(features), dim=1)
        pooled = torch.sum(features * weights, dim=1)
        return pooled, weights


class GlobalPointNet(_AttentionPointNet):
    def __init__(self, in_channels: int) -> None:
        super().__init__(in_channels=in_channels, hidden=(32, 48, 64))


class AnchorPointNet(_AttentionPointNet):
    def __init__(self, in_channels: int) -> None:
        super().__init__(in_channels=in_channels, hidden=(32, 48, 64))


class AnchorVoxelNet(nn.Module):
    """3D convolution over the fixed anchor grid."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(64, 96, kernel_size=(3, 3, 3), padding=0),
            nn.BatchNorm3d(96),
            nn.ReLU(),
            nn.Conv3d(96, 128, kernel_size=(5, 1, 1), padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 64, kernel_size=(3, 1, 1), padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.net(x.permute(0, 4, 1, 2, 3))
        return x.reshape(batch_size, 64)


class GlobalRNN(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.loc_head = nn.Sequential(nn.Linear(hidden_dim, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor,
        c0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        global_vec, (hn, cn) = self.rnn(x, (h0, c0))
        global_loc = self.loc_head(global_vec)
        return global_vec, global_loc, hn, cn


class AnchorRNN(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor,
        c0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_vec, (hn, cn) = self.rnn(x, (h0, c0))
        return anchor_vec, hn, cn


class GlobalModule(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.pointnet = GlobalPointNet(feature_dim)
        self.rnn = GlobalRNN()

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor,
        c0: torch.Tensor,
        batch_size: int,
        length_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        point_vec, weights = self.pointnet(x)
        point_vec = point_vec.view(batch_size, length_size, 64)
        global_vec, global_loc, hn, cn = self.rnn(point_vec, h0, c0)
        return global_vec, global_loc, weights, hn, cn


class AnchorModule(nn.Module):
    def __init__(
        self,
        point_feature_dim: int,
        nsample: int = 8,
        template_points: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        template = anchor_init() if template_points is None else template_points.detach().clone().float()
        self.register_buffer("template_points", template)
        self.z_size, self.y_size, self.x_size, _ = template.shape
        self.anchor_size = self.z_size * self.y_size * self.x_size
        self.nsample = int(nsample)
        self.pointnet = AnchorPointNet(in_channels=point_feature_dim + 3)
        self.voxel = AnchorVoxelNet()
        self.rnn = AnchorRNN()

    def forward(
        self,
        x: torch.Tensor,
        global_loc: torch.Tensor,
        h0: torch.Tensor,
        c0: torch.Tensor,
        batch_size: int,
        length_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        global_loc = global_loc.view(batch_size * length_size, 1, 2).expand(-1, self.anchor_size, -1)
        anchors = self.template_points.view(1, self.anchor_size, 3).expand(batch_size * length_size, -1, -1).clone()
        anchors[:, :, :2] = anchors[:, :, :2] + global_loc

        grouped = anchor_grouping(anchors, self.nsample, xyz=x[..., :3], points=x[..., 3:])
        grouped = grouped.reshape(batch_size * length_size * self.anchor_size, grouped.shape[2], grouped.shape[3])
        voxel_points, attn_weights = self.pointnet(grouped)
        voxel_points = voxel_points.view(batch_size * length_size, self.z_size, self.y_size, self.x_size, 64)
        voxel_vec = self.voxel(voxel_points).view(batch_size, length_size, 64)
        anchor_vec, hn, cn = self.rnn(voxel_vec, h0, c0)
        return anchor_vec, attn_weights, hn, cn


def _rotation_6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to rotation matrices."""

    x_axis = F.normalize(x[..., :3], dim=-1)
    z_axis = F.normalize(torch.cross(x_axis, x[..., 3:], dim=-1), dim=-1)
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    return torch.stack((x_axis, y_axis, z_axis), dim=-1)


def _matrix_to_axis_angle(matrix: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Convert rotation matrices to axis-angle vectors."""

    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"matrix must end with [3,3], got {tuple(matrix.shape)}")
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    cos_angle = ((trace - 1.0) * 0.5).clamp(min=-1.0 + eps, max=1.0 - eps)
    angle = torch.acos(cos_angle)
    vector = torch.stack(
        (
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ),
        dim=-1,
    )
    sin_angle = torch.sin(angle)
    scale = angle / (2.0 * sin_angle).clamp_min(eps)
    small = angle < 1.0e-4
    scale = torch.where(small, torch.full_like(scale, 0.5), scale)
    return vector * scale.unsqueeze(-1)


def expand_mmmesh_rotmat_to_smpl_pose(rotmat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand mmMesh's 9 predicted rotations to SMPL global/body rotations.

    Upstream mmMesh predicts 9 rotations. Rotation 0 is applied outside SMPL as
    a world transform. The remaining rotations are inserted into a 24-joint SMPL
    pose with identity matrices for joints not predicted by mmMesh.
    """

    if rotmat.ndim != 5 or rotmat.shape[-3:] != (9, 3, 3):
        raise ValueError(f"rotmat must have shape [B, T, 9, 3, 3], got {tuple(rotmat.shape)}")

    batch_size, length_size = rotmat.shape[:2]
    flat = rotmat.reshape(batch_size * length_size, 9, 3, 3)
    eye = torch.eye(3, device=rotmat.device, dtype=rotmat.dtype).view(1, 1, 3, 3)
    identity = eye.expand(flat.shape[0], 1, 3, 3)
    full_pose = torch.cat(
        (
            identity,
            flat[:, 1:3],
            identity,
            flat[:, 3:5],
            identity.expand(-1, 10, -1, -1),
            flat[:, 5:9],
            identity.expand(-1, 4, -1, -1),
        ),
        dim=1,
    )
    return flat[:, 0], full_pose


class MMMeshSMPLDecoder(nn.Module):
    """Optional SMPL decoder for :class:`MMMeshModel`.

    The decoder uses the same pose expansion as upstream mmMesh, then applies
    the predicted world rotation and translation outside the SMPL body model.
    Licensed SMPL assets are not bundled; pass ``model_path`` or prebuilt male
    and female modules when assets are available.
    """

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        male_model: nn.Module | None = None,
        female_model: nn.Module | None = None,
        model_type: str = "smpl",
        gender_threshold: float = 0.5,
        joint_count: int | None = None,
        return_vertices: bool = True,
    ) -> None:
        super().__init__()
        if (male_model is None or female_model is None) and model_path is None:
            raise ValueError("MMMeshSMPLDecoder requires model_path or both male_model and female_model")
        self.model_type = str(model_type).strip().lower()
        self.gender_threshold = float(gender_threshold)
        self.joint_count = int(joint_count) if joint_count is not None else None
        self.return_vertices = bool(return_vertices)
        if male_model is None or female_model is None:
            try:
                import smplx
            except ImportError as exc:  # pragma: no cover - dependency declared in pyproject
                raise ImportError("MMMeshSMPLDecoder requires smplx when model_path is used") from exc
            path = str(Path(model_path).expanduser())
            male_model = smplx.create(model_path=path, model_type=model_type, gender="male")
            female_model = smplx.create(model_path=path, model_type=model_type, gender="female")
        self.male_model = male_model
        self.female_model = female_model

    @staticmethod
    def _decode_one(
        model: nn.Module,
        betas: torch.Tensor,
        full_pose: torch.Tensor,
        *,
        return_verts: bool,
        model_type: str,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if model_type == "smplx":
            global_orient = _matrix_to_axis_angle(full_pose[:, :1])
            body_pose = _matrix_to_axis_angle(full_pose[:, 1:22])
            pose2rot = True
            batch_size = int(betas.shape[0])
            zero_kwargs: dict[str, torch.Tensor] = {
                "jaw_pose": betas.new_zeros(batch_size, 3),
                "leye_pose": betas.new_zeros(batch_size, 3),
                "reye_pose": betas.new_zeros(batch_size, 3),
                "expression": betas.new_zeros(batch_size, int(getattr(model, "num_expression_coeffs", 10))),
            }
            if bool(getattr(model, "use_pca", False)):
                hand_dim = int(getattr(model, "num_pca_comps", 6))
                zero_kwargs["left_hand_pose"] = betas.new_zeros(batch_size, hand_dim)
                zero_kwargs["right_hand_pose"] = betas.new_zeros(batch_size, hand_dim)
            else:
                zero_kwargs["left_hand_pose"] = betas.new_zeros(batch_size, 15, 3)
                zero_kwargs["right_hand_pose"] = betas.new_zeros(batch_size, 15, 3)
        else:
            global_orient = full_pose[:, :1]
            body_pose = full_pose[:, 1:]
            pose2rot = False
            zero_kwargs = {}
        output = model(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            transl=None,
            pose2rot=pose2rot,
            return_verts=return_verts,
            **zero_kwargs,
        )
        vertices = getattr(output, "vertices", None)
        joints = getattr(output, "joints", None)
        if joints is None and isinstance(output, dict):
            joints = output.get("joints")
            vertices = output.get("vertices", vertices)
        if joints is None:
            raise KeyError("SMPL decoder output must expose joints")
        return vertices, joints

    def forward(
        self,
        rotmat: torch.Tensor,
        translation: torch.Tensor,
        betas: torch.Tensor,
        gender: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if translation.shape[:2] != rotmat.shape[:2] or translation.shape[-1] != 3:
            raise ValueError("translation must have shape [B, T, 3]")
        if betas.shape[:2] != rotmat.shape[:2] or betas.shape[-1] != 10:
            raise ValueError("betas must have shape [B, T, 10]")

        batch_size, length_size = rotmat.shape[:2]
        world_rot, full_pose = expand_mmmesh_rotmat_to_smpl_pose(rotmat)
        flat_betas = betas.reshape(batch_size * length_size, 10)
        flat_translation = translation.reshape(batch_size * length_size, 1, 3)
        flat_gender = gender.reshape(batch_size * length_size, -1)[:, 0]
        male_mask = flat_gender > self.gender_threshold
        female_mask = ~male_mask

        vertices_out: torch.Tensor | None = None
        joints_out: torch.Tensor | None = None
        for mask, body_model in ((male_mask, self.male_model), (female_mask, self.female_model)):
            if not bool(mask.any()):
                continue
            vertices, joints = self._decode_one(
                body_model,
                flat_betas[mask],
                full_pose[mask],
                return_verts=self.return_vertices,
                model_type=self.model_type,
            )
            if self.joint_count is not None:
                joints = joints[:, : self.joint_count]
            joints = torch.bmm(world_rot[mask], joints.transpose(1, 2)).transpose(1, 2) + flat_translation[mask]
            if joints_out is None:
                joints_out = joints.new_zeros((batch_size * length_size, joints.shape[1], 3))
            joints_out[mask] = joints

            if vertices is not None:
                vertices = torch.bmm(world_rot[mask], vertices.transpose(1, 2)).transpose(1, 2) + flat_translation[mask]
                if vertices_out is None:
                    vertices_out = vertices.new_zeros((batch_size * length_size, vertices.shape[1], 3))
                vertices_out[mask] = vertices

        if joints_out is None:
            raise RuntimeError("MMMeshSMPLDecoder did not decode any samples")
        result = {
            "joints": joints_out.view(batch_size, length_size, joints_out.shape[1], 3),
            "full_pose": full_pose.view(batch_size, length_size, 24, 3, 3),
            "world_rot": world_rot.view(batch_size, length_size, 3, 3),
        }
        if vertices_out is not None:
            result["vertices"] = vertices_out.view(batch_size, length_size, vertices_out.shape[1], 3)
        return result


class CombineModule(nn.Module):
    """Combine global and anchor temporal features into body parameters."""

    def __init__(self, rot_joints: int = 9, betas_dim: int = 10) -> None:
        super().__init__()
        self.rot_joints = int(rot_joints)
        self.betas_dim = int(betas_dim)
        out_dim = self.rot_joints * 6 + 3 + self.betas_dim + 1
        self.net = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, out_dim))

    def forward(
        self,
        global_vec: torch.Tensor,
        anchor_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, length_size, _ = global_vec.shape
        x = self.net(torch.cat((global_vec, anchor_vec), dim=-1))
        rot6d = x[:, :, : self.rot_joints * 6].reshape(batch_size, length_size, self.rot_joints, 6)
        rotmat = _rotation_6d_to_matrix(rot6d)
        start = self.rot_joints * 6
        translation = x[:, :, start : start + 3]
        betas = x[:, :, start + 3 : start + 3 + self.betas_dim]
        gender_logit = x[:, :, start + 3 + self.betas_dim :]
        return rotmat, translation, betas, gender_logit


@dataclass(frozen=True)
class MMMeshState:
    h_global: torch.Tensor
    c_global: torch.Tensor
    h_anchor: torch.Tensor
    c_anchor: torch.Tensor


class MMMeshModel(nn.Module):
    """mmMesh-style model with Wave2Body tensor contracts.

    Args:
        input_dim: Per-point input dimension. Upstream mmMesh uses 6.
        body_decoder: Optional callable ``(rotmat, translation, betas, gender)``
            returning a mapping or tuple with decoded vertices/joints.
    """

    def __init__(
        self,
        input_dim: int = 6,
        *,
        keep_raw_dims: int = 4,
        point_feature_dim: int = 24,
        anchor_nsample: int = 8,
        body_decoder: Callable[..., object] | None = None,
        decode_translation_mode: str = "translation",
    ) -> None:
        super().__init__()
        self.base = BasePointNet(input_dim=input_dim, keep_raw_dims=keep_raw_dims, feature_dim=point_feature_dim)
        self.global_module = GlobalModule(feature_dim=self.base.output_dim)
        self.anchor_module = AnchorModule(point_feature_dim=self.base.output_dim, nsample=anchor_nsample)
        self.combine = CombineModule()
        self.body_decoder = body_decoder
        mode = str(decode_translation_mode).strip().lower()
        if mode not in {"translation", "root_xy_plus_translation_z"}:
            raise ValueError("decode_translation_mode must be 'translation' or 'root_xy_plus_translation_z'")
        self.decode_translation_mode = mode

    @staticmethod
    def initial_state(
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> MMMeshState:
        zeros = torch.zeros((3, batch_size, 64), device=device, dtype=dtype)
        return MMMeshState(
            h_global=zeros.clone(),
            c_global=zeros.clone(),
            h_anchor=zeros.clone(),
            c_anchor=zeros.clone(),
        )

    def forward(
        self,
        mmwave_points: torch.Tensor,
        gender: torch.Tensor | None = None,
        state: MMMeshState | None = None,
    ) -> dict[str, torch.Tensor | MMMeshState | object]:
        if mmwave_points.ndim == 3:
            mmwave_points = mmwave_points.unsqueeze(1)
        if mmwave_points.ndim != 4:
            raise ValueError("mmwave_points must have shape [B, T, N, C] or [B, N, C]")

        batch_size, length_size, point_size, input_dim = mmwave_points.shape
        if gender is not None and gender.ndim == 2 and int(gender.shape[0]) == batch_size:
            gender = gender.unsqueeze(1).expand(batch_size, length_size, -1)
        flat = mmwave_points.reshape(batch_size * length_size, point_size, input_dim)
        features = self.base(flat)

        if state is None:
            state = self.initial_state(batch_size, device=mmwave_points.device, dtype=mmwave_points.dtype)

        global_vec, global_loc, global_weights, h_global, c_global = self.global_module(
            features,
            state.h_global,
            state.c_global,
            batch_size,
            length_size,
        )
        anchor_vec, anchor_weights, h_anchor, c_anchor = self.anchor_module(
            features,
            global_loc,
            state.h_anchor,
            state.c_anchor,
            batch_size,
            length_size,
        )
        rotmat, translation, betas, gender_logit = self.combine(global_vec, anchor_vec)
        gender_for_decode = gender if gender is not None else torch.round(torch.sigmoid(gender_logit))

        outputs: dict[str, torch.Tensor | MMMeshState | object] = {
            "pred_rotmat": rotmat,
            "pred_translation": translation,
            "pred_betas": betas,
            "pred_gender_logit": gender_logit,
            "pred_root_xy": global_loc,
            "global_attention": global_weights,
            "anchor_attention": anchor_weights,
            "state": MMMeshState(h_global=h_global, c_global=c_global, h_anchor=h_anchor, c_anchor=c_anchor),
        }

        if self.body_decoder is not None:
            decode_translation = translation
            if self.decode_translation_mode == "root_xy_plus_translation_z":
                decode_translation = torch.cat((global_loc, translation[..., 2:3]), dim=-1)
            outputs["pred_decode_translation"] = decode_translation
            body = self.body_decoder(rotmat, decode_translation, betas, gender_for_decode)
            outputs["body"] = body
            if isinstance(body, dict) and torch.is_tensor(body.get("joints")):
                joints = body["joints"]
                outputs["pred_pose"] = joints[:, -1] if joints.ndim == 4 else joints

        return outputs


def _as_tensor_from_body(body: object, key: str) -> torch.Tensor | None:
    if isinstance(body, dict):
        value = body.get(key)
        return value if torch.is_tensor(value) else None
    return None


class MMMeshLoss(nn.Module):
    """Training loss for the mmMesh port.

    The upstream code combines SMPL vertex/joint L1 losses with translation,
    shape, root-location, and gender terms.  This loss keeps the same terms
    optional so Wave2Body can train either the neural body-parameter heads only
    or the full SMPL-decoded path when local SMPL assets are available.
    """

    def __init__(
        self,
        *,
        rotmat_weight: float = 0.0,
        translation_weight: float = 1.0,
        root_xy_weight: float = 1.0,
        betas_weight: float = 0.1,
        gender_weight: float = 1.0,
        joints_weight: float = 1.0,
        vertices_weight: float = 0.001,
        gender_loss: str = "hinge",
    ) -> None:
        super().__init__()
        self.rotmat_weight = float(rotmat_weight)
        self.translation_weight = float(translation_weight)
        self.root_xy_weight = float(root_xy_weight)
        self.betas_weight = float(betas_weight)
        self.gender_weight = float(gender_weight)
        self.joints_weight = float(joints_weight)
        self.vertices_weight = float(vertices_weight)
        self.gender_loss = str(gender_loss).strip().lower()

    @staticmethod
    def _align_prediction_and_target(
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pred.shape == target.shape:
            return pred, target
        if (
            pred.ndim == target.ndim + 1
            and int(pred.shape[0]) == int(target.shape[0])
            and tuple(pred.shape[2:]) == tuple(target.shape[1:])
        ):
            return pred[:, -1], target
        if (
            pred.ndim == target.ndim
            and int(pred.shape[0]) == int(target.shape[0])
            and int(target.shape[1]) == 1
            and tuple(pred.shape[2:]) == tuple(target.shape[2:])
        ):
            return pred[:, -1], target[:, 0]
        return pred, target

    @staticmethod
    def _weighted_l1(
        losses: dict[str, torch.Tensor],
        total: torch.Tensor,
        name: str,
        pred: torch.Tensor | None,
        target: torch.Tensor | None,
        weight: float,
    ) -> torch.Tensor:
        if weight <= 0 or pred is None or target is None:
            return total
        pred, target = MMMeshLoss._align_prediction_and_target(pred, target)
        if pred.shape != target.shape:
            raise ValueError(f"{name} shape mismatch: {pred.shape} vs {target.shape}")
        loss = F.l1_loss(pred.float(), target.detach().float())
        losses[name] = loss
        return total + weight * loss

    def _gender_loss(self, pred_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.detach().to(device=pred_logit.device, dtype=pred_logit.dtype)
        if target.shape != pred_logit.shape:
            target = target.reshape_as(pred_logit)
        if self.gender_loss == "hinge":
            return F.relu(target + (1.0 - 2.0 * target) * pred_logit).mean()
        if self.gender_loss in {"bce", "bcewithlogits"}:
            return F.binary_cross_entropy_with_logits(pred_logit.float(), target.float())
        raise ValueError(f"Unsupported gender_loss: {self.gender_loss!r}")

    def forward(
        self,
        outputs: dict[str, Any],
        *,
        target_rotmat: torch.Tensor | None = None,
        target_translation: torch.Tensor | None = None,
        target_root_xy: torch.Tensor | None = None,
        target_betas: torch.Tensor | None = None,
        target_gender: torch.Tensor | None = None,
        target_joints: torch.Tensor | None = None,
        target_vertices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pred_translation = outputs.get("pred_translation")
        if not torch.is_tensor(pred_translation):
            raise KeyError("MMMeshLoss expects outputs['pred_translation']")

        total = pred_translation.new_zeros(())
        losses: dict[str, torch.Tensor] = {}
        total = self._weighted_l1(
            losses,
            total,
            "rotmat_loss",
            outputs.get("pred_rotmat") if torch.is_tensor(outputs.get("pred_rotmat")) else None,
            target_rotmat,
            self.rotmat_weight,
        )
        total = self._weighted_l1(
            losses,
            total,
            "translation_loss",
            pred_translation,
            target_translation,
            self.translation_weight,
        )
        total = self._weighted_l1(
            losses,
            total,
            "root_xy_loss",
            outputs.get("pred_root_xy") if torch.is_tensor(outputs.get("pred_root_xy")) else None,
            target_root_xy if target_root_xy is not None else (
                target_translation[..., :2] if target_translation is not None else None
            ),
            self.root_xy_weight,
        )
        total = self._weighted_l1(
            losses,
            total,
            "betas_loss",
            outputs.get("pred_betas") if torch.is_tensor(outputs.get("pred_betas")) else None,
            target_betas,
            self.betas_weight,
        )

        pred_gender_logit = outputs.get("pred_gender_logit")
        if self.gender_weight > 0 and torch.is_tensor(pred_gender_logit) and target_gender is not None:
            gender_loss = self._gender_loss(pred_gender_logit, target_gender)
            losses["gender_loss"] = gender_loss
            total = total + self.gender_weight * gender_loss

        body = outputs.get("body")
        pred_joints = _as_tensor_from_body(body, "joints")
        if pred_joints is None:
            pred_joints = _as_tensor_from_body(body, "pred_joints")
        pred_vertices = _as_tensor_from_body(body, "vertices")
        if pred_vertices is None:
            pred_vertices = _as_tensor_from_body(body, "pred_vertices")

        total = self._weighted_l1(losses, total, "joints_loss", pred_joints, target_joints, self.joints_weight)
        total = self._weighted_l1(
            losses,
            total,
            "vertices_loss",
            pred_vertices,
            target_vertices,
            self.vertices_weight,
        )

        if not losses:
            raise ValueError("No mmMesh loss terms were active; provide at least one target with nonzero weight.")
        losses["total_loss"] = total
        return losses


def create_mmmesh_model(config: dict[str, Any] | None = None, **overrides: Any) -> MMMeshModel:
    """Instantiate :class:`MMMeshModel` from a small config dictionary."""

    cfg: dict[str, Any] = dict(config or {})
    cfg.update(overrides)
    body_decoder = cfg.get("body_decoder")
    decoder_cfg = cfg.get("body_decoder_config")
    if body_decoder is None and isinstance(decoder_cfg, dict) and bool(decoder_cfg.get("enabled", False)):
        decoder_kwargs = dict(decoder_cfg)
        decoder_kwargs.pop("enabled", None)
        body_decoder = MMMeshSMPLDecoder(**decoder_kwargs)
        cfg["body_decoder"] = body_decoder
    allowed = {
        "input_dim",
        "keep_raw_dims",
        "point_feature_dim",
        "anchor_nsample",
        "body_decoder",
        "decode_translation_mode",
    }
    kwargs = {key: value for key, value in cfg.items() if key in allowed}
    return MMMeshModel(**kwargs)
