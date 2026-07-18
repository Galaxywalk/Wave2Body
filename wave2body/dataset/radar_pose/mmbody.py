# -*- coding: utf-8 -*-
"""
MMBody Dataset for Token Translation Training

Independent implementation that doesn't rely on old project's __getitem__.
This avoids custom CUDA point-sampling ops in DataLoader workers.
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import Any, Dict

from .base import BaseRadarPoseDataset
from .mmbody_compact import MMBodyCompactStore, is_mmbody_compact
from .point_sampling import pad_or_sample_points
from .root_alignment import apply_input_root_alignment


_MMESH_ROT_JOINTS = np.asarray([0, 1, 2, 4, 5, 16, 17, 18, 19], dtype=np.int64)


def _mesh_key(mesh: np.lib.npyio.NpzFile, names: tuple[str, ...]) -> str | None:
    keys = {str(key).lower(): str(key) for key in mesh.files}
    for name in names:
        key = keys.get(name.lower())
        if key is not None:
            return key
    return None


def _mesh_value(mesh: np.lib.npyio.NpzFile, *names: str) -> Any | None:
    key = _mesh_key(mesh, tuple(names))
    return None if key is None else mesh[key]


def _float_vector(value: Any, *, length: int | None = None) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if length is not None:
        if arr.size < length:
            return None
        arr = arr[:length]
    return arr.astype(np.float32, copy=False)


def _gender_vector(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.dtype.kind in {"U", "S", "O"}:
        text = str(arr.reshape(-1)[0]).strip().lower()
        if text in {"m", "male", "man", "1", "true"}:
            return np.asarray([1.0], dtype=np.float32)
        if text in {"f", "female", "woman", "0", "false"}:
            return np.asarray([0.0], dtype=np.float32)
        return None
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return None
    return flat[:1].astype(np.float32, copy=False)


def _axis_angle_to_rotmat(axis_angle: np.ndarray) -> np.ndarray:
    vectors = np.asarray(axis_angle, dtype=np.float32).reshape(-1, 3)
    theta = np.linalg.norm(vectors, axis=1, keepdims=True)
    axis = np.divide(vectors, theta, out=np.zeros_like(vectors), where=theta > 1.0e-8)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zeros = np.zeros_like(x)
    k = np.stack(
        [
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros,
        ],
        axis=-1,
    ).reshape(-1, 3, 3)
    eye = np.eye(3, dtype=np.float32).reshape(1, 3, 3)
    sin = np.sin(theta).reshape(-1, 1, 1)
    cos = np.cos(theta).reshape(-1, 1, 1)
    rotmat = eye + sin * k + (1.0 - cos) * np.matmul(k, k)
    small = theta.reshape(-1) <= 1.0e-8
    if np.any(small):
        rotmat[small] = eye
    return rotmat.astype(np.float32, copy=False)


def _quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32).reshape(-1, 4)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = np.divide(q, norm, out=np.zeros_like(q), where=norm > 1.0e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(-1, 3, 3).astype(np.float32, copy=False)


def _select_mmmesh_rotmats(rotmat: np.ndarray) -> np.ndarray | None:
    flat = np.asarray(rotmat, dtype=np.float32).reshape(-1, 3, 3)
    if flat.shape[0] >= 24:
        return flat[_MMESH_ROT_JOINTS].astype(np.float32, copy=False)
    if flat.shape[0] >= 9:
        return flat[:9].astype(np.float32, copy=False)
    return None


def _rotmat_target(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.shape[-2:] == (3, 3):
        return _select_mmmesh_rotmats(arr)
    if arr.shape[-1:] == (4,):
        return _select_mmmesh_rotmats(_quat_to_rotmat(arr))
    if arr.size % 3 == 0:
        return _select_mmmesh_rotmats(_axis_angle_to_rotmat(arr))
    return None


def z_up_to_y_down(data):
    """
    Convert from Z-up to Y-down coordinate system.

    MMBody uses Z-up (head has positive Z, feet have negative Z).
    AMASS uses Y-down (head has negative Y, feet have positive Y).

    Transform: (X, Y, Z) -> (X, -Z, Y)

    Args:
        data: numpy array with shape [..., 3] or [..., >3] where first 3 dims are XYZ

    Returns:
        transformed data with same shape
    """
    transformed = data.copy()
    if data.shape[-1] >= 3:
        # -Z becomes new Y (so head has negative Y)
        # Y becomes new Z
        transformed[..., 1] = -data[..., 2]
        transformed[..., 2] = data[..., 1]
    return transformed


class MMBodyDataset(BaseRadarPoseDataset):
    """
    MMBody dataset for token translation training.

    Independent implementation that loads data directly without relying on
    old project's dataset class, avoiding CUDA multiprocessing issues.
    """

    ALL_TEST_SCENARIOS = ["lab1", "lab2", "furnished", "rain", "smoke", "poor_lighting", "occlusion"]
    ALL_TRAIN_SCENARIOS = [f"sequence_{i}" for i in range(20)]

    def __init__(self, config, split='train', vqvae_model=None, device=None):
        config = dict(config)
        data_format = dict(config.get("data_format", {}))
        data_format.setdefault("point_sampling", "random")
        data_format.setdefault("input_root_crop", True)
        config["data_format"] = data_format
        mmbody_cfg = dict(config.get("mmbody", {})) if isinstance(config.get("mmbody", {}), dict) else {}
        self.return_mmmesh_targets = bool(
            data_format.get("return_mmmesh_targets", mmbody_cfg.get("return_mmmesh_targets", False))
        )
        self.return_mmmesh_vertices = bool(
            data_format.get("return_mmmesh_vertices", mmbody_cfg.get("return_mmmesh_vertices", True))
        )
        self._compact_store = None
        self._compact_frame_keys = {}

        # Map split names: 'val' -> 'test' for MMBody dataset
        self._actual_split = 'test' if split == 'val' else split

        # Call parent init
        super().__init__(config, split, vqvae_model, device)
        if self._compact_store is not None and self.return_mmmesh_targets:
            raise ValueError("Compact mmBody stores only radar points and 22-joint poses; mmMesh targets are unavailable")

    def _load_data_list(self):
        """Build DataFrame of radar/mesh paths."""
        if is_mmbody_compact(self.root):
            return self._load_compact_data_list()
        scenarios = self._get_scenarios()
        split_path = os.path.join(self.root, self._actual_split)

        df_list = []
        if self._actual_split == "train":
            for seq in scenarios:
                seq_path = os.path.join(split_path, seq)
                if os.path.exists(seq_path):
                    df = self._load_sequence(seq_path)
                    if len(df) > 0:
                        df_list.append(df)
        else:  # test
            for scenario in scenarios:
                base = os.path.join(split_path, scenario)
                if not os.path.exists(base):
                    continue
                seq_dirs = sorted(
                    [d for d in os.listdir(base) if d.startswith("sequence_")],
                    key=lambda x: int(x.split("_")[-1])
                )
                for seq in seq_dirs:
                    df = self._load_sequence(os.path.join(base, seq))
                    if len(df) > 0:
                        df_list.append(df)

        if df_list:
            return pd.concat(df_list, ignore_index=True)
        return pd.DataFrame()

    def _load_compact_data_list(self):
        self._compact_store = MMBodyCompactStore(self.root)
        split_records = [
            record for record in self._compact_store.samples
            if str(record["split"]) == self._actual_split
        ]
        self._compact_frame_keys = {
            (str(record["scenario"]), str(record["sequence_name"]), int(record["frame"])): bytes(record["radar_key"])
            for record in split_records
        }
        scenario_list = list(self._get_scenarios())
        scenarios = set(scenario_list)
        scenario_order = {name: index for index, name in enumerate(scenario_list)}
        selected = [record for record in split_records if str(record["scenario"]) in scenarios]
        selected.sort(
            key=lambda record: (
                scenario_order[str(record["scenario"])],
                int(record["sequence"]),
                int(record["frame"]),
            )
        )
        if self._actual_split == "train" and self.used_data_quantity < 1.0:
            limited = []
            sequence_names = sorted(
                {str(record["sequence_name"]) for record in selected},
                key=lambda x: int(x.split("_")[-1]),
            )
            for sequence_name in sequence_names:
                sequence_records = [record for record in selected if str(record["sequence_name"]) == sequence_name]
                limited.extend(sequence_records[: int(len(sequence_records) * self.used_data_quantity)])
            selected = limited
        return pd.DataFrame(
            [
                {
                    "Sequence": int(record["sequence"]),
                    "SequenceName": str(record["sequence_name"]),
                    "Scenario": str(record["scenario"]),
                    "Frame": int(record["frame"]),
                    "RadarKey": bytes(record["radar_key"]),
                    "PoseKey": bytes(record["pose_key"]),
                }
                for record in selected
            ]
        )

    def _cache_extra_spec(self):
        return {
            "actual_split": str(self._actual_split),
            "val_scenes_cfg": self.config.get("val_dataset", {}).get("scenes", "all"),
            "return_mmmesh_targets": bool(getattr(self, "return_mmmesh_targets", False)),
            "return_mmmesh_vertices": bool(getattr(self, "return_mmmesh_vertices", True)),
        }

    def _get_scenarios(self):
        """Get scenario list based on split and config."""
        if self._actual_split == "train":
            return self.ALL_TRAIN_SCENARIOS

        scenes_cfg = self.config.get("val_dataset", {}).get("scenes", "all")
        if scenes_cfg == "all":
            return self.ALL_TEST_SCENARIOS
        if isinstance(scenes_cfg, list):
            return scenes_cfg
        return self.ALL_TEST_SCENARIOS

    def _load_sequence(self, seq_path):
        """Load radar/mesh paths for a sequence."""
        seq_name = os.path.basename(seq_path)
        seq_id = int(seq_name.split("_")[-1])
        parent_name = os.path.basename(os.path.dirname(seq_path))
        scenario_name = seq_name if parent_name == "train" else parent_name

        # Load radar files
        radar_dir = os.path.join(seq_path, "radar")
        if not os.path.exists(radar_dir):
            return pd.DataFrame()

        radar_files = sorted(
            [f for f in os.listdir(radar_dir) if f.endswith(".npy")],
            key=lambda x: int(x.split("_")[1].split(".")[0])
        )
        n_files = int(len(radar_files) * self.used_data_quantity)

        # Load mesh files
        mesh_dir = os.path.join(seq_path, "mesh")
        if not os.path.exists(mesh_dir):
            return pd.DataFrame()

        mesh_files = sorted(
            [f for f in os.listdir(mesh_dir) if f.endswith(".npz")],
            key=lambda x: int(x.split("_")[1].split(".")[0])
        )

        # Build DataFrame
        data = []
        for f in radar_files[:n_files]:
            frame = int(f.split("_")[1].split(".")[0])
            mesh_file = f"frame_{frame}.npz"
            if mesh_file in mesh_files:
                data.append({
                    "Sequence": seq_id,
                    "SequenceName": seq_name,
                    "Scenario": scenario_name,
                    "Frame": frame,
                    "Radar": os.path.join(radar_dir, f),
                    "Mesh": os.path.join(mesh_dir, mesh_file)
                })
        return pd.DataFrame(data)

    def _load_sample(self, idx):
        """Load radar frames and ground truth joints."""
        row = self.data_list.iloc[int(idx)]
        frame = row["Frame"]

        if self._compact_store is not None:
            gt_joints = self._compact_store.read_pose(bytes(row["PoseKey"]))[: self.num_joints]
            radar_list = []
            start = max(0, frame - self.merge_nframes // 2)
            end = start + self.merge_nframes
            scenario = str(row["Scenario"])
            sequence_name = str(row["SequenceName"])
            for frame_id in range(start, end):
                radar_key = self._compact_frame_keys.get((scenario, sequence_name, int(frame_id)))
                if radar_key is None:
                    continue
                data = self._compact_store.read_radar(radar_key)
                data[:, 3] *= 1e38
                data[:, 4] /= 100
                data = apply_input_root_alignment(data, root=gt_joints[0], cfg=self.root_alignment)
                radar_list.append(data)
            return radar_list, gt_joints

        # Load ground truth (22 joints)
        mesh = np.load(row["Mesh"])
        gt_joints = mesh["joints"][:self.num_joints].astype(np.float32)

        # Load merged radar frames
        radar_list = []
        start = max(0, frame - self.merge_nframes // 2)
        end = start + self.merge_nframes

        for f in range(start, end):
            path = row["Radar"].replace(f"frame_{frame}", f"frame_{f}")
            if os.path.exists(path):
                data = np.load(path).astype(np.float32)
                data[:, 3] *= 1e38  # Scale intensity (following mmdiff)
                data[:, 5] /= 100   # Scale velocity
                data = data[:, [0, 1, 2, 3, 5]]  # xyz, intensity, velocity

                data = apply_input_root_alignment(data, root=gt_joints[0], cfg=self.root_alignment)
                radar_list.append(data)

        return radar_list, gt_joints

    def close(self) -> None:
        if self._compact_store is not None:
            self._compact_store.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _load_mmmesh_targets(self, mesh_path: str) -> Dict[str, torch.Tensor]:
        """Load optional mmMesh supervision fields from an MMBody mesh npz."""
        result: Dict[str, torch.Tensor] = {}
        with np.load(mesh_path, allow_pickle=True) as mesh:
            joints = _mesh_value(mesh, "joints", "Joints", "target_joints")
            if joints is not None:
                target_joints = np.asarray(joints, dtype=np.float32)
                if target_joints.ndim >= 2 and target_joints.shape[-1] == 3:
                    target_joints = target_joints.reshape(-1, 3)[: self.num_joints]
                    result["target_joints"] = torch.from_numpy(target_joints.copy()).float()
                    if target_joints.shape[0] > 0:
                        result["target_root_xy"] = torch.from_numpy(target_joints[0, :2].copy()).float()

            if getattr(self, "return_mmmesh_vertices", True):
                vertices = _mesh_value(mesh, "vertices", "Vertices", "verts", "v", "target_vertices")
                if vertices is not None:
                    target_vertices = np.asarray(vertices, dtype=np.float32)
                    if target_vertices.ndim >= 2 and target_vertices.shape[-1] == 3:
                        result["target_vertices"] = torch.from_numpy(target_vertices.reshape(-1, 3).copy()).float()

            translation = _float_vector(
                _mesh_value(mesh, "trans", "translation", "target_translation", "root_translation"),
                length=3,
            )
            if translation is not None:
                result["target_translation"] = torch.from_numpy(translation.copy()).float()
                result["target_root_xy"] = torch.from_numpy(translation[:2].copy()).float()

            betas = _float_vector(_mesh_value(mesh, "betas", "beta", "shape", "Shape", "target_betas"), length=10)
            if betas is not None:
                result["target_betas"] = torch.from_numpy(betas.copy()).float()

            gender = _gender_vector(_mesh_value(mesh, "gender", "Gender", "sex", "target_gender"))
            if gender is not None:
                result["target_gender"] = torch.from_numpy(gender.copy()).float()

            rotmat = _rotmat_target(
                _mesh_value(
                    mesh,
                    "pquat",
                    "pmat",
                    "rotmat",
                    "rotation",
                    "rotations",
                    "pose_rotmat",
                    "pose",
                    "poses",
                    "target_rotmat",
                )
            )
            if rotmat is not None:
                result["target_rotmat"] = torch.from_numpy(rotmat.copy()).float()

        return result

    def __len__(self):
        if isinstance(self.data_list, pd.DataFrame):
            return len(self.data_list)
        return len(self.data_list) if self.data_list is not None else 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary with:
        - mmwave_points: [n_points, input_channels] point cloud tensor
        - target_poses: [num_joints, 3] ground truth pose
        """
        row = self.data_list.iloc[int(idx)]
        cached = self._get_cached_item(idx)
        if cached is not None:
            result = dict(cached)
            if getattr(self, "return_mmmesh_targets", False):
                result.update(self._load_mmmesh_targets(str(row["Mesh"])))
            result.update(
                {
                    'mmbody_scenario': str(row.get("Scenario", "unknown")),
                    'mmbody_sequence': str(row.get("SequenceName", f"sequence_{int(row['Sequence'])}")),
                    'mmbody_frame': int(row.get("Frame", idx)),
                }
            )
            return result

        radar_list, gt_joints = self._load_sample(idx)

        # Merge frames
        if len(radar_list) > 0:
            radar_data = np.concatenate(radar_list, axis=0)
        else:
            radar_data = np.zeros((0, self.input_channels), dtype=np.float32)

        gt_joints_absolute = gt_joints.copy()

        # Normalize gt_joints (pelvis-centered)
        if self.target_root_normalized:
            pelvis = gt_joints[0]
            gt_joints = gt_joints - pelvis

        # Convert to tensor
        gt_tensor = torch.from_numpy(gt_joints).float()
        gt_absolute_tensor = torch.from_numpy(gt_joints_absolute).float()
        radar_tensor = torch.from_numpy(radar_data).float()

        # Process points (downsample/pad to n_points) - CPU only, no FPS
        radar_tensor = self._process_points_cpu(radar_tensor)

        # Build output dictionary
        result = {
            'mmwave_points': radar_tensor,
            'target_poses': gt_tensor,
            'target_poses_absolute': gt_absolute_tensor,
            'mmbody_scenario': str(row.get("Scenario", "unknown")),
            'mmbody_sequence': str(row.get("SequenceName", f"sequence_{int(row['Sequence'])}")),
            'mmbody_frame': int(row.get("Frame", idx)),
        }
        if getattr(self, "return_mmmesh_targets", False):
            result.update(self._load_mmmesh_targets(str(row["Mesh"])))

        return result

    def _process_points_cpu(self, points: torch.Tensor) -> torch.Tensor:
        """
        Downsample or pad point cloud to n_points.
        MMBody defaults to random sampling for historical compatibility, but
        honors data_format.point_sampling for diagnostics and new recipes.
        """
        return pad_or_sample_points(
            points,
            n_points=self.n_points,
            input_channels=self.input_channels,
            strategy=self.point_sampling,
        )
