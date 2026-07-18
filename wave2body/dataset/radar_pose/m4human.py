# -*- coding: utf-8 -*-
"""M4Human RF3DPose LMDB adapter for Wave2Body radar-pose training."""

from __future__ import annotations

import copy
import gzip
import os
import pickle
import struct
from pathlib import Path
from typing import Any

import lmdb
import msgpack
import numpy as np
import smplx
import torch
from torch.utils.data import Dataset

from ...utils.paths import resolve_dataset_uri
from .point_sampling import VALID_POINT_SAMPLING_STRATEGIES, pad_or_sample_points
from .root_alignment import apply_input_root_alignment, resolve_root_alignment_config


VALID_SPLIT_IDS = ("s1", "s2", "s3")
VALID_SUBSETS = ("train", "val", "test")
M4HUMAN_TEMPORAL_OFFSET_MODES: dict[str, tuple[int, ...]] = {
    "current1": (0,),
    "hist2": (-1, 0),
    "hist4": (-3, -2, -1, 0),
    "hist5": (-4, -3, -2, -1, 0),
    "hist8": (-7, -6, -5, -4, -3, -2, -1, 0),
    "center5_oracle": (-2, -1, 0, 1, 2),
}

GENDER_INFO = {
    "P1": 0, "P3": 0, "P4": 0, "P7": 0, "P8": 0, "P10": 0, "P13": 0, "P15": 0,
    "P2": 1, "P5": 1, "P6": 1, "P9": 1, "P11": 1, "P12": 1, "P14": 1, "P16": 1,
    "P17": 1, "P18": 1, "P19": 1, "P20": 1,
}

NON_VALID_INDICATOR = [
    [1, 12, 58], [1, 13, 491], [1, 31, 324], [1, 36, 284], [1, 37, 43],
    [5, 43, 551], [5, 43, 552], [5, 47, 136], [5, 47, 140],
]
NON_VALID_INDICATOR_SET = {tuple(x) for x in NON_VALID_INDICATOR}
_LMDB_ENV_CACHE: dict[tuple[int, str], tuple[lmdb.Environment, int]] = {}


def _acquire_lmdb_env(path: Path) -> lmdb.Environment:
    key = (os.getpid(), str(path))
    cached = _LMDB_ENV_CACHE.get(key)
    if cached is not None:
        env, refcount = cached
        _LMDB_ENV_CACHE[key] = (env, refcount + 1)
        return env
    env = lmdb.open(str(path), readonly=True, subdir=False, lock=False, readahead=False)
    _LMDB_ENV_CACHE[key] = (env, 1)
    return env


def _release_lmdb_env(path: Path) -> None:
    key = (os.getpid(), str(path))
    cached = _LMDB_ENV_CACHE.get(key)
    if cached is None:
        return
    env, refcount = cached
    if refcount <= 1:
        env.close()
        _LMDB_ENV_CACHE.pop(key, None)
        return
    _LMDB_ENV_CACHE[key] = (env, refcount - 1)


def _parse_temporal_offsets(fmt: dict, m4: dict) -> tuple[str, tuple[int, ...]]:
    """Resolve explicit M4Human frame offsets.

    The M4Human default is history-only with offsets ``[-T+1, ..., 0]``.
    Explicit modes compare causal and oracle windows while keeping that default
    unchanged.
    """

    raw_offsets = m4.get("temporal_offsets", fmt.get("temporal_offsets"))
    raw_mode = m4.get("frame_mode", m4.get("temporal_mode", fmt.get("frame_mode", fmt.get("temporal_mode"))))
    if raw_offsets is not None:
        offsets = tuple(int(x) for x in raw_offsets)
        if not offsets:
            raise ValueError("M4Human temporal_offsets must not be empty")
        return str(raw_mode or "custom"), offsets

    if raw_mode not in (None, "", "default"):
        mode = str(raw_mode).strip().lower()
        if mode not in M4HUMAN_TEMPORAL_OFFSET_MODES:
            raise ValueError(
                f"Unknown M4Human temporal mode {raw_mode!r}; "
                f"expected one of {sorted(M4HUMAN_TEMPORAL_OFFSET_MODES)}"
            )
        return mode, M4HUMAN_TEMPORAL_OFFSET_MODES[mode]

    temporal_window = int(fmt.get("merge_nframes", m4.get("temporal_window", 4)))
    if temporal_window <= 0:
        raise ValueError(f"M4Human temporal_window/merge_nframes must be positive, got {temporal_window}")
    return f"hist{temporal_window}", tuple(range(-temporal_window + 1, 1))


def _decode_np(obj: Any) -> Any:
    if isinstance(obj, dict) and obj.get("__nd__") is True:
        arr = np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"]))
        return arr.reshape(tuple(obj["shape"]))
    if isinstance(obj, dict):
        return {k: _decode_np(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_np(v) for v in obj]
    return obj


def _unpack_dict_np(blob: bytes) -> dict:
    return _decode_np(msgpack.unpackb(blob, raw=False))


def _unpack_ndarray(blob: bytes) -> np.ndarray:
    if len(blob) < 4:
        raise ValueError(f"Invalid ndarray blob: too short for ndim header ({len(blob)} bytes)")
    offset = 0
    ndim = struct.unpack_from("=I", blob, offset)[0]
    if ndim <= 0 or ndim > 8:
        raise ValueError(f"Invalid ndarray blob: unsupported ndim={ndim}")
    offset += 4
    header_size = offset + 4 * ndim
    if len(blob) < header_size:
        raise ValueError(f"Invalid ndarray blob: too short for shape header ndim={ndim}")
    shape = struct.unpack_from("=" + "I" * ndim, blob, offset)
    offset += 4 * ndim
    expected = int(np.prod(shape, dtype=np.int64))
    arr = np.frombuffer(blob, dtype=np.float32, offset=offset)
    if arr.size != expected:
        raise ValueError(f"Invalid ndarray blob: expected {expected} float32 values for shape={shape}, got {arr.size}")
    return arr.reshape(shape)


def _rodrigues(rvec: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rvec)
    if theta < 1e-8:
        return np.eye(3)
    r = rvec / theta
    k = np.array([
        [0.0, -r[2], r[1]],
        [r[2], 0.0, -r[0]],
        [-r[1], r[0], 0.0],
    ])
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def _inv_rodrigues(rot: np.ndarray) -> np.ndarray:
    theta = np.arccos(np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0))
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([
        rot[2, 1] - rot[1, 2],
        rot[0, 2] - rot[2, 0],
        rot[1, 0] - rot[0, 1],
    ]) / (2.0 * np.sin(theta))
    return (axis * theta).astype(np.float32)


def _calibrate_param_to_radar(param: dict, calib: dict) -> dict:
    root_rot = _rodrigues(param["root_orient"])
    cam_rot = calib["vicon_to_cam_rotmatrix"] @ root_rot
    radar_rot = np.linalg.inv(calib["radar_to_cam_rotmatrix"]) @ cam_rot

    cam_trans = (
        calib["vicon_to_cam_rotmatrix"] @ param["joints"][0]
        + calib["vicon_to_cam_tvec"] / 1000.0
    )
    radar_trans = np.linalg.inv(calib["radar_to_cam_rotmatrix"]) @ (
        cam_trans - calib["radar_to_cam_tvec"]
    )

    out = copy.deepcopy(param)
    out["root_orient"] = _inv_rodrigues(radar_rot)
    out["trans"] = radar_trans + (param["trans"] - param["joints"][0])
    return out


class M4HumanLMDBPoseDataset(Dataset):
    """
    Read M4Human RF3DPose LMDB samples and expose Wave2Body's radar-pose input
    contract.

    Returns:
        mmwave_points: [n_points, 4] float tensor, xyz + intensity
        target_poses: [num_joints, 3] float tensor in meters, pelvis-centered when normalized
        target_poses_absolute: [num_joints, 3] float tensor in the same radar-normalized
            frame as mmwave_points for support diagnostics
    """

    def __init__(self, config: dict, split: str = "train", device: str | None = None):
        super().__init__()
        if split not in VALID_SUBSETS:
            raise ValueError(f"split must be train/val/test, got {split}")
        self.config = config
        self.split = split
        self.device = device or "cpu"

        fmt = config.get("data_format", {})
        m4 = config.get("m4human", {})
        self.n_points = int(fmt.get("n_points", 1024))
        if self.n_points <= 0:
            raise ValueError(f"data_format.n_points must be positive, got {self.n_points}")
        self.input_channels = int(fmt.get("radar_input_c", 4))
        if self.input_channels != 4:
            raise ValueError("M4Human radar_points are xyz+intensity; set radar_input_c: 4")
        self.num_joints = int(fmt.get("num_joints", 22))
        if self.num_joints <= 0:
            raise ValueError(f"data_format.num_joints must be positive, got {self.num_joints}")
        self.temporal_mode, self.temporal_offsets = _parse_temporal_offsets(fmt, m4)
        self.temporal_window = len(self.temporal_offsets)
        self.root_alignment = resolve_root_alignment_config(
            fmt,
            default_input_root_normalized=False,
            default_input_root_crop=False,
        )
        self.normalized = self.root_alignment.normalized
        self.input_root_normalized = self.root_alignment.input_root_normalized
        self.target_root_normalized = self.root_alignment.target_root_normalized
        self.input_root_crop = self.root_alignment.input_root_crop
        self.input_root_crop_x_range = self.root_alignment.input_root_crop_x_range
        self.input_root_crop_y_range = self.root_alignment.input_root_crop_y_range
        self.input_root_crop_z_range = self.root_alignment.input_root_crop_z_range
        self.normalize_z = bool(m4.get("normalize_z", True))
        self.intensity_scale = float(m4.get("intensity_scale", 1.0))
        self.random_sample_train = bool(m4.get("random_sample_train", True))
        self.point_sampling = str(fmt.get("point_sampling", "")).strip().lower()
        if self.point_sampling and self.point_sampling not in VALID_POINT_SAMPLING_STRATEGIES:
            raise ValueError(
                f"data_format.point_sampling must be one of {sorted(VALID_POINT_SAMPLING_STRATEGIES)}, "
                f"got {self.point_sampling!r}"
            )
        self.load_pose = bool(m4.get("load_pose", True))
        self.return_temporal_frames = bool(m4.get("return_temporal_frames", False))
        self.pose_source = str(m4.get("pose_source", "smplx")).strip().lower()
        if self.pose_source not in {"smplx", "param_joints"}:
            raise ValueError("m4human.pose_source must be one of {'smplx', 'param_joints'}")
        self.cache_mode = "none"

        if any(key in m4 for key in ("scale", "scale_id", "protocol")) or any(
            key in config for key in ("scale", "scale_id", "protocol")
        ):
            raise ValueError("M4Human scale/protocol options are no longer supported; remove them from the config.")
        self.split_id = str(m4.get("split", config.get("split_id", "s3")))
        if self.split_id not in VALID_SPLIT_IDS:
            raise ValueError(f"m4human.split must be one of {VALID_SPLIT_IDS}, got {self.split_id!r}")
        self.root = resolve_dataset_uri(str(config.get("dataset_path") or m4.get("cached_root") or ""))
        if not self.root:
            raise ValueError("Set dataset_path or m4human.cached_root to the M4Human dataset root")
        self.cache_path = Path(self.root) / "rf3dpose_all"
        self.smplx_model_root = Path(resolve_dataset_uri(str(m4.get("smplx_model_root", Path(self.root) / "models"))))

        self.env_paths = {
            "radar_pc": self.cache_path / "radar_pc.lmdb",
            "params": self.cache_path / "params.lmdb",
            "calib": self.cache_path / "calib.lmdb",
        }
        missing = [str(p) for p in self.env_paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing M4Human LMDB files: {missing}")

        self.lmdb_envs: dict[str, lmdb.Environment] | None = None
        self._lmdb_owner_pid: int | None = None
        self.smplx_models: dict[str, torch.nn.Module] | None = None
        self._smplx_owner_pid: int | None = None
        self.data_list = self._load_split_indices()

        max_samples_cfg = m4.get("max_samples", {})
        max_samples = None
        if isinstance(max_samples_cfg, dict):
            max_samples = max_samples_cfg.get(split)
        elif max_samples_cfg not in (None, "", "null"):
            max_samples = max_samples_cfg
        if max_samples is not None:
            max_samples = int(max_samples)
            if max_samples < 0:
                raise ValueError(f"m4human.max_samples must be non-negative, got {max_samples}")
            self.data_list = self.data_list[:max_samples]
        self._temporal_index: dict[tuple[int, int, int], list[int]] = {}
        self._temporal_sequences: dict[tuple[int, int], list[int]] = {}
        self._build_temporal_lookup()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["lmdb_envs"] = None
        state["_lmdb_owner_pid"] = None
        state["smplx_models"] = None
        state["_smplx_owner_pid"] = None
        return state

    def close(self) -> None:
        if self.lmdb_envs is not None:
            if self._lmdb_owner_pid == os.getpid():
                for name in list(self.lmdb_envs):
                    _release_lmdb_env(self.env_paths[name])
        self.lmdb_envs = None
        self._lmdb_owner_pid = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _open_lmdb_envs(self) -> None:
        for name in self.env_paths:
            self._ensure_lmdb_env(name)

    def _ensure_lmdb_env(self, name: str) -> lmdb.Environment:
        if name not in self.env_paths:
            raise KeyError(f"Unknown M4Human LMDB env: {name}")
        cur_pid = os.getpid()
        if self.lmdb_envs is None or self._lmdb_owner_pid != cur_pid:
            self.lmdb_envs = {}
            self._lmdb_owner_pid = cur_pid
        if name not in self.lmdb_envs:
            self.lmdb_envs[name] = _acquire_lmdb_env(self.env_paths[name])
        return self.lmdb_envs[name]

    def _ensure_smplx_models(self) -> None:
        cur_pid = os.getpid()
        if self.smplx_models is not None and self._smplx_owner_pid == cur_pid:
            return
        if not self.smplx_model_root.exists():
            raise FileNotFoundError(f"Missing SMPL-X model root for M4Human: {self.smplx_model_root}")

        models: dict[str, torch.nn.Module] = {}
        for gender in ("male", "female"):
            try:
                model = smplx.create(
                    model_path=str(self.smplx_model_root),
                    model_type="smplx",
                    gender=gender,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load {gender} SMPL-X model from {self.smplx_model_root}"
                ) from exc
            model.eval()
            models[gender] = model
        self.smplx_models = models
        self._smplx_owner_pid = cur_pid

    def _load_split_indices(self) -> list[list[int]]:
        indices_path = self.cache_path / "indeces.pkl.gz"
        if not indices_path.exists():
            raise FileNotFoundError(f"Missing split index file: {indices_path}")
        with gzip.open(indices_path, "rb") as f:
            split_indices = pickle.load(f)
        candidates: list[tuple[int, str, list[list[int]]]] = []
        for route_key, route_splits in split_indices.items():
            try:
                raw_indicators = route_splits[self.split_id][self.split]
            except KeyError:
                continue
            indicators = [
                list(x)
                for x in raw_indicators
                if tuple(x) not in NON_VALID_INDICATOR_SET
            ]
            candidates.append((len(indicators), str(route_key), indicators))
        if not candidates:
            raise KeyError(
                f"Unknown M4Human split={self.split_id}, subset={self.split}"
            )
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        _, source_key, indicators = candidates[0]
        self.source_index_key = source_key

        return indicators

    def _build_temporal_lookup(self) -> None:
        temporal_index: dict[tuple[int, int, int], list[int]] = {}
        temporal_sequences: dict[tuple[int, int], list[int]] = {}
        for indicator in self.data_list:
            sub_id, action_id, frame_id = (int(indicator[0]), int(indicator[1]), int(indicator[2]))
            key = (sub_id, action_id, frame_id)
            temporal_index[key] = [sub_id, action_id, frame_id]
            temporal_sequences.setdefault((sub_id, action_id), []).append(frame_id)
        for key, frames in temporal_sequences.items():
            temporal_sequences[key] = sorted(set(frames))
        self._temporal_index = temporal_index
        self._temporal_sequences = temporal_sequences

    def _ensure_temporal_lookup(self) -> None:
        if not getattr(self, "_temporal_index", None) or not getattr(self, "_temporal_sequences", None):
            self._build_temporal_lookup()

    def _resolve_temporal_indicator(self, indicator: list[int], offset: int) -> list[int]:
        self._ensure_temporal_lookup()
        sub_id, action_id, current_frame = (int(indicator[0]), int(indicator[1]), int(indicator[2]))
        target_frame = current_frame + int(offset)
        exact = self._temporal_index.get((sub_id, action_id, target_frame))
        if exact is not None:
            return exact

        frames = self._temporal_sequences.get((sub_id, action_id), [])
        if not frames:
            return [sub_id, action_id, current_frame]

        if target_frame <= frames[0]:
            nearest = frames[0]
        elif target_frame >= frames[-1]:
            nearest = frames[-1]
        else:
            nearest = min(frames, key=lambda frame: (abs(frame - target_frame), frame))
        return self._temporal_index[(sub_id, action_id, int(nearest))]

    def __len__(self) -> int:
        return len(self.data_list)

    def _read_radar_pc(self, indicator: list[int]) -> np.ndarray:
        env = self._ensure_lmdb_env("radar_pc")
        with env.begin() as txn:
            blob = txn.get(str(indicator).encode())
        if blob is None:
            raise KeyError(f"Missing radar_pc for indicator {indicator}")
        radar = _unpack_ndarray(blob).astype(np.float32, copy=True)
        if radar.ndim != 2 or radar.shape[1] < self.input_channels:
            raise ValueError(
                f"Invalid radar_pc shape for indicator {indicator}: expected [N,>={self.input_channels}], got {radar.shape}"
            )
        return radar

    def _read_params_and_calib(self, indicator: list[int]) -> tuple[dict, dict]:
        params_env = self._ensure_lmdb_env("params")
        calib_env = self._ensure_lmdb_env("calib")
        with params_env.begin() as txn_param, calib_env.begin() as txn_calib:
            param_blob = txn_param.get(str(indicator).encode())
            calib_blob = txn_calib.get(str(indicator).encode())
        if param_blob is None or calib_blob is None:
            raise KeyError(f"Missing params/calib for indicator {indicator}")
        return _unpack_dict_np(param_blob), _unpack_dict_np(calib_blob)

    def _resolve_temporal_indicators(self, idx: int) -> list[list[int]]:
        indicator = self.data_list[int(idx)]
        selected: list[list[int]] = []
        for offset in self.temporal_offsets:
            resolved = self._resolve_temporal_indicator(indicator, int(offset))
            selected.append(resolved)
        return selected

    def _load_temporal_radar(self, idx: int) -> list[np.ndarray]:
        selected = []
        for resolved in self._resolve_temporal_indicators(idx):
            selected.append(self._read_radar_pc(resolved).copy())
        return selected

    def _load_joints(self, indicator: list[int]) -> np.ndarray:
        params, calib = self._read_params_and_calib(indicator)
        if self.pose_source == "param_joints":
            joints = np.asarray(params["joints"], dtype=np.float32)
            if joints.ndim != 2 or joints.shape[1] < 3 or joints.shape[0] < self.num_joints:
                raise ValueError(
                    f"Invalid params.joints for indicator {indicator}: "
                    f"expected [>={self.num_joints},3], got {joints.shape}"
                )
            vicon_to_cam_rot = np.asarray(calib["vicon_to_cam_rotmatrix"], dtype=np.float32)
            vicon_to_cam_t = np.asarray(calib["vicon_to_cam_tvec"], dtype=np.float32) / 1000.0
            radar_to_cam_rot = np.asarray(calib["radar_to_cam_rotmatrix"], dtype=np.float32)
            radar_to_cam_t = np.asarray(calib["radar_to_cam_tvec"], dtype=np.float32)
            joints_cam = (vicon_to_cam_rot @ joints[:, :3].T).T + vicon_to_cam_t
            joints_radar = (np.linalg.inv(radar_to_cam_rot) @ (joints_cam - radar_to_cam_t).T).T
            if self.normalize_z:
                joints_radar = joints_radar.copy()
                joints_radar[:, 2] -= 1.5
            return joints_radar[: self.num_joints].astype(np.float32, copy=False)

        subject_key = f"P{indicator[0]}"
        if subject_key not in GENDER_INFO:
            raise KeyError(f"Unknown M4Human subject gender for {subject_key}")
        params["gender"] = GENDER_INFO[subject_key]
        radar_params = _calibrate_param_to_radar(params, calib)
        if self.normalize_z:
            radar_params["trans"] = np.asarray(radar_params["trans"], dtype=np.float32).copy()
            radar_params["trans"][..., 2] -= 1.5

        self._ensure_smplx_models()
        assert self.smplx_models is not None
        gender = "male" if float(radar_params["gender"]) > 0.5 else "female"
        smplx_model = self.smplx_models[gender]
        with torch.no_grad():
            output = smplx_model(
                betas=torch.as_tensor(radar_params["betas"][None], dtype=torch.float32),
                body_pose=torch.as_tensor(radar_params["pose_body"][None], dtype=torch.float32),
                global_orient=torch.as_tensor(radar_params["root_orient"][None], dtype=torch.float32),
                transl=torch.as_tensor(radar_params["trans"][None], dtype=torch.float32),
            )
            vertices_mm = output.vertices.detach().squeeze(0) * 1000.0
            regressor = smplx_model.J_regressor[: self.num_joints].to(vertices_mm.device)
            joints_m = torch.einsum("ij,jc->ic", regressor, vertices_mm) / 1000.0
        return joints_m.cpu().numpy().astype(np.float32)

    def _process_points(self, points: torch.Tensor) -> torch.Tensor:
        if points.numel() == 0:
            return torch.zeros((self.n_points, self.input_channels), dtype=torch.float32)

        valid = torch.any(torch.abs(points) > 0, dim=1)
        points = points[valid]
        if points.numel() == 0:
            return torch.zeros((self.n_points, self.input_channels), dtype=torch.float32)

        strategy = getattr(self, "point_sampling", "")
        if not strategy:
            strategy = "random" if self.split == "train" and self.random_sample_train else "linspace"
        return pad_or_sample_points(
            points,
            n_points=self.n_points,
            input_channels=self.input_channels,
            strategy=strategy,
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        indicator = self.data_list[int(idx)]
        joints = self._load_joints(indicator) if getattr(self, "load_pose", True) else None
        root = joints[0] if joints is not None and joints.shape[0] > 0 else None
        subject_key = f"P{int(indicator[0])}"
        gender_value = float(GENDER_INFO.get(subject_key, 0))
        if hasattr(self, "temporal_offsets"):
            temporal_indicators = self._resolve_temporal_indicators(int(idx))
            radar_frames = [self._read_radar_pc(resolved).copy() for resolved in temporal_indicators]
        else:
            temporal_indicators = [indicator]
            radar_frames = self._load_temporal_radar(int(idx))
        processed_frames = []
        for frame in radar_frames:
            frame = frame.astype(np.float32, copy=True)
            if frame.ndim != 2:
                raise ValueError(f"Expected 2D radar frame, got shape={frame.shape}")
            if frame.shape[1] < self.input_channels:
                raise ValueError(f"Expected at least {self.input_channels} radar channels, got {frame.shape}")
            frame = frame[:, : self.input_channels]
            if self.normalize_z and frame.size:
                frame[:, 2] -= 1.5
            if self.intensity_scale != 1.0 and frame.size:
                frame[:, 3] *= self.intensity_scale
            cfg = getattr(self, "root_alignment", None)
            if cfg is not None:
                frame = apply_input_root_alignment(frame, root=root, cfg=cfg)
            processed_frames.append(frame)

        if getattr(self, "return_temporal_frames", False):
            frame_tensors = [self._process_points(torch.from_numpy(frame).float()) for frame in processed_frames]
            if frame_tensors:
                radar_tensor = torch.stack(frame_tensors, dim=0)
            else:
                radar_tensor = torch.zeros((0, self.n_points, self.input_channels), dtype=torch.float32)
        else:
            radar = np.concatenate(processed_frames, axis=0) if processed_frames else np.zeros((0, 4), dtype=np.float32)
            radar_tensor = self._process_points(torch.from_numpy(radar).float())

        if joints is not None:
            joints_absolute = joints.copy()
            if getattr(self, "target_root_normalized", getattr(self, "normalized", True)):
                joints = joints - joints[:1]
            target_tensor = torch.from_numpy(joints).float()
            target_absolute_tensor = torch.from_numpy(joints_absolute).float()
        else:
            target_tensor = torch.zeros((self.num_joints, 3), dtype=torch.float32)
            target_absolute_tensor = torch.zeros((self.num_joints, 3), dtype=torch.float32)

        return {
            "mmwave_points": radar_tensor,
            "target_poses": target_tensor,
            "target_poses_absolute": target_absolute_tensor,
            "target_gender": torch.tensor([gender_value], dtype=torch.float32),
            "indicator": torch.tensor(indicator, dtype=torch.long),
            "temporal_indicators": torch.tensor(temporal_indicators, dtype=torch.long),
        }
