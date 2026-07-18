# -*- coding: utf-8 -*-
"""M4Human joint-only dataset loader."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ...utils.paths import resolve_dataset_uri
from .registry import register_dataset


@register_dataset("m4human")
class M4HumanJointsDataset(Dataset):
    """Cached M4Human joint-only loader for VQ-VAE training."""

    def __init__(self, root_path: str, num_joints: int, device: str, label: int, name: str, **kwargs):
        self.root_path = Path(resolve_dataset_uri(str(root_path)))
        self.num_joints = int(num_joints)
        self.device = device
        self.label = label
        self.name = name
        if any(key in kwargs for key in ("scale", "scale_id", "protocol")):
            raise ValueError("M4Human scale/protocol options are no longer supported.")
        split_id = kwargs.pop("split_id", None)
        if split_id is None:
            split_id = kwargs.pop("split", "s3")
        self.split_id = str(split_id)
        self.subset = str(kwargs.pop("subset", "train"))
        self.normalize_z = bool(kwargs.pop("normalize_z", True))
        self.normalized = bool(kwargs.pop("normalized", True))
        self.temporal_window = int(kwargs.pop("temporal_window", 4))
        self.smplx_model_root = kwargs.pop("smplx_model_root", str(self.root_path / "models"))
        self.pose_source = str(kwargs.pop("pose_source", "smplx")).strip().lower()
        if self.pose_source not in {"param_joints", "smplx"}:
            raise ValueError("M4Human pose_source must be one of {'param_joints', 'smplx'}")
        self.cache_build_workers = max(0, int(kwargs.pop("cache_build_workers", 0) or 0))
        self.cache_build_chunk_size = max(1, int(kwargs.pop("cache_build_chunk_size", 4096) or 4096))
        self.cache_build_prefetch = max(
            1,
            int(kwargs.pop("cache_build_prefetch", max(1, self.cache_build_workers * 2)) or 1),
        )
        self.lmdb_readahead = bool(kwargs.pop("lmdb_readahead", True))
        max_samples = kwargs.pop("max_samples", None)
        self.max_samples = int(max_samples) if max_samples not in (None, "", "null") else None
        self.use_cache = bool(kwargs.pop("use_cache", True))
        self.cache_dir = Path(kwargs.pop("cache_dir", self.root_path / ".cache"))
        self.joints_data = self._load_with_cache()

    def _cache_key(self) -> str:
        max_tag = self.max_samples if self.max_samples is not None else "all"
        norm_tag = "norm" if self.normalized else "raw"
        z_tag = "z1" if self.normalize_z else "z0"
        return (
            f"{self.name}_{self.split_id}_{self.subset}_"
            f"{self.pose_source}_{norm_tag}_{z_tag}_max{max_tag}_joints{self.num_joints}.pt"
        )

    def _get_cache_path(self) -> Path:
        return self.cache_dir / self._cache_key()

    def _legacy_cache_path(self) -> Path | None:
        return None

    def _load_with_cache(self) -> torch.Tensor:
        cache_path = self._get_cache_path()
        if self.use_cache:
            candidate_paths = [cache_path]
            legacy_path = self._legacy_cache_path()
            if legacy_path is not None and legacy_path != cache_path:
                candidate_paths.append(legacy_path)
            for candidate_path in candidate_paths:
                if candidate_path.exists():
                    print(f"[INFO] Loading {self.name} from cache: {candidate_path}")
                    return torch.load(candidate_path, map_location="cpu", weights_only=True).to(self.device)

        data = self._load_data()
        if self.use_cache and len(data) > 0:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                torch.save(data.cpu(), cache_path)
                print(f"[INFO] Cached to: {cache_path}")
            except Exception as e:
                print(f"[WARN] Cache save failed: {e}")
        return data

    def _load_data(self) -> torch.Tensor:
        from ..radar_pose.m4human import M4HumanLMDBPoseDataset

        config = {
            "dataset_path": str(self.root_path),
            "data_format": {
                "n_points": 1,
                "merge_nframes": self.temporal_window,
                "normalized": self.normalized,
                "radar_input_c": 4,
                "num_joints": self.num_joints,
            },
            "m4human": {
                "cached_root": str(self.root_path),
                "smplx_model_root": str(resolve_dataset_uri(str(self.smplx_model_root))),
                "split": self.split_id,
                "temporal_window": self.temporal_window,
                "normalize_z": self.normalize_z,
                "random_sample_train": False,
                "pose_source": self.pose_source,
                "max_samples": {self.subset: self.max_samples},
            },
        }
        dataset = M4HumanLMDBPoseDataset(config=config, split=self.subset, device="cpu")
        print(f"[INFO] Loading {len(dataset)} frames from {self.name}...")
        if self.pose_source == "param_joints":
            try:
                return self._load_param_joints(dataset)
            finally:
                dataset.close()

        joints_data = torch.empty((len(dataset), self.num_joints, 3), dtype=torch.float32, device=self.device)
        try:
            for i, indicator in enumerate(tqdm(dataset.data_list, desc=f"Loading {self.name}", leave=False)):
                joints = dataset._load_joints(indicator)
                if self.normalized:
                    joints = joints - joints[:1]
                joints_data[i] = torch.from_numpy(joints[: self.num_joints].astype(np.float32))
        finally:
            dataset.close()
        print(f"[INFO] Loaded {len(joints_data)} frames for {self.name}.")
        return joints_data

    def _load_param_joints(self, dataset) -> torch.Tensor:
        env_paths = {name: str(path) for name, path in dataset.env_paths.items()}
        indicators = dataset.data_list
        joints_data = torch.empty((len(indicators), self.num_joints, 3), dtype=torch.float32, device=self.device)
        chunks = [
            (start, indicators[start : start + self.cache_build_chunk_size])
            for start in range(0, len(indicators), self.cache_build_chunk_size)
        ]
        workers = min(self.cache_build_workers, len(chunks))
        if workers <= 1:
            for start, chunk in tqdm(chunks, desc=f"Loading {self.name}", leave=False):
                joints = _load_m4human_param_joints_chunk(
                    env_paths,
                    chunk,
                    num_joints=self.num_joints,
                    normalized=self.normalized,
                    normalize_z=self.normalize_z,
                    lmdb_readahead=self.lmdb_readahead,
                )
                joints_data[start : start + len(chunk)] = torch.from_numpy(joints)
            print(f"[INFO] Loaded {len(joints_data)} frames for {self.name}.")
            return joints_data

        max_in_flight = max(workers, self.cache_build_prefetch)
        next_idx = 0
        futures = {}

        def submit_available(executor):
            nonlocal next_idx
            while next_idx < len(chunks) and len(futures) < max_in_flight:
                start, chunk = chunks[next_idx]
                futures[
                    executor.submit(
                        _load_m4human_param_joints_chunk,
                        env_paths,
                        chunk,
                        self.num_joints,
                        self.normalized,
                        self.normalize_z,
                        self.lmdb_readahead,
                    )
                ] = (start, len(chunk))
                next_idx += 1

        with ProcessPoolExecutor(max_workers=workers) as executor:
            submit_available(executor)
            progress = tqdm(total=len(chunks), desc=f"Loading {self.name}", leave=False)
            try:
                while futures:
                    for future in as_completed(futures):
                        start, count = futures.pop(future)
                        joints_data[start : start + count] = torch.from_numpy(future.result())
                        progress.update(1)
                        submit_available(executor)
                        break
            finally:
                progress.close()
        print(f"[INFO] Loaded {len(joints_data)} frames for {self.name}.")
        return joints_data

    def __len__(self):
        return self.joints_data.shape[0]

    def __getitem__(self, idx):
        return self.joints_data[idx], self.label


def _load_m4human_param_joints_chunk(
    env_paths: dict[str, str],
    indicators: list[list[int]],
    num_joints: int,
    normalized: bool,
    normalize_z: bool,
    lmdb_readahead: bool,
) -> np.ndarray:
    from ..radar_pose.m4human import _unpack_dict_np

    params_env = lmdb.open(env_paths["params"], readonly=True, subdir=False, lock=False, readahead=lmdb_readahead)
    calib_env = lmdb.open(env_paths["calib"], readonly=True, subdir=False, lock=False, readahead=lmdb_readahead)
    joints_data = np.empty((len(indicators), num_joints, 3), dtype=np.float32)
    try:
        with params_env.begin() as txn_param, calib_env.begin() as txn_calib:
            for i, indicator in enumerate(indicators):
                key = str(indicator).encode()
                param_blob = txn_param.get(key)
                calib_blob = txn_calib.get(key)
                if param_blob is None or calib_blob is None:
                    raise KeyError(f"Missing params/calib for indicator {indicator}")
                params = _unpack_dict_np(param_blob)
                calib = _unpack_dict_np(calib_blob)
                joints = _m4human_param_joints_to_radar(params, calib, indicator, num_joints, normalize_z)
                if normalized:
                    joints = joints - joints[:1]
                joints_data[i] = joints
    finally:
        params_env.close()
        calib_env.close()
    return joints_data


def _m4human_param_joints_to_radar(
    params: dict,
    calib: dict,
    indicator: list[int],
    num_joints: int,
    normalize_z: bool,
) -> np.ndarray:
    joints = np.asarray(params["joints"], dtype=np.float32)
    if joints.ndim != 2 or joints.shape[1] < 3 or joints.shape[0] < num_joints:
        raise ValueError(
            f"Invalid params.joints for indicator {indicator}: "
            f"expected [>={num_joints},3], got {joints.shape}"
        )
    vicon_to_cam_rot = np.asarray(calib["vicon_to_cam_rotmatrix"], dtype=np.float32)
    vicon_to_cam_t = np.asarray(calib["vicon_to_cam_tvec"], dtype=np.float32) / 1000.0
    radar_to_cam_rot = np.asarray(calib["radar_to_cam_rotmatrix"], dtype=np.float32)
    radar_to_cam_t = np.asarray(calib["radar_to_cam_tvec"], dtype=np.float32)
    joints_cam = (vicon_to_cam_rot @ joints[:, :3].T).T + vicon_to_cam_t
    joints_radar = np.linalg.solve(radar_to_cam_rot, (joints_cam - radar_to_cam_t).T).T
    if normalize_z:
        joints_radar = joints_radar.copy()
        joints_radar[:, 2] -= 1.5
    return joints_radar[:num_joints].astype(np.float32, copy=False)


__all__ = ["M4HumanJointsDataset"]
