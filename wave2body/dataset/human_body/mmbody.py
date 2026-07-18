# -*- coding: utf-8 -*-
"""MMBody joint-only dataset loader."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm.auto import tqdm

from .base import BaseJointsDataset
from .registry import register_dataset
from .transforms import PreprocessingUtils
from ..radar_pose.mmbody_compact import MMBodyCompactStore, is_mmbody_compact


@register_dataset("mmbody")
class MMBodyJointsDataset(BaseJointsDataset):
    """MMBody dataset loader (one frame per file)."""

    def __init__(self, root_path: str, num_joints: int, device: str, label: int, name: str, **kwargs):
        self.splits = list(kwargs.pop("splits", ["train", "test"]))
        max_samples = kwargs.pop("max_samples", None)
        self.max_samples = int(max_samples) if max_samples not in (None, "", "null") else None
        super().__init__(root_path, num_joints, device, label, name, **kwargs)

    def _get_cache_path(self) -> Path:
        if self.max_samples is None:
            return super()._get_cache_path()
        norm_tag = "norm" if self.normalized else "raw"
        split_tag = "-".join(str(split) for split in self.splits)
        return self.cache_dir / f"{self.name}_{split_tag}_{norm_tag}_max{self.max_samples}_joints{self.num_joints}.pt"

    def _get_file_paths(self) -> List[Path]:
        if not self.root_path.exists():
            print(f"[WARN] MMBody dir not found: {self.root_path}")
            return []
        files = []
        for split in self.splits:
            split_dir = self.root_path / split
            if split_dir.exists():
                files.extend(split_dir.rglob("mesh/frame_*.npz"))
        files = sorted(files)
        if self.max_samples is not None:
            files = files[: self.max_samples]
        return files

    def _load_compact_data(self) -> torch.Tensor:
        if self.num_joints > 22:
            raise ValueError(f"Compact mmBody provides 22 joints, requested {self.num_joints}")
        store = MMBodyCompactStore(self.root_path)
        try:
            records = [record for record in store.samples if str(record["split"]) in self.splits]
            records.sort(
                key=lambda record: (
                    f"{record['split']}/"
                    + (
                        f"{record['sequence_name']}"
                        if str(record["split"]) == "train"
                        else f"{record['scenario']}/{record['sequence_name']}"
                    )
                    + f"/mesh/frame_{int(record['frame'])}.npz"
                )
            )
            if self.max_samples is not None:
                records = records[: self.max_samples]
            joints_data = torch.empty(
                (len(records), self.num_joints, 3),
                dtype=torch.float32,
                device=self.device,
            )
            for index, record in enumerate(tqdm(records, desc="Loading compact MMBody", leave=False)):
                joints = store.read_pose(bytes(record["pose_key"]))[: self.num_joints]
                joints_data[index] = torch.from_numpy(self.preprocess(joints))
            print(f"[INFO] Loaded {len(joints_data)} compact frames for MMBody.")
            return joints_data
        finally:
            store.close()

    def _load_data(self) -> torch.Tensor:
        if is_mmbody_compact(self.root_path):
            return self._load_compact_data()
        file_paths = self._get_file_paths()
        if not file_paths:
            return torch.empty(0, self.num_joints, 3, dtype=torch.float32, device=self.device)

        print(f"[INFO] Loading {len(file_paths)} frames from MMBody...")

        with np.load(file_paths[0]) as data:
            if data["joints"].shape[0] < self.num_joints:
                raise ValueError(f"Requested {self.num_joints} joints, file has {data['joints'].shape[0]}")

        joints_data = torch.empty((len(file_paths), self.num_joints, 3), dtype=torch.float32, device=self.device)
        workers = min(self.cache_build_workers, len(file_paths))
        if workers > 1:
            self._load_frames_parallel(file_paths, joints_data, workers)
            print(f"[INFO] Loaded {len(joints_data)} frames for MMBody.")
            return joints_data

        for i, fp in enumerate(tqdm(file_paths, desc="Loading MMBody", leave=False)):
            try:
                with np.load(fp) as data:
                    joints = self.preprocess(data["joints"][: self.num_joints].astype(np.float32))
                    joints_data[i] = torch.from_numpy(joints)
            except Exception as e:
                warnings.warn(f"Failed to load {fp.name}: {e}")
                joints_data[i].fill_(0)

        print(f"[INFO] Loaded {len(joints_data)} frames for MMBody.")
        return joints_data

    def _load_frames_parallel(self, file_paths: list[Path], joints_data: torch.Tensor, workers: int) -> None:
        max_in_flight = max(workers, self.cache_build_prefetch)
        next_idx = 0
        futures = {}

        def submit_available(executor):
            nonlocal next_idx
            while next_idx < len(file_paths) and len(futures) < max_in_flight:
                futures[executor.submit(self._load_one_frame, file_paths[next_idx])] = next_idx
                next_idx += 1

        with ThreadPoolExecutor(max_workers=workers) as executor:
            submit_available(executor)
            progress = tqdm(total=len(file_paths), desc="Loading MMBody", leave=False)
            try:
                while futures:
                    for future in as_completed(futures):
                        idx = futures.pop(future)
                        try:
                            joints_data[idx] = torch.from_numpy(future.result())
                        except Exception as e:
                            warnings.warn(f"Failed to load {file_paths[idx].name}: {e}")
                            joints_data[idx].fill_(0)
                        progress.update(1)
                        submit_available(executor)
                        break
            finally:
                progress.close()

    def _load_one_frame(self, file_path: Path) -> np.ndarray:
        with np.load(file_path) as data:
            return self.preprocess(data["joints"][: self.num_joints].astype(np.float32))

    def preprocess(self, joints: np.ndarray) -> np.ndarray:
        if self.normalized:
            return PreprocessingUtils.pelvis_normalization(joints)
        return joints


__all__ = ["MMBodyJointsDataset"]
