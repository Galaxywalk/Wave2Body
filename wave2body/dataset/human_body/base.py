# -*- coding: utf-8 -*-
"""Base class for human-body joint-only datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class BaseJointsDataset(Dataset, ABC):
    """Base class for joints datasets with caching support.

    Human-body pose-only datasets are root-relative by default. Pass
    ``normalized=False`` to keep source coordinates when the source contains an
    absolute root.
    """

    def __init__(
        self,
        root_path: str,
        num_joints: int,
        device: str,
        label: int,
        name: str,
        use_cache: bool = True,
        cache_dir: str | None = None,
        **kwargs,
    ):
        self.root_path = Path(root_path)
        self.num_joints = num_joints
        self.device = device
        self.label = label
        self.name = name
        self.normalized = bool(kwargs.pop("normalized", True))
        self.cache_build_workers = max(0, int(kwargs.pop("cache_build_workers", 0) or 0))
        self.cache_build_prefetch = max(
            1,
            int(kwargs.pop("cache_build_prefetch", max(1, self.cache_build_workers * 2)) or 1),
        )
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else self.root_path / ".cache"

        self.joints_data = self._load_with_cache()

    @abstractmethod
    def _get_file_paths(self) -> List[Path]:
        """Return list of data file paths."""
        pass

    def _get_cache_path(self) -> Path:
        norm_tag = "norm" if self.normalized else "raw"
        return self.cache_dir / f"{self.name}_{norm_tag}_joints{self.num_joints}.pt"

    def _load_with_cache(self) -> torch.Tensor:
        """Load data with caching."""
        cache_path = self._get_cache_path()

        if self.use_cache and cache_path.exists():
            print(f"[INFO] Loading {self.name} from cache: {cache_path}")
            try:
                data = torch.load(cache_path, map_location="cpu", weights_only=True).to(self.device)
                print(f"[INFO] Loaded {len(data)} frames from cache.")
                return data
            except Exception as e:
                print(f"[WARN] Cache load failed, reloading: {e}")

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
        """Load data from source files."""
        file_paths = self._get_file_paths()
        if not file_paths:
            print(f"[WARN] No files found in {self.root_path}")
            return torch.empty(0, self.num_joints, 3, dtype=torch.float32, device=self.device)

        print(f"[INFO] Loading {len(file_paths)} files from {self.root_path.name}...")

        workers = min(self.cache_build_workers, len(file_paths))
        if workers > 1:
            frame_counts = self._scan_frame_counts_parallel(file_paths, workers)
            total_frames = int(sum(frame_counts))
        else:
            frame_counts = []
            total_frames = 0
            for fp in tqdm(file_paths, desc=f"Scanning {self.root_path.name}", leave=False):
                try:
                    with np.load(fp) as data:
                        count = int(len(data["joints"]))
                    frame_counts.append(count)
                    total_frames += count
                except Exception as e:
                    frame_counts.append(0)
                    print(f"[WARN] Could not read {fp}: {e}")

        if total_frames == 0:
            return torch.empty(0, self.num_joints, 3, dtype=torch.float32, device=self.device)

        joints_data = torch.empty((total_frames, self.num_joints, 3), dtype=torch.float32, device=self.device)
        if workers > 1:
            self._load_files_parallel(file_paths, frame_counts, joints_data, workers)
            print(f"[INFO] Loaded {len(joints_data)} frames for {self.root_path.name}.")
            return joints_data

        idx = 0
        for fp in tqdm(file_paths, desc=f"Loading {self.root_path.name}", leave=False):
            try:
                with np.load(fp) as data:
                    joints = self.preprocess(data["joints"].astype(np.float32))
                    n = joints.shape[0]
                    joints_data[idx : idx + n] = torch.from_numpy(joints)
                    idx += n
            except Exception as e:
                print(f"[WARN] Could not load {fp}: {e}")

        if idx < total_frames:
            joints_data = joints_data[:idx]

        print(f"[INFO] Loaded {len(joints_data)} frames for {self.root_path.name}.")
        return joints_data

    def _scan_frame_counts_parallel(self, file_paths: list[Path], workers: int) -> list[int]:
        counts = [0] * len(file_paths)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(self._read_frame_count, fp): i
                for i, fp in enumerate(file_paths)
            }
            for future in tqdm(
                as_completed(future_to_idx),
                total=len(future_to_idx),
                desc=f"Scanning {self.root_path.name}",
                leave=False,
            ):
                idx = future_to_idx[future]
                try:
                    counts[idx] = int(future.result())
                except Exception as e:
                    print(f"[WARN] Could not read {file_paths[idx]}: {e}")
        return counts

    def _load_files_parallel(
        self,
        file_paths: list[Path],
        frame_counts: list[int],
        joints_data: torch.Tensor,
        workers: int,
    ) -> None:
        offsets = np.cumsum([0] + frame_counts[:-1], dtype=np.int64)
        max_in_flight = max(workers, self.cache_build_prefetch)
        next_idx = 0
        futures = {}

        def submit_available(executor):
            nonlocal next_idx
            while next_idx < len(file_paths) and len(futures) < max_in_flight:
                if frame_counts[next_idx] > 0:
                    futures[executor.submit(self._load_file_joints, file_paths[next_idx])] = next_idx
                next_idx += 1

        with ThreadPoolExecutor(max_workers=workers) as executor:
            submit_available(executor)
            progress = tqdm(total=len(file_paths), desc=f"Loading {self.root_path.name}", leave=False)
            completed = 0
            try:
                while futures:
                    for future in as_completed(futures):
                        idx = futures.pop(future)
                        try:
                            joints = future.result()
                            start = int(offsets[idx])
                            end = start + int(joints.shape[0])
                            joints_data[start:end] = torch.from_numpy(joints)
                        except Exception as e:
                            print(f"[WARN] Could not load {file_paths[idx]}: {e}")
                        completed += 1
                        progress.update(1)
                        submit_available(executor)
                        break
            finally:
                progress.update(max(0, len(file_paths) - completed - len(futures)))
                progress.close()

    def _read_frame_count(self, file_path: Path) -> int:
        with np.load(file_path) as data:
            return int(len(data["joints"]))

    def _load_file_joints(self, file_path: Path) -> np.ndarray:
        with np.load(file_path) as data:
            return self.preprocess(data["joints"].astype(np.float32))

    def preprocess(self, joints: np.ndarray) -> np.ndarray:
        """Preprocessing hook. Override for custom preprocessing."""
        return joints

    def __len__(self):
        return self.joints_data.shape[0]

    def __getitem__(self, idx):
        return self.joints_data[idx], self.label


__all__ = ["BaseJointsDataset"]
