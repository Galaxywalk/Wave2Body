# -*- coding: utf-8 -*-
"""Base dataset class for end-to-end token translation training."""

import json
import hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Any, Dict, Optional

from ...utils.paths import resolve_dataset_uri
from .point_sampling import VALID_POINT_SAMPLING_STRATEGIES, pad_or_sample_points
from .root_alignment import crop_point_cloud, resolve_root_alignment_config


def cropping(radar_pc, x_range=[-1.67, 1.67], y_range=[-1, 1], z_range=[-1.67, 1.67]):
    """Crop radar point cloud to a bounding box."""
    return crop_point_cloud(
        radar_pc,
        x_range=tuple(float(v) for v in x_range),
        y_range=tuple(float(v) for v in y_range),
        z_range=tuple(float(v) for v in z_range),
    )


class BaseRadarPoseDataset(Dataset):
    """
    Base class for mmWave radar pose estimation datasets with token support.

    For end-to-end training, this dataset provides:
    1. mmWave point cloud
    2. Ground truth pose coordinates
    3. (Optional) Pre-computed pose tokens from VQ-VAE

    Subclasses should implement:
    - _load_data_list(): Build list of data samples
    - _load_sample(idx): Load (radar_frames, gt_joints) for a sample
    """

    def __init__(
        self,
        config: Dict,
        split: str = 'train',
        vqvae_model: Optional[torch.nn.Module] = None,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.config = config
        self.split = split
        self.device = device or 'cpu'
        self.vqvae_model = vqvae_model

        # Common data format params
        fmt = config.get("data_format", {})
        self.n_points = fmt.get("n_points", 200)
        self.merge_nframes = fmt.get("merge_nframes", 1)
        self.root_alignment = resolve_root_alignment_config(fmt)
        self.normalized = self.root_alignment.normalized
        self.input_root_normalized = self.root_alignment.input_root_normalized
        self.target_root_normalized = self.root_alignment.target_root_normalized
        self.input_root_crop = self.root_alignment.input_root_crop
        self.input_root_crop_x_range = self.root_alignment.input_root_crop_x_range
        self.input_root_crop_y_range = self.root_alignment.input_root_crop_y_range
        self.input_root_crop_z_range = self.root_alignment.input_root_crop_z_range
        self.input_channels = fmt.get("radar_input_c", 5)
        self.num_joints = fmt.get("num_joints", 22)
        self.point_sampling = str(fmt.get("point_sampling", "fps")).strip().lower()
        if self.point_sampling not in VALID_POINT_SAMPLING_STRATEGIES:
            raise ValueError(
                f"data_format.point_sampling must be one of {sorted(VALID_POINT_SAMPLING_STRATEGIES)}, "
                f"got {self.point_sampling!r}"
            )

        # Dataset path
        self.root = resolve_dataset_uri(str(config.get("dataset_path") or ""))

        # Random seed
        self.random_seed = config.get("random_seed", 42)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        # Data quantity (for training subset)
        self.used_data_quantity = 1.0
        if split == "train" and "used_data_quantity" in config:
            self.used_data_quantity = config["used_data_quantity"]

        # Precompute tokens flag
        self.precompute_tokens = config.get("precompute_tokens", False)

        # Optional tensor cache.
        self.cache_mode = "none"
        self.cache_path: Optional[str] = None
        self._cache_file: Optional[Path] = None
        self._cache_on_gpu = False
        self.cache_build_workers = 0
        self.cache_build_batch_size = 512
        self.cache_build_prefetch_factor = 2
        self._cached_mmwave_points: Optional[torch.Tensor] = None
        self._cached_target_poses: Optional[torch.Tensor] = None
        self._cached_target_poses_absolute: Optional[torch.Tensor] = None

        # Load data list (implemented by subclass)
        self.data_list = self._load_data_list()
        self._init_tensor_cache()

    def _load_data_list(self):
        """Build list of data samples. Override in subclass."""
        raise NotImplementedError

    def _load_sample(self, idx):
        """Load raw sample data. Override in subclass. Returns (radar_data_list, gt_joints)."""
        raise NotImplementedError

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary with:
        - mmwave_points: [n_points, input_channels] point cloud tensor
        - target_poses: [num_joints, 3] ground truth pose
        - target_tokens: (optional) [num_tokens, token_dim] if precomputed
        """
        cached = self._get_cached_item(idx)
        if cached is not None:
            return cached

        radar_list, gt_joints = self._load_sample(idx)

        # Merge frames
        if len(radar_list) > 0:
            radar_data = np.concatenate(radar_list, axis=0)
        else:
            radar_data = np.zeros((0, self.input_channels))

        # Convert to tensor
        radar_tensor = torch.from_numpy(radar_data).float()

        # Process points (downsample/pad to n_points)
        radar_tensor = self._process_points(radar_tensor)

        gt_joints_absolute = gt_joints.copy()
        if self.target_root_normalized:
            pelvis = gt_joints[0]
            gt_joints = gt_joints - pelvis

        gt_tensor = torch.from_numpy(gt_joints).float()
        gt_absolute_tensor = torch.from_numpy(gt_joints_absolute).float()

        # Build output dictionary
        result = {
            'mmwave_points': radar_tensor,
            'target_poses': gt_tensor,
            'target_poses_absolute': gt_absolute_tensor,
        }

        # Optionally precompute tokens using VQ-VAE encoder
        if self.precompute_tokens and self.vqvae_model is not None:
            with torch.no_grad():
                gt_batch = gt_tensor.unsqueeze(0).to(self.device)
                encoding_indices, quantized_features, _ = self.vqvae_model.encode(
                    gt_batch, train=False
                )
                result['target_tokens'] = quantized_features.squeeze(0).cpu()
                result['target_indices'] = encoding_indices.squeeze(0).cpu()

        return result

    def _process_points(self, points: torch.Tensor) -> torch.Tensor:
        """Downsample or pad point cloud to n_points."""
        return pad_or_sample_points(
            points,
            n_points=self.n_points,
            input_channels=self.input_channels,
            strategy=self.point_sampling,
        )

    def _cache_extra_spec(self) -> Dict[str, Any]:
        """
        Subclasses may override to include extra fields that affect sample contents.
        """
        return {}

    def _cache_spec(self) -> Dict[str, Any]:
        payload = {
            "dataset_class": self.__class__.__name__,
            "split": str(self.split),
            "root": str(self.root),
            "n_points": int(self.n_points),
            "merge_nframes": int(self.merge_nframes),
            "normalized": bool(self.normalized),
            "input_root_normalized": bool(self.input_root_normalized),
            "target_root_normalized": bool(self.target_root_normalized),
            "input_root_crop": bool(self.input_root_crop),
            "input_root_crop_x_range": [float(v) for v in self.input_root_crop_x_range],
            "input_root_crop_y_range": [float(v) for v in self.input_root_crop_y_range],
            "input_root_crop_z_range": [float(v) for v in self.input_root_crop_z_range],
            "input_channels": int(self.input_channels),
            "num_joints": int(self.num_joints),
            "point_sampling": str(self.point_sampling),
            "used_data_quantity": float(self.used_data_quantity),
            "random_seed": int(self.random_seed),
            "data_len": int(len(self.data_list)),
        }
        payload.update(self._cache_extra_spec())
        return payload

    def _cache_fingerprint(self) -> str:
        text = json.dumps(self._cache_spec(), sort_keys=True, ensure_ascii=True)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    def _resolve_cache_cfg(self) -> tuple[str, Optional[Path]]:
        cache_cfg = (
            self.config.get("data", {}).get("cache", {})
            if isinstance(self.config.get("data", {}), dict)
            else {}
        )
        mode = str(cache_cfg.get("mode", "none")).strip().lower()
        if mode not in {"none", "cpu_tensor", "gpu_tensor"}:
            print(f"[WARN] Unknown cache mode `{mode}`, fallback to `none`.")
            mode = "none"
        path_like = cache_cfg.get("path", None)
        if path_like in (None, "", "null"):
            cache_dir = Path(self.root) / ".tensor_cache"
        else:
            cache_dir = Path(path_like)
        return mode, cache_dir

    def _resolve_process_points_fn(self):
        if hasattr(self, "_process_points_cpu"):
            return getattr(self, "_process_points_cpu")
        return self._process_points

    def _prepare_sample_tensors(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        radar_list, gt_joints = self._load_sample(idx)
        if len(radar_list) > 0:
            radar_data = np.concatenate(radar_list, axis=0)
        else:
            radar_data = np.zeros((0, self.input_channels), dtype=np.float32)

        gt_joints_absolute = gt_joints.copy()
        if self.target_root_normalized:
            pelvis = gt_joints[0]
            gt_joints = gt_joints - pelvis

        radar_tensor = torch.from_numpy(radar_data).float()
        process_points = self._resolve_process_points_fn()
        radar_tensor = process_points(radar_tensor)
        gt_tensor = torch.from_numpy(gt_joints).float()
        gt_absolute_tensor = torch.from_numpy(gt_joints_absolute).float()
        return radar_tensor, gt_tensor, gt_absolute_tensor

    def _build_tensor_cache(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_items = len(self.data_list)
        workers = max(0, int(getattr(self, "cache_build_workers", 0) or 0))
        if workers > 0 and n_items > 0:
            return self._build_tensor_cache_with_loader(workers)

        mmwave_items = []
        pose_items = []
        pose_absolute_items = []
        for idx in range(n_items):
            mmwave, pose, pose_absolute = self._prepare_sample_tensors(idx)
            mmwave_items.append(mmwave)
            pose_items.append(pose)
            pose_absolute_items.append(pose_absolute)
        if mmwave_items:
            mmwave_all = torch.stack(mmwave_items, dim=0).contiguous()
            pose_all = torch.stack(pose_items, dim=0).contiguous()
            pose_absolute_all = torch.stack(pose_absolute_items, dim=0).contiguous()
        else:
            mmwave_all = torch.empty((0, self.n_points, self.input_channels), dtype=torch.float32)
            pose_all = torch.empty((0, self.num_joints, 3), dtype=torch.float32)
            pose_absolute_all = torch.empty((0, self.num_joints, 3), dtype=torch.float32)
        return mmwave_all, pose_all, pose_absolute_all

    def _build_tensor_cache_with_loader(self, workers: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = max(1, int(getattr(self, "cache_build_batch_size", 512) or 512))
        kwargs: dict[str, Any] = {
            "dataset": self,
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": int(workers),
            "pin_memory": False,
            "persistent_workers": False,
            "drop_last": False,
        }
        if workers > 0:
            kwargs["prefetch_factor"] = max(1, int(getattr(self, "cache_build_prefetch_factor", 2) or 2))
        loader = DataLoader(**kwargs)
        mmwave_batches = []
        pose_batches = []
        pose_absolute_batches = []
        prepared = 0
        total = len(self.data_list)
        print(
            f"[Cache] Building tensors with DataLoader workers={workers}, batch_size={batch_size} ...",
            flush=True,
        )
        for batch in loader:
            mmwave = batch["mmwave_points"].cpu().contiguous()
            pose = batch["target_poses"].cpu().contiguous()
            pose_absolute = batch["target_poses_absolute"].cpu().contiguous()
            mmwave_batches.append(mmwave)
            pose_batches.append(pose)
            pose_absolute_batches.append(pose_absolute)
            prepared += int(mmwave.shape[0])
            if prepared % 5120 == 0 or prepared >= total:
                print(f"[Cache] Prepared {prepared}/{total} samples ...", flush=True)
        if mmwave_batches:
            return (
                torch.cat(mmwave_batches, dim=0).contiguous(),
                torch.cat(pose_batches, dim=0).contiguous(),
                torch.cat(pose_absolute_batches, dim=0).contiguous(),
            )
        return (
            torch.empty((0, self.n_points, self.input_channels), dtype=torch.float32),
            torch.empty((0, self.num_joints, 3), dtype=torch.float32),
            torch.empty((0, self.num_joints, 3), dtype=torch.float32),
        )

    def _save_tensor_cache(
        self,
        mmwave_all: torch.Tensor,
        pose_all: torch.Tensor,
        pose_absolute_all: torch.Tensor,
    ) -> None:
        if self._cache_file is None:
            return
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "fingerprint": self._cache_fingerprint(),
                "spec": self._cache_spec(),
                "mmwave_points": mmwave_all.cpu(),
                "target_poses": pose_all.cpu(),
                "target_poses_absolute": pose_absolute_all.cpu(),
            },
            self._cache_file,
        )

    def _load_tensor_cache(self) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self._cache_file is None or not self._cache_file.exists():
            return None
        try:
            payload = torch.load(self._cache_file, map_location="cpu", weights_only=False)
            mmwave_all = payload["mmwave_points"]
            pose_all = payload["target_poses"]
            pose_absolute_all = payload.get("target_poses_absolute")
            if pose_absolute_all is None:
                print(f"[WARN] Cache `{self._cache_file}` has no absolute target poses, rebuilding cache.")
                return None
            if int(mmwave_all.shape[0]) != len(self.data_list):
                print(
                    f"[WARN] Cache length mismatch ({mmwave_all.shape[0]} vs {len(self.data_list)}), rebuilding cache."
                )
                return None
            return mmwave_all, pose_all, pose_absolute_all
        except Exception as exc:
            print(f"[WARN] Failed to load dataset cache `{self._cache_file}`: {exc}")
            return None

    def _materialize_cache_device(
        self,
        mmwave_all: torch.Tensor,
        pose_all: torch.Tensor,
        pose_absolute_all: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cache_mode != "gpu_tensor":
            self._cache_on_gpu = False
            return mmwave_all, pose_all, pose_absolute_all

        use_cuda = str(self.device).startswith("cuda") and torch.cuda.is_available()
        if not use_cuda:
            print("[WARN] cache.mode=gpu_tensor but CUDA/device unavailable, fallback to cpu_tensor cache.")
            self.cache_mode = "cpu_tensor"
            self._cache_on_gpu = False
            return mmwave_all, pose_all, pose_absolute_all

        dev = torch.device(self.device)
        self._cache_on_gpu = True
        return (
            mmwave_all.to(dev, non_blocking=False),
            pose_all.to(dev, non_blocking=False),
            pose_absolute_all.to(dev, non_blocking=False),
        )

    def _init_tensor_cache(self) -> None:
        mode, cache_dir = self._resolve_cache_cfg()
        self.cache_mode = mode
        self.cache_path = str(cache_dir) if cache_dir is not None else None
        cache_cfg = (
            self.config.get("data", {}).get("cache", {})
            if isinstance(self.config.get("data", {}), dict)
            else {}
        )
        self.cache_build_workers = max(0, int(cache_cfg.get("build_workers", 0) or 0))
        self.cache_build_batch_size = max(1, int(cache_cfg.get("build_batch_size", 512) or 512))
        self.cache_build_prefetch_factor = max(1, int(cache_cfg.get("build_prefetch_factor", 2) or 2))
        if self.cache_mode == "none":
            return

        fp = self._cache_fingerprint()
        self._cache_file = Path(cache_dir) / f"{self.__class__.__name__}_{self.split}_{fp}.pt"
        loaded = self._load_tensor_cache()
        if loaded is None:
            print(
                f"[Cache] Building {self.__class__.__name__} split={self.split} cache "
                f"(mode={self.cache_mode}, samples={len(self.data_list)}, "
                f"build_workers={self.cache_build_workers}) ...",
                flush=True,
            )
            mmwave_all, pose_all, pose_absolute_all = self._build_tensor_cache()
            self._save_tensor_cache(mmwave_all, pose_all, pose_absolute_all)
        else:
            mmwave_all, pose_all, pose_absolute_all = loaded
            print(
                f"[Cache] Loaded {self.__class__.__name__} split={self.split} cache "
                f"(mode={self.cache_mode}, samples={len(self.data_list)}) from {self._cache_file}",
                flush=True,
            )

        (
            self._cached_mmwave_points,
            self._cached_target_poses,
            self._cached_target_poses_absolute,
        ) = self._materialize_cache_device(mmwave_all, pose_all, pose_absolute_all)

    def _get_cached_item(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        if self._cached_mmwave_points is None or self._cached_target_poses is None:
            return None
        result = {
            "mmwave_points": self._cached_mmwave_points[idx],
            "target_poses": self._cached_target_poses[idx],
        }
        if self._cached_target_poses_absolute is not None:
            result["target_poses_absolute"] = self._cached_target_poses_absolute[idx]
        if self.precompute_tokens and self.vqvae_model is not None:
            with torch.no_grad():
                gt_batch = result["target_poses"].unsqueeze(0).to(self.device)
                encoding_indices, quantized_features, _ = self.vqvae_model.encode(gt_batch, train=False)
                result["target_tokens"] = quantized_features.squeeze(0).cpu()
                result["target_indices"] = encoding_indices.squeeze(0).cpu()
        return result
