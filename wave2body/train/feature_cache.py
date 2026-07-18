# -*- coding: utf-8 -*-
"""Frozen PT/VQ-VAE feature cache helpers."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from wave2body.utils.runtime_utils import autocast_context
except ImportError:  # Support package-style imports: wave2body.train.feature_cache.
    from ..utils.runtime_utils import autocast_context


@dataclass
class FrozenFeatureCacheConfig:
    enabled: bool
    batch_size: int
    num_workers: int
    pin_memory: bool
    prefetch_factor: int
    path: Optional[str]
    reuse: bool
    device: str
    max_gpu_cache_fraction: float
    batch_loader: str


class PrecomputedFeatureDataset(Dataset):
    """Dataset backed by cached frozen-backbone features."""

    def __init__(
        self,
        target_poses: torch.Tensor,
        cached_mmwave_tokens: torch.Tensor,
        cached_target_z_e: torch.Tensor,
        cached_target_indices: torch.Tensor,
        cached_mmwave_token_mask: torch.Tensor | None = None,
        signature: dict[str, Any] | None = None,
    ):
        lengths = {
            int(target_poses.shape[0]),
            int(cached_mmwave_tokens.shape[0]),
            int(cached_target_z_e.shape[0]),
            int(cached_target_indices.shape[0]),
        }
        if cached_mmwave_token_mask is not None:
            lengths.add(int(cached_mmwave_token_mask.shape[0]))
        if len(lengths) != 1:
            raise ValueError(f"Cached feature length mismatch: {sorted(lengths)}")
        self.target_poses = target_poses.contiguous()
        self.cached_mmwave_tokens = cached_mmwave_tokens.contiguous()
        self.cached_target_z_e = cached_target_z_e.contiguous()
        self.cached_target_indices = cached_target_indices.contiguous().to(torch.long)
        self.cached_mmwave_token_mask = (
            cached_mmwave_token_mask.contiguous().to(torch.bool)
            if cached_mmwave_token_mask is not None
            else None
        )
        self.signature = dict(signature or {})

    def __len__(self) -> int:
        return int(self.target_poses.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "target_poses": self.target_poses[idx],
            "cached_mmwave_tokens": self.cached_mmwave_tokens[idx],
            "cached_target_z_e": self.cached_target_z_e[idx],
            "cached_target_indices": self.cached_target_indices[idx],
            **(
                {"cached_mmwave_token_mask": self.cached_mmwave_token_mask[idx]}
                if self.cached_mmwave_token_mask is not None
                else {}
            ),
        }

    @property
    def device(self) -> torch.device:
        return self.target_poses.device

    def nbytes(self) -> int:
        tensors = (
            self.target_poses,
            self.cached_mmwave_tokens,
            self.cached_target_z_e,
            self.cached_target_indices,
        )
        total = int(sum(t.numel() * t.element_size() for t in tensors))
        if self.cached_mmwave_token_mask is not None:
            total += int(self.cached_mmwave_token_mask.numel() * self.cached_mmwave_token_mask.element_size())
        return total

    def to(self, device: torch.device | str) -> "PrecomputedFeatureDataset":
        device_obj = torch.device(device)
        if self.device == device_obj:
            return self
        return PrecomputedFeatureDataset(
            target_poses=self.target_poses.to(device_obj, non_blocking=True),
            cached_mmwave_tokens=self.cached_mmwave_tokens.to(device_obj, non_blocking=True),
            cached_target_z_e=self.cached_target_z_e.to(device_obj, non_blocking=True),
            cached_target_indices=self.cached_target_indices.to(device_obj, non_blocking=True),
            cached_mmwave_token_mask=(
                self.cached_mmwave_token_mask.to(device_obj, non_blocking=True)
                if self.cached_mmwave_token_mask is not None
                else None
            ),
            signature=self.signature,
        )

    def batch(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        idx = indices.to(device=self.device, dtype=torch.long, non_blocking=True)
        batch = {
            "target_poses": self.target_poses.index_select(0, idx),
            "cached_mmwave_tokens": self.cached_mmwave_tokens.index_select(0, idx),
            "cached_target_z_e": self.cached_target_z_e.index_select(0, idx),
            "cached_target_indices": self.cached_target_indices.index_select(0, idx),
        }
        if self.cached_mmwave_token_mask is not None:
            batch["cached_mmwave_token_mask"] = self.cached_mmwave_token_mask.index_select(0, idx)
        return batch


class PrecomputedFeatureBatchLoader:
    """Batch cached feature tensors directly, avoiding per-sample DataLoader collation."""

    def __init__(
        self,
        dataset: PrecomputedFeatureDataset,
        *,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        generator: torch.Generator | None = None,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.generator = generator
        self.num_workers = 0
        self.pin_memory = False
        self.persistent_workers = False
        self.prefetch_factor = None
        self.worker_init_fn = None
        self.collate_fn = None

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.dataset)
        if self.shuffle:
            order = torch.randperm(n, generator=self.generator)
        else:
            order = torch.arange(n, dtype=torch.long)
        limit = (n // self.batch_size) * self.batch_size if self.drop_last else n
        for start in range(0, limit, self.batch_size):
            end = min(start + self.batch_size, n)
            if self.drop_last and end - start < self.batch_size:
                break
            yield self.dataset.batch(order[start:end])


def unwrap_dataset(dataset: Dataset) -> Dataset:
    target = dataset
    visited = set()
    while target is not None and id(target) not in visited:
        visited.add(id(target))
        base = getattr(target, "base_dataset", None)
        if base is not None:
            target = base
            continue
        nested = getattr(target, "dataset", None)
        if nested is not None:
            target = nested
            continue
        break
    return target


def dataset_signature(dataset: Dataset) -> dict[str, Any]:
    sig: dict[str, Any] = {
        "dataset_class": dataset.__class__.__name__,
        "length": int(len(dataset)),
    }
    if hasattr(dataset, "indices"):
        indices = [int(x) for x in getattr(dataset, "indices")]
        sig["indices_hash"] = hashlib.sha1(
            json.dumps(indices, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest()[:16]
        sig["indices_len"] = len(indices)

    base = unwrap_dataset(dataset)
    sig["base_class"] = base.__class__.__name__
    if hasattr(base, "_cache_spec"):
        try:
            sig["base_spec"] = base._cache_spec()
        except Exception:
            pass
    else:
        for attr_name in ("root", "split", "n_points", "merge_nframes", "input_channels", "num_joints"):
            if hasattr(base, attr_name):
                sig[attr_name] = getattr(base, attr_name)
    return sig


def dataset_cache_mode(dataset) -> str:
    target = dataset
    visited = set()
    while target is not None and id(target) not in visited:
        visited.add(id(target))
        mode = getattr(target, "cache_mode", None)
        if mode is not None:
            return str(mode)
        base = getattr(target, "base_dataset", None)
        if base is not None:
            target = base
            continue
        target = getattr(target, "dataset", None)
    return "none"


def resolve_frozen_feature_cache_config(
    training_cfg: dict,
    *,
    freeze_point_transformer: bool | None = None,
    freeze_vqvae: bool,
    freeze_mmwave_encoder: bool | None = None,
    log_prefix: str = "[Runtime]",
) -> FrozenFeatureCacheConfig:
    raw_cfg = training_cfg.get("frozen_feature_cache", None)
    cfg: dict = {}
    enabled_raw: Any = False
    if isinstance(raw_cfg, bool):
        enabled_raw = raw_cfg
    elif isinstance(raw_cfg, dict):
        cfg = raw_cfg
        enabled_raw = raw_cfg.get("enabled", "auto")
    elif raw_cfg is not None:
        print(f"{log_prefix} invalid training.frozen_feature_cache={type(raw_cfg)}, fallback to disabled.")

    if isinstance(enabled_raw, str):
        enabled = enabled_raw.strip().lower() == "true"
        if enabled_raw.strip().lower() == "auto":
            encoder_frozen = freeze_mmwave_encoder if freeze_mmwave_encoder is not None else freeze_point_transformer
            enabled = bool(encoder_frozen and freeze_vqvae)
    else:
        enabled = bool(enabled_raw)

    encoder_frozen = freeze_mmwave_encoder if freeze_mmwave_encoder is not None else freeze_point_transformer
    if enabled and (not encoder_frozen or not freeze_vqvae):
        print(f"{log_prefix} frozen_feature_cache requires frozen mmWave encoder and VQ-VAE; disabling.")
        enabled = False

    path = cfg.get("path", None)
    if path in ("", "null"):
        path = None
    return FrozenFeatureCacheConfig(
        enabled=enabled,
        batch_size=max(1, int(cfg.get("batch_size", max(64, int(training_cfg.get("batch_size", 32)))))),
        num_workers=max(0, int(cfg.get("num_workers", 0))),
        pin_memory=bool(cfg.get("pin_memory", True)),
        prefetch_factor=max(1, int(cfg.get("prefetch_factor", 2))),
        path=(str(path) if path is not None else None),
        reuse=bool(cfg.get("reuse", True)),
        device=str(cfg.get("device", "auto")).strip().lower(),
        max_gpu_cache_fraction=float(cfg.get("max_gpu_cache_fraction", 0.60)),
        batch_loader=str(cfg.get("batch_loader", "auto")).strip().lower(),
    )


def resolve_feature_cache_root(
    *,
    cache_cfg: FrozenFeatureCacheConfig,
    training_cfg: dict,
    reference_dataset: Dataset,
) -> Path:
    if cache_cfg.path:
        return Path(cache_cfg.path)
    base = unwrap_dataset(reference_dataset)
    root = getattr(base, "root", None)
    if isinstance(root, str) and root:
        return Path(root) / ".frozen_feature_cache"
    return Path(str(training_cfg.get("checkpoint_dir", "."))) / "_frozen_feature_cache"


def save_precomputed_feature_dataset(
    dataset: PrecomputedFeatureDataset,
    output_path: str | Path,
    *,
    signature: dict[str, Any] | None = None,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sig = dict(signature or dataset.signature or {})
    torch.save(
        {
            "signature": sig,
            "target_poses": dataset.target_poses.cpu(),
            "cached_mmwave_tokens": dataset.cached_mmwave_tokens.cpu(),
            "cached_target_z_e": dataset.cached_target_z_e.cpu(),
            "cached_target_indices": dataset.cached_target_indices.cpu(),
            "cached_mmwave_token_mask": (
                dataset.cached_mmwave_token_mask.cpu()
                if dataset.cached_mmwave_token_mask is not None
                else None
            ),
        },
        output,
    )


def load_precomputed_feature_dataset(
    input_path: str | Path,
    *,
    expected_signature: dict[str, Any] | None = None,
) -> Optional[PrecomputedFeatureDataset]:
    path = Path(input_path)
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    stored_sig = dict(payload.get("signature", {}))
    if expected_signature is not None and stored_sig != dict(expected_signature):
        raise ValueError(f"Frozen feature cache signature mismatch: {path}")
    return PrecomputedFeatureDataset(
        target_poses=payload["target_poses"],
        cached_mmwave_tokens=payload["cached_mmwave_tokens"],
        cached_target_z_e=payload["cached_target_z_e"],
        cached_target_indices=payload["cached_target_indices"],
        cached_mmwave_token_mask=payload.get("cached_mmwave_token_mask"),
        signature=stored_sig,
    )


def _format_gib(num_bytes: int | float) -> str:
    return f"{float(num_bytes) / (1024.0**3):.2f}GiB"


def materialize_cached_features(
    dataset: PrecomputedFeatureDataset,
    *,
    cache_cfg: FrozenFeatureCacheConfig,
    runtime_device: torch.device,
    log_prefix: str,
    role: str,
) -> PrecomputedFeatureDataset:
    requested = cache_cfg.device
    if requested in {"", "none"}:
        requested = "cpu"
    if requested not in {"auto", "cpu", "cuda"}:
        print(f"{log_prefix} unknown frozen_feature_cache.device={requested!r}; using auto.")
        requested = "auto"

    if requested == "cpu" or runtime_device.type != "cuda":
        return dataset.to("cpu")

    needed = dataset.nbytes()
    if requested == "auto":
        free_bytes, total_bytes = torch.cuda.mem_get_info(runtime_device)
        budget = int(float(cache_cfg.max_gpu_cache_fraction) * float(free_bytes))
        if needed > budget:
            print(
                f"{log_prefix} {role} frozen cache stays on CPU "
                f"(need={_format_gib(needed)}, free={_format_gib(free_bytes)}, "
                f"budget={_format_gib(budget)})."
            )
            return dataset.to("cpu")
        print(
            f"{log_prefix} {role} frozen cache -> cuda "
            f"(need={_format_gib(needed)}, free={_format_gib(free_bytes)}, "
            f"total={_format_gib(total_bytes)})."
        )
        return dataset.to(runtime_device)

    try:
        print(f"{log_prefix} {role} frozen cache -> cuda (forced, need={_format_gib(needed)}).")
        return dataset.to(runtime_device)
    except RuntimeError as exc:
        print(f"{log_prefix} failed to move {role} frozen cache to cuda: {exc}; using CPU cache.")
        if runtime_device.type == "cuda":
            torch.cuda.empty_cache()
        return dataset.to("cpu")


@torch.inference_mode()
def build_precomputed_feature_dataset(
    *,
    model,
    dataset: Dataset,
    device: torch.device,
    runtime,
    cache_cfg: FrozenFeatureCacheConfig,
    desc: str,
    signature: dict[str, Any] | None = None,
) -> PrecomputedFeatureDataset:
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=cache_cfg.batch_size,
        shuffle=False,
        num_workers=cache_cfg.num_workers,
        pin_memory=cache_cfg.pin_memory,
        drop_last=False,
        persistent_workers=(cache_cfg.num_workers > 0),
    )
    if cache_cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = cache_cfg.prefetch_factor
    loader = DataLoader(**loader_kwargs)

    if hasattr(model, "point_transformer"):
        model.point_transformer.eval()
    if hasattr(model, "mmwave_tokenizer"):
        model.mmwave_tokenizer.eval()
    model.vqvae.eval()
    pose_items = []
    mmwave_token_items = []
    target_ze_items = []
    target_idx_items = []
    mmwave_mask_items = []
    for batch in tqdm(loader, desc=desc):
        mmwave_points = batch["mmwave_points"].to(device, non_blocking=True)
        target_poses = batch["target_poses"].to(device, non_blocking=True)
        with autocast_context(runtime):
            if hasattr(model, "encode_mmwave_tokens"):
                encoded = model.encode_mmwave_tokens(mmwave_points)
                mmwave_tokens = encoded["tokens"] if isinstance(encoded, dict) else encoded
                mmwave_mask = encoded.get("mask") if isinstance(encoded, dict) else None
            else:
                _, mmwave_tokens = model.point_transformer(mmwave_points)
                mmwave_mask = None
            target_inter = model.vqvae.encode_with_intermediates(target_poses)
        pose_items.append(target_poses.detach().cpu())
        mmwave_token_items.append(mmwave_tokens.detach().cpu())
        if mmwave_mask is None:
            mmwave_mask_items.append(
                torch.ones(mmwave_tokens.shape[:2], device="cpu", dtype=torch.bool)
            )
        else:
            mmwave_mask_items.append(mmwave_mask.detach().cpu().to(torch.bool))
        target_ze_items.append(target_inter["z_e"].detach().cpu())
        target_idx_items.append(target_inter["indices"].detach().cpu().to(torch.long))

    if not pose_items:
        raise RuntimeError("No samples available for frozen feature cache precompute.")

    return PrecomputedFeatureDataset(
        target_poses=torch.cat(pose_items, dim=0),
        cached_mmwave_tokens=torch.cat(mmwave_token_items, dim=0),
        cached_mmwave_token_mask=torch.cat(mmwave_mask_items, dim=0),
        cached_target_z_e=torch.cat(target_ze_items, dim=0),
        cached_target_indices=torch.cat(target_idx_items, dim=0),
        signature=signature,
    )


def rebuild_loader_like(
    loader: DataLoader,
    dataset: Dataset,
    *,
    shuffle: bool,
    drop_last: bool | None = None,
    use_direct_batch_loader: bool = False,
) -> DataLoader:
    if use_direct_batch_loader:
        return PrecomputedFeatureBatchLoader(
            dataset=dataset,
            batch_size=loader.batch_size,
            shuffle=shuffle,
            drop_last=loader.drop_last if drop_last is None else bool(drop_last),
            generator=loader.generator,
        )

    kwargs = dict(
        dataset=dataset,
        batch_size=loader.batch_size,
        shuffle=shuffle,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last if drop_last is None else bool(drop_last),
        worker_init_fn=loader.worker_init_fn,
        generator=loader.generator,
        persistent_workers=(loader.persistent_workers and loader.num_workers > 0),
        collate_fn=loader.collate_fn,
    )
    prefetch_factor = getattr(loader, "prefetch_factor", None)
    if loader.num_workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**kwargs)


def _use_direct_batch_loader(dataset: PrecomputedFeatureDataset, cache_cfg: FrozenFeatureCacheConfig) -> bool:
    requested = cache_cfg.batch_loader
    if requested in {"1", "true", "yes", "direct"}:
        return True
    if requested in {"0", "false", "no", "dataloader"}:
        return False
    return dataset.device.type == "cuda"


def maybe_enable_frozen_feature_cache(
    *,
    data_bundle,
    model,
    training_cfg: dict,
    runtime,
    device: torch.device,
    log_prefix: str,
    cache_tag: str | None = None,
):
    freeze_encoder = bool(
        training_cfg.get(
            "freeze_point_transformer",
            training_cfg.get("freeze_mmwave_tokenizer", True),
        )
    )
    cache_cfg = resolve_frozen_feature_cache_config(
        training_cfg=training_cfg,
        freeze_point_transformer=bool(training_cfg.get("freeze_point_transformer", freeze_encoder)),
        freeze_mmwave_encoder=freeze_encoder,
        freeze_vqvae=bool(training_cfg.get("freeze_vqvae", True)),
        log_prefix=log_prefix,
    )
    if not cache_cfg.enabled:
        print(f"{log_prefix} frozen_feature_cache=disabled")
        return data_bundle

    start_t = time.perf_counter()
    train_dataset_sig = dataset_signature(data_bundle.train_loader.dataset)
    val_dataset_sig = dataset_signature(data_bundle.val_loader.dataset)
    cache_key_payload = {
        "cache_tag": str(cache_tag or ""),
        "runtime_precision": str(getattr(runtime, "precision", "unknown")),
        "train_dataset_sig": train_dataset_sig,
        "val_dataset_sig": val_dataset_sig,
    }
    cache_hash = hashlib.sha1(
        json.dumps(cache_key_payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    cache_root = resolve_feature_cache_root(
        cache_cfg=cache_cfg,
        training_cfg=training_cfg,
        reference_dataset=data_bundle.train_loader.dataset,
    )
    train_sig = {"role": "train", **cache_key_payload}
    val_sig = {"role": "val", **cache_key_payload}
    train_cache_file = cache_root / f"train_{cache_hash}.pt"
    val_cache_file = cache_root / f"val_{cache_hash}.pt"

    train_cached_dataset = None
    val_cached_dataset = None
    source = "build"
    if cache_cfg.reuse:
        try:
            train_cached_dataset = load_precomputed_feature_dataset(train_cache_file, expected_signature=train_sig)
            val_cached_dataset = load_precomputed_feature_dataset(val_cache_file, expected_signature=val_sig)
        except ValueError as exc:
            print(f"{log_prefix} frozen feature cache rejected: {exc}")
            train_cached_dataset = None
            val_cached_dataset = None
        if (train_cached_dataset is None) != (val_cached_dataset is None):
            train_cached_dataset = None
            val_cached_dataset = None
        if train_cached_dataset is not None and val_cached_dataset is not None:
            source = "disk"

    if train_cached_dataset is None or val_cached_dataset is None:
        train_cached_dataset = build_precomputed_feature_dataset(
            model=model,
            dataset=data_bundle.train_loader.dataset,
            device=device,
            runtime=runtime,
            cache_cfg=cache_cfg,
            desc=f"{log_prefix} Build Frozen Cache [train]",
            signature=train_sig,
        )
        val_cached_dataset = build_precomputed_feature_dataset(
            model=model,
            dataset=data_bundle.val_loader.dataset,
            device=device,
            runtime=runtime,
            cache_cfg=cache_cfg,
            desc=f"{log_prefix} Build Frozen Cache [val]",
            signature=val_sig,
        )
        if cache_cfg.reuse:
            try:
                save_precomputed_feature_dataset(train_cached_dataset, train_cache_file, signature=train_sig)
                save_precomputed_feature_dataset(val_cached_dataset, val_cache_file, signature=val_sig)
                source = "build+save"
            except Exception as exc:
                source = "build"
                print(f"{log_prefix} failed to save frozen feature cache: {exc}")

    train_cached_dataset = materialize_cached_features(
        train_cached_dataset,
        cache_cfg=cache_cfg,
        runtime_device=device,
        log_prefix=log_prefix,
        role="train",
    )
    val_cached_dataset = materialize_cached_features(
        val_cached_dataset,
        cache_cfg=cache_cfg,
        runtime_device=device,
        log_prefix=log_prefix,
        role="val",
    )
    train_direct = _use_direct_batch_loader(train_cached_dataset, cache_cfg)
    val_direct = _use_direct_batch_loader(val_cached_dataset, cache_cfg)
    train_loader = rebuild_loader_like(
        data_bundle.train_loader,
        train_cached_dataset,
        shuffle=True,
        drop_last=True,
        use_direct_batch_loader=train_direct,
    )
    val_loader = rebuild_loader_like(
        data_bundle.val_loader,
        val_cached_dataset,
        shuffle=False,
        drop_last=False,
        use_direct_batch_loader=val_direct,
    )
    elapsed = time.perf_counter() - start_t
    print(
        f"{log_prefix} frozen_feature_cache=enabled | source={source}, prep={elapsed:.1f}s, "
        f"train={len(train_cached_dataset)}, val={len(val_cached_dataset)}, "
        f"build_bs={cache_cfg.batch_size}, workers={cache_cfg.num_workers}, "
        f"train_device={train_cached_dataset.device}, val_device={val_cached_dataset.device}, "
        f"direct_loader={train_direct}/{val_direct}, cache_root={cache_root}"
    )
    return data_bundle.__class__(
        train_loader=train_loader,
        val_loader=val_loader,
        train_count=len(train_cached_dataset),
        val_count=len(val_cached_dataset),
        split_manifest=getattr(data_bundle, "split_manifest", {}),
        test_loader=getattr(data_bundle, "test_loader", None),
        test_count=getattr(data_bundle, "test_count", 0),
    )
