# -*- coding: utf-8 -*-
"""Shared training helpers for experiment runners."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml

try:
    from wave2body.utils.paths import artifact_root, resolve_dataset_paths
except ImportError:  # pragma: no cover - package import fallback
    from ..utils.paths import artifact_root, resolve_dataset_paths


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries without mutating either input."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def expand_env_vars(value: Any) -> Any:
    """Expand environment variables and user home markers in nested config data."""
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_env_vars(item) for item in value)
    if isinstance(value, Mapping):
        return {key: expand_env_vars(item) for key, item in value.items()}
    return value


def load_config(config_path: str | Path, *, expand_env: bool = True, resolve_datasets: bool = True) -> dict[str, Any]:
    """Load YAML config with optional ``base_config`` recursive merge support."""
    path = Path(config_path).expanduser()
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise TypeError(f"Config must be a dict at {path}, got {type(config)}")

    base_ref = config.pop("base_config", None)
    if base_ref:
        base_path = Path(str(base_ref)).expanduser()
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        config = deep_merge(load_config(base_path, expand_env=expand_env, resolve_datasets=False), config)

    if expand_env:
        config = expand_env_vars(config)
    return resolve_dataset_paths(config) if resolve_datasets else config


def resolve_path(base: str | Path, path_like: str | Path | None, *, prefer_existing: bool = True) -> Path | None:
    """Resolve a possibly relative path against cwd first, then ``base``."""
    if path_like in (None, "", "null"):
        return None
    base_path = Path(base).expanduser()
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p
    cwd_candidate = p.resolve()
    base_candidate = (base_path / p).resolve()
    if prefer_existing:
        if cwd_candidate.exists():
            return cwd_candidate
        if base_candidate.exists():
            return base_candidate
    return base_candidate


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and Torch RNGs."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _iter_params(params_or_module: Any) -> Iterable[torch.nn.Parameter]:
    if isinstance(params_or_module, torch.nn.Module):
        return params_or_module.parameters()
    return params_or_module


def safe_param_count(params_or_module: Any, *, trainable_only: bool = False) -> int:
    """Count initialized parameters and skip lazy/uninitialized tensors safely."""
    total = 0
    for param in _iter_params(params_or_module):
        try:
            if getattr(param, "is_lazy", False):
                continue
            if getattr(param, "_is_uninitialized", False):
                continue
            if trainable_only and not bool(param.requires_grad):
                continue
            total += int(param.numel())
        except (RuntimeError, ValueError):
            continue
    return total


def load_optional_model_states_from_checkpoint(model: torch.nn.Module, ckpt: Mapping[str, Any], prefix: str = "") -> None:
    """Load standard optional PT/VQ-VAE/translator states from a checkpoint."""
    tag = f"{prefix} " if prefix else ""
    state_targets = (
        ("point_transformer_state_dict", "point_transformer"),
        ("vqvae_state_dict", "vqvae"),
        ("token_translator_state_dict", "token_translator"),
        ("root_head_state_dict", "root_head"),
    )
    for state_key, attr in state_targets:
        module = getattr(model, attr, None)
        if module is None:
            continue
        if state_key not in ckpt:
            print(f"{tag}{state_key} not found in checkpoint")
            continue
        state = dict(ckpt[state_key])
        for wrapped_prefix in ("module.", "_orig_mod."):
            if state and all(k.startswith(wrapped_prefix) for k in state):
                state = {k[len(wrapped_prefix) :]: v for k, v in state.items()}
        missing, unexpected = module.load_state_dict(state, strict=False)
        print(f"{tag}Loaded {state_key} (missing={len(missing)}, unexpected={len(unexpected)})")


def file_sha256(path: str | Path | None) -> str | None:
    """Return SHA256 for a file, or None when no readable file is available."""
    if path in (None, "", "null"):
        return None
    p = Path(path).expanduser()
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=str(cwd), stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def git_state(cwd: str | Path) -> dict[str, Any]:
    """Return lightweight git provenance for a run manifest."""
    root_text = _run_git(["rev-parse", "--show-toplevel"], Path(cwd))
    root = Path(root_text) if root_text else Path(cwd).resolve()
    commit = _run_git(["rev-parse", "HEAD"], root)
    status = _run_git(["status", "--porcelain"], root) or ""
    diff = _run_git(["diff", "--binary"], root) or ""
    staged = _run_git(["diff", "--cached", "--binary"], root) or ""
    dirty_hash = hashlib.sha1((status + "\n" + diff + "\n" + staged).encode("utf-8")).hexdigest()
    return {
        "root": str(root),
        "commit": commit,
        "dirty": bool(status),
        "dirty_hash": dirty_hash if status else None,
    }


def write_run_manifest(
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    dataset_split: str | None = None,
    coord_transform: str | None = None,
    output_paths: Mapping[str, str | Path] | None = None,
    extra: Mapping[str, Any] | None = None,
    filename: str = "run_manifest.json",
) -> Path:
    """Write a machine-readable manifest for a training/evaluation run."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path_obj = Path(config_path).expanduser().resolve() if config_path not in (None, "", "null") else None
    checkpoint_path_obj = (
        Path(checkpoint_path).expanduser().resolve() if checkpoint_path not in (None, "", "null") else None
    )
    cfg = dict(config or {})
    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset", {}), Mapping) else {}
    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [str(x) for x in sys.argv],
        "cwd": str(Path.cwd()),
        "git": git_state(Path.cwd()),
        "config": {
            "path": str(config_path_obj) if config_path_obj else None,
            "sha256": file_sha256(config_path_obj),
        },
        "checkpoint": {
            "path": str(checkpoint_path_obj) if checkpoint_path_obj else None,
            "sha256": file_sha256(checkpoint_path_obj),
        },
        "dataset": {
            "name": dataset_cfg.get("name"),
            "split": dataset_split,
            "dataset_path": cfg.get("dataset_path"),
        },
        "coord_transform": coord_transform if coord_transform is not None else cfg.get("coord_transform"),
        "environment": {
            "python": sys.version.replace("\n", " "),
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "wave2body_artifact_root": os.environ.get("WAVE2BODY_ARTIFACT_ROOT"),
        },
        "output_paths": {k: str(v) for k, v in dict(output_paths or {}).items()},
    }
    if extra:
        manifest["extra"] = dict(extra)

    path = out_dir / filename
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return path
