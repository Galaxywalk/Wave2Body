# -*- coding: utf-8 -*-
"""Repository and runtime path helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml


def repo_root() -> Path:
    """Return the repository root for an installed or editable checkout."""
    return Path(__file__).resolve().parents[2]


DEFAULT_WORKSPACE_ROOT = repo_root() / "workspace"
DEFAULT_ARTIFACT_ROOT = DEFAULT_WORKSPACE_ROOT / "wave2body_artifacts"


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return value
    return None


def _expand_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser()


def workspace_root(default: str | Path | None = None) -> Path:
    """Resolve the root directory that contains artifacts, VQ-VAE runs, and caches."""
    value = _first_env("WAVE2BODY_WORKSPACE_ROOT")
    if value is not None:
        return _expand_path(value)

    artifact_value = _first_env("WAVE2BODY_ARTIFACT_ROOT")
    if artifact_value is not None:
        return _expand_path(artifact_value).parent

    if default not in (None, ""):
        return _expand_path(default)
    return DEFAULT_WORKSPACE_ROOT


def artifact_root(default: str | Path | None = None) -> Path:
    """Resolve the root directory for run artifacts and generated configs."""
    value = _first_env("WAVE2BODY_ARTIFACT_ROOT")
    if value is not None:
        return _expand_path(value)
    if default not in (None, ""):
        return _expand_path(default)
    return workspace_root() / "wave2body_artifacts"


DATASET_URI_PREFIX = "dataset://"
ARTIFACT_URI_PREFIX = "artifact://"
WORKSPACE_URI_PREFIX = "workspace://"


def default_dataset_path_config() -> Path:
    """Resolve the machine-local dataset path map config."""
    value = _first_env("WAVE2BODY_DATASET_PATHS")
    if value is not None:
        return _expand_path(value)

    root = repo_root()
    local_path = root / "configs" / "dataset_paths.local.yaml"
    if local_path.exists():
        return local_path
    return root / "configs" / "dataset_paths.example.yaml"


def load_dataset_path_map(config_path: str | Path | None = None) -> dict[str, str]:
    """Load dataset path aliases from the configured YAML map."""
    path = _expand_path(config_path) if config_path else default_dataset_path_config()
    if not path.exists():
        return {}

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise TypeError(f"Dataset path config must be a mapping: {path}")

    datasets = data.get("datasets", data)
    if not isinstance(datasets, Mapping):
        raise TypeError(f"Dataset path config must contain a mapping under `datasets`: {path}")

    resolved: dict[str, str] = {}
    for name, spec in datasets.items():
        value = spec.get("path") if isinstance(spec, Mapping) else spec
        if value in (None, "", "null"):
            continue
        resolved[str(name)] = str(_expand_path(value))
    return resolved


def resolve_dataset_uri(value: str, path_map: Mapping[str, str] | None = None) -> str:
    """Resolve a ``dataset://name`` URI through the dataset path map."""
    if not value.startswith(DATASET_URI_PREFIX):
        return value

    rest = value[len(DATASET_URI_PREFIX) :]
    key, sep, suffix = rest.partition("/")
    if not key:
        raise ValueError(f"Invalid dataset URI: {value!r}")

    paths = path_map if path_map is not None else load_dataset_path_map()
    if key not in paths:
        available = ", ".join(sorted(paths)) or "<none>"
        raise KeyError(f"Unknown dataset path key {key!r} in {value!r}; available keys: {available}")

    base = Path(paths[key]).expanduser()
    return str(base / suffix) if sep else str(base)


def resolve_runtime_uri(value: str) -> str:
    """Resolve Wave2Body runtime-root URIs used by portable configs."""
    if value.startswith(ARTIFACT_URI_PREFIX):
        suffix = value[len(ARTIFACT_URI_PREFIX) :].lstrip("/")
        return str(artifact_root() / suffix)
    if value.startswith(WORKSPACE_URI_PREFIX):
        suffix = value[len(WORKSPACE_URI_PREFIX) :].lstrip("/")
        return str(workspace_root() / suffix)
    return value


def resolve_dataset_paths(value: Any, path_map: Mapping[str, str] | None = None) -> Any:
    """Recursively resolve dataset/runtime URIs in nested config values."""
    if isinstance(value, str):
        return resolve_runtime_uri(resolve_dataset_uri(value, path_map=path_map))
    if isinstance(value, list):
        return [resolve_dataset_paths(item, path_map=path_map) for item in value]
    if isinstance(value, tuple):
        return tuple(resolve_dataset_paths(item, path_map=path_map) for item in value)
    if isinstance(value, Mapping):
        return {key: resolve_dataset_paths(item, path_map=path_map) for key, item in value.items()}
    return value


def python_executable() -> str:
    """Resolve the Python executable used for launcher subprocesses."""
    return _first_env("WAVE2BODY_PYTHON") or sys.executable


__all__ = [
    "DATASET_URI_PREFIX",
    "ARTIFACT_URI_PREFIX",
    "DEFAULT_ARTIFACT_ROOT",
    "DEFAULT_WORKSPACE_ROOT",
    "WORKSPACE_URI_PREFIX",
    "artifact_root",
    "default_dataset_path_config",
    "load_dataset_path_map",
    "python_executable",
    "resolve_dataset_paths",
    "resolve_dataset_uri",
    "resolve_runtime_uri",
    "repo_root",
    "workspace_root",
]
