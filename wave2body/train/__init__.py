# -*- coding: utf-8 -*-
"""Training module."""

from __future__ import annotations

__all__ = [
    "load_config",
    "resolve_path",
    "safe_param_count",
    "set_global_seed",
    "write_run_manifest",
]


def __getattr__(name: str):
    if name in __all__:
        from .common import load_config, resolve_path, safe_param_count, set_global_seed, write_run_manifest

        values = {
            "load_config": load_config,
            "resolve_path": resolve_path,
            "safe_param_count": safe_param_count,
            "set_global_seed": set_global_seed,
            "write_run_manifest": write_run_manifest,
        }
        globals().update(values)
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
