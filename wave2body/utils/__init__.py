# -*- coding: utf-8 -*-
"""Utility modules."""

from __future__ import annotations

__all__ = ["compute_mpjpe", "compute_p_mpjpe"]


def __getattr__(name: str):
    if name in __all__:
        from .metrics import compute_mpjpe, compute_p_mpjpe

        values = {
            "compute_mpjpe": compute_mpjpe,
            "compute_p_mpjpe": compute_p_mpjpe,
        }
        globals().update(values)
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
