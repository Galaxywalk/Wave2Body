# -*- coding: utf-8 -*-
"""Registry for human-body joint-only datasets."""

from __future__ import annotations

from typing import Callable


DATASET_REGISTRY: dict[str, type] = {}


def register_dataset(name: str) -> Callable[[type], type]:
    """Decorator to register dataset classes by config name."""

    def decorator(cls: type) -> type:
        DATASET_REGISTRY[str(name)] = cls
        return cls

    return decorator


__all__ = ["DATASET_REGISTRY", "register_dataset"]
