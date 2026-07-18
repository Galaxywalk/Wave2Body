# -*- coding: utf-8 -*-
"""M4Human upstream test action-group evaluation accumulators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch

try:
    from ..dataset.radar_pose.m4human_protocols import (
        M4HUMAN_ACTION_GROUP_ORDER,
        action_indices_for_group,
        summarize_metric_by_action_group,
    )
except ImportError:  # Support scripts that put src/ directly on sys.path.
    from wave2body.dataset.radar_pose.m4human_protocols import (
        M4HUMAN_ACTION_GROUP_ORDER,
        action_indices_for_group,
        summarize_metric_by_action_group,
    )


def m4human_action_ids_from_indicator(indicator, *, device: torch.device | None = None) -> torch.Tensor:
    """Extract 1-based M4Human action ids from batched indicators."""

    if torch.is_tensor(indicator):
        values = indicator.to(device=device, non_blocking=True).long()
    else:
        values = torch.as_tensor(indicator, device=device, dtype=torch.long)
    if values.dim() != 2 or int(values.shape[1]) < 2:
        raise ValueError(f"Expected M4Human indicator [B,>=2], got shape={tuple(values.shape)}")
    return values[:, 1]


@dataclass
class M4HumanActionGroupAccumulator:
    """Accumulate sample-level metrics by upstream M4Human test action groups.

    Inputs are per-sample metric values in the caller's unit, usually meters for
    MPJPE/P-MPJPE. ``summary(scale=1000)`` converts meter inputs to millimeters.
    """

    metric_names: Iterable[str]
    device: torch.device | None = None
    _sum_by_action: dict[str, torch.Tensor] = field(init=False, repr=False)
    _count_by_action: dict[str, torch.Tensor] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.metric_names)
        if not names:
            raise ValueError("metric_names must not be empty")
        self.metric_names = names
        self._sum_by_action = {
            name: torch.zeros(50, device=self.device, dtype=torch.float64) for name in names
        }
        self._count_by_action = {
            name: torch.zeros(50, device=self.device, dtype=torch.float64) for name in names
        }

    def update(self, metric_name: str, action_ids_1based, values) -> None:
        """Add per-sample values for one metric.

        ``action_ids_1based`` and ``values`` must both be one-dimensional and
        have matching lengths. Invalid action ids are ignored.
        """

        if metric_name not in self._sum_by_action:
            raise KeyError(f"Unknown M4Human action-group metric: {metric_name!r}")
        action_ids = torch.as_tensor(action_ids_1based, device=self.device, dtype=torch.long).reshape(-1)
        metric_values = torch.as_tensor(values, device=self.device, dtype=torch.float64).reshape(-1)
        if action_ids.numel() != metric_values.numel():
            raise ValueError(
                f"action/value count mismatch: {int(action_ids.numel())} vs {int(metric_values.numel())}"
            )
        action_idx = action_ids - 1
        valid = (action_idx >= 0) & (action_idx < 50)
        if not bool(valid.any().item()):
            return
        ids = action_idx[valid]
        vals = metric_values[valid]
        self._sum_by_action[metric_name].index_add_(0, ids, vals)
        self._count_by_action[metric_name].index_add_(0, ids, torch.ones_like(vals, dtype=torch.float64))

    def summary(self, *, scale: float = 1.0, include_per_action: bool = True) -> dict[str, dict[str, object]]:
        """Return JSON-serializable group metrics for each accumulated metric."""

        out: dict[str, dict[str, object]] = {}
        for name in self.metric_names:
            sums = self._sum_by_action[name].detach().cpu().numpy()
            counts = self._count_by_action[name].detach().cpu().numpy()
            by_group, group_counts = summarize_metric_by_action_group(sums, counts, scale=scale)
            item: dict[str, object] = {
                "by_group": by_group,
                "num_samples_by_group": group_counts,
            }
            if include_per_action:
                per_action: dict[str, float | None] = {}
                for idx in range(50):
                    count = float(counts[idx])
                    if count <= 0:
                        per_action[str(idx + 1)] = None
                    else:
                        per_action[str(idx + 1)] = float(sums[idx] / count * float(scale))
                item["per_action"] = per_action
            out[name] = item
        return out


def summarize_vector_metric_by_action_group(
    sum_by_action: torch.Tensor | np.ndarray,
    count_by_action: torch.Tensor | np.ndarray,
    *,
    scale: float = 1.0,
) -> tuple[dict[str, list[float] | None], dict[str, int]]:
    """Return sample-micro-averaged vector values for semantic M4Human groups."""

    if torch.is_tensor(sum_by_action):
        sums = sum_by_action.detach().cpu().numpy()
    else:
        sums = np.asarray(sum_by_action)
    if torch.is_tensor(count_by_action):
        counts = count_by_action.detach().cpu().numpy()
    else:
        counts = np.asarray(count_by_action)
    sums = np.asarray(sums, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if sums.ndim != 2 or sums.shape[0] != 50:
        raise ValueError(f"Expected vector sums [50,J], got {sums.shape}")
    if counts.shape != (50,):
        raise ValueError(f"Expected 50 action counts, got {counts.shape}")

    metrics: dict[str, list[float] | None] = {}
    sample_counts: dict[str, int] = {}
    for name in M4HUMAN_ACTION_GROUP_ORDER:
        idx = action_indices_for_group(name)
        total_count = float(np.nansum(counts[idx]))
        sample_counts[name] = int(round(total_count))
        if total_count <= 0:
            metrics[name] = None
        else:
            values = np.nansum(sums[idx], axis=0) / total_count * float(scale)
            metrics[name] = [float(item) for item in values.tolist()]
    return metrics, sample_counts


@dataclass
class M4HumanActionGroupVectorAccumulator:
    """Accumulate per-joint metric vectors by semantic M4Human action groups."""

    metric_names: Iterable[str]
    device: torch.device | None = None
    _sum_by_action: dict[str, torch.Tensor | None] = field(init=False, repr=False)
    _count_by_action: dict[str, torch.Tensor] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.metric_names)
        if not names:
            raise ValueError("metric_names must not be empty")
        self.metric_names = names
        self._sum_by_action = {name: None for name in names}
        self._count_by_action = {
            name: torch.zeros(50, device=self.device, dtype=torch.float64) for name in names
        }

    def update(self, metric_name: str, action_ids_1based, values) -> None:
        """Add per-sample vector values with shape ``[B, J]`` for one metric."""

        if metric_name not in self._sum_by_action:
            raise KeyError(f"Unknown M4Human action-group vector metric: {metric_name!r}")
        action_ids = torch.as_tensor(action_ids_1based, device=self.device, dtype=torch.long).reshape(-1)
        metric_values = torch.as_tensor(values, device=self.device, dtype=torch.float64)
        if metric_values.dim() != 2:
            raise ValueError(f"Expected vector metric values [B,J], got shape={tuple(metric_values.shape)}")
        if action_ids.numel() != int(metric_values.shape[0]):
            raise ValueError(
                f"action/value count mismatch: {int(action_ids.numel())} vs {int(metric_values.shape[0])}"
            )
        existing = self._sum_by_action[metric_name]
        if existing is None:
            self._sum_by_action[metric_name] = torch.zeros(
                (50, int(metric_values.shape[1])),
                device=self.device,
                dtype=torch.float64,
            )
        elif int(existing.shape[1]) != int(metric_values.shape[1]):
            raise ValueError(
                f"joint count mismatch for {metric_name}: {int(existing.shape[1])} vs {int(metric_values.shape[1])}"
            )

        action_idx = action_ids - 1
        valid = (action_idx >= 0) & (action_idx < 50)
        if not bool(valid.any().item()):
            return
        ids = action_idx[valid]
        vals = metric_values[valid]
        sums = self._sum_by_action[metric_name]
        assert sums is not None
        sums.index_add_(0, ids, vals)
        self._count_by_action[metric_name].index_add_(
            0,
            ids,
            torch.ones(int(vals.shape[0]), device=self.device, dtype=torch.float64),
        )

    def summary(self, *, scale: float = 1.0) -> dict[str, dict[str, object]]:
        """Return JSON-serializable per-group vector metrics."""

        out: dict[str, dict[str, object]] = {}
        for name in self.metric_names:
            sums = self._sum_by_action[name]
            if sums is None:
                continue
            by_group, group_counts = summarize_vector_metric_by_action_group(
                sums,
                self._count_by_action[name],
                scale=scale,
            )
            out[name] = {
                "by_group": by_group,
                "num_samples_by_group": group_counts,
            }
        return out


__all__ = [
    "M4HumanActionGroupAccumulator",
    "M4HumanActionGroupVectorAccumulator",
    "m4human_action_ids_from_indicator",
    "summarize_vector_metric_by_action_group",
]
