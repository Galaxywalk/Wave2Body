# -*- coding: utf-8 -*-
"""Action-group-aware low-error qualitative sample selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..dataset.radar_pose.m4human_protocols import M4HUMAN_ACTION_GROUPS


DEFAULT_QUALITATIVE_GROUPS: tuple[str, ...] = ("1_30", "31_35", "36_50")
ALL_M4HUMAN_ACTION_IDS: tuple[int, ...] = tuple(range(1, 51))


@dataclass
class SampleCandidate:
    """A qualitative sample candidate with arrays needed for rendering."""

    action_group: str
    action_id: int
    subject_id: int
    frame_id: int
    global_index: int
    absolute_mpjpe_mm: float
    root_error_mm: float
    root_aligned_mpjpe_mm: float
    mmwave_points: np.ndarray
    gt_pose: np.ndarray
    pred_pose: np.ndarray
    indicator: np.ndarray
    temporal_indicators: np.ndarray | None = None
    split_id: str | None = None
    eval_split: str | None = None
    config: str | None = None
    checkpoint: str | None = None

    def record(self, *, npz_path: str | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "action_group": self.action_group,
            "action_id": int(self.action_id),
            "subject_id": int(self.subject_id),
            "frame_id": int(self.frame_id),
            "global_index": int(self.global_index),
            "absolute_mpjpe_mm": float(self.absolute_mpjpe_mm),
            "root_error_mm": float(self.root_error_mm),
            "root_aligned_mpjpe_mm": float(self.root_aligned_mpjpe_mm),
            "split_id": self.split_id,
            "eval_split": self.eval_split,
            "config": self.config,
            "checkpoint": self.checkpoint,
        }
        if npz_path is not None:
            out["npz_path"] = str(npz_path)
        return out

    def save_npz(self, path: str | Path) -> None:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            mmwave_points=self.mmwave_points.astype(np.float32, copy=False),
            gt_pose=self.gt_pose.astype(np.float32, copy=False),
            pred_pose=self.pred_pose.astype(np.float32, copy=False),
            indicator=self.indicator.astype(np.int64, copy=False),
            temporal_indicators=(
                np.asarray(self.temporal_indicators, dtype=np.int64)
                if self.temporal_indicators is not None
                else np.empty((0, 3), dtype=np.int64)
            ),
            absolute_mpjpe_mm=np.asarray(self.absolute_mpjpe_mm, dtype=np.float32),
            root_error_mm=np.asarray(self.root_error_mm, dtype=np.float32),
            root_aligned_mpjpe_mm=np.asarray(self.root_aligned_mpjpe_mm, dtype=np.float32),
            action_group=np.asarray(self.action_group),
            split_id=np.asarray(self.split_id or ""),
            eval_split=np.asarray(self.eval_split or ""),
            config=np.asarray(self.config or ""),
            checkpoint=np.asarray(self.checkpoint or ""),
        )


@dataclass
class PairedComparisonCandidate:
    """A same-frame Wave2Body vs WiCompass/PT qualitative candidate."""

    action_group: str
    action_id: int
    subject_id: int
    frame_id: int
    global_index: int
    wave2body_mpjpe_mm: float
    wicompass_mpjpe_mm: float
    wave2body_root_error_mm: float
    wicompass_root_error_mm: float
    wave2body_root_aligned_mpjpe_mm: float
    wicompass_root_aligned_mpjpe_mm: float
    mmwave_points: np.ndarray
    gt_pose: np.ndarray
    wave2body_pred_pose: np.ndarray
    wicompass_pred_pose: np.ndarray
    indicator: np.ndarray
    temporal_indicators: np.ndarray | None = None
    split_id: str | None = None
    eval_split: str | None = None
    wave2body_config: str | None = None
    wave2body_checkpoint: str | None = None
    wicompass_config: str | None = None
    wicompass_checkpoint: str | None = None

    @property
    def delta_mpjpe_mm(self) -> float:
        return float(self.wicompass_mpjpe_mm - self.wave2body_mpjpe_mm)

    def record(self, *, npz_path: str | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "split_id": self.split_id,
            "eval_split": self.eval_split,
            "action_group": self.action_group,
            "action_id": int(self.action_id),
            "subject_id": int(self.subject_id),
            "frame_id": int(self.frame_id),
            "global_index": int(self.global_index),
            "wave2body_mpjpe_mm": float(self.wave2body_mpjpe_mm),
            "wicompass_mpjpe_mm": float(self.wicompass_mpjpe_mm),
            "delta_mpjpe_mm": float(self.delta_mpjpe_mm),
            "wave2body_root_error_mm": float(self.wave2body_root_error_mm),
            "wicompass_root_error_mm": float(self.wicompass_root_error_mm),
            "wave2body_root_aligned_mpjpe_mm": float(self.wave2body_root_aligned_mpjpe_mm),
            "wicompass_root_aligned_mpjpe_mm": float(self.wicompass_root_aligned_mpjpe_mm),
            "wave2body_config": self.wave2body_config,
            "wave2body_checkpoint": self.wave2body_checkpoint,
            "wicompass_config": self.wicompass_config,
            "wicompass_checkpoint": self.wicompass_checkpoint,
        }
        if npz_path is not None:
            out["npz_path"] = str(npz_path)
        return out

    def save_npz(self, path: str | Path) -> None:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            mmwave_points=self.mmwave_points.astype(np.float32, copy=False),
            gt_pose=self.gt_pose.astype(np.float32, copy=False),
            wave2body_pred_pose=self.wave2body_pred_pose.astype(np.float32, copy=False),
            wicompass_pred_pose=self.wicompass_pred_pose.astype(np.float32, copy=False),
            indicator=self.indicator.astype(np.int64, copy=False),
            temporal_indicators=(
                np.asarray(self.temporal_indicators, dtype=np.int64)
                if self.temporal_indicators is not None
                else np.empty((0, 3), dtype=np.int64)
            ),
            wave2body_mpjpe_mm=np.asarray(self.wave2body_mpjpe_mm, dtype=np.float32),
            wicompass_mpjpe_mm=np.asarray(self.wicompass_mpjpe_mm, dtype=np.float32),
            delta_mpjpe_mm=np.asarray(self.delta_mpjpe_mm, dtype=np.float32),
            wave2body_root_error_mm=np.asarray(self.wave2body_root_error_mm, dtype=np.float32),
            wicompass_root_error_mm=np.asarray(self.wicompass_root_error_mm, dtype=np.float32),
            wave2body_root_aligned_mpjpe_mm=np.asarray(self.wave2body_root_aligned_mpjpe_mm, dtype=np.float32),
            wicompass_root_aligned_mpjpe_mm=np.asarray(self.wicompass_root_aligned_mpjpe_mm, dtype=np.float32),
            action_group=np.asarray(self.action_group),
            split_id=np.asarray(self.split_id or ""),
            eval_split=np.asarray(self.eval_split or ""),
            wave2body_config=np.asarray(self.wave2body_config or ""),
            wave2body_checkpoint=np.asarray(self.wave2body_checkpoint or ""),
            wicompass_config=np.asarray(self.wicompass_config or ""),
            wicompass_checkpoint=np.asarray(self.wicompass_checkpoint or ""),
        )


def action_group_for_action(action_id: int, groups: tuple[str, ...] = DEFAULT_QUALITATIVE_GROUPS) -> str | None:
    """Return the first configured semantic action group containing ``action_id``."""

    action = int(action_id)
    for group in groups:
        if action in M4HUMAN_ACTION_GROUPS[group]:
            return group
    return None


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _metric_to_mm(value: Any) -> np.ndarray:
    return _to_numpy(value).reshape(-1).astype(np.float64, copy=False) * 1000.0


class PairedAdvantageSampleSelector:
    """Keep difficult same-frame samples where Wave2Body beats WiCompass/PT."""

    def __init__(
        self,
        *,
        top_k: int = 2,
        action_groups: tuple[str, ...] = ("31_35", "36_50"),
        min_wicompass_mpjpe_mm: float = 100.0,
        min_delta_mpjpe_mm: float = 5.0,
        max_per_action: int = 1,
        max_per_subject: int = 1,
        min_frame_gap: int = 30,
        pool_size: int | None = None,
        split_id: str | None = None,
        eval_split: str | None = None,
        wave2body_config: str | None = None,
        wave2body_checkpoint: str | None = None,
        wicompass_config: str | None = None,
        wicompass_checkpoint: str | None = None,
    ) -> None:
        if int(top_k) <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        for group in action_groups:
            if group not in M4HUMAN_ACTION_GROUPS:
                raise KeyError(f"Unknown M4Human action group: {group!r}")
        self.top_k = int(top_k)
        self.action_groups = tuple(action_groups)
        self.min_wicompass_mpjpe_mm = float(min_wicompass_mpjpe_mm)
        self.min_delta_mpjpe_mm = float(min_delta_mpjpe_mm)
        self.max_per_action = int(max_per_action)
        self.max_per_subject = int(max_per_subject)
        self.min_frame_gap = int(min_frame_gap)
        self.pool_size = int(pool_size or max(50, self.top_k * 50))
        self.split_id = split_id
        self.eval_split = eval_split
        self.wave2body_config = wave2body_config
        self.wave2body_checkpoint = wave2body_checkpoint
        self.wicompass_config = wicompass_config
        self.wicompass_checkpoint = wicompass_checkpoint
        self._pools: dict[str, list[PairedComparisonCandidate]] = {
            group: [] for group in self.action_groups
        }
        self._seen_count = 0
        self._eligible_count = 0

    @property
    def seen_count(self) -> int:
        return self._seen_count

    @property
    def eligible_count(self) -> int:
        return self._eligible_count

    def update_from_batch(
        self,
        *,
        batch: dict[str, Any],
        wave2body_pred_pose: Any,
        wicompass_pred_pose: Any,
        gt_pose: Any,
        mmwave_points: Any,
        wave2body_mpjpe: Any,
        wicompass_mpjpe: Any,
        wave2body_root_error: Any,
        wicompass_root_error: Any,
        wave2body_root_aligned_mpjpe: Any,
        wicompass_root_aligned_mpjpe: Any,
    ) -> None:
        indicators = _to_numpy(batch["indicator"]).astype(np.int64, copy=False)
        temporal = _to_numpy(batch["temporal_indicators"]).astype(np.int64, copy=False) if "temporal_indicators" in batch else None
        wave_pred = _to_numpy(wave2body_pred_pose).astype(np.float32, copy=False)
        pt_pred = _to_numpy(wicompass_pred_pose).astype(np.float32, copy=False)
        gt = _to_numpy(gt_pose).astype(np.float32, copy=False)
        points = _to_numpy(mmwave_points).astype(np.float32, copy=False)
        wave_mpjpe = _metric_to_mm(wave2body_mpjpe)
        pt_mpjpe = _metric_to_mm(wicompass_mpjpe)
        wave_root = _metric_to_mm(wave2body_root_error)
        pt_root = _metric_to_mm(wicompass_root_error)
        wave_root_aligned = _metric_to_mm(wave2body_root_aligned_mpjpe)
        pt_root_aligned = _metric_to_mm(wicompass_root_aligned_mpjpe)

        batch_size = int(indicators.shape[0])
        for local_idx in range(batch_size):
            indicator = indicators[local_idx]
            action_id = int(indicator[1])
            group = action_group_for_action(action_id, self.action_groups)
            if group is None:
                continue
            delta = float(pt_mpjpe[local_idx] - wave_mpjpe[local_idx])
            if float(pt_mpjpe[local_idx]) < self.min_wicompass_mpjpe_mm:
                continue
            if delta < self.min_delta_mpjpe_mm:
                continue
            self._eligible_count += 1
            candidate = PairedComparisonCandidate(
                action_group=group,
                action_id=action_id,
                subject_id=int(indicator[0]),
                frame_id=int(indicator[2]),
                global_index=self._seen_count + local_idx,
                wave2body_mpjpe_mm=float(wave_mpjpe[local_idx]),
                wicompass_mpjpe_mm=float(pt_mpjpe[local_idx]),
                wave2body_root_error_mm=float(wave_root[local_idx]),
                wicompass_root_error_mm=float(pt_root[local_idx]),
                wave2body_root_aligned_mpjpe_mm=float(wave_root_aligned[local_idx]),
                wicompass_root_aligned_mpjpe_mm=float(pt_root_aligned[local_idx]),
                mmwave_points=np.asarray(points[local_idx], dtype=np.float32).copy(),
                gt_pose=np.asarray(gt[local_idx], dtype=np.float32).copy(),
                wave2body_pred_pose=np.asarray(wave_pred[local_idx], dtype=np.float32).copy(),
                wicompass_pred_pose=np.asarray(pt_pred[local_idx], dtype=np.float32).copy(),
                indicator=np.asarray(indicator, dtype=np.int64).copy(),
                temporal_indicators=(
                    np.asarray(temporal[local_idx], dtype=np.int64).copy()
                    if temporal is not None and int(temporal.shape[0]) > local_idx
                    else None
                ),
                split_id=self.split_id,
                eval_split=self.eval_split,
                wave2body_config=self.wave2body_config,
                wave2body_checkpoint=self.wave2body_checkpoint,
                wicompass_config=self.wicompass_config,
                wicompass_checkpoint=self.wicompass_checkpoint,
            )
            self._insert(candidate)
        self._seen_count += batch_size

    def _insert(self, candidate: PairedComparisonCandidate) -> None:
        pool = self._pools[candidate.action_group]
        pool.append(candidate)
        pool.sort(key=self._rank_key)
        if len(pool) > self.pool_size:
            del pool[self.pool_size :]

    @staticmethod
    def _rank_key(candidate: PairedComparisonCandidate) -> tuple[float, float, float]:
        return (-candidate.delta_mpjpe_mm, -candidate.wicompass_mpjpe_mm, candidate.wave2body_mpjpe_mm)

    def selected(self) -> dict[str, list[PairedComparisonCandidate]]:
        out: dict[str, list[PairedComparisonCandidate]] = {}
        for group, pool in self._pools.items():
            selected = self._select_with_limits(pool, enforce_action=True, enforce_subject=True)
            if len(selected) < self.top_k:
                selected = self._select_with_limits(pool, enforce_action=False, enforce_subject=True)
            if len(selected) < self.top_k:
                selected = self._select_with_limits(pool, enforce_action=False, enforce_subject=False)
            out[group] = selected
        return out

    def flat_selected(self) -> list[PairedComparisonCandidate]:
        return [candidate for group in self.action_groups for candidate in self.selected().get(group, [])]

    def _select_with_limits(
        self,
        pool: list[PairedComparisonCandidate],
        *,
        enforce_action: bool,
        enforce_subject: bool,
    ) -> list[PairedComparisonCandidate]:
        selected: list[PairedComparisonCandidate] = []
        action_counts: dict[int, int] = {}
        subject_counts: dict[int, int] = {}
        for candidate in sorted(pool, key=self._rank_key):
            if len(selected) >= self.top_k:
                break
            if enforce_action and self.max_per_action > 0 and action_counts.get(candidate.action_id, 0) >= self.max_per_action:
                continue
            if enforce_subject and self.max_per_subject > 0 and subject_counts.get(candidate.subject_id, 0) >= self.max_per_subject:
                continue
            if self.min_frame_gap > 0 and self._too_close(candidate, selected):
                continue
            selected.append(candidate)
            action_counts[candidate.action_id] = action_counts.get(candidate.action_id, 0) + 1
            subject_counts[candidate.subject_id] = subject_counts.get(candidate.subject_id, 0) + 1
        return selected

    def _too_close(self, candidate: PairedComparisonCandidate, selected: list[PairedComparisonCandidate]) -> bool:
        for item in selected:
            if item.subject_id != candidate.subject_id or item.action_id != candidate.action_id:
                continue
            if abs(item.frame_id - candidate.frame_id) < self.min_frame_gap:
                return True
        return False


class LowErrorSampleSelector:
    """Keep a bounded low-MPJPE candidate pool per M4Human action group."""

    def __init__(
        self,
        *,
        top_k: int = 3,
        action_groups: tuple[str, ...] = DEFAULT_QUALITATIVE_GROUPS,
        max_per_action: int = 1,
        min_frame_gap: int = 30,
        pool_size: int | None = None,
        split_id: str | None = None,
        eval_split: str | None = None,
        config: str | None = None,
        checkpoint: str | None = None,
    ) -> None:
        if int(top_k) <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        for group in action_groups:
            if group not in M4HUMAN_ACTION_GROUPS:
                raise KeyError(f"Unknown M4Human action group: {group!r}")
        self.top_k = int(top_k)
        self.action_groups = tuple(action_groups)
        self.max_per_action = int(max_per_action)
        self.min_frame_gap = int(min_frame_gap)
        self.pool_size = int(pool_size or max(50, self.top_k * 50))
        self.split_id = split_id
        self.eval_split = eval_split
        self.config = config
        self.checkpoint = checkpoint
        self._pools: dict[str, list[SampleCandidate]] = {group: [] for group in self.action_groups}
        self._seen_count = 0

    @property
    def seen_count(self) -> int:
        return self._seen_count

    def update_from_batch(
        self,
        *,
        batch: dict[str, Any],
        pred_pose: Any,
        gt_pose: Any,
        mmwave_points: Any,
        absolute_mpjpe: Any,
        root_error: Any,
        root_aligned_mpjpe: Any,
    ) -> None:
        indicators = _to_numpy(batch["indicator"]).astype(np.int64, copy=False)
        temporal = _to_numpy(batch["temporal_indicators"]).astype(np.int64, copy=False) if "temporal_indicators" in batch else None
        pred = _to_numpy(pred_pose).astype(np.float32, copy=False)
        gt = _to_numpy(gt_pose).astype(np.float32, copy=False)
        points = _to_numpy(mmwave_points).astype(np.float32, copy=False)
        abs_values = _to_numpy(absolute_mpjpe).reshape(-1).astype(np.float64, copy=False) * 1000.0
        root_values = _to_numpy(root_error).reshape(-1).astype(np.float64, copy=False) * 1000.0
        root_aligned_values = _to_numpy(root_aligned_mpjpe).reshape(-1).astype(np.float64, copy=False) * 1000.0

        batch_size = int(indicators.shape[0])
        for local_idx in range(batch_size):
            indicator = indicators[local_idx]
            action_id = int(indicator[1])
            group = action_group_for_action(action_id, self.action_groups)
            if group is None:
                continue
            candidate = SampleCandidate(
                action_group=group,
                action_id=action_id,
                subject_id=int(indicator[0]),
                frame_id=int(indicator[2]),
                global_index=self._seen_count + local_idx,
                absolute_mpjpe_mm=float(abs_values[local_idx]),
                root_error_mm=float(root_values[local_idx]),
                root_aligned_mpjpe_mm=float(root_aligned_values[local_idx]),
                mmwave_points=np.asarray(points[local_idx], dtype=np.float32).copy(),
                gt_pose=np.asarray(gt[local_idx], dtype=np.float32).copy(),
                pred_pose=np.asarray(pred[local_idx], dtype=np.float32).copy(),
                indicator=np.asarray(indicator, dtype=np.int64).copy(),
                temporal_indicators=(
                    np.asarray(temporal[local_idx], dtype=np.int64).copy()
                    if temporal is not None and int(temporal.shape[0]) > local_idx
                    else None
                ),
                split_id=self.split_id,
                eval_split=self.eval_split,
                config=self.config,
                checkpoint=self.checkpoint,
            )
            self._insert(candidate)
        self._seen_count += batch_size

    def _insert(self, candidate: SampleCandidate) -> None:
        pool = self._pools[candidate.action_group]
        pool.append(candidate)
        pool.sort(key=lambda item: item.absolute_mpjpe_mm)
        if len(pool) > self.pool_size:
            del pool[self.pool_size :]

    def selected(self) -> dict[str, list[SampleCandidate]]:
        """Return selected candidates per group, sorted by absolute MPJPE."""

        out: dict[str, list[SampleCandidate]] = {}
        for group, pool in self._pools.items():
            selected: list[SampleCandidate] = []
            action_counts: dict[int, int] = {}
            for candidate in sorted(pool, key=lambda item: item.absolute_mpjpe_mm):
                if len(selected) >= self.top_k:
                    break
                if self.max_per_action > 0 and action_counts.get(candidate.action_id, 0) >= self.max_per_action:
                    continue
                if self.min_frame_gap > 0 and self._too_close(candidate, selected):
                    continue
                selected.append(candidate)
                action_counts[candidate.action_id] = action_counts.get(candidate.action_id, 0) + 1
            if len(selected) < self.top_k:
                chosen = {(item.subject_id, item.action_id, item.frame_id) for item in selected}
                for candidate in sorted(pool, key=lambda item: item.absolute_mpjpe_mm):
                    key = (candidate.subject_id, candidate.action_id, candidate.frame_id)
                    if key in chosen:
                        continue
                    selected.append(candidate)
                    chosen.add(key)
                    if len(selected) >= self.top_k:
                        break
            out[group] = selected
        return out

    def flat_selected(self) -> list[SampleCandidate]:
        return [candidate for group in self.action_groups for candidate in self.selected().get(group, [])]

    def _too_close(self, candidate: SampleCandidate, selected: list[SampleCandidate]) -> bool:
        for item in selected:
            if item.subject_id != candidate.subject_id or item.action_id != candidate.action_id:
                continue
            if abs(item.frame_id - candidate.frame_id) < self.min_frame_gap:
                return True
        return False



@dataclass
class ErrorBucketRecord:
    """Lightweight per-sample record used for error-bucket qualitative selection."""

    dataset: str
    dataset_display: str
    global_index: int
    absolute_mpjpe_mm: float
    root_error_mm: float
    root_aligned_mpjpe_mm: float
    point_count: int
    metadata: dict[str, Any]
    bin_index: int | None = None
    bin_start_error_mm: float | None = None
    bin_end_error_mm: float | None = None
    bin_center_error_mm: float | None = None
    rank: int | None = None
    fallback_low_points: bool = False
    selection_metric: str = "absolute_mpjpe_mm"
    npz_path: str | None = None

    def record(self, *, npz_path: str | None = None) -> dict[str, Any]:
        out = {
            "dataset": self.dataset,
            "dataset_display": self.dataset_display,
            "global_index": int(self.global_index),
            "absolute_mpjpe_mm": float(self.absolute_mpjpe_mm),
            "root_error_mm": float(self.root_error_mm),
            "root_aligned_mpjpe_mm": float(self.root_aligned_mpjpe_mm),
            "point_count": int(self.point_count),
            "bin_index": None if self.bin_index is None else int(self.bin_index),
            "bin_start_error_mm": self.bin_start_error_mm,
            "bin_end_error_mm": self.bin_end_error_mm,
            "bin_center_error_mm": self.bin_center_error_mm,
            "rank": None if self.rank is None else int(self.rank),
            "fallback_low_points": bool(self.fallback_low_points),
            "selection_metric": self.selection_metric,
            "metadata": dict(self.metadata),
        }
        path_value = npz_path if npz_path is not None else self.npz_path
        if path_value is not None:
            out["npz_path"] = str(path_value)
        return out


def _point_count(points: Any) -> np.ndarray:
    arr = _to_numpy(points).astype(np.float32, copy=False)
    if arr.ndim < 3:
        return np.zeros((0,), dtype=np.int64)
    flat = arr.reshape(arr.shape[0], -1, arr.shape[-1])
    finite = np.isfinite(flat).all(axis=-1)
    nonzero = np.abs(flat[..., :3]).sum(axis=-1) > 0.0
    return (finite & nonzero).sum(axis=1).astype(np.int64, copy=False)


def _batch_metadata(batch: dict[str, Any], local_idx: int) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if "indicator" in batch:
        indicator = _to_numpy(batch["indicator"])[local_idx].astype(np.int64, copy=False)
        metadata.update(
            {
                "subject_id": int(indicator[0]),
                "action_id": int(indicator[1]),
                "frame_id": int(indicator[2]),
                "indicator": indicator.tolist(),
            }
        )
    if "temporal_indicators" in batch:
        temporal = _to_numpy(batch["temporal_indicators"])
        if int(temporal.shape[0]) > local_idx:
            metadata["temporal_indicators"] = temporal[local_idx].astype(np.int64, copy=False).tolist()
    for key in ("mmbody_scenario", "mmbody_sequence", "mmbody_frame"):
        if key not in batch:
            continue
        value = batch[key]
        if torch.is_tensor(value):
            item = value[local_idx].detach().cpu().item()
        elif isinstance(value, (list, tuple)):
            item = value[local_idx]
        else:
            item = value
        if isinstance(item, np.generic):
            item = item.item()
        metadata[key] = item
    if "frame_id" not in metadata and "mmbody_frame" in metadata:
        metadata["frame_id"] = int(metadata["mmbody_frame"])
    return metadata


class ErrorBucketMetricCollector:
    """Collect sample-level errors, then choose representative quantile buckets."""

    VALID_METRICS = ("absolute_mpjpe_mm", "root_error_mm", "root_aligned_mpjpe_mm")

    def __init__(
        self,
        *,
        dataset: str,
        dataset_display: str,
        bins: int = 10,
        metric: str = "absolute_mpjpe_mm",
        min_points: int = 10,
        top_k_candidates: int = 10,
    ) -> None:
        if int(bins) <= 0:
            raise ValueError(f"bins must be positive, got {bins}")
        if metric not in self.VALID_METRICS:
            raise ValueError(f"metric must be one of {self.VALID_METRICS}, got {metric!r}")
        self.dataset = str(dataset)
        self.dataset_display = str(dataset_display)
        self.bins = int(bins)
        self.metric = str(metric)
        self.min_points = int(min_points)
        self.top_k_candidates = int(top_k_candidates)
        self.records: list[ErrorBucketRecord] = []
        self._seen_count = 0

    @property
    def seen_count(self) -> int:
        return self._seen_count

    def update_from_batch(
        self,
        *,
        batch: dict[str, Any],
        pred_pose: Any,
        gt_pose: Any,
        mmwave_points: Any,
        absolute_mpjpe: Any,
        root_error: Any,
        root_aligned_mpjpe: Any,
    ) -> None:
        abs_values = _metric_to_mm(absolute_mpjpe)
        root_values = _metric_to_mm(root_error)
        root_aligned_values = _metric_to_mm(root_aligned_mpjpe)
        point_counts = _point_count(mmwave_points)
        batch_size = int(abs_values.shape[0])
        for local_idx in range(batch_size):
            self.records.append(
                ErrorBucketRecord(
                    dataset=self.dataset,
                    dataset_display=self.dataset_display,
                    global_index=self._seen_count + local_idx,
                    absolute_mpjpe_mm=float(abs_values[local_idx]),
                    root_error_mm=float(root_values[local_idx]),
                    root_aligned_mpjpe_mm=float(root_aligned_values[local_idx]),
                    point_count=int(point_counts[local_idx]) if int(point_counts.shape[0]) > local_idx else 0,
                    metadata=_batch_metadata(batch, local_idx),
                    selection_metric=self.metric,
                )
            )
        self._seen_count += batch_size

    def selected(self) -> tuple[list[ErrorBucketRecord], list[ErrorBucketRecord]]:
        if len(self.records) < self.bins:
            raise RuntimeError(f"Cannot split {len(self.records)} samples into {self.bins} bins")
        ordered = sorted(self.records, key=lambda item: getattr(item, self.metric))
        buckets = np.array_split(np.asarray(ordered, dtype=object), self.bins)
        selected: list[ErrorBucketRecord] = []
        candidates: list[ErrorBucketRecord] = []
        for bin_index, bucket_array in enumerate(buckets):
            bucket = list(bucket_array.tolist())
            if not bucket:
                continue
            values = np.asarray([getattr(item, self.metric) for item in bucket], dtype=np.float64)
            center = float(np.median(values))
            start = float(values.min())
            end = float(values.max())
            valid = [item for item in bucket if item.point_count >= self.min_points]
            fallback = len(valid) == 0
            pool = valid if valid else bucket
            ranked = sorted(pool, key=lambda item: (abs(getattr(item, self.metric) - center), -item.point_count))
            for rank, item in enumerate(ranked[: self.top_k_candidates], start=1):
                clone = ErrorBucketRecord(**item.__dict__)
                clone.bin_index = int(bin_index)
                clone.bin_start_error_mm = start
                clone.bin_end_error_mm = end
                clone.bin_center_error_mm = center
                clone.rank = int(rank)
                clone.fallback_low_points = bool(fallback)
                clone.selection_metric = self.metric
                candidates.append(clone)
            chosen = ErrorBucketRecord(**ranked[0].__dict__)
            chosen.bin_index = int(bin_index)
            chosen.bin_start_error_mm = start
            chosen.bin_end_error_mm = end
            chosen.bin_center_error_mm = center
            chosen.rank = 1
            chosen.fallback_low_points = bool(fallback)
            chosen.selection_metric = self.metric
            selected.append(chosen)
        return selected, candidates


class ErrorBucketExportCollector:
    """Export arrays only for sample indices selected by ErrorBucketMetricCollector."""

    def __init__(self, *, selected: list[ErrorBucketRecord], output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.selected_by_index = {int(item.global_index): item for item in selected}
        self.exported: list[ErrorBucketRecord] = []
        self._seen_count = 0

    @property
    def seen_count(self) -> int:
        return self._seen_count

    def update_from_batch(
        self,
        *,
        batch: dict[str, Any],
        pred_pose: Any,
        gt_pose: Any,
        mmwave_points: Any,
        absolute_mpjpe: Any,
        root_error: Any,
        root_aligned_mpjpe: Any,
    ) -> None:
        pred = _to_numpy(pred_pose).astype(np.float32, copy=False)
        gt = _to_numpy(gt_pose).astype(np.float32, copy=False)
        points = _to_numpy(mmwave_points).astype(np.float32, copy=False)
        batch_size = int(pred.shape[0])
        for local_idx in range(batch_size):
            global_index = self._seen_count + local_idx
            record = self.selected_by_index.get(global_index)
            if record is None:
                continue
            safe_dataset = record.dataset.replace("/", "_")
            file_name = f"{safe_dataset}_bin{int(record.bin_index):02d}_idx{global_index:07d}.npz"
            path = self.output_dir / file_name
            metadata = dict(record.metadata)
            metadata.update(_batch_metadata(batch, local_idx))
            np.savez_compressed(
                path,
                mmwave_points=np.asarray(points[local_idx], dtype=np.float32),
                gt_pose=np.asarray(gt[local_idx], dtype=np.float32),
                pred_pose=np.asarray(pred[local_idx], dtype=np.float32),
                absolute_mpjpe_mm=np.asarray(record.absolute_mpjpe_mm, dtype=np.float32),
                root_error_mm=np.asarray(record.root_error_mm, dtype=np.float32),
                root_aligned_mpjpe_mm=np.asarray(record.root_aligned_mpjpe_mm, dtype=np.float32),
                bin_index=np.asarray(record.bin_index if record.bin_index is not None else -1, dtype=np.int64),
                global_index=np.asarray(global_index, dtype=np.int64),
                metadata_json=np.asarray(json_dumps(metadata)),
            )
            exported = ErrorBucketRecord(**record.__dict__)
            exported.metadata = metadata
            exported.npz_path = str(path)
            self.exported.append(exported)
        self._seen_count += batch_size


def json_dumps(value: Any) -> str:
    """JSON dump helper that tolerates numpy scalar/list values."""

    def default(obj: Any) -> Any:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    import json

    return json.dumps(value, sort_keys=True, default=default)

class LowErrorPerActionSelector:
    """Keep low-MPJPE candidates independently for each M4Human action id."""

    def __init__(
        self,
        *,
        top_k: int = 1,
        action_ids: tuple[int, ...] = ALL_M4HUMAN_ACTION_IDS,
        min_frame_gap: int = 30,
        pool_size: int | None = None,
        split_id: str | None = None,
        eval_split: str | None = None,
        config: str | None = None,
        checkpoint: str | None = None,
    ) -> None:
        if int(top_k) <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        ids = tuple(int(action_id) for action_id in action_ids)
        invalid = [action_id for action_id in ids if action_id < 1 or action_id > 50]
        if invalid:
            raise ValueError(f"M4Human action ids must be in [1, 50], got {invalid}")
        self.top_k = int(top_k)
        self.action_ids = ids
        self.action_id_set = set(ids)
        self.min_frame_gap = int(min_frame_gap)
        self.pool_size = int(pool_size or max(20, self.top_k * 20))
        self.split_id = split_id
        self.eval_split = eval_split
        self.config = config
        self.checkpoint = checkpoint
        self._pools: dict[int, list[SampleCandidate]] = {action_id: [] for action_id in ids}
        self._seen_count = 0

    @property
    def seen_count(self) -> int:
        return self._seen_count

    def update_from_batch(
        self,
        *,
        batch: dict[str, Any],
        pred_pose: Any,
        gt_pose: Any,
        mmwave_points: Any,
        absolute_mpjpe: Any,
        root_error: Any,
        root_aligned_mpjpe: Any,
    ) -> None:
        indicators = _to_numpy(batch["indicator"]).astype(np.int64, copy=False)
        temporal = _to_numpy(batch["temporal_indicators"]).astype(np.int64, copy=False) if "temporal_indicators" in batch else None
        pred = _to_numpy(pred_pose).astype(np.float32, copy=False)
        gt = _to_numpy(gt_pose).astype(np.float32, copy=False)
        points = _to_numpy(mmwave_points).astype(np.float32, copy=False)
        abs_values = _to_numpy(absolute_mpjpe).reshape(-1).astype(np.float64, copy=False) * 1000.0
        root_values = _to_numpy(root_error).reshape(-1).astype(np.float64, copy=False) * 1000.0
        root_aligned_values = _to_numpy(root_aligned_mpjpe).reshape(-1).astype(np.float64, copy=False) * 1000.0

        batch_size = int(indicators.shape[0])
        for local_idx in range(batch_size):
            indicator = indicators[local_idx]
            action_id = int(indicator[1])
            if action_id not in self.action_id_set:
                continue
            group = action_group_for_action(action_id, DEFAULT_QUALITATIVE_GROUPS) or str(action_id)
            candidate = SampleCandidate(
                action_group=group,
                action_id=action_id,
                subject_id=int(indicator[0]),
                frame_id=int(indicator[2]),
                global_index=self._seen_count + local_idx,
                absolute_mpjpe_mm=float(abs_values[local_idx]),
                root_error_mm=float(root_values[local_idx]),
                root_aligned_mpjpe_mm=float(root_aligned_values[local_idx]),
                mmwave_points=np.asarray(points[local_idx], dtype=np.float32).copy(),
                gt_pose=np.asarray(gt[local_idx], dtype=np.float32).copy(),
                pred_pose=np.asarray(pred[local_idx], dtype=np.float32).copy(),
                indicator=np.asarray(indicator, dtype=np.int64).copy(),
                temporal_indicators=(
                    np.asarray(temporal[local_idx], dtype=np.int64).copy()
                    if temporal is not None and int(temporal.shape[0]) > local_idx
                    else None
                ),
                split_id=self.split_id,
                eval_split=self.eval_split,
                config=self.config,
                checkpoint=self.checkpoint,
            )
            pool = self._pools[action_id]
            pool.append(candidate)
            pool.sort(key=lambda item: item.absolute_mpjpe_mm)
            if len(pool) > self.pool_size:
                del pool[self.pool_size :]
        self._seen_count += batch_size

    def selected(self) -> dict[int, list[SampleCandidate]]:
        out: dict[int, list[SampleCandidate]] = {}
        for action_id, pool in self._pools.items():
            selected: list[SampleCandidate] = []
            for candidate in sorted(pool, key=lambda item: item.absolute_mpjpe_mm):
                if len(selected) >= self.top_k:
                    break
                if self.min_frame_gap > 0 and self._too_close(candidate, selected):
                    continue
                selected.append(candidate)
            if len(selected) < self.top_k:
                chosen = {(item.subject_id, item.frame_id) for item in selected}
                for candidate in sorted(pool, key=lambda item: item.absolute_mpjpe_mm):
                    key = (candidate.subject_id, candidate.frame_id)
                    if key in chosen:
                        continue
                    selected.append(candidate)
                    chosen.add(key)
                    if len(selected) >= self.top_k:
                        break
            out[action_id] = selected
        return out

    def flat_selected(self) -> list[SampleCandidate]:
        return [candidate for action_id in self.action_ids for candidate in self.selected().get(action_id, [])]

    def _too_close(self, candidate: SampleCandidate, selected: list[SampleCandidate]) -> bool:
        for item in selected:
            if item.subject_id != candidate.subject_id:
                continue
            if abs(item.frame_id - candidate.frame_id) < self.min_frame_gap:
                return True
        return False
