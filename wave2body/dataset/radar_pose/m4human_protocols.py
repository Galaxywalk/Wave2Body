# -*- coding: utf-8 -*-
"""M4Human split and action-protocol helpers.

Wave2Body uses the full M4Human data route and the official ``s1``/``s2``/``s3``
split ids. The M4Human paper defines semantic action protocols as IP
``1..30``, SIP ``31..35``, NIP ``36..50``, and ALL ``1..50``. The released
upstream evaluator also contains legacy buckets ``1..35``, ``36..40``, and
``41..50``; keep those as explicit legacy ranges instead of using them for
paper-facing semantic protocol results.
"""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


M4HUMAN_ACTION_GROUPS: dict[str, tuple[int, ...]] = {
    "1_30": tuple(range(1, 31)),
    "31_35": tuple(range(31, 36)),
    "36_50": tuple(range(36, 51)),
    "1_50": tuple(range(1, 51)),
}

M4HUMAN_ACTION_GROUP_ORDER: tuple[str, ...] = ("1_30", "31_35", "36_50", "1_50")

M4HUMAN_ACTION_GROUP_TAGS: dict[str, str] = {
    "1_30": "mean_1_30",
    "31_35": "mean_31_35",
    "36_50": "mean_36_50",
    "1_50": "mean_1_50",
}

M4HUMAN_LEGACY_ACTION_RANGES: dict[str, tuple[int, ...]] = {
    "1_35": tuple(range(1, 36)),
    "36_40": tuple(range(36, 41)),
    "41_50": tuple(range(41, 51)),
    "1_50": tuple(range(1, 51)),
}

M4HUMAN_NON_VALID_INDICATORS: tuple[tuple[int, int, int], ...] = (
    (1, 12, 58),
    (1, 13, 491),
    (1, 31, 324),
    (1, 36, 284),
    (1, 37, 43),
    (5, 43, 551),
    (5, 43, 552),
    (5, 47, 136),
    (5, 47, 140),
)
M4HUMAN_NON_VALID_INDICATOR_SET = set(M4HUMAN_NON_VALID_INDICATORS)


EXPECTED_OFFICIAL_ALL_COUNTS: dict[str, dict[str, dict[str, int]]] = {
    "s1": {
        "train": {"1_50": 495506, "1_35": 346942},
        "val": {"1_50": 33811, "1_35": 23674},
        "test": {"1_50": 131806, "1_35": 92286},
    },
    "s2": {
        "train": {"1_50": 493925, "1_35": 345754},
        "val": {"1_50": 33987, "1_35": 23786},
        "test": {"1_50": 133211, "1_35": 93362},
    },
    "s3": {
        "train": {"1_50": 488947, "1_35": 343331},
        "val": {"1_50": 39959, "1_35": 39959},
        "test": {"1_50": 132217, "1_35": 79612},
    },
}


def action_to_index(action_id: int) -> int:
    action = int(action_id)
    if action < 1 or action > 50:
        raise ValueError(f"M4Human action id must be in [1, 50], got {action_id!r}")
    return action - 1


def action_indices_for_group(group_name: str) -> list[int]:
    try:
        actions = M4HUMAN_ACTION_GROUPS[group_name]
    except KeyError as exc:
        raise KeyError(f"Unknown M4Human action group: {group_name!r}") from exc
    return [action_to_index(action_id) for action_id in actions]


def count_by_action_group(indicators: Iterable[Iterable[int]]) -> dict[str, int]:
    counts = {name: 0 for name in M4HUMAN_ACTION_GROUP_ORDER}
    action_sets = {name: set(actions) for name, actions in M4HUMAN_ACTION_GROUPS.items()}
    for indicator in indicators:
        values = list(indicator)
        if len(values) < 2:
            continue
        action_id = int(values[1])
        for name in M4HUMAN_ACTION_GROUP_ORDER:
            if action_id in action_sets[name]:
                counts[name] += 1
    return counts


def summarize_metric_by_action_group(
    sum_by_action: Iterable[float],
    count_by_action: Iterable[float],
    *,
    scale: float = 1.0,
) -> tuple[dict[str, float | None], dict[str, int]]:
    """Return sample-micro-averaged values for semantic M4Human action protocols."""

    sums = np.asarray(list(sum_by_action), dtype=np.float64)
    counts = np.asarray(list(count_by_action), dtype=np.float64)
    if sums.shape[0] != 50 or counts.shape[0] != 50:
        raise ValueError(f"Expected 50 action bins, got sums={sums.shape}, counts={counts.shape}")

    metrics: dict[str, float | None] = {}
    sample_counts: dict[str, int] = {}
    for name in M4HUMAN_ACTION_GROUP_ORDER:
        idx = action_indices_for_group(name)
        total_count = float(np.nansum(counts[idx]))
        sample_counts[name] = int(round(total_count))
        if total_count <= 0:
            metrics[name] = None
        else:
            metrics[name] = float(np.nansum(sums[idx]) / total_count * float(scale))
    return metrics, sample_counts


def summarize_mpjpe_by_action_group(
    sum_mpjpe_m_by_action: Iterable[float],
    count_by_action: Iterable[float],
) -> tuple[dict[str, float | None], dict[str, int]]:
    """Return sample-micro-averaged MPJPE in mm for each action group."""

    return summarize_metric_by_action_group(
        sum_mpjpe_m_by_action,
        count_by_action,
        scale=1000.0,
    )


def _subject_set(indicators: Iterable[Iterable[int]]) -> set[int]:
    return {int(list(item)[0]) for item in indicators}


def _action_set(indicators: Iterable[Iterable[int]]) -> set[int]:
    return {int(list(item)[1]) for item in indicators}


def _select_full_route(
    split_indices: Mapping[str, Mapping[str, Mapping[str, list[list[int]]]]],
    *,
    split_id: str,
    subset: str,
) -> list[list[int]]:
    candidates: list[tuple[int, str, list[list[int]]]] = []
    for route_key, route_splits in split_indices.items():
        try:
            raw_items = route_splits[split_id][subset]
        except KeyError:
            continue
        items = [item for item in raw_items if tuple(item) not in M4HUMAN_NON_VALID_INDICATOR_SET]
        candidates.append((len(items), str(route_key), items))
    if not candidates:
        raise KeyError(f"Missing M4Human indices for split={split_id}, subset={subset}")
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def validate_official_m4human_splits(indices_path: str | Path) -> dict[str, object]:
    """Validate effective full-data M4Human splits used for upstream test groups.

    M4Human evaluation excludes a small fixed set of invalid indicators; counts
    returned here mirror the dataset adapter and paper table sample counts.
    """

    with gzip.open(Path(indices_path), "rb") as f:
        split_indices: Mapping[str, Mapping[str, Mapping[str, list[list[int]]]]] = pickle.load(f)

    observed: dict[str, dict[str, dict[str, int]]] = {}
    for split_id in ("s1", "s2", "s3"):
        observed[split_id] = {}
        for subset in ("train", "val", "test"):
            indicators = _select_full_route(split_indices, split_id=split_id, subset=subset)
            observed[split_id][subset] = count_by_action_group(indicators)

    mismatches: list[dict[str, object]] = []
    for split_id, subsets in EXPECTED_OFFICIAL_ALL_COUNTS.items():
        for subset, expected_counts in subsets.items():
            got_counts = observed[split_id][subset]
            for name, expected in expected_counts.items():
                got = got_counts[name]
                if got != expected:
                    mismatches.append(
                        {
                            "split": split_id,
                            "subset": subset,
                            "action_group": name,
                            "expected": expected,
                            "observed": got,
                        }
                    )

    def _valid(items):
        return [item for item in items if tuple(item) not in M4HUMAN_NON_VALID_INDICATOR_SET]

    s2_train_subjects = _subject_set(_valid(_select_full_route(split_indices, split_id="s2", subset="train")))
    s2_val_subjects = _subject_set(_valid(_select_full_route(split_indices, split_id="s2", subset="val")))
    s2_test_subjects = _subject_set(_valid(_select_full_route(split_indices, split_id="s2", subset="test")))
    s3_train_actions = _action_set(_valid(_select_full_route(split_indices, split_id="s3", subset="train")))
    s3_val_actions = _action_set(_valid(_select_full_route(split_indices, split_id="s3", subset="val")))
    s3_test_actions = _action_set(_valid(_select_full_route(split_indices, split_id="s3", subset="test")))

    return {
        "indices_path": str(indices_path),
        "observed_counts": observed,
        "expected_counts": EXPECTED_OFFICIAL_ALL_COUNTS,
        "mismatches": mismatches,
        "s2_subjects": {
            "train": sorted(s2_train_subjects),
            "val": sorted(s2_val_subjects),
            "test": sorted(s2_test_subjects),
            "train_test_disjoint": not bool(s2_train_subjects & s2_test_subjects),
            "val_test_disjoint": not bool(s2_val_subjects & s2_test_subjects),
        },
        "s3_actions": {
            "train": sorted(s3_train_actions),
            "val": sorted(s3_val_actions),
            "test": sorted(s3_test_actions),
            "train_val_test_disjoint": not bool(
                (s3_train_actions & s3_val_actions)
                or (s3_train_actions & s3_test_actions)
                or (s3_val_actions & s3_test_actions)
            ),
        },
        "warnings": [
            "Upstream s3 val has zero 36_40/41_50 action-group samples; skip those scalars or log NaN."
        ],
    }
