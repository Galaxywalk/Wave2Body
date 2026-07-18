# -*- coding: utf-8 -*-

import numpy as np
import torch

from wave2body.dataset.radar_pose.m4human_protocols import (
    M4HUMAN_ACTION_GROUPS,
    M4HUMAN_LEGACY_ACTION_RANGES,
    count_by_action_group,
    summarize_mpjpe_by_action_group,
)
from wave2body.evaluate.m4human_action_groups import (
    M4HumanActionGroupAccumulator,
    M4HumanActionGroupVectorAccumulator,
    m4human_action_ids_from_indicator,
)


def test_m4human_action_groups_match_semantic_protocols():
    assert M4HUMAN_ACTION_GROUPS["1_30"] == tuple(range(1, 31))
    assert M4HUMAN_ACTION_GROUPS["31_35"] == tuple(range(31, 36))
    assert M4HUMAN_ACTION_GROUPS["36_50"] == tuple(range(36, 51))
    assert M4HUMAN_ACTION_GROUPS["1_50"] == tuple(range(1, 51))
    assert M4HUMAN_LEGACY_ACTION_RANGES["1_35"] == tuple(range(1, 36))
    assert M4HUMAN_LEGACY_ACTION_RANGES["36_40"] == tuple(range(36, 41))
    assert M4HUMAN_LEGACY_ACTION_RANGES["41_50"] == tuple(range(41, 51))


def test_count_by_action_group_uses_action_id_column():
    indicators = [
        [1, 30, 100],
        [1, 31, 100],
        [1, 35, 100],
        [1, 36, 100],
        [1, 50, 100],
    ]

    assert count_by_action_group(indicators) == {
        "1_30": 1,
        "31_35": 2,
        "36_50": 2,
        "1_50": 5,
    }


def test_summarize_mpjpe_by_action_group_is_sample_micro_average_in_mm():
    sums = np.zeros(50, dtype=np.float64)
    counts = np.zeros(50, dtype=np.float64)
    sums[29] = 0.10
    counts[29] = 1
    sums[30] = 0.40
    counts[30] = 2
    sums[49] = 0.90
    counts[49] = 3

    metrics, sample_counts = summarize_mpjpe_by_action_group(sums, counts)

    np.testing.assert_allclose(metrics["1_30"], 100.0)
    np.testing.assert_allclose(metrics["31_35"], 200.0)
    np.testing.assert_allclose(metrics["36_50"], 300.0)
    np.testing.assert_allclose(metrics["1_50"], (0.10 + 0.40 + 0.90) / 6.0 * 1000.0)
    assert sample_counts == {"1_30": 1, "31_35": 2, "36_50": 3, "1_50": 6}


def test_action_group_accumulator_accepts_batched_indicators():
    indicators = torch.tensor([[2, 30, 0], [2, 31, 0], [2, 50, 0]])
    action_ids = m4human_action_ids_from_indicator(indicators)
    acc = M4HumanActionGroupAccumulator(metric_names=("mpjpe",))
    acc.update("mpjpe", action_ids, torch.tensor([0.10, 0.20, 0.50]))

    summary = acc.summary(scale=1000.0)
    np.testing.assert_allclose(summary["mpjpe"]["by_group"]["1_30"], 100.0)
    np.testing.assert_allclose(summary["mpjpe"]["by_group"]["31_35"], 200.0)
    np.testing.assert_allclose(summary["mpjpe"]["by_group"]["36_50"], 500.0)
    np.testing.assert_allclose(summary["mpjpe"]["by_group"]["1_50"], (0.10 + 0.20 + 0.50) / 3.0 * 1000.0)
    assert summary["mpjpe"]["num_samples_by_group"]["1_50"] == 3


def test_action_group_vector_accumulator_reports_per_joint_micro_average():
    indicators = torch.tensor([[2, 30, 0], [2, 31, 0], [2, 50, 0]])
    action_ids = m4human_action_ids_from_indicator(indicators)
    acc = M4HumanActionGroupVectorAccumulator(metric_names=("mpjpe",))
    acc.update(
        "mpjpe",
        action_ids,
        torch.tensor(
            [
                [0.10, 0.20],
                [0.30, 0.50],
                [0.70, 1.10],
            ]
        ),
    )

    summary = acc.summary(scale=1000.0)["mpjpe"]

    np.testing.assert_allclose(summary["by_group"]["1_30"], [100.0, 200.0])
    np.testing.assert_allclose(summary["by_group"]["31_35"], [300.0, 500.0])
    np.testing.assert_allclose(summary["by_group"]["36_50"], [700.0, 1100.0])
    np.testing.assert_allclose(summary["by_group"]["1_50"], [(0.10 + 0.30 + 0.70) / 3 * 1000.0, 600.0])
    assert summary["num_samples_by_group"]["1_50"] == 3
