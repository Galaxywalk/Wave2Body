# -*- coding: utf-8 -*-
"""Dataset module.

Radar-pose datasets live in :mod:`wave2body.dataset.radar_pose`.
Joint-only human-body datasets live in :mod:`wave2body.dataset.human_body`.
"""

from __future__ import annotations

__all__ = [
    # Radar-Pose datasets
    "BaseRadarPoseDataset",
    "cropping",
    "MMBodyDataset",
    "M4HumanLMDBPoseDataset",
    "M4HUMAN_TEMPORAL_OFFSET_MODES",
    "M4HUMAN_ACTION_GROUP_ORDER",
    "M4HUMAN_ACTION_GROUPS",
    "count_by_action_group",
    "summarize_metric_by_action_group",
    "summarize_mpjpe_by_action_group",
    "validate_official_m4human_splits",
    # Joint-only datasets
    "AMASSJointsDataset",
    "M4HumanJointsDataset",
    "MMBodyJointsDataset",
    "create_human_body_dataset",
    "PreprocessingUtils",
]


_EXPORTS = {
    "BaseRadarPoseDataset": ("radar_pose", "BaseRadarPoseDataset"),
    "cropping": ("radar_pose", "cropping"),
    "MMBodyDataset": ("radar_pose", "MMBodyDataset"),
    "M4HumanLMDBPoseDataset": ("radar_pose", "M4HumanLMDBPoseDataset"),
    "M4HUMAN_TEMPORAL_OFFSET_MODES": ("radar_pose", "M4HUMAN_TEMPORAL_OFFSET_MODES"),
    "M4HUMAN_ACTION_GROUP_ORDER": ("radar_pose", "M4HUMAN_ACTION_GROUP_ORDER"),
    "M4HUMAN_ACTION_GROUPS": ("radar_pose", "M4HUMAN_ACTION_GROUPS"),
    "count_by_action_group": ("radar_pose", "count_by_action_group"),
    "summarize_metric_by_action_group": ("radar_pose", "summarize_metric_by_action_group"),
    "summarize_mpjpe_by_action_group": ("radar_pose", "summarize_mpjpe_by_action_group"),
    "validate_official_m4human_splits": ("radar_pose", "validate_official_m4human_splits"),
    "AMASSJointsDataset": ("human_body", "AMASSJointsDataset"),
    "M4HumanJointsDataset": ("human_body", "M4HumanJointsDataset"),
    "MMBodyJointsDataset": ("human_body", "MMBodyJointsDataset"),
    "create_human_body_dataset": ("human_body", "create_dataset"),
    "PreprocessingUtils": ("human_body", "PreprocessingUtils"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = __import__(f"{__name__}.{module_name}", fromlist=[attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
