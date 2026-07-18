# -*- coding: utf-8 -*-
"""Single-stage token-based human pose estimation training."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    from ..checkpoint import load_checkpoint, load_optional_component_states, module_state_dict, save_checkpoint
    from ..common import load_config, safe_param_count, set_global_seed, write_run_manifest
    from ..radar_pose_data import LimitedDataLoader, build_radar_pose_dataloaders, create_radar_pose_dataset
    from ..feature_cache import maybe_enable_frozen_feature_cache
    from ..optim import build_hpe_token_optimizer
    from ...evaluate.m4human_action_groups import (
        M4HumanActionGroupAccumulator,
        M4HumanActionGroupVectorAccumulator,
        m4human_action_ids_from_indicator,
    )
    from ...evaluate.token_alignment import TokenAlignmentCollector, rank_part_codebook_candidates
    from ...evaluate.token_accuracy import PartTokenTopKAccumulator, parse_topk_list
    from ...models.hpe_token import TokenHPEEstimatorModel
    from ...models.hpe_token.decoder import decode_part_token_logits, part_codebook_distance_logits
    from ...utils.runtime_utils import autocast_context, build_grad_scaler, configure_runtime, maybe_compile
    from ._paths import resolve_run_root, write_effective_config
    from .hpe_token_training import (
        _accumulate_metric,
        _apply_optional_soft_token_loss,
        _encoder_trainable_count,
        _finalize_metric_tensors,
        _forward_batch,
        _point_augmentation_from_config,
        _pose_decode_from_config,
        _select_pose_outputs,
        _soft_token_loss_from_config,
        _token_loss_from_config,
        train_hpe_token_epoch,
        validate_hpe_token,
        _zero_metric_tensors,
    )
except ImportError:  # Support scripts that put the repo root directly on sys.path.
    from wave2body.train.checkpoint import load_checkpoint, load_optional_component_states, module_state_dict, save_checkpoint
    from wave2body.train.common import load_config, safe_param_count, set_global_seed, write_run_manifest
    from wave2body.train.radar_pose_data import LimitedDataLoader, build_radar_pose_dataloaders, create_radar_pose_dataset
    from wave2body.train.feature_cache import maybe_enable_frozen_feature_cache
    from wave2body.train.optim import build_hpe_token_optimizer
    from wave2body.evaluate.m4human_action_groups import (
        M4HumanActionGroupAccumulator,
        M4HumanActionGroupVectorAccumulator,
        m4human_action_ids_from_indicator,
    )
    from wave2body.evaluate.token_alignment import TokenAlignmentCollector, rank_part_codebook_candidates
    from wave2body.evaluate.token_accuracy import PartTokenTopKAccumulator, parse_topk_list
    from wave2body.models.hpe_token import TokenHPEEstimatorModel
    from wave2body.models.hpe_token.decoder import decode_part_token_logits, part_codebook_distance_logits
    from wave2body.utils.runtime_utils import autocast_context, build_grad_scaler, configure_runtime, maybe_compile
    from wave2body.train.pipelines._paths import resolve_run_root, write_effective_config
    from wave2body.train.pipelines.hpe_token_training import (
        _accumulate_metric,
        _apply_optional_soft_token_loss,
        _encoder_trainable_count,
        _finalize_metric_tensors,
        _forward_batch,
        _point_augmentation_from_config,
        _pose_decode_from_config,
        _select_pose_outputs,
        _soft_token_loss_from_config,
        _token_loss_from_config,
        train_hpe_token_epoch,
        validate_hpe_token,
        _zero_metric_tensors,
    )


EXPERIMENT_NAME = "hpe_token"
STAGE_NAME = "hpe_token"
DISPLAY_NAME = "HPE Token"
METRIC_NAMESPACE = "hpe_token"


def _dataset_eval_label(config: dict[str, Any]) -> str:
    name = str(config.get("dataset", {}).get("name", "dataset")).strip().lower()
    return name.replace("dataset", "") or name or "dataset"


def _eval_suffix(config: dict[str, Any], eval_split: str) -> str:
    split_id = config.get("m4human", {}).get("split")
    if split_id:
        return f"{eval_split}_{split_id}"
    return f"{eval_split}_{_dataset_eval_label(config)}"


def _normalize_training_config(config: dict[str, Any]) -> dict[str, Any]:
    training = config.setdefault("training", {})
    training["stage"] = STAGE_NAME
    training["stage_display_name"] = DISPLAY_NAME
    training["metric_namespace"] = METRIC_NAMESPACE
    training.setdefault("freeze_mmwave_tokenizer", True)
    training.setdefault("freeze_vqvae", True)
    training.setdefault("frozen_feature_cache", {"enabled": False})
    training.setdefault("latent_loss_weight", 0.3)
    training.setdefault("token_loss_weight", 0.0)
    training.setdefault("pose_loss_weight", 1.0)
    return config


def _apply_token_classifier_only_training(model: torch.nn.Module, training_cfg: dict[str, Any]) -> None:
    if not bool(training_cfg.get("freeze_token_translator_except_classifier", False)):
        return

    translator = getattr(model, "token_translator", None)
    raw_translator = getattr(translator, "_orig_mod", translator)
    token_classifier = getattr(raw_translator, "token_classifier", None)
    if token_classifier is None:
        raise ValueError(
            "training.freeze_token_translator_except_classifier requires "
            "token_translator.token_classifier"
        )

    for param in raw_translator.parameters():
        param.requires_grad = False
    for param in token_classifier.parameters():
        param.requires_grad = True
    setattr(model, "freeze_token_translator_except_classifier", True)


def _cache_tag(config: dict[str, Any], pretrained: dict[str, Any]) -> str:
    data_format = config.get("data_format", {})
    payload = {
        "stage": STAGE_NAME,
        "data_format": {
            key: data_format.get(key)
            for key in (
                "n_points",
                "merge_nframes",
                "radar_input_c",
                "point_sampling",
                "normalized",
                "input_root_normalized",
                "input_root_crop",
                "input_root_crop_x_range",
                "input_root_crop_y_range",
                "input_root_crop_z_range",
                "target_root_normalized",
            )
        },
        "mmwave_tokenizer": config.get("mmwave_tokenizer", {}),
        "sensor_tokens": config.get("sensor_tokens", {}),
        "mmwave_tokenizer_ckpt": pretrained.get("mmwave_tokenizer"),
        "vqvae_ckpt": pretrained.get("vqvae"),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)


def _write_summary(
    output_root: Path,
    *,
    config_path: str | Path | None,
    checkpoint_dir: Path,
    log_dir: Path,
    runtime,
    epochs: int,
    best_val_mpjpe: float,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = {
        "stage": STAGE_NAME,
        "best_val_mpjpe_mm": best_val_mpjpe * 1000.0,
        "output_root": str(output_root),
        "checkpoint_dir": str(checkpoint_dir),
        "best_checkpoint": str(checkpoint_dir / "best.pth"),
        "latest_checkpoint": str(checkpoint_dir / "latest.pth"),
        "tensorboard_dir": str(log_dir),
        "config": str(config_path) if config_path else None,
        "precision": runtime.precision,
        "amp_enabled": runtime.amp_enabled,
        "epochs_configured": epochs,
        "history": history,
    }
    (output_root / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _build_eval_loader(
    dataset,
    *,
    training_cfg: dict[str, Any],
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> DataLoader:
    eval_batch_size = int(batch_size or training_cfg.get("eval_batch_size", training_cfg.get("batch_size", 32)))
    workers = int(num_workers if num_workers is not None else training_cfg.get("eval_num_workers", training_cfg.get("num_workers", 4)))
    pin_memory = bool(training_cfg.get("pin_memory", True)) and workers >= 0
    persistent_workers = bool(training_cfg.get("persistent_workers", workers > 0)) and workers > 0
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": eval_batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "drop_last": False,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = int(training_cfg.get("prefetch_factor", 2))
    return DataLoader(**kwargs)


class _PointEnsembleDataset(Dataset):
    """Return stacked deterministic point views for test-time ensemble eval."""

    def __init__(self, config: dict[str, Any], *, split: str, views: list[dict[str, Any]]) -> None:
        if not views:
            raise ValueError("Point ensemble requires at least one view.")
        self.datasets = []
        self.views = []
        for view in views:
            view_cfg = copy.deepcopy(config)
            data_format = view_cfg.setdefault("data_format", {})
            if "n_points" in view and view["n_points"] is not None:
                data_format["n_points"] = int(view["n_points"])
            if view.get("point_sampling"):
                data_format["point_sampling"] = str(view["point_sampling"])
            self.datasets.append(create_radar_pose_dataset(view_cfg, split=split, device="cpu"))
            self.views.append(
                {
                    "n_points": data_format.get("n_points"),
                    "merge_nframes": data_format.get("merge_nframes"),
                    "point_sampling": data_format.get("point_sampling"),
                    "weight": float(view.get("weight", 1.0)),
                }
            )
        lengths = {len(dataset) for dataset in self.datasets}
        if len(lengths) != 1:
            raise ValueError(f"Point ensemble view datasets have different lengths: {sorted(lengths)}")
        weights = torch.as_tensor([view["weight"] for view in self.views], dtype=torch.float32)
        if bool((weights < 0).any().item()) or float(weights.sum().item()) <= 0.0:
            raise ValueError("Point ensemble view weights must be non-negative with positive sum.")
        self.view_weights = weights

    def __len__(self) -> int:
        return len(self.datasets[0])

    @staticmethod
    def _pad_view_points(points: torch.Tensor, max_points: int) -> torch.Tensor:
        if int(points.shape[0]) == int(max_points):
            return points
        out = points.new_zeros((int(max_points), int(points.shape[-1])))
        out[: int(points.shape[0]), :] = points
        return out

    def __getitem__(self, idx: int) -> dict[str, Any]:
        items = [dataset[idx] for dataset in self.datasets]
        max_points = max(int(item["mmwave_points"].shape[0]) for item in items)
        out = dict(items[0])
        out["mmwave_points"] = torch.stack(
            [self._pad_view_points(item["mmwave_points"], max_points) for item in items],
            dim=0,
        )
        out["point_ensemble_weights"] = self.view_weights
        return out


def _point_ensemble_views_from_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = config.get("evaluation", {}).get("point_ensemble")
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return []
    raw_views = cfg.get("views")
    if not isinstance(raw_views, list) or not raw_views:
        raise ValueError("evaluation.point_ensemble.views must be a non-empty list.")
    base_format = config.get("data_format", {})
    views: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_views):
        if not isinstance(raw, dict):
            raise ValueError(f"evaluation.point_ensemble.views[{idx}] must be a mapping.")
        view = dict(raw)
        if view.get("n_points") is not None:
            n_points = int(view["n_points"])
            if n_points <= 0:
                raise ValueError("evaluation.point_ensemble view n_points must be > 0.")
            view["n_points"] = n_points
        else:
            view["n_points"] = base_format.get("n_points")
        if view.get("merge_nframes") is not None:
            raise ValueError("evaluation.point_ensemble per-view merge_nframes is no longer supported.")
        view["merge_nframes"] = base_format.get("merge_nframes")
        view["point_sampling"] = str(view.get("point_sampling") or base_format.get("point_sampling", "random"))
        weight = float(view.get("weight", 1.0))
        if weight < 0.0:
            raise ValueError("evaluation.point_ensemble view weight must be non-negative.")
        view["weight"] = weight
        views.append(view)
    if sum(float(view["weight"]) for view in views) <= 0.0:
        raise ValueError("evaluation.point_ensemble view weights must have positive sum.")
    return views


def _checkpoint_ensemble_from_config(config: dict[str, Any]) -> dict[str, Any] | None:
    cfg = config.get("evaluation", {}).get("checkpoint_ensemble")
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return None
    raw_checkpoints = cfg.get("checkpoints", [])
    if not isinstance(raw_checkpoints, list):
        raise ValueError("evaluation.checkpoint_ensemble.checkpoints must be a list.")
    checkpoints = [str(item).strip() for item in raw_checkpoints if str(item).strip()]
    if len(checkpoints) < 2:
        raise ValueError("evaluation.checkpoint_ensemble needs at least two checkpoints.")

    raw_weights = cfg.get("weights")
    if raw_weights is None:
        weights = [1.0] * len(checkpoints)
    elif isinstance(raw_weights, list):
        weights = [float(item) for item in raw_weights]
    else:
        raise ValueError("evaluation.checkpoint_ensemble.weights must be a list when provided.")
    if len(weights) != len(checkpoints):
        raise ValueError("evaluation.checkpoint_ensemble.weights must match checkpoints length.")
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError("evaluation.checkpoint_ensemble.weights must be non-negative with positive sum.")
    weight_sum = float(sum(weights))
    return {
        "checkpoints": checkpoints,
        "weights": [float(weight) / weight_sum for weight in weights],
    }


def _per_sample_mpjpe(pred_pose: torch.Tensor, target_pose: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pred_pose.float() - target_pose.float(), dim=-1).mean(dim=1)


def _root_aligned(pose: torch.Tensor, root_index: int = 0) -> torch.Tensor:
    return pose.float() - pose.float()[:, root_index : root_index + 1, :]


def _sanitize_metric_name(value: str) -> str:
    safe = []
    for char in str(value):
        safe.append(char if char.isalnum() else "_")
    return "_".join(part for part in "".join(safe).strip("_").lower().split("_") if part)


def _logit_decode_specs_from_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = config.get("evaluation", {}).get("logit_decodes")
    if not cfg:
        return []
    if isinstance(cfg, dict):
        if not bool(cfg.get("enabled", True)):
            return []
        cfg = cfg.get("specs", [])
    if not isinstance(cfg, list):
        raise ValueError("evaluation.logit_decodes must be a list or {enabled, specs} mapping.")

    specs: list[dict[str, Any]] = []
    for idx, raw in enumerate(cfg):
        if not isinstance(raw, dict):
            raise ValueError(f"evaluation.logit_decodes[{idx}] must be a mapping.")
        source = str(raw.get("source", "learned")).strip().lower()
        if source not in {
            "learned",
            "learned_classifier",
            "codebook_distance",
            "distance",
            "vq_distance",
            "blend",
            "learned_codebook_blend",
        }:
            raise ValueError(f"Unsupported logit decode source: {source!r}")
        mode = str(raw.get("mode", "argmax")).strip().lower()
        if mode not in {"argmax", "argmax_st", "sample_st", "soft_expectation"}:
            raise ValueError(f"Unsupported logit decode mode: {mode!r}")
        name = _sanitize_metric_name(
            raw.get(
                "name",
                f"{source}_{mode}_t{float(raw.get('temperature', 1.0)):g}_s{float(raw.get('scale', 1.0)):g}",
            )
        )
        if not name:
            raise ValueError(f"evaluation.logit_decodes[{idx}] has an empty metric name.")
        spec = {
            "name": name,
            "source": source,
            "mode": mode,
            "temperature": float(raw.get("temperature", 1.0)),
            "scale": float(raw.get("scale", raw.get("codebook_distance_scale", 1.0))),
        }
        if source in {"blend", "learned_codebook_blend"}:
            spec["learned_weight"] = float(raw.get("learned_weight", raw.get("classifier_weight", 1.0)))
            spec["codebook_weight"] = float(raw.get("codebook_weight", raw.get("distance_weight", 1.0)))
            if spec["learned_weight"] < 0 or spec["codebook_weight"] < 0:
                raise ValueError("evaluation.logit_decodes blend weights must be non-negative.")
            if spec["learned_weight"] == 0 and spec["codebook_weight"] == 0:
                raise ValueError("evaluation.logit_decodes blend needs at least one positive weight.")
        specs.append(spec)
    return specs


def _token_topk_from_config(config: dict[str, Any]) -> dict[str, Any] | None:
    cfg = config.get("evaluation", {}).get("token_topk")
    if cfg is None:
        return None
    if isinstance(cfg, bool):
        if not cfg:
            return None
        return {"topk": (1, 3, 5, 10), "include_slots": True}
    if isinstance(cfg, str):
        return {"topk": parse_topk_list(cfg), "include_slots": True}
    if not isinstance(cfg, dict):
        raise ValueError("evaluation.token_topk must be a bool, string, or mapping.")
    if not bool(cfg.get("enabled", True)):
        return None
    return {
        "topk": parse_topk_list(cfg.get("topk", "1,3,5,10")),
        "include_slots": bool(cfg.get("include_slots", True)),
    }


def _token_alignment_from_config(config: dict[str, Any]) -> dict[str, int] | None:
    cfg = config.get("evaluation", {}).get("token_alignment")
    if cfg is None:
        return None
    if isinstance(cfg, bool):
        if not cfg:
            return None
        cfg = {"enabled": True}
    if not isinstance(cfg, dict):
        raise ValueError("evaluation.token_alignment must be a bool or mapping.")
    if not bool(cfg.get("enabled", True)):
        return None
    if str(config.get("dataset", {}).get("name", "")).strip().lower() != "m4human":
        raise ValueError("evaluation.token_alignment currently supports only M4Human Stage-I evaluation.")
    training_stage = str(config.get("training", {}).get("stage", "")).strip().lower()
    if training_stage != "hpe_token" or bool(config.get("root_head")):
        raise ValueError(
            "evaluation.token_alignment requires a Stage-I HPE-token config "
            "without the absolute root head."
        )
    shuffle_seed = int(cfg.get("shuffle_seed", 0))
    if shuffle_seed < 0:
        raise ValueError(
            f"evaluation.token_alignment.shuffle_seed must be non-negative, got {shuffle_seed}"
        )
    return {"shuffle_seed": shuffle_seed}


def _token_topk_oracle_from_config(config: dict[str, Any]) -> dict[str, Any] | None:
    cfg = config.get("evaluation", {}).get("token_topk_oracle")
    if cfg is None:
        return None
    if isinstance(cfg, bool):
        if not cfg:
            return None
        return {"topk": (1, 3, 5, 10), "source": "learned"}
    if isinstance(cfg, str):
        return {"topk": parse_topk_list(cfg), "source": "learned"}
    if not isinstance(cfg, dict):
        raise ValueError("evaluation.token_topk_oracle must be a bool, string, or mapping.")
    if not bool(cfg.get("enabled", True)):
        return None
    source = str(cfg.get("source", "learned")).strip().lower()
    if source not in {"learned", "learned_classifier"}:
        raise ValueError(f"Unsupported token_topk_oracle source: {source!r}")
    return {
        "topk": parse_topk_list(cfg.get("topk", "1,3,5,10")),
        "source": source,
    }


def _decode_part_token_indices(*, vqvae, indices: torch.Tensor) -> torch.Tensor:
    """Decode global part-token indices through a part-level VQ-VAE."""
    if indices.dim() != 2:
        raise ValueError(f"indices must be [B,T], got {tuple(indices.shape)}")
    if not hasattr(vqvae, "layout") or not hasattr(vqvae, "quantizers"):
        raise RuntimeError("Expected a part-level VQ-VAE with layout and quantizers.")
    layout = vqvae.layout
    part_indices = layout.split_global_indices(indices.long())
    part_tokens: dict[str, torch.Tensor] = {}
    for part_name in layout.part_names:
        idx_local = part_indices[part_name]
        codebook = vqvae.quantizers[part_name].codebook.to(device=indices.device)
        flat = idx_local.reshape(-1)
        z_part = codebook[flat].view(idx_local.shape[0], idx_local.shape[1], codebook.shape[-1])
        part_tokens[part_name] = z_part
    return vqvae.decode(layout.concat_part_tokens(part_tokens))


def _topk_oracle_indices_from_logits(
    *,
    layout,
    pred_logits: torch.Tensor,
    target_indices: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Use target tokens when they appear in each slot's top-k candidate set."""
    if pred_logits is None:
        raise ValueError("pred_logits is required for token top-k oracle decode")
    if pred_logits.dim() != 3:
        raise ValueError(f"pred_logits must be [B,T,C], got {tuple(pred_logits.shape)}")
    if target_indices.dim() != 2:
        raise ValueError(f"target_indices must be [B,T], got {tuple(target_indices.shape)}")
    if pred_logits.shape[:2] != target_indices.shape:
        raise ValueError(
            f"logit prefix {tuple(pred_logits.shape[:2])} does not match target shape {tuple(target_indices.shape)}"
        )

    target_indices = target_indices.long()
    local_indices_by_part: dict[str, torch.Tensor] = {}
    for part_name in layout.part_names:
        token_slice = layout.part_token_slices[part_name]
        offset = int(layout.part_codebook_offsets[part_name])
        codebook_size = int(layout.part_codebook_size[part_name])
        end = offset + codebook_size
        if offset < 0 or codebook_size <= 0 or end > int(pred_logits.shape[-1]):
            raise ValueError(
                f"Invalid codebook slice for {part_name}: offset={offset}, size={codebook_size}, "
                f"logits_dim={int(pred_logits.shape[-1])}"
            )
        logits_local = pred_logits[:, token_slice, offset:end]
        target_local = target_indices[:, token_slice] - offset
        valid_target = (target_local >= 0) & (target_local < codebook_size)
        effective_k = min(int(topk), codebook_size)
        top_idx = torch.topk(logits_local.float(), k=effective_k, dim=-1).indices
        top1 = top_idx[..., 0]
        target_in_topk = (top_idx == target_local.unsqueeze(-1)).any(dim=-1) & valid_target
        chosen = torch.where(target_in_topk, target_local, top1)
        local_indices_by_part[part_name] = chosen.long()
    return layout.merge_local_indices_to_global(local_indices_by_part).long()


def _merge_action_group_metrics(
    metrics: dict[str, Any],
    accumulator: M4HumanActionGroupAccumulator | None,
    *,
    include_per_action: bool,
) -> None:
    if accumulator is None:
        return
    summary = accumulator.summary(scale=1000.0, include_per_action=include_per_action)
    mpjpe_summary = summary.get("mpjpe", {})
    by_group = mpjpe_summary.get("by_group", {})
    counts = mpjpe_summary.get("num_samples_by_group", {})
    metrics["mpjpe_by_action_group_mm"] = by_group
    metrics["num_samples_by_action_group"] = counts
    if include_per_action and "per_action" in mpjpe_summary:
        metrics["mpjpe_by_action_mm"] = mpjpe_summary["per_action"]


def _batch_string_values(batch: dict[str, Any], key: str, batch_size: int) -> list[str] | None:
    if key not in batch:
        return None
    values = batch[key]
    if isinstance(values, str):
        out = [values for _ in range(batch_size)]
    elif isinstance(values, torch.Tensor):
        flat = values.detach().cpu().reshape(-1).tolist()
        out = [str(item) for item in flat]
    elif isinstance(values, (list, tuple)):
        out = [str(item) for item in values]
    else:
        out = [str(values)]
    if len(out) != int(batch_size):
        return None
    return out


def _update_mmbody_scenario_totals(
    totals: dict[str, dict[str, torch.Tensor]],
    scenario_names: list[str] | None,
    *,
    per_sample_mpjpe: torch.Tensor,
    per_sample_root_aligned_mpjpe: torch.Tensor,
    token_acc_per_sample: torch.Tensor,
    logit_decode_per_sample: dict[str, dict[str, torch.Tensor]] | None = None,
) -> None:
    if not scenario_names:
        return
    for scenario in sorted(set(scenario_names)):
        mask_values = [name == scenario for name in scenario_names]
        if not any(mask_values):
            continue
        mask = torch.as_tensor(mask_values, device=per_sample_mpjpe.device, dtype=torch.bool)
        item = totals.setdefault(
            scenario,
            {
                "count": per_sample_mpjpe.new_zeros((), dtype=torch.float32),
                "mpjpe": per_sample_mpjpe.new_zeros((), dtype=torch.float32),
                "root_aligned_mpjpe": per_sample_mpjpe.new_zeros((), dtype=torch.float32),
                "token_acc": per_sample_mpjpe.new_zeros((), dtype=torch.float32),
                "logit_decode": {},
            },
        )
        item["count"] = item["count"] + mask.float().sum()
        item["mpjpe"] = item["mpjpe"] + per_sample_mpjpe[mask].sum()
        item["root_aligned_mpjpe"] = item["root_aligned_mpjpe"] + per_sample_root_aligned_mpjpe[mask].sum()
        item["token_acc"] = item["token_acc"] + token_acc_per_sample[mask].sum()
        if logit_decode_per_sample:
            logit_totals = item["logit_decode"]
            for name, values in logit_decode_per_sample.items():
                logit_item = logit_totals.setdefault(
                    name,
                    {
                        "mpjpe": per_sample_mpjpe.new_zeros((), dtype=torch.float32),
                        "root_aligned_mpjpe": per_sample_mpjpe.new_zeros((), dtype=torch.float32),
                        "token_acc": per_sample_mpjpe.new_zeros((), dtype=torch.float32),
                    },
                )
                logit_item["mpjpe"] = logit_item["mpjpe"] + values["mpjpe"][mask].sum()
                logit_item["root_aligned_mpjpe"] = (
                    logit_item["root_aligned_mpjpe"] + values["root_aligned_mpjpe"][mask].sum()
                )
                logit_item["token_acc"] = logit_item["token_acc"] + values["token_acc"][mask].sum()


def _finalize_mmbody_scenario_totals(totals: dict[str, dict[str, torch.Tensor]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for scenario, item in sorted(totals.items()):
        count = float(item["count"].item())
        denom = max(count, 1.0)
        scenario_out: dict[str, Any] = {
            "num_samples": int(round(count)),
            "mpjpe_mm": float((item["mpjpe"] / denom).item() * 1000.0),
            "root_aligned_mpjpe_mm": float((item["root_aligned_mpjpe"] / denom).item() * 1000.0),
            "token_acc": float((item["token_acc"] / denom).item()),
        }
        logit_decode = item.get("logit_decode", {})
        if logit_decode:
            scenario_out["logit_decode"] = {
                name: {
                    "mpjpe_mm": float((values["mpjpe"] / denom).item() * 1000.0),
                    "root_aligned_mpjpe_mm": float((values["root_aligned_mpjpe"] / denom).item() * 1000.0),
                    "token_acc": float((values["token_acc"] / denom).item()),
                }
                for name, values in sorted(logit_decode.items())
            }
        out[scenario] = scenario_out
    return out


def _merge_checkpoint_outputs(
    model: TokenHPEEstimatorModel,
    outputs_list: list[dict[str, torch.Tensor]],
    *,
    weights: list[float] | None = None,
) -> dict[str, torch.Tensor]:
    if not outputs_list:
        raise ValueError("Expected at least one checkpoint output.")
    if len(outputs_list) == 1:
        return outputs_list[0]

    merged = dict(outputs_list[0])
    stacked_z_e = torch.stack([item["pred_z_e"].float() for item in outputs_list], dim=0)
    if weights is None:
        weight_tensor = stacked_z_e.new_full((len(outputs_list),), 1.0 / float(len(outputs_list)))
    else:
        weight_tensor = stacked_z_e.new_tensor([float(item) for item in weights]).reshape(-1)
        if int(weight_tensor.numel()) != len(outputs_list):
            raise ValueError(f"Expected {len(outputs_list)} checkpoint weights, got {int(weight_tensor.numel())}.")
        if bool((weight_tensor < 0).any().item()) or float(weight_tensor.sum().item()) <= 0.0:
            raise ValueError("Checkpoint ensemble weights must be non-negative with positive sum.")
        weight_tensor = weight_tensor / weight_tensor.sum()

    pred_z_e = (stacked_z_e * weight_tensor.view(-1, 1, 1, 1)).sum(dim=0)
    pred_z_q, pred_indices = model.quantize_tokens(pred_z_e)
    pred_pose = model.vqvae.decode(pred_z_q)

    pred_logits_values = [item.get("pred_logits") for item in outputs_list]
    if all(value is not None for value in pred_logits_values):
        stacked_logits = torch.stack([value.float() for value in pred_logits_values if value is not None], dim=0)
        merged["pred_logits"] = (stacked_logits * weight_tensor.view(-1, 1, 1, 1)).sum(dim=0)
    elif any(value is not None for value in pred_logits_values):
        raise ValueError("All checkpoint ensemble members must either provide pred_logits or omit them.")

    merged["pred_z_e"] = pred_z_e
    merged["pred_z_q"] = pred_z_q
    merged["pred_indices"] = pred_indices
    merged["pred_pose"] = pred_pose
    return merged


def _forward_checkpoint_ensemble(
    models: list[TokenHPEEstimatorModel],
    batch: dict,
    device: torch.device,
    runtime,
    *,
    weights: list[float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not models:
        raise ValueError("Expected at least one checkpoint ensemble model.")
    target_poses_ref: torch.Tensor | None = None
    outputs_list: list[dict[str, torch.Tensor]] = []
    for model in models:
        target_poses, outputs = _forward_batch(model, batch, device, runtime)
        if target_poses_ref is None:
            target_poses_ref = target_poses
        outputs_list.append(outputs)
    if target_poses_ref is None:
        raise RuntimeError("Checkpoint ensemble did not produce target poses.")
    return target_poses_ref, _merge_checkpoint_outputs(models[0], outputs_list, weights=weights)


def _recipe_label(training_cfg: dict[str, Any]) -> str:
    if bool(training_cfg.get("freeze_mmwave_tokenizer", True)):
        return "single_stage_frozen_tokenizers_train_human_token_translator"
    return "single_stage_finetune_mmwave_tokenizer_train_human_token_translator"


@torch.no_grad()
def evaluate_hpe_token(
    *,
    model: TokenHPEEstimatorModel,
    dataloader: DataLoader,
    criterion,
    device: torch.device,
    runtime,
    config: dict[str, Any],
    split_name: str,
    soft_token_criterion=None,
    soft_token_weight: float = 0.0,
    include_action_groups: bool = True,
    include_per_action: bool = False,
    checkpoint_models: list[TokenHPEEstimatorModel] | None = None,
    checkpoint_weights: list[float] | None = None,
) -> dict[str, Any]:
    """Evaluate HPE-token and optionally summarize M4Human upstream test action groups."""

    max_key = f"max_{split_name}_batches"
    max_batches = config.get("training", {}).get(max_key)
    max_batches = int(max_batches) if max_batches else None
    display_name = DISPLAY_NAME
    training_cfg = config.get("training", {})
    pose_decode_cfg = _pose_decode_from_config(training_cfg)
    eval_models = list(checkpoint_models) if checkpoint_models is not None else [model]
    if not eval_models:
        raise ValueError("Expected at least one evaluation model.")
    model = eval_models[0]
    for eval_model in eval_models:
        eval_model.eval()
    totals = _zero_metric_tensors(
        [
            "total_loss",
            "latent_loss",
            "token_loss",
            "pose_loss",
            "soft_token_loss",
            "soft_token_weighted_loss",
            "soft_token_exact_acc",
            "soft_token_pred_regret_mm",
            "soft_token_valid_count",
        ],
        device,
    )
    logit_decode_specs = _logit_decode_specs_from_config(config)
    logit_decode_totals = {
        spec["name"]: {
            "mpjpe": torch.zeros((), device=device),
            "root_aligned_mpjpe": torch.zeros((), device=device),
            "token_acc": torch.zeros((), device=device),
            "entropy": torch.zeros((), device=device),
        }
        for spec in logit_decode_specs
    }
    total_mpjpe = torch.zeros((), device=device)
    total_root_error = torch.zeros((), device=device)
    total_root_aligned_mpjpe = torch.zeros((), device=device)
    total_token_acc = torch.zeros((), device=device)
    pose_decode_metric_totals = (
        {
            "hard_mpjpe": torch.zeros((), device=device),
            "pose_decode_mpjpe": torch.zeros((), device=device),
            "pose_decode_token_acc": torch.zeros((), device=device),
            "pose_decode_entropy": torch.zeros((), device=device),
        }
        if pose_decode_cfg is not None
        else None
    )
    group_accumulator = (
        M4HumanActionGroupAccumulator(["mpjpe"], device=device)
        if include_action_groups and str(config.get("dataset", {}).get("name", "")).lower() == "m4human"
        else None
    )
    group_joint_accumulator = (
        M4HumanActionGroupVectorAccumulator(["mpjpe", "root_aligned_mpjpe"], device=device)
        if include_action_groups and str(config.get("dataset", {}).get("name", "")).lower() == "m4human"
        else None
    )
    token_topk_cfg = _token_topk_from_config(config)
    token_topk_accumulator = (
        PartTokenTopKAccumulator(
            model.vqvae.layout,
            topk=token_topk_cfg["topk"],
            include_slots=bool(token_topk_cfg["include_slots"]),
        )
        if token_topk_cfg is not None
        else None
    )
    token_alignment_cfg = _token_alignment_from_config(config)
    token_alignment_collector = (
        TokenAlignmentCollector(
            model.vqvae.layout,
            shuffle_seed=token_alignment_cfg["shuffle_seed"],
        )
        if token_alignment_cfg is not None
        else None
    )
    token_topk_oracle_cfg = _token_topk_oracle_from_config(config)
    token_topk_oracle_totals = (
        {
            int(k): {
                "mpjpe": torch.zeros((), device=device),
                "root_aligned_mpjpe": torch.zeros((), device=device),
                "token_acc": torch.zeros((), device=device),
            }
            for k in token_topk_oracle_cfg["topk"]
        }
        if token_topk_oracle_cfg is not None
        else None
    )
    mmbody_scenario_totals: dict[str, dict[str, torch.Tensor]] = {}
    root_index = int(config.get("vqvae", {}).get("root_index", 0))
    joint_error_sum_m: torch.Tensor | None = None
    root_aligned_joint_error_sum_m: torch.Tensor | None = None
    sample_count = 0
    num_batches = 0
    data_time_sum = 0.0
    step_time_sum = 0.0
    import time

    t0 = time.perf_counter()
    loop_end_t = time.perf_counter()
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluate {display_name} {split_name}")):
        if max_batches is not None and batch_idx >= max_batches:
            break
        data_time_sum += time.perf_counter() - loop_end_t
        step_t0 = time.perf_counter()

        if len(eval_models) > 1:
            target_poses, outputs = _forward_checkpoint_ensemble(
                eval_models,
                batch,
                device,
                runtime,
                weights=checkpoint_weights,
            )
        else:
            target_poses, outputs = _forward_batch(model, batch, device, runtime)
        batch_size = int(target_poses.shape[0])
        pose_selection = _select_pose_outputs(model, outputs, training_cfg)
        loss_dict = criterion(
            pred_z_e=outputs["pred_z_e"],
            target_z_e=outputs["target_z_e"],
            pred_logits=outputs["pred_logits"],
            target_indices=outputs["target_indices"],
            pred_pose=pose_selection["loss_pose"],
            target_pose=target_poses,
        )
        loss_dict = _apply_optional_soft_token_loss(
            loss_dict,
            outputs,
            soft_token_criterion=soft_token_criterion,
            soft_token_weight=soft_token_weight,
        )
        pred_pose = pose_selection["metric_pose"]
        per_joint_errors_m = torch.linalg.norm(pred_pose.float() - target_poses.float(), dim=-1)
        if joint_error_sum_m is None:
            joint_error_sum_m = torch.zeros(int(per_joint_errors_m.shape[1]), device=device, dtype=torch.float64)
        joint_error_sum_m += per_joint_errors_m.double().sum(dim=0)
        per_sample_mpjpe = _per_sample_mpjpe(pred_pose, target_poses)
        per_sample_root_error = torch.linalg.norm(
            pred_pose.float()[:, root_index, :] - target_poses.float()[:, root_index, :],
            dim=-1,
        )
        pred_root_aligned = _root_aligned(pred_pose, root_index)
        target_root_aligned = _root_aligned(target_poses, root_index)
        root_aligned_joint_errors_m = torch.linalg.norm(
            pred_root_aligned.float() - target_root_aligned.float(),
            dim=-1,
        )
        if root_aligned_joint_error_sum_m is None:
            root_aligned_joint_error_sum_m = torch.zeros(
                int(root_aligned_joint_errors_m.shape[1]),
                device=device,
                dtype=torch.float64,
            )
        root_aligned_joint_error_sum_m += root_aligned_joint_errors_m.double().sum(dim=0)
        per_sample_root_aligned_mpjpe = _per_sample_mpjpe(
            pred_root_aligned,
            target_root_aligned,
        )
        token_acc_per_sample = (pose_selection["metric_indices"] == outputs["target_indices"]).float().mean(dim=1)
        scenario_names = _batch_string_values(batch, "mmbody_scenario", batch_size)
        if token_alignment_collector is not None:
            if "indicator" not in batch:
                raise ValueError(
                    "evaluation.token_alignment requires batch indicator metadata "
                    "[subject, action, frame]."
                )
            with autocast_context(runtime):
                ranked_indices = rank_part_codebook_candidates(
                    vqvae=model.vqvae,
                    pred_z_e=outputs["pred_z_e"].detach(),
                    pred_indices=outputs["pred_indices"].detach(),
                    max_k=max(token_alignment_collector.topk),
                )
            token_alignment_collector.update(
                pred_indices=outputs["pred_indices"],
                target_indices=outputs["target_indices"],
                indicators=batch["indicator"],
                root_relative_mpjpe=per_sample_root_aligned_mpjpe,
                ranked_indices=ranked_indices,
            )
        logit_decode_per_sample: dict[str, dict[str, torch.Tensor]] | None = {} if scenario_names else None
        if token_topk_accumulator is not None:
            token_topk_accumulator.update_logits(outputs["pred_logits"].detach(), outputs["target_indices"].detach())
            token_topk_accumulator.update_quantized(
                pose_selection["metric_indices"].detach(),
                outputs["target_indices"].detach(),
            )
        if pose_decode_metric_totals is not None and pose_selection["decoded"] is not None:
            decoded = pose_selection["decoded"]
            pose_decode_metric_totals["hard_mpjpe"] = _accumulate_metric(
                pose_decode_metric_totals["hard_mpjpe"],
                _per_sample_mpjpe(outputs["pred_pose"], target_poses).sum(),
                weight=1,
            )
            pose_decode_metric_totals["pose_decode_mpjpe"] = _accumulate_metric(
                pose_decode_metric_totals["pose_decode_mpjpe"],
                _per_sample_mpjpe(decoded.pose, target_poses).sum(),
                weight=1,
            )
            pose_decode_metric_totals["pose_decode_token_acc"] = _accumulate_metric(
                pose_decode_metric_totals["pose_decode_token_acc"],
                (decoded.indices == outputs["target_indices"]).float().mean(dim=1).sum(),
                weight=1,
            )
            pose_decode_metric_totals["pose_decode_entropy"] = _accumulate_metric(
                pose_decode_metric_totals["pose_decode_entropy"],
                decoded.entropy.float().mean(dim=1).sum(),
                weight=1,
            )
        if token_topk_oracle_totals is not None and token_topk_oracle_cfg is not None:
            oracle_logits = outputs.get("pred_logits")
            if oracle_logits is None:
                raise RuntimeError("evaluation.token_topk_oracle requires model outputs['pred_logits'].")
            for k in token_topk_oracle_totals:
                oracle_indices = _topk_oracle_indices_from_logits(
                    layout=model.vqvae.layout,
                    pred_logits=oracle_logits,
                    target_indices=outputs["target_indices"],
                    topk=int(k),
                )
                oracle_pose = _decode_part_token_indices(vqvae=model.vqvae, indices=oracle_indices)
                oracle_mpjpe = _per_sample_mpjpe(oracle_pose, target_poses)
                oracle_root_aligned_mpjpe = _per_sample_mpjpe(
                    _root_aligned(oracle_pose, root_index),
                    _root_aligned(target_poses, root_index),
                )
                oracle_token_acc = (oracle_indices == outputs["target_indices"]).float().mean(dim=1)
                token_topk_oracle_totals[k]["mpjpe"] = _accumulate_metric(
                    token_topk_oracle_totals[k]["mpjpe"],
                    oracle_mpjpe.sum(),
                    weight=1,
                )
                token_topk_oracle_totals[k]["root_aligned_mpjpe"] = _accumulate_metric(
                    token_topk_oracle_totals[k]["root_aligned_mpjpe"],
                    oracle_root_aligned_mpjpe.sum(),
                    weight=1,
                )
                token_topk_oracle_totals[k]["token_acc"] = _accumulate_metric(
                    token_topk_oracle_totals[k]["token_acc"],
                    oracle_token_acc.sum(),
                    weight=1,
                )
        for spec in logit_decode_specs:
            source = str(spec["source"])
            name = spec["name"]
            if source in {"learned", "learned_classifier"}:
                pred_logits = outputs.get("pred_logits")
                if pred_logits is None:
                    raise RuntimeError("Logit decode source 'learned' requires model outputs['pred_logits'].")
            elif source in {"blend", "learned_codebook_blend"}:
                learned_logits = outputs.get("pred_logits")
                if learned_logits is None:
                    raise RuntimeError("Logit decode source 'blend' requires model outputs['pred_logits'].")
                codebook_logits = part_codebook_distance_logits(
                    vqvae=model.vqvae,
                    pred_z_e=outputs["pred_z_e"],
                    scale=float(spec["scale"]),
                )
                pred_logits = (
                    float(spec["learned_weight"]) * learned_logits.float()
                    + float(spec["codebook_weight"]) * codebook_logits
                )
            else:
                pred_logits = part_codebook_distance_logits(
                    vqvae=model.vqvae,
                    pred_z_e=outputs["pred_z_e"],
                    scale=float(spec["scale"]),
                )
            decoded = decode_part_token_logits(
                vqvae=model.vqvae,
                pred_logits=pred_logits,
                mode=spec["mode"],
                temperature=float(spec["temperature"]),
            )
            decoded_mpjpe = _per_sample_mpjpe(decoded.pose, target_poses)
            decoded_root_aligned_mpjpe = _per_sample_mpjpe(
                _root_aligned(decoded.pose, root_index),
                _root_aligned(target_poses, root_index),
            )
            decoded_token_acc = (decoded.indices == outputs["target_indices"]).float().mean(dim=1)
            if logit_decode_per_sample is not None:
                logit_decode_per_sample[name] = {
                    "mpjpe": decoded_mpjpe.detach(),
                    "root_aligned_mpjpe": decoded_root_aligned_mpjpe.detach(),
                    "token_acc": decoded_token_acc.detach(),
                }
            logit_decode_totals[name]["mpjpe"] = _accumulate_metric(
                logit_decode_totals[name]["mpjpe"],
                decoded_mpjpe.sum(),
                weight=1,
            )
            logit_decode_totals[name]["root_aligned_mpjpe"] = _accumulate_metric(
                logit_decode_totals[name]["root_aligned_mpjpe"],
                decoded_root_aligned_mpjpe.sum(),
                weight=1,
            )
            logit_decode_totals[name]["token_acc"] = _accumulate_metric(
                logit_decode_totals[name]["token_acc"],
                decoded_token_acc.sum(),
                weight=1,
            )
            logit_decode_totals[name]["entropy"] = _accumulate_metric(
                logit_decode_totals[name]["entropy"],
                decoded.entropy.float().mean(dim=1).sum(),
                weight=1,
            )
        _update_mmbody_scenario_totals(
            mmbody_scenario_totals,
            scenario_names,
            per_sample_mpjpe=per_sample_mpjpe.detach(),
            per_sample_root_aligned_mpjpe=per_sample_root_aligned_mpjpe.detach(),
            token_acc_per_sample=token_acc_per_sample.detach(),
            logit_decode_per_sample=logit_decode_per_sample,
        )
        for key in totals:
            if key in loss_dict:
                totals[key] = _accumulate_metric(
                    totals[key],
                    loss_dict[key],
                    weight=1 if key == "soft_token_valid_count" else batch_size,
                )
        total_mpjpe = _accumulate_metric(total_mpjpe, per_sample_mpjpe.sum(), weight=1)
        total_root_error = _accumulate_metric(total_root_error, per_sample_root_error.sum(), weight=1)
        total_root_aligned_mpjpe = _accumulate_metric(
            total_root_aligned_mpjpe,
            per_sample_root_aligned_mpjpe.sum(),
            weight=1,
        )
        total_token_acc = _accumulate_metric(total_token_acc, token_acc_per_sample.sum(), weight=1)
        if group_accumulator is not None and "indicator" in batch:
            action_ids = m4human_action_ids_from_indicator(batch["indicator"], device=device)
            group_accumulator.update("mpjpe", action_ids, per_sample_mpjpe)
            if group_joint_accumulator is not None:
                group_joint_accumulator.update("mpjpe", action_ids, per_joint_errors_m)
                group_joint_accumulator.update("root_aligned_mpjpe", action_ids, root_aligned_joint_errors_m)
        sample_count += batch_size
        num_batches += 1
        step_time_sum += time.perf_counter() - step_t0
        loop_end_t = time.perf_counter()

    if num_batches == 0:
        raise RuntimeError(f"No {split_name} batches were processed for {display_name}.")
    metrics: dict[str, Any] = _finalize_metric_tensors(totals, sample_count, count_keys={"soft_token_valid_count"})
    metrics["mpjpe"] = float((total_mpjpe / float(max(1, sample_count))).item())
    metrics["mpjpe_mm"] = metrics["mpjpe"] * 1000.0
    if joint_error_sum_m is not None:
        metrics["mpjpe_by_joint_mm"] = [
            float(item) for item in (joint_error_sum_m / float(max(1, sample_count)) * 1000.0).detach().cpu().tolist()
        ]
    metrics["root_error"] = float((total_root_error / float(max(1, sample_count))).item())
    metrics["root_error_mm"] = metrics["root_error"] * 1000.0
    metrics["root_aligned_mpjpe"] = float((total_root_aligned_mpjpe / float(max(1, sample_count))).item())
    metrics["root_aligned_mpjpe_mm"] = metrics["root_aligned_mpjpe"] * 1000.0
    if root_aligned_joint_error_sum_m is not None:
        metrics["root_aligned_mpjpe_by_joint_mm"] = [
            float(item)
            for item in (root_aligned_joint_error_sum_m / float(max(1, sample_count)) * 1000.0)
            .detach()
            .cpu()
            .tolist()
        ]
    metrics["token_acc"] = float((total_token_acc / float(max(1, sample_count))).item())
    if pose_decode_metric_totals is not None:
        for key, value in pose_decode_metric_totals.items():
            metrics[key] = float((value / float(max(1, sample_count))).item())
        metrics["hard_mpjpe_mm"] = metrics["hard_mpjpe"] * 1000.0
        metrics["pose_decode_mpjpe_mm"] = metrics["pose_decode_mpjpe"] * 1000.0
        metrics["pose_decode"] = pose_decode_cfg
    for spec in logit_decode_specs:
        name = spec["name"]
        prefix = f"logit_decode_{name}"
        item = logit_decode_totals[name]
        metrics[f"{prefix}_mpjpe"] = float((item["mpjpe"] / float(max(1, sample_count))).item())
        metrics[f"{prefix}_mpjpe_mm"] = metrics[f"{prefix}_mpjpe"] * 1000.0
        metrics[f"{prefix}_root_aligned_mpjpe"] = float(
            (item["root_aligned_mpjpe"] / float(max(1, sample_count))).item()
        )
        metrics[f"{prefix}_root_aligned_mpjpe_mm"] = metrics[f"{prefix}_root_aligned_mpjpe"] * 1000.0
        metrics[f"{prefix}_token_acc"] = float((item["token_acc"] / float(max(1, sample_count))).item())
        metrics[f"{prefix}_entropy"] = float((item["entropy"] / float(max(1, sample_count))).item())
    if logit_decode_specs:
        metrics["logit_decode_specs"] = logit_decode_specs
    if token_alignment_collector is not None:
        metrics["token_alignment"] = token_alignment_collector.summary()
    if token_topk_accumulator is not None:
        token_topk = token_topk_accumulator.summary()
        metrics["token_topk"] = token_topk
        logit_overall = token_topk["logit"]["overall"]
        for key, value in logit_overall.items():
            if key != "count":
                metrics[f"token_topk_logit_{key}"] = float(value)
        quantized_overall = token_topk["quantized"]["overall"]
        for key, value in quantized_overall.items():
            if key != "count":
                metrics[f"token_topk_quantized_{key}"] = float(value)
    if token_topk_oracle_totals is not None and token_topk_oracle_cfg is not None:
        oracle_summary: dict[str, Any] = {
            "source": token_topk_oracle_cfg["source"],
            "topk": [int(k) for k in token_topk_oracle_cfg["topk"]],
            "overall": {},
        }
        for k, item in token_topk_oracle_totals.items():
            key = f"top{int(k)}"
            mpjpe = float((item["mpjpe"] / float(max(1, sample_count))).item())
            root_aligned_mpjpe = float((item["root_aligned_mpjpe"] / float(max(1, sample_count))).item())
            token_acc = float((item["token_acc"] / float(max(1, sample_count))).item())
            oracle_summary["overall"][key] = {
                "mpjpe_mm": mpjpe * 1000.0,
                "root_aligned_mpjpe_mm": root_aligned_mpjpe * 1000.0,
                "token_acc": token_acc,
                "num_samples": int(sample_count),
            }
            metrics[f"token_topk_oracle_{key}_mpjpe_mm"] = mpjpe * 1000.0
            metrics[f"token_topk_oracle_{key}_root_aligned_mpjpe_mm"] = root_aligned_mpjpe * 1000.0
            metrics[f"token_topk_oracle_{key}_token_acc"] = token_acc
        metrics["token_topk_oracle"] = oracle_summary
    if mmbody_scenario_totals:
        metrics["mmbody_scenario_metrics"] = _finalize_mmbody_scenario_totals(mmbody_scenario_totals)
    metrics["num_samples"] = int(sample_count)
    metrics["num_batches"] = int(num_batches)
    metrics["samples_per_sec"] = float(sample_count / max(time.perf_counter() - t0, 1e-6))
    metrics["data_time_sec"] = float(data_time_sum)
    metrics["step_time_sec"] = float(step_time_sum)
    _merge_action_group_metrics(metrics, group_accumulator, include_per_action=include_per_action)
    if group_joint_accumulator is not None:
        joint_summary = group_joint_accumulator.summary(scale=1000.0)
        if "mpjpe" in joint_summary:
            metrics["mpjpe_by_action_group_by_joint_mm"] = joint_summary["mpjpe"]["by_group"]
        if "root_aligned_mpjpe" in joint_summary:
            metrics["root_aligned_mpjpe_by_action_group_by_joint_mm"] = joint_summary["root_aligned_mpjpe"]["by_group"]
    return metrics


def _load_hpe_token_model_for_eval(
    config: dict[str, Any],
    *,
    checkpoint: str | Path,
    device: torch.device,
) -> TokenHPEEstimatorModel:
    training_cfg = config.get("training", {})
    model = TokenHPEEstimatorModel(
        mmwave_tokenizer_config=config["mmwave_tokenizer"],
        sensor_token_config=config.get("sensor_tokens", {}),
        token_translator_config=config["token_translator"],
        vqvae_config=config["vqvae"],
        device=str(device),
        freeze_mmwave_tokenizer=bool(training_cfg.get("freeze_mmwave_tokenizer", True)),
        freeze_token_translator=bool(training_cfg.get("freeze_token_translator", False)),
        freeze_vqvae=bool(training_cfg.get("freeze_vqvae", True)),
    )
    pretrained = config.get("pretrained", {})
    model.load_pretrained_weights(
        mmwave_tokenizer_path=pretrained.get("mmwave_tokenizer"),
        vqvae_path=pretrained.get("vqvae"),
        device=str(device),
    )
    ckpt = load_checkpoint(checkpoint, map_location=device)
    load_optional_component_states(model, ckpt, prefix="[HPE Token Eval]")
    return model.to(device)


def run_hpe_token_evaluation(
    config: dict[str, Any],
    *,
    checkpoint: str | Path,
    config_path: str | Path | None = None,
    eval_split: str = "test",
    device: str = "cuda",
    output_path: str | Path | None = None,
    max_eval_batches: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    include_action_groups: bool = True,
    include_per_action: bool = False,
) -> dict[str, Any]:
    config = _normalize_training_config(config)
    if eval_split not in {"train", "val", "test"}:
        raise ValueError(f"eval_split must be train/val/test, got {eval_split!r}")
    training_cfg = config.setdefault("training", {})
    if max_eval_batches is not None:
        training_cfg[f"max_{eval_split}_batches"] = int(max_eval_batches)
    device_obj = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    runtime = configure_runtime(training_cfg, device_obj, default_amp=True, log_prefix="[HPE Token Eval Runtime]")

    point_ensemble_views = _point_ensemble_views_from_config(config)
    if point_ensemble_views:
        dataset = _PointEnsembleDataset(config, split=eval_split, views=point_ensemble_views)
    else:
        dataset = create_radar_pose_dataset(config, split=eval_split, device="cpu")
    loader = _build_eval_loader(dataset, training_cfg=training_cfg, batch_size=batch_size, num_workers=num_workers)
    if max_eval_batches is not None:
        loader = LimitedDataLoader(loader, int(max_eval_batches))
    checkpoint_ensemble = _checkpoint_ensemble_from_config(config)
    if checkpoint_ensemble is not None:
        checkpoint_models = [
            _load_hpe_token_model_for_eval(config, checkpoint=item, device=device_obj)
            for item in checkpoint_ensemble["checkpoints"]
        ]
        model = checkpoint_models[0]
        checkpoint_weights = list(checkpoint_ensemble["weights"])
    else:
        model = _load_hpe_token_model_for_eval(config, checkpoint=checkpoint, device=device_obj)
        checkpoint_models = None
        checkpoint_weights = None
    criterion = _token_loss_from_config(training_cfg)
    soft_token_criterion, soft_token_weight = _soft_token_loss_from_config(training_cfg, model.vqvae.layout)
    if soft_token_criterion is not None:
        soft_token_criterion = soft_token_criterion.to(device_obj)
    metrics = evaluate_hpe_token(
        model=model,
        dataloader=loader,
        criterion=criterion,
        soft_token_criterion=soft_token_criterion,
        soft_token_weight=soft_token_weight,
        device=device_obj,
        runtime=runtime,
        config=config,
        split_name=eval_split,
        include_action_groups=include_action_groups,
        include_per_action=include_per_action,
        checkpoint_models=checkpoint_models,
        checkpoint_weights=checkpoint_weights,
    )
    payload = {
        "stage": STAGE_NAME,
        "dataset": config.get("dataset", {}).get("name"),
        "eval_split": eval_split,
        "split_id": config.get("m4human", {}).get("split"),
        "checkpoint": str(checkpoint),
        "config": str(config_path) if config_path else None,
        "precision": runtime.precision,
        "amp_enabled": runtime.amp_enabled,
        "point_ensemble": point_ensemble_views or None,
        "checkpoint_ensemble": checkpoint_ensemble,
        "metrics": metrics,
    }
    out_path = Path(output_path).expanduser() if output_path else (
        Path(checkpoint).expanduser().parent.parent
        / f"metrics_{_eval_suffix(config, eval_split)}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["output_path"] = str(out_path)
    print(
        "[HPE Token Eval] "
        f"{_eval_suffix(config, eval_split)} | "
        f"mpjpe={metrics['mpjpe_mm']:.2f}mm, "
        f"root_aligned={metrics['root_aligned_mpjpe_mm']:.2f}mm, "
        f"token_acc={metrics['token_acc'] * 100.0:.1f}%"
    )
    by_group = metrics.get("mpjpe_by_action_group_mm")
    if isinstance(by_group, dict):
        parts = [f"{key}={value:.2f}mm" for key, value in by_group.items() if value is not None]
        if parts:
            print("[HPE Token Eval] Action groups | " + ", ".join(parts))
    print(f"[HPE Token Eval] Wrote {out_path}")
    return payload


def run_hpe_token(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    resume: str | Path | None = None,
    work_dir: str | Path | None = None,
    device: str = "cuda",
    seed: int | None = None,
    reset_epoch: bool = False,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> dict[str, Any]:
    config = _normalize_training_config(config)
    if seed is not None:
        config.setdefault("training", {})["seed"] = int(seed)
    training_cfg = config.get("training", {})
    if training_cfg.get("seed") is not None:
        set_global_seed(int(training_cfg["seed"]), deterministic=bool(training_cfg.get("deterministic", False)))

    default_name = str(training_cfg.get("run_name", STAGE_NAME))
    output_root, checkpoint_dir, log_dir = resolve_run_root(
        config,
        work_dir=work_dir,
        default_name=default_name,
        experiment_name=EXPERIMENT_NAME,
    )
    effective_config_path = write_effective_config(config, output_root)
    device_obj = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    runtime = configure_runtime(training_cfg, device_obj, default_amp=True, log_prefix="[HPE Token Runtime]")

    model = TokenHPEEstimatorModel(
        mmwave_tokenizer_config=config["mmwave_tokenizer"],
        sensor_token_config=config.get("sensor_tokens", {}),
        token_translator_config=config["token_translator"],
        vqvae_config=config["vqvae"],
        device=str(device_obj),
        freeze_mmwave_tokenizer=bool(training_cfg.get("freeze_mmwave_tokenizer", True)),
        freeze_token_translator=bool(training_cfg.get("freeze_token_translator", False)),
        freeze_vqvae=bool(training_cfg.get("freeze_vqvae", True)),
    )
    pretrained = config.get("pretrained", {})
    model.load_pretrained_weights(
        mmwave_tokenizer_path=pretrained.get("mmwave_tokenizer"),
        vqvae_path=pretrained.get("vqvae"),
        device=str(device_obj),
    )
    model = model.to(device_obj)

    data_bundle = build_radar_pose_dataloaders(
        config=config,
        checkpoint_dir=output_root,
        include_test=False,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
    )
    write_run_manifest(
        output_root,
        config_path=effective_config_path,
        checkpoint_path=resume or pretrained.get("mmwave_tokenizer"),
        config=config,
        dataset_split=str(data_bundle.split_manifest.get("protocol", data_bundle.split_manifest.get("val_split", ""))),
        coord_transform=config.get("coord_transform"),
        output_paths={"output_root": output_root, "checkpoint_dir": checkpoint_dir, "log_dir": log_dir},
        extra={
            "stage": STAGE_NAME,
            "split_manifest": data_bundle.split_manifest,
            "mmwave_tokenizer_checkpoint": pretrained.get("mmwave_tokenizer"),
            "body_vqvae_checkpoint": pretrained.get("vqvae"),
            "sensor_token_dim": model.sensor_token_dim,
            "sensor_token_mode": model.sensor_token_mode,
            "recipe": _recipe_label(training_cfg),
        },
    )

    data_bundle = maybe_enable_frozen_feature_cache(
        data_bundle=data_bundle,
        model=model,
        training_cfg=training_cfg,
        runtime=runtime,
        device=device_obj,
        log_prefix="[HPE Token Runtime]",
        cache_tag=_cache_tag(config, pretrained),
    )
    _apply_token_classifier_only_training(model, training_cfg)
    print(f"[HPE Token] Train samples: {data_bundle.train_count}")
    print(f"[HPE Token] Val samples: {data_bundle.val_count}")
    print(
        "[HPE Token] Model | "
        f"mmwave_tokenizer={config['mmwave_tokenizer'].get('method')} "
        f"sensor_tokens={model.sensor_token_num}x{model.sensor_token_dim} "
        f"human_tokens={config['token_translator'].get('num_pose_tokens')}"
    )

    criterion = _token_loss_from_config(training_cfg)
    soft_token_criterion, soft_token_weight = _soft_token_loss_from_config(training_cfg, model.vqvae.layout)
    if soft_token_criterion is not None:
        soft_token_criterion = soft_token_criterion.to(device_obj)
        print(f"[HPE Token] Soft token CE enabled | weight={soft_token_weight:.4g}")
    point_augmentation_cfg = _point_augmentation_from_config(training_cfg)
    if point_augmentation_cfg is not None:
        print(
            "[HPE Token] Point augmentation enabled | "
            f"clutter={point_augmentation_cfg['clutter_replace_prob']:.2f}/"
            f"{point_augmentation_cfg['clutter_fraction']:.2f}, "
            f"dropout={point_augmentation_cfg['point_dropout_prob']:.2f}/"
            f"{point_augmentation_cfg['point_dropout_fraction']:.2f}, "
            f"cutout={point_augmentation_cfg['spatial_cutout_prob']:.2f}/"
            f"{point_augmentation_cfg['spatial_cutout_width']:.2f}, "
            f"xyz_jitter={point_augmentation_cfg['xyz_jitter_std']:.3f}"
        )
    optimizer = build_hpe_token_optimizer(model, config)
    grad_scaler = build_grad_scaler(runtime)
    epochs = int(training_cfg.get("epochs", 20))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(training_cfg.get("min_lr", 1e-6)),
    )

    start_epoch = 0
    best_val_mpjpe = float("inf")
    if resume:
        ckpt = load_checkpoint(resume, map_location=device_obj)
        load_optional_component_states(model, ckpt, prefix="[HPE Token]")
        if not reset_epoch:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if ckpt.get("amp_state_dict") and grad_scaler.is_enabled():
                grad_scaler.load_state_dict(ckpt["amp_state_dict"])
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            best_val_mpjpe = float(ckpt.get("best_val_mpjpe", float("inf")))
        else:
            print("[HPE Token] Resetting epoch after resume.")

    model.token_translator = maybe_compile(
        model.token_translator,
        runtime,
        module_name="hpe_token_translator",
        log_prefix="[HPE Token Runtime]",
    )
    print(
        "[HPE Token] Trainable parameters | "
        f"human_token_translator={safe_param_count([p for p in model.token_translator.parameters() if p.requires_grad]):,}, "
        f"mmwave_tokenizer={_encoder_trainable_count(model):,}, "
        f"body_vqvae={safe_param_count([p for p in model.vqvae.parameters() if p.requires_grad]):,}"
    )

    history: list[dict[str, Any]] = []
    writer = SummaryWriter(str(log_dir))
    writer.add_text("hpe_token/run_name", str(training_cfg.get("run_name", "")), 0)
    writer.add_text("hpe_token/mmwave_tokenizer_checkpoint", str(pretrained.get("mmwave_tokenizer", "")), 0)
    writer.add_text("hpe_token/body_vqvae_checkpoint", str(pretrained.get("vqvae", "")), 0)
    try:
        for epoch in range(start_epoch, epochs):
            print(f"\n[HPE Token] Epoch {epoch + 1}/{epochs} | lr={optimizer.param_groups[0]['lr']:.6e}")
            train_metrics = train_hpe_token_epoch(
                model=model,
                dataloader=data_bundle.train_loader,
                criterion=criterion,
                soft_token_criterion=soft_token_criterion,
                soft_token_weight=soft_token_weight,
                optimizer=optimizer,
                device=device_obj,
                epoch=epoch,
                writer=writer,
                config=config,
                runtime=runtime,
                grad_scaler=grad_scaler,
            )
            val_metrics = validate_hpe_token(
                model=model,
                dataloader=data_bundle.val_loader,
                criterion=criterion,
                soft_token_criterion=soft_token_criterion,
                soft_token_weight=soft_token_weight,
                device=device_obj,
                epoch=epoch,
                writer=writer,
                config=config,
                runtime=runtime,
            )
            scheduler.step()
            history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
            print(
                "[HPE Token] Val | "
                f"loss={val_metrics['total_loss']:.4f}, pose={val_metrics['mpjpe'] * 1000.0:.2f}mm, "
                f"token_acc={val_metrics['token_acc'] * 100.0:.1f}%"
            )

            checkpoint = {
                "epoch": epoch,
                "stage": STAGE_NAME,
                "mmwave_tokenizer_state_dict": module_state_dict(model.mmwave_tokenizer),
                "vqvae_state_dict": module_state_dict(model.vqvae),
                "token_translator_state_dict": module_state_dict(model.token_translator),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "amp_state_dict": grad_scaler.state_dict() if grad_scaler.is_enabled() else None,
                "best_val_mpjpe": best_val_mpjpe,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "config": config,
            }
            if float(val_metrics["mpjpe"]) < best_val_mpjpe:
                best_val_mpjpe = float(val_metrics["mpjpe"])
                checkpoint["best_val_mpjpe"] = best_val_mpjpe
                save_checkpoint(checkpoint_dir / "best.pth", checkpoint)
                print(f"[HPE Token] Saved best model (val_mpjpe={best_val_mpjpe * 1000.0:.2f}mm)")
            save_checkpoint(checkpoint_dir / "latest.pth", checkpoint)
            _write_summary(
                output_root,
                config_path=effective_config_path,
                checkpoint_dir=checkpoint_dir,
                log_dir=log_dir,
                runtime=runtime,
                epochs=epochs,
                best_val_mpjpe=best_val_mpjpe,
                history=history,
            )
    finally:
        writer.close()

    summary = _write_summary(
        output_root,
        config_path=effective_config_path,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        runtime=runtime,
        epochs=epochs,
        best_val_mpjpe=best_val_mpjpe,
        history=history,
    )
    print(f"[HPE Token] Training completed. Best val MPJPE: {best_val_mpjpe * 1000.0:.2f} mm")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train token-based human pose estimation")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--resume", type=str, default=None, help="Resume HPE token checkpoint")
    parser.add_argument("--work-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--reset-epoch", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    args = parser.parse_args()
    run_hpe_token(
        load_config(args.config),
        config_path=args.config,
        resume=args.resume,
        work_dir=args.work_dir,
        device=args.device,
        seed=args.seed,
        reset_epoch=args.reset_epoch,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )


if __name__ == "__main__":
    main()
