# -*- coding: utf-8 -*-
"""Reusable HPE-token training helpers."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    from ..common import safe_param_count
    from ...models.hpe_token.decoder import decode_part_token_logits, part_codebook_distance_logits
    from ...models.token_translation.soft_token_loss import SoftMPJPETokenCELoss
    from ...models.token_translation import TokenTranslatorLoss
    from ...utils.metrics import compute_mpjpe
    from ...utils.runtime_utils import autocast_context, mark_compiled_step
    from .hpe_token_ablation_losses import (
        ABLATION_LOSS_COUNT_KEYS,
        ABLATION_LOSS_METRIC_KEYS,
        _ablation_loss_config_from_training_config,
        _apply_optional_ablation_losses,
    )
except ImportError:  # Support scripts that put src/ directly on sys.path.
    from wave2body.train.common import safe_param_count
    from wave2body.models.hpe_token.decoder import decode_part_token_logits, part_codebook_distance_logits
    from wave2body.models.token_translation.soft_token_loss import SoftMPJPETokenCELoss
    from wave2body.models.token_translation import TokenTranslatorLoss
    from wave2body.utils.metrics import compute_mpjpe
    from wave2body.utils.runtime_utils import autocast_context, mark_compiled_step
    from wave2body.train.pipelines.hpe_token_ablation_losses import (
        ABLATION_LOSS_COUNT_KEYS,
        ABLATION_LOSS_METRIC_KEYS,
        _ablation_loss_config_from_training_config,
        _apply_optional_ablation_losses,
    )


_COUNT_METRIC_KEYS = {
    "soft_token_valid_count",
}


def _token_loss_from_config(training_cfg: dict) -> TokenTranslatorLoss:
    return TokenTranslatorLoss(
        latent_loss_weight=float(training_cfg.get("latent_loss_weight", 1.0)),
        token_loss_weight=float(training_cfg.get("token_loss_weight", 0.5)),
        pose_loss_weight=float(training_cfg.get("pose_loss_weight", 0.1)),
        label_smoothing=float(training_cfg.get("label_smoothing", 0.0)),
        normalize_latent=bool(training_cfg.get("normalize_latent", False)),
    )


def _soft_token_loss_from_config(training_cfg: dict, layout) -> tuple[SoftMPJPETokenCELoss | None, float]:
    cfg = training_cfg.get("soft_token_ce")
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return None, 0.0
    weight = float(cfg.get("weight", 1.0))
    if weight <= 0:
        return None, 0.0
    table_path = cfg.get("table_path")
    if not table_path:
        raise ValueError("training.soft_token_ce.table_path is required when soft-token CE is enabled")
    criterion = SoftMPJPETokenCELoss(
        layout=layout,
        table_path=table_path,
        tau_mm=float(cfg.get("tau_mm", 40.0)),
        d_clip_mm=float(cfg.get("d_clip_mm", 150.0)),
    )
    return criterion, weight


def _apply_optional_soft_token_loss(
    loss_dict: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    *,
    soft_token_criterion: SoftMPJPETokenCELoss | None,
    soft_token_weight: float,
) -> dict[str, torch.Tensor]:
    if soft_token_criterion is None or soft_token_weight <= 0:
        return loss_dict
    soft = soft_token_criterion(
        pred_logits=outputs["pred_logits"],
        target_indices=outputs["target_indices"],
    )
    merged = dict(loss_dict)
    weighted = float(soft_token_weight) * soft["soft_token_loss"]
    merged["total_loss"] = merged["total_loss"] + weighted
    merged["soft_token_loss"] = soft["soft_token_loss"]
    merged["soft_token_weighted_loss"] = weighted
    for key in ("soft_token_exact_acc", "soft_token_pred_regret_mm", "soft_token_valid_count"):
        if key in soft:
            merged[key] = soft[key]
    return merged


_COUNT_METRIC_KEYS.update(ABLATION_LOSS_COUNT_KEYS)


def _prob_from_config(cfg: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = float(cfg.get(key, default))
    if value < 0.0 or value > 1.0:
        raise ValueError(f"training.point_augmentation.{key} must be in [0, 1], got {value}")
    return value


def _fraction_from_config(cfg: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = float(cfg.get(key, default))
    if value < 0.0 or value > 1.0:
        raise ValueError(f"training.point_augmentation.{key} must be in [0, 1], got {value}")
    return value


def _range_from_config(cfg: dict[str, Any], key: str, default: tuple[float, float]) -> tuple[float, float]:
    value = cfg.get(key, default)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"training.point_augmentation.{key} must be a two-item range.")
    low = float(value[0])
    high = float(value[1])
    if not low < high:
        raise ValueError(f"training.point_augmentation.{key} must satisfy low < high, got {value!r}")
    return low, high


def _cutout_axes_from_config(cfg: dict[str, Any]) -> tuple[int, ...]:
    axes = cfg.get("spatial_cutout_axes", cfg.get("spatial_cutout_axis", "xyz"))
    if isinstance(axes, str):
        value = axes.strip().lower()
        if value in {"random", "xyz", "all"}:
            return (0, 1, 2)
        if value in {"x", "0"}:
            return (0,)
        if value in {"y", "1"}:
            return (1,)
        if value in {"z", "2"}:
            return (2,)
        raise ValueError(f"Unsupported training.point_augmentation.spatial_cutout_axis={axes!r}")
    if not isinstance(axes, (list, tuple)):
        raise ValueError("training.point_augmentation.spatial_cutout_axes must be a string or list.")
    mapping = {"x": 0, "y": 1, "z": 2, "0": 0, "1": 1, "2": 2}
    out: list[int] = []
    for axis in axes:
        key = str(axis).strip().lower()
        if key not in mapping:
            raise ValueError(f"Unsupported spatial cutout axis {axis!r}")
        out.append(mapping[key])
    if not out:
        raise ValueError("training.point_augmentation.spatial_cutout_axes cannot be empty.")
    return tuple(sorted(set(out)))


def _point_augmentation_from_config(training_cfg: dict) -> dict[str, Any] | None:
    cfg = training_cfg.get("point_augmentation")
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return None

    clutter_fraction = _fraction_from_config(
        cfg,
        "clutter_fraction",
        float(cfg.get("clutter_replace_fraction", 0.0)),
    )
    out = {
        "enabled": True,
        "clutter_replace_prob": _prob_from_config(
            cfg,
            "clutter_replace_prob",
            float(cfg.get("clutter_prob", 0.0)),
        ),
        "clutter_fraction": clutter_fraction,
        "point_dropout_prob": _prob_from_config(cfg, "point_dropout_prob", 0.0),
        "point_dropout_fraction": _fraction_from_config(cfg, "point_dropout_fraction", 0.0),
        "spatial_cutout_prob": _prob_from_config(cfg, "spatial_cutout_prob", 0.0),
        "spatial_cutout_width": float(cfg.get("spatial_cutout_width", 0.0)),
        "spatial_cutout_axes": _cutout_axes_from_config(cfg),
        "xyz_jitter_std": float(cfg.get("xyz_jitter_std", 0.0)),
        "feature_jitter_std": float(cfg.get("feature_jitter_std", 0.0)),
        "x_range": _range_from_config(cfg, "x_range", (-1.0, 1.0)),
        "y_range": _range_from_config(cfg, "y_range", (-1.0, 1.0)),
        "z_range": _range_from_config(cfg, "z_range", (-1.0, 1.0)),
    }
    if out["spatial_cutout_width"] < 0:
        raise ValueError("training.point_augmentation.spatial_cutout_width must be >= 0")
    if out["xyz_jitter_std"] < 0 or out["feature_jitter_std"] < 0:
        raise ValueError("training.point_augmentation jitter std values must be >= 0")
    return out


def _valid_point_mask(points: torch.Tensor) -> torch.Tensor:
    xyz = points[..., :3].float()
    finite = torch.isfinite(points).all(dim=-1)
    return finite & (xyz.abs().sum(dim=-1) > 0.0)


def _augmentation_count(num_valid: int, fraction: float) -> int:
    if num_valid <= 0 or fraction <= 0.0:
        return 0
    return min(num_valid, max(1, int(round(float(num_valid) * float(fraction)))))


def _apply_point_augmentation(points: torch.Tensor, cfg: dict[str, Any] | None) -> torch.Tensor:
    """Apply training-only point perturbations to padded mmWave point clouds."""
    if cfg is None or not bool(cfg.get("enabled", False)):
        return points
    if points.dim() != 3 or points.shape[-1] < 3:
        raise ValueError(f"mmwave_points must be [B,N,C>=3], got {tuple(points.shape)}")

    out = points.clone()
    device = out.device
    dtype = out.dtype
    ranges = (
        tuple(cfg["x_range"]),
        tuple(cfg["y_range"]),
        tuple(cfg["z_range"]),
    )

    clutter_prob = float(cfg.get("clutter_replace_prob", 0.0))
    clutter_fraction = float(cfg.get("clutter_fraction", 0.0))
    cutout_prob = float(cfg.get("spatial_cutout_prob", 0.0))
    cutout_width = float(cfg.get("spatial_cutout_width", 0.0))
    dropout_prob = float(cfg.get("point_dropout_prob", 0.0))
    dropout_fraction = float(cfg.get("point_dropout_fraction", 0.0))
    axes = tuple(int(axis) for axis in cfg.get("spatial_cutout_axes", (0, 1, 2)))

    for batch_idx in range(int(out.shape[0])):
        valid = _valid_point_mask(out[batch_idx])
        valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
        num_valid = int(valid_idx.numel())
        if num_valid <= 0:
            continue

        if clutter_prob > 0.0 and clutter_fraction > 0.0 and float(torch.rand((), device=device).item()) < clutter_prob:
            replace_count = _augmentation_count(num_valid, clutter_fraction)
            replace_idx = valid_idx[torch.randperm(num_valid, device=device)[:replace_count]]
            xyz = []
            for low, high in ranges:
                xyz.append(torch.empty(replace_count, device=device, dtype=dtype).uniform_(float(low), float(high)))
            out[batch_idx, replace_idx, :3] = torch.stack(xyz, dim=-1)
            if out.shape[-1] > 3:
                source_features = points[batch_idx, valid_idx, 3:].to(device=device, dtype=dtype)
                source_idx = torch.randint(0, num_valid, (replace_count,), device=device)
                out[batch_idx, replace_idx, 3:] = source_features[source_idx]

        if cutout_prob > 0.0 and cutout_width > 0.0 and float(torch.rand((), device=device).item()) < cutout_prob:
            valid = _valid_point_mask(out[batch_idx])
            if bool(valid.any().item()):
                axis = axes[int(torch.randint(0, len(axes), (), device=device).item())]
                low, high = ranges[axis]
                center = float(torch.empty((), device=device).uniform_(float(low), float(high)).item())
                half_width = 0.5 * cutout_width
                cut = valid & (out[batch_idx, :, axis].float() >= center - half_width)
                cut = cut & (out[batch_idx, :, axis].float() <= center + half_width)
                out[batch_idx, cut, :] = 0.0

        if dropout_prob > 0.0 and dropout_fraction > 0.0 and float(torch.rand((), device=device).item()) < dropout_prob:
            valid = _valid_point_mask(out[batch_idx])
            valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
            num_valid = int(valid_idx.numel())
            drop_count = _augmentation_count(num_valid, dropout_fraction)
            if drop_count > 0:
                drop_idx = valid_idx[torch.randperm(num_valid, device=device)[:drop_count]]
                out[batch_idx, drop_idx, :] = 0.0

    valid = _valid_point_mask(out)
    xyz_jitter_std = float(cfg.get("xyz_jitter_std", 0.0))
    if xyz_jitter_std > 0.0 and bool(valid.any().item()):
        xyz_noise = torch.randn_like(out[..., :3]) * xyz_jitter_std
        out[..., :3] = torch.where(valid.unsqueeze(-1), out[..., :3] + xyz_noise, out[..., :3])
    feature_jitter_std = float(cfg.get("feature_jitter_std", 0.0))
    if feature_jitter_std > 0.0 and out.shape[-1] > 3 and bool(valid.any().item()):
        feat_noise = torch.randn_like(out[..., 3:]) * feature_jitter_std
        out[..., 3:] = torch.where(valid.unsqueeze(-1), out[..., 3:] + feat_noise, out[..., 3:])
    return out


def _pose_decode_from_config(training_cfg: dict) -> dict[str, Any] | None:
    cfg = training_cfg.get("pose_decode")
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return None
    source = str(cfg.get("source", "codebook_distance")).strip().lower()
    if source not in {
        "learned",
        "learned_classifier",
        "codebook_distance",
        "distance",
        "vq_distance",
        "blend",
        "learned_codebook_blend",
    }:
        raise ValueError(f"Unsupported training.pose_decode.source: {source!r}")
    mode = str(cfg.get("mode", "soft_expectation")).strip().lower()
    if mode not in {"argmax", "argmax_st", "sample_st", "soft_expectation"}:
        raise ValueError(f"Unsupported training.pose_decode.mode: {mode!r}")
    out = {
        "source": source,
        "mode": mode,
        "temperature": float(cfg.get("temperature", cfg.get("eval_temperature", 1.0))),
        "scale": float(cfg.get("scale", cfg.get("codebook_distance_scale", 1.0))),
        "use_for_loss": bool(cfg.get("use_for_loss", True)),
        "use_for_metric": bool(cfg.get("use_for_metric", True)),
    }
    if source in {"blend", "learned_codebook_blend"}:
        out["learned_weight"] = float(cfg.get("learned_weight", cfg.get("classifier_weight", 1.0)))
        out["codebook_weight"] = float(cfg.get("codebook_weight", cfg.get("distance_weight", 1.0)))
        if out["learned_weight"] < 0 or out["codebook_weight"] < 0:
            raise ValueError("training.pose_decode blend weights must be non-negative.")
        if out["learned_weight"] == 0 and out["codebook_weight"] == 0:
            raise ValueError("training.pose_decode blend needs at least one positive weight.")
    return out


def _decode_pose_from_config(model, outputs: dict[str, torch.Tensor], cfg: dict[str, Any]):
    source = str(cfg["source"])
    if source in {"learned", "learned_classifier"}:
        pred_logits = outputs.get("pred_logits")
        if pred_logits is None:
            raise RuntimeError("training.pose_decode source 'learned' requires model outputs['pred_logits'].")
    elif source in {"blend", "learned_codebook_blend"}:
        learned_logits = outputs.get("pred_logits")
        if learned_logits is None:
            raise RuntimeError("training.pose_decode source 'blend' requires model outputs['pred_logits'].")
        codebook_logits = part_codebook_distance_logits(
            vqvae=model.vqvae,
            pred_z_e=outputs["pred_z_e"],
            scale=float(cfg["scale"]),
        )
        pred_logits = float(cfg["learned_weight"]) * learned_logits.float() + float(cfg["codebook_weight"]) * codebook_logits
    else:
        pred_logits = part_codebook_distance_logits(
            vqvae=model.vqvae,
            pred_z_e=outputs["pred_z_e"],
            scale=float(cfg["scale"]),
        )
    return decode_part_token_logits(
        vqvae=model.vqvae,
        pred_logits=pred_logits,
        mode=cfg["mode"],
        temperature=float(cfg["temperature"]),
    )


def _select_pose_outputs(model, outputs: dict[str, torch.Tensor], training_cfg: dict) -> dict[str, Any]:
    cfg = _pose_decode_from_config(training_cfg)
    selected = {
        "cfg": cfg,
        "decoded": None,
        "loss_pose": outputs["pred_pose"],
        "metric_pose": outputs["pred_pose"],
        "metric_indices": outputs["pred_indices"],
    }
    if cfg is None:
        return selected
    decoded = _decode_pose_from_config(model, outputs, cfg)
    selected["decoded"] = decoded
    if bool(cfg["use_for_loss"]):
        selected["loss_pose"] = decoded.pose
    if bool(cfg["use_for_metric"]):
        selected["metric_pose"] = decoded.pose
        selected["metric_indices"] = decoded.indices
    return selected


def _zero_metric_tensors(keys: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.zeros((), device=device) for key in keys}


def _accumulate_metric(total: torch.Tensor, value: torch.Tensor | float | int, *, weight: int = 1) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return total + value.detach().float() * int(weight)
    return total + torch.tensor(float(value) * int(weight), device=total.device)


def _finalize_metric_tensors(
    totals: dict[str, torch.Tensor],
    denom: int,
    *,
    count_keys: set[str] | None = None,
) -> dict[str, float]:
    scale = float(max(1, int(denom)))
    count_keys = count_keys or set()
    return {
        key: float(value.item()) if key in count_keys else float((value / scale).item())
        for key, value in totals.items()
    }


def _set_frozen_modules_eval(model) -> None:
    if getattr(model, "freeze_point_transformer", False) and hasattr(model, "point_transformer"):
        model.point_transformer.eval()
    if getattr(model, "freeze_mmwave_tokenizer", False) and hasattr(model, "mmwave_tokenizer"):
        model.mmwave_tokenizer.eval()
    if getattr(model, "freeze_token_translator", False) and hasattr(model, "token_translator"):
        model.token_translator.eval()
    if getattr(model, "freeze_vqvae", False) and hasattr(model, "vqvae"):
        model.vqvae.eval()


def _encoder_trainable_count(model) -> int:
    params = []
    seen_modules: set[int] = set()
    for attr in ("point_transformer", "mmwave_tokenizer", "sensor_encoder"):
        module = getattr(model, attr, None)
        if module is None or id(module) in seen_modules:
            continue
        seen_modules.add(id(module))
        params.extend([p for p in module.parameters() if p.requires_grad])
    return safe_param_count(params)


def _stage_display(config: dict, default: str = "HPE Token") -> str:
    training_cfg = config.get("training", {})
    return str(training_cfg.get("stage_display_name", default))


def _metric_namespace(config: dict, default: str = "hpe_token") -> str:
    training_cfg = config.get("training", {})
    return str(training_cfg.get("metric_namespace", default)).strip().strip("/") or default


def _merge_point_view_outputs(
    model,
    outputs_list: list[dict[str, torch.Tensor]],
    *,
    view_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if not outputs_list:
        raise ValueError("Expected at least one point-view output.")

    merged = dict(outputs_list[0])
    stacked_z_e = torch.stack([item["pred_z_e"].float() for item in outputs_list], dim=0)
    if view_weights is None:
        weights = stacked_z_e.new_full((len(outputs_list),), 1.0 / float(len(outputs_list)))
    else:
        weights = view_weights.to(device=stacked_z_e.device, dtype=stacked_z_e.dtype).reshape(-1)
        if int(weights.numel()) != len(outputs_list):
            raise ValueError(f"Expected {len(outputs_list)} point-view weights, got {int(weights.numel())}.")
        if bool((weights < 0).any().item()) or float(weights.sum().item()) <= 0.0:
            raise ValueError("Point-view ensemble weights must be non-negative with positive sum.")
        weights = weights / weights.sum()
    pred_z_e = (stacked_z_e * weights.view(-1, 1, 1, 1)).sum(dim=0)
    pred_z_q, pred_indices = model.quantize_tokens(pred_z_e)
    pred_pose = model.vqvae.decode(pred_z_q)

    pred_logits_values = [item.get("pred_logits") for item in outputs_list]
    if all(value is not None for value in pred_logits_values):
        stacked_logits = torch.stack([value.float() for value in pred_logits_values], dim=0)
        merged["pred_logits"] = (stacked_logits * weights.view(-1, 1, 1, 1)).sum(dim=0)

    merged["pred_z_e"] = pred_z_e
    merged["pred_z_q"] = pred_z_q
    merged["pred_indices"] = pred_indices
    merged["pred_pose"] = pred_pose
    return merged


def _forward_batch(model, batch: dict, device: torch.device, runtime, *, point_augmentation: dict[str, Any] | None = None):
    target_poses = batch["target_poses"].to(device, non_blocking=True)
    cached_mmwave_tokens = batch.get("cached_mmwave_tokens")
    cached_mmwave_token_mask = batch.get("cached_mmwave_token_mask")
    cached_target_z_e = batch.get("cached_target_z_e")
    cached_target_indices = batch.get("cached_target_indices")
    with autocast_context(runtime):
        if cached_mmwave_tokens is not None and cached_target_z_e is not None and cached_target_indices is not None:
            kwargs = {
                "mmwave_points": None,
                "target_poses": target_poses,
                "mmwave_tokens": cached_mmwave_tokens.to(device, non_blocking=True),
                "target_z_e": cached_target_z_e.to(device, non_blocking=True),
                "target_indices": cached_target_indices.to(device, non_blocking=True).to(torch.long),
            }
            if cached_mmwave_token_mask is not None and hasattr(model, "mmwave_tokenizer"):
                kwargs["mmwave_token_mask"] = cached_mmwave_token_mask.to(device, non_blocking=True).to(torch.bool)
            outputs = model(**kwargs)
        else:
            mmwave_points = batch["mmwave_points"].to(device, non_blocking=True)
            if point_augmentation is not None:
                if mmwave_points.dim() == 4:
                    batch_size, num_views, num_points, channels = mmwave_points.shape
                    flat_points = mmwave_points.reshape(batch_size * num_views, num_points, channels)
                    flat_points = _apply_point_augmentation(flat_points, point_augmentation)
                    mmwave_points = flat_points.reshape(batch_size, num_views, num_points, channels)
                else:
                    mmwave_points = _apply_point_augmentation(mmwave_points, point_augmentation)
            if mmwave_points.dim() == 4:
                view_outputs = [
                    model(mmwave_points[:, view_idx, :, :], target_poses)
                    for view_idx in range(int(mmwave_points.shape[1]))
                ]
                view_weights = batch.get("point_ensemble_weights")
                if view_weights is not None:
                    view_weights = view_weights.to(device, non_blocking=True)
                    if view_weights.dim() > 1:
                        view_weights = view_weights[0]
                outputs = _merge_point_view_outputs(model, view_outputs, view_weights=view_weights)
            else:
                outputs = model(mmwave_points, target_poses)
    return target_poses, outputs


def train_hpe_token_epoch(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: TokenTranslatorLoss,
    soft_token_criterion: SoftMPJPETokenCELoss | None,
    soft_token_weight: float,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    config: dict,
    runtime,
    grad_scaler,
) -> dict[str, float]:
    training_cfg = config.get("training", {})
    pose_decode_cfg = _pose_decode_from_config(training_cfg)
    ablation_loss_cfg = _ablation_loss_config_from_training_config(training_cfg)
    point_augmentation_cfg = _point_augmentation_from_config(training_cfg)
    grad_accum_steps = max(1, int(training_cfg.get("grad_accum_steps", 1)))
    max_batches = training_cfg.get("max_train_batches")
    max_batches = int(max_batches) if max_batches else None
    log_interval = int(training_cfg.get("log_interval", 10))
    grad_norm = float(training_cfg.get("grad_norm", 0.0))

    model.train()
    _set_frozen_modules_eval(model)

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
            *ABLATION_LOSS_METRIC_KEYS,
        ],
        device,
    )
    total_mpjpe = torch.zeros((), device=device)
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
    num_batches = 0
    sample_count = 0
    optimizer_steps = 0
    data_time_sum = 0.0
    step_time_sum = 0.0
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    display_name = _stage_display(config)
    metric_namespace = _metric_namespace(config)
    pbar = tqdm(dataloader, desc=f"Train {display_name} Epoch {epoch + 1}")
    optimizer.zero_grad(set_to_none=True)
    loop_end_t = time.perf_counter()
    for batch_idx, batch in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break
        data_time_sum += time.perf_counter() - loop_end_t
        step_t0 = time.perf_counter()

        mark_compiled_step(runtime)
        target_poses, outputs = _forward_batch(
            model,
            batch,
            device,
            runtime,
            point_augmentation=point_augmentation_cfg,
        )
        batch_size = int(target_poses.shape[0])
        sample_count += batch_size
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
        loss_dict = _apply_optional_ablation_losses(
            loss_dict,
            outputs,
            model=model,
            layout=model.vqvae.layout,
            ablation_cfg=ablation_loss_cfg,
            target_poses=target_poses,
        )
        loss = loss_dict["total_loss"]
        loss_to_backprop = loss / float(grad_accum_steps)
        if grad_scaler.is_enabled():
            grad_scaler.scale(loss_to_backprop).backward()
        else:
            loss_to_backprop.backward()

        reached_accum = (batch_idx + 1) % grad_accum_steps == 0
        is_last = ((batch_idx + 1) >= len(dataloader)) or (
            max_batches is not None and (batch_idx + 1) >= int(max_batches)
        )
        if reached_accum or is_last:
            if grad_norm > 0:
                if grad_scaler.is_enabled():
                    grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
            if grad_scaler.is_enabled():
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        with torch.no_grad():
            mpjpe = compute_mpjpe(pose_selection["metric_pose"], target_poses)
            token_acc = (pose_selection["metric_indices"] == outputs["target_indices"]).float().mean()
            if pose_decode_metric_totals is not None and pose_selection["decoded"] is not None:
                decoded = pose_selection["decoded"]
                pose_decode_metric_totals["hard_mpjpe"] = _accumulate_metric(
                    pose_decode_metric_totals["hard_mpjpe"],
                    compute_mpjpe(outputs["pred_pose"], target_poses),
                    weight=batch_size,
                )
                pose_decode_metric_totals["pose_decode_mpjpe"] = _accumulate_metric(
                    pose_decode_metric_totals["pose_decode_mpjpe"],
                    compute_mpjpe(decoded.pose, target_poses),
                    weight=batch_size,
                )
                pose_decode_metric_totals["pose_decode_token_acc"] = _accumulate_metric(
                    pose_decode_metric_totals["pose_decode_token_acc"],
                    (decoded.indices == outputs["target_indices"]).float().mean(),
                    weight=batch_size,
                )
                pose_decode_metric_totals["pose_decode_entropy"] = _accumulate_metric(
                    pose_decode_metric_totals["pose_decode_entropy"],
                    decoded.entropy.float().mean(),
                    weight=batch_size,
                )
        for key in totals:
            if key in loss_dict:
                totals[key] = _accumulate_metric(
                    totals[key],
                    loss_dict[key],
                    weight=1 if key in _COUNT_METRIC_KEYS else batch_size,
                )
        total_mpjpe = _accumulate_metric(total_mpjpe, mpjpe, weight=batch_size)
        total_token_acc = _accumulate_metric(total_token_acc, token_acc, weight=batch_size)
        num_batches += 1

        if batch_idx % max(1, log_interval) == 0:
            loss_value = float(loss.detach().item())
            mpjpe_value = float(mpjpe.detach().item())
            token_acc_value = float(token_acc.detach().item())
            pbar.set_postfix(loss=f"{loss_value:.4f}", mpjpe=f"{mpjpe_value * 1000:.2f}mm")
            step = epoch * max(1, len(dataloader)) + batch_idx
            writer.add_scalar(f"{metric_namespace}/train/loss", loss_value, step)
            writer.add_scalar(f"{metric_namespace}/train/mpjpe_mm", mpjpe_value * 1000.0, step)
            writer.add_scalar(f"{metric_namespace}/train/token_acc", token_acc_value, step)
        step_time_sum += time.perf_counter() - step_t0
        loop_end_t = time.perf_counter()

    if num_batches == 0:
        raise RuntimeError(f"No training batches were processed for {display_name}.")
    metrics = _finalize_metric_tensors(totals, sample_count, count_keys=_COUNT_METRIC_KEYS)
    metrics["mpjpe"] = float((total_mpjpe / float(max(1, sample_count))).item())
    metrics["token_acc"] = float((total_token_acc / float(max(1, sample_count))).item())
    if pose_decode_metric_totals is not None:
        for key, value in pose_decode_metric_totals.items():
            metrics[key] = float((value / float(max(1, sample_count))).item())
        metrics["hard_mpjpe_mm"] = metrics["hard_mpjpe"] * 1000.0
        metrics["pose_decode_mpjpe_mm"] = metrics["pose_decode_mpjpe"] * 1000.0
        metrics["pose_decode"] = pose_decode_cfg
    metrics["samples_per_sec"] = float(sample_count / max(time.perf_counter() - t0, 1e-6))
    metrics["data_time_sec"] = float(data_time_sum)
    metrics["step_time_sec"] = float(step_time_sum)
    metrics["optimizer_steps"] = int(optimizer_steps)
    metrics["grad_accum_steps"] = int(grad_accum_steps)
    metrics["max_memory_mb"] = float(torch.cuda.max_memory_allocated(device=device) / (1024.0**2)) if device.type == "cuda" else 0.0
    return metrics


@torch.no_grad()
def validate_hpe_token(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: TokenTranslatorLoss,
    soft_token_criterion: SoftMPJPETokenCELoss | None,
    soft_token_weight: float,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    config: dict,
    runtime,
) -> dict[str, float]:
    training_cfg = config.get("training", {})
    max_batches = training_cfg.get("max_val_batches")
    max_batches = int(max_batches) if max_batches else None
    pose_decode_cfg = _pose_decode_from_config(training_cfg)
    ablation_loss_cfg = _ablation_loss_config_from_training_config(training_cfg)
    display_name = _stage_display(config)
    metric_namespace = _metric_namespace(config)
    model.eval()
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
            *ABLATION_LOSS_METRIC_KEYS,
        ],
        device,
    )
    total_mpjpe = torch.zeros((), device=device)
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
    num_batches = 0
    sample_count = 0
    data_time_sum = 0.0
    step_time_sum = 0.0
    t0 = time.perf_counter()
    loop_end_t = time.perf_counter()
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Validate {display_name}")):
        if max_batches is not None and batch_idx >= max_batches:
            break
        data_time_sum += time.perf_counter() - loop_end_t
        step_t0 = time.perf_counter()
        mark_compiled_step(runtime)
        target_poses, outputs = _forward_batch(model, batch, device, runtime)
        batch_size = int(target_poses.shape[0])
        sample_count += batch_size
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
        loss_dict = _apply_optional_ablation_losses(
            loss_dict,
            outputs,
            model=model,
            layout=model.vqvae.layout,
            ablation_cfg=ablation_loss_cfg,
            target_poses=target_poses,
        )
        mpjpe = compute_mpjpe(pose_selection["metric_pose"], target_poses)
        token_acc = (pose_selection["metric_indices"] == outputs["target_indices"]).float().mean()
        if pose_decode_metric_totals is not None and pose_selection["decoded"] is not None:
            decoded = pose_selection["decoded"]
            pose_decode_metric_totals["hard_mpjpe"] = _accumulate_metric(
                pose_decode_metric_totals["hard_mpjpe"],
                compute_mpjpe(outputs["pred_pose"], target_poses),
                weight=batch_size,
            )
            pose_decode_metric_totals["pose_decode_mpjpe"] = _accumulate_metric(
                pose_decode_metric_totals["pose_decode_mpjpe"],
                compute_mpjpe(decoded.pose, target_poses),
                weight=batch_size,
            )
            pose_decode_metric_totals["pose_decode_token_acc"] = _accumulate_metric(
                pose_decode_metric_totals["pose_decode_token_acc"],
                (decoded.indices == outputs["target_indices"]).float().mean(),
                weight=batch_size,
            )
            pose_decode_metric_totals["pose_decode_entropy"] = _accumulate_metric(
                pose_decode_metric_totals["pose_decode_entropy"],
                decoded.entropy.float().mean(),
                weight=batch_size,
            )
        for key in totals:
            if key in loss_dict:
                totals[key] = _accumulate_metric(
                    totals[key],
                    loss_dict[key],
                    weight=1 if key in _COUNT_METRIC_KEYS else batch_size,
                )
        total_mpjpe = _accumulate_metric(total_mpjpe, mpjpe, weight=batch_size)
        total_token_acc = _accumulate_metric(total_token_acc, token_acc, weight=batch_size)
        num_batches += 1
        step_time_sum += time.perf_counter() - step_t0
        loop_end_t = time.perf_counter()

    if num_batches == 0:
        raise RuntimeError(f"No validation batches were processed for {display_name}.")
    metrics = _finalize_metric_tensors(totals, sample_count, count_keys=_COUNT_METRIC_KEYS)
    metrics["mpjpe"] = float((total_mpjpe / float(max(1, sample_count))).item())
    metrics["token_acc"] = float((total_token_acc / float(max(1, sample_count))).item())
    if pose_decode_metric_totals is not None:
        for key, value in pose_decode_metric_totals.items():
            metrics[key] = float((value / float(max(1, sample_count))).item())
        metrics["hard_mpjpe_mm"] = metrics["hard_mpjpe"] * 1000.0
        metrics["pose_decode_mpjpe_mm"] = metrics["pose_decode_mpjpe"] * 1000.0
        metrics["pose_decode"] = pose_decode_cfg
    metrics["samples_per_sec"] = float(sample_count / max(time.perf_counter() - t0, 1e-6))
    metrics["data_time_sec"] = float(data_time_sum)
    metrics["step_time_sec"] = float(step_time_sum)
    writer.add_scalar(f"{metric_namespace}/val/loss", metrics["total_loss"], epoch)
    writer.add_scalar(f"{metric_namespace}/val/mpjpe_mm", metrics["mpjpe"] * 1000.0, epoch)
    writer.add_scalar(f"{metric_namespace}/val/token_acc", metrics["token_acc"], epoch)
    return metrics
