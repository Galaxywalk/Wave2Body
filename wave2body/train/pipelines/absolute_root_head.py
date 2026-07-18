# -*- coding: utf-8 -*-
"""Absolute-root HPE-token training."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    from ..checkpoint import load_checkpoint, load_state_dict_shape_safe, module_state_dict, save_checkpoint
    from ..common import load_config, safe_param_count, set_global_seed, write_run_manifest
    from ..radar_pose_data import build_radar_pose_dataloaders
    from ...evaluate.m4human_action_groups import (
        M4HumanActionGroupAccumulator,
        M4HumanActionGroupVectorAccumulator,
        m4human_action_ids_from_indicator,
    )
    from ...models.hpe_token.absolute_root import AbsoluteRootHeadEstimatorModel
    from ...models.token_translation import TokenTranslatorLoss
    from ...utils.metrics import compute_per_sample_p_mpjpe
    from ...utils.runtime_utils import autocast_context, build_grad_scaler, configure_runtime, mark_compiled_step, maybe_compile
    from ._paths import resolve_run_root, write_effective_config
    from .hpe_token_training import _encoder_trainable_count, _token_loss_from_config
except ImportError:  # Support scripts that put src/ directly on sys.path.
    from wave2body.train.checkpoint import load_checkpoint, load_state_dict_shape_safe, module_state_dict, save_checkpoint
    from wave2body.train.common import load_config, safe_param_count, set_global_seed, write_run_manifest
    from wave2body.train.radar_pose_data import build_radar_pose_dataloaders
    from wave2body.evaluate.m4human_action_groups import (
        M4HumanActionGroupAccumulator,
        M4HumanActionGroupVectorAccumulator,
        m4human_action_ids_from_indicator,
    )
    from wave2body.models.hpe_token.absolute_root import AbsoluteRootHeadEstimatorModel
    from wave2body.models.token_translation import TokenTranslatorLoss
    from wave2body.utils.metrics import compute_per_sample_p_mpjpe
    from wave2body.utils.runtime_utils import autocast_context, build_grad_scaler, configure_runtime, mark_compiled_step, maybe_compile
    from wave2body.train.pipelines._paths import resolve_run_root, write_effective_config
    from wave2body.train.pipelines.hpe_token_training import _encoder_trainable_count, _token_loss_from_config


EXPERIMENT_NAME = "absolute_root_head"
ABSOLUTE_METRIC_KEY = "absolute_mpjpe"
STAGE_NAME = "absolute_root_head"


def _zero_metric_tensors(keys: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.zeros((), device=device) for key in keys}


def _accumulate(total: torch.Tensor, value: torch.Tensor | float, *, weight: int) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return total + value.detach().float() * int(weight)
    return total + torch.tensor(float(value) * int(weight), device=total.device)


def _per_sample_mpjpe(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.norm(pred.float() - target.float(), dim=-1).mean(dim=-1)


def _root_aligned(pred: torch.Tensor, root_index: int) -> torch.Tensor:
    return pred - pred[:, root_index : root_index + 1]


def _batch_metric_values(
    *,
    pred_abs: torch.Tensor,
    target_abs: torch.Tensor,
    root_index: int,
    compute_p_mpjpe: bool,
    pred_relative: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    pred_root = pred_abs[:, root_index]
    target_root = target_abs[:, root_index]
    values = {
        "absolute_mpjpe": _per_sample_mpjpe(pred_abs, target_abs),
        "root_error": torch.norm(pred_root.float() - target_root.float(), dim=-1),
        "root_aligned_mpjpe": _per_sample_mpjpe(_root_aligned(pred_abs, root_index), _root_aligned(target_abs, root_index)),
    }
    if compute_p_mpjpe:
        p_values = compute_per_sample_p_mpjpe(pred_abs, target_abs)
        values["p_mpjpe"] = torch.as_tensor(p_values, device=pred_abs.device, dtype=torch.float32)
    if pred_relative is not None:
        oracle_root_pose = pred_relative.float() + target_abs[:, root_index : root_index + 1].float()
        values["oracle_root_mpjpe"] = _per_sample_mpjpe(oracle_root_pose, target_abs)
    return values


def _loss_from_batch(
    *,
    criterion: TokenTranslatorLoss,
    outputs: dict[str, torch.Tensor],
    target_relative: torch.Tensor,
    target_absolute: torch.Tensor,
    root_index: int,
    training_cfg: dict,
) -> dict[str, torch.Tensor]:
    token_loss = criterion(
        pred_z_e=outputs["pred_z_e"],
        target_z_e=outputs["target_z_e"],
        pred_logits=outputs["pred_logits"],
        target_indices=outputs["target_indices"],
        pred_pose=outputs["pred_pose_relative"],
        target_pose=target_relative,
    )
    absolute_pose_loss = F.mse_loss(outputs["pred_pose_absolute"].float(), target_absolute.float())
    root_loss = F.mse_loss(outputs["pred_root"].float(), target_absolute[:, root_index].float())
    absolute_weight = float(training_cfg.get("absolute_pose_loss_weight", 1.0))
    root_weight = float(training_cfg.get("root_loss_weight", 0.5))

    merged = dict(token_loss)
    merged["relative_total_loss"] = token_loss["total_loss"]
    merged["absolute_pose_loss"] = absolute_pose_loss
    merged["root_loss"] = root_loss
    merged["total_loss"] = token_loss["total_loss"] + absolute_weight * absolute_pose_loss + root_weight * root_loss
    return merged


def _use_root_only_loss(training_cfg: dict) -> bool:
    """Skip constant frozen-token losses when only the root head is trainable."""

    default_enabled = bool(training_cfg.get("freeze_token_translator", False))
    return bool(training_cfg.get("root_only_loss", default_enabled))


def _root_only_loss_from_batch(
    *,
    outputs: dict[str, torch.Tensor],
    target_absolute: torch.Tensor,
    root_index: int,
    training_cfg: dict,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    absolute_pose_loss = F.mse_loss(outputs["pred_pose_absolute"].float(), target_absolute.float())
    root_loss = F.mse_loss(outputs["pred_root"].float(), target_absolute[:, root_index].float())
    absolute_weight = float(training_cfg.get("absolute_pose_loss_weight", 1.0))
    root_weight = float(training_cfg.get("root_loss_weight", 0.5))
    zero = torch.zeros((), device=device, dtype=absolute_pose_loss.dtype)
    total = absolute_weight * absolute_pose_loss + root_weight * root_loss
    return {
        "total_loss": total,
        "relative_total_loss": zero,
        "latent_loss": zero,
        "token_loss": zero,
        "pose_loss": zero,
        "absolute_pose_loss": absolute_pose_loss,
        "root_loss": root_loss,
    }


def _build_absolute_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    training_cfg = config.get("training", {})
    base_lr = float(training_cfg.get("lr", 5e-5))
    weight_decay = float(training_cfg.get("weight_decay", 0.01))
    translator_lr_mult = float(training_cfg.get("translator_lr_mult", 1.0))
    root_lr_mult = float(training_cfg.get("root_head_lr_mult", training_cfg.get("root_lr_mult", 1.0)))
    encoder_lr_mult = float(training_cfg.get("sensor_encoder_lr_mult", 0.2))
    vqvae_lr_mult = float(training_cfg.get("vqvae_lr_mult", 0.05))

    groups: list[dict[str, Any]] = []
    seen: set[int] = set()

    def collect(module: torch.nn.Module | None) -> list[torch.nn.Parameter]:
        params: list[torch.nn.Parameter] = []
        if module is None:
            return params
        for param in module.parameters():
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            params.append(param)
        return params

    translator_params = collect(getattr(model, "token_translator", None))
    root_params = collect(getattr(model, "root_head", None))
    encoder_params = collect(getattr(model, "mmwave_tokenizer", None))
    vqvae_params = collect(getattr(model, "vqvae", None))
    if translator_params:
        groups.append({"params": translator_params, "lr": base_lr * translator_lr_mult, "name": "translator"})
    if root_params:
        groups.append({"params": root_params, "lr": base_lr * root_lr_mult, "name": "root_head"})
    if encoder_params:
        groups.append({"params": encoder_params, "lr": base_lr * encoder_lr_mult, "name": "sensor_encoder"})
    if vqvae_params:
        groups.append({"params": vqvae_params, "lr": base_lr * vqvae_lr_mult, "name": "vqvae"})
    if not groups:
        raise RuntimeError("No trainable parameters found for absolute HPE-token optimizer.")
    print(
        "[AbsoluteRootHead] Optimizer param groups | "
        f"translator={safe_param_count(translator_params):,}, root={safe_param_count(root_params):,}, "
        f"sensor_encoder={safe_param_count(encoder_params):,}, vqvae={safe_param_count(vqvae_params):,}"
    )
    return torch.optim.AdamW(groups, lr=base_lr, weight_decay=weight_decay)


def _set_frozen_modules_eval(model: AbsoluteRootHeadEstimatorModel) -> None:
    if getattr(model, "freeze_mmwave_tokenizer", False):
        model.mmwave_tokenizer.eval()
    if getattr(model, "freeze_token_translator", False):
        model.token_translator.eval()
    if getattr(model, "freeze_vqvae", False):
        model.vqvae.eval()


def _load_absolute_component_states(model: AbsoluteRootHeadEstimatorModel, ckpt: dict[str, Any], *, prefix: str) -> None:
    tag = f"{prefix} " if prefix else ""
    targets = (
        ("mmwave_tokenizer_state_dict", "mmwave_tokenizer"),
        ("vqvae_state_dict", "vqvae"),
        ("token_translator_state_dict", "token_translator"),
        ("root_head_state_dict", "root_head"),
    )
    for state_key, attr in targets:
        if state_key not in ckpt:
            continue
        module = getattr(model, attr, None)
        if module is None:
            continue
        load_target = getattr(module, "_orig_mod", module)
        load_state_dict_shape_safe(
            load_target,
            ckpt[state_key],
            module_name=f"{tag}{state_key}",
            strict=False,
        )


def train_absolute_epoch(
    *,
    model: AbsoluteRootHeadEstimatorModel,
    dataloader: DataLoader,
    criterion: TokenTranslatorLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    config: dict,
    runtime,
    grad_scaler,
) -> dict[str, float]:
    training_cfg = config.get("training", {})
    grad_accum_steps = max(1, int(training_cfg.get("grad_accum_steps", 1)))
    max_batches = training_cfg.get("max_train_batches")
    max_batches = int(max_batches) if max_batches else None
    log_interval = int(training_cfg.get("log_interval", 20))
    grad_norm = float(training_cfg.get("grad_norm", 0.0))
    root_index = int(model.root_index)
    root_only_loss = _use_root_only_loss(training_cfg)

    model.train()
    _set_frozen_modules_eval(model)
    totals = _zero_metric_tensors(
        ["total_loss", "relative_total_loss", "latent_loss", "token_loss", "pose_loss", "absolute_pose_loss", "root_loss"],
        device,
    )
    metric_totals = _zero_metric_tensors(["absolute_mpjpe", "root_error", "root_aligned_mpjpe"], device)
    token_acc_total = torch.zeros((), device=device)
    sample_count = 0
    num_batches = 0
    optimizer_steps = 0
    data_time_sum = 0.0
    step_time_sum = 0.0
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    pbar = tqdm(dataloader, desc=f"Train absolute_root_head epoch {epoch + 1}")
    optimizer.zero_grad(set_to_none=True)
    loop_end_t = time.perf_counter()
    for batch_idx, batch in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break
        data_time_sum += time.perf_counter() - loop_end_t
        step_t0 = time.perf_counter()
        mark_compiled_step(runtime)

        mmwave_points = batch["mmwave_points"].to(device, non_blocking=True)
        target_relative = None if root_only_loss else batch["target_poses"].to(device, non_blocking=True)
        target_absolute = batch["target_poses_absolute"].to(device, non_blocking=True)
        batch_size = int(target_absolute.shape[0])
        with autocast_context(runtime):
            outputs = model(mmwave_points, None if root_only_loss else target_relative)
            if root_only_loss:
                loss_dict = _root_only_loss_from_batch(
                    outputs=outputs,
                    target_absolute=target_absolute,
                    root_index=root_index,
                    training_cfg=training_cfg,
                    device=device,
                )
            else:
                loss_dict = _loss_from_batch(
                    criterion=criterion,
                    outputs=outputs,
                    target_relative=target_relative,
                    target_absolute=target_absolute,
                    root_index=root_index,
                    training_cfg=training_cfg,
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
            metric_values = _batch_metric_values(
                pred_abs=outputs["pred_pose_absolute"],
                target_abs=target_absolute,
                root_index=root_index,
                compute_p_mpjpe=False,
            )
            if root_only_loss:
                token_acc = torch.zeros((), device=device)
            else:
                token_acc = (outputs["pred_indices"] == outputs["target_indices"]).float().mean()

        sample_count += batch_size
        num_batches += 1
        for key in totals:
            if key in loss_dict:
                totals[key] = _accumulate(totals[key], loss_dict[key], weight=batch_size)
        for key, value in metric_values.items():
            metric_totals[key] = _accumulate(metric_totals[key], value.mean(), weight=batch_size)
        token_acc_total = _accumulate(token_acc_total, token_acc, weight=batch_size)

        if batch_idx % max(1, log_interval) == 0:
            loss_value = float(loss.detach().item())
            abs_mpjpe = float(metric_values["absolute_mpjpe"].mean().detach().item())
            root_error = float(metric_values["root_error"].mean().detach().item())
            pbar.set_postfix(loss=f"{loss_value:.4f}", abs=f"{abs_mpjpe * 1000:.1f}mm", root=f"{root_error * 1000:.1f}mm")
            step = epoch * max(1, len(dataloader)) + batch_idx
            writer.add_scalar("absolute_root_head/train/loss", loss_value, step)
            writer.add_scalar("absolute_root_head/train/absolute_mpjpe_mm", abs_mpjpe * 1000.0, step)
            writer.add_scalar("absolute_root_head/train/root_error_mm", root_error * 1000.0, step)
            writer.add_scalar("absolute_root_head/train/token_acc", float(token_acc.detach().item()), step)
        step_time_sum += time.perf_counter() - step_t0
        loop_end_t = time.perf_counter()

    if num_batches == 0:
        raise RuntimeError("No training batches were processed for absolute_root_head.")
    denom = float(max(1, sample_count))
    metrics = {key: float((value / denom).item()) for key, value in totals.items()}
    metrics.update({key: float((value / denom).item()) for key, value in metric_totals.items()})
    metrics["token_acc"] = float((token_acc_total / denom).item())
    metrics["samples_per_sec"] = float(sample_count / max(time.perf_counter() - t0, 1e-6))
    metrics["data_time_sec"] = float(data_time_sum)
    metrics["step_time_sec"] = float(step_time_sum)
    metrics["optimizer_steps"] = int(optimizer_steps)
    metrics["grad_accum_steps"] = int(grad_accum_steps)
    metrics["max_memory_mb"] = float(torch.cuda.max_memory_allocated(device=device) / (1024.0**2)) if device.type == "cuda" else 0.0
    return metrics


@torch.no_grad()
def evaluate_absolute_root_head(
    *,
    model: AbsoluteRootHeadEstimatorModel,
    dataloader: DataLoader,
    criterion: TokenTranslatorLoss,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter | None,
    config: dict,
    runtime,
    split_name: str,
    sample_collector: Any | None = None,
) -> dict[str, Any]:
    eval_cfg = config.get("evaluation", {})
    training_cfg = config.get("training", {})
    max_key = "max_test_batches" if split_name == "test" else "max_val_batches"
    max_batches = training_cfg.get(max_key)
    max_batches = int(max_batches) if max_batches else None
    compute_p = bool(eval_cfg.get("compute_p_mpjpe", True))
    use_action_groups = bool(eval_cfg.get("m4human_action_groups", True))
    include_per_action = bool(eval_cfg.get("include_per_action", False))
    root_index = int(model.root_index)
    root_only_loss = _use_root_only_loss(training_cfg)

    model.eval()
    totals = _zero_metric_tensors(
        ["total_loss", "relative_total_loss", "latent_loss", "token_loss", "pose_loss", "absolute_pose_loss", "root_loss"],
        device,
    )
    metric_names = ["absolute_mpjpe", "root_error", "root_aligned_mpjpe", "oracle_root_mpjpe"]
    if compute_p:
        metric_names.append("p_mpjpe")
    metric_totals = _zero_metric_tensors(metric_names, device)
    token_acc_total = torch.zeros((), device=device)
    action_acc = (
        M4HumanActionGroupAccumulator(metric_names=metric_names, device=device)
        if use_action_groups
        else None
    )
    action_joint_acc = (
        M4HumanActionGroupVectorAccumulator(["absolute_mpjpe", "root_aligned_mpjpe"], device=device)
        if use_action_groups
        else None
    )
    joint_error_sum_m: torch.Tensor | None = None
    root_aligned_joint_error_sum_m: torch.Tensor | None = None
    sample_count = 0
    num_batches = 0
    data_time_sum = 0.0
    step_time_sum = 0.0
    t0 = time.perf_counter()
    loop_end_t = time.perf_counter()

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluate absolute_root_head {split_name}")):
        if max_batches is not None and batch_idx >= max_batches:
            break
        data_time_sum += time.perf_counter() - loop_end_t
        step_t0 = time.perf_counter()
        mark_compiled_step(runtime)
        mmwave_points = batch["mmwave_points"].to(device, non_blocking=True)
        target_relative = None if root_only_loss else batch["target_poses"].to(device, non_blocking=True)
        target_absolute = batch["target_poses_absolute"].to(device, non_blocking=True)
        batch_size = int(target_absolute.shape[0])
        with autocast_context(runtime):
            outputs = model(mmwave_points, None if root_only_loss else target_relative)
            if root_only_loss:
                loss_dict = _root_only_loss_from_batch(
                    outputs=outputs,
                    target_absolute=target_absolute,
                    root_index=root_index,
                    training_cfg=training_cfg,
                    device=device,
                )
            else:
                loss_dict = _loss_from_batch(
                    criterion=criterion,
                    outputs=outputs,
                    target_relative=target_relative,
                    target_absolute=target_absolute,
                    root_index=root_index,
                    training_cfg=training_cfg,
                )
        metric_values = _batch_metric_values(
            pred_abs=outputs["pred_pose_absolute"],
            target_abs=target_absolute,
            root_index=root_index,
            compute_p_mpjpe=compute_p,
            pred_relative=outputs["pred_pose_relative"],
        )
        per_joint_errors_m = torch.linalg.norm(outputs["pred_pose_absolute"].float() - target_absolute.float(), dim=-1)
        if joint_error_sum_m is None:
            joint_error_sum_m = torch.zeros(int(per_joint_errors_m.shape[1]), device=device, dtype=torch.float64)
        joint_error_sum_m += per_joint_errors_m.double().sum(dim=0)
        root_aligned_joint_errors_m = torch.linalg.norm(
            _root_aligned(outputs["pred_pose_absolute"], root_index).float()
            - _root_aligned(target_absolute, root_index).float(),
            dim=-1,
        )
        if root_aligned_joint_error_sum_m is None:
            root_aligned_joint_error_sum_m = torch.zeros(
                int(root_aligned_joint_errors_m.shape[1]),
                device=device,
                dtype=torch.float64,
            )
        root_aligned_joint_error_sum_m += root_aligned_joint_errors_m.double().sum(dim=0)
        if root_only_loss:
            token_acc = torch.zeros((), device=device)
        else:
            token_acc = (outputs["pred_indices"] == outputs["target_indices"]).float().mean()
        if sample_collector is not None:
            sample_collector.update_from_batch(
                batch=batch,
                pred_pose=outputs["pred_pose_absolute"].detach(),
                gt_pose=target_absolute.detach(),
                mmwave_points=mmwave_points.detach(),
                absolute_mpjpe=metric_values["absolute_mpjpe"].detach(),
                root_error=metric_values["root_error"].detach(),
                root_aligned_mpjpe=metric_values["root_aligned_mpjpe"].detach(),
            )
        sample_count += batch_size
        num_batches += 1

        for key in totals:
            if key in loss_dict:
                totals[key] = _accumulate(totals[key], loss_dict[key], weight=batch_size)
        for key, value in metric_values.items():
            metric_totals[key] = _accumulate(metric_totals[key], value.mean(), weight=batch_size)
        token_acc_total = _accumulate(token_acc_total, token_acc, weight=batch_size)

        if action_acc is not None and "indicator" in batch:
            action_ids = m4human_action_ids_from_indicator(batch["indicator"], device=device)
            for key, value in metric_values.items():
                action_acc.update(key, action_ids, value)
            if action_joint_acc is not None:
                action_joint_acc.update("absolute_mpjpe", action_ids, per_joint_errors_m)
                action_joint_acc.update("root_aligned_mpjpe", action_ids, root_aligned_joint_errors_m)
        step_time_sum += time.perf_counter() - step_t0
        loop_end_t = time.perf_counter()

    if num_batches == 0:
        raise RuntimeError(f"No {split_name} batches were processed for absolute_root_head.")
    denom = float(max(1, sample_count))
    metrics: dict[str, Any] = {key: float((value / denom).item()) for key, value in totals.items()}
    metrics.update({key: float((value / denom).item()) for key, value in metric_totals.items()})
    for key in metric_names:
        if key in metrics:
            metrics[f"{key}_mm"] = float(metrics[key] * 1000.0)
    metrics["mpjpe"] = metrics["absolute_mpjpe"]
    metrics["mpjpe_mm"] = metrics["absolute_mpjpe_mm"]
    if joint_error_sum_m is not None:
        metrics["mpjpe_by_joint_mm"] = [
            float(item) for item in (joint_error_sum_m / denom * 1000.0).detach().cpu().tolist()
        ]
        metrics["absolute_mpjpe_by_joint_mm"] = metrics["mpjpe_by_joint_mm"]
    if root_aligned_joint_error_sum_m is not None:
        metrics["root_aligned_mpjpe_by_joint_mm"] = [
            float(item) for item in (root_aligned_joint_error_sum_m / denom * 1000.0).detach().cpu().tolist()
        ]
    metrics["token_acc"] = float((token_acc_total / denom).item())
    metrics["samples_per_sec"] = float(sample_count / max(time.perf_counter() - t0, 1e-6))
    metrics["data_time_sec"] = float(data_time_sum)
    metrics["step_time_sec"] = float(step_time_sum)
    metrics["num_samples"] = int(sample_count)
    metrics["num_batches"] = int(num_batches)
    metrics["root_index"] = int(root_index)
    if action_acc is not None:
        metrics["m4human_action_groups"] = action_acc.summary(scale=1000.0, include_per_action=include_per_action)
    if action_joint_acc is not None:
        joint_summary = action_joint_acc.summary(scale=1000.0)
        if "absolute_mpjpe" in joint_summary:
            metrics["mpjpe_by_action_group_by_joint_mm"] = joint_summary["absolute_mpjpe"]["by_group"]
            metrics["absolute_mpjpe_by_action_group_by_joint_mm"] = joint_summary["absolute_mpjpe"]["by_group"]
        if "root_aligned_mpjpe" in joint_summary:
            metrics["root_aligned_mpjpe_by_action_group_by_joint_mm"] = joint_summary["root_aligned_mpjpe"]["by_group"]

    if writer is not None:
        writer.add_scalar(f"absolute_root_head/{split_name}/loss", metrics["total_loss"], epoch)
        writer.add_scalar(f"absolute_root_head/{split_name}/absolute_mpjpe_mm", metrics["absolute_mpjpe"] * 1000.0, epoch)
        writer.add_scalar(f"absolute_root_head/{split_name}/root_error_mm", metrics["root_error"] * 1000.0, epoch)
        writer.add_scalar(f"absolute_root_head/{split_name}/root_aligned_mpjpe_mm", metrics["root_aligned_mpjpe"] * 1000.0, epoch)
        writer.add_scalar(f"absolute_root_head/{split_name}/oracle_root_mpjpe_mm", metrics["oracle_root_mpjpe"] * 1000.0, epoch)
        if "p_mpjpe" in metrics:
            writer.add_scalar(f"absolute_root_head/{split_name}/p_mpjpe_mm", metrics["p_mpjpe"] * 1000.0, epoch)
        writer.add_scalar(f"absolute_root_head/{split_name}/token_acc", metrics["token_acc"], epoch)
    return metrics


def _write_summary(
    output_root: Path,
    *,
    config_path: str | Path | None,
    checkpoint_dir: Path,
    log_dir: Path,
    runtime,
    epochs: int,
    best_val_absolute_mpjpe: float,
    history: list[dict[str, Any]],
    final_val_metrics: dict[str, Any] | None,
    final_test_metrics: dict[str, Any] | None,
    completed: bool,
) -> dict[str, Any]:
    summary = {
        "stage": STAGE_NAME,
        "completed": bool(completed),
        "best_val_absolute_mpjpe_mm": best_val_absolute_mpjpe * 1000.0,
        "best_val_mpjpe_mm": best_val_absolute_mpjpe * 1000.0,
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
        "final_val": final_val_metrics,
        "final_test": final_test_metrics,
    }
    (output_root / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metrics_payload = {"val": final_val_metrics, "test": final_test_metrics, "summary": summary}
    (output_root / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return summary


def _dataset_eval_label(config: dict[str, Any]) -> str:
    name = str(config.get("dataset", {}).get("name", "dataset")).strip().lower()
    return name.replace("dataset", "") or name or "dataset"


def _eval_suffix(config: dict[str, Any], eval_split: str) -> str:
    split_id = config.get("m4human", {}).get("split")
    if split_id:
        return f"{eval_split}_{split_id}"
    return f"{eval_split}_{_dataset_eval_label(config)}"


def _select_eval_loader(data_bundle, eval_split: str):
    if eval_split == "train":
        return data_bundle.train_loader
    if eval_split == "val":
        return data_bundle.val_loader
    if eval_split == "test":
        if data_bundle.test_loader is None:
            raise RuntimeError("Test loader was not built for absolute-root evaluation.")
        return data_bundle.test_loader
    raise ValueError(f"eval_split must be train/val/test, got {eval_split!r}")


def run_absolute_root_head_evaluation(
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
    include_per_action: bool = False,
    sample_collector: Any | None = None,
) -> dict[str, Any]:
    if eval_split not in {"train", "val", "test"}:
        raise ValueError(f"eval_split must be train/val/test, got {eval_split!r}")
    training_cfg = config.setdefault("training", {})
    training_cfg["stage"] = STAGE_NAME
    training_cfg["stage_display_name"] = "Absolute Root Head"
    training_cfg["metric_namespace"] = STAGE_NAME
    training_cfg["freeze_mmwave_tokenizer"] = True
    training_cfg["freeze_token_translator"] = True
    training_cfg["freeze_vqvae"] = True
    if batch_size is not None:
        training_cfg["batch_size"] = int(batch_size)
    if num_workers is not None:
        workers = int(num_workers)
        training_cfg["num_workers"] = workers
        if workers <= 0:
            training_cfg["pin_memory"] = False
            training_cfg["persistent_workers"] = False
    if max_eval_batches is not None:
        training_cfg[f"max_{eval_split}_batches"] = int(max_eval_batches)
    config.setdefault("evaluation", {})["include_per_action"] = bool(include_per_action)

    device_obj = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    runtime = configure_runtime(training_cfg, device_obj, default_amp=True, log_prefix="[AbsoluteRootHead Eval Runtime]")
    model = AbsoluteRootHeadEstimatorModel(
        mmwave_tokenizer_config=config["mmwave_tokenizer"],
        sensor_token_config=config.get("sensor_tokens", {}),
        token_translator_config=config["token_translator"],
        vqvae_config=config["vqvae"],
        root_head_config=config.get("root_head", {}),
        device=str(device_obj),
        freeze_mmwave_tokenizer=True,
        freeze_token_translator=True,
        freeze_vqvae=True,
    )
    pretrained = config.get("pretrained", {})
    model.load_pretrained_weights(
        mmwave_tokenizer_path=pretrained.get("mmwave_tokenizer"),
        vqvae_path=pretrained.get("vqvae"),
        hpe_token_path=pretrained.get("hpe_token"),
        device=str(device_obj),
    )
    ckpt = load_checkpoint(checkpoint, map_location=device_obj)
    _load_absolute_component_states(model, ckpt, prefix="[AbsoluteRootHead Eval]")
    model = model.to(device_obj)

    data_bundle = build_radar_pose_dataloaders(
        config=config,
        checkpoint_dir=None,
        include_test=True,
        max_train_batches=max_eval_batches if eval_split == "train" else None,
        max_val_batches=max_eval_batches if eval_split == "val" else None,
        max_test_batches=max_eval_batches if eval_split == "test" else None,
    )
    loader = _select_eval_loader(data_bundle, eval_split)
    criterion = _token_loss_from_config(training_cfg)
    metrics = evaluate_absolute_root_head(
        model=model,
        dataloader=loader,
        criterion=criterion,
        device=device_obj,
        epoch=0,
        writer=None,
        config=config,
        runtime=runtime,
        split_name=eval_split,
        sample_collector=sample_collector,
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
        "metrics": metrics,
    }
    out_path = Path(output_path).expanduser() if output_path else (
        Path(checkpoint).expanduser().parent.parent / f"metrics_{_eval_suffix(config, eval_split)}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["output_path"] = str(out_path)
    print(
        "[AbsoluteRootHead Eval] "
        f"{_eval_suffix(config, eval_split)} | "
        f"abs={metrics['absolute_mpjpe'] * 1000.0:.2f}mm, "
        f"root={metrics['root_error'] * 1000.0:.2f}mm, "
        f"root_aligned={metrics['root_aligned_mpjpe'] * 1000.0:.2f}mm, "
        f"oracle_root={metrics['oracle_root_mpjpe'] * 1000.0:.2f}mm"
    )
    if "p_mpjpe" in metrics:
        print(f"[AbsoluteRootHead Eval] p_mpjpe={metrics['p_mpjpe'] * 1000.0:.2f}mm")
    print(f"[AbsoluteRootHead Eval] Wrote {out_path}")
    return payload


def run_absolute_root_head(
    config: dict,
    *,
    config_path: str | Path | None = None,
    resume: str | Path | None = None,
    work_dir: str | Path | None = None,
    device: str = "cuda",
    seed: int | None = None,
    reset_epoch: bool = False,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    max_test_batches: int | None = None,
) -> dict[str, Any]:
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
    runtime = configure_runtime(training_cfg, device_obj, default_amp=True, log_prefix="[AbsoluteRootHead Runtime]")

    model = AbsoluteRootHeadEstimatorModel(
        mmwave_tokenizer_config=config["mmwave_tokenizer"],
        sensor_token_config=config.get("sensor_tokens", {}),
        token_translator_config=config["token_translator"],
        vqvae_config=config["vqvae"],
        root_head_config=config.get("root_head", {}),
        device=str(device_obj),
        freeze_mmwave_tokenizer=bool(training_cfg.get("freeze_mmwave_tokenizer", False)),
        freeze_token_translator=bool(training_cfg.get("freeze_token_translator", False)),
        freeze_vqvae=bool(training_cfg.get("freeze_vqvae", True)),
    )
    pretrained = config.get("pretrained", {})
    hpe_token_checkpoint = pretrained.get("hpe_token")
    model.load_pretrained_weights(
        mmwave_tokenizer_path=pretrained.get("mmwave_tokenizer"),
        vqvae_path=pretrained.get("vqvae"),
        hpe_token_path=hpe_token_checkpoint,
        device=str(device_obj),
    )
    model = model.to(device_obj)

    data_bundle = build_radar_pose_dataloaders(
        config=config,
        checkpoint_dir=output_root,
        include_test=True,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        max_test_batches=max_test_batches,
    )
    write_run_manifest(
        output_root,
        config_path=effective_config_path,
        checkpoint_path=resume or hpe_token_checkpoint or pretrained.get("mmwave_tokenizer"),
        config=config,
        dataset_split=(
            f"{config.get('m4human', {}).get('split')}/"
            f"{config.get('m4human', {}).get('frame_mode', config.get('m4human', {}).get('temporal_mode', 'default'))}"
        ),
        coord_transform=config.get("coord_transform", "m4human_radar_absolute_z_shifted_root_head"),
        output_paths={"output_root": output_root, "checkpoint_dir": checkpoint_dir, "log_dir": log_dir},
        extra={
            "stage": STAGE_NAME,
            "split_manifest": data_bundle.split_manifest,
            "hpe_token_checkpoint": hpe_token_checkpoint,
            "mmwave_tokenizer_checkpoint": pretrained.get("mmwave_tokenizer"),
            "vqvae_checkpoint": pretrained.get("vqvae"),
            "sensor_token_dim": model.sensor_token_dim,
            "sensor_token_mode": model.sensor_token_mode,
            "temporal_offsets": config.get("m4human", {}).get("temporal_offsets"),
            "frame_mode": config.get("m4human", {}).get("frame_mode"),
        },
    )
    print(f"[AbsoluteRootHead] Train samples: {data_bundle.train_count}")
    print(f"[AbsoluteRootHead] Val samples: {data_bundle.val_count}")
    print(f"[AbsoluteRootHead] Test samples: {data_bundle.test_count}")
    print(
        "[AbsoluteRootHead] Model | "
        f"mmwave_tokenizer={config['mmwave_tokenizer'].get('method')} sensor_mode={model.sensor_token_mode} "
        f"sensor_dim={model.sensor_token_dim} root_index={model.root_index} "
        f"root_architecture={getattr(model.root_head, 'architecture', 'unknown')} "
        f"root_prior={getattr(model.root_head, 'prior_mode', 'unknown')} "
        f"root_only_loss={_use_root_only_loss(training_cfg)}"
    )

    criterion = _token_loss_from_config(training_cfg)
    optimizer = _build_absolute_optimizer(model, config)
    grad_scaler = build_grad_scaler(runtime)
    epochs = int(training_cfg.get("epochs", 25))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(training_cfg.get("min_lr", 1e-7)),
    )

    start_epoch = 0
    best_val_absolute_mpjpe = float("inf")
    if resume:
        ckpt = load_checkpoint(resume, map_location=device_obj)
        _load_absolute_component_states(model, ckpt, prefix="[AbsoluteRootHead]")
        if not reset_epoch:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if ckpt.get("amp_state_dict") and grad_scaler.is_enabled():
                grad_scaler.load_state_dict(ckpt["amp_state_dict"])
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            best_val_absolute_mpjpe = float(ckpt.get("best_val_absolute_mpjpe", ckpt.get("best_val_mpjpe", float("inf"))))
        else:
            print("[AbsoluteRootHead] Resetting epoch after resume.")

    model.token_translator = maybe_compile(
        model.token_translator,
        runtime,
        module_name="absolute_root_head_translator",
        log_prefix="[AbsoluteRootHead Runtime]",
    )
    model.root_head = maybe_compile(
        model.root_head,
        runtime,
        module_name="absolute_root_head_root_head",
        log_prefix="[AbsoluteRootHead Runtime]",
    )
    print(
        "[AbsoluteRootHead] Trainable parameters | "
        f"translator={safe_param_count([p for p in model.token_translator.parameters() if p.requires_grad]):,}, "
        f"root={safe_param_count([p for p in model.root_head.parameters() if p.requires_grad]):,}, "
        f"sensor_encoder={_encoder_trainable_count(model):,}, "
        f"vqvae={safe_param_count([p for p in model.vqvae.parameters() if p.requires_grad]):,}"
    )

    history: list[dict[str, Any]] = []
    final_val_metrics: dict[str, Any] | None = None
    final_test_metrics: dict[str, Any] | None = None
    writer = SummaryWriter(str(log_dir))
    writer.add_text("absolute_root_head/run_name", str(training_cfg.get("run_name", "")), 0)
    writer.add_text("absolute_root_head/hpe_token_checkpoint", str(hpe_token_checkpoint or ""), 0)
    try:
        for epoch in range(start_epoch, epochs):
            print(f"\n[AbsoluteRootHead] Epoch {epoch + 1}/{epochs} | lr={optimizer.param_groups[0]['lr']:.6e}")
            train_metrics = train_absolute_epoch(
                model=model,
                dataloader=data_bundle.train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device_obj,
                epoch=epoch,
                writer=writer,
                config=config,
                runtime=runtime,
                grad_scaler=grad_scaler,
            )
            val_metrics = evaluate_absolute_root_head(
                model=model,
                dataloader=data_bundle.val_loader,
                criterion=criterion,
                device=device_obj,
                epoch=epoch,
                writer=writer,
                config=config,
                runtime=runtime,
                split_name="val",
            )
            scheduler.step()
            history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
            print(
                "[AbsoluteRootHead] Val | "
                f"loss={val_metrics['total_loss']:.4f}, abs={val_metrics['absolute_mpjpe'] * 1000.0:.2f}mm, "
                f"root={val_metrics['root_error'] * 1000.0:.2f}mm, "
                f"root_aligned={val_metrics['root_aligned_mpjpe'] * 1000.0:.2f}mm"
            )

            checkpoint = {
                "epoch": epoch,
                "stage": STAGE_NAME,
                "mmwave_tokenizer_state_dict": module_state_dict(model.mmwave_tokenizer),
                "vqvae_state_dict": module_state_dict(model.vqvae),
                "token_translator_state_dict": module_state_dict(model.token_translator),
                "root_head_state_dict": module_state_dict(model.root_head),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "amp_state_dict": grad_scaler.state_dict() if grad_scaler.is_enabled() else None,
                "best_val_absolute_mpjpe": best_val_absolute_mpjpe,
                "best_val_mpjpe": best_val_absolute_mpjpe,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "config": config,
            }
            if float(val_metrics[ABSOLUTE_METRIC_KEY]) < best_val_absolute_mpjpe:
                best_val_absolute_mpjpe = float(val_metrics[ABSOLUTE_METRIC_KEY])
                checkpoint["best_val_absolute_mpjpe"] = best_val_absolute_mpjpe
                checkpoint["best_val_mpjpe"] = best_val_absolute_mpjpe
                save_checkpoint(checkpoint_dir / "best.pth", checkpoint)
                print(f"[AbsoluteRootHead] Saved best model (val_abs_mpjpe={best_val_absolute_mpjpe * 1000.0:.2f}mm)")
            save_checkpoint(checkpoint_dir / "latest.pth", checkpoint)
            _write_summary(
                output_root,
                config_path=effective_config_path,
                checkpoint_dir=checkpoint_dir,
                log_dir=log_dir,
                runtime=runtime,
                epochs=epochs,
                best_val_absolute_mpjpe=best_val_absolute_mpjpe,
                history=history,
                final_val_metrics=val_metrics,
                final_test_metrics=final_test_metrics,
                completed=False,
            )

        best_path = checkpoint_dir / "best.pth"
        if best_path.exists():
            ckpt = load_checkpoint(best_path, map_location=device_obj)
            _load_absolute_component_states(model, ckpt, prefix="[AbsoluteRootHead best]")
        final_val_metrics = evaluate_absolute_root_head(
            model=model,
            dataloader=data_bundle.val_loader,
            criterion=criterion,
            device=device_obj,
            epoch=epochs,
            writer=writer,
            config=config,
            runtime=runtime,
            split_name="val",
        )
        if data_bundle.test_loader is not None:
            final_test_metrics = evaluate_absolute_root_head(
                model=model,
                dataloader=data_bundle.test_loader,
                criterion=criterion,
                device=device_obj,
                epoch=epochs,
                writer=writer,
                config=config,
                runtime=runtime,
                split_name="test",
            )
            print(
                "[AbsoluteRootHead] Test | "
                f"abs={final_test_metrics['absolute_mpjpe'] * 1000.0:.2f}mm, "
                f"root={final_test_metrics['root_error'] * 1000.0:.2f}mm, "
                f"root_aligned={final_test_metrics['root_aligned_mpjpe'] * 1000.0:.2f}mm"
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
        best_val_absolute_mpjpe=best_val_absolute_mpjpe,
        history=history,
        final_val_metrics=final_val_metrics,
        final_test_metrics=final_test_metrics,
        completed=True,
    )
    print(f"[AbsoluteRootHead] Training completed. Best val absolute MPJPE: {best_val_absolute_mpjpe * 1000.0:.2f} mm")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train absolute-root-head estimator")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint")
    parser.add_argument("--work-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--reset-epoch", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    args = parser.parse_args()
    run_absolute_root_head(
        load_config(args.config),
        config_path=args.config,
        resume=args.resume,
        work_dir=args.work_dir,
        device=args.device,
        seed=args.seed,
        reset_epoch=args.reset_epoch,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        max_test_batches=args.max_test_batches,
    )


if __name__ == "__main__":
    main()
