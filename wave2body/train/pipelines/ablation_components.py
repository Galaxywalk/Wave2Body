# -*- coding: utf-8 -*-
"""Training and evaluation pipeline for controlled component ablations."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..checkpoint import load_checkpoint, load_optional_component_states, module_state_dict, save_checkpoint
from ..common import safe_param_count, set_global_seed, write_run_manifest
from ..optim import build_hpe_token_optimizer
from ..radar_pose_data import build_radar_pose_dataloaders
from ...evaluate.m4human_action_groups import M4HumanActionGroupAccumulator, m4human_action_ids_from_indicator
from ...models.ablations import (
    AblationAbsoluteRootHeadEstimatorModel,
    AblationTokenHPEEstimatorModel,
    NoBodyTokenPooledMLP,
    NoBodyTokenRelativeRootHeadMLP,
)
from ...utils.metrics import compute_per_sample_p_mpjpe
from ...utils.runtime_utils import autocast_context, build_grad_scaler, configure_runtime, maybe_compile
from ._paths import resolve_run_root, write_effective_config
from .absolute_root_head import (
    _build_absolute_optimizer,
    _load_absolute_component_states,
    evaluate_absolute_root_head,
    train_absolute_epoch,
)
from .hpe_token import _write_summary as _write_hpe_summary
from .hpe_token_training import _soft_token_loss_from_config, _token_loss_from_config, train_hpe_token_epoch, validate_hpe_token


EXPERIMENT_NAME = "ablation_components_current"


def _target_absolute(batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    key = "target_poses_absolute"
    if key not in batch:
        available = ", ".join(sorted(str(name) for name in batch))
        raise KeyError(f"Expected {key!r} in batch. Available keys: {available}")
    return batch[key].to(device, non_blocking=True).float()


def _target_relative(batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    key = "target_poses"
    if key not in batch:
        available = ", ".join(sorted(str(name) for name in batch))
        raise KeyError(f"Expected {key!r} in batch. Available keys: {available}")
    return batch[key].to(device, non_blocking=True).float()


def _per_sample_mpjpe(pred_pose: torch.Tensor, target_pose: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pred_pose.float() - target_pose.float(), dim=-1).mean(dim=1)


def _root_aligned(pose: torch.Tensor, root_index: int) -> torch.Tensor:
    return pose.float() - pose.float()[:, root_index : root_index + 1]


def _metric_values(
    pred_pose: torch.Tensor,
    target_pose: torch.Tensor,
    *,
    root_index: int,
    compute_p_mpjpe: bool,
) -> dict[str, torch.Tensor]:
    values = {
        "absolute_mpjpe": _per_sample_mpjpe(pred_pose, target_pose),
        "root_error": torch.linalg.norm(
            pred_pose.float()[:, root_index] - target_pose.float()[:, root_index],
            dim=-1,
        ),
        "root_aligned_mpjpe": _per_sample_mpjpe(_root_aligned(pred_pose, root_index), _root_aligned(target_pose, root_index)),
    }
    if compute_p_mpjpe:
        values["p_mpjpe"] = torch.as_tensor(
            compute_per_sample_p_mpjpe(pred_pose, target_pose),
            device=pred_pose.device,
            dtype=torch.float32,
        )
    return values


def _relative_root_metric_values(
    outputs: dict[str, torch.Tensor],
    target_relative: torch.Tensor,
    target_absolute: torch.Tensor,
    *,
    root_index: int,
    compute_p_mpjpe: bool,
) -> dict[str, torch.Tensor]:
    values = _metric_values(
        outputs["pred_pose_absolute"],
        target_absolute,
        root_index=root_index,
        compute_p_mpjpe=compute_p_mpjpe,
    )
    values["relative_mpjpe"] = _per_sample_mpjpe(outputs["pred_pose_relative"], target_relative)
    oracle_root_pose = outputs["pred_pose_relative"].float() + target_absolute[:, root_index : root_index + 1].float()
    values["oracle_root_mpjpe"] = _per_sample_mpjpe(oracle_root_pose, target_absolute)
    return values


def _relative_root_loss_from_batch(
    *,
    outputs: dict[str, torch.Tensor],
    target_relative: torch.Tensor,
    target_absolute: torch.Tensor,
    root_index: int,
    training_cfg: dict[str, Any],
) -> dict[str, torch.Tensor]:
    relative_pose_loss = F.mse_loss(outputs["pred_pose_relative"].float(), target_relative.float())
    absolute_pose_loss = F.mse_loss(outputs["pred_pose_absolute"].float(), target_absolute.float())
    root_loss = F.mse_loss(outputs["pred_root"].float(), target_absolute[:, root_index].float())
    relative_weight = float(training_cfg.get("relative_pose_loss_weight", 1.0))
    absolute_weight = float(training_cfg.get("absolute_pose_loss_weight", 0.25))
    root_weight = float(training_cfg.get("root_loss_weight", 1.0))
    total = relative_weight * relative_pose_loss + absolute_weight * absolute_pose_loss + root_weight * root_loss
    return {
        "total_loss": total,
        "relative_pose_loss": relative_pose_loss,
        "absolute_pose_loss": absolute_pose_loss,
        "root_loss": root_loss,
    }


def _finalize_metric_totals(
    totals: dict[str, torch.Tensor],
    sample_count: int,
    *,
    scale: float = 1.0,
) -> dict[str, float]:
    denom = float(max(1, int(sample_count)))
    return {name: float((value / denom * float(scale)).detach().cpu().item()) for name, value in totals.items()}


def evaluate_no_body_token(
    *,
    model: NoBodyTokenPooledMLP,
    dataloader,
    device: torch.device,
    runtime,
    config: dict[str, Any],
    split_name: str,
) -> dict[str, Any]:
    model.eval()
    eval_cfg = config.get("evaluation", {})
    root_index = int(eval_cfg.get("root_index", config.get("vqvae", {}).get("root_index", 0)))
    compute_p_mpjpe = bool(eval_cfg.get("compute_p_mpjpe", True))
    metric_names = ["absolute_mpjpe", "root_error", "root_aligned_mpjpe"] + (["p_mpjpe"] if compute_p_mpjpe else [])
    totals = {name: torch.zeros((), device=device) for name in metric_names}
    action_acc = (
        M4HumanActionGroupAccumulator(metric_names=metric_names, device=device)
        if bool(eval_cfg.get("m4human_action_groups", True))
        else None
    )
    sample_count = 0
    num_batches = 0
    t0 = time.perf_counter()
    max_batches = config.get("training", {}).get(f"max_{split_name}_batches")
    max_batches = int(max_batches) if max_batches else None
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluate NoBodyToken {split_name}")):
            if max_batches is not None and batch_idx >= max_batches:
                break
            mmwave_points = batch["mmwave_points"].to(device, non_blocking=True)
            target_pose = _target_absolute(batch, device)
            with autocast_context(runtime):
                outputs = model(mmwave_points)
            pred_pose = outputs["pred_pose"]
            values = _metric_values(pred_pose, target_pose, root_index=root_index, compute_p_mpjpe=compute_p_mpjpe)
            batch_size = int(target_pose.shape[0])
            for name, value in values.items():
                totals[name] = totals[name] + value.detach().float().sum()
            if action_acc is not None and "indicator" in batch:
                action_ids = m4human_action_ids_from_indicator(batch["indicator"], device=device)
                for name, value in values.items():
                    action_acc.update(name, action_ids, value.detach())
            sample_count += batch_size
            num_batches += 1
    if num_batches == 0:
        raise RuntimeError(f"No {split_name} batches were processed for no-body-token ablation.")
    metrics = _finalize_metric_totals(totals, sample_count)
    for name in list(metrics):
        metrics[f"{name}_mm"] = metrics[name] * 1000.0
    metrics["mpjpe"] = metrics["absolute_mpjpe"]
    metrics["mpjpe_mm"] = metrics["absolute_mpjpe_mm"]
    metrics["num_samples"] = int(sample_count)
    metrics["num_batches"] = int(num_batches)
    metrics["samples_per_sec"] = float(sample_count / max(time.perf_counter() - t0, 1e-6))
    if action_acc is not None:
        metrics["m4human_action_groups"] = action_acc.summary(scale=1000.0)
    return metrics


def train_no_body_token_epoch(
    *,
    model: NoBodyTokenPooledMLP,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    runtime,
    grad_scaler,
    config: dict[str, Any],
    epoch: int,
    writer: SummaryWriter,
) -> dict[str, float]:
    training_cfg = config.get("training", {})
    grad_accum_steps = max(1, int(training_cfg.get("grad_accum_steps", 1)))
    max_batches = training_cfg.get("max_train_batches")
    max_batches = int(max_batches) if max_batches else None
    grad_norm = float(training_cfg.get("grad_norm", 0.0))
    log_interval = int(training_cfg.get("log_interval", 20))
    root_index = int(config.get("evaluation", {}).get("root_index", 0))
    model.train()
    total_loss = torch.zeros((), device=device)
    total_mpjpe = torch.zeros((), device=device)
    sample_count = 0
    pending_backward_steps = 0
    optimizer.zero_grad(set_to_none=True)
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"NoBodyToken train {epoch + 1}")):
        if max_batches is not None and batch_idx >= max_batches:
            break
        mmwave_points = batch["mmwave_points"].to(device, non_blocking=True)
        target_pose = _target_absolute(batch, device)
        with autocast_context(runtime):
            outputs = model(mmwave_points)
            loss = F.mse_loss(outputs["pred_pose"].float(), target_pose.float())
            loss_scaled = loss / grad_accum_steps
        if grad_scaler.is_enabled():
            grad_scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()
        pending_backward_steps += 1
        should_step = ((batch_idx + 1) % grad_accum_steps) == 0
        if should_step:
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
            pending_backward_steps = 0
        batch_size = int(target_pose.shape[0])
        total_loss = total_loss + loss.detach().float() * batch_size
        total_mpjpe = total_mpjpe + _metric_values(
            outputs["pred_pose"].detach(),
            target_pose,
            root_index=root_index,
            compute_p_mpjpe=False,
        )["absolute_mpjpe"].sum()
        sample_count += batch_size
        if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
            writer.add_scalar("train/loss", float(loss.detach().cpu().item()), epoch * 100000 + batch_idx)
    if pending_backward_steps > 0:
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
    if sample_count == 0:
        raise RuntimeError("No train batches were processed for no-body-token ablation.")
    return {
        "loss": float((total_loss / sample_count).detach().cpu().item()),
        "absolute_mpjpe": float((total_mpjpe / sample_count).detach().cpu().item()),
        "absolute_mpjpe_mm": float((total_mpjpe / sample_count * 1000.0).detach().cpu().item()),
    }


def _build_no_body_relative_root_optimizer(
    model: NoBodyTokenRelativeRootHeadMLP,
    config: dict[str, Any],
) -> torch.optim.Optimizer:
    training_cfg = config.get("training", {})
    base_lr = float(training_cfg.get("lr", 5e-5))
    weight_decay = float(training_cfg.get("weight_decay", 0.01))
    root_lr_mult = float(training_cfg.get("root_head_lr_mult", training_cfg.get("root_lr_mult", 1.0)))
    tokenizer_lr_mult = float(training_cfg.get("mmwave_tokenizer_lr_mult", 1.0))
    groups: list[dict[str, Any]] = []

    regressor_params = [param for param in model.regressor.parameters() if param.requires_grad]
    root_params = [param for param in model.root_head.parameters() if param.requires_grad]
    tokenizer_params = [param for param in model.mmwave_tokenizer.parameters() if param.requires_grad]
    if regressor_params:
        groups.append({"params": regressor_params, "lr": base_lr, "name": "regressor"})
    if root_params:
        groups.append({"params": root_params, "lr": base_lr * root_lr_mult, "name": "root_head"})
    if tokenizer_params:
        groups.append({"params": tokenizer_params, "lr": base_lr * tokenizer_lr_mult, "name": "mmwave_tokenizer"})
    if not groups:
        raise RuntimeError("No trainable parameters found for no-body relative-root optimizer.")
    print(
        "[NoBodyRelativeRoot] Optimizer param groups | "
        f"regressor={safe_param_count(regressor_params):,}, "
        f"root={safe_param_count(root_params):,}, "
        f"mmwave_tokenizer={safe_param_count(tokenizer_params):,}"
    )
    return torch.optim.AdamW(groups, lr=base_lr, weight_decay=weight_decay)


def evaluate_no_body_relative_root(
    *,
    model: NoBodyTokenRelativeRootHeadMLP,
    dataloader,
    device: torch.device,
    runtime,
    config: dict[str, Any],
    split_name: str,
) -> dict[str, Any]:
    model.eval()
    eval_cfg = config.get("evaluation", {})
    root_index = int(eval_cfg.get("root_index", config.get("data_format", {}).get("root_index", 0)))
    compute_p_mpjpe = bool(eval_cfg.get("compute_p_mpjpe", True))
    metric_names = [
        "absolute_mpjpe",
        "root_error",
        "root_aligned_mpjpe",
        "relative_mpjpe",
        "oracle_root_mpjpe",
    ] + (["p_mpjpe"] if compute_p_mpjpe else [])
    totals = {name: torch.zeros((), device=device) for name in metric_names}
    loss_totals = {
        "total_loss": torch.zeros((), device=device),
        "relative_pose_loss": torch.zeros((), device=device),
        "absolute_pose_loss": torch.zeros((), device=device),
        "root_loss": torch.zeros((), device=device),
    }
    action_acc = (
        M4HumanActionGroupAccumulator(metric_names=metric_names, device=device)
        if bool(eval_cfg.get("m4human_action_groups", True))
        else None
    )
    sample_count = 0
    num_batches = 0
    t0 = time.perf_counter()
    max_batches = config.get("training", {}).get(f"max_{split_name}_batches")
    max_batches = int(max_batches) if max_batches else None
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluate NoBodyRelativeRoot {split_name}")):
            if max_batches is not None and batch_idx >= max_batches:
                break
            mmwave_points = batch["mmwave_points"].to(device, non_blocking=True)
            target_relative = _target_relative(batch, device)
            target_absolute = _target_absolute(batch, device)
            with autocast_context(runtime):
                outputs = model(mmwave_points)
                loss_dict = _relative_root_loss_from_batch(
                    outputs=outputs,
                    target_relative=target_relative,
                    target_absolute=target_absolute,
                    root_index=root_index,
                    training_cfg=config.get("training", {}),
                )
            values = _relative_root_metric_values(
                outputs,
                target_relative,
                target_absolute,
                root_index=root_index,
                compute_p_mpjpe=compute_p_mpjpe,
            )
            batch_size = int(target_absolute.shape[0])
            for name, value in values.items():
                totals[name] = totals[name] + value.detach().float().sum()
            for name, value in loss_dict.items():
                loss_totals[name] = loss_totals[name] + value.detach().float() * batch_size
            if action_acc is not None and "indicator" in batch:
                action_ids = m4human_action_ids_from_indicator(batch["indicator"], device=device)
                for name, value in values.items():
                    action_acc.update(name, action_ids, value.detach())
            sample_count += batch_size
            num_batches += 1
    if num_batches == 0:
        raise RuntimeError(f"No {split_name} batches were processed for no-body relative-root ablation.")
    metrics = _finalize_metric_totals({**totals, **loss_totals}, sample_count)
    for name in metric_names:
        if name in metrics:
            metrics[f"{name}_mm"] = metrics[name] * 1000.0
    metrics["mpjpe"] = metrics["absolute_mpjpe"]
    metrics["mpjpe_mm"] = metrics["absolute_mpjpe_mm"]
    metrics["num_samples"] = int(sample_count)
    metrics["num_batches"] = int(num_batches)
    metrics["samples_per_sec"] = float(sample_count / max(time.perf_counter() - t0, 1e-6))
    if action_acc is not None:
        metrics["m4human_action_groups"] = action_acc.summary(scale=1000.0)
    return metrics


def train_no_body_relative_root_epoch(
    *,
    model: NoBodyTokenRelativeRootHeadMLP,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    runtime,
    grad_scaler,
    config: dict[str, Any],
    epoch: int,
    writer: SummaryWriter,
) -> dict[str, float]:
    training_cfg = config.get("training", {})
    grad_accum_steps = max(1, int(training_cfg.get("grad_accum_steps", 1)))
    max_batches = training_cfg.get("max_train_batches")
    max_batches = int(max_batches) if max_batches else None
    grad_norm = float(training_cfg.get("grad_norm", 0.0))
    log_interval = int(training_cfg.get("log_interval", 20))
    root_index = int(config.get("evaluation", {}).get("root_index", config.get("data_format", {}).get("root_index", 0)))
    model.train()
    if getattr(model, "freeze_mmwave_tokenizer", False):
        model.mmwave_tokenizer.eval()
    totals = {
        "total_loss": torch.zeros((), device=device),
        "relative_pose_loss": torch.zeros((), device=device),
        "absolute_pose_loss": torch.zeros((), device=device),
        "root_loss": torch.zeros((), device=device),
        "absolute_mpjpe": torch.zeros((), device=device),
        "relative_mpjpe": torch.zeros((), device=device),
        "root_error": torch.zeros((), device=device),
    }
    sample_count = 0
    pending_backward_steps = 0
    optimizer.zero_grad(set_to_none=True)
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"NoBodyRelativeRoot train {epoch + 1}")):
        if max_batches is not None and batch_idx >= max_batches:
            break
        mmwave_points = batch["mmwave_points"].to(device, non_blocking=True)
        target_relative = _target_relative(batch, device)
        target_absolute = _target_absolute(batch, device)
        with autocast_context(runtime):
            outputs = model(mmwave_points)
            loss_dict = _relative_root_loss_from_batch(
                outputs=outputs,
                target_relative=target_relative,
                target_absolute=target_absolute,
                root_index=root_index,
                training_cfg=training_cfg,
            )
            loss_scaled = loss_dict["total_loss"] / grad_accum_steps
        if grad_scaler.is_enabled():
            grad_scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()
        pending_backward_steps += 1
        should_step = ((batch_idx + 1) % grad_accum_steps) == 0
        if should_step:
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
            pending_backward_steps = 0
        batch_size = int(target_absolute.shape[0])
        metric_values = _relative_root_metric_values(
            outputs,
            target_relative,
            target_absolute,
            root_index=root_index,
            compute_p_mpjpe=False,
        )
        for name, value in loss_dict.items():
            totals[name] = totals[name] + value.detach().float() * batch_size
        for name in ("absolute_mpjpe", "relative_mpjpe", "root_error"):
            totals[name] = totals[name] + metric_values[name].detach().float().sum()
        sample_count += batch_size
        if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
            step = epoch * 100000 + batch_idx
            writer.add_scalar("train/loss", float(loss_dict["total_loss"].detach().cpu().item()), step)
            writer.add_scalar(
                "train/absolute_mpjpe_mm",
                float(metric_values["absolute_mpjpe"].mean().detach().cpu().item() * 1000.0),
                step,
            )
            writer.add_scalar(
                "train/root_error_mm",
                float(metric_values["root_error"].mean().detach().cpu().item() * 1000.0),
                step,
            )
    if pending_backward_steps > 0:
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
    if sample_count == 0:
        raise RuntimeError("No train batches were processed for no-body relative-root ablation.")
    metrics = {name: float((value / sample_count).detach().cpu().item()) for name, value in totals.items()}
    metrics["absolute_mpjpe_mm"] = metrics["absolute_mpjpe"] * 1000.0
    metrics["relative_mpjpe_mm"] = metrics["relative_mpjpe"] * 1000.0
    metrics["root_error_mm"] = metrics["root_error"] * 1000.0
    return metrics


def _save_no_body_checkpoint(
    path: Path,
    *,
    model: NoBodyTokenPooledMLP,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    save_checkpoint(
        path,
        {
            "epoch": int(epoch),
            "metrics": metrics,
            "config": config,
            "model_state_dict": module_state_dict(model),
            "mmwave_tokenizer_state_dict": module_state_dict(model.mmwave_tokenizer),
            "regressor_state_dict": module_state_dict(model.regressor),
            "optimizer_state_dict": optimizer.state_dict(),
        },
    )


def _save_no_body_relative_root_checkpoint(
    path: Path,
    *,
    model: NoBodyTokenRelativeRootHeadMLP,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    save_checkpoint(
        path,
        {
            "epoch": int(epoch),
            "metrics": metrics,
            "config": config,
            "model_state_dict": module_state_dict(model),
            "mmwave_tokenizer_state_dict": module_state_dict(model.mmwave_tokenizer),
            "regressor_state_dict": module_state_dict(model.regressor),
            "root_head_state_dict": module_state_dict(model.root_head),
            "optimizer_state_dict": optimizer.state_dict(),
        },
    )


def run_no_body_token(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    work_dir: str | Path | None = None,
    device: str = "cuda",
    resume: str | Path | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    max_test_batches: int | None = None,
) -> dict[str, Any]:
    training_cfg = config.setdefault("training", {})
    training_cfg["stage"] = "ablation_no_body_token"
    training_cfg.setdefault("run_name", "m4human_no_body_token_pooled_mlp")
    training_cfg.setdefault("freeze_mmwave_tokenizer", True)
    if max_train_batches is not None:
        training_cfg["max_train_batches"] = int(max_train_batches)
    if max_val_batches is not None:
        training_cfg["max_val_batches"] = int(max_val_batches)
    if max_test_batches is not None:
        training_cfg["max_test_batches"] = int(max_test_batches)
    if training_cfg.get("seed") is not None:
        set_global_seed(int(training_cfg["seed"]), deterministic=bool(training_cfg.get("deterministic", False)))

    output_root, checkpoint_dir, log_dir = resolve_run_root(
        config,
        work_dir=work_dir,
        default_name=str(training_cfg["run_name"]),
        experiment_name=EXPERIMENT_NAME,
    )
    effective_config_path = write_effective_config(config, output_root)
    device_obj = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    runtime = configure_runtime(training_cfg, device_obj, default_amp=True, log_prefix="[NoBodyToken Runtime]")
    raw_model = NoBodyTokenPooledMLP(config).to(device_obj)
    raw_model.load_pretrained_weights(config.get("pretrained", {}).get("mmwave_tokenizer"), device=device_obj)
    model = maybe_compile(raw_model, runtime, log_prefix="[NoBodyToken Runtime]")
    data_bundle = build_radar_pose_dataloaders(
        config=config,
        checkpoint_dir=output_root,
        include_test=True,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        max_test_batches=max_test_batches,
    )
    optimizer = torch.optim.AdamW(
        [param for param in raw_model.parameters() if param.requires_grad],
        lr=float(training_cfg.get("lr", 5e-5)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
    )
    if resume:
        ckpt = load_checkpoint(resume, map_location=device_obj)
        load_optional_component_states(raw_model, ckpt, prefix="[NoBodyToken]")
        if "regressor_state_dict" in ckpt:
            raw_model.regressor.load_state_dict(ckpt["regressor_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    grad_scaler = build_grad_scaler(runtime)
    writer = SummaryWriter(str(log_dir))
    write_run_manifest(
        output_root,
        config_path=effective_config_path,
        checkpoint_path=resume or config.get("pretrained", {}).get("mmwave_tokenizer"),
        config=config,
        dataset_split=str(config.get("m4human", {}).get("split", "m4human")),
        coord_transform=config.get("coord_transform"),
        output_paths={"output_root": output_root, "checkpoint_dir": checkpoint_dir, "log_dir": log_dir},
        extra={"stage": training_cfg["stage"], "trainable_params": safe_param_count(raw_model, trainable_only=True)},
    )

    best_val = float("inf")
    history: list[dict[str, Any]] = []
    epochs = int(training_cfg.get("epochs", 25))
    try:
        for epoch in range(epochs):
            train_metrics = train_no_body_token_epoch(
                model=model,
                dataloader=data_bundle.train_loader,
                optimizer=optimizer,
                device=device_obj,
                runtime=runtime,
                grad_scaler=grad_scaler,
                config=config,
                epoch=epoch,
                writer=writer,
            )
            val_metrics = evaluate_no_body_token(
                model=model,
                dataloader=data_bundle.val_loader,
                device=device_obj,
                runtime=runtime,
                config=config,
                split_name="val",
            )
            val_key = float(val_metrics["absolute_mpjpe"])
            history.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})
            _save_no_body_checkpoint(
                checkpoint_dir / "latest.pth",
                model=raw_model,
                optimizer=optimizer,
                epoch=epoch + 1,
                metrics=val_metrics,
                config=config,
            )
            if val_key < best_val:
                best_val = val_key
                _save_no_body_checkpoint(
                    checkpoint_dir / "best.pth",
                    model=raw_model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    metrics=val_metrics,
                    config=config,
                )
    finally:
        writer.close()

    best_ckpt = load_checkpoint(checkpoint_dir / "best.pth", map_location=device_obj)
    load_optional_component_states(raw_model, best_ckpt, prefix="[NoBodyToken best]")
    if "regressor_state_dict" in best_ckpt:
        raw_model.regressor.load_state_dict(best_ckpt["regressor_state_dict"], strict=False)
    test_metrics = evaluate_no_body_token(
        model=model,
        dataloader=data_bundle.test_loader,
        device=device_obj,
        runtime=runtime,
        config=config,
        split_name="test",
    )
    metrics_path = output_root / "metrics.json"
    metrics_path.write_text(json.dumps({"val": history[-1]["val"], "test": test_metrics}, indent=2), encoding="utf-8")
    summary = {
        "stage": training_cfg["stage"],
        "best_val_absolute_mpjpe_mm": best_val * 1000.0,
        "output_root": str(output_root),
        "best_checkpoint": str(checkpoint_dir / "best.pth"),
        "metrics": str(metrics_path),
        "history": history,
        "test": test_metrics,
    }
    (output_root / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_no_body_relative_root(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    work_dir: str | Path | None = None,
    device: str = "cuda",
    resume: str | Path | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    max_test_batches: int | None = None,
) -> dict[str, Any]:
    training_cfg = config.setdefault("training", {})
    training_cfg["stage"] = "ablation_no_body_relative_root"
    training_cfg.setdefault("run_name", "m4human_no_body_token_relative_root")
    training_cfg.setdefault("freeze_mmwave_tokenizer", True)
    training_cfg.setdefault("relative_pose_loss_weight", 1.0)
    training_cfg.setdefault("absolute_pose_loss_weight", 0.25)
    training_cfg.setdefault("root_loss_weight", 1.0)
    training_cfg.setdefault("root_head_lr_mult", 6.0)
    if max_train_batches is not None:
        training_cfg["max_train_batches"] = int(max_train_batches)
    if max_val_batches is not None:
        training_cfg["max_val_batches"] = int(max_val_batches)
    if max_test_batches is not None:
        training_cfg["max_test_batches"] = int(max_test_batches)
    if training_cfg.get("seed") is not None:
        set_global_seed(int(training_cfg["seed"]), deterministic=bool(training_cfg.get("deterministic", False)))

    output_root, checkpoint_dir, log_dir = resolve_run_root(
        config,
        work_dir=work_dir,
        default_name=str(training_cfg["run_name"]),
        experiment_name=EXPERIMENT_NAME,
    )
    effective_config_path = write_effective_config(config, output_root)
    device_obj = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    runtime = configure_runtime(training_cfg, device_obj, default_amp=True, log_prefix="[NoBodyRelativeRoot Runtime]")
    raw_model = NoBodyTokenRelativeRootHeadMLP(config).to(device_obj)
    raw_model.load_pretrained_weights(config.get("pretrained", {}).get("mmwave_tokenizer"), device=device_obj)
    model = maybe_compile(raw_model, runtime, log_prefix="[NoBodyRelativeRoot Runtime]")
    data_bundle = build_radar_pose_dataloaders(
        config=config,
        checkpoint_dir=output_root,
        include_test=True,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        max_test_batches=max_test_batches,
    )
    optimizer = _build_no_body_relative_root_optimizer(raw_model, config)
    if resume:
        ckpt = load_checkpoint(resume, map_location=device_obj)
        load_optional_component_states(raw_model, ckpt, prefix="[NoBodyRelativeRoot]")
        if "regressor_state_dict" in ckpt:
            raw_model.regressor.load_state_dict(ckpt["regressor_state_dict"], strict=False)
        if "root_head_state_dict" in ckpt:
            raw_model.root_head.load_state_dict(ckpt["root_head_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    grad_scaler = build_grad_scaler(runtime)
    writer = SummaryWriter(str(log_dir))
    write_run_manifest(
        output_root,
        config_path=effective_config_path,
        checkpoint_path=resume or config.get("pretrained", {}).get("mmwave_tokenizer"),
        config=config,
        dataset_split=str(config.get("m4human", {}).get("split", "m4human")),
        coord_transform=config.get("coord_transform"),
        output_paths={"output_root": output_root, "checkpoint_dir": checkpoint_dir, "log_dir": log_dir},
        extra={"stage": training_cfg["stage"], "trainable_params": safe_param_count(raw_model, trainable_only=True)},
    )

    best_val = float("inf")
    history: list[dict[str, Any]] = []
    epochs = int(training_cfg.get("epochs", 25))
    try:
        for epoch in range(epochs):
            train_metrics = train_no_body_relative_root_epoch(
                model=model,
                dataloader=data_bundle.train_loader,
                optimizer=optimizer,
                device=device_obj,
                runtime=runtime,
                grad_scaler=grad_scaler,
                config=config,
                epoch=epoch,
                writer=writer,
            )
            val_metrics = evaluate_no_body_relative_root(
                model=model,
                dataloader=data_bundle.val_loader,
                device=device_obj,
                runtime=runtime,
                config=config,
                split_name="val",
            )
            val_key = float(val_metrics["absolute_mpjpe"])
            history.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})
            _save_no_body_relative_root_checkpoint(
                checkpoint_dir / "latest.pth",
                model=raw_model,
                optimizer=optimizer,
                epoch=epoch + 1,
                metrics=val_metrics,
                config=config,
            )
            if val_key < best_val:
                best_val = val_key
                _save_no_body_relative_root_checkpoint(
                    checkpoint_dir / "best.pth",
                    model=raw_model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    metrics=val_metrics,
                    config=config,
                )
    finally:
        writer.close()

    best_ckpt = load_checkpoint(checkpoint_dir / "best.pth", map_location=device_obj)
    load_optional_component_states(raw_model, best_ckpt, prefix="[NoBodyRelativeRoot best]")
    if "regressor_state_dict" in best_ckpt:
        raw_model.regressor.load_state_dict(best_ckpt["regressor_state_dict"], strict=False)
    if "root_head_state_dict" in best_ckpt:
        raw_model.root_head.load_state_dict(best_ckpt["root_head_state_dict"], strict=False)
    test_metrics = evaluate_no_body_relative_root(
        model=model,
        dataloader=data_bundle.test_loader,
        device=device_obj,
        runtime=runtime,
        config=config,
        split_name="test",
    )
    metrics_path = output_root / "metrics.json"
    metrics_path.write_text(json.dumps({"val": history[-1]["val"], "test": test_metrics}, indent=2), encoding="utf-8")
    summary = {
        "stage": training_cfg["stage"],
        "best_val_absolute_mpjpe_mm": best_val * 1000.0,
        "output_root": str(output_root),
        "best_checkpoint": str(checkpoint_dir / "best.pth"),
        "metrics": str(metrics_path),
        "history": history,
        "test": test_metrics,
    }
    (output_root / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_ablation_hpe_token(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    resume: str | Path | None = None,
    work_dir: str | Path | None = None,
    device: str = "cuda",
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> dict[str, Any]:
    training_cfg = config.setdefault("training", {})
    training_cfg["stage"] = "hpe_token"
    training_cfg["stage_display_name"] = "HPE Token Ablation"
    training_cfg["metric_namespace"] = "hpe_token"
    training_cfg.setdefault("freeze_mmwave_tokenizer", False)
    training_cfg.setdefault("freeze_vqvae", True)
    training_cfg.setdefault("frozen_feature_cache", {"enabled": False})
    training_cfg.setdefault("latent_loss_weight", 0.3)
    training_cfg.setdefault("token_loss_weight", 0.0)
    training_cfg.setdefault("pose_loss_weight", 1.0)
    if max_train_batches is not None:
        training_cfg["max_train_batches"] = int(max_train_batches)
    if max_val_batches is not None:
        training_cfg["max_val_batches"] = int(max_val_batches)
    if training_cfg.get("seed") is not None:
        set_global_seed(int(training_cfg["seed"]), deterministic=bool(training_cfg.get("deterministic", False)))

    output_root, checkpoint_dir, log_dir = resolve_run_root(
        config,
        work_dir=work_dir,
        default_name=str(training_cfg.get("run_name", "hpe_token_ablation")),
        experiment_name=EXPERIMENT_NAME,
    )
    effective_config_path = write_effective_config(config, output_root)
    device_obj = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    runtime = configure_runtime(training_cfg, device_obj, default_amp=True, log_prefix="[HPE Ablation Runtime]")
    raw_model = AblationTokenHPEEstimatorModel(
        mmwave_tokenizer_config=config["mmwave_tokenizer"],
        sensor_token_config=config.get("sensor_tokens", {}),
        token_translator_config=config["token_translator"],
        vqvae_config=config["vqvae"],
        device=str(device_obj),
        freeze_mmwave_tokenizer=bool(training_cfg.get("freeze_mmwave_tokenizer", False)),
        freeze_token_translator=bool(training_cfg.get("freeze_token_translator", False)),
        freeze_vqvae=bool(training_cfg.get("freeze_vqvae", True)),
    )
    pretrained = config.get("pretrained", {})
    raw_model.load_pretrained_weights(
        mmwave_tokenizer_path=pretrained.get("mmwave_tokenizer"),
        vqvae_path=pretrained.get("vqvae"),
        device=str(device_obj),
    )
    raw_model = raw_model.to(device_obj)
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
        checkpoint_path=resume or pretrained.get("mmwave_tokenizer") or pretrained.get("vqvae"),
        config=config,
        dataset_split=str(config.get("m4human", {}).get("split", "m4human")),
        coord_transform=config.get("coord_transform"),
        output_paths={"output_root": output_root, "checkpoint_dir": checkpoint_dir, "log_dir": log_dir},
        extra={"stage": "hpe_token_ablation"},
    )
    if resume:
        ckpt = load_checkpoint(resume, map_location=device_obj)
        load_optional_component_states(raw_model, ckpt, prefix="[HPE Ablation]")
    criterion = _token_loss_from_config(training_cfg)
    soft_token_criterion, soft_token_weight = _soft_token_loss_from_config(training_cfg, raw_model.vqvae.layout)
    if soft_token_criterion is not None:
        soft_token_criterion = soft_token_criterion.to(device_obj)
    optimizer = build_hpe_token_optimizer(raw_model, config)
    model = maybe_compile(raw_model, runtime, log_prefix="[HPE Ablation Runtime]")
    grad_scaler = build_grad_scaler(runtime)
    writer = SummaryWriter(str(log_dir))
    best_val = float("inf")
    history: list[dict[str, Any]] = []
    epochs = int(training_cfg.get("epochs", 25))
    try:
        for epoch in range(epochs):
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
            history.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})
            val_mpjpe = float(val_metrics["mpjpe"])
            payload = {
                "epoch": epoch + 1,
                "config": config,
                "metrics": val_metrics,
                "mmwave_tokenizer_state_dict": module_state_dict(raw_model.mmwave_tokenizer),
                "vqvae_state_dict": module_state_dict(raw_model.vqvae),
                "token_translator_state_dict": module_state_dict(raw_model.token_translator),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint_dir / "latest.pth", payload)
            if val_mpjpe < best_val:
                best_val = val_mpjpe
                save_checkpoint(checkpoint_dir / "best.pth", payload)
    finally:
        writer.close()
    return _write_hpe_summary(
        output_root,
        config_path=config_path,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        runtime=runtime,
        epochs=epochs,
        best_val_mpjpe=best_val,
        history=history,
    )


def run_ablation_absolute_root(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    resume: str | Path | None = None,
    work_dir: str | Path | None = None,
    device: str = "cuda",
    reset_epoch: bool = False,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    max_test_batches: int | None = None,
) -> dict[str, Any]:
    del reset_epoch
    training_cfg = config.setdefault("training", {})
    training_cfg["stage"] = "absolute_root_head"
    training_cfg["stage_display_name"] = "Absolute Root Head Ablation"
    training_cfg["metric_namespace"] = "absolute_root_head"
    training_cfg.setdefault("freeze_mmwave_tokenizer", True)
    training_cfg.setdefault("freeze_token_translator", False)
    training_cfg.setdefault("freeze_vqvae", True)
    if max_train_batches is not None:
        training_cfg["max_train_batches"] = int(max_train_batches)
    if max_val_batches is not None:
        training_cfg["max_val_batches"] = int(max_val_batches)
    if max_test_batches is not None:
        training_cfg["max_test_batches"] = int(max_test_batches)
    if training_cfg.get("seed") is not None:
        set_global_seed(int(training_cfg["seed"]), deterministic=bool(training_cfg.get("deterministic", False)))

    output_root, checkpoint_dir, log_dir = resolve_run_root(
        config,
        work_dir=work_dir,
        default_name=str(training_cfg.get("run_name", "absolute_root_ablation")),
        experiment_name=EXPERIMENT_NAME,
    )
    effective_config_path = write_effective_config(config, output_root)
    device_obj = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    runtime = configure_runtime(training_cfg, device_obj, default_amp=True, log_prefix="[Absolute Ablation Runtime]")
    model = AblationAbsoluteRootHeadEstimatorModel(
        mmwave_tokenizer_config=config["mmwave_tokenizer"],
        sensor_token_config=config.get("sensor_tokens", {}),
        token_translator_config=config["token_translator"],
        vqvae_config=config["vqvae"],
        root_head_config=config.get("root_head", {}),
        device=str(device_obj),
        freeze_mmwave_tokenizer=bool(training_cfg.get("freeze_mmwave_tokenizer", True)),
        freeze_token_translator=bool(training_cfg.get("freeze_token_translator", False)),
        freeze_vqvae=bool(training_cfg.get("freeze_vqvae", True)),
    )
    pretrained = config.get("pretrained", {})
    model.load_pretrained_weights(
        mmwave_tokenizer_path=pretrained.get("mmwave_tokenizer"),
        vqvae_path=pretrained.get("vqvae"),
        hpe_token_path=pretrained.get("hpe_token"),
        device=str(device_obj),
    )
    model = model.to(device_obj)
    if resume:
        ckpt = load_checkpoint(resume, map_location=device_obj)
        _load_absolute_component_states(model, ckpt, prefix="[Absolute Ablation]")
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
        checkpoint_path=resume or pretrained.get("hpe_token") or pretrained.get("mmwave_tokenizer"),
        config=config,
        dataset_split=str(config.get("m4human", {}).get("split", "m4human")),
        coord_transform=config.get("coord_transform"),
        output_paths={"output_root": output_root, "checkpoint_dir": checkpoint_dir, "log_dir": log_dir},
        extra={"stage": "absolute_root_ablation"},
    )
    criterion = _token_loss_from_config(training_cfg)
    optimizer = _build_absolute_optimizer(model, config)
    grad_scaler = build_grad_scaler(runtime)
    writer = SummaryWriter(str(log_dir))
    best_val = float("inf")
    history: list[dict[str, Any]] = []
    epochs = int(training_cfg.get("epochs", 25))
    try:
        for epoch in range(epochs):
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
            history.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})
            payload = {
                "epoch": epoch + 1,
                "config": config,
                "metrics": val_metrics,
                "mmwave_tokenizer_state_dict": module_state_dict(model.mmwave_tokenizer),
                "vqvae_state_dict": module_state_dict(model.vqvae),
                "token_translator_state_dict": module_state_dict(model.token_translator),
                "root_head_state_dict": module_state_dict(model.root_head),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint_dir / "latest.pth", payload)
            val_abs = float(val_metrics["absolute_mpjpe"])
            if val_abs < best_val:
                best_val = val_abs
                save_checkpoint(checkpoint_dir / "best.pth", payload)
    finally:
        writer.close()

    best_ckpt = load_checkpoint(checkpoint_dir / "best.pth", map_location=device_obj)
    _load_absolute_component_states(model, best_ckpt, prefix="[Absolute Ablation best]")
    test_metrics = evaluate_absolute_root_head(
        model=model,
        dataloader=data_bundle.test_loader,
        criterion=criterion,
        device=device_obj,
        epoch=epochs,
        writer=None,
        config=config,
        runtime=runtime,
        split_name="test",
    )
    summary = {
        "stage": "absolute_root_head",
        "best_val_absolute_mpjpe_mm": best_val * 1000.0,
        "output_root": str(output_root),
        "checkpoint_dir": str(checkpoint_dir),
        "best_checkpoint": str(checkpoint_dir / "best.pth"),
        "tensorboard_dir": str(log_dir),
        "config": str(config_path) if config_path else None,
        "history": history,
        "test": test_metrics,
    }
    (output_root / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_root / "metrics.json").write_text(json.dumps({"test": test_metrics}, indent=2), encoding="utf-8")
    return summary
