# -*- coding: utf-8 -*-
"""Generic training helpers for config-driven pose baselines."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from ..models.baselines import PoseBaselineBundle, move_pose_baseline_batch, pose_baseline_step
    from ..utils.runtime_utils import autocast_context
except ImportError:  # Support scripts that put src/ directly on sys.path.
    from wave2body.models.baselines import PoseBaselineBundle, move_pose_baseline_batch, pose_baseline_step
    from wave2body.utils.runtime_utils import autocast_context


def _last_frame_if_temporal(pred: torch.Tensor) -> torch.Tensor:
    return pred[:, -1] if pred.ndim == 4 else pred


def _decoded_joints_from_outputs(outputs: dict[str, Any]) -> torch.Tensor | None:
    pred = outputs.get("pred_pose")
    if torch.is_tensor(pred):
        return _last_frame_if_temporal(pred)
    body = outputs.get("body")
    if isinstance(body, dict):
        joints = body.get("joints")
        if not torch.is_tensor(joints):
            joints = body.get("pred_joints")
        if torch.is_tensor(joints):
            return _last_frame_if_temporal(joints)
    return None


def _target_joints_from_batch(outputs: dict[str, Any], batch: dict[str, Any]) -> torch.Tensor | None:
    target = None
    if isinstance(outputs.get("body"), dict):
        target = batch.get("target_poses_absolute")
    if not torch.is_tensor(target):
        target = batch.get("target_poses")
    return target if torch.is_tensor(target) else None


def _mean_joint_error_m(outputs: dict[str, Any], batch: dict[str, Any]) -> torch.Tensor | None:
    pred = _decoded_joints_from_outputs(outputs)
    target = _target_joints_from_batch(outputs, batch)
    if not torch.is_tensor(pred) or not torch.is_tensor(target):
        return None
    if pred.shape != target.shape:
        return None
    return torch.norm(pred.float() - target.float(), dim=-1).mean()


def _update_totals(totals: dict[str, float], loss_dict: dict[str, torch.Tensor], weight: int) -> None:
    for key, value in loss_dict.items():
        if torch.is_tensor(value) and value.ndim == 0:
            totals[key] = totals.get(key, 0.0) + float(value.detach().item()) * weight


def _finalize_metrics(
    totals: dict[str, float],
    *,
    sample_count: int,
    batch_count: int,
    elapsed: float,
    data_time_sum: float,
    step_time_sum: float,
    optimizer_steps: int = 0,
    grad_accum_steps: int = 1,
    device: torch.device | None = None,
) -> dict[str, float | int]:
    if sample_count <= 0 or batch_count <= 0:
        raise RuntimeError("Empty pose-baseline dataloader")
    metrics: dict[str, float | int] = {key: value / sample_count for key, value in totals.items()}
    if "mean_joint_error_m" in metrics:
        metrics["mpjpe_mm"] = float(metrics["mean_joint_error_m"]) * 1000.0
    metrics["num_samples"] = int(sample_count)
    metrics["num_batches"] = int(batch_count)
    metrics["samples_per_sec"] = float(sample_count / max(elapsed, 1.0e-6))
    metrics["data_time_sec"] = float(data_time_sum)
    metrics["step_time_sec"] = float(step_time_sum)
    metrics["optimizer_steps"] = int(optimizer_steps)
    metrics["grad_accum_steps"] = int(grad_accum_steps)
    if device is not None and device.type == "cuda":
        metrics["max_memory_mb"] = float(torch.cuda.max_memory_allocated(device=device) / (1024.0**2))
    elif device is not None:
        metrics["max_memory_mb"] = 0.0
    return metrics


def train_pose_baseline_one_epoch(
    *,
    bundle: PoseBaselineBundle,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    grad_scaler,
    device: torch.device,
    runtime,
    epoch: int = 0,
    grad_norm: float = 1.0,
    grad_accum_steps: int = 1,
    writer: Any = None,
    log_interval: int = 20,
    global_step_start: int = 0,
) -> tuple[dict[str, float | int], int]:
    """Train any config-driven pose baseline for one epoch."""

    bundle.model.train()
    grad_accum_steps = max(1, int(grad_accum_steps))
    num_batches = len(dataloader)
    if num_batches == 0:
        raise RuntimeError("Empty train dataloader")

    totals: dict[str, float] = {}
    sample_count = 0
    optimizer_steps = 0
    data_time_sum = 0.0
    step_time_sum = 0.0
    global_step = int(global_step_start)
    epoch_t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    optimizer.zero_grad(set_to_none=True)
    loop_end_t = time.perf_counter()
    pbar = tqdm(dataloader, desc=f"Train {bundle.name} epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        data_time_sum += time.perf_counter() - loop_end_t
        step_t0 = time.perf_counter()
        moved_batch = move_pose_baseline_batch(batch, device)
        bsz = int(moved_batch["mmwave_points"].shape[0])
        sample_count += bsz

        with autocast_context(runtime):
            outputs, loss_dict = pose_baseline_step(bundle, moved_batch)
        joint_error = _mean_joint_error_m(outputs, moved_batch)
        if joint_error is not None and "mean_joint_error_m" not in loss_dict:
            loss_dict["mean_joint_error_m"] = joint_error.detach()

        loss_to_backprop = loss_dict["total_loss"] / float(grad_accum_steps)
        if grad_scaler.is_enabled():
            grad_scaler.scale(loss_to_backprop).backward()
        else:
            loss_to_backprop.backward()

        reached_accum_boundary = ((batch_idx + 1) % grad_accum_steps == 0)
        is_last_batch = (batch_idx + 1) >= num_batches
        if reached_accum_boundary or is_last_batch:
            if grad_norm > 0:
                if grad_scaler.is_enabled():
                    grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(bundle.model.parameters(), grad_norm)
            if grad_scaler.is_enabled():
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        _update_totals(totals, loss_dict, bsz)
        if batch_idx % max(1, int(log_interval)) == 0:
            total_value = float(loss_dict["total_loss"].detach().item())
            postfix = {"loss": f"{total_value:.4f}"}
            if "mean_joint_error_m" in loss_dict:
                postfix["pose_mm"] = f"{float(loss_dict['mean_joint_error_m'].detach().item()) * 1000.0:.1f}"
            pbar.set_postfix(postfix)
            if writer is not None:
                writer.add_scalar(f"{bundle.name}/train/total_loss", total_value, global_step)
                if "mean_joint_error_m" in loss_dict:
                    writer.add_scalar(
                        f"{bundle.name}/train/mean_joint_error_mm",
                        float(loss_dict["mean_joint_error_m"].detach().item()) * 1000.0,
                        global_step,
                    )
                writer.add_scalar(f"{bundle.name}/train/lr", optimizer.param_groups[0]["lr"], global_step)
        global_step += 1
        step_time_sum += time.perf_counter() - step_t0
        loop_end_t = time.perf_counter()

    metrics = _finalize_metrics(
        totals,
        sample_count=sample_count,
        batch_count=num_batches,
        elapsed=time.perf_counter() - epoch_t0,
        data_time_sum=data_time_sum,
        step_time_sum=step_time_sum,
        optimizer_steps=optimizer_steps,
        grad_accum_steps=grad_accum_steps,
        device=device,
    )
    return metrics, global_step


@torch.no_grad()
def validate_pose_baseline_one_epoch(
    *,
    bundle: PoseBaselineBundle,
    dataloader: DataLoader,
    device: torch.device,
    runtime,
    desc: str | None = None,
) -> dict[str, float | int]:
    """Validate any config-driven pose baseline for one epoch."""

    bundle.model.eval()
    num_batches = len(dataloader)
    if num_batches == 0:
        raise RuntimeError("Empty validation dataloader")

    totals: dict[str, float] = {}
    sample_count = 0
    data_time_sum = 0.0
    step_time_sum = 0.0
    t0 = time.perf_counter()
    loop_end_t = time.perf_counter()
    for batch in tqdm(dataloader, desc=desc or f"Validate {bundle.name}"):
        data_time_sum += time.perf_counter() - loop_end_t
        step_t0 = time.perf_counter()
        moved_batch = move_pose_baseline_batch(batch, device)
        bsz = int(moved_batch["mmwave_points"].shape[0])
        sample_count += bsz
        with autocast_context(runtime):
            outputs, loss_dict = pose_baseline_step(bundle, moved_batch)
        joint_error = _mean_joint_error_m(outputs, moved_batch)
        if joint_error is not None and "mean_joint_error_m" not in loss_dict:
            loss_dict["mean_joint_error_m"] = joint_error
        _update_totals(totals, loss_dict, bsz)
        step_time_sum += time.perf_counter() - step_t0
        loop_end_t = time.perf_counter()

    return _finalize_metrics(
        totals,
        sample_count=sample_count,
        batch_count=num_batches,
        elapsed=time.perf_counter() - t0,
        data_time_sum=data_time_sum,
        step_time_sum=step_time_sum,
    )


__all__ = [
    "train_pose_baseline_one_epoch",
    "validate_pose_baseline_one_epoch",
]
