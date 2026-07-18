# -*- coding: utf-8 -*-
"""Shared training helpers for PointTransformer pose baselines."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from ..models.point_transformer.pose_baseline import PTPoseModel, PoseMPJPELoss
    from ..utils.metrics import compute_p_mpjpe
    from ..utils.runtime_utils import autocast_context
except ImportError:  # Support scripts that put src/ directly on sys.path.
    from wave2body.models.point_transformer.pose_baseline import PTPoseModel, PoseMPJPELoss
    from wave2body.utils.metrics import compute_p_mpjpe
    from wave2body.utils.runtime_utils import autocast_context


@torch.no_grad()
def materialize_lazy_modules(
    model: PTPoseModel,
    config: dict,
    device: torch.device,
    runtime,
) -> None:
    """Run one dummy forward so lazy PT layers have concrete shapes."""
    n_points = int(config["data_format"]["n_points"])
    in_c = int(config["data_format"]["radar_input_c"])
    dummy = torch.zeros((1, n_points, in_c), dtype=torch.float32, device=device)
    model.eval()
    with autocast_context(runtime):
        _ = model(dummy)
    model.train()


def compute_lr_for_epoch(
    epoch: int,
    max_epochs: int,
    base_lr: float,
    scheduler_type: str = "cosine",
    warmup_epochs: int = 0,
    min_lr: float = 1e-6,
) -> float:
    if scheduler_type in ("none", "constant"):
        return float(base_lr)

    warmup_epochs = int(max(0, warmup_epochs))
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return float(base_lr) * float(epoch + 1) / float(warmup_epochs)

    if scheduler_type == "cosine":
        denom = max(1, max_epochs - warmup_epochs - 1)
        progress = float(epoch - warmup_epochs) / float(denom)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_lr) + (float(base_lr) - float(min_lr)) * cosine

    raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _compute_joint_errors(pred_pose, target_pose):
    pred = _to_numpy(pred_pose)
    gt = _to_numpy(target_pose)
    return np.linalg.norm(pred - gt, axis=-1)


def train_one_epoch(
    model: PTPoseModel,
    dataloader: DataLoader,
    criterion: PoseMPJPELoss,
    optimizer: torch.optim.Optimizer,
    grad_scaler,
    device: torch.device,
    epoch: int,
    grad_norm: float,
    runtime,
    grad_accum_steps: int,
    writer: Any,
    log_interval: int,
    global_step_start: int,
) -> tuple[dict, int]:
    model.train()
    total = {
        "total_loss": torch.zeros((), device=device),
        "pose_loss": torch.zeros((), device=device),
        "mean_joint_error_m": torch.zeros((), device=device),
    }
    num_batches = len(dataloader)
    if num_batches == 0:
        raise RuntimeError("Empty train dataloader")
    grad_accum_steps = max(1, int(grad_accum_steps))
    data_time_sum = 0.0
    step_time_sum = 0.0
    optimizer_steps = 0
    sample_count = 0
    epoch_t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
    global_step = global_step_start
    optimizer.zero_grad(set_to_none=True)
    loop_end_t = time.perf_counter()

    for batch_idx, batch in enumerate(pbar):
        data_time_sum += time.perf_counter() - loop_end_t
        step_t0 = time.perf_counter()
        mmwave_points = batch["mmwave_points"].to(device, non_blocking=True)
        target_poses = batch["target_poses"].to(device, non_blocking=True)
        sample_count += int(mmwave_points.shape[0])

        if runtime.compile_enabled and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        with autocast_context(runtime):
            outputs = model(mmwave_points)
            loss_dict = criterion(outputs["pred_pose"].float(), target_poses.float())
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
            if grad_scaler.is_enabled():
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        for key in total:
            total[key] = total[key] + loss_dict[key].detach()

        if batch_idx % max(1, log_interval) == 0:
            total_loss_value = float(loss_dict["total_loss"].detach().item())
            pose_mm_value = float(loss_dict["mean_joint_error_m"].detach().item() * 1000.0)
            pbar.set_postfix(
                {
                    "loss": f"{total_loss_value:.4f}",
                    "pose_mm": f"{pose_mm_value:.1f}",
                }
            )
            if writer is not None:
                writer.add_scalar("train/total_loss", total_loss_value, global_step)
                writer.add_scalar("train/pose_loss", float(loss_dict["pose_loss"].detach().item()), global_step)
                writer.add_scalar("train/mean_joint_error_mm", pose_mm_value, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
        global_step += 1
        step_time_sum += time.perf_counter() - step_t0
        loop_end_t = time.perf_counter()

    avg = {k: float((v / num_batches).item()) for k, v in total.items()}
    avg["mpjpe_mm"] = avg["mean_joint_error_m"] * 1000.0
    elapsed = max(time.perf_counter() - epoch_t0, 1e-6)
    avg["samples_per_sec"] = float(sample_count / elapsed)
    avg["data_time_sec"] = float(data_time_sum)
    avg["step_time_sec"] = float(step_time_sum)
    avg["optimizer_steps"] = int(optimizer_steps)
    avg["grad_accum_steps"] = int(grad_accum_steps)
    if device.type == "cuda":
        avg["max_memory_mb"] = float(torch.cuda.max_memory_allocated(device=device) / (1024.0**2))
    else:
        avg["max_memory_mb"] = 0.0
    return avg, global_step


@torch.no_grad()
def validate_one_epoch(
    model: PTPoseModel,
    dataloader: DataLoader,
    criterion: PoseMPJPELoss,
    device: torch.device,
    runtime,
    eval_cfg: dict,
) -> tuple[dict, dict]:
    model.eval()
    total = {
        "total_loss": 0.0,
        "pose_loss": 0.0,
        "mean_joint_error_m": 0.0,
    }
    all_pred = []
    all_gt = []
    num_batches = len(dataloader)
    if num_batches == 0:
        raise RuntimeError("Empty val dataloader")
    data_time_sum = 0.0
    step_time_sum = 0.0
    sample_count = 0
    t0 = time.perf_counter()
    loop_end_t = time.perf_counter()

    for batch in tqdm(dataloader, desc="Validate"):
        data_time_sum += time.perf_counter() - loop_end_t
        step_t0 = time.perf_counter()
        mmwave_points = batch["mmwave_points"].to(device, non_blocking=True)
        target_poses = batch["target_poses"].to(device, non_blocking=True).float()
        sample_count += int(mmwave_points.shape[0])

        if runtime.compile_enabled and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        with autocast_context(runtime):
            outputs = model(mmwave_points)
            loss_dict = criterion(outputs["pred_pose"].float(), target_poses)
        for key in total:
            total[key] += float(loss_dict[key].item())

        all_pred.append(outputs["pred_pose"].detach().float().cpu())
        all_gt.append(target_poses.detach().cpu())
        step_time_sum += time.perf_counter() - step_t0
        loop_end_t = time.perf_counter()

    avg = {k: v / num_batches for k, v in total.items()}
    avg["mpjpe_mm"] = avg["mean_joint_error_m"] * 1000.0
    elapsed = max(time.perf_counter() - t0, 1e-6)
    avg["samples_per_sec"] = float(sample_count / elapsed)
    avg["data_time_sec"] = float(data_time_sum)
    avg["step_time_sec"] = float(step_time_sum)

    pred = torch.cat(all_pred, dim=0).numpy()
    gt = torch.cat(all_gt, dim=0).numpy()
    errors = _compute_joint_errors(pred, gt)
    avg["mpjpe_mm"] = float(errors.mean() * 1000.0)
    if bool(eval_cfg.get("p_mpjpe", True)):
        avg["p_mpjpe_mm"] = float(compute_p_mpjpe(pred, gt).item() * 1000.0)
    return avg, {}


__all__ = [
    "compute_lr_for_epoch",
    "materialize_lazy_modules",
    "set_optimizer_lr",
    "train_one_epoch",
    "validate_one_epoch",
]
