# -*- coding: utf-8 -*-
"""Reusable PointTransformer pose training pipeline."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    from ..checkpoint import (
        load_checkpoint,
        load_point_transformer_checkpoint,
        load_state_dict_shape_safe,
        module_state_dict,
        save_checkpoint,
    )
    from ..common import load_config, safe_param_count, set_global_seed, write_run_manifest
    from ..radar_pose_data import LimitedDataLoader, build_radar_pose_dataloaders, create_radar_pose_dataset, dataset_name
    from ..pt_pose import (
        compute_lr_for_epoch,
        materialize_lazy_modules,
        set_optimizer_lr,
        train_one_epoch,
        validate_one_epoch,
    )
    from ...dataset.radar_pose.m4human_protocols import summarize_mpjpe_by_action_group
    from ...evaluate.m4human_action_groups import (
        M4HumanActionGroupAccumulator,
        M4HumanActionGroupVectorAccumulator,
        m4human_action_ids_from_indicator,
    )
    from ...models.point_transformer.pose_baseline import PTPoseModel, PoseMPJPELoss
    from ...utils.metrics import compute_per_sample_p_mpjpe
    from ...utils.runtime_utils import autocast_context, build_grad_scaler, configure_runtime, maybe_compile
except ImportError:  # Support scripts that put src/ directly on sys.path.
    from wave2body.dataset.radar_pose.m4human_protocols import summarize_mpjpe_by_action_group
    from wave2body.evaluate.m4human_action_groups import (
        M4HumanActionGroupAccumulator,
        M4HumanActionGroupVectorAccumulator,
        m4human_action_ids_from_indicator,
    )
    from wave2body.train.checkpoint import (
        load_checkpoint,
        load_point_transformer_checkpoint,
        load_state_dict_shape_safe,
        module_state_dict,
        save_checkpoint,
    )
    from wave2body.train.common import load_config, safe_param_count, set_global_seed, write_run_manifest
    from wave2body.train.radar_pose_data import LimitedDataLoader, build_radar_pose_dataloaders, create_radar_pose_dataset, dataset_name
    from wave2body.train.pt_pose_utils import (
        compute_lr_for_epoch,
        materialize_lazy_modules,
        set_optimizer_lr,
        train_one_epoch,
        validate_one_epoch,
    )
    from wave2body.models.point_transformer.pose_baseline import PTPoseModel, PoseMPJPELoss
    from wave2body.utils.metrics import compute_per_sample_p_mpjpe
    from wave2body.utils.runtime_utils import autocast_context, build_grad_scaler, configure_runtime, maybe_compile

from ._paths import resolve_run_root, write_effective_config


def apply_pt_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    cfg = dict(config)
    cfg.setdefault("training", {})
    if getattr(args, "split_id", None):
        cfg.setdefault("m4human", {})["split"] = args.split_id
    if getattr(args, "batch_size", None) is not None:
        cfg["training"]["batch_size"] = int(args.batch_size)
    if getattr(args, "epochs", None) is not None:
        cfg["training"]["epochs"] = int(args.epochs)
    if getattr(args, "precision", None):
        cfg["training"]["precision"] = args.precision
        cfg["training"]["amp"] = args.precision != "fp32"
    if getattr(args, "compile", None) is not None:
        current = cfg["training"].get("compile", {})
        if isinstance(current, dict):
            current["enabled"] = bool(args.compile)
            cfg["training"]["compile"] = current
        else:
            cfg["training"]["compile"] = bool(args.compile)
    if getattr(args, "run_name", None):
        cfg["training"]["run_name"] = args.run_name
    return cfg


def _per_sample_mpjpe(pred_pose: torch.Tensor, target_pose: torch.Tensor) -> torch.Tensor:
    return torch.norm(pred_pose.float() - target_pose.float(), dim=-1).mean(dim=1)


def _root_aligned(pose: torch.Tensor, root_index: int) -> torch.Tensor:
    return pose - pose[:, root_index : root_index + 1]


def _needs_metric(eval_cfg: dict[str, Any], name: str) -> bool:
    diagnostics = eval_cfg.get("diagnostic_metrics", [])
    if isinstance(diagnostics, str):
        diagnostics = [diagnostics]
    return str(name) in {str(item) for item in diagnostics}


@torch.no_grad()
def evaluate_mpjpe_only(
    *,
    model: PTPoseModel,
    dataloader: DataLoader,
    criterion: PoseMPJPELoss,
    device: torch.device,
    runtime,
    desc: str,
    eval_cfg: dict | None = None,
) -> dict[str, Any]:
    model.eval()
    eval_cfg = eval_cfg or {}
    use_action_groups = bool(eval_cfg.get("m4human_action_groups", False))
    primary_metric = str(eval_cfg.get("primary_metric", "")).strip().lower()
    compute_absolute_alias = primary_metric == "absolute_mpjpe" or _needs_metric(eval_cfg, "absolute_mpjpe")
    compute_root_diagnostics = _needs_metric(eval_cfg, "root_error") or _needs_metric(eval_cfg, "root_aligned_mpjpe")
    compute_p = bool(eval_cfg.get("compute_p_mpjpe", eval_cfg.get("p_mpjpe", False))) or _needs_metric(eval_cfg, "p_mpjpe")
    root_index = int(eval_cfg.get("root_index", 0))
    totals = {
        "total_loss": 0.0,
        "pose_loss": 0.0,
        "mean_joint_error_m": 0.0,
    }
    sample_count = 0
    mpjpe_sum_m = 0.0
    root_error_sum_m = 0.0
    root_aligned_mpjpe_sum_m = 0.0
    p_mpjpe_sum_m = 0.0
    action_mpjpe_sum = torch.zeros(50, device=device, dtype=torch.float64)
    action_count = torch.zeros(50, device=device, dtype=torch.float64)
    joint_error_sum_m: torch.Tensor | None = None
    root_aligned_joint_error_sum_m: torch.Tensor | None = None
    action_joint_metric_names = ["mpjpe"]
    if compute_root_diagnostics:
        action_joint_metric_names.append("root_aligned_mpjpe")
    action_scalar_metric_names = ["absolute_mpjpe"]
    if compute_root_diagnostics:
        action_scalar_metric_names.extend(["root_error", "root_aligned_mpjpe"])
    if compute_p:
        action_scalar_metric_names.append("p_mpjpe")
    action_scalar_acc = (
        M4HumanActionGroupAccumulator(action_scalar_metric_names, device=device)
        if use_action_groups
        else None
    )
    action_joint_acc = (
        M4HumanActionGroupVectorAccumulator(action_joint_metric_names, device=device)
        if use_action_groups
        else None
    )
    data_time_sum = 0.0
    step_time_sum = 0.0
    t0 = time.perf_counter()
    loop_end_t = time.perf_counter()
    for batch in tqdm(dataloader, desc=desc):
        data_time_sum += time.perf_counter() - loop_end_t
        step_t0 = time.perf_counter()
        mmwave_points = batch["mmwave_points"].to(device, non_blocking=True)
        target_poses = batch["target_poses"].to(device, non_blocking=True).float()
        bsz = int(mmwave_points.shape[0])
        with autocast_context(runtime):
            outputs = model(mmwave_points)
        pred_pose = outputs["pred_pose"].float()
        loss_dict = criterion(pred_pose, target_poses)
        for key in totals:
            totals[key] += float(loss_dict[key].item()) * bsz
        per_joint_errors_m = torch.norm(pred_pose.float() - target_poses.float(), dim=-1)
        if joint_error_sum_m is None:
            joint_error_sum_m = torch.zeros(int(per_joint_errors_m.shape[1]), device=device, dtype=torch.float64)
        joint_error_sum_m += per_joint_errors_m.double().sum(dim=0)
        per_sample_mpjpe_m = _per_sample_mpjpe(pred_pose, target_poses)
        mpjpe_sum_m += float(per_sample_mpjpe_m.sum().item())
        root_aligned_joint_errors_m = None
        if compute_root_diagnostics:
            pred_root = pred_pose[:, root_index]
            target_root = target_poses[:, root_index]
            pred_root_aligned = _root_aligned(pred_pose, root_index)
            target_root_aligned = _root_aligned(target_poses, root_index)
            root_error = torch.norm(pred_root - target_root, dim=-1)
            root_aligned_joint_errors_m = torch.norm(
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
            root_aligned_mpjpe = root_aligned_joint_errors_m.mean(dim=1)
            root_error_sum_m += float(root_error.sum().item())
            root_aligned_mpjpe_sum_m += float(root_aligned_mpjpe.sum().item())
        if compute_p:
            p_values = compute_per_sample_p_mpjpe(pred_pose, target_poses)
            p_mpjpe_sum_m += float(p_values.sum())
        if use_action_groups and "indicator" in batch:
            indicator = batch["indicator"]
            action_ids_1based = m4human_action_ids_from_indicator(indicator, device=device)
            action_ids = action_ids_1based - 1
            valid = (action_ids >= 0) & (action_ids < 50)
            if bool(valid.any()):
                ids = action_ids[valid]
                action_mpjpe_sum.index_add_(0, ids, per_sample_mpjpe_m[valid].double())
                action_count.index_add_(0, ids, torch.ones_like(per_sample_mpjpe_m[valid], dtype=torch.float64))
            if action_joint_acc is not None:
                action_joint_acc.update("mpjpe", action_ids_1based, per_joint_errors_m)
                if root_aligned_joint_errors_m is not None:
                    action_joint_acc.update("root_aligned_mpjpe", action_ids_1based, root_aligned_joint_errors_m)
            if action_scalar_acc is not None:
                action_scalar_acc.update("absolute_mpjpe", action_ids_1based, per_sample_mpjpe_m)
                if compute_root_diagnostics:
                    action_scalar_acc.update("root_error", action_ids_1based, root_error)
                    action_scalar_acc.update("root_aligned_mpjpe", action_ids_1based, root_aligned_mpjpe)
                if compute_p:
                    action_scalar_acc.update(
                        "p_mpjpe",
                        action_ids_1based,
                        torch.as_tensor(p_values, device=device, dtype=torch.float32),
                    )
        sample_count += bsz
        step_time_sum += time.perf_counter() - step_t0
        loop_end_t = time.perf_counter()
    if sample_count <= 0:
        raise RuntimeError(f"Empty {desc.lower()} dataloader")
    metrics = {key: value / sample_count for key, value in totals.items()}
    metrics["mpjpe_mm"] = float(mpjpe_sum_m / sample_count * 1000.0)
    if joint_error_sum_m is not None:
        metrics["mpjpe_by_joint_mm"] = [
            float(item) for item in (joint_error_sum_m / float(sample_count) * 1000.0).detach().cpu().tolist()
        ]
    if compute_absolute_alias:
        metrics["absolute_mpjpe"] = float(mpjpe_sum_m / sample_count)
        metrics["absolute_mpjpe_mm"] = float(metrics["absolute_mpjpe"] * 1000.0)
    if compute_root_diagnostics:
        metrics["root_error"] = float(root_error_sum_m / sample_count)
        metrics["root_error_mm"] = float(metrics["root_error"] * 1000.0)
        metrics["root_aligned_mpjpe"] = float(root_aligned_mpjpe_sum_m / sample_count)
        metrics["root_aligned_mpjpe_mm"] = float(metrics["root_aligned_mpjpe"] * 1000.0)
        if root_aligned_joint_error_sum_m is not None:
            metrics["root_aligned_mpjpe_by_joint_mm"] = [
                float(item)
                for item in (root_aligned_joint_error_sum_m / float(sample_count) * 1000.0).detach().cpu().tolist()
            ]
    if compute_p:
        metrics["p_mpjpe"] = float(p_mpjpe_sum_m / sample_count)
        metrics["p_mpjpe_mm"] = float(metrics["p_mpjpe"] * 1000.0)
    metrics["num_samples"] = int(sample_count)
    if use_action_groups:
        by_action_group, sample_counts = summarize_mpjpe_by_action_group(
            action_mpjpe_sum.detach().cpu().tolist(),
            action_count.detach().cpu().tolist(),
        )
        metrics["mpjpe_by_action_group_mm"] = by_action_group
        metrics["num_samples_by_action_group"] = sample_counts
        per_action: dict[str, float | None] = {}
        action_sums_cpu = action_mpjpe_sum.detach().cpu().tolist()
        action_counts_cpu = action_count.detach().cpu().tolist()
        for idx, (value_sum, count) in enumerate(zip(action_sums_cpu, action_counts_cpu, strict=True), start=1):
            per_action[str(idx)] = None if float(count) <= 0 else float(value_sum / count * 1000.0)
        metrics["mpjpe_by_action_mm"] = per_action
        if action_joint_acc is not None:
            joint_summary = action_joint_acc.summary(scale=1000.0)
            if "mpjpe" in joint_summary:
                metrics["mpjpe_by_action_group_by_joint_mm"] = joint_summary["mpjpe"]["by_group"]
            if "root_aligned_mpjpe" in joint_summary:
                metrics["root_aligned_mpjpe_by_action_group_by_joint_mm"] = joint_summary["root_aligned_mpjpe"]["by_group"]
        if action_scalar_acc is not None:
            metrics["m4human_action_groups"] = action_scalar_acc.summary(scale=1000.0)
    metrics["samples_per_sec"] = float(sample_count / max(time.perf_counter() - t0, 1e-6))
    metrics["data_time_sec"] = float(data_time_sum)
    metrics["step_time_sec"] = float(step_time_sum)
    return metrics


def _log_m4human_action_groups(writer: SummaryWriter, split: str, metrics: dict[str, Any], step: int) -> None:
    by_action_group = metrics.get("mpjpe_by_action_group_mm")
    if not isinstance(by_action_group, dict):
        return
    for name, value in by_action_group.items():
        if value is not None:
            writer.add_scalar(f"pt/{split}/mpjpe_mm/{name}", float(value), step)


def _format_m4human_action_groups(metrics: dict[str, Any]) -> str:
    by_action_group = metrics.get("mpjpe_by_action_group_mm")
    if not isinstance(by_action_group, dict):
        return ""
    parts = [f"{name}={value:.2f}" for name, value in by_action_group.items() if value is not None]
    return " | " + ", ".join(parts) if parts else ""


def _evaluate(
    *,
    model: PTPoseModel,
    dataloader: DataLoader,
    criterion: PoseMPJPELoss,
    device: torch.device,
    runtime,
    config: dict,
    desc: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    eval_cfg = config.get("evaluation", {})
    mpjpe_only = bool(eval_cfg.get("mpjpe_only", False)) or dataset_name(config).lower().startswith("m4human")
    if mpjpe_only:
        return evaluate_mpjpe_only(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            device=device,
            runtime=runtime,
            desc=desc,
            eval_cfg=eval_cfg,
        ), {}
    return validate_one_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
        runtime=runtime,
        eval_cfg=eval_cfg,
    )


def _build_eval_loader(
    dataset,
    *,
    training_cfg: dict[str, Any],
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> DataLoader:
    eval_batch_size = int(batch_size or training_cfg.get("eval_batch_size", training_cfg.get("batch_size", 32)))
    workers = int(num_workers if num_workers is not None else training_cfg.get("eval_num_workers", training_cfg.get("num_workers", 4)))
    pin_memory = bool(training_cfg.get("pin_memory", True))
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


def _load_pt_pose_model_for_eval(
    config: dict[str, Any],
    *,
    checkpoint: str | Path,
    device: torch.device,
    runtime,
) -> PTPoseModel:
    model = PTPoseModel(
        point_transformer_config=config["point_transformer"],
        device=str(device),
    ).to(device)
    materialize_lazy_modules(model, config=config, device=device, runtime=runtime)
    ckpt = load_checkpoint(checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    if not isinstance(state, dict):
        raise KeyError(f"Checkpoint {checkpoint} does not contain model_state_dict")
    load_state_dict_shape_safe(model, state, module_name="PTPoseModel Eval", strict=False)
    return model.to(device)


def run_pt_pose_evaluation(
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
) -> dict[str, Any]:
    if eval_split not in {"train", "val", "test"}:
        raise ValueError(f"eval_split must be train/val/test, got {eval_split!r}")
    training_cfg = config.setdefault("training", {})
    eval_cfg = config.setdefault("evaluation", {})
    eval_cfg["mpjpe_only"] = True
    eval_cfg["m4human_action_groups"] = bool(include_action_groups)
    if not bool(config.get("data_format", {}).get("normalized", True)):
        eval_cfg.setdefault("primary_metric", "absolute_mpjpe")
        diagnostics = eval_cfg.get("diagnostic_metrics", [])
        if isinstance(diagnostics, str):
            diagnostics = [diagnostics]
        for name in ("absolute_mpjpe", "root_error", "root_aligned_mpjpe", "p_mpjpe"):
            if name not in diagnostics:
                diagnostics.append(name)
        eval_cfg["diagnostic_metrics"] = diagnostics
        eval_cfg["compute_p_mpjpe"] = True
    elif "primary_metric" not in eval_cfg:
        eval_cfg["primary_metric"] = "mpjpe"

    if max_eval_batches is not None:
        training_cfg[f"max_{eval_split}_batches"] = int(max_eval_batches)
    device_obj = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    runtime = configure_runtime(training_cfg, device_obj, default_amp=True, log_prefix="[PT Eval Runtime]")

    dataset = create_radar_pose_dataset(config, split=eval_split, device="cpu")
    loader = _build_eval_loader(dataset, training_cfg=training_cfg, batch_size=batch_size, num_workers=num_workers)
    if max_eval_batches is not None:
        loader = LimitedDataLoader(loader, int(max_eval_batches))
    model = _load_pt_pose_model_for_eval(config, checkpoint=checkpoint, device=device_obj, runtime=runtime)
    criterion = PoseMPJPELoss().to(device_obj)
    metrics, _ = _evaluate(
        model=model,
        dataloader=loader,
        criterion=criterion,
        device=device_obj,
        runtime=runtime,
        config=config,
        desc=f"Evaluate PT {eval_split}",
    )
    payload = {
        "stage": "hpe_point_transformer",
        "eval_split": eval_split,
        "split_id": config.get("m4human", {}).get("split"),
        "checkpoint": str(checkpoint),
        "config": str(config_path) if config_path else None,
        "precision": runtime.precision,
        "amp_enabled": runtime.amp_enabled,
        "metrics": metrics,
    }
    out_path = Path(output_path).expanduser() if output_path else (
        Path(checkpoint).expanduser().parent.parent
        / f"metrics_{eval_split}_{payload['split_id']}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        "[PT Eval] "
        f"{payload['split_id']}/{eval_split} | "
        f"mpjpe={metrics['mpjpe_mm']:.2f}mm"
    )
    by_group = metrics.get("mpjpe_by_action_group_mm")
    if isinstance(by_group, dict):
        parts = [f"{key}={value:.2f}mm" for key, value in by_group.items() if value is not None]
        if parts:
            print("[PT Eval] Action groups | " + ", ".join(parts))
    print(f"[PT Eval] Wrote {out_path}")
    payload["output_path"] = str(out_path)
    return payload


def run_pt_pose(
    config: dict,
    *,
    config_path: str | Path | None = None,
    resume: str | Path | None = None,
    work_dir: str | Path | None = None,
    device: str = "cuda",
    experiment_name: str = "pt_pose",
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    max_test_batches: int | None = None,
) -> dict[str, Any]:
    set_global_seed(int(config.get("random_seed", config.get("training", {}).get("seed", 42))))
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    runtime = configure_runtime(config.get("training", {}), device_obj, default_amp=True, log_prefix="[PT Runtime]")

    default_name = datetime.now().strftime("pt_pose_%Y%m%d_%H%M%S")
    output_root, checkpoint_dir, log_dir = resolve_run_root(
        config,
        work_dir=work_dir,
        default_name=default_name,
        experiment_name=experiment_name,
    )
    effective_config_path = write_effective_config(config, output_root)
    data_bundle = build_radar_pose_dataloaders(
        config,
        checkpoint_dir=output_root,
        include_test=True,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        max_test_batches=max_test_batches,
    )
    write_run_manifest(
        output_root,
        config_path=effective_config_path,
        checkpoint_path=resume,
        config=config,
        dataset_split=str(data_bundle.split_manifest.get("protocol", data_bundle.split_manifest.get("val_split", "train/val"))),
        coord_transform=config.get("coord_transform"),
        output_paths={"output_root": output_root, "checkpoint_dir": checkpoint_dir, "log_dir": log_dir},
        extra={"stage": "pt_pose", "split_manifest": data_bundle.split_manifest},
    )

    train_cfg = config.get("training", {})

    model = PTPoseModel(
        point_transformer_config=config["point_transformer"],
        device=str(device_obj),
    ).to(device_obj)
    materialize_lazy_modules(model, config=config, device=device_obj, runtime=runtime)

    pretrained_pt = config.get("pretrained", {}).get("point_transformer")
    if pretrained_pt:
        load_point_transformer_checkpoint(model.pt, pretrained_pt, map_location=device_obj, strict=False)

    resume_ckpt = None
    if resume:
        resume_ckpt = load_checkpoint(resume, map_location=device_obj)
        load_state_dict_shape_safe(model, resume_ckpt["model_state_dict"], module_name="PTPoseModel", strict=False)

    model.pt = maybe_compile(model.pt, runtime, module_name="pt_backbone", log_prefix="[PT Runtime]")
    criterion = PoseMPJPELoss().to(device_obj)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    grad_scaler = build_grad_scaler(runtime)

    start_epoch = 0
    global_step = 0
    history: list[dict[str, Any]] = []
    best_val_mpjpe_mm = float("inf")
    final_val_metrics: dict[str, Any] | None = None
    final_test_metrics: dict[str, Any] | None = None
    if resume_ckpt is not None:
        if "optimizer_state_dict" in resume_ckpt:
            try:
                optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            except ValueError as exc:
                print(f"[PT] Skipping incompatible optimizer state from resume checkpoint: {exc}")
        if resume_ckpt.get("amp_state_dict") and grad_scaler.is_enabled():
            grad_scaler.load_state_dict(resume_ckpt["amp_state_dict"])
        start_epoch = int(resume_ckpt.get("epoch", -1)) + 1
        global_step = int(resume_ckpt.get("global_step", 0))
        history = list(resume_ckpt.get("history", []))
        best_val_mpjpe_mm = float(resume_ckpt.get("best_val_mpjpe_mm", float("inf")))
        final_val_metrics = resume_ckpt.get("final_val_metrics")
        final_test_metrics = resume_ckpt.get("final_test_metrics")

    print(f"[PT] Trainable parameters: {safe_param_count(model, trainable_only=True):,}")
    max_epochs = int(train_cfg.get("epochs", 80))
    writer = SummaryWriter(str(log_dir))
    for epoch in range(start_epoch, max_epochs):
        lr = compute_lr_for_epoch(
            epoch=epoch,
            max_epochs=max_epochs,
            base_lr=float(train_cfg.get("lr", 3e-4)),
            scheduler_type=str(train_cfg.get("scheduler", "cosine")).lower(),
            warmup_epochs=int(train_cfg.get("warmup_epochs", 5)),
            min_lr=float(train_cfg.get("min_lr", 1e-6)),
        )
        set_optimizer_lr(optimizer, lr)
        print(f"\n[PT] Epoch {epoch + 1}/{max_epochs} | lr={lr:.6e}")
        train_metrics, global_step = train_one_epoch(
            model=model,
            dataloader=data_bundle.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            grad_scaler=grad_scaler,
            device=device_obj,
            epoch=epoch,
            grad_norm=float(train_cfg.get("grad_norm", 1.0)),
            runtime=runtime,
            grad_accum_steps=int(train_cfg.get("grad_accum_steps", 1)),
            writer=writer,
            log_interval=int(train_cfg.get("log_interval", 20)),
            global_step_start=global_step,
        )
        row: dict[str, Any] = {"epoch": epoch, "lr": lr, "train": train_metrics}
        is_best = False
        if (epoch + 1) % max(1, int(train_cfg.get("val_freq", 1))) == 0:
            val_metrics, _ = _evaluate(
                model=model,
                dataloader=data_bundle.val_loader,
                criterion=criterion,
                device=device_obj,
                runtime=runtime,
                config=config,
                desc="Validate PT",
            )
            row["val"] = val_metrics
            final_val_metrics = val_metrics
            is_best = float(val_metrics["mpjpe_mm"]) < best_val_mpjpe_mm
            best_val_mpjpe_mm = min(best_val_mpjpe_mm, float(val_metrics["mpjpe_mm"]))
            writer.add_scalar("pt/val/mpjpe_mm", float(val_metrics["mpjpe_mm"]), epoch + 1)
            _log_m4human_action_groups(writer, "val", val_metrics, epoch + 1)
            print(f"[PT] Val | mpjpe={val_metrics['mpjpe_mm']:.2f}mm{_format_m4human_action_groups(val_metrics)}")

        history.append(row)
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": module_state_dict(model),
            "point_transformer_state_dict": module_state_dict(model.pt),
            "optimizer_state_dict": optimizer.state_dict(),
            "amp_state_dict": grad_scaler.state_dict() if grad_scaler.is_enabled() else None,
            "best_val_mpjpe_mm": best_val_mpjpe_mm,
            "config": config,
            "history": history,
            "final_val_metrics": final_val_metrics,
            "final_test_metrics": final_test_metrics,
            "split_manifest": data_bundle.split_manifest,
            "output_root": str(output_root),
        }
        save_checkpoint(checkpoint_dir / "latest.pth", checkpoint)
        if is_best:
            save_checkpoint(checkpoint_dir / "best.pth", checkpoint)
            print(f"[PT] Saved best model: {best_val_mpjpe_mm:.2f} mm")
        (output_root / "training_results.json").write_text(
            json.dumps(
                {
                    "best_val_mpjpe_mm": best_val_mpjpe_mm,
                    "epochs_completed": epoch + 1,
                    "history": history,
                    "final_val_metrics": final_val_metrics,
                    "final_test_metrics": final_test_metrics,
                    "precision": runtime.precision,
                    "amp_enabled": runtime.amp_enabled,
                    "output_root": str(output_root),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    best_path = checkpoint_dir / "best.pth"
    if best_path.exists():
        ckpt = load_checkpoint(best_path, map_location=device_obj)
        state = ckpt.get("model_state_dict", ckpt)
        if not isinstance(state, dict):
            raise KeyError(f"Checkpoint {best_path} does not contain model_state_dict")
        load_target = getattr(model, "_orig_mod", model)
        load_state_dict_shape_safe(load_target, state, module_name="PTPoseModel best", strict=False)

    final_val_metrics, _ = _evaluate(
        model=model,
        dataloader=data_bundle.val_loader,
        criterion=criterion,
        device=device_obj,
        runtime=runtime,
        config=config,
        desc="Validate best PT",
    )
    writer.add_scalar("pt/best_val/mpjpe_mm", float(final_val_metrics["mpjpe_mm"]), max_epochs)
    _log_m4human_action_groups(writer, "best_val", final_val_metrics, max_epochs)
    print(f"[PT] Best-val checkpoint | val mpjpe={final_val_metrics['mpjpe_mm']:.2f}mm{_format_m4human_action_groups(final_val_metrics)}")

    if data_bundle.test_loader is not None:
        final_test_metrics, _ = _evaluate(
            model=model,
            dataloader=data_bundle.test_loader,
            criterion=criterion,
            device=device_obj,
            runtime=runtime,
            config=config,
            desc="Test best PT",
        )
        writer.add_scalar("pt/final_test/mpjpe_mm", float(final_test_metrics["mpjpe_mm"]), max_epochs)
        _log_m4human_action_groups(writer, "final_test", final_test_metrics, max_epochs)
        print(f"[PT] Best-val checkpoint | test mpjpe={final_test_metrics['mpjpe_mm']:.2f}mm{_format_m4human_action_groups(final_test_metrics)}")

    (output_root / "training_results.json").write_text(
        json.dumps(
            {
                "best_val_mpjpe_mm": best_val_mpjpe_mm,
                "epochs_completed": max_epochs,
                "history": history,
                "final_val_metrics": final_val_metrics,
                "final_test_metrics": final_test_metrics,
                "final_metrics_note": "final_test_metrics are evaluated once from the validation-selected best checkpoint after training.",
                "precision": runtime.precision,
                "amp_enabled": runtime.amp_enabled,
                "output_root": str(output_root),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    writer.close()
    summary = {
        "stage": "pt_pose",
        "best_val_mpjpe_mm": best_val_mpjpe_mm,
        "final_metrics": {
            "val": final_val_metrics,
            "test": final_test_metrics,
        },
        "final_metrics_note": "Test metrics are evaluated once from the validation-selected best checkpoint after training.",
        "output_root": str(output_root),
        "best_checkpoint": str(checkpoint_dir / "best.pth"),
        "latest_checkpoint": str(checkpoint_dir / "latest.pth"),
    }
    print(f"[PT] Training completed. Best val MPJPE: {best_val_mpjpe_mm:.2f} mm")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PointTransformer pose model")
    parser.add_argument("--config", type=str, required=True, help="YAML config")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--split-id", type=str, default=None, choices=["s1", "s2", "s3"])
    parser.add_argument("--precision", type=str, default=None, choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--compile", dest="compile", action="store_true", default=None)
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    args = parser.parse_args()
    config = apply_pt_cli_overrides(load_config(args.config), args)
    run_pt_pose(
        config,
        config_path=args.config,
        resume=args.resume,
        work_dir=args.work_dir,
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        max_test_batches=args.max_test_batches,
    )


if __name__ == "__main__":
    main()
