# -*- coding: utf-8 -*-
"""Generic training pipeline for config-driven pose baselines."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

try:
    from ..checkpoint import load_checkpoint, load_state_dict_shape_safe, module_state_dict, save_checkpoint
    from ..common import safe_param_count, set_global_seed, write_run_manifest
    from ..pose_baseline_utils import train_pose_baseline_one_epoch, validate_pose_baseline_one_epoch
    from ..radar_pose_data import build_radar_pose_dataloaders
    from ...models.baselines import build_pose_baseline
    from ...utils.runtime_utils import build_grad_scaler, configure_runtime, maybe_compile
except ImportError:  # Support scripts that put src/ directly on sys.path.
    from wave2body.models.baselines import build_pose_baseline
    from wave2body.train.checkpoint import load_checkpoint, load_state_dict_shape_safe, module_state_dict, save_checkpoint
    from wave2body.train.common import safe_param_count, set_global_seed, write_run_manifest
    from wave2body.train.pose_baseline_utils import train_pose_baseline_one_epoch, validate_pose_baseline_one_epoch
    from wave2body.train.radar_pose_data import build_radar_pose_dataloaders
    from wave2body.utils.runtime_utils import build_grad_scaler, configure_runtime, maybe_compile

from ._paths import resolve_run_root, write_effective_config


def _optimizer_from_config(parameters, training_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_name = str(training_cfg.get("optimizer", "adamw")).strip().lower()
    lr = float(training_cfg.get("lr", 3.0e-4))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    if optimizer_name == "adam":
        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported pose baseline optimizer: {optimizer_name!r}")


def _primary_metric(metrics: dict[str, Any]) -> tuple[str, float]:
    if "mpjpe_mm" in metrics:
        return "mpjpe_mm", float(metrics["mpjpe_mm"])
    if "total_loss" in metrics:
        return "total_loss", float(metrics["total_loss"])
    raise KeyError("Metrics must contain mpjpe_mm or total_loss")


def run_pose_baseline(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    resume: str | Path | None = None,
    work_dir: str | Path | None = None,
    device: str = "cuda",
    experiment_name: str = "hpe_baselines",
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    max_test_batches: int | None = None,
) -> dict[str, Any]:
    """Train a config-driven pose baseline on a radar-pose dataloader."""

    set_global_seed(int(config.get("random_seed", config.get("training", {}).get("seed", 42))))
    requested_device = str(device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for pose-baseline training, but torch.cuda.is_available() is false. "
            "Check the NVIDIA driver/runtime and CUDA_VISIBLE_DEVICES before launching training."
        )
    device_obj = torch.device(requested_device)
    training_cfg = config.setdefault("training", {})
    runtime = configure_runtime(training_cfg, device_obj, default_amp=True, log_prefix="[PoseBaseline Runtime]")

    baseline_name = str(config.get("baseline", {}).get("name", config.get("model", {}).get("name", "pose_baseline")))
    default_name = datetime.now().strftime(f"{baseline_name}_%Y%m%d_%H%M%S")
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
        extra={"stage": "pose_baseline", "baseline": baseline_name, "split_manifest": data_bundle.split_manifest},
    )

    bundle = build_pose_baseline(config, device=str(device_obj))
    bundle.model.to(device_obj)

    resume_ckpt = None
    if resume:
        resume_ckpt = load_checkpoint(resume, map_location=device_obj)
        state = resume_ckpt.get("model_state_dict", resume_ckpt)
        if not isinstance(state, dict):
            raise KeyError(f"Checkpoint {resume} does not contain a model_state_dict")
        load_state_dict_shape_safe(bundle.model, state, module_name=f"{bundle.name}PoseBaseline", strict=False)

    bundle.model = maybe_compile(bundle.model, runtime, module_name=f"{bundle.name}_baseline", log_prefix="[PoseBaseline Runtime]")
    optimizer = _optimizer_from_config(bundle.model.parameters(), training_cfg)
    grad_scaler = build_grad_scaler(runtime)

    start_epoch = 0
    global_step = 0
    history: list[dict[str, Any]] = []
    best_metric = float("inf")
    best_metric_name = "mpjpe_mm"
    final_val_metrics: dict[str, Any] | None = None
    final_test_metrics: dict[str, Any] | None = None
    if resume_ckpt is not None:
        if "optimizer_state_dict" in resume_ckpt:
            try:
                optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            except ValueError as exc:
                print(f"[PoseBaseline] Skipping incompatible optimizer state: {exc}")
        if resume_ckpt.get("amp_state_dict") and grad_scaler.is_enabled():
            grad_scaler.load_state_dict(resume_ckpt["amp_state_dict"])
        start_epoch = int(resume_ckpt.get("epoch", -1)) + 1
        global_step = int(resume_ckpt.get("global_step", 0))
        history = list(resume_ckpt.get("history", []))
        best_metric = float(resume_ckpt.get("best_metric", float("inf")))
        best_metric_name = str(resume_ckpt.get("best_metric_name", best_metric_name))
        final_val_metrics = resume_ckpt.get("final_val_metrics")
        final_test_metrics = resume_ckpt.get("final_test_metrics")

    print(f"[PoseBaseline] Baseline={bundle.name}")
    print(f"[PoseBaseline] Trainable parameters: {safe_param_count(bundle.model, trainable_only=True):,}")
    writer = SummaryWriter(str(log_dir))
    max_epochs = int(training_cfg.get("epochs", 1))

    for epoch in range(start_epoch, max_epochs):
        print(f"\n[PoseBaseline] Epoch {epoch + 1}/{max_epochs}")
        train_metrics, global_step = train_pose_baseline_one_epoch(
            bundle=bundle,
            dataloader=data_bundle.train_loader,
            optimizer=optimizer,
            grad_scaler=grad_scaler,
            device=device_obj,
            runtime=runtime,
            epoch=epoch,
            grad_norm=float(training_cfg.get("grad_norm", training_cfg.get("gradient_clipping", 1.0))),
            grad_accum_steps=int(training_cfg.get("grad_accum_steps", 1)),
            writer=writer,
            log_interval=int(training_cfg.get("log_interval", 20)),
            global_step_start=global_step,
        )
        row: dict[str, Any] = {"epoch": epoch, "train": train_metrics}

        val_metrics = validate_pose_baseline_one_epoch(
            bundle=bundle,
            dataloader=data_bundle.val_loader,
            device=device_obj,
            runtime=runtime,
            desc=f"Validate {bundle.name}",
        )
        row["val"] = val_metrics
        final_val_metrics = val_metrics
        metric_name, metric_value = _primary_metric(val_metrics)
        best_metric_name = metric_name
        is_best = metric_value < best_metric
        best_metric = min(best_metric, metric_value)
        writer.add_scalar(f"{bundle.name}/val/{metric_name}", metric_value, epoch + 1)
        print(f"[PoseBaseline] Val | {metric_name}={metric_value:.4f}")

        history.append(row)
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "baseline": bundle.name,
            "model_state_dict": module_state_dict(bundle.model),
            "optimizer_state_dict": optimizer.state_dict(),
            "amp_state_dict": grad_scaler.state_dict() if grad_scaler.is_enabled() else None,
            "best_metric": best_metric,
            "best_metric_name": best_metric_name,
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
            print(f"[PoseBaseline] Saved best model: {best_metric_name}={best_metric:.4f}")

        (output_root / "training_results.json").write_text(
            json.dumps(
                {
                    "baseline": bundle.name,
                    "best_metric_name": best_metric_name,
                    "best_metric": best_metric,
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
            raise KeyError(f"Checkpoint {best_path} does not contain a model_state_dict")
        load_target = getattr(bundle.model, "_orig_mod", bundle.model)
        load_state_dict_shape_safe(load_target, state, module_name=f"{bundle.name}PoseBaseline best", strict=False)

    final_val_metrics = validate_pose_baseline_one_epoch(
        bundle=bundle,
        dataloader=data_bundle.val_loader,
        device=device_obj,
        runtime=runtime,
        desc=f"Validate best {bundle.name}",
    )
    final_val_metric_name, final_val_metric_value = _primary_metric(final_val_metrics)
    writer.add_scalar(f"{bundle.name}/best_val/{final_val_metric_name}", final_val_metric_value, max_epochs)
    print(f"[PoseBaseline] Best-val checkpoint | val {final_val_metric_name}={final_val_metric_value:.4f}")

    if data_bundle.test_loader is not None:
        final_test_metrics = validate_pose_baseline_one_epoch(
            bundle=bundle,
            dataloader=data_bundle.test_loader,
            device=device_obj,
            runtime=runtime,
            desc=f"Test best {bundle.name}",
        )
        test_metric_name, test_metric_value = _primary_metric(final_test_metrics)
        writer.add_scalar(f"{bundle.name}/final_test/{test_metric_name}", test_metric_value, max_epochs)
        print(f"[PoseBaseline] Best-val checkpoint | test {test_metric_name}={test_metric_value:.4f}")

    writer.close()
    (output_root / "training_results.json").write_text(
        json.dumps(
            {
                "baseline": bundle.name,
                "best_metric_name": best_metric_name,
                "best_metric": best_metric,
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
    summary = {
        "stage": "pose_baseline",
        "baseline": bundle.name,
        "best_metric_name": best_metric_name,
        "best_metric": best_metric,
        "final_metrics": {"val": final_val_metrics, "test": final_test_metrics},
        "final_metrics_note": "Test metrics are evaluated once from the validation-selected best checkpoint after training.",
        "output_root": str(output_root),
        "best_checkpoint": str(checkpoint_dir / "best.pth"),
        "latest_checkpoint": str(checkpoint_dir / "latest.pth"),
    }
    print(f"[PoseBaseline] Training completed. Best {best_metric_name}: {best_metric:.4f}")
    return summary


__all__ = ["run_pose_baseline"]
