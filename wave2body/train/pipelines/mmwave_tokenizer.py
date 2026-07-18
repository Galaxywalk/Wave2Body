# -*- coding: utf-8 -*-
"""Training and export pipeline for self-supervised mmWave tokenizers."""

from __future__ import annotations

import json
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional runtime dependency
    SummaryWriter = None


from ...models.mmwave_tokenizer import create_mmwave_tokenizer
from ...utils.runtime_utils import (
    autocast_context,
    build_grad_scaler,
    configure_runtime,
    mark_compiled_step,
    maybe_compile,
)
from ..checkpoint import load_checkpoint, module_state_dict, save_checkpoint
from ..common import safe_param_count, set_global_seed, write_run_manifest
from ..mmwave_tokenizer_data import build_mmwave_tokenizer_dataloaders, create_mmwave_tokenizer_dataset
from ._paths import resolve_run_root, write_effective_config


DEFAULT_MMWAVE_TOKENIZER_EXPERIMENT = "mmwave_tokenizer"


def _base_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def _move_batch(
    batch: dict[str, Any],
    device: torch.device,
    *,
    use_target_pose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    points = batch["mmwave_points"].to(device, non_blocking=True).float()
    target_abs = batch.get("target_poses_absolute") if use_target_pose else None
    if target_abs is not None:
        target_abs = target_abs.to(device, non_blocking=True).float()
    return points, target_abs


def _add_weighted_metric(acc: dict[str, Any], key: str, value: Any, batch_size: int) -> None:
    if torch.is_tensor(value):
        weighted = value.detach().float() * int(batch_size)
        acc[key] = acc[key] + weighted if key in acc else weighted
    elif isinstance(value, (int, float)):
        acc[key] = acc.get(key, 0.0) + float(value) * int(batch_size)


def _accumulate(acc: dict[str, Any], outputs: dict[str, Any], batch_size: int) -> None:
    for key, value in outputs.get("losses", {}).items():
        _add_weighted_metric(acc, key, value, batch_size)
    for key, value in outputs.get("stats", {}).items():
        _add_weighted_metric(acc, f"stat_{key}", value, batch_size)


def _finalize(acc: dict[str, Any], sample_count: int) -> dict[str, float]:
    denom = max(1, int(sample_count))
    out = {}
    for key, value in sorted(acc.items()):
        if torch.is_tensor(value):
            out[key] = float(value.detach().cpu().item()) / denom
        else:
            out[key] = float(value) / denom
    out["num_samples"] = int(sample_count)
    return out


def _loader_perf_config(dataloader) -> dict[str, Any]:
    return {
        "batch_size": getattr(dataloader, "batch_size", None),
        "num_workers": int(getattr(dataloader, "num_workers", 0) or 0),
        "pin_memory": bool(getattr(dataloader, "pin_memory", False)),
        "persistent_workers": bool(getattr(dataloader, "persistent_workers", False)),
        "prefetch_factor": getattr(dataloader, "prefetch_factor", None),
    }


def _write_tensorboard_epoch(
    writer,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    *,
    namespace: str,
) -> None:
    if writer is None:
        return
    for split, metrics in (("train", train_metrics), ("val", val_metrics)):
        for key, value in metrics.items():
            if key == "num_samples":
                continue
            writer.add_scalar(f"{namespace}/{split}/{key}", float(value), int(epoch))
    writer.add_scalar(f"{namespace}/best/val_loss_candidate", float(val_metrics.get("total_loss", float("nan"))), int(epoch))
    writer.flush()


def _run_epoch(
    *,
    model: torch.nn.Module,
    dataloader,
    optimizer: optim.Optimizer | None,
    scaler,
    runtime,
    device: torch.device,
    grad_norm: float | None,
    desc: str,
    sync_timing: bool = False,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(mode=train)
    acc: dict[str, Any] = {}
    sample_count = 0
    data_wait_sec = 0.0
    step_compute_sec = 0.0
    step_count = 0
    log_every = max(1, int(getattr(runtime, "log_every", 25)))
    iterator = tqdm(dataloader, desc=desc)
    fetch_start = time.perf_counter()
    for batch_idx, batch in enumerate(iterator):
        batch_ready = time.perf_counter()
        data_wait_sec += batch_ready - fetch_start
        step_start = batch_ready
        mark_compiled_step(runtime)
        points, target_abs = _move_batch(batch, device)
        batch_size = int(points.shape[0])
        if train:
            optimizer.zero_grad(set_to_none=True)
        grad_context = nullcontext() if train else torch.inference_mode()
        with grad_context:
            with autocast_context(runtime):
                outputs = model(points, target_poses_absolute=target_abs)
                loss = outputs["losses"]["total_loss"]
        if train:
            scaler.scale(loss).backward()
            if grad_norm is not None and float(grad_norm) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_norm))
            scaler.step(optimizer)
            scaler.update()
            updater = getattr(_base_model(model), "update_teacher", None)
            if updater is not None:
                updater()
        _accumulate(acc, outputs, batch_size)
        sample_count += batch_size
        if sync_timing and device.type == "cuda":
            torch.cuda.synchronize(device)
        step_compute_sec += time.perf_counter() - step_start
        step_count += 1
        if batch_idx == 0 or (batch_idx + 1) % log_every == 0:
            total_seen_sec = data_wait_sec + step_compute_sec
            wait_fraction = data_wait_sec / max(total_seen_sec, 1e-9)
            iterator.set_postfix(
                loss=f"{float(loss.detach().float().item()):.4f}",
                wait=f"{wait_fraction:.2f}",
            )
        fetch_start = time.perf_counter()
    metrics = _finalize(acc, sample_count)
    total_sec = data_wait_sec + step_compute_sec
    metrics.update(
        {
            "data_wait_sec": float(data_wait_sec),
            "step_compute_sec": float(step_compute_sec),
            "epoch_wall_sec": float(total_sec),
            "data_wait_fraction": float(data_wait_sec / max(total_sec, 1e-9)),
            "samples_per_sec": float(sample_count / max(total_sec, 1e-9)),
            "steps": int(step_count),
            "avg_data_wait_sec": float(data_wait_sec / max(1, step_count)),
            "avg_step_compute_sec": float(step_compute_sec / max(1, step_count)),
        }
    )
    return metrics


def _token_ids_np(outputs: dict[str, Any]) -> np.ndarray:
    token_ids = outputs.get("token_ids")
    if token_ids is None:
        return np.full(outputs["valid_mask"].shape, -1, dtype=np.int64)
    return token_ids.detach().cpu().numpy().astype(np.int64)


def _indicator_np(batch: dict[str, Any]) -> np.ndarray | None:
    indicator = batch.get("indicator")
    if not torch.is_tensor(indicator):
        return None
    return indicator.detach().cpu().numpy()


@torch.no_grad()
def _write_preview(model: torch.nn.Module, dataloader, device: torch.device, output_root: Path) -> Path | None:
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        return None
    model.eval()
    points, target_abs = _move_batch(batch, device)
    outputs = model(points, target_poses_absolute=target_abs)
    path = output_root / "token_preview_val.npz"
    np.savez_compressed(
        path,
        token_ids=_token_ids_np(outputs),
        token_embeddings=outputs["token_embeddings"].detach().cpu().float().numpy(),
        centers=outputs["centers"].detach().cpu().float().numpy(),
        valid_mask=outputs["valid_mask"].detach().cpu().numpy(),
        indicators=_indicator_np(batch),
    )
    return path


def run_mmwave_tokenizer_training(
    config: dict[str, Any],
    *,
    config_path: str | Path | None,
    resume: str | Path | None = None,
    work_dir: str | Path | None = None,
    device: str = "cuda",
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    experiment_name: str = DEFAULT_MMWAVE_TOKENIZER_EXPERIMENT,
    stage_name: str = "mmwave_tokenizer",
    log_prefix: str = "[MMWaveTokenizer Runtime]",
    tensorboard_namespace: str = "mmwave_tokenizer",
) -> dict[str, Any]:
    set_global_seed(int(config.get("random_seed", config.get("training", {}).get("seed", 42))))
    device_obj = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    runtime = configure_runtime(config.get("training", {}), device_obj, default_amp=True, log_prefix=log_prefix)
    default_name = datetime.now().strftime("m4human_mmwave_tokenizer_%Y%m%d_%H%M%S")
    output_root, checkpoint_dir, log_dir = resolve_run_root(
        config,
        work_dir=work_dir,
        default_name=default_name,
        experiment_name=experiment_name,
    )
    tb_dir = output_root / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    config.setdefault("training", {})["log_dir"] = str(tb_dir)
    effective_config_path = write_effective_config(config, output_root)
    data_bundle = build_mmwave_tokenizer_dataloaders(
        config,
        checkpoint_dir=output_root,
        include_test=False,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
    )
    write_run_manifest(
        output_root,
        config_path=effective_config_path,
        checkpoint_path=resume,
        config=config,
        dataset_split=str(data_bundle.split_manifest.get("protocol", data_bundle.split_manifest.get("val_split", "train/val"))),
        output_paths={"output_root": output_root, "checkpoint_dir": checkpoint_dir, "log_dir": log_dir, "tensorboard": tb_dir},
        extra={
            "stage": stage_name,
            "split_manifest": data_bundle.split_manifest,
            "method": config.get("mmwave_tokenizer", {}).get("method"),
            "patch_policy": config.get("mmwave_tokenizer", {}).get("patch_policy"),
            "train_loader": _loader_perf_config(data_bundle.train_loader),
            "val_loader": _loader_perf_config(data_bundle.val_loader),
        },
    )

    tokenizer_cfg = config.get("mmwave_tokenizer", {})
    method = str(tokenizer_cfg.get("method", "mae")).strip().lower()
    if method != "mae":
        raise ValueError(
            "Production mmWave tokenizer training only supports method='mae'. "
            "Historical methods live under wave2body.models.mmwave_tokenizer.baselines."
        )
    patch_policy = str(tokenizer_cfg.get("patch_policy", "spatial_fps_knn")).strip().lower()
    if patch_policy != "spatial_fps_knn":
        raise ValueError(
            "Production mmWave tokenizer training only supports patch_policy='spatial_fps_knn'. "
            "Body-part-assisted grouping belongs to historical baseline experiments."
        )

    model = create_mmwave_tokenizer(config["mmwave_tokenizer"]).to(device_obj)
    model = maybe_compile(model, runtime, module_name="mmwave_tokenizer", log_prefix=log_prefix)
    print(
        f"{log_prefix} "
        f"method={config['mmwave_tokenizer'].get('method')} "
        f"patch_policy={config['mmwave_tokenizer'].get('patch_policy')} "
        f"params={safe_param_count(_base_model(model)):,}"
    )

    train_cfg = config.get("training", {})
    optimizer = optim.AdamW(
        _base_model(model).parameters(),
        lr=float(train_cfg.get("lr", 3.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1.0e-4)),
    )
    scaler = build_grad_scaler(runtime)
    writer = SummaryWriter(str(tb_dir)) if SummaryWriter is not None else None
    if writer is None:
        print(f"{log_prefix} TensorBoard SummaryWriter unavailable; scalar logging disabled.")
    else:
        writer.add_text(f"{tensorboard_namespace}/run_name", str(config.get("training", {}).get("run_name", "")), 0)
        writer.add_text(f"{tensorboard_namespace}/method", str(config.get("mmwave_tokenizer", {}).get("method", "")), 0)
        writer.add_text(
            f"{tensorboard_namespace}/patch_policy",
            str(config.get("mmwave_tokenizer", {}).get("patch_policy", "")),
            0,
        )
    start_epoch = 1
    best_val = float("inf")
    history: list[dict[str, Any]] = []
    try:
        if resume:
            ckpt = load_checkpoint(resume, map_location=device_obj)
            _base_model(model).load_state_dict(ckpt["model_state_dict"], strict=False)
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val = float(ckpt.get("best_val_loss", best_val))

        epochs = int(train_cfg.get("epochs", 1))
        grad_norm = train_cfg.get("grad_norm", None)
        sync_timing = bool(train_cfg.get("sync_timing", False))
        for epoch in range(start_epoch, epochs + 1):
            train_metrics = _run_epoch(
                model=model,
                dataloader=data_bundle.train_loader,
                optimizer=optimizer,
                scaler=scaler,
                runtime=runtime,
                device=device_obj,
                grad_norm=float(grad_norm) if grad_norm is not None else None,
                desc=f"train {epoch}/{epochs}",
                sync_timing=sync_timing,
            )
            val_metrics = _run_epoch(
                model=model,
                dataloader=data_bundle.val_loader,
                optimizer=None,
                scaler=scaler,
                runtime=runtime,
                device=device_obj,
                grad_norm=None,
                desc=f"val {epoch}/{epochs}",
                sync_timing=sync_timing,
            )
            val_loss = float(val_metrics.get("total_loss", float("inf")))
            is_best = val_loss < best_val
            if is_best:
                best_val = val_loss
            _write_tensorboard_epoch(
                writer,
                epoch,
                train_metrics,
                val_metrics,
                namespace=tensorboard_namespace,
            )
            if writer is not None:
                writer.add_scalar(f"{tensorboard_namespace}/best/val_loss", float(best_val), int(epoch))
            payload = {
                "epoch": int(epoch),
                "best_val_loss": float(best_val),
                "config": config,
                "model_state_dict": module_state_dict(_base_model(model)),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
            latest_path = save_checkpoint(checkpoint_dir / "latest.pt", payload)
            if is_best:
                save_checkpoint(checkpoint_dir / "best.pt", payload)
            row = {"epoch": int(epoch), "train": train_metrics, "val": val_metrics, "latest_checkpoint": str(latest_path)}
            history.append(row)
            (output_root / "training_summary.json").write_text(
                json.dumps(
                    {
                        "experiment": experiment_name,
                        "stage": stage_name,
                        "config_path": str(effective_config_path),
                        "checkpoint_dir": str(checkpoint_dir),
                        "tensorboard_dir": str(tb_dir),
                        "history": history,
                        "best_val_loss": float(best_val),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"{log_prefix} epoch={epoch} train_loss={train_metrics.get('total_loss'):.6f} val_loss={val_loss:.6f}")

        preview_path = _write_preview(model, data_bundle.val_loader, device_obj, output_root)
        if preview_path is not None:
            print(f"{log_prefix} wrote token preview: {preview_path}")
    finally:
        if writer is not None:
            writer.close()
    return {"output_root": str(output_root), "checkpoint_dir": str(checkpoint_dir), "best_val_loss": float(best_val)}


@torch.no_grad()
def export_mmwave_tokens(
    config: dict[str, Any],
    *,
    checkpoint: str,
    split: str,
    output_path: str | None,
    max_batches: int | None,
    device: str,
) -> Path:
    if not checkpoint:
        raise ValueError("--checkpoint is required for export")
    device_obj = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    ckpt = load_checkpoint(checkpoint, map_location=device_obj)
    requested_export_no_pose = bool(config.get("evaluation", {}).get("export_no_pose", False))
    if isinstance(ckpt.get("config"), dict):
        config = ckpt["config"]
        if requested_export_no_pose:
            config.setdefault("evaluation", {})["export_no_pose"] = True
            if str(config.get("dataset", {}).get("name", "")).lower() == "m4human":
                config.setdefault("m4human", {})["load_pose"] = False
    export_no_pose = bool(config.get("evaluation", {}).get("export_no_pose", False))
    model = create_mmwave_tokenizer(config["mmwave_tokenizer"]).to(device_obj)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    dataset = create_mmwave_tokenizer_dataset(config, split=split, device="cpu")
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("training", {}).get("batch_size", 32)),
        shuffle=False,
        num_workers=int(config.get("training", {}).get("num_workers", 0)),
        pin_memory=bool(config.get("training", {}).get("pin_memory", False)),
    )
    token_ids, embeddings, centers, masks, indicators = [], [], [], [], []
    for batch_idx, batch in enumerate(tqdm(loader, desc=f"export {split}")):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        points, target_abs = _move_batch(batch, device_obj, use_target_pose=not export_no_pose)
        outputs = model(points, target_poses_absolute=target_abs)
        token_ids.append(_token_ids_np(outputs))
        embeddings.append(outputs["token_embeddings"].detach().cpu().float().numpy())
        centers.append(outputs["centers"].detach().cpu().float().numpy())
        masks.append(outputs["valid_mask"].detach().cpu().numpy())
        indicator = _indicator_np(batch)
        if indicator is not None:
            indicators.append(indicator)

    if not embeddings:
        raise RuntimeError(f"No batches exported for split={split}; check dataset and --max-export-batches")

    out = Path(output_path).expanduser() if output_path else Path(checkpoint).expanduser().parent / f"{split}_mmwave_tokens.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        token_ids=np.concatenate(token_ids, axis=0),
        token_embeddings=np.concatenate(embeddings, axis=0),
        centers=np.concatenate(centers, axis=0),
        valid_mask=np.concatenate(masks, axis=0),
        indicators=np.concatenate(indicators, axis=0) if indicators else None,
        method=str(config.get("mmwave_tokenizer", {}).get("method")),
        patch_policy=str(config.get("mmwave_tokenizer", {}).get("patch_policy")),
        checkpoint=str(checkpoint),
        export_no_pose=export_no_pose,
    )
    print(f"[MMWaveTokenizer Export] exported tokens: {out}")
    return out
