#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Human Pose Tokenizer Training Script

Train the human pose tokenizer on joint pose data (AMASS, MMBody, M4Human, or combined).

Usage:
    python -m wave2body.train.human_pose_tokenizer --config wave2body/configs/human_pose_tokenizer/stage_a_part_k96_amass_mmbody_m4human.yaml --work-dir /tmp/human_pose_tokenizer_stage_a
"""

import argparse
import inspect
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from wave2body.models.human_body_tokenizer import create_joint_tokenizer, JointVQVAELoss
from wave2body.dataset.human_body import create_dataset
from wave2body.train.common import load_config, set_global_seed
from wave2body.train.joint_batch_loader import JointTensorBatchLoader, pack_joints_dataset
from wave2body.utils.runtime_utils import (
    autocast_context,
    build_grad_scaler,
    configure_runtime,
    dataset_cache_mode,
    maybe_compile,
)


def get_quantizer_metrics(model) -> dict:
    target = model.module if hasattr(model, 'module') else model
    if hasattr(target, 'get_quantizer_metrics'):
        return target.get_quantizer_metrics()
    return {}


def save_checkpoint(model, optimizer, epoch, loss, path, scheduler=None, grad_scaler=None, best_val_loss=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if best_val_loss is not None:
        checkpoint['best_val_loss'] = float(best_val_loss)
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    if grad_scaler is not None and hasattr(grad_scaler, 'state_dict'):
        checkpoint['grad_scaler_state_dict'] = grad_scaler.state_dict()
    torch.save(checkpoint, path)


def resolve_enabled_datasets(config: dict) -> list[dict]:
    """Resolve dataset config entries for the trainer and cache-prep tooling."""
    data_config = config.get('data', {})
    enabled_datasets = list(data_config.get('dataset_configs') or [])
    if not enabled_datasets:
        raise ValueError("Human pose tokenizer config must define data.dataset_configs.")

    requested_names = data_config.get('datasets') or []
    if requested_names:
        requested = set(requested_names)
        enabled_datasets = [ds for ds in enabled_datasets if ds.get('name') in requested]
    return enabled_datasets


def build_adamw_optimizer(parameters, optimizer_cfg: dict, device: torch.device):
    """Build AdamW with optional fused/foreach acceleration and safe fallback."""
    cfg = dict(optimizer_cfg)
    requested_fused = cfg.pop('fused', None)
    requested_foreach = cfg.pop('foreach', None)
    supported = inspect.signature(optim.AdamW).parameters

    if requested_fused is not None:
        if 'fused' not in supported:
            print("[Optimizer] AdamW fused flag is not supported by this PyTorch; using default AdamW.")
        elif bool(requested_fused) and device.type != 'cuda':
            print("[Optimizer] AdamW fused=true requires CUDA; using default AdamW.")
        else:
            cfg['fused'] = bool(requested_fused)

    if requested_foreach is not None:
        if 'foreach' not in supported:
            print("[Optimizer] AdamW foreach flag is not supported by this PyTorch; ignoring it.")
        elif cfg.get('fused') is not None:
            print("[Optimizer] AdamW fused is enabled; ignoring foreach to avoid incompatible modes.")
        else:
            cfg['foreach'] = bool(requested_foreach)

    try:
        optimizer = optim.AdamW(parameters, **cfg)
    except (RuntimeError, TypeError) as exc:
        if 'fused' not in cfg and 'foreach' not in cfg:
            raise
        print(f"[Optimizer] Accelerated AdamW init failed ({exc}); retrying without fused/foreach.")
        cfg.pop('fused', None)
        cfg.pop('foreach', None)
        optimizer = optim.AdamW(parameters, **cfg)

    mode = (
        'fused' if cfg.get('fused') else
        'foreach' if cfg.get('foreach') else
        'default'
    )
    print(f"[Optimizer] AdamW mode={mode}")
    return optimizer


def _log_interval(value, default: int) -> int:
    interval = default if value is None else int(value)
    return max(0, interval)


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    epoch,
    writer,
    runtime,
    grad_scaler,
    grad_accum_steps=1,
    log_interval=1,
    quantizer_log_interval=1,
    max_batches=None,
):
    model.train()
    losses, recon_losses, commit_losses = [], [], []
    sample_count = 0
    optimizer_steps = 0
    t0 = time.perf_counter()
    data_time_sum = 0.0
    step_time_sum = 0.0
    grad_accum_steps = max(1, int(grad_accum_steps))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    show_progress = sys.stderr.isatty()
    progress_bar = tqdm(loader, desc=f'Epoch {epoch}', disable=not show_progress, mininterval=0.5)
    optimizer.zero_grad(set_to_none=True)
    loop_end_t = time.perf_counter()
    for batch_idx, (joints, _) in enumerate(progress_bar):
        if max_batches is not None and batch_idx >= max_batches:
            break
        data_time_sum += time.perf_counter() - loop_end_t
        step_t0 = time.perf_counter()

        joints = joints.to(device, non_blocking=True)
        sample_count += int(joints.shape[0])
        joints_visible = torch.ones(joints.shape[:2], device=device, dtype=torch.bool)

        with autocast_context(runtime):
            recovered, _, e_loss = model(joints=joints, joints_visible=joints_visible, train=True)
            loss_dict = criterion(recovered, joints, e_loss)

        loss = loss_dict['total_loss']
        loss_to_backprop = loss / float(grad_accum_steps)
        if grad_scaler.is_enabled():
            grad_scaler.scale(loss_to_backprop).backward()
        else:
            loss_to_backprop.backward()

        reached_accum_boundary = ((batch_idx + 1) % grad_accum_steps == 0)
        is_last_batch = ((batch_idx + 1) >= len(loader)) or (
            max_batches is not None and (batch_idx + 1) >= int(max_batches)
        )
        if reached_accum_boundary or is_last_batch:
            if grad_scaler.is_enabled():
                grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if grad_scaler.is_enabled():
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        losses.append(loss.item())
        recon_losses.append(loss_dict['recon_loss'].item())
        if 'commitment_loss' in loss_dict:
            commit_losses.append(loss_dict['commitment_loss'].item())

        if show_progress:
            progress_bar.set_postfix({
                'Loss': f'{np.mean(losses):.6f}',
                'Recon': f'{np.mean(recon_losses):.6f}',
            })

        if writer and log_interval > 0 and (batch_idx % log_interval == 0):
            global_step = epoch * len(loader) + batch_idx
            for k, v in loss_dict.items():
                writer.add_scalar(f'Train/{k}', v.item(), global_step)
        if writer and quantizer_log_interval > 0 and (batch_idx % quantizer_log_interval == 0):
            global_step = epoch * len(loader) + batch_idx
            quantizer_metrics = get_quantizer_metrics(model)
            for k, v in quantizer_metrics.items():
                writer.add_scalar(f'Train/quantizer/{k}', v, global_step)
        step_time_sum += time.perf_counter() - step_t0
        loop_end_t = time.perf_counter()

    elapsed = max(time.perf_counter() - t0, 1e-6)
    max_memory_mb = 0.0
    if device.type == "cuda":
        max_memory_mb = float(torch.cuda.max_memory_allocated(device=device) / (1024.0 ** 2))
    return {
        'total_loss': np.mean(losses),
        'recon_loss': np.mean(recon_losses),
        'commitment_loss': np.mean(commit_losses) if commit_losses else 0.0,
        'num_samples': sample_count,
        'samples_per_sec': sample_count / elapsed,
        'elapsed_sec': elapsed,
        'data_time_sec': data_time_sum,
        'step_time_sec': step_time_sum,
        'optimizer_steps': optimizer_steps,
        'grad_accum_steps': grad_accum_steps,
        'max_memory_mb': max_memory_mb,
    }


def validate_epoch(
    model,
    loader,
    criterion,
    device,
    runtime,
    max_batches=None,
    quantizer_metric_interval=1,
):
    model.eval()
    losses, recon_losses, commit_losses = [], [], []
    quantizer_metrics_accum = {}
    quantizer_metric_steps = 0
    sample_count = 0
    t0 = time.perf_counter()
    data_time_sum = 0.0
    step_time_sum = 0.0
    loop_end_t = time.perf_counter()

    with torch.inference_mode():
        for batch_idx, (joints, _) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            data_time_sum += time.perf_counter() - loop_end_t
            step_t0 = time.perf_counter()
            joints = joints.to(device, non_blocking=True)
            sample_count += int(joints.shape[0])
            joints_visible = torch.ones(joints.shape[:2], device=device, dtype=torch.bool)
            with autocast_context(runtime):
                recovered, _, e_loss = model(joints=joints, joints_visible=joints_visible, train=False)
                loss_dict = criterion(recovered, joints, e_loss)

            losses.append(loss_dict['total_loss'].item())
            recon_losses.append(loss_dict['recon_loss'].item())
            if 'commitment_loss' in loss_dict:
                commit_losses.append(loss_dict['commitment_loss'].item())
            if quantizer_metric_interval > 0 and (batch_idx % quantizer_metric_interval == 0):
                quantizer_metrics = get_quantizer_metrics(model)
                if quantizer_metrics:
                    quantizer_metric_steps += 1
                    for k, v in quantizer_metrics.items():
                        quantizer_metrics_accum[k] = quantizer_metrics_accum.get(k, 0.0) + float(v)
            step_time_sum += time.perf_counter() - step_t0
            loop_end_t = time.perf_counter()

    elapsed = max(time.perf_counter() - t0, 1e-6)
    metrics = {
        'total_loss': np.mean(losses),
        'recon_loss': np.mean(recon_losses),
        'commitment_loss': np.mean(commit_losses) if commit_losses else 0.0,
        'num_samples': sample_count,
        'samples_per_sec': sample_count / elapsed,
        'elapsed_sec': elapsed,
        'data_time_sec': data_time_sum,
        'step_time_sec': step_time_sum,
    }
    if quantizer_metric_steps > 0:
        for k, v in quantizer_metrics_accum.items():
            metrics[f'quantizer/{k}'] = v / quantizer_metric_steps
    return metrics


def main(argv: list[str] | None = None) -> dict | None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--work-dir', default='./work_dirs/vqvae', help='Working directory')
    parser.add_argument('--resume', help='Resume training checkpoint')
    parser.add_argument(
        '--resume-model-only',
        action='store_true',
        help='Load model weights from --resume but reset optimizer, scheduler, and epoch.',
    )
    parser.add_argument('--gpus', help='Specify GPUs, e.g. "0,1"')
    parser.add_argument('--epochs', type=int, help='Override epochs')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--num-workers', type=int, help='Override dataloader num_workers')
    parser.add_argument(
        '--batch-loader',
        choices=['dataloader', 'tensor'],
        help='Override data.batch_loader. tensor batches packed joint caches directly.',
    )
    parser.add_argument('--max-dataset-samples', type=int, help='Limit samples per configured dataset before cache build')
    parser.add_argument('--log-interval', type=int, help='Override train TensorBoard scalar log interval in batches')
    parser.add_argument('--dataset-device', choices=['cpu', 'cuda'], help='Dataset tensor device')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic cudnn (slower)')
    parser.add_argument('--max-train-batches', type=int, help='Limit train batches per epoch (smoke test)')
    parser.add_argument('--max-val-batches', type=int, help='Limit val batches per epoch (smoke test)')
    args = parser.parse_args(argv)

    config = load_config(args.config)

    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.num_workers is not None:
        config['training']['num_workers'] = args.num_workers
    if args.batch_loader is not None:
        config.setdefault('data', {})['batch_loader'] = args.batch_loader
    if args.max_dataset_samples is not None:
        if int(args.max_dataset_samples) <= 0:
            raise ValueError('--max-dataset-samples must be positive')
        for dataset_cfg in config.setdefault('data', {}).get('dataset_configs', []):
            dataset_cfg.setdefault('params', {})['max_samples'] = int(args.max_dataset_samples)
    if args.log_interval is not None:
        config['training']['log_interval'] = args.log_interval
    if args.dataset_device:
        config.setdefault('data', {})['dataset_device'] = args.dataset_device
    if args.deterministic:
        config['training']['deterministic'] = True

    deterministic = bool(config.get('training', {}).get('deterministic', False))
    set_global_seed(config.get('seed', 42), deterministic=deterministic)

    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    runtime = configure_runtime(
        config.get('training', {}),
        device,
        default_amp=True,
        log_prefix='[VQ Runtime]',
    )

    # Get datasets from config.
    data_config = config.get('data', {})
    enabled_datasets = resolve_enabled_datasets(config)

    if not enabled_datasets:
        raise ValueError("No datasets enabled. Check config.")

    print("\n--- Dataset List ---")
    for ds in enabled_datasets:
        print(f"  {ds['name']}: {ds['path']}")

    # Create model
    model = create_joint_tokenizer(config['model']).float().to(device)
    if runtime.compile_enabled and torch.cuda.device_count() <= 1:
        model = maybe_compile(model, runtime, module_name='vqvae', log_prefix='[VQ Runtime]')
    elif runtime.compile_enabled and torch.cuda.device_count() > 1:
        print("[VQ Runtime] compile is enabled but DataParallel is active; skip compile.")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Create dataset
    dataset_device = config.get('data', {}).get('dataset_device', 'cuda' if torch.cuda.is_available() else 'cpu')
    if dataset_device == 'cuda' and not torch.cuda.is_available():
        print("[WARN] dataset_device=cuda but CUDA is unavailable, fallback to cpu")
        dataset_device = 'cpu'

    train_dataset, _ = create_dataset(
        enabled_datasets,
        config['model']['num_joints'],
        dataset_device,
        use_cache=bool(data_config.get('use_cache', True)),
        cache_dir=data_config.get('cache_dir'),
        cache_build_workers=data_config.get('cache_build_workers'),
        cache_build_chunk_size=data_config.get('cache_build_chunk_size'),
    )
    num_workers = int(config.get('training', {}).get('num_workers', 0))
    pin_memory = bool(config.get('training', {}).get('pin_memory', True))
    persistent_workers = bool(config.get('training', {}).get('persistent_workers', num_workers > 0))
    prefetch_factor = int(config.get('training', {}).get('prefetch_factor', 2))
    batch_size = int(config['training']['batch_size'])
    train_size = int(config['data']['train_ratio'] * len(train_dataset))
    val_size = len(train_dataset) - train_size
    batch_loader_mode = str(data_config.get("batch_loader", "dataloader")).strip().lower()

    cache_mode = "none"
    if batch_loader_mode in {"tensor", "tensor_batch", "joint_tensor"}:
        packed_dataset = pack_joints_dataset(train_dataset)
        if packed_dataset is None:
            print("[WARN] data.batch_loader=tensor requested but dataset is not CPU joint tensors; using DataLoader.")
            batch_loader_mode = "dataloader"
        else:
            split_indices = torch.randperm(len(packed_dataset))
            train_indices = split_indices[:train_size]
            val_indices = split_indices[train_size:]
            train_loader = JointTensorBatchLoader(
                packed_dataset,
                train_indices,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=pin_memory,
            )
            val_loader = JointTensorBatchLoader(
                packed_dataset,
                val_indices,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=pin_memory,
            )
            cache_mode = "joint_tensor_batch"
            num_workers = 0
            persistent_workers = False
            print(
                f"[Data] Packed joint tensor loader enabled: "
                f"samples={len(packed_dataset):,}, nbytes={packed_dataset.nbytes() / (1024 ** 3):.2f} GiB"
            )

    if batch_loader_mode not in {"tensor", "tensor_batch", "joint_tensor"}:
        train_ds, val_ds = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        cache_mode = dataset_cache_mode(train_ds)
        if cache_mode == 'gpu_tensor' and num_workers > 0:
            print("[WARN] data.cache.mode=gpu_tensor requires num_workers=0, forcing num_workers=0")
            num_workers = 0
            persistent_workers = False
        if dataset_device == 'cuda' and num_workers > 0:
            print("[WARN] dataset_device=cuda with num_workers>0 is not supported, forcing num_workers=0")
            num_workers = 0
            persistent_workers = False
        if dataset_device == 'cuda' or cache_mode == 'gpu_tensor':
            pin_memory = False
            persistent_workers = False

        train_loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(persistent_workers and num_workers > 0),
        )
        val_loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(persistent_workers and num_workers > 0),
        )
        if num_workers > 0:
            train_loader_kwargs['prefetch_factor'] = prefetch_factor
            val_loader_kwargs['prefetch_factor'] = prefetch_factor

        train_loader = DataLoader(train_ds, **train_loader_kwargs)
        val_loader = DataLoader(val_ds, **val_loader_kwargs)

    print(
        f"[Perf] deterministic={runtime.deterministic}, tf32={runtime.tf32}, "
        f"matmul_precision={runtime.matmul_precision}, dataset_device={dataset_device}, "
        f"cache_mode={cache_mode}, batch_loader={batch_loader_mode}, "
        f"num_workers={num_workers}, pin_memory={pin_memory}, batch_size={batch_size}"
    )

    criterion = JointVQVAELoss(**config['loss'])
    optimizer = build_adamw_optimizer(model.parameters(), config['training']['optimizer'], device)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **config['training']['scheduler'])
    grad_scaler = build_grad_scaler(runtime)
    grad_accum_steps = int(config.get('training', {}).get('grad_accum_steps', 1))
    if grad_accum_steps < 1:
        raise ValueError(f"training.grad_accum_steps must be >=1, got {grad_accum_steps}")
    log_interval = _log_interval(config.get('training', {}).get('log_interval'), 1)
    quantizer_log_interval = _log_interval(
        config.get('training', {}).get('quantizer_log_interval'),
        log_interval if log_interval > 0 else 1,
    )
    val_quantizer_log_interval = _log_interval(
        config.get('training', {}).get('val_quantizer_log_interval'),
        quantizer_log_interval if quantizer_log_interval > 0 else 1,
    )

    # Setup work dir
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    with open(work_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    writer = SummaryWriter(work_dir / 'tensorboard')

    # Training loop
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        if args.resume_model_only:
            print(f"Loaded model weights from {args.resume}; optimizer/scheduler reset.")
        else:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if 'grad_scaler_state_dict' in ckpt and hasattr(grad_scaler, 'load_state_dict'):
                grad_scaler.load_state_dict(ckpt['grad_scaler_state_dict'])
            best_val_loss = float(ckpt.get('best_val_loss', ckpt.get('loss', best_val_loss)))
            start_epoch = ckpt['epoch'] + 1
            print(f"Resumed from epoch {start_epoch} (best_val_loss={best_val_loss:.6f})")

    print(f"\nStarting training for {config['training']['epochs']} epochs...")

    for epoch in range(start_epoch, config['training']['epochs']):
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            writer,
            runtime,
            grad_scaler,
            grad_accum_steps=grad_accum_steps,
            log_interval=log_interval,
            quantizer_log_interval=quantizer_log_interval,
            max_batches=args.max_train_batches,
        )
        val_metrics = validate_epoch(
            model,
            val_loader,
            criterion,
            device,
            runtime,
            max_batches=args.max_val_batches,
            quantizer_metric_interval=val_quantizer_log_interval,
        )
        scheduler.step()

        print(f'Epoch {epoch:03d}: '
              f'Train Loss={train_metrics["total_loss"]:.6f} (Recon={train_metrics["recon_loss"]:.6f}) | '
              f'Val Loss={val_metrics["total_loss"]:.6f} (Recon={val_metrics["recon_loss"]:.6f})')
        print(
            f'  Throughput: train={train_metrics["samples_per_sec"]:.1f} samp/s '
            f'({train_metrics["num_samples"]} samples, {train_metrics["elapsed_sec"]:.1f}s), '
            f'val={val_metrics["samples_per_sec"]:.1f} samp/s '
            f'({val_metrics["num_samples"]} samples, {val_metrics["elapsed_sec"]:.1f}s)'
        )
        print(
            f'  Timings: train(data={train_metrics["data_time_sec"]:.1f}s, step={train_metrics["step_time_sec"]:.1f}s, '
            f'optim_steps={train_metrics["optimizer_steps"]}, accum={train_metrics["grad_accum_steps"]}), '
            f'val(data={val_metrics["data_time_sec"]:.1f}s, step={val_metrics["step_time_sec"]:.1f}s)'
        )
        if "max_memory_mb" in train_metrics and train_metrics["max_memory_mb"] > 0:
            print(f'  Memory: train max_alloc={train_metrics["max_memory_mb"]:.1f} MiB')

        for k, v in val_metrics.items():
            writer.add_scalar(f'Val/{k}', v, epoch)

        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_metrics['total_loss'],
                work_dir / 'best_model.pth',
                scheduler,
                grad_scaler,
                best_val_loss,
            )
            print(f'  -> New best model saved!')

        if epoch % config['training']['save_interval'] == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_metrics['total_loss'],
                work_dir / f'checkpoint_epoch_{epoch:03d}.pth',
                scheduler,
                grad_scaler,
                best_val_loss,
            )

    save_checkpoint(model, optimizer, config['training']['epochs']-1, val_metrics['total_loss'],
                   work_dir / 'final_model.pth', scheduler, grad_scaler, best_val_loss)

    # Save training results
    results = {
        'timestamp': datetime.now().isoformat(),
        'epochs': config['training']['epochs'],
        'best_val_loss': best_val_loss,
        'final_val_loss': val_metrics['total_loss'],
        'final_train_samples_per_sec': train_metrics.get('samples_per_sec'),
        'final_val_samples_per_sec': val_metrics.get('samples_per_sec'),
        'final_train_elapsed_sec': train_metrics.get('elapsed_sec'),
        'final_val_elapsed_sec': val_metrics.get('elapsed_sec'),
        'final_train_data_time_sec': train_metrics.get('data_time_sec'),
        'final_train_step_time_sec': train_metrics.get('step_time_sec'),
        'final_train_max_memory_mb': train_metrics.get('max_memory_mb'),
        'grad_accum_steps': grad_accum_steps,
        'log_interval': log_interval,
        'quantizer_log_interval': quantizer_log_interval,
        'val_quantizer_log_interval': val_quantizer_log_interval,
        'precision': runtime.precision,
        'amp_enabled': runtime.amp_enabled,
        'smoke_max_train_batches': args.max_train_batches,
        'smoke_max_val_batches': args.max_val_batches,
    }
    with open(work_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    writer.close()
    print(f"\nTraining completed! Best val loss: {best_val_loss:.6f}")
    return {
        **results,
        'stage': 'vqvae',
        'output_root': str(work_dir),
        'best_checkpoint': str(work_dir / 'best_model.pth'),
        'final_checkpoint': str(work_dir / 'final_model.pth'),
        'latest_checkpoint': str(work_dir / 'final_model.pth'),
    }


if __name__ == "__main__":
    main()
