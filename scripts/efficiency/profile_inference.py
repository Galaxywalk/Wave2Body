#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Profile forward-only inference FLOPs and latency for PT and Wave2Body."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.efficiency.profile_training_flops import (  # noqa: E402
    DEFAULT_POINT_TRANSFORMER_CONFIG,
    _component_param_summary,
    _fmt_si,
    _instrument_components,
    _parse_profile_batches,
    _profile_flops_streaming,
)
from wave2body.models.hpe_token.absolute_root import AbsoluteRootHeadEstimatorModel  # noqa: E402
from wave2body.models.point_transformer.pose_baseline import PTPoseModel  # noqa: E402
from wave2body.train.checkpoint import load_point_transformer_checkpoint  # noqa: E402
from wave2body.train.common import load_config, set_global_seed  # noqa: E402
from wave2body.train.pt_pose_utils import materialize_lazy_modules  # noqa: E402
from wave2body.train.radar_pose_data import build_radar_pose_dataloaders  # noqa: E402
from wave2body.utils.paths import artifact_root  # noqa: E402
from wave2body.utils.runtime_utils import autocast_context, configure_runtime  # noqa: E402


DEFAULT_WAVE2BODY_FINAL_CONFIG = (
    REPO_ROOT
    / "wave2body"
    / "configs"
    / "hpe_token"
    / "m4human_s2_k96_queryroot_pointiwmean_absolute_trainable_translator.yaml"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default="both", help="both, point_transformer, wave2body_final, or comma list.")
    parser.add_argument("--point-transformer-config", default=str(DEFAULT_POINT_TRANSFORMER_CONFIG))
    parser.add_argument("--wave2body-config", default=str(DEFAULT_WAVE2BODY_FINAL_CONFIG))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-name", help="Report run name. Defaults to a timestamped name.")
    parser.add_argument("--work-dir", help="Output directory. Defaults under WAVE2BODY artifact root.")
    parser.add_argument("--output-json", help="Write the report to this path.")
    parser.add_argument("--batch-size", type=int, help="Override both configs' inference batch size.")
    parser.add_argument("--point-transformer-batch-size", type=int)
    parser.add_argument("--wave2body-batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--flops-batches", default="all", help="Batches for FLOPs profile, or 'all'.")
    parser.add_argument("--latency-batches", default="all", help="Batches for latency timing, or 'all'.")
    parser.add_argument("--latency-repeats", type=int, default=1, help="Repeat latency timing and report aggregate stats.")
    parser.add_argument("--warmup-batches", type=int, default=5, help="Warmup batches before latency timing.")
    parser.add_argument("--flops-warmup-batches", type=int, default=0, help="Warmup batches before FLOPs profiling.")
    parser.add_argument(
        "--single-sample-latency-batches",
        default="256",
        help="B=1 batches for single-sample latency timing, or 'all'.",
    )
    parser.add_argument(
        "--single-sample-warmup-batches",
        type=int,
        default=32,
        help="B=1 warmup batches before single-sample latency timing.",
    )
    parser.add_argument(
        "--single-sample-latency-repeats",
        type=int,
        default=3,
        help="Repeat B=1 latency timing and report aggregate stats.",
    )
    parser.add_argument("--no-pretrained", action="store_true", help="Skip pretrained checkpoint loading.")
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        default=True,
        help="Disable torch.compile while profiling. Enabled by default for clean traces.",
    )
    parser.add_argument("--enable-compile", dest="disable_compile", action="store_false")
    return parser


def _default_run_name() -> str:
    return f"inference_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _resolve_work_root(args: argparse.Namespace, run_name: str) -> Path:
    if args.work_dir:
        return Path(args.work_dir).expanduser()
    return artifact_root() / "experiments" / "efficiency" / "results" / run_name


def _parse_methods(value: str) -> list[str]:
    text = str(value).strip().lower()
    if text == "both":
        return ["point_transformer", "wave2body_final"]
    methods = [item.strip() for item in text.split(",") if item.strip()]
    valid = {"point_transformer", "wave2body_final"}
    unknown = sorted(set(methods) - valid)
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")
    if not methods:
        raise ValueError("--methods resolved to an empty list")
    return methods


def _set_compile_disabled(training: dict[str, Any]) -> None:
    current = training.get("compile", {})
    if isinstance(current, dict):
        current = dict(current)
        current["enabled"] = False
        training["compile"] = current
    else:
        training["compile"] = {"enabled": False}


def _prepare_config(
    config: dict[str, Any],
    *,
    batch_size: int | None,
    num_workers: int | None,
    disable_compile: bool,
) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    training = cfg.setdefault("training", {})
    if batch_size is not None:
        training["batch_size"] = int(batch_size)
    if num_workers is not None:
        workers = int(num_workers)
        training["num_workers"] = workers
        if workers <= 0:
            training["pin_memory"] = False
            training["persistent_workers"] = False
    if disable_compile:
        _set_compile_disabled(training)
    return cfg


def _config_with_batch_size(config: dict[str, Any], batch_size: int) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg.setdefault("training", {})["batch_size"] = int(batch_size)
    return cfg


def _device_from_arg(value: str) -> torch.device:
    if str(value) == "cpu":
        return torch.device("cpu")
    return torch.device(value if torch.cuda.is_available() else "cpu")


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _loader_for_split(config: dict[str, Any], split: str):
    bundle = build_radar_pose_dataloaders(config=config, checkpoint_dir=None, include_test=True)
    if split == "test":
        if bundle.test_loader is None:
            raise RuntimeError("Requested test split, but test_loader is unavailable.")
        return bundle.test_loader, int(bundle.test_count)
    return bundle.val_loader, int(bundle.val_count)


def _max_batches_from_arg(value: str) -> int | None:
    return _parse_profile_batches(value, profile_full_train=False)


def _transform_flops_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile_mode": profile["profile_mode"],
        "profile_batches_requested": profile["profile_batches_requested"],
        "profile_batches": profile["profile_batches"],
        "profile_samples": profile["profile_samples"],
        "avg_batch_samples": profile["avg_batch_samples"],
        "avg_forward_batch_flops": profile["avg_train_step_flops"],
        "avg_forward_sample_flops": profile["avg_train_sample_flops"],
        "avg_forward_batch_flops_by_component": profile["avg_train_step_flops_by_component"],
        "profiler_total_flops": profile["profiler_total_flops"],
        "unsupported_or_zero_flop_events": profile["unsupported_or_zero_flop_events"],
    }


def _measure_latency(
    *,
    loader,
    prepare_batch: Callable[[Any], tuple[Any, int]],
    forward_prepared: Callable[[Any], Any],
    device: torch.device,
    warmup_batches: int,
    max_batches: int | None,
    measure_memory: bool = False,
) -> dict[str, Any]:
    batch_times: list[float] = []
    sample_count = 0
    timed_batches = 0
    warmup_remaining = max(0, int(warmup_batches))
    memory_reset = False
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and timed_batches >= int(max_batches):
                break
            prepared, batch_size = prepare_batch(batch)
            if warmup_remaining > 0:
                _ = forward_prepared(prepared)
                _cuda_sync(device)
                warmup_remaining -= 1
                continue
            if measure_memory and device.type == "cuda" and not memory_reset:
                _cuda_sync(device)
                torch.cuda.reset_peak_memory_stats(device=device)
                memory_reset = True
            _cuda_sync(device)
            t0 = time.perf_counter()
            _ = forward_prepared(prepared)
            _cuda_sync(device)
            elapsed = time.perf_counter() - t0
            batch_times.append(float(elapsed))
            sample_count += int(batch_size)
            timed_batches += 1
    if timed_batches <= 0 or sample_count <= 0:
        raise RuntimeError("No batches were timed. Check --latency-batches/--warmup-batches.")
    total_sec = float(sum(batch_times))
    payload = {
        "timed_batches": int(timed_batches),
        "timed_samples": int(sample_count),
        "total_forward_sec": total_sec,
        "mean_batch_latency_ms": float(mean(batch_times) * 1000.0),
        "latency_ms_per_sample": float(total_sec / sample_count * 1000.0),
        "fps": float(sample_count / max(total_sec, 1e-12)),
        "excludes": ["data_loading", "host_to_device_transfer", "loss", "metrics"],
    }
    if measure_memory:
        if device.type == "cuda":
            _cuda_sync(device)
            payload["peak_memory_allocated_mib"] = float(torch.cuda.max_memory_allocated(device=device) / (1024.0**2))
            payload["peak_memory_reserved_mib"] = float(torch.cuda.max_memory_reserved(device=device) / (1024.0**2))
        else:
            payload["peak_memory_allocated_mib"] = 0.0
            payload["peak_memory_reserved_mib"] = 0.0
    return payload


def _measure_latency_repeated(
    *,
    repeats: int,
    loader_factory: Callable[[], Any],
    prepare_batch: Callable[[Any], tuple[Any, int]],
    forward_prepared: Callable[[Any], Any],
    device: torch.device,
    warmup_batches: int,
    max_batches: int | None,
    measure_memory: bool = False,
) -> dict[str, Any]:
    repeats = max(1, int(repeats))
    runs = [
        _measure_latency(
            loader=loader_factory(),
            prepare_batch=prepare_batch,
            forward_prepared=forward_prepared,
            device=device,
            warmup_batches=warmup_batches,
            max_batches=max_batches,
            measure_memory=measure_memory,
        )
        for _ in range(repeats)
    ]
    if repeats == 1:
        payload = dict(runs[0])
        payload["repeats"] = 1
        payload["runs"] = runs
        return payload

    latency_values = [float(item["latency_ms_per_sample"]) for item in runs]
    fps_values = [float(item["fps"]) for item in runs]
    payload = dict(runs[0])
    payload.update(
        {
            "repeats": repeats,
            "runs": runs,
            "latency_ms_per_sample_mean": float(mean(latency_values)),
            "latency_ms_per_sample_median": float(median(latency_values)),
            "latency_ms_per_sample_std": float(stdev(latency_values)) if repeats > 1 else 0.0,
            "fps_mean": float(mean(fps_values)),
            "fps_median": float(median(fps_values)),
            "fps_std": float(stdev(fps_values)) if repeats > 1 else 0.0,
        }
    )
    payload["latency_ms_per_sample"] = payload["latency_ms_per_sample_mean"]
    payload["fps"] = payload["fps_mean"]
    payload["total_forward_sec"] = float(sum(float(item["total_forward_sec"]) for item in runs))
    payload["timed_samples"] = int(sum(int(item["timed_samples"]) for item in runs))
    payload["timed_batches"] = int(sum(int(item["timed_batches"]) for item in runs))
    if measure_memory:
        payload["peak_memory_allocated_mib"] = float(max(float(item.get("peak_memory_allocated_mib", 0.0)) for item in runs))
        payload["peak_memory_reserved_mib"] = float(max(float(item.get("peak_memory_reserved_mib", 0.0)) for item in runs))
    return payload


def _profile_point_transformer(
    *,
    config_path: Path,
    config: dict[str, Any],
    split: str,
    device: torch.device,
    flops_batches: int | None,
    flops_warmup_batches: int,
    latency_batches: int | None,
    latency_warmup_batches: int,
    latency_repeats: int,
    single_sample_latency_batches: int | None,
    single_sample_warmup_batches: int,
    single_sample_latency_repeats: int,
    no_pretrained: bool,
) -> dict[str, Any]:
    training_cfg = config.get("training", {})
    set_global_seed(int(training_cfg.get("seed", config.get("random_seed", 42))))
    runtime = configure_runtime(training_cfg, device, default_amp=True, log_prefix="[Inference/PT]")
    loader, split_count = _loader_for_split(config, split)
    loader_factory = lambda: _loader_for_split(config, split)[0]
    single_sample_config = _config_with_batch_size(config, 1)
    single_sample_loader_factory = lambda: _loader_for_split(single_sample_config, split)[0]
    model = PTPoseModel(point_transformer_config=config["point_transformer"], device=str(device)).to(device)
    materialize_lazy_modules(model, config=config, device=device, runtime=runtime)
    if not no_pretrained:
        checkpoint = config.get("pretrained", {}).get("point_transformer")
        if checkpoint:
            load_point_transformer_checkpoint(model.pt, checkpoint, map_location=device, strict=False)
    model.eval()
    pt = model.pt
    components = [
        {"name": "point_transformer.forward", "object": pt, "attribute": "forward", "profile": True, "params": False},
        {"name": "point_transformer.backbone", "object": pt.backbone, "attribute": "forward", "profile": True, "params": True},
        {"name": "point_transformer.backbone_proj", "object": pt.backbone_proj, "attribute": "forward", "profile": True, "params": True},
        {"name": "point_transformer.transformer", "object": pt.transformer, "attribute": "forward", "profile": True, "params": True},
        {"name": "point_transformer.fc2", "object": pt.fc2, "attribute": "forward", "profile": True, "params": True},
        {"name": "point_transformer.fc3", "object": pt.fc3, "attribute": "forward", "profile": True, "params": True},
    ]

    def prepare(batch):
        mmwave_points = batch["mmwave_points"].to(device, non_blocking=True)
        return mmwave_points, int(mmwave_points.shape[0])

    def forward_prepared(mmwave_points):
        with autocast_context(runtime):
            return model(mmwave_points)

    def step(batch) -> int:
        prepared, batch_size = prepare(batch)
        with torch.inference_mode():
            _ = forward_prepared(prepared)
        return batch_size

    flops_raw = _profile_flops_streaming(
        step_fn=step,
        loader=loader,
        components=components,
        device=device,
        profile_batches=flops_batches,
        warmup_batches=flops_warmup_batches,
        max_batches=None,
        train_batches_per_epoch=len(loader),
    )
    latency = _measure_latency_repeated(
        repeats=latency_repeats,
        loader_factory=loader_factory,
        prepare_batch=prepare,
        forward_prepared=forward_prepared,
        device=device,
        warmup_batches=latency_warmup_batches,
        max_batches=latency_batches,
        measure_memory=True,
    )
    single_sample_latency = _measure_latency_repeated(
        repeats=single_sample_latency_repeats,
        loader_factory=single_sample_loader_factory,
        prepare_batch=prepare,
        forward_prepared=forward_prepared,
        device=device,
        warmup_batches=single_sample_warmup_batches,
        max_batches=single_sample_latency_batches,
    )
    return {
        "method": "point_transformer",
        "config_path": str(config_path),
        "split": split,
        "split_samples": split_count,
        "batch_size": int(config.get("training", {}).get("batch_size", 0)),
        "params": _component_param_summary(model, components),
        "flops": _transform_flops_profile(flops_raw),
        "latency": latency,
        "single_sample_latency": single_sample_latency,
    }


def _profile_wave2body_final(
    *,
    config_path: Path,
    config: dict[str, Any],
    split: str,
    device: torch.device,
    flops_batches: int | None,
    flops_warmup_batches: int,
    latency_batches: int | None,
    latency_warmup_batches: int,
    latency_repeats: int,
    single_sample_latency_batches: int | None,
    single_sample_warmup_batches: int,
    single_sample_latency_repeats: int,
    no_pretrained: bool,
) -> dict[str, Any]:
    training_cfg = config.get("training", {})
    set_global_seed(int(training_cfg.get("seed", config.get("random_seed", 42))))
    runtime = configure_runtime(training_cfg, device, default_amp=True, log_prefix="[Inference/Wave2Body]")
    loader, split_count = _loader_for_split(config, split)
    loader_factory = lambda: _loader_for_split(config, split)[0]
    single_sample_config = _config_with_batch_size(config, 1)
    single_sample_loader_factory = lambda: _loader_for_split(single_sample_config, split)[0]
    model = AbsoluteRootHeadEstimatorModel(
        mmwave_tokenizer_config=config["mmwave_tokenizer"],
        sensor_token_config=config.get("sensor_tokens", {}),
        token_translator_config=config["token_translator"],
        vqvae_config=config["vqvae"],
        root_head_config=config.get("root_head", {}),
        device=str(device),
        freeze_mmwave_tokenizer=True,
        freeze_token_translator=True,
        freeze_vqvae=True,
    )
    pretrained = config.get("pretrained", {}) if not no_pretrained else {}
    model.load_pretrained_weights(
        mmwave_tokenizer_path=pretrained.get("mmwave_tokenizer"),
        vqvae_path=pretrained.get("vqvae"),
        hpe_token_path=pretrained.get("hpe_token"),
        device=str(device),
    )
    model = model.to(device)
    model.eval()
    components = [
        {"name": "wave2body.forward", "object": model, "attribute": "forward", "profile": True, "params": False},
        {"name": "mmwave_tokenizer_encode", "object": model, "attribute": "encode_mmwave_tokens", "profile": True, "params": False},
        {"name": "token_translator", "object": model, "attribute": "_translate", "profile": True, "params": False},
        {"name": "human_pose_tokenizer_quantize", "object": model, "attribute": "quantize_tokens", "profile": True, "params": False},
        {"name": "human_pose_tokenizer_decode", "object": model.vqvae, "attribute": "decode", "profile": True, "params": False},
        {"name": "root_head", "object": model.root_head, "attribute": "forward_with_components", "profile": True, "params": True},
        {"name": "mmwave_tokenizer", "object": model.mmwave_tokenizer, "profile": False, "params": True},
        {"name": "token_translator", "object": model.token_translator, "profile": False, "params": True},
        {"name": "human_pose_tokenizer_vqvae", "object": model.vqvae, "profile": False, "params": True},
    ]

    def prepare(batch):
        mmwave_points = batch["mmwave_points"].to(device, non_blocking=True)
        return mmwave_points, int(mmwave_points.shape[0])

    def forward_prepared(mmwave_points):
        with autocast_context(runtime):
            return model(mmwave_points=mmwave_points, target_poses=None)

    def step(batch) -> int:
        prepared, batch_size = prepare(batch)
        with torch.inference_mode():
            _ = forward_prepared(prepared)
        return batch_size

    flops_raw = _profile_flops_streaming(
        step_fn=step,
        loader=loader,
        components=components,
        device=device,
        profile_batches=flops_batches,
        warmup_batches=flops_warmup_batches,
        max_batches=None,
        train_batches_per_epoch=len(loader),
    )
    latency = _measure_latency_repeated(
        repeats=latency_repeats,
        loader_factory=loader_factory,
        prepare_batch=prepare,
        forward_prepared=forward_prepared,
        device=device,
        warmup_batches=latency_warmup_batches,
        max_batches=latency_batches,
        measure_memory=True,
    )
    single_sample_latency = _measure_latency_repeated(
        repeats=single_sample_latency_repeats,
        loader_factory=single_sample_loader_factory,
        prepare_batch=prepare,
        forward_prepared=forward_prepared,
        device=device,
        warmup_batches=single_sample_warmup_batches,
        max_batches=single_sample_latency_batches,
    )
    return {
        "method": "wave2body_final",
        "config_path": str(config_path),
        "split": split,
        "split_samples": split_count,
        "batch_size": int(config.get("training", {}).get("batch_size", 0)),
        "inference_path": [
            "mmwave_tokenizer_encode",
            "token_translator",
            "human_pose_tokenizer_quantize",
            "human_pose_tokenizer_decode",
            "absolute_root_head",
        ],
        "params": _component_param_summary(model, components),
        "flops": _transform_flops_profile(flops_raw),
        "latency": latency,
        "single_sample_latency": single_sample_latency,
    }


def _comparison_table(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in results:
        rows.append(
            {
                "Method": item["method"],
                "Split": item["split"],
                "Batch Size": item["batch_size"],
                "Trainable Params": int(item["params"]["trainable"]),
                "Total Params": int(item["params"]["total"]),
                "FLOPs / Sample": float(item["flops"]["avg_forward_sample_flops"]),
                "Latency ms / Sample": float(item["latency"]["latency_ms_per_sample"]),
                "FPS": float(item["latency"]["fps"]),
                "Latency B1 ms / Sample": float(item["single_sample_latency"]["latency_ms_per_sample"]),
                "Throughput B192 FPS": float(item["latency"]["fps"]),
                "Peak Memory B192 MiB": float(item["latency"].get("peak_memory_allocated_mib", 0.0)),
                "Peak Reserved Memory B192 MiB": float(item["latency"].get("peak_memory_reserved_mib", 0.0)),
                "Profiled FLOPs Samples": int(item["flops"]["profile_samples"]),
                "Timed Samples": int(item["latency"]["timed_samples"]),
                "Timed B1 Samples": int(item["single_sample_latency"]["timed_samples"]),
            }
        )
    return rows


def _print_report(report: dict[str, Any]) -> None:
    print("\n[Inference Efficiency] Summary")
    for row in report["comparison_table"]:
        print(
            f"- {row['Method']}: flops/sample={_fmt_si(row['FLOPs / Sample'])}, "
            f"latency_b1={row['Latency B1 ms / Sample']:.4f} ms, "
            f"throughput_b192={row['Throughput B192 FPS']:.2f} samples/s, "
            f"peak_mem_b192={row['Peak Memory B192 MiB']:.1f} MiB, split={row['Split']}"
        )


def _write_table_source_csv(report: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "method",
        "params_m",
        "flops_per_sample",
        "latency_b1_ms_per_sample",
        "throughput_b192_fps",
        "peak_memory_b192_mib",
        "peak_reserved_memory_b192_mib",
        "batch_size",
        "split",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["comparison_table"]:
            writer.writerow(
                {
                    "method": row["Method"],
                    "params_m": f"{int(row['Total Params']) / 1e6:.2f}",
                    "flops_per_sample": _fmt_si(float(row["FLOPs / Sample"])),
                    "latency_b1_ms_per_sample": f"{float(row['Latency B1 ms / Sample']):.4f}",
                    "throughput_b192_fps": f"{float(row['Throughput B192 FPS']):.2f}",
                    "peak_memory_b192_mib": f"{float(row['Peak Memory B192 MiB']):.1f}",
                    "peak_reserved_memory_b192_mib": f"{float(row['Peak Reserved Memory B192 MiB']):.1f}",
                    "batch_size": row["Batch Size"],
                    "split": row["Split"],
                }
            )


def _make_report(
    *,
    run_name: str,
    work_root: Path,
    device: torch.device,
    results: list[dict[str, Any]],
    output_json: Path,
) -> dict[str, Any]:
    return {
        "benchmark": "inference_forward_flops_latency",
        "run_name": run_name,
        "work_root": str(work_root),
        "device": str(device),
        "methodology": {
            "flops": "torch.profiler.profile(with_flops=True) on forward-only model inference.",
            "latency": "CUDA-synchronized model forward time only, excluding data loading, host-to-device transfer, loss, and metrics.",
            "single_sample_latency": "Latency table values use batch size 1. Throughput and peak memory table values use the configured batch size.",
            "note": "Latency/FPS are hardware-, precision-, batch-size-, and implementation-dependent.",
        },
        "results": results,
        "comparison_table": _comparison_table(results),
        "summary_path": str(output_json),
    }


def _write_report(
    *,
    run_name: str,
    work_root: Path,
    device: torch.device,
    results: list[dict[str, Any]],
    output_json: Path,
) -> dict[str, Any]:
    report = _make_report(
        run_name=run_name,
        work_root=work_root,
        device=device,
        results=results,
        output_json=output_json,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_table_source_csv(report, output_json.parent / "inference_table_source.csv")
    print(f"[Inference Efficiency] Wrote interim summary: {output_json}")
    return report


def main() -> None:
    args = _build_parser().parse_args()
    methods = _parse_methods(args.methods)
    flops_batches = _parse_profile_batches(args.flops_batches, profile_full_train=False)
    latency_batches = _max_batches_from_arg(args.latency_batches)
    single_sample_latency_batches = _max_batches_from_arg(args.single_sample_latency_batches)
    if int(args.warmup_batches) < 0 or int(args.flops_warmup_batches) < 0 or int(args.single_sample_warmup_batches) < 0:
        raise ValueError("Warmup batch counts cannot be negative.")
    if int(args.latency_repeats) <= 0:
        raise ValueError("--latency-repeats must be positive")
    if int(args.single_sample_latency_repeats) <= 0:
        raise ValueError("--single-sample-latency-repeats must be positive")

    run_name = args.run_name or _default_run_name()
    work_root = _resolve_work_root(args, run_name)
    work_root.mkdir(parents=True, exist_ok=True)
    device = _device_from_arg(args.device)
    results: list[dict[str, Any]] = []
    output_json = Path(args.output_json).expanduser() if args.output_json else work_root / "inference_efficiency_summary.json"

    if "point_transformer" in methods:
        print("[Inference Efficiency] Profiling point_transformer...")
        pt_config_path = Path(args.point_transformer_config).expanduser()
        pt_batch = args.point_transformer_batch_size if args.point_transformer_batch_size is not None else args.batch_size
        pt_config = _prepare_config(
            load_config(pt_config_path),
            batch_size=pt_batch,
            num_workers=args.num_workers,
            disable_compile=args.disable_compile,
        )
        results.append(
            _profile_point_transformer(
                config_path=pt_config_path,
                config=pt_config,
                split=args.split,
                device=device,
                flops_batches=flops_batches,
                flops_warmup_batches=args.flops_warmup_batches,
                latency_batches=latency_batches,
                latency_warmup_batches=args.warmup_batches,
                latency_repeats=args.latency_repeats,
                single_sample_latency_batches=single_sample_latency_batches,
                single_sample_warmup_batches=args.single_sample_warmup_batches,
                single_sample_latency_repeats=args.single_sample_latency_repeats,
                no_pretrained=args.no_pretrained,
            )
        )
        _write_report(
            run_name=run_name,
            work_root=work_root,
            device=device,
            results=results,
            output_json=output_json,
        )

    if "wave2body_final" in methods:
        print("[Inference Efficiency] Profiling wave2body_final...")
        wave_config_path = Path(args.wave2body_config).expanduser()
        wave_batch = args.wave2body_batch_size if args.wave2body_batch_size is not None else args.batch_size
        wave_config = _prepare_config(
            load_config(wave_config_path),
            batch_size=wave_batch,
            num_workers=args.num_workers,
            disable_compile=args.disable_compile,
        )
        results.append(
            _profile_wave2body_final(
                config_path=wave_config_path,
                config=wave_config,
                split=args.split,
                device=device,
                flops_batches=flops_batches,
                flops_warmup_batches=args.flops_warmup_batches,
                latency_batches=latency_batches,
                latency_warmup_batches=args.warmup_batches,
                latency_repeats=args.latency_repeats,
                single_sample_latency_batches=single_sample_latency_batches,
                single_sample_warmup_batches=args.single_sample_warmup_batches,
                single_sample_latency_repeats=args.single_sample_latency_repeats,
                no_pretrained=args.no_pretrained,
            )
        )
        _write_report(
            run_name=run_name,
            work_root=work_root,
            device=device,
            results=results,
            output_json=output_json,
        )

    report = _write_report(
        run_name=run_name,
        work_root=work_root,
        device=device,
        results=results,
        output_json=output_json,
    )
    _print_report(report)
    print(f"[Inference Efficiency] Wrote final summary: {output_json}")


if __name__ == "__main__":
    main()
