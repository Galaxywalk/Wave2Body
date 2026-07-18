#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Profile training FLOPs for Wave2Body stages and the PointTransformer baseline."""

from __future__ import annotations

import argparse
import copy
import functools
import json
import math
import sys
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.optim as optim
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wave2body.dataset.human_body import create_dataset as create_pose_dataset  # noqa: E402
from wave2body.models.hpe_token import TokenHPEEstimatorModel  # noqa: E402
from wave2body.models.hpe_token.absolute_root import AbsoluteRootHeadEstimatorModel  # noqa: E402
from wave2body.models.human_body_tokenizer import JointVQVAELoss, create_joint_tokenizer  # noqa: E402
from wave2body.models.mmwave_tokenizer import create_mmwave_tokenizer  # noqa: E402
from wave2body.models.baselines import (  # noqa: E402
    build_pose_baseline,
    compute_pose_baseline_loss,
    forward_pose_baseline,
    move_pose_baseline_batch,
)
from wave2body.models.point_transformer.pose_baseline import PTPoseModel, PoseMPJPELoss  # noqa: E402
from wave2body.train.common import load_config, safe_param_count, set_global_seed  # noqa: E402
from wave2body.train.human_pose_tokenizer import (  # noqa: E402
    build_adamw_optimizer,
    resolve_enabled_datasets,
)
from wave2body.train.joint_batch_loader import JointTensorBatchLoader, pack_joints_dataset  # noqa: E402
from wave2body.train.mmwave_tokenizer_data import build_mmwave_tokenizer_dataloaders  # noqa: E402
from wave2body.train.optim import build_hpe_token_optimizer  # noqa: E402
from wave2body.train.pipelines.hpe_token import _apply_token_classifier_only_training, _normalize_training_config  # noqa: E402
from wave2body.train.pipelines.hpe_token_ablation_losses import (  # noqa: E402
    _ablation_loss_config_from_training_config,
    _apply_optional_ablation_losses,
)
from wave2body.train.pipelines.hpe_token_training import (  # noqa: E402
    _apply_optional_soft_token_loss,
    _forward_batch,
    _point_augmentation_from_config,
    _select_pose_outputs,
    _soft_token_loss_from_config,
    _token_loss_from_config,
)
from wave2body.train.pipelines.absolute_root_head import (  # noqa: E402
    _build_absolute_optimizer,
    _loss_from_batch as _absolute_loss_from_batch,
    _set_frozen_modules_eval as _set_absolute_frozen_modules_eval,
)
from wave2body.train.pipelines.mmwave_tokenizer import _move_batch as _move_mmwave_batch  # noqa: E402
from wave2body.train.pipelines.pose_baseline import _optimizer_from_config as _pose_baseline_optimizer_from_config  # noqa: E402
from wave2body.train.pt_pose_utils import materialize_lazy_modules  # noqa: E402
from wave2body.train.radar_pose_data import build_radar_pose_dataloaders  # noqa: E402
from wave2body.utils.paths import artifact_root  # noqa: E402
from wave2body.utils.runtime_utils import (  # noqa: E402
    autocast_context,
    build_grad_scaler,
    configure_runtime,
    dataset_cache_mode,
)


DEFAULT_MMWAVE_CONFIG = (
    REPO_ROOT / "wave2body" / "configs" / "mmwave_tokenizer" / "m4human_all_mae_spatial_k96_fast.yaml"
)
DEFAULT_HUMAN_POSE_STAGE_A_CONFIG = (
    REPO_ROOT / "wave2body" / "configs" / "human_pose_tokenizer" / "stage_a_part_k96_amass_mmbody_m4human.yaml"
)
DEFAULT_HUMAN_POSE_STAGE_B_CONFIG = (
    REPO_ROOT / "wave2body" / "configs" / "human_pose_tokenizer" / "stage_b_part_k96_mmbody_m4human.yaml"
)
DEFAULT_HPE_TOKEN_CONFIG = REPO_ROOT / "wave2body" / "configs" / "hpe_token" / "m4human_s2_k96_xattn.yaml"
DEFAULT_ABSOLUTE_ROOT_CONFIG = (
    REPO_ROOT
    / "wave2body"
    / "configs"
    / "hpe_token"
    / "m4human_s2_k96_queryroot_pointiwmean_absolute_trainable_translator.yaml"
)
DEFAULT_POINT_TRANSFORMER_CONFIG = (
    REPO_ROOT / "wave2body" / "configs" / "hpe_point_transformer" / "m4human_absolute_hist4_s2_bf16.yaml"
)
DEFAULT_MMDIFF_M4HUMAN_CONFIG = REPO_ROOT / "wave2body" / "configs" / "hpe_baselines" / "mmdiff_m4human_s2.yaml"
DEFAULT_MMMESH_M4HUMAN_CONFIG = REPO_ROOT / "wave2body" / "configs" / "hpe_baselines" / "mmmesh_m4human_s2.yaml"

WAVE2BODY_TRAINING_STAGES = (
    "mmwave_tokenizer",
    "human_pose_tokenizer_stage_a",
    "human_pose_tokenizer_stage_b",
    "hpe_token",
    "absolute_root_head",
)
BASELINE_STAGES = ("point_transformer", "mmdiff_m4human", "mmmesh_m4human")
ALL_STAGES = (*WAVE2BODY_TRAINING_STAGES, *BASELINE_STAGES)


class _NullWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        default="wave2body",
        help=(
            "Comma-separated stages, or one of wave2body/all. "
            f"Known stages: {', '.join(ALL_STAGES)}"
        ),
    )
    parser.add_argument("--mmwave-config", default=str(DEFAULT_MMWAVE_CONFIG))
    parser.add_argument("--human-pose-stage-a-config", default=str(DEFAULT_HUMAN_POSE_STAGE_A_CONFIG))
    parser.add_argument("--human-pose-stage-b-config", default=str(DEFAULT_HUMAN_POSE_STAGE_B_CONFIG))
    parser.add_argument("--hpe-token-config", default=str(DEFAULT_HPE_TOKEN_CONFIG))
    parser.add_argument("--absolute-root-config", default=str(DEFAULT_ABSOLUTE_ROOT_CONFIG))
    parser.add_argument("--point-transformer-config", default=str(DEFAULT_POINT_TRANSFORMER_CONFIG))
    parser.add_argument("--mmdiff-m4human-config", default=str(DEFAULT_MMDIFF_M4HUMAN_CONFIG))
    parser.add_argument("--mmmesh-m4human-config", default=str(DEFAULT_MMMESH_M4HUMAN_CONFIG))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-name", help="Report run name. Defaults to a timestamped name.")
    parser.add_argument("--work-dir", help="Output directory. Defaults under WAVE2BODY artifact root.")
    parser.add_argument("--output-json", help="Write the FLOPs report to this path.")
    parser.add_argument("--epochs", type=int, help="Override configured training epochs for total FLOPs.")
    parser.add_argument(
        "--profile-batches",
        default="1",
        help="Number of train batches to profile and average, or 'all' for a streaming full train-set pass.",
    )
    parser.add_argument(
        "--profile-full-train",
        action="store_true",
        help="Shortcut for --profile-batches all. Use --warmup-batches 0 for an exact full-pass average.",
    )
    parser.add_argument("--warmup-batches", type=int, default=1, help="Warmup train batches before profiling.")
    parser.add_argument(
        "--max-train-batches",
        type=int,
        help="Cap train batches per epoch for smoke tests; total FLOPs use this capped value when set.",
    )
    parser.add_argument("--batch-size", type=int, help="Override all configs' training.batch_size.")
    parser.add_argument("--mmwave-batch-size", type=int, help="Override mmWave tokenizer batch size.")
    parser.add_argument("--human-pose-batch-size", type=int, help="Override both human-pose tokenizer stages.")
    parser.add_argument("--human-pose-stage-a-batch-size", type=int, help="Override human-pose tokenizer stage A batch size.")
    parser.add_argument("--human-pose-stage-b-batch-size", type=int, help="Override human-pose tokenizer stage B batch size.")
    parser.add_argument("--hpe-token-batch-size", type=int, help="Override HPE-token batch size.")
    parser.add_argument("--absolute-root-batch-size", type=int, help="Override absolute-root-head batch size.")
    parser.add_argument("--point-transformer-batch-size", type=int, help="Override PointTransformer batch size.")
    parser.add_argument("--mmdiff-m4human-batch-size", type=int, help="Override M4Human mmDiff baseline batch size.")
    parser.add_argument("--mmmesh-m4human-batch-size", type=int, help="Override M4Human mmMesh baseline batch size.")
    parser.add_argument("--num-workers", type=int, help="Override DataLoader workers for all stages.")
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        default=True,
        help="Disable torch.compile while profiling. Enabled by default for cleaner profiler traces.",
    )
    parser.add_argument(
        "--enable-compile",
        dest="disable_compile",
        action="store_false",
        help="Honor each config's compile setting.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Skip pretrained checkpoint loading for HPE-token. FLOPs path is unchanged for shape-compatible configs.",
    )
    return parser


def _default_run_name() -> str:
    return f"training_flops_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _resolve_work_root(args: argparse.Namespace, run_name: str) -> Path:
    if args.work_dir:
        return Path(args.work_dir).expanduser()
    return artifact_root() / "experiments" / "efficiency" / "results" / run_name


def _parse_stages(value: str) -> list[str]:
    normalized = str(value).strip().lower()
    if normalized in {"wave2body", "wave2body_all", "full"}:
        return list(WAVE2BODY_TRAINING_STAGES)
    if normalized == "all":
        return list(ALL_STAGES)
    stages = [item.strip().lower() for item in normalized.split(",") if item.strip()]
    unknown = sorted(set(stages) - set(ALL_STAGES))
    if unknown:
        raise ValueError(f"Unknown stages: {', '.join(unknown)}")
    if not stages:
        raise ValueError("--stages resolved to an empty stage list")
    return stages


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
    epochs: int | None,
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
    if epochs is not None:
        training["epochs"] = int(epochs)
        if "scheduler" in training and isinstance(training["scheduler"], dict):
            training["scheduler"]["T_max"] = int(epochs)
    return cfg


def _device_from_arg(value: str) -> torch.device:
    if str(value) == "cpu":
        return torch.device("cpu")
    return torch.device(value if torch.cuda.is_available() else "cpu")


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _limited_batches(loader: Iterable[Any], count: int) -> list[Any]:
    batches = []
    for batch in loader:
        batches.append(batch)
        if len(batches) >= int(count):
            break
    return batches


def _num_train_batches(*, train_count: int | None, batch_size: int, loader_len: int | None, max_train_batches: int | None) -> int:
    if loader_len is not None:
        batches = int(loader_len)
    elif train_count is not None:
        batches = int(train_count) // int(batch_size)
    else:
        batches = 0
    if max_train_batches is not None:
        batches = min(batches, int(max_train_batches))
    return int(batches)


def _parse_profile_batches(value: str, *, profile_full_train: bool) -> int | None:
    if profile_full_train:
        return None
    text = str(value).strip().lower()
    if text in {"all", "full", "full_train"}:
        return None
    count = int(text)
    if count <= 0:
        raise ValueError("--profile-batches must be positive or 'all'")
    return count


def _module_children_components(module: torch.nn.Module, *, prefix: str = "model") -> list[dict[str, Any]]:
    return [
        {
            "name": f"{prefix}.{name}",
            "object": child,
            "attribute": "forward",
            "profile": True,
            "params": True,
        }
        for name, child in module.named_children()
    ]


def _component_param_summary(model: torch.nn.Module, components: list[dict[str, Any]]) -> dict[str, Any]:
    by_component: dict[str, dict[str, int]] = {}
    for spec in components:
        if not bool(spec.get("params", False)):
            continue
        obj = spec["object"]
        if not isinstance(obj, torch.nn.Module):
            continue
        by_component[spec["name"]] = {
            "total": safe_param_count(obj),
            "trainable": safe_param_count(obj, trainable_only=True),
        }
    return {
        "total": safe_param_count(model),
        "trainable": safe_param_count(model, trainable_only=True),
        "by_component": by_component,
    }


def _wrap_callable(label: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with torch.profiler.record_function(f"component:{label}"):
            return fn(*args, **kwargs)

    return wrapped


@contextmanager
def _instrument_components(components: list[dict[str, Any]]):
    patches: list[tuple[Any, str, Any]] = []
    try:
        for spec in components:
            if not bool(spec.get("profile", True)):
                continue
            obj = spec["object"]
            attr = str(spec.get("attribute", "forward"))
            original = getattr(obj, attr)
            setattr(obj, attr, _wrap_callable(str(spec["name"]), original))
            patches.append((obj, attr, original))
        yield
    finally:
        for obj, attr, original in reversed(patches):
            setattr(obj, attr, original)


def _nearest_component_key(event) -> str | None:
    current = event
    while current is not None:
        key = str(getattr(current, "key", ""))
        if key.startswith("component:"):
            return key.split(":", 1)[1]
        current = getattr(current, "cpu_parent", None)
    return None


def _empty_flop_accumulator() -> dict[str, Any]:
    return {
        "total_flops": 0.0,
        "by_component": {},
        "unsupported_or_zero_flop_events": 0,
    }


def _accumulate_profiler_events(acc: dict[str, Any], events, components: list[dict[str, Any]]) -> None:
    by_component = acc["by_component"]
    for event in events:
        flops = float(getattr(event, "flops", 0) or 0)
        if flops <= 0:
            acc["unsupported_or_zero_flop_events"] += 1
            continue
        acc["total_flops"] += flops
        component = _nearest_component_key(event) or "unattributed"
        by_component[component] = by_component.get(component, 0.0) + flops

    component_names = {str(spec["name"]) for spec in components}
    if "backward_optimizer" in component_names and "unattributed" in by_component:
        # CUDA autograd matmul/conv FLOPs often lose their Python record_function
        # parent. Forward and loss are explicitly scoped, so remaining positive
        # FLOPs in training profiles are treated as backward/optimizer work.
        by_component["backward_optimizer"] = by_component.get("backward_optimizer", 0.0) + by_component.pop("unattributed")


def _finalize_profile_accumulator(
    acc: dict[str, Any],
    *,
    profile_batch_count: int,
    sample_count: int,
    profile_mode: str,
    profile_batches_requested: int | str | None,
    train_batches_per_epoch: int | None = None,
) -> dict[str, Any]:
    total = float(acc["total_flops"])
    by_component = dict(acc["by_component"])
    profile_batch_count = int(profile_batch_count)
    sample_count = int(sample_count)
    coverage = None
    if train_batches_per_epoch:
        coverage = float(profile_batch_count / max(1, int(train_batches_per_epoch)))
    return {
        "profile_mode": profile_mode,
        "profile_batches_requested": profile_batches_requested,
        "profile_batches": profile_batch_count,
        "profile_samples": sample_count,
        "profile_train_batch_coverage": coverage,
        "avg_batch_samples": float(sample_count / max(1, profile_batch_count)),
        "avg_train_step_flops": float(total / max(1, profile_batch_count)),
        "avg_train_sample_flops": float(total / max(1, sample_count)),
        "avg_train_step_flops_by_component": {
            key: float(value / max(1, profile_batch_count))
            for key, value in sorted(by_component.items(), key=lambda item: item[0])
        },
        "profiler_total_flops": total,
        "unsupported_or_zero_flop_events": int(acc["unsupported_or_zero_flop_events"]),
    }


def _profile_flops(
    *,
    step_fn: Callable[[Any], int],
    profile_batches: list[Any],
    components: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    if not profile_batches:
        raise RuntimeError("No batches available for FLOPs profiling.")

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    sample_count = 0
    with _instrument_components(components):
        _cuda_sync(device)
        with torch.profiler.profile(
            activities=activities,
            with_flops=True,
            record_shapes=True,
            profile_memory=False,
        ) as prof:
            for batch in profile_batches:
                sample_count += int(step_fn(batch))
                prof.step()
        _cuda_sync(device)

    acc = _empty_flop_accumulator()
    _accumulate_profiler_events(acc, prof.events(), components)
    return _finalize_profile_accumulator(
        acc,
        profile_batch_count=len(profile_batches),
        sample_count=sample_count,
        profile_mode="fixed_batches",
        profile_batches_requested=len(profile_batches),
    )


def _profile_flops_streaming(
    *,
    step_fn: Callable[[Any], int],
    loader,
    components: list[dict[str, Any]],
    device: torch.device,
    profile_batches: int | None,
    warmup_batches: int,
    max_batches: int | None,
    train_batches_per_epoch: int | None,
) -> dict[str, Any]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    acc = _empty_flop_accumulator()
    sample_count = 0
    profiled_batches = 0
    warmup_remaining = max(0, int(warmup_batches))
    requested = "all" if profile_batches is None else int(profile_batches)
    with _instrument_components(components):
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            if warmup_remaining > 0:
                _ = step_fn(batch)
                warmup_remaining -= 1
                continue
            if profile_batches is not None and profiled_batches >= int(profile_batches):
                break
            _cuda_sync(device)
            with torch.profiler.profile(
                activities=activities,
                with_flops=True,
                record_shapes=True,
                profile_memory=False,
            ) as prof:
                sample_count += int(step_fn(batch))
                prof.step()
            _cuda_sync(device)
            _accumulate_profiler_events(acc, prof.events(), components)
            profiled_batches += 1
    if profiled_batches <= 0:
        raise RuntimeError("No batches were profiled. Check --profile-batches/--warmup-batches/--max-train-batches.")
    return _finalize_profile_accumulator(
        acc,
        profile_batch_count=profiled_batches,
        sample_count=sample_count,
        profile_mode="streaming_full_train" if profile_batches is None else "streaming_fixed_batches",
        profile_batches_requested=requested,
        train_batches_per_epoch=train_batches_per_epoch,
    )


def _scale_flops(profile: dict[str, Any], *, steps_per_epoch: int, samples_per_epoch: int | None, epochs: int) -> dict[str, Any]:
    multiplier = int(steps_per_epoch) * int(epochs)
    step_flops = float(profile["avg_train_step_flops"])
    sample_flops = float(profile["avg_train_sample_flops"])
    component_step = dict(profile["avg_train_step_flops_by_component"])
    total = step_flops * float(multiplier)
    component_total = {key: float(value) * float(multiplier) for key, value in component_step.items()}
    sample_scaled_total = None
    component_sample_total = None
    if samples_per_epoch is not None:
        sample_multiplier = int(samples_per_epoch) * int(epochs)
        sample_scaled_total = sample_flops * float(sample_multiplier)
        component_sample_total = {
            key: (float(value) / max(float(profile.get("avg_batch_samples", 1.0)), 1.0e-12)) * float(sample_multiplier)
            for key, value in component_step.items()
        }
    component_share = {
        key: (float(value) / total if total > 0 else 0.0)
        for key, value in sorted(component_total.items(), key=lambda item: item[0])
    }
    component_sample_share = None
    if component_sample_total is not None and sample_scaled_total is not None:
        component_sample_share = {
            key: (float(value) / sample_scaled_total if sample_scaled_total > 0 else 0.0)
            for key, value in sorted(component_sample_total.items(), key=lambda item: item[0])
        }
    return {
        **profile,
        "train_batches_per_epoch": int(steps_per_epoch),
        "train_samples_per_epoch": int(samples_per_epoch) if samples_per_epoch is not None else None,
        "epochs": int(epochs),
        "total_train_steps": int(multiplier),
        "total_training_flops": float(total),
        "total_training_flops_by_component": component_total,
        "component_share": component_share,
        "total_training_flops_sample_scaled": None if sample_scaled_total is None else float(sample_scaled_total),
        "total_training_flops_by_component_sample_scaled": component_sample_total,
        "component_share_sample_scaled": component_sample_share,
    }


def _take_warmup_and_profile_batches(
    loader,
    *,
    warmup_batches: int,
    profile_batches: int | None,
    max_batches: int | None = None,
) -> tuple[list[Any], list[Any]]:
    needed = max(0, int(warmup_batches)) + max(1, int(profile_batches))
    if max_batches is not None:
        needed = min(needed, max(1, int(max_batches)))
    batches = _limited_batches(loader, needed)
    if len(batches) < max(1, int(profile_batches)):
        raise RuntimeError(f"Need at least {profile_batches} profile batches, got {len(batches)}.")
    warmup = batches[: max(0, int(warmup_batches))]
    profile = batches[max(0, int(warmup_batches)) :]
    if len(profile) < int(profile_batches):
        profile = batches[-int(profile_batches) :]
        warmup = batches[: max(0, len(batches) - len(profile))]
    return warmup, profile


def _finalize_stage_report(
    *,
    stage: str,
    config_path: Path,
    config: dict[str, Any],
    model: torch.nn.Module,
    components: list[dict[str, Any]],
    profile: dict[str, Any],
    train_count: int | None,
    train_batches_per_epoch: int,
    epochs: int,
) -> dict[str, Any]:
    batch_size = int(config.get("training", {}).get("batch_size", 0))
    samples_per_epoch = None
    if batch_size > 0:
        samples_per_epoch = int(train_batches_per_epoch) * batch_size
        if train_count is not None:
            samples_per_epoch = min(int(train_count), samples_per_epoch)
    flops = _scale_flops(
        profile,
        steps_per_epoch=train_batches_per_epoch,
        samples_per_epoch=samples_per_epoch,
        epochs=epochs,
    )
    return {
        "stage": stage,
        "config_path": str(config_path),
        "dataset": config.get("dataset", {}).get("name") or config.get("data", {}).get("datasets"),
        "batch_size": int(config.get("training", {}).get("batch_size", 0)),
        "train_samples": int(train_count) if train_count is not None else None,
        "params": _component_param_summary(model, components),
        "flops": flops,
    }


def _run_warmup(step_fn: Callable[[Any], int], batches: list[Any], device: torch.device) -> None:
    for batch in batches:
        _ = step_fn(batch)
    _cuda_sync(device)


def _profile_mmwave_tokenizer(
    *,
    config_path: Path,
    config: dict[str, Any],
    device: torch.device,
    profile_batches: int | None,
    warmup_batches: int,
    max_train_batches: int | None,
) -> dict[str, Any]:
    set_global_seed(int(config.get("random_seed", config.get("training", {}).get("seed", 42))))
    runtime = configure_runtime(config.get("training", {}), device, default_amp=True, log_prefix="[FLOPs/MMWave]")
    data_bundle = build_mmwave_tokenizer_dataloaders(
        config,
        checkpoint_dir=None,
        include_test=False,
        max_train_batches=max_train_batches,
        max_val_batches=1,
    )
    model = create_mmwave_tokenizer(config["mmwave_tokenizer"]).to(device)
    train_cfg = config.get("training", {})
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1.0e-4)),
    )
    scaler = build_grad_scaler(runtime)
    grad_norm = train_cfg.get("grad_norm", None)
    components = [
        {"name": "model_forward_unscoped", "object": model, "attribute": "forward", "profile": True, "params": False},
        *_module_children_components(model),
        {"name": "loss", "object": None, "profile": False, "params": False},
        {"name": "backward_optimizer", "object": None, "profile": False, "params": False},
    ]

    def step(batch) -> int:
        model.train()
        points, target_abs = _move_mmwave_batch(batch, device)
        batch_size = int(points.shape[0])
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(runtime):
            outputs = model(points, target_poses_absolute=target_abs)
            with torch.profiler.record_function("component:loss"):
                loss = outputs["losses"]["total_loss"]
        with torch.profiler.record_function("component:backward_optimizer"):
            scaler.scale(loss).backward()
            if grad_norm is not None and float(grad_norm) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_norm))
            scaler.step(optimizer)
            scaler.update()
        optimizer.zero_grad(set_to_none=True)
        return batch_size

    train_batches = _num_train_batches(
        train_count=data_bundle.train_count,
        batch_size=int(config["training"]["batch_size"]),
        loader_len=len(data_bundle.train_loader),
        max_train_batches=max_train_batches,
    )
    profile = _profile_flops_streaming(
        step_fn=step,
        loader=data_bundle.train_loader,
        components=components,
        device=device,
        profile_batches=profile_batches,
        warmup_batches=warmup_batches,
        max_batches=max_train_batches,
        train_batches_per_epoch=train_batches,
    )
    epochs = int(config.get("training", {}).get("epochs", 1))
    return _finalize_stage_report(
        stage="mmwave_tokenizer",
        config_path=config_path,
        config=config,
        model=model,
        components=components,
        profile=profile,
        train_count=data_bundle.train_count,
        train_batches_per_epoch=train_batches,
        epochs=epochs,
    )


def _build_human_pose_train_loader(config: dict[str, Any]):
    data_config = config.get("data", {})
    enabled_datasets = resolve_enabled_datasets(config)
    dataset_device = data_config.get("dataset_device", "cuda" if torch.cuda.is_available() else "cpu")
    if dataset_device == "cuda" and not torch.cuda.is_available():
        dataset_device = "cpu"
    train_dataset, _ = create_pose_dataset(
        enabled_datasets,
        config["model"]["num_joints"],
        dataset_device,
        use_cache=bool(data_config.get("use_cache", True)),
        cache_dir=data_config.get("cache_dir"),
        cache_build_workers=data_config.get("cache_build_workers"),
        cache_build_chunk_size=data_config.get("cache_build_chunk_size"),
    )
    training = config.get("training", {})
    batch_size = int(training["batch_size"])
    train_size = int(float(data_config.get("train_ratio", 0.9)) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    num_workers = int(training.get("num_workers", 0))
    pin_memory = bool(training.get("pin_memory", True))
    persistent_workers = bool(training.get("persistent_workers", num_workers > 0))
    prefetch_factor = int(training.get("prefetch_factor", 2))
    batch_loader_mode = str(data_config.get("batch_loader", "dataloader")).strip().lower()

    if batch_loader_mode in {"tensor", "tensor_batch", "joint_tensor"}:
        packed_dataset = pack_joints_dataset(train_dataset)
        if packed_dataset is not None:
            split_indices = torch.randperm(len(packed_dataset))
            train_indices = split_indices[:train_size]
            loader = JointTensorBatchLoader(
                packed_dataset,
                train_indices,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=pin_memory,
            )
            return loader, int(train_size), "joint_tensor_batch"

    train_ds, _ = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    cache_mode = dataset_cache_mode(train_ds)
    if cache_mode == "gpu_tensor" or dataset_device == "cuda":
        num_workers = 0
        pin_memory = False
        persistent_workers = False
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
        "drop_last": False,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(train_ds, **kwargs), int(train_size), cache_mode


def _profile_human_pose_tokenizer(
    *,
    stage: str,
    config_path: Path,
    config: dict[str, Any],
    device: torch.device,
    profile_batches: int | None,
    warmup_batches: int,
    max_train_batches: int | None,
) -> dict[str, Any]:
    set_global_seed(int(config.get("seed", 42)), deterministic=bool(config.get("training", {}).get("deterministic", False)))
    runtime = configure_runtime(config.get("training", {}), device, default_amp=True, log_prefix=f"[FLOPs/{stage}]")
    loader, train_count, cache_mode = _build_human_pose_train_loader(config)
    model = create_joint_tokenizer(config["model"]).float().to(device)
    criterion = JointVQVAELoss(**config["loss"])
    optimizer = build_adamw_optimizer(model.parameters(), config["training"]["optimizer"], device)
    scaler = build_grad_scaler(runtime)
    grad_accum_steps = max(1, int(config.get("training", {}).get("grad_accum_steps", 1)))
    components = [
        {"name": "model_forward_unscoped", "object": model, "attribute": "forward", "profile": True, "params": False},
        *_module_children_components(model),
        {"name": "loss", "object": None, "profile": False, "params": False},
        {"name": "backward_optimizer", "object": None, "profile": False, "params": False},
    ]

    def step(batch) -> int:
        model.train()
        joints = batch[0] if isinstance(batch, (tuple, list)) else batch["joints"]
        joints = joints.to(device, non_blocking=True)
        joints_visible = torch.ones(joints.shape[:2], device=device, dtype=torch.bool)
        batch_size = int(joints.shape[0])
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(runtime):
            recovered, _, e_loss = model(joints=joints, joints_visible=joints_visible, train=True)
            with torch.profiler.record_function("component:loss"):
                loss_dict = criterion(recovered, joints, e_loss)
                loss = loss_dict["total_loss"] / float(grad_accum_steps)
        with torch.profiler.record_function("component:backward_optimizer"):
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return batch_size

    loader_len = len(loader) if hasattr(loader, "__len__") else math.ceil(train_count / int(config["training"]["batch_size"]))
    train_batches = _num_train_batches(
        train_count=train_count,
        batch_size=int(config["training"]["batch_size"]),
        loader_len=loader_len,
        max_train_batches=max_train_batches,
    )
    profile = _profile_flops_streaming(
        step_fn=step,
        loader=loader,
        components=components,
        device=device,
        profile_batches=profile_batches,
        warmup_batches=warmup_batches,
        max_batches=max_train_batches,
        train_batches_per_epoch=train_batches,
    )
    epochs = int(config.get("training", {}).get("epochs", 1))
    report = _finalize_stage_report(
        stage=stage,
        config_path=config_path,
        config=config,
        model=model,
        components=components,
        profile=profile,
        train_count=train_count,
        train_batches_per_epoch=train_batches,
        epochs=epochs,
    )
    report["data_loader"] = {"cache_mode": cache_mode}
    return report


def _profile_hpe_token(
    *,
    config_path: Path,
    config: dict[str, Any],
    device: torch.device,
    profile_batches: int | None,
    warmup_batches: int,
    max_train_batches: int | None,
    no_pretrained: bool,
) -> dict[str, Any]:
    config = _normalize_training_config(config)
    training_cfg = config.get("training", {})
    set_global_seed(int(training_cfg.get("seed", config.get("random_seed", 42))))
    runtime = configure_runtime(training_cfg, device, default_amp=True, log_prefix="[FLOPs/HPE Token]")
    model = TokenHPEEstimatorModel(
        mmwave_tokenizer_config=config["mmwave_tokenizer"],
        sensor_token_config=config.get("sensor_tokens", {}),
        token_translator_config=config["token_translator"],
        vqvae_config=config["vqvae"],
        device=str(device),
        freeze_mmwave_tokenizer=bool(training_cfg.get("freeze_mmwave_tokenizer", True)),
        freeze_token_translator=bool(training_cfg.get("freeze_token_translator", False)),
        freeze_vqvae=bool(training_cfg.get("freeze_vqvae", True)),
    )
    pretrained = config.get("pretrained", {}) if not no_pretrained else {}
    model.load_pretrained_weights(
        mmwave_tokenizer_path=pretrained.get("mmwave_tokenizer"),
        vqvae_path=pretrained.get("vqvae"),
        device=str(device),
    )
    model = model.to(device)
    _apply_token_classifier_only_training(model, training_cfg)
    data_bundle = build_radar_pose_dataloaders(
        config=config,
        checkpoint_dir=None,
        include_test=False,
        max_train_batches=max_train_batches,
        max_val_batches=1,
    )
    criterion = _token_loss_from_config(training_cfg)
    soft_token_criterion, soft_token_weight = _soft_token_loss_from_config(training_cfg, model.vqvae.layout)
    if soft_token_criterion is not None:
        soft_token_criterion = soft_token_criterion.to(device)
    ablation_loss_cfg = _ablation_loss_config_from_training_config(training_cfg)
    point_augmentation_cfg = _point_augmentation_from_config(training_cfg)
    optimizer = build_hpe_token_optimizer(model, config)
    scaler = build_grad_scaler(runtime)
    grad_norm = float(training_cfg.get("grad_norm", 0.0))
    components = [
        {"name": "hpe_forward_unscoped", "object": model, "attribute": "forward", "profile": True, "params": False},
        {
            "name": "mmwave_tokenizer_encode",
            "object": model,
            "attribute": "encode_mmwave_tokens",
            "profile": True,
            "params": False,
        },
        {"name": "token_translator", "object": model, "attribute": "_translate", "profile": True, "params": False},
        {"name": "human_pose_tokenizer_quantize", "object": model, "attribute": "quantize_tokens", "profile": True, "params": False},
        {
            "name": "human_pose_tokenizer_target_encode",
            "object": model.vqvae,
            "attribute": "encode_with_intermediates",
            "profile": True,
            "params": False,
        },
        {"name": "human_pose_tokenizer_decode", "object": model.vqvae, "attribute": "decode", "profile": True, "params": False},
        {"name": "mmwave_tokenizer", "object": model.mmwave_tokenizer, "profile": False, "params": True},
        {"name": "token_translator", "object": model.token_translator, "profile": False, "params": True},
        {"name": "human_pose_tokenizer_vqvae", "object": model.vqvae, "profile": False, "params": True},
        {"name": "loss", "object": None, "profile": False, "params": False},
        {"name": "backward_optimizer", "object": None, "profile": False, "params": False},
    ]

    def step(batch) -> int:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        target_poses, outputs = _forward_batch(
            model,
            batch,
            device,
            runtime,
            point_augmentation=point_augmentation_cfg,
        )
        batch_size = int(target_poses.shape[0])
        pose_selection = _select_pose_outputs(model, outputs, training_cfg)
        with torch.profiler.record_function("component:loss"):
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
        with torch.profiler.record_function("component:backward_optimizer"):
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
                optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return batch_size

    train_batches = _num_train_batches(
        train_count=data_bundle.train_count,
        batch_size=int(config["training"]["batch_size"]),
        loader_len=len(data_bundle.train_loader),
        max_train_batches=max_train_batches,
    )
    profile = _profile_flops_streaming(
        step_fn=step,
        loader=data_bundle.train_loader,
        components=components,
        device=device,
        profile_batches=profile_batches,
        warmup_batches=warmup_batches,
        max_batches=max_train_batches,
        train_batches_per_epoch=train_batches,
    )
    epochs = int(config.get("training", {}).get("epochs", 1))
    return _finalize_stage_report(
        stage="hpe_token",
        config_path=config_path,
        config=config,
        model=model,
        components=components,
        profile=profile,
        train_count=data_bundle.train_count,
        train_batches_per_epoch=train_batches,
        epochs=epochs,
    )


def _profile_absolute_root_head(
    *,
    config_path: Path,
    config: dict[str, Any],
    device: torch.device,
    profile_batches: int | None,
    warmup_batches: int,
    max_train_batches: int | None,
    no_pretrained: bool,
) -> dict[str, Any]:
    training_cfg = config.get("training", {})
    set_global_seed(int(training_cfg.get("seed", config.get("random_seed", 42))))
    runtime = configure_runtime(training_cfg, device, default_amp=True, log_prefix="[FLOPs/AbsoluteRoot]")
    model = AbsoluteRootHeadEstimatorModel(
        mmwave_tokenizer_config=config["mmwave_tokenizer"],
        sensor_token_config=config.get("sensor_tokens", {}),
        token_translator_config=config["token_translator"],
        vqvae_config=config["vqvae"],
        root_head_config=config.get("root_head", {}),
        device=str(device),
        freeze_mmwave_tokenizer=bool(training_cfg.get("freeze_mmwave_tokenizer", False)),
        freeze_token_translator=bool(training_cfg.get("freeze_token_translator", False)),
        freeze_vqvae=bool(training_cfg.get("freeze_vqvae", True)),
    )
    pretrained = config.get("pretrained", {}) if not no_pretrained else {}
    model.load_pretrained_weights(
        mmwave_tokenizer_path=pretrained.get("mmwave_tokenizer"),
        vqvae_path=pretrained.get("vqvae"),
        hpe_token_path=pretrained.get("hpe_token"),
        device=str(device),
    )
    model = model.to(device)
    data_bundle = build_radar_pose_dataloaders(
        config=config,
        checkpoint_dir=None,
        include_test=False,
        max_train_batches=max_train_batches,
        max_val_batches=1,
    )
    criterion = _token_loss_from_config(training_cfg)
    optimizer = _build_absolute_optimizer(model, config)
    scaler = build_grad_scaler(runtime)
    grad_norm = float(training_cfg.get("grad_norm", 0.0))
    root_index = int(model.root_index)
    components = [
        {"name": "absolute_root_forward_unscoped", "object": model, "attribute": "forward", "profile": True, "params": False},
        {
            "name": "mmwave_tokenizer_encode",
            "object": model,
            "attribute": "encode_mmwave_tokens",
            "profile": True,
            "params": False,
        },
        {"name": "token_translator", "object": model, "attribute": "_translate", "profile": True, "params": False},
        {"name": "human_pose_tokenizer_quantize", "object": model, "attribute": "quantize_tokens", "profile": True, "params": False},
        {
            "name": "human_pose_tokenizer_target_encode",
            "object": model.vqvae,
            "attribute": "encode_with_intermediates",
            "profile": True,
            "params": False,
        },
        {"name": "human_pose_tokenizer_decode", "object": model.vqvae, "attribute": "decode", "profile": True, "params": False},
        {"name": "root_head", "object": model.root_head, "attribute": "forward_with_components", "profile": True, "params": True},
        {"name": "mmwave_tokenizer", "object": model.mmwave_tokenizer, "profile": False, "params": True},
        {"name": "token_translator", "object": model.token_translator, "profile": False, "params": True},
        {"name": "human_pose_tokenizer_vqvae", "object": model.vqvae, "profile": False, "params": True},
        {"name": "loss", "object": None, "profile": False, "params": False},
        {"name": "backward_optimizer", "object": None, "profile": False, "params": False},
    ]

    def step(batch) -> int:
        model.train()
        _set_absolute_frozen_modules_eval(model)
        mmwave_points = batch["mmwave_points"].to(device, non_blocking=True)
        target_relative = batch["target_poses"].to(device, non_blocking=True)
        target_absolute = batch["target_poses_absolute"].to(device, non_blocking=True)
        batch_size = int(target_absolute.shape[0])
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(runtime):
            outputs = model(mmwave_points, target_relative)
            with torch.profiler.record_function("component:loss"):
                loss_dict = _absolute_loss_from_batch(
                    criterion=criterion,
                    outputs=outputs,
                    target_relative=target_relative,
                    target_absolute=target_absolute,
                    root_index=root_index,
                    training_cfg=training_cfg,
                )
                loss = loss_dict["total_loss"]
        with torch.profiler.record_function("component:backward_optimizer"):
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
                optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return batch_size

    train_batches = _num_train_batches(
        train_count=data_bundle.train_count,
        batch_size=int(config["training"]["batch_size"]),
        loader_len=len(data_bundle.train_loader),
        max_train_batches=max_train_batches,
    )
    profile = _profile_flops_streaming(
        step_fn=step,
        loader=data_bundle.train_loader,
        components=components,
        device=device,
        profile_batches=profile_batches,
        warmup_batches=warmup_batches,
        max_batches=max_train_batches,
        train_batches_per_epoch=train_batches,
    )
    epochs = int(config.get("training", {}).get("epochs", 1))
    return _finalize_stage_report(
        stage="absolute_root_head",
        config_path=config_path,
        config=config,
        model=model,
        components=components,
        profile=profile,
        train_count=data_bundle.train_count,
        train_batches_per_epoch=train_batches,
        epochs=epochs,
    )


def _profile_point_transformer(
    *,
    config_path: Path,
    config: dict[str, Any],
    device: torch.device,
    profile_batches: int,
    warmup_batches: int,
    max_train_batches: int | None,
) -> dict[str, Any]:
    training_cfg = config.get("training", {})
    set_global_seed(int(training_cfg.get("seed", config.get("random_seed", 42))))
    runtime = configure_runtime(training_cfg, device, default_amp=True, log_prefix="[FLOPs/PT]")
    data_bundle = build_radar_pose_dataloaders(
        config=config,
        checkpoint_dir=None,
        include_test=False,
        max_train_batches=max_train_batches,
        max_val_batches=1,
    )
    model = PTPoseModel(point_transformer_config=config["point_transformer"], device=str(device)).to(device)
    materialize_lazy_modules(model, config=config, device=device, runtime=runtime)
    criterion = PoseMPJPELoss().to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("lr", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    scaler = build_grad_scaler(runtime)
    grad_norm = float(training_cfg.get("grad_norm", 1.0))
    pt = model.pt
    components = [
        {"name": "pt_forward_unscoped", "object": pt, "attribute": "forward", "profile": True, "params": False},
        {"name": "point_transformer.backbone", "object": pt.backbone, "attribute": "forward", "profile": True, "params": True},
        {
            "name": "point_transformer.backbone_proj",
            "object": pt.backbone_proj,
            "attribute": "forward",
            "profile": True,
            "params": True,
        },
        {"name": "point_transformer.in_norm", "object": pt.in_norm, "attribute": "forward", "profile": True, "params": True},
        {
            "name": "point_transformer.transformer",
            "object": pt.transformer,
            "attribute": "forward",
            "profile": True,
            "params": True,
        },
        {"name": "point_transformer.out_norm", "object": pt.out_norm, "attribute": "forward", "profile": True, "params": True},
        {"name": "point_transformer.fc2", "object": pt.fc2, "attribute": "forward", "profile": True, "params": True},
        {"name": "point_transformer.fc3", "object": pt.fc3, "attribute": "forward", "profile": True, "params": True},
        {"name": "loss", "object": None, "profile": False, "params": False},
        {"name": "backward_optimizer", "object": None, "profile": False, "params": False},
    ]

    def step(batch) -> int:
        model.train()
        mmwave_points = batch["mmwave_points"].to(device, non_blocking=True)
        target_poses = batch["target_poses"].to(device, non_blocking=True)
        batch_size = int(mmwave_points.shape[0])
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(runtime):
            outputs = model(mmwave_points)
            with torch.profiler.record_function("component:loss"):
                loss_dict = criterion(outputs["pred_pose"].float(), target_poses.float())
                loss = loss_dict["total_loss"]
        with torch.profiler.record_function("component:backward_optimizer"):
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
                optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return batch_size

    train_batches = _num_train_batches(
        train_count=data_bundle.train_count,
        batch_size=int(config["training"]["batch_size"]),
        loader_len=len(data_bundle.train_loader),
        max_train_batches=max_train_batches,
    )
    profile = _profile_flops_streaming(
        step_fn=step,
        loader=data_bundle.train_loader,
        components=components,
        device=device,
        profile_batches=profile_batches,
        warmup_batches=warmup_batches,
        max_batches=max_train_batches,
        train_batches_per_epoch=train_batches,
    )
    epochs = int(config.get("training", {}).get("epochs", 1))
    return _finalize_stage_report(
        stage="point_transformer",
        config_path=config_path,
        config=config,
        model=model,
        components=components,
        profile=profile,
        train_count=data_bundle.train_count,
        train_batches_per_epoch=train_batches,
        epochs=epochs,
    )


def _profile_pose_baseline(
    *,
    stage: str,
    config_path: Path,
    config: dict[str, Any],
    device: torch.device,
    profile_batches: int | None,
    warmup_batches: int,
    max_train_batches: int | None,
) -> dict[str, Any]:
    training_cfg = config.get("training", {})
    set_global_seed(int(training_cfg.get("seed", config.get("random_seed", 42))))
    baseline_name = str(
        config.get("baseline", {}).get("name", config.get("model", {}).get("name", stage))
    ).strip().lower()
    runtime = configure_runtime(training_cfg, device, default_amp=True, log_prefix=f"[FLOPs/{baseline_name}]")
    data_bundle = build_radar_pose_dataloaders(
        config=config,
        checkpoint_dir=None,
        include_test=False,
        max_train_batches=max_train_batches,
        max_val_batches=1,
    )
    bundle = build_pose_baseline(config, device=str(device))
    bundle.model.to(device)
    optimizer = _pose_baseline_optimizer_from_config(bundle.model.parameters(), training_cfg)
    scaler = build_grad_scaler(runtime)
    grad_norm = float(training_cfg.get("grad_norm", training_cfg.get("gradient_clipping", 1.0)))
    components = [
        {
            "name": f"{baseline_name}_forward_unscoped",
            "object": bundle.model,
            "attribute": "forward",
            "profile": True,
            "params": False,
        },
        *_module_children_components(bundle.model, prefix=f"{baseline_name}.model"),
        {"name": "loss", "object": None, "profile": False, "params": False},
        {"name": "backward_optimizer", "object": None, "profile": False, "params": False},
    ]

    def step(batch) -> int:
        bundle.model.train()
        moved_batch = move_pose_baseline_batch(batch, device)
        batch_size = int(moved_batch["mmwave_points"].shape[0])
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(runtime):
            outputs = forward_pose_baseline(bundle, moved_batch)
            with torch.profiler.record_function("component:loss"):
                loss_dict = compute_pose_baseline_loss(bundle, outputs, moved_batch)
                loss = loss_dict["total_loss"]
        with torch.profiler.record_function("component:backward_optimizer"):
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(bundle.model.parameters(), grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(bundle.model.parameters(), grad_norm)
                optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return batch_size

    train_batches = _num_train_batches(
        train_count=data_bundle.train_count,
        batch_size=int(config["training"]["batch_size"]),
        loader_len=len(data_bundle.train_loader),
        max_train_batches=max_train_batches,
    )
    profile = _profile_flops_streaming(
        step_fn=step,
        loader=data_bundle.train_loader,
        components=components,
        device=device,
        profile_batches=profile_batches,
        warmup_batches=warmup_batches,
        max_batches=max_train_batches,
        train_batches_per_epoch=train_batches,
    )
    epochs = int(config.get("training", {}).get("epochs", 1))
    return _finalize_stage_report(
        stage=stage,
        config_path=config_path,
        config=config,
        model=bundle.model,
        components=components,
        profile=profile,
        train_count=data_bundle.train_count,
        train_batches_per_epoch=train_batches,
        epochs=epochs,
    )


def _fmt_si(value: float) -> str:
    abs_value = abs(float(value))
    for suffix, scale in (("EFLOPs", 1e18), ("PFLOPs", 1e15), ("TFLOPs", 1e12), ("GFLOPs", 1e9), ("MFLOPs", 1e6)):
        if abs_value >= scale:
            return f"{float(value) / scale:.3f} {suffix}"
    return f"{float(value):.3f} FLOPs"


def _top_components(flops: dict[str, Any], *, limit: int = 8) -> list[str]:
    shares = flops.get("component_share_sample_scaled") or flops.get("component_share", {})
    totals = flops.get("total_training_flops_by_component_sample_scaled") or flops.get("total_training_flops_by_component", {})
    items = sorted(totals.items(), key=lambda item: float(item[1]), reverse=True)
    out = []
    for name, value in items[:limit]:
        out.append(f"{name}={_fmt_si(float(value))} ({float(shares.get(name, 0.0)) * 100.0:.1f}%)")
    return out


def _comparison_table(stage_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for report in stage_reports:
        flops = report["flops"]
        params = report["params"]
        rows.append(
            {
                "Stage": report["stage"],
                "Trainable Params": int(params["trainable"]),
                "Total Params": int(params["total"]),
                "Train Batches/Epoch": int(flops["train_batches_per_epoch"]),
                "Train Samples/Epoch": flops.get("train_samples_per_epoch"),
                "Epochs": int(flops["epochs"]),
                "Avg Train Step FLOPs": float(flops["avg_train_step_flops"]),
                "Avg Train Sample FLOPs": float(flops["avg_train_sample_flops"]),
                "Total Training FLOPs": float(flops.get("total_training_flops_sample_scaled") or flops["total_training_flops"]),
                "Total Training FLOPs Step-Scaled": float(flops["total_training_flops"]),
                "Top FLOPs Components": "; ".join(_top_components(flops)),
            }
        )
    return rows


def _print_report(report: dict[str, Any]) -> None:
    print("\n[Training FLOPs] Summary")
    for row in report["comparison_table"]:
        print(
            f"- {row['Stage']}: total={_fmt_si(row['Total Training FLOPs'])}, "
            f"step={_fmt_si(row['Avg Train Step FLOPs'])}, "
            f"params(train/total)={row['Trainable Params']:,}/{row['Total Params']:,}, "
            f"steps={row['Train Batches/Epoch'] * row['Epochs']:,}"
        )
        if row["Top FLOPs Components"]:
            print(f"  components: {row['Top FLOPs Components']}")


def main() -> None:
    args = _build_parser().parse_args()
    profile_batches = _parse_profile_batches(args.profile_batches, profile_full_train=bool(args.profile_full_train))
    if int(args.warmup_batches) < 0:
        raise ValueError("--warmup-batches cannot be negative")
    if args.epochs is not None and int(args.epochs) <= 0:
        raise ValueError("--epochs must be positive when provided")

    stages = _parse_stages(args.stages)
    run_name = args.run_name or _default_run_name()
    work_root = _resolve_work_root(args, run_name)
    work_root.mkdir(parents=True, exist_ok=True)
    device = _device_from_arg(args.device)

    def batch(default: int | None, specific: int | None) -> int | None:
        return specific if specific is not None else (default if default is not None else args.batch_size)

    stage_configs: dict[str, tuple[Path, dict[str, Any]]] = {}
    for stage, path, stage_batch in (
        ("mmwave_tokenizer", args.mmwave_config, batch(None, args.mmwave_batch_size)),
        (
            "human_pose_tokenizer_stage_a",
            args.human_pose_stage_a_config,
            batch(args.human_pose_batch_size, args.human_pose_stage_a_batch_size),
        ),
        (
            "human_pose_tokenizer_stage_b",
            args.human_pose_stage_b_config,
            batch(args.human_pose_batch_size, args.human_pose_stage_b_batch_size),
        ),
        ("hpe_token", args.hpe_token_config, batch(None, args.hpe_token_batch_size)),
        ("absolute_root_head", args.absolute_root_config, batch(None, args.absolute_root_batch_size)),
        ("point_transformer", args.point_transformer_config, batch(None, args.point_transformer_batch_size)),
        ("mmdiff_m4human", args.mmdiff_m4human_config, batch(None, args.mmdiff_m4human_batch_size)),
        ("mmmesh_m4human", args.mmmesh_m4human_config, batch(None, args.mmmesh_m4human_batch_size)),
    ):
        if stage not in stages:
            continue
        config_path = Path(path).expanduser()
        stage_configs[stage] = (
            config_path,
            _prepare_config(
                load_config(config_path),
                batch_size=stage_batch,
                num_workers=args.num_workers,
                disable_compile=args.disable_compile,
                epochs=args.epochs,
            ),
        )

    reports: list[dict[str, Any]] = []
    for stage in stages:
        config_path, cfg = stage_configs[stage]
        print(f"[Training FLOPs] Profiling {stage} | config={config_path}")
        if stage == "mmwave_tokenizer":
            reports.append(
                _profile_mmwave_tokenizer(
                    config_path=config_path,
                    config=cfg,
                    device=device,
                    profile_batches=profile_batches,
                    warmup_batches=args.warmup_batches,
                    max_train_batches=args.max_train_batches,
                )
            )
        elif stage == "human_pose_tokenizer_stage_a":
            reports.append(
                _profile_human_pose_tokenizer(
                    stage=stage,
                    config_path=config_path,
                    config=cfg,
                    device=device,
                    profile_batches=profile_batches,
                    warmup_batches=args.warmup_batches,
                    max_train_batches=args.max_train_batches,
                )
            )
        elif stage == "human_pose_tokenizer_stage_b":
            reports.append(
                _profile_human_pose_tokenizer(
                    stage=stage,
                    config_path=config_path,
                    config=cfg,
                    device=device,
                    profile_batches=profile_batches,
                    warmup_batches=args.warmup_batches,
                    max_train_batches=args.max_train_batches,
                )
            )
        elif stage == "hpe_token":
            reports.append(
                _profile_hpe_token(
                    config_path=config_path,
                    config=cfg,
                    device=device,
                    profile_batches=profile_batches,
                    warmup_batches=args.warmup_batches,
                    max_train_batches=args.max_train_batches,
                    no_pretrained=args.no_pretrained,
                )
            )
        elif stage == "absolute_root_head":
            reports.append(
                _profile_absolute_root_head(
                    config_path=config_path,
                    config=cfg,
                    device=device,
                    profile_batches=profile_batches,
                    warmup_batches=args.warmup_batches,
                    max_train_batches=args.max_train_batches,
                    no_pretrained=args.no_pretrained,
                )
            )
        elif stage == "point_transformer":
            reports.append(
                _profile_point_transformer(
                    config_path=config_path,
                    config=cfg,
                    device=device,
                    profile_batches=profile_batches,
                    warmup_batches=args.warmup_batches,
                    max_train_batches=args.max_train_batches,
                )
            )
        elif stage in {"mmdiff_m4human", "mmmesh_m4human"}:
            reports.append(
                _profile_pose_baseline(
                    stage=stage,
                    config_path=config_path,
                    config=cfg,
                    device=device,
                    profile_batches=profile_batches,
                    warmup_batches=args.warmup_batches,
                    max_train_batches=args.max_train_batches,
                )
            )
        else:  # pragma: no cover - guarded by _parse_stages
            raise AssertionError(stage)

    wave2body_stage_reports = [item for item in reports if item["stage"] in WAVE2BODY_TRAINING_STAGES]
    wave2body_total = sum(
        float(item["flops"].get("total_training_flops_sample_scaled") or item["flops"]["total_training_flops"])
        for item in wave2body_stage_reports
    )
    report = {
        "benchmark": "training_flops_by_stage",
        "run_name": run_name,
        "work_root": str(work_root),
        "device": str(device),
        "methodology": {
            "profiler": "torch.profiler.profile(with_flops=True)",
            "total_training_flops": "primary table uses mean profiled train-sample FLOPs multiplied by train_samples_per_epoch and configured epochs; step-scaled totals are also stored",
            "captured_path": "forward + loss + backward + optimizer for real train batches",
            "note": (
                "PyTorch FLOPs are reported for supported operators such as matmul/conv. "
                "Component attribution follows nearest component:* profiler scope; backward is grouped under "
                "backward_optimizer because PyTorch backward ops are not reliably attributable to the original module. "
                "On CUDA, positive-FLOP events without a component parent are also folded into backward_optimizer."
            ),
        },
        "wave2body_total_training_flops": float(wave2body_total),
        "wave2body_total_training_flops_text": _fmt_si(wave2body_total),
        "stages": reports,
        "comparison_table": _comparison_table(reports),
    }
    output_json = Path(args.output_json).expanduser() if args.output_json else work_root / "training_flops_summary.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    report["summary_path"] = str(output_json)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_report(report)
    print(f"[Training FLOPs] Wrote summary: {output_json}")


if __name__ == "__main__":
    main()
