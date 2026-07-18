# -*- coding: utf-8 -*-
"""Runtime/performance helpers shared by training and evaluation scripts."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RuntimeConfig:
    device: torch.device
    precision: str
    amp_enabled: bool
    autocast_dtype: Optional[torch.dtype]
    use_grad_scaler: bool
    deterministic: bool
    tf32: bool
    matmul_precision: str
    compile_enabled: bool
    compile_mode: Optional[str]
    compile_dynamic: bool
    compile_fullgraph: bool


def _supported_precision(precision: str) -> str:
    precision = str(precision).strip().lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"Unsupported precision `{precision}`. Expected one of: fp32/fp16/bf16")
    return precision


def _default_amp_precision() -> str:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bf16"
    return "fp16"


def configure_runtime(
    training_cfg: dict,
    device: torch.device,
    *,
    default_amp: bool = False,
    log_prefix: str = "[Runtime]",
) -> RuntimeConfig:
    """
    Resolve precision/AMP/TF32/compile policy and apply backend switches.

    Behavior:
    - `training.precision` has highest priority (fp32/fp16/bf16).
    - `training.amp=false` forces fp32 even when precision requests fp16/bf16.
    - if precision is absent, uses `training.amp` or `default_amp`.
    """
    requested_precision = training_cfg.get("precision", None)
    amp_cfg = training_cfg.get("amp", None)

    if requested_precision is not None:
        requested_precision = _supported_precision(requested_precision)
        if requested_precision != "fp32" and amp_cfg is False:
            print(
                f"{log_prefix} precision={requested_precision} but amp=false; "
                "falling back to fp32."
            )
            precision = "fp32"
        else:
            precision = requested_precision
    else:
        use_amp = bool(amp_cfg) if amp_cfg is not None else bool(default_amp)
        precision = _default_amp_precision() if use_amp else "fp32"

    # CUDA-only mixed precision.
    if device.type != "cuda" and precision != "fp32":
        print(f"{log_prefix} Non-CUDA device detected, forcing precision=fp32.")
        precision = "fp32"

    if precision == "bf16" and device.type == "cuda" and (not torch.cuda.is_bf16_supported()):
        print(f"{log_prefix} bf16 not supported on this device/runtime, falling back to fp16.")
        precision = "fp16"

    amp_enabled = device.type == "cuda" and precision in {"fp16", "bf16"}
    autocast_dtype = (
        torch.float16 if precision == "fp16" else
        torch.bfloat16 if precision == "bf16" else
        None
    )
    use_grad_scaler = bool(amp_enabled and precision == "fp16")

    deterministic = bool(training_cfg.get("deterministic", False))
    tf32 = bool(training_cfg.get("tf32", True))
    matmul_precision = str(training_cfg.get("matmul_precision", "high"))
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision(matmul_precision)
        except Exception as exc:
            print(f"{log_prefix} Failed to set matmul precision `{matmul_precision}`: {exc}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        torch.backends.cudnn.deterministic = deterministic
        if deterministic:
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = bool(training_cfg.get("cudnn_benchmark", True))

    compile_cfg = training_cfg.get("compile", {})
    if isinstance(compile_cfg, bool):
        compile_enabled = bool(compile_cfg)
        compile_mode = "reduce-overhead"
        compile_dynamic = False
        compile_fullgraph = False
    else:
        compile_enabled = bool(compile_cfg.get("enabled", False))
        compile_mode = str(compile_cfg.get("mode", "reduce-overhead"))
        compile_dynamic = bool(compile_cfg.get("dynamic", False))
        compile_fullgraph = bool(compile_cfg.get("fullgraph", False))

    runtime = RuntimeConfig(
        device=device,
        precision=precision,
        amp_enabled=amp_enabled,
        autocast_dtype=autocast_dtype,
        use_grad_scaler=use_grad_scaler,
        deterministic=deterministic,
        tf32=tf32,
        matmul_precision=matmul_precision,
        compile_enabled=compile_enabled,
        compile_mode=compile_mode,
        compile_dynamic=compile_dynamic,
        compile_fullgraph=compile_fullgraph,
    )
    print(
        f"{log_prefix} precision={runtime.precision}, amp={runtime.amp_enabled}, "
        f"grad_scaler={runtime.use_grad_scaler}, tf32={runtime.tf32}, "
        f"matmul_precision={runtime.matmul_precision}, compile={runtime.compile_enabled}"
    )
    return runtime


def build_grad_scaler(runtime: RuntimeConfig):
    if not runtime.use_grad_scaler:
        class _DisabledScaler:
            @staticmethod
            def is_enabled() -> bool:
                return False

            @staticmethod
            def scale(loss):
                return loss

            @staticmethod
            def step(optimizer):
                optimizer.step()

            @staticmethod
            def update():
                return None

            @staticmethod
            def unscale_(_optimizer):
                return None

            @staticmethod
            def state_dict() -> dict:
                return {}

            @staticmethod
            def load_state_dict(_state):
                return None

        return _DisabledScaler()

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)


@contextmanager
def autocast_context(runtime: RuntimeConfig):
    if runtime.amp_enabled:
        with torch.autocast(
            device_type=runtime.device.type,
            dtype=runtime.autocast_dtype,
            enabled=True,
        ):
            yield
    else:
        yield


def maybe_compile(module, runtime: RuntimeConfig, *, module_name: str = "model", log_prefix: str = "[Runtime]"):
    if not runtime.compile_enabled:
        return module
    if not hasattr(torch, "compile"):
        print(f"{log_prefix} torch.compile unavailable, skip compile for {module_name}.")
        return module
    try:
        compiled = torch.compile(
            module,
            mode=runtime.compile_mode,
            dynamic=runtime.compile_dynamic,
            fullgraph=runtime.compile_fullgraph,
        )
        print(
            f"{log_prefix} compiled {module_name} with mode={runtime.compile_mode}, "
            f"dynamic={runtime.compile_dynamic}, fullgraph={runtime.compile_fullgraph}"
        )
        return compiled
    except Exception as exc:
        print(f"{log_prefix} failed to compile {module_name}: {exc}. Using eager mode.")
        return module


def mark_compiled_step(runtime: RuntimeConfig) -> None:
    """Mark iteration boundaries for torch.compile CUDA graph capture when available."""
    if not getattr(runtime, "compile_enabled", False):
        return
    compiler = getattr(torch, "compiler", None)
    marker = getattr(compiler, "cudagraph_mark_step_begin", None) if compiler is not None else None
    if marker is not None:
        marker()


def dataset_cache_mode(dataset) -> str:
    """
    Resolve cache mode from wave2body.dataset or nested subset wrappers.
    """
    target = dataset
    visited = set()
    while target is not None and id(target) not in visited:
        visited.add(id(target))
        mode = getattr(target, "cache_mode", None)
        if mode is not None:
            return str(mode)
        base = getattr(target, "base_dataset", None)
        if base is not None:
            target = base
            continue
        nested = getattr(target, "dataset", None)
        target = nested
    return "none"
