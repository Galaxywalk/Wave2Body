# -*- coding: utf-8 -*-
"""Checkpoint helpers shared by staged training pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from ..utils.paths import resolve_runtime_uri


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint_path = Path(resolve_runtime_uri(str(path))).expanduser()
    return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def _looks_like_state_dict(obj: Any) -> bool:
    if not isinstance(obj, Mapping) or not obj:
        return False
    return all(torch.is_tensor(v) for v in obj.values())


def _strip_prefix(state_dict: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _strip_wrapping_prefixes(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = dict(state_dict)
    for prefix in ("module.", "_orig_mod."):
        if out and all(k.startswith(prefix) for k in out):
            out = {k[len(prefix) :]: v for k, v in out.items()}
    return out


def normalized_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return a state dict without DataParallel/torch.compile wrapper prefixes."""
    out = _strip_wrapping_prefixes(state_dict)
    return {key.replace("._orig_mod.", "."): value for key, value in out.items()}


def module_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return a checkpoint-friendly module state dict.

    ``torch.compile`` wraps the original module and prefixes state-dict keys with
    ``_orig_mod.``. Saving normalized keys keeps checkpoints loadable by eager
    modules and compiled modules alike.
    """
    return normalized_state_dict(module.state_dict())


def extract_point_transformer_state_dict(source: str | Path | Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Extract a PointTransformer state dict from current and historical checkpoints."""
    ckpt: Mapping[str, Any]
    if isinstance(source, (str, Path)):
        ckpt = load_checkpoint(source)
    else:
        ckpt = source

    for key in ("point_transformer_state_dict", "pt_state_dict"):
        state = ckpt.get(key) if isinstance(ckpt, Mapping) else None
        if _looks_like_state_dict(state):
            return _strip_wrapping_prefixes(state)

    candidate = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, Mapping) else ckpt
    if not _looks_like_state_dict(candidate):
        raise KeyError("Checkpoint does not contain a usable PointTransformer state dict")

    candidate = _strip_wrapping_prefixes(candidate)
    for prefix in ("pt.", "module.pt.", "point_transformer.", "module.point_transformer."):
        stripped = _strip_prefix(candidate, prefix)
        if stripped:
            return _strip_wrapping_prefixes(stripped)
    return dict(candidate)


def load_state_dict_shape_safe(
    module: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    module_name: str = "module",
    strict: bool = False,
) -> dict[str, Any]:
    """Load matching checkpoint tensors and skip incompatible shapes."""
    state_dict = normalized_state_dict(state_dict)
    model_state = module.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped: list[tuple[str, tuple[int, ...], tuple[int, ...] | None]] = []
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        try:
            dst_shape = tuple(model_state[key].shape)
        except RuntimeError:
            dst_shape = None
        src_shape = tuple(value.shape)
        if dst_shape is not None and src_shape != dst_shape:
            skipped.append((key, src_shape, dst_shape))
            continue
        filtered[key] = value

    msg = module.load_state_dict(filtered, strict=strict)
    report = {
        "loaded": len(filtered),
        "missing": list(msg.missing_keys),
        "unexpected": list(msg.unexpected_keys),
        "shape_skipped": skipped,
    }
    print(
        f"[{module_name}] loaded={report['loaded']} | "
        f"missing={len(report['missing'])} | unexpected={len(report['unexpected'])} | "
        f"shape_skipped={len(skipped)}"
    )
    for key, src_shape, dst_shape in skipped[:8]:
        print(f"[{module_name}] skip shape mismatch: {key} ckpt{src_shape} -> model{dst_shape}")
    if len(skipped) > 8:
        print(f"[{module_name}] ... and {len(skipped) - 8} more mismatched keys")
    return report


def load_point_transformer_checkpoint(
    module: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = False,
) -> dict[str, Any]:
    state = extract_point_transformer_state_dict(load_checkpoint(checkpoint_path, map_location=map_location))
    return load_state_dict_shape_safe(module, state, module_name="PointTransformer", strict=strict)


def load_optional_component_states(
    model: torch.nn.Module,
    ckpt: Mapping[str, Any],
    *,
    prefix: str = "",
) -> None:
    """Load sensor encoder, VQ-VAE, and token-translator states when present."""
    tag = f"{prefix} " if prefix else ""
    targets = (
        ("point_transformer_state_dict", "point_transformer"),
        ("mmwave_tokenizer_state_dict", "mmwave_tokenizer"),
        ("sensor_encoder_state_dict", "sensor_encoder"),
        ("vqvae_state_dict", "vqvae"),
        ("token_translator_state_dict", "token_translator"),
        ("root_head_state_dict", "root_head"),
    )
    for state_key, attr in targets:
        module = getattr(model, attr, None)
        if module is None or state_key not in ckpt:
            continue
        load_state_dict_shape_safe(module, ckpt[state_key], module_name=f"{tag}{state_key}", strict=False)


def save_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), output)
    return output
