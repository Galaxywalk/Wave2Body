# -*- coding: utf-8 -*-
"""Decode part-token logits through a frozen part VQ-VAE codebook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

DecodeMode = Literal["argmax", "argmax_st", "sample_st", "soft_expectation"]


@dataclass
class LogitTokenDecodeOutput:
    z_q: torch.Tensor
    indices: torch.Tensor
    pose: torch.Tensor
    probs: torch.Tensor
    entropy: torch.Tensor


def _one_hot(indices: torch.Tensor, num_classes: int, dtype: torch.dtype) -> torch.Tensor:
    return F.one_hot(indices.long(), num_classes=int(num_classes)).to(dtype=dtype)


def _local_weights(
    logits_local: torch.Tensor,
    *,
    mode: DecodeMode,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tau = max(float(temperature), 1e-6)
    logits_fp32 = logits_local.float()
    probs = torch.softmax(logits_fp32 / tau, dim=-1)
    entropy = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=-1)

    if mode in {"argmax", "argmax_st"}:
        idx = torch.argmax(probs, dim=-1)
        hard = _one_hot(idx, probs.shape[-1], probs.dtype)
        weights = hard if mode == "argmax" else hard + probs - probs.detach()
        return weights, idx, probs, entropy

    if mode == "sample_st":
        weights = F.gumbel_softmax(logits_fp32, tau=tau, hard=True, dim=-1)
        idx = torch.argmax(weights, dim=-1)
        return weights, idx, probs, entropy

    if mode == "soft_expectation":
        idx = torch.argmax(probs, dim=-1)
        return probs, idx, probs, entropy

    raise ValueError(f"Unsupported logits decode mode: {mode}")


def decode_part_token_logits(
    *,
    vqvae,
    pred_logits: torch.Tensor,
    mode: DecodeMode = "argmax_st",
    temperature: float = 1.0,
) -> LogitTokenDecodeOutput:
    """Decode global part-token logits with body-part-local codebook slices.

    The translator emits global logits ``[B, T, sum(K_part)]`` for historical
    compatibility, but each token may only choose codes from its own body part.
    This helper masks the choice by slicing each part's legal range before
    producing codebook embeddings.
    """
    if pred_logits is None:
        raise ValueError("pred_logits is required for logits-first decoding")
    if pred_logits.dim() != 3:
        raise ValueError(f"Expected pred_logits [B,T,C], got {tuple(pred_logits.shape)}")
    if not hasattr(vqvae, "layout") or not hasattr(vqvae, "quantizers"):
        raise RuntimeError("Expected a part-level VQ-VAE with layout and quantizers.")

    layout = vqvae.layout
    batch_size = int(pred_logits.shape[0])
    if int(pred_logits.shape[1]) != int(layout.total_tokens):
        raise ValueError(f"Token count mismatch: logits={pred_logits.shape[1]}, layout={layout.total_tokens}")

    part_tokens: dict[str, torch.Tensor] = {}
    part_indices_local: dict[str, torch.Tensor] = {}
    token_entropy = pred_logits.new_zeros((batch_size, int(layout.total_tokens)), dtype=torch.float32)
    global_probs = torch.zeros_like(pred_logits)

    for part_name in layout.part_names:
        token_slice = layout.part_token_slices[part_name]
        offset = int(layout.part_codebook_offsets[part_name])
        codebook_size = int(layout.part_codebook_size[part_name])
        end = offset + codebook_size
        if offset < 0 or codebook_size <= 0 or end > int(pred_logits.shape[-1]):
            raise ValueError(
                f"Invalid codebook slice for {part_name}: offset={offset}, size={codebook_size}, "
                f"logits_dim={int(pred_logits.shape[-1])}"
            )

        logits_local = pred_logits[:, token_slice, offset:end]
        weights, idx_local, probs_local, entropy = _local_weights(
            logits_local,
            mode=mode,
            temperature=temperature,
        )

        codebook = vqvae.quantizers[part_name].codebook.to(device=pred_logits.device)
        z_part = torch.matmul(weights.to(dtype=codebook.dtype), codebook)
        part_tokens[part_name] = z_part
        part_indices_local[part_name] = idx_local.long()
        token_entropy[:, token_slice] = entropy
        global_probs[:, token_slice, offset:end] = probs_local.to(dtype=pred_logits.dtype)

    z_q = layout.concat_part_tokens(part_tokens)
    indices = layout.merge_local_indices_to_global(part_indices_local).long()
    pose = vqvae.decode(z_q)
    return LogitTokenDecodeOutput(
        z_q=z_q,
        indices=indices,
        pose=pose,
        probs=global_probs,
        entropy=token_entropy,
    )


def part_codebook_distance_logits(
    *,
    vqvae,
    pred_z_e: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Convert continuous part tokens to global logits from VQ codebook distances.

    For each body-part-local codebook, this returns ``-scale * ||z_e - e_k||^2``
    in the valid global code slice. Argmax decoding of these logits is therefore
    equivalent to nearest-codebook quantization, while still exposing a full
    posterior-logit interface.
    """
    if pred_z_e.dim() != 3:
        raise ValueError(f"Expected pred_z_e [B,T,D], got {tuple(pred_z_e.shape)}")
    if not hasattr(vqvae, "layout") or not hasattr(vqvae, "quantizers"):
        raise RuntimeError("Expected a part-level VQ-VAE with layout and quantizers.")
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")

    layout = vqvae.layout
    batch_size, token_count, _ = pred_z_e.shape
    if int(token_count) != int(layout.total_tokens):
        raise ValueError(f"Token count mismatch: pred_z_e={token_count}, layout={layout.total_tokens}")

    total_codes = int(sum(int(layout.part_codebook_size[name]) for name in layout.part_names))
    logits = pred_z_e.new_full((int(batch_size), int(token_count), total_codes), float("-inf"), dtype=torch.float32)
    z_e_fp32 = pred_z_e.float()
    scale_value = float(scale)

    for part_name in layout.part_names:
        token_slice = layout.part_token_slices[part_name]
        offset = int(layout.part_codebook_offsets[part_name])
        codebook_size = int(layout.part_codebook_size[part_name])
        end = offset + codebook_size
        codebook = vqvae.quantizers[part_name].codebook.to(device=pred_z_e.device, dtype=torch.float32)
        z_part = z_e_fp32[:, token_slice, :]
        dist = (
            torch.sum(z_part * z_part, dim=-1, keepdim=True)
            + torch.sum(codebook * codebook, dim=-1)
            - 2.0 * torch.matmul(z_part, codebook.t())
        )
        logits[:, token_slice, offset:end] = -scale_value * dist

    return logits


__all__ = ["DecodeMode", "LogitTokenDecodeOutput", "decode_part_token_logits", "part_codebook_distance_logits"]
