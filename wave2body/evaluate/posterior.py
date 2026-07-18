# -*- coding: utf-8 -*-
"""Shared posterior candidate utilities for part-token models."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError(f"Failed to parse non-empty int list from: {text}")
    return values


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError(f"Failed to parse non-empty float list from: {text}")
    return values


def sample_confidence(part_conf: torch.Tensor, layout, agg: str) -> torch.Tensor:
    if agg == "mean":
        return part_conf.mean(dim=1)
    if agg == "min":
        return part_conf.min(dim=1).values
    if agg == "weighted_mean":
        weights = torch.tensor(
            [len(layout.part_joints[name]) for name in layout.part_names],
            dtype=part_conf.dtype,
            device=part_conf.device,
        )
        return (part_conf * weights.unsqueeze(0)).sum(dim=1) / torch.clamp(weights.sum(), min=1e-8)
    raise ValueError(f"Unsupported sample-confidence aggregation: {agg}")


def part_confidence_to_joint_confidence(part_conf: torch.Tensor, layout) -> torch.Tensor:
    bsz = int(part_conf.shape[0])
    num_joints = int(layout.num_joints)
    out_sum = torch.zeros((bsz, num_joints), dtype=part_conf.dtype, device=part_conf.device)
    out_cnt = torch.zeros((bsz, num_joints), dtype=part_conf.dtype, device=part_conf.device)
    for pid, name in enumerate(layout.part_names):
        idx = torch.as_tensor(layout.part_joints[name], dtype=torch.long, device=part_conf.device)
        out_sum[:, idx] += part_conf[:, pid].unsqueeze(1)
        out_cnt[:, idx] += 1.0
    return out_sum / torch.clamp(out_cnt, min=1.0)


def decode_from_global_indices(model, indices_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(model.vqvae, "layout") or not hasattr(model.vqvae, "quantizers"):
        raise RuntimeError("Expected part-level VQ-VAE with layout + quantizers.")
    layout = model.vqvae.layout
    bsz = int(indices_global.shape[0])
    part_idx_local = layout.split_global_indices(indices_global.long())
    part_tokens_q: dict[str, torch.Tensor] = {}
    for name in layout.part_names:
        idx_local = part_idx_local[name].long()
        z_q = model.vqvae.quantizers[name].dequantize(idx_local.reshape(-1)).view(
            bsz, -1, int(model.vqvae.token_dim)
        )
        part_tokens_q[name] = z_q
    tokens_q = layout.concat_part_tokens(part_tokens_q)
    pose = model.vqvae.decode(tokens_q)
    return tokens_q, pose


def candidate_indices_and_nll(
    *,
    logits_global: torch.Tensor,
    layout,
    temperature: float,
    sample_mode: bool,
    generator: Optional[torch.Generator],
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits_global.dim() != 3:
        raise ValueError(f"Expected logits [B,N,C], got {tuple(logits_global.shape)}")
    bsz = int(logits_global.shape[0])
    part_idx_local: dict[str, torch.Tensor] = {}
    nll_terms: list[torch.Tensor] = []

    tau = max(float(temperature), 1e-6)
    for name in layout.part_names:
        tok_slice = layout.part_token_slices[name]
        start = int(layout.part_codebook_offsets[name])
        end = start + int(layout.part_codebook_size[name])
        logits_part = logits_global[:, tok_slice, start:end] / tau
        log_probs = torch.log_softmax(logits_part, dim=-1)

        if sample_mode:
            probs_2d = torch.softmax(logits_part, dim=-1).reshape(-1, logits_part.shape[-1])
            idx_flat = torch.multinomial(
                probs_2d,
                num_samples=1,
                replacement=True,
                generator=generator,
            ).squeeze(1)
            idx_local = idx_flat.view(bsz, -1)
        else:
            idx_local = torch.argmax(log_probs, dim=-1)

        chosen_logp = log_probs.gather(dim=-1, index=idx_local.unsqueeze(-1)).squeeze(-1)
        nll_terms.append(-chosen_logp.mean(dim=1))
        part_idx_local[name] = idx_local.long()

    indices_global = layout.merge_local_indices_to_global(part_idx_local).long()
    seq_nll = torch.stack(nll_terms, dim=1).mean(dim=1)
    return indices_global, seq_nll


def gather_candidate(cands: torch.Tensor, sel_idx: torch.Tensor) -> torch.Tensor:
    bsz = int(sel_idx.shape[0])
    bidx = torch.arange(bsz, device=sel_idx.device, dtype=torch.long)
    return cands[sel_idx, bidx]


def select_candidate_indices(
    *,
    selector: str,
    cand_sample_mpjpe_mm: torch.Tensor,
    cand_sample_conf: torch.Tensor,
    cand_seq_nll: torch.Tensor,
    random_state: np.random.Generator,
) -> torch.Tensor:
    k = int(cand_sample_mpjpe_mm.shape[0])
    bsz = int(cand_sample_mpjpe_mm.shape[1])
    device = cand_sample_mpjpe_mm.device

    if selector == "argmax":
        return torch.zeros((bsz,), device=device, dtype=torch.long)
    if selector == "oracle_best_of_k":
        return torch.argmin(cand_sample_mpjpe_mm, dim=0).long()
    if selector == "confidence_selected_of_k":
        return torch.argmax(cand_sample_conf, dim=0).long()
    if selector == "entropy_selected_of_k":
        return torch.argmin(cand_seq_nll, dim=0).long()
    if selector == "random_of_k":
        idx_np = random_state.integers(low=0, high=k, size=(bsz,), endpoint=False)
        return torch.as_tensor(idx_np, dtype=torch.long, device=device)
    raise ValueError(f"Unsupported selector: {selector}")


__all__ = [
    "candidate_indices_and_nll",
    "decode_from_global_indices",
    "gather_candidate",
    "parse_float_list",
    "parse_int_list",
    "part_confidence_to_joint_confidence",
    "sample_confidence",
    "select_candidate_indices",
]
