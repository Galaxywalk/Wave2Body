# -*- coding: utf-8 -*-
"""Soft token CE based on part-level MPJPE regret."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_tuple(value: Iterable[str]) -> tuple[str, ...]:
    return tuple(str(x) for x in value)


def load_soft_distance_table(path: str | Path) -> dict[str, np.ndarray]:
    """Load the soft-token NPZ artifact and validate required arrays."""
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(f"Soft target table not found: {table_path}")

    data = np.load(table_path, allow_pickle=False)
    required = {"distance_mm", "slot_valid", "observed_counts"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise KeyError(f"Soft target table missing keys: {missing}")
    return {key: data[key] for key in data.files}


class SoftMPJPETokenCELoss(nn.Module):
    """
    Soft token CE over each part-local K-way codebook slice.

    `pred_logits` still has the existing global shape [B, N, P*K], but for each
    token this loss only consumes the valid local slice belonging to that token's
    body part. The soft target row is selected by the target local code.
    """

    def __init__(
        self,
        *,
        layout,
        table_path: str | Path,
        tau_mm: float = 40.0,
        d_clip_mm: float = 150.0,
        valid_thresholds_mm: tuple[float, ...] = (20.0, 40.0, 80.0),
        eps: float = 1e-8,
    ):
        super().__init__()
        if tau_mm <= 0:
            raise ValueError(f"tau_mm must be > 0, got {tau_mm}")
        if d_clip_mm <= 0:
            raise ValueError(f"d_clip_mm must be > 0, got {d_clip_mm}")

        payload = load_soft_distance_table(table_path)
        distance_mm = np.asarray(payload["distance_mm"], dtype=np.float32)
        slot_valid = np.asarray(payload["slot_valid"], dtype=np.bool_)
        observed_counts = np.asarray(payload["observed_counts"], dtype=np.int64)

        if distance_mm.ndim != 4:
            raise ValueError(f"distance_mm must be [P,S,K,K], got {distance_mm.shape}")
        if slot_valid.shape != distance_mm.shape[:2]:
            raise ValueError(
                f"slot_valid shape {slot_valid.shape} does not match distance prefix {distance_mm.shape[:2]}"
            )
        if observed_counts.shape != distance_mm.shape[:3]:
            raise ValueError(
                f"observed_counts shape {observed_counts.shape} does not match distance prefix {distance_mm.shape[:3]}"
            )

        self.layout = layout
        self.part_names = _as_tuple(layout.part_names)
        self.tau_mm = float(tau_mm)
        self.d_clip_mm = float(d_clip_mm)
        self.valid_thresholds_mm = tuple(float(x) for x in valid_thresholds_mm)
        self.eps = float(eps)

        table_part_count, table_slot_count, table_k, table_k2 = distance_mm.shape
        if table_k != table_k2:
            raise ValueError(f"distance_mm last dims must be square KxK, got {distance_mm.shape[-2:]}")
        if table_part_count != len(self.part_names):
            raise ValueError(f"Table has {table_part_count} parts, layout has {len(self.part_names)}")

        codebook_sizes = [int(layout.part_codebook_size[name]) for name in self.part_names]
        if len(set(codebook_sizes)) != 1:
            raise ValueError(f"Soft loss currently requires uniform per-part K, got {codebook_sizes}")
        if int(codebook_sizes[0]) != int(table_k):
            raise ValueError(f"Table K={table_k} does not match layout K={codebook_sizes[0]}")

        max_slots = max(int(layout.part_token_num[name]) for name in self.part_names)
        if table_slot_count < max_slots:
            raise ValueError(f"Table S={table_slot_count} is smaller than layout max slots={max_slots}")

        if np.nanmin(distance_mm) < -1e-6:
            raise ValueError("distance_mm contains negative values")

        self.k_local = int(table_k)
        self.register_buffer("distance_mm", torch.from_numpy(np.maximum(distance_mm, 0.0)))
        self.register_buffer("slot_valid", torch.from_numpy(slot_valid.astype(np.bool_)))
        self.register_buffer("observed_counts", torch.from_numpy(observed_counts))

    def _soft_targets(self, distances_mm: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(distances_mm, min=0.0, max=self.d_clip_mm)
        return F.softmax(-clipped / self.tau_mm, dim=-1)

    def forward(
        self,
        *,
        pred_logits: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if pred_logits is None:
            raise ValueError("pred_logits is required for soft token CE")
        if target_indices is None:
            raise ValueError("target_indices is required for soft token CE")
        if pred_logits.dim() != 3:
            raise ValueError(f"pred_logits must be [B,N,C], got {tuple(pred_logits.shape)}")
        if target_indices.dim() != 2:
            raise ValueError(f"target_indices must be [B,N], got {tuple(target_indices.shape)}")
        if pred_logits.shape[:2] != target_indices.shape:
            raise ValueError(
                f"pred_logits prefix {tuple(pred_logits.shape[:2])} does not match target_indices {tuple(target_indices.shape)}"
            )

        device = pred_logits.device
        dtype = pred_logits.dtype
        total_loss = torch.zeros((), device=device, dtype=dtype)
        valid_count = torch.zeros((), device=device, dtype=dtype)
        exact_count = torch.zeros((), device=device, dtype=dtype)
        regret_sum = torch.zeros((), device=device, dtype=dtype)
        soft_valid_counts = {
            float(th): torch.zeros((), device=device, dtype=dtype)
            for th in self.valid_thresholds_mm
        }

        batch_indices = torch.arange(pred_logits.shape[0], device=device)

        for part_idx, part_name in enumerate(self.part_names):
            token_slice = self.layout.part_token_slices[part_name]
            code_offset = int(self.layout.part_codebook_offsets[part_name])
            n_slots = int(self.layout.part_token_num[part_name])

            for slot_idx in range(n_slots):
                if not bool(self.slot_valid[part_idx, slot_idx].item()):
                    continue

                token_idx = int(token_slice.start) + slot_idx
                logits_local = pred_logits[:, token_idx, code_offset : code_offset + self.k_local]
                target_local = target_indices[:, token_idx] - code_offset
                valid_mask = (target_local >= 0) & (target_local < self.k_local)
                if not bool(valid_mask.any().item()):
                    continue

                target_local_valid = target_local[valid_mask].long()
                logits_valid = logits_local[valid_mask]
                distances = self.distance_mm[part_idx, slot_idx].to(device=device, dtype=dtype)
                selected_distances = distances.index_select(0, target_local_valid)
                soft_targets = self._soft_targets(selected_distances)
                log_probs = F.log_softmax(logits_valid, dim=-1)
                loss_vec = -(soft_targets * log_probs).sum(dim=-1)

                pred_local = torch.argmax(logits_valid, dim=-1)
                local_batch = batch_indices[valid_mask]
                pred_regret = selected_distances[torch.arange(pred_local.numel(), device=device), pred_local]

                total_loss = total_loss + loss_vec.sum()
                token_count = torch.as_tensor(float(loss_vec.numel()), device=device, dtype=dtype)
                valid_count = valid_count + token_count
                exact_count = exact_count + (pred_local == target_local_valid).to(dtype).sum()
                regret_sum = regret_sum + pred_regret.sum()
                for threshold_mm in self.valid_thresholds_mm:
                    soft_valid_counts[float(threshold_mm)] = soft_valid_counts[float(threshold_mm)] + (
                        pred_regret <= float(threshold_mm)
                    ).to(dtype).sum()

                # Keep local_batch referenced so shape regressions surface in tests.
                if local_batch.numel() != pred_local.numel():
                    raise RuntimeError("Internal valid-mask batch indexing mismatch")

        denom = torch.clamp(valid_count, min=1.0)
        loss = total_loss / denom
        out: Dict[str, torch.Tensor] = {
            "soft_token_loss": loss,
            "soft_token_valid_count": valid_count,
            "soft_token_exact_acc": exact_count / denom,
            "soft_token_pred_regret_mm": regret_sum / denom,
        }
        for threshold_mm, count in soft_valid_counts.items():
            key = f"soft_valid_acc_{int(threshold_mm)}mm"
            out[key] = count / denom
        return out


def verify_soft_target_rows(
    distance_mm: np.ndarray,
    slot_valid: np.ndarray,
    *,
    tau_mm: float,
    d_clip_mm: float,
    atol: float = 1e-5,
) -> dict[str, float | bool]:
    """Verify row-normalization after the softmax transform."""
    if tau_mm <= 0:
        raise ValueError(f"tau_mm must be > 0, got {tau_mm}")
    valid = np.asarray(slot_valid, dtype=bool)
    distances = np.asarray(distance_mm, dtype=np.float64)
    clipped = np.clip(distances, 0.0, float(d_clip_mm))
    logits = -clipped / float(tau_mm)
    logits = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.clip(probs.sum(axis=-1, keepdims=True), a_min=1e-12, a_max=None)

    row_sums = probs.sum(axis=-1)
    valid_row_sums = row_sums[valid]
    if valid_row_sums.size == 0:
        return {"ok": False, "max_abs_error": float("inf")}
    max_abs_error = float(np.max(np.abs(valid_row_sums - 1.0)))
    return {"ok": bool(max_abs_error <= float(atol)), "max_abs_error": max_abs_error}
