# -*- coding: utf-8 -*-
"""Anatomical geometry, sparsity stratification, and causal temporal smoothness diagnostics."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..dataset.radar_pose import M4HumanLMDBPoseDataset, MMBodyDataset
from ..dataset.radar_pose.root_alignment import apply_input_root_alignment
from ..utils.metrics import compute_per_sample_p_mpjpe
from ..visualization.pose import SMPL22_BONE_EDGES


ALL_BONE_EDGES: tuple[tuple[int, int], ...] = tuple(SMPL22_BONE_EDGES)
LIMB_BONE_EDGES: tuple[tuple[int, int], ...] = (
    (1, 4),
    (2, 5),
    (4, 7),
    (5, 8),
    (16, 18),
    (17, 19),
    (18, 20),
    (19, 21),
)
DISTAL_BONE_EDGES: tuple[tuple[int, int], ...] = (
    (4, 7),
    (5, 8),
    (18, 20),
    (19, 21),
)
BILATERAL_BONE_PAIRS: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
    ((0, 1), (0, 2)),
    ((1, 4), (2, 5)),
    ((4, 7), (5, 8)),
    ((7, 10), (8, 11)),
    ((12, 13), (12, 14)),
    ((13, 16), (14, 17)),
    ((16, 18), (17, 19)),
    ((18, 20), (19, 21)),
)

_EDGE_TO_IDX = {edge: idx for idx, edge in enumerate(ALL_BONE_EDGES)}
_LIMB_IDX = [_EDGE_TO_IDX[edge] for edge in LIMB_BONE_EDGES]
_DISTAL_IDX = [_EDGE_TO_IDX[edge] for edge in DISTAL_BONE_EDGES]
_LEFT_IDX = [_EDGE_TO_IDX[pair[0]] for pair in BILATERAL_BONE_PAIRS]
_RIGHT_IDX = [_EDGE_TO_IDX[pair[1]] for pair in BILATERAL_BONE_PAIRS]
_U_IDX = [edge[0] for edge in ALL_BONE_EDGES]
_V_IDX = [edge[1] for edge in ALL_BONE_EDGES]

MMBODY_SCENARIOS: tuple[str, ...] = (
    "lab1",
    "lab2",
    "furnished",
    "rain",
    "smoke",
    "poor_lighting",
    "occlusion",
)


def compute_anatomical_pose_metrics(
    pred_m: torch.Tensor,
    gt_m: torch.Tensor,
    *,
    root_index: int = 0,
    pa_chunk_size: int = 16384,
) -> dict[str, np.ndarray]:
    """Compute per-sample geometric pose and SMPL-22 skeletal bone-length metrics (in mm)."""
    pred_f = pred_m.float()
    gt_f = gt_m.float()
    abs_mpjpe_mm = (torch.linalg.norm(pred_f - gt_f, dim=-1).mean(dim=-1) * 1000.0).double().cpu().numpy()
    root_err_mm = (
        torch.linalg.norm(pred_f[:, root_index] - gt_f[:, root_index], dim=-1) * 1000.0
    ).double().cpu().numpy()
    pred_ra = pred_f - pred_f[:, root_index : root_index + 1]
    gt_ra = gt_f - gt_f[:, root_index : root_index + 1]
    ra_mpjpe_mm = (torch.linalg.norm(pred_ra - gt_ra, dim=-1).mean(dim=-1) * 1000.0).double().cpu().numpy()

    n = int(pred_f.shape[0])
    p_mpjpe_mm = np.empty(n, dtype=np.float64)
    chunk = max(1, int(pa_chunk_size))
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        p_mpjpe_mm[start:end] = compute_per_sample_p_mpjpe(pred_f[start:end], gt_f[start:end]).astype(np.float64) * 1000.0

    pred_mm = pred_f * 1000.0
    gt_mm = gt_f * 1000.0
    pred_len = torch.linalg.norm(pred_mm[:, _U_IDX] - pred_mm[:, _V_IDX], dim=-1)
    gt_len = torch.linalg.norm(gt_mm[:, _U_IDX] - gt_mm[:, _V_IDX], dim=-1)
    abs_bone_err = (pred_len - gt_len).abs()

    mble_all_mm = abs_bone_err.mean(dim=-1).double().cpu().numpy()
    mble_limbs_mm = abs_bone_err[:, _LIMB_IDX].mean(dim=-1).double().cpu().numpy()
    mble_distal_mm = abs_bone_err[:, _DISTAL_IDX].mean(dim=-1).double().cpu().numpy()
    distal_diff = pred_len[:, _DISTAL_IDX] - gt_len[:, _DISTAL_IDX]
    distal_bias_mm = distal_diff.mean(dim=-1).double().cpu().numpy()
    distal_bias_rel_pct = (
        (distal_diff / gt_len[:, _DISTAL_IDX].clamp_min(1.0e-3) * 100.0).mean(dim=-1).double().cpu().numpy()
    )
    pred_lr_diff = pred_len[:, _LEFT_IDX] - pred_len[:, _RIGHT_IDX]
    gt_lr_diff = gt_len[:, _LEFT_IDX] - gt_len[:, _RIGHT_IDX]
    lr_asym_err_mm = (pred_lr_diff - gt_lr_diff).abs().mean(dim=-1).double().cpu().numpy()

    return {
        "abs_mpjpe_mm": abs_mpjpe_mm,
        "root_error_mm": root_err_mm,
        "root_aligned_mpjpe_mm": ra_mpjpe_mm,
        "pa_mpjpe_mm": p_mpjpe_mm,
        "mble_all_mm": mble_all_mm,
        "mble_limbs_mm": mble_limbs_mm,
        "mble_distal_mm": mble_distal_mm,
        "distal_bias_mm": distal_bias_mm,
        "distal_bias_rel_pct": distal_bias_rel_pct,
        "lr_asym_err_mm": lr_asym_err_mm,
    }


def summarize_metric_vectors(vecs: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    """Return sample count and masked mean for each metric vector."""
    cnt = int(mask.sum())
    out: dict[str, Any] = {"num_samples": cnt}
    if cnt == 0:
        return out
    for key, values in vecs.items():
        out[key] = float(values[mask].mean())
    return out


def extract_sequence_metadata(dataset: Any) -> dict[str, Any]:
    """Extract ordered sequence identifiers and frame indices for temporal and scenario slicing."""
    if isinstance(dataset, M4HumanLMDBPoseDataset):
        arr = np.asarray(dataset.data_list, dtype=np.int64)
        return {
            "kind": "m4human",
            "subject_ids": arr[:, 0],
            "action_ids": arr[:, 1],
            "frame_ids": arr[:, 2],
        }
    if isinstance(dataset, MMBodyDataset):
        df = dataset.data_list
        return {
            "kind": "mmbody",
            "scenarios": df["Scenario"].astype(str).to_numpy(),
            "sequences": df["Sequence"].astype(np.int64).to_numpy(),
            "frame_ids": df["Frame"].astype(np.int64).to_numpy(),
        }
    raise TypeError(f"Unsupported dataset for sequence metadata extraction: {type(dataset)}")


def compute_same_prev_mask(seq_meta: dict[str, Any]) -> np.ndarray:
    """Return boolean mask where index ``t`` immediately follows ``t-1`` in the same sequence."""
    frame_ids = np.asarray(seq_meta["frame_ids"], dtype=np.int64)
    n = len(frame_ids)
    same_prev = np.zeros(n, dtype=bool)
    if n <= 1:
        return same_prev
    if seq_meta["kind"] == "m4human":
        same_prev[1:] = (
            (seq_meta["subject_ids"][1:] == seq_meta["subject_ids"][:-1])
            & (seq_meta["action_ids"][1:] == seq_meta["action_ids"][:-1])
            & (frame_ids[1:] == frame_ids[:-1] + 1)
        )
        return same_prev
    if seq_meta["kind"] == "mmbody":
        same_prev[1:] = (
            (seq_meta["scenarios"][1:] == seq_meta["scenarios"][:-1])
            & (seq_meta["sequences"][1:] == seq_meta["sequences"][:-1])
            & (frame_ids[1:] == frame_ids[:-1] + 1)
        )
        return same_prev
    raise ValueError(f"Unsupported seq_meta kind={seq_meta.get('kind')!r}")


def apply_temporal_filter(
    pred_m: torch.Tensor,
    seq_meta: dict[str, Any] | np.ndarray,
    filter_kind: str,
) -> torch.Tensor:
    """Apply a temporal filter resetting strictly at sequence/gap boundaries.

    Supported ``filter_kind`` values:
    - ``raw``: identity passthrough.
    - ``causal_median3``: strictly causal 3-frame median over ``{t-2, t-1, t}``.
    - ``causal_ema_<alpha>`` / ``ema_<alpha>``: strictly causal EMA
      ``y_tilde[t] = alpha * y[t] + (1 - alpha) * y_tilde[t-1]``.
    - ``median3``: centered 3-frame median over ``{t-1, t, t+1}`` within contiguous runs.
    """
    kind = str(filter_kind).strip().lower()
    if kind == "raw":
        return pred_m

    same_prev = seq_meta if isinstance(seq_meta, np.ndarray) else compute_same_prev_mask(seq_meta)
    arr = pred_m.detach().float().cpu().numpy().copy()
    n = arr.shape[0]
    if n == 0:
        return torch.from_numpy(arr)

    if kind == "causal_median3":
        out = arr.copy()
        if n >= 3:
            valid3 = np.zeros(n, dtype=bool)
            valid3[2:] = same_prev[2:] & same_prev[1:-1]
            med_all = np.median(np.stack([arr[:-2], arr[1:-1], arr[2:]], axis=0), axis=0)
            out[2:][valid3[2:]] = med_all[valid3[2:]]
        return torch.from_numpy(out)

    if kind == "median3":
        out = arr.copy()
        if n >= 3:
            valid_mid = same_prev[1:-1] & same_prev[2:]
            med_all = np.median(np.stack([arr[:-2], arr[1:-1], arr[2:]], axis=0), axis=0)
            out[1:-1][valid_mid] = med_all[valid_mid]
        return torch.from_numpy(out)

    if kind.startswith("causal_ema_") or kind.startswith("ema_"):
        alpha = float(kind.split("_")[-1])
        one_minus = 1.0 - alpha
        out = np.empty_like(arr)
        out[0] = arr[0]
        for i in range(1, n):
            if same_prev[i]:
                out[i] = alpha * arr[i] + one_minus * out[i - 1]
            else:
                out[i] = arr[i]
        return torch.from_numpy(out)

    raise ValueError(f"Unsupported temporal filter_kind={filter_kind!r}")


def compute_temporal_smoothness_metrics(
    pred_m: torch.Tensor,
    gt_m: torch.Tensor,
    seq_meta: dict[str, Any] | np.ndarray,
    *,
    root_index: int = 0,
) -> dict[str, float | int]:
    """Compute velocity (MPJVE) and acceleration errors over strictly contiguous frame runs."""
    same_prev = seq_meta if isinstance(seq_meta, np.ndarray) else compute_same_prev_mask(seq_meta)
    vel_valid = same_prev[1:]
    acc_valid = same_prev[1:-1] & same_prev[2:]

    pred_mm = pred_m.detach().float().cpu() * 1000.0
    gt_mm = gt_m.detach().float().cpu() * 1000.0
    pred_ra = pred_mm - pred_mm[:, root_index : root_index + 1]
    gt_ra = gt_mm - gt_mm[:, root_index : root_index + 1]

    pv = pred_mm[1:] - pred_mm[:-1]
    gv = gt_mm[1:] - gt_mm[:-1]
    pv_ra = pred_ra[1:] - pred_ra[:-1]
    gv_ra = gt_ra[1:] - gt_ra[:-1]

    n_vel = int(vel_valid.sum())
    n_acc = int(acc_valid.sum())
    mpjve_abs = float(torch.linalg.norm((pv - gv)[vel_valid], dim=-1).mean().item()) if n_vel > 0 else 0.0
    mpjve_ra = float(torch.linalg.norm((pv_ra - gv_ra)[vel_valid], dim=-1).mean().item()) if n_vel > 0 else 0.0

    pa = pred_mm[2:] - 2.0 * pred_mm[1:-1] + pred_mm[:-2]
    ga = gt_mm[2:] - 2.0 * gt_mm[1:-1] + gt_mm[:-2]
    pa_ra = pred_ra[2:] - 2.0 * pred_ra[1:-1] + pred_ra[:-2]
    ga_ra = gt_ra[2:] - 2.0 * gt_ra[1:-1] + gt_ra[:-2]

    accel_err_abs = float(torch.linalg.norm((pa - ga)[acc_valid], dim=-1).mean().item()) if n_acc > 0 else 0.0
    accel_err_ra = float(torch.linalg.norm((pa_ra - ga_ra)[acc_valid], dim=-1).mean().item()) if n_acc > 0 else 0.0
    pred_accel_mag = float(torch.linalg.norm(pa[acc_valid], dim=-1).mean().item()) if n_acc > 0 else 0.0
    gt_accel_mag = float(torch.linalg.norm(ga[acc_valid], dim=-1).mean().item()) if n_acc > 0 else 0.0

    return {
        "mpjve_abs_mm_per_frame": mpjve_abs,
        "mpjve_ra_mm_per_frame": mpjve_ra,
        "accel_err_abs_mm_per_frame2": accel_err_abs,
        "accel_err_ra_mm_per_frame2": accel_err_ra,
        "pred_accel_mag_mm_per_frame2": pred_accel_mag,
        "gt_accel_mag_mm_per_frame2": gt_accel_mag,
        "n_vel_pairs": n_vel,
        "n_acc_triples": n_acc,
    }


def compute_pre_cap_point_counts(
    dataset: Any,
    *,
    canon_gt: torch.Tensor | None = None,
    cache_path: Path | None = None,
) -> np.ndarray:
    """Count valid pre-downsampling radar reflections per test window for sparsity slicing."""
    if cache_path is not None and cache_path.is_file():
        return np.load(cache_path)

    n = len(dataset)
    out = np.empty(n, dtype=np.int32)
    t0 = time.perf_counter()

    if isinstance(dataset, M4HumanLMDBPoseDataset):
        if canon_gt is None:
            canon_gt = torch.stack([dataset[i]["target_poses_absolute"] for i in range(n)], dim=0)
        root_cfg = getattr(dataset, "root_alignment", None)
        roots_np = canon_gt[:, 0].cpu().numpy() if root_cfg is not None else None
        in_ch = int(dataset.input_channels)
        norm_z = bool(dataset.normalize_z)
        int_scale = float(dataset.intensity_scale)
        has_offsets = hasattr(dataset, "temporal_offsets")
        frame_cnt_cache: dict[tuple[int, int, int], int] = {}
        dataset._ensure_lmdb_env("radar_pc")
        for idx in range(n):
            t_inds = dataset._resolve_temporal_indicators(idx) if has_offsets else [dataset.data_list[idx]]
            if root_cfg is None:
                total_c = 0
                for ind in t_inds:
                    key = (int(ind[0]), int(ind[1]), int(ind[2]))
                    cnt = frame_cnt_cache.get(key)
                    if cnt is None:
                        arr = dataset._read_radar_pc(ind)[:, :in_ch]
                        if norm_z and arr.size:
                            arr = arr.copy()
                            arr[:, 2] -= 1.5
                        if int_scale != 1.0 and arr.size:
                            arr[:, 3] *= int_scale
                        cnt = int(np.any(np.abs(arr[:, :4]) > 0, axis=-1).sum())
                        frame_cnt_cache[key] = cnt
                    total_c += cnt
                out[idx] = total_c
            else:
                assert roots_np is not None
                root = roots_np[idx]
                total_c = 0
                for ind in t_inds:
                    arr = dataset._read_radar_pc(ind)[:, :in_ch].copy()
                    if norm_z and arr.size:
                        arr[:, 2] -= 1.5
                    if int_scale != 1.0 and arr.size:
                        arr[:, 3] *= int_scale
                    arr = apply_input_root_alignment(arr, root=root, cfg=root_cfg)
                    total_c += int(np.any(np.abs(arr[:, :4]) > 0, axis=-1).sum())
                out[idx] = total_c
    elif isinstance(dataset, MMBodyDataset):
        for idx in range(n):
            rlist, _ = dataset._load_sample(idx)
            if not rlist:
                out[idx] = 0
            else:
                raw = np.concatenate(rlist, axis=0)
                out[idx] = int(np.any(np.abs(raw[:, :4]) > 0, axis=-1).sum())
    else:
        raise TypeError(f"Unsupported dataset for pre-cap point counting: {type(dataset)}")

    print(f"[diagnostics] Computed pre-cap radar counts ({n} samples) in {time.perf_counter() - t0:.1f}s", flush=True)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, out)
    return out


__all__ = [
    "ALL_BONE_EDGES",
    "BILATERAL_BONE_PAIRS",
    "DISTAL_BONE_EDGES",
    "LIMB_BONE_EDGES",
    "MMBODY_SCENARIOS",
    "apply_temporal_filter",
    "compute_anatomical_pose_metrics",
    "compute_pre_cap_point_counts",
    "compute_same_prev_mask",
    "compute_temporal_smoothness_metrics",
    "extract_sequence_metadata",
    "summarize_metric_vectors",
]
