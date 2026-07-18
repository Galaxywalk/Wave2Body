# -*- coding: utf-8 -*-
"""Stage-I quantized body-token alignment analysis."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


SAME_ACTION_CROSS_SUBJECT_STRATEGY = "same_action_cross_subject_permutation"
DIFFERENT_ACTION_SAME_SUBJECT_STRATEGY = "different_action_same_subject_permutation"
NEGATIVE_STRATEGY = SAME_ACTION_CROSS_SUBJECT_STRATEGY


def _as_indicator_array(indicators: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(indicators):
        values = indicators.detach().cpu().numpy()
    else:
        values = np.asarray(indicators)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError(f"indicators must be [N,>=3] [subject, action, frame], got {values.shape}")
    if values.shape[0] <= 0:
        raise ValueError("indicators must contain at least one sample")
    return np.asarray(values, dtype=np.int64)


def build_same_action_cross_subject_permutation(
    indicators: torch.Tensor | np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Build a deterministic bijection to same-action, cross-subject targets.

    Samples are grouped by action and then into contiguous subject blocks.
    Items inside every subject block are shuffled before the concatenated action
    group is circularly shifted by its largest subject-block size. A valid
    bijection exists under this construction when no subject owns more than
    half of an action group.
    """

    if int(seed) < 0:
        raise ValueError(f"shuffle seed must be non-negative, got {seed}")
    values = _as_indicator_array(indicators)
    subjects = values[:, 0]
    actions = values[:, 1]
    permutation = np.full(values.shape[0], -1, dtype=np.int64)
    rng = np.random.default_rng(int(seed))

    for action_id in np.unique(actions):
        action_indices = np.flatnonzero(actions == action_id)
        action_subjects = subjects[action_indices]
        unique_subjects = np.unique(action_subjects)
        if unique_subjects.size < 2:
            raise ValueError(
                "same-action cross-subject permutation is impossible for "
                f"action={int(action_id)}: found one subject {unique_subjects.tolist()}"
            )

        blocks: list[np.ndarray] = []
        block_sizes: list[int] = []
        for subject_id in unique_subjects:
            block = action_indices[action_subjects == subject_id].copy()
            rng.shuffle(block)
            blocks.append(block)
            block_sizes.append(int(block.size))

        ordered = np.concatenate(blocks)
        largest = max(block_sizes)
        if largest * 2 > int(ordered.size):
            counts = {
                str(int(subject_id)): int(size)
                for subject_id, size in zip(unique_subjects.tolist(), block_sizes)
            }
            raise ValueError(
                "same-action cross-subject permutation is impossible because one "
                f"subject exceeds half of action={int(action_id)}: counts={counts}"
            )

        shifted = np.roll(ordered, -largest)
        if np.any(subjects[ordered] == subjects[shifted]):
            raise RuntimeError(
                f"internal permutation error: action={int(action_id)} retained a same-subject pair"
            )
        permutation[ordered] = shifted

    expected = np.arange(values.shape[0], dtype=np.int64)
    if np.any(permutation < 0) or not np.array_equal(np.sort(permutation), expected):
        raise RuntimeError("internal permutation error: target mapping is not a bijection")
    if np.any(actions != actions[permutation]):
        raise RuntimeError("internal permutation error: action labels changed")
    if np.any(subjects == subjects[permutation]):
        raise RuntimeError("internal permutation error: subject labels did not change")
    return permutation


def build_different_action_same_subject_permutation(
    indicators: torch.Tensor | np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Build a deterministic bijection to different-action, same-subject targets.

    Samples are grouped by subject and then into contiguous action blocks. Items
    inside each action block are shuffled before the subject group is circularly
    shifted by its largest action-block size. A valid bijection exists under
    this construction when no action owns more than half of a subject group.
    """

    if int(seed) < 0:
        raise ValueError(f"shuffle seed must be non-negative, got {seed}")
    values = _as_indicator_array(indicators)
    subjects = values[:, 0]
    actions = values[:, 1]
    permutation = np.full(values.shape[0], -1, dtype=np.int64)
    rng = np.random.default_rng(int(seed))

    for subject_id in np.unique(subjects):
        subject_indices = np.flatnonzero(subjects == subject_id)
        subject_actions = actions[subject_indices]
        unique_actions = np.unique(subject_actions)
        if unique_actions.size < 2:
            raise ValueError(
                "different-action same-subject permutation is impossible for "
                f"subject={int(subject_id)}: found one action {unique_actions.tolist()}"
            )

        blocks: list[np.ndarray] = []
        block_sizes: list[int] = []
        for action_id in unique_actions:
            block = subject_indices[subject_actions == action_id].copy()
            rng.shuffle(block)
            blocks.append(block)
            block_sizes.append(int(block.size))

        ordered = np.concatenate(blocks)
        largest = max(block_sizes)
        if largest * 2 > int(ordered.size):
            counts = {
                str(int(action_id)): int(size)
                for action_id, size in zip(unique_actions.tolist(), block_sizes)
            }
            raise ValueError(
                "different-action same-subject permutation is impossible because one "
                f"action exceeds half of subject={int(subject_id)}: counts={counts}"
            )

        shifted = np.roll(ordered, -largest)
        if np.any(actions[ordered] == actions[shifted]):
            raise RuntimeError(
                f"internal permutation error: subject={int(subject_id)} retained a same-action pair"
            )
        permutation[ordered] = shifted

    expected = np.arange(values.shape[0], dtype=np.int64)
    if np.any(permutation < 0) or not np.array_equal(np.sort(permutation), expected):
        raise RuntimeError("internal permutation error: target mapping is not a bijection")
    if np.any(subjects != subjects[permutation]):
        raise RuntimeError("internal permutation error: subject labels changed")
    if np.any(actions == actions[permutation]):
        raise RuntimeError("internal permutation error: action labels did not change")
    return permutation


def permutation_signature(permutation: np.ndarray) -> str:
    values = np.asarray(permutation, dtype="<i8")
    if values.ndim != 1:
        raise ValueError(f"permutation must be one-dimensional, got {values.shape}")
    return "sha256:" + hashlib.sha256(values.tobytes()).hexdigest()


def _as_index_array(values: torch.Tensor | np.ndarray, *, name: str) -> np.ndarray:
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    out = np.asarray(values)
    if out.ndim != 2:
        raise ValueError(f"{name} must be [N,T], got {out.shape}")
    return np.asarray(out, dtype=np.int64)


def _as_ranked_index_array(
    values: torch.Tensor | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    out = np.asarray(values)
    if out.ndim != 3:
        raise ValueError(f"{name} must be [N,T,K], got {out.shape}")
    return np.asarray(out, dtype=np.int64)


def summarize_exact_agreement(
    pred_indices: torch.Tensor | np.ndarray,
    target_indices: torch.Tensor | np.ndarray,
    *,
    layout: Any,
) -> dict[str, Any]:
    """Summarize exact global-code agreement overall and by body part."""

    pred = _as_index_array(pred_indices, name="pred_indices")
    target = _as_index_array(target_indices, name="target_indices")
    if pred.shape != target.shape:
        raise ValueError(f"pred/target index shape mismatch: {pred.shape} vs {target.shape}")
    expected_tokens = int(layout.total_tokens)
    if pred.shape[1] != expected_tokens:
        raise ValueError(f"expected {expected_tokens} token slots, got {pred.shape[1]}")

    correct = pred == target

    def item(mask: np.ndarray) -> dict[str, int | float]:
        count = int(mask.size)
        matches = int(mask.sum())
        return {
            "count": count,
            "matches": matches,
            "agreement": float(matches / count) if count else 0.0,
        }

    by_part: dict[str, dict[str, int | float]] = {}
    for part_name in layout.part_names:
        token_slice = layout.part_token_slices[part_name]
        by_part[str(part_name)] = item(correct[:, token_slice])
    return {"overall": item(correct), "by_part": by_part}


def summarize_topk_agreement(
    ranked_indices: torch.Tensor | np.ndarray,
    target_indices: torch.Tensor | np.ndarray,
    *,
    layout: Any,
    topk: tuple[int, ...] = (1, 3, 5),
) -> dict[str, Any]:
    """Summarize whether each target code appears in ranked code candidates."""

    ranked = _as_ranked_index_array(ranked_indices, name="ranked_indices")
    target = _as_index_array(target_indices, name="target_indices")
    if ranked.shape[:2] != target.shape:
        raise ValueError(
            f"ranked/target prefix mismatch: {ranked.shape[:2]} vs {target.shape}"
        )
    expected_tokens = int(layout.total_tokens)
    if target.shape[1] != expected_tokens:
        raise ValueError(f"expected {expected_tokens} token slots, got {target.shape[1]}")

    ks = tuple(sorted({int(k) for k in topk}))
    if not ks or any(k <= 0 for k in ks):
        raise ValueError(f"top-k values must be positive, got {topk}")
    if max(ks) > ranked.shape[2]:
        raise ValueError(
            f"ranked candidates provide K={ranked.shape[2]}, but requested top-{max(ks)}"
        )

    masks = {
        f"top{k}": (ranked[:, :, :k] == target[:, :, None]).any(axis=-1)
        for k in ks
    }

    def item(part_slice: slice | None = None) -> dict[str, int | float]:
        selected = masks if part_slice is None else {
            key: mask[:, part_slice] for key, mask in masks.items()
        }
        count = int(next(iter(selected.values())).size)
        result: dict[str, int | float] = {"count": count}
        for key, mask in selected.items():
            matches = int(mask.sum())
            result[f"{key}_matches"] = matches
            result[key] = float(matches / count) if count else 0.0
        return result

    by_part = {
        str(part_name): item(layout.part_token_slices[part_name])
        for part_name in layout.part_names
    }
    return {"topk": list(ks), "overall": item(), "by_part": by_part}


def rank_part_codebook_candidates(
    *,
    vqvae: Any,
    pred_z_e: torch.Tensor,
    pred_indices: torch.Tensor,
    max_k: int,
) -> torch.Tensor:
    """Rank part-local codes using the quantizer's native distance arithmetic.

    The actual quantized prediction is kept first, matching the deployed
    quantizer's argmin and tie-breaking exactly. Remaining candidates follow
    ascending squared-L2 distance in the same autocast context as quantization.
    Returned IDs use the global, part-offset code indexing convention.
    """

    if pred_z_e.dim() != 3 or pred_indices.dim() != 2:
        raise ValueError(
            f"Expected pred_z_e [B,T,D] and pred_indices [B,T], got "
            f"{tuple(pred_z_e.shape)} and {tuple(pred_indices.shape)}"
        )
    if pred_z_e.shape[:2] != pred_indices.shape:
        raise ValueError(
            f"embedding/index prefix mismatch: {tuple(pred_z_e.shape[:2])} vs "
            f"{tuple(pred_indices.shape)}"
        )
    max_k = int(max_k)
    if max_k <= 0:
        raise ValueError(f"max_k must be positive, got {max_k}")
    if not hasattr(vqvae, "layout") or not hasattr(vqvae, "quantizers"):
        raise RuntimeError("Expected a part-level VQ-VAE with layout and quantizers")

    layout = vqvae.layout
    ranked_parts: list[torch.Tensor] = []
    for part_name in layout.part_names:
        token_slice = layout.part_token_slices[part_name]
        offset = int(layout.part_codebook_offsets[part_name])
        codebook_size = int(layout.part_codebook_size[part_name])
        if max_k > codebook_size:
            raise ValueError(
                f"Requested top-{max_k}, but {part_name} has {codebook_size} codes"
            )

        z_part = pred_z_e[:, token_slice, :]
        codebook = vqvae.quantizers[part_name].codebook
        distances = (
            torch.sum(z_part * z_part, dim=-1, keepdim=True)
            + torch.sum(codebook * codebook, dim=-1)
            - 2.0 * torch.matmul(z_part, codebook.t())
        )
        nearest = torch.topk(
            distances,
            k=max_k,
            dim=-1,
            largest=False,
            sorted=True,
        ).indices

        chosen_local = pred_indices[:, token_slice].long() - offset
        valid = (chosen_local >= 0) & (chosen_local < codebook_size)
        if not bool(valid.all().item()):
            raise ValueError(f"pred_indices contain codes outside {part_name}'s codebook")

        # ``topk`` may choose a different representative for an exact-distance
        # tie than ``argmin``. Keep the deployed quantizer choice at rank one,
        # then retain the nearest non-duplicate runners-up.
        chosen = chosen_local.unsqueeze(-1)
        positions = torch.arange(max_k, device=nearest.device).view(1, 1, -1)
        positions = positions.expand_as(nearest)
        positions = positions + (nearest == chosen).to(positions.dtype) * max_k
        other_order = torch.argsort(positions, dim=-1, stable=True)
        other = torch.gather(nearest, dim=-1, index=other_order)
        ranked_local = torch.cat([chosen, other[..., : max_k - 1]], dim=-1)
        ranked_parts.append(ranked_local + offset)

    return torch.cat(ranked_parts, dim=1)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"rank input must be one-dimensional, got {values.shape}")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        sorted_ranks[start:stop] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = sorted_ranks
    return ranks


def spearman_rank_correlation(
    left: torch.Tensor | np.ndarray,
    right: torch.Tensor | np.ndarray,
) -> float | None:
    """Return tie-aware Spearman rho, or None for a constant input."""

    left_values = np.asarray(
        left.detach().cpu().numpy() if torch.is_tensor(left) else left,
        dtype=np.float64,
    ).reshape(-1)
    right_values = np.asarray(
        right.detach().cpu().numpy() if torch.is_tensor(right) else right,
        dtype=np.float64,
    ).reshape(-1)
    if left_values.shape != right_values.shape:
        raise ValueError(f"correlation shape mismatch: {left_values.shape} vs {right_values.shape}")
    if left_values.size < 2:
        return None
    if not np.all(np.isfinite(left_values)) or not np.all(np.isfinite(right_values)):
        raise ValueError("Spearman inputs must contain only finite values")
    if np.all(left_values == left_values[0]) or np.all(right_values == right_values[0]):
        return None

    left_ranks = _average_ranks(left_values)
    right_ranks = _average_ranks(right_values)
    rho = float(np.corrcoef(left_ranks, right_ranks)[0, 1])
    if not np.isfinite(rho):
        return None
    return float(np.clip(rho, -1.0, 1.0))


def _agreement_gain(
    paired: dict[str, Any],
    shuffled: dict[str, Any],
) -> dict[str, Any]:
    part_names = tuple(paired["by_part"])
    if part_names != tuple(shuffled["by_part"]):
        raise ValueError("paired and shuffled part layouts differ")
    return {
        "overall": float(paired["overall"]["agreement"] - shuffled["overall"]["agreement"]),
        "by_part": {
            name: float(
                paired["by_part"][name]["agreement"]
                - shuffled["by_part"][name]["agreement"]
            )
            for name in part_names
        },
    }


def _topk_agreement_gain(
    paired: dict[str, Any],
    control: dict[str, Any],
) -> dict[str, Any]:
    if tuple(paired["topk"]) != tuple(control["topk"]):
        raise ValueError("paired and control top-k values differ")
    part_names = tuple(paired["by_part"])
    if part_names != tuple(control["by_part"]):
        raise ValueError("paired and control part layouts differ")
    keys = tuple(f"top{int(k)}" for k in paired["topk"])
    return {
        "overall": {
            key: float(paired["overall"][key] - control["overall"][key])
            for key in keys
        },
        "by_part": {
            name: {
                key: float(
                    paired["by_part"][name][key]
                    - control["by_part"][name][key]
                )
                for key in keys
            }
            for name in part_names
        },
    }


@dataclass
class TokenAlignmentCollector:
    """Collect sample-level Stage-I outputs and finalize alignment metrics."""

    layout: Any
    shuffle_seed: int = 0
    topk: tuple[int, ...] = (1, 3, 5)
    _pred_items: list[torch.Tensor] = field(default_factory=list, init=False, repr=False)
    _ranked_items: list[torch.Tensor] = field(default_factory=list, init=False, repr=False)
    _target_items: list[torch.Tensor] = field(default_factory=list, init=False, repr=False)
    _indicator_items: list[torch.Tensor] = field(default_factory=list, init=False, repr=False)
    _error_items: list[torch.Tensor] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.shuffle_seed = int(self.shuffle_seed)
        if self.shuffle_seed < 0:
            raise ValueError(f"shuffle seed must be non-negative, got {self.shuffle_seed}")
        self.topk = tuple(sorted({int(k) for k in self.topk}))
        if not self.topk or any(k <= 0 for k in self.topk):
            raise ValueError(f"top-k values must be positive, got {self.topk}")

    def update(
        self,
        *,
        pred_indices: torch.Tensor,
        target_indices: torch.Tensor,
        indicators: torch.Tensor,
        root_relative_mpjpe: torch.Tensor,
        ranked_indices: torch.Tensor | None = None,
    ) -> None:
        if pred_indices.dim() != 2 or target_indices.dim() != 2:
            raise ValueError("pred_indices and target_indices must be [B,T]")
        if pred_indices.shape != target_indices.shape:
            raise ValueError(
                f"pred/target index shape mismatch: {tuple(pred_indices.shape)} vs "
                f"{tuple(target_indices.shape)}"
            )
        batch_size = int(pred_indices.shape[0])
        if indicators.dim() != 2 or int(indicators.shape[0]) != batch_size or int(indicators.shape[1]) < 3:
            raise ValueError(
                f"indicators must be [B,>=3] [subject, action, frame] with B={batch_size}, "
                f"got {tuple(indicators.shape)}"
            )
        errors = root_relative_mpjpe.reshape(-1)
        if int(errors.numel()) != batch_size:
            raise ValueError(
                f"root_relative_mpjpe must contain {batch_size} values, got {int(errors.numel())}"
            )
        if ranked_indices is not None:
            if ranked_indices.dim() != 3:
                raise ValueError(
                    f"ranked_indices must be [B,T,K], got {tuple(ranked_indices.shape)}"
                )
            if ranked_indices.shape[:2] != target_indices.shape:
                raise ValueError(
                    f"ranked/target prefix mismatch: {tuple(ranked_indices.shape[:2])} vs "
                    f"{tuple(target_indices.shape)}"
                )
            if int(ranked_indices.shape[2]) < max(self.topk):
                raise ValueError(
                    f"ranked candidates provide K={int(ranked_indices.shape[2])}, "
                    f"but requested top-{max(self.topk)}"
                )
            self._ranked_items.append(
                ranked_indices.detach().to(device="cpu", dtype=torch.long).contiguous()
            )
        elif self._ranked_items:
            raise ValueError("ranked_indices must be provided consistently for every batch")

        self._pred_items.append(pred_indices.detach().to(device="cpu", dtype=torch.long).contiguous())
        self._target_items.append(target_indices.detach().to(device="cpu", dtype=torch.long).contiguous())
        self._indicator_items.append(indicators.detach().to(device="cpu", dtype=torch.long).contiguous())
        self._error_items.append(errors.detach().to(device="cpu", dtype=torch.float64).contiguous())

    def summary(self) -> dict[str, Any]:
        if not self._pred_items:
            raise RuntimeError("No samples were collected for token alignment")
        pred = torch.cat(self._pred_items, dim=0).numpy()
        target = torch.cat(self._target_items, dim=0).numpy()
        indicators = torch.cat(self._indicator_items, dim=0).numpy()
        root_relative_mpjpe = torch.cat(self._error_items, dim=0).numpy()
        if self._ranked_items and len(self._ranked_items) != len(self._pred_items):
            raise ValueError("ranked_indices were not provided consistently for every batch")
        ranked = (
            torch.cat(self._ranked_items, dim=0).numpy()
            if self._ranked_items
            else None
        )

        paired = summarize_exact_agreement(pred, target, layout=self.layout)
        permutation = build_same_action_cross_subject_permutation(
            indicators,
            seed=self.shuffle_seed,
        )
        different_action_permutation = build_different_action_same_subject_permutation(
            indicators,
            seed=self.shuffle_seed,
        )
        shuffled = summarize_exact_agreement(pred, target[permutation], layout=self.layout)
        different_action_same_subject = summarize_exact_agreement(
            pred, target[different_action_permutation], layout=self.layout
        )
        per_sample_agreement = (pred == target).mean(axis=1, dtype=np.float64)
        token_mismatch = 1.0 - per_sample_agreement
        rho = spearman_rank_correlation(token_mismatch, root_relative_mpjpe)

        codebook_topk = None
        topk_spearman = None
        if ranked is not None:
            if not np.array_equal(ranked[:, :, 0], pred):
                raise ValueError("ranked_indices top-1 must equal pred_indices")
            topk_spearman = {}
            for k in self.topk:
                per_sample_hits = (
                    ranked[:, :, :k] == target[:, :, None]
                ).any(axis=-1)
                per_sample_miss_rate = 1.0 - per_sample_hits.mean(
                    axis=1, dtype=np.float64
                )
                topk_spearman[f"top{k}"] = spearman_rank_correlation(
                    per_sample_miss_rate, root_relative_mpjpe
                )
            paired_topk = summarize_topk_agreement(
                ranked, target, layout=self.layout, topk=self.topk
            )
            shuffled_topk = summarize_topk_agreement(
                ranked, target[permutation], layout=self.layout, topk=self.topk
            )
            different_action_topk = summarize_topk_agreement(
                ranked,
                target[different_action_permutation],
                layout=self.layout,
                topk=self.topk,
            )
            codebook_topk = {
                "definition": "target_token_in_nearest_codebook_candidates",
                "ranking": "quantizer_selected_top1_then_ascending_native_squared_l2",
                "topk": list(self.topk),
                "paired": paired_topk,
                "shuffled": shuffled_topk,
                "gain": _topk_agreement_gain(paired_topk, shuffled_topk),
                "different_action_same_subject": different_action_topk,
                "different_action_same_subject_gain": _topk_agreement_gain(
                    paired_topk, different_action_topk
                ),
            }

        result = {
            "definition": "quantized_exact_body_token_agreement",
            "num_samples": int(pred.shape[0]),
            "tokens_per_sample": int(pred.shape[1]),
            "part_token_counts": {
                str(name): int(self.layout.part_token_num[name])
                for name in self.layout.part_names
            },
            "paired": paired,
            "shuffled": shuffled,
            "gain": _agreement_gain(paired, shuffled),
            "different_action_same_subject": different_action_same_subject,
            "different_action_same_subject_gain": _agreement_gain(
                paired, different_action_same_subject
            ),
            "spearman": {
                "token_mismatch_vs_root_relative_mpjpe": rho,
                "topk_miss_rate_vs_root_relative_mpjpe": topk_spearman,
                "valid": rho is not None,
                "num_samples": int(pred.shape[0]),
                "root_relative_mpjpe_unit": "m",
            },
            "negative_protocol": {
                "strategy": NEGATIVE_STRATEGY,
                "shuffle_seed": int(self.shuffle_seed),
                "permutation_signature": permutation_signature(permutation),
                "num_actions": int(np.unique(indicators[:, 1]).size),
                "num_subjects": int(np.unique(indicators[:, 0]).size),
                "verified_same_action": bool(
                    np.all(indicators[:, 1] == indicators[permutation, 1])
                ),
                "verified_different_subject": bool(
                    np.all(indicators[:, 0] != indicators[permutation, 0])
                ),
                "verified_bijection": bool(
                    np.array_equal(
                        np.sort(permutation),
                        np.arange(permutation.size, dtype=np.int64),
                    )
                ),
            },
            "different_action_same_subject_protocol": {
                "strategy": DIFFERENT_ACTION_SAME_SUBJECT_STRATEGY,
                "shuffle_seed": int(self.shuffle_seed),
                "permutation_signature": permutation_signature(
                    different_action_permutation
                ),
                "num_actions": int(np.unique(indicators[:, 1]).size),
                "num_subjects": int(np.unique(indicators[:, 0]).size),
                "verified_same_subject": bool(
                    np.all(
                        indicators[:, 0]
                        == indicators[different_action_permutation, 0]
                    )
                ),
                "verified_different_action": bool(
                    np.all(
                        indicators[:, 1]
                        != indicators[different_action_permutation, 1]
                    )
                ),
                "verified_bijection": bool(
                    np.array_equal(
                        np.sort(different_action_permutation),
                        np.arange(different_action_permutation.size, dtype=np.int64),
                    )
                ),
            },
        }
        if codebook_topk is not None:
            result["codebook_topk"] = codebook_topk
        return result


__all__ = [
    "DIFFERENT_ACTION_SAME_SUBJECT_STRATEGY",
    "NEGATIVE_STRATEGY",
    "SAME_ACTION_CROSS_SUBJECT_STRATEGY",
    "TokenAlignmentCollector",
    "build_different_action_same_subject_permutation",
    "build_same_action_cross_subject_permutation",
    "permutation_signature",
    "rank_part_codebook_candidates",
    "spearman_rank_correlation",
    "summarize_exact_agreement",
    "summarize_topk_agreement",
]
