# -*- coding: utf-8 -*-
"""Part-token accuracy helpers for body-part VQ-VAE models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


def parse_topk_list(value: str | Iterable[int]) -> tuple[int, ...]:
    """Parse and validate a top-k list."""
    if isinstance(value, str):
        items = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        items = [int(item) for item in value]
    if not items:
        raise ValueError("top-k list must not be empty")
    if any(k <= 0 for k in items):
        raise ValueError(f"top-k values must be positive, got {items}")
    return tuple(sorted(set(items)))


def _topk_key(k: int) -> str:
    return f"top{int(k)}"


def _empty_topk_counts(topk: tuple[int, ...]) -> dict[str, int]:
    return {_topk_key(k): 0 for k in topk}


def _empty_top1_counts() -> dict[str, int]:
    return {"top1": 0}


def _ratio(correct: int, count: int) -> float:
    if count <= 0:
        return 0.0
    return float(correct) / float(count)


@dataclass
class PartTokenTopKAccumulator:
    """Accumulate local logit top-k and quantized-token top-1 accuracies.

    ``logit`` metrics use the valid local codebook slice for each body part.
    ``quantized`` metrics match the existing training-time token_acc definition:
    predicted global VQ indices after quantizing the continuous translator output
    are compared directly against target global indices.
    """

    layout: object
    topk: tuple[int, ...] = (1, 3, 5)
    include_slots: bool = True

    def __post_init__(self) -> None:
        self.topk = parse_topk_list(self.topk)
        self.part_names = tuple(str(name) for name in self.layout.part_names)

        self.logit_correct = _empty_topk_counts(self.topk)
        self.logit_count = 0
        self.logit_part_correct = {name: _empty_topk_counts(self.topk) for name in self.part_names}
        self.logit_part_count = {name: 0 for name in self.part_names}
        self.logit_slot_correct = {
            name: [_empty_topk_counts(self.topk) for _ in range(int(self.layout.part_token_num[name]))]
            for name in self.part_names
        }
        self.logit_slot_count = {
            name: [0 for _ in range(int(self.layout.part_token_num[name]))]
            for name in self.part_names
        }

        self.quantized_correct = _empty_top1_counts()
        self.quantized_count = 0
        self.quantized_part_correct = {name: _empty_top1_counts() for name in self.part_names}
        self.quantized_part_count = {name: 0 for name in self.part_names}
        self.quantized_slot_correct = {
            name: [_empty_top1_counts() for _ in range(int(self.layout.part_token_num[name]))]
            for name in self.part_names
        }
        self.quantized_slot_count = {
            name: [0 for _ in range(int(self.layout.part_token_num[name]))]
            for name in self.part_names
        }

    def update_logits(self, pred_logits: torch.Tensor, target_indices: torch.Tensor) -> None:
        """Accumulate posterior top-k accuracy from global logits."""
        if pred_logits is None:
            raise ValueError("pred_logits is required for logit top-k accuracy")
        if pred_logits.dim() != 3:
            raise ValueError(f"pred_logits must be [B,T,C], got {tuple(pred_logits.shape)}")
        if target_indices.dim() != 2:
            raise ValueError(f"target_indices must be [B,T], got {tuple(target_indices.shape)}")
        if pred_logits.shape[:2] != target_indices.shape:
            raise ValueError(
                f"logit prefix {tuple(pred_logits.shape[:2])} does not match target shape {tuple(target_indices.shape)}"
            )

        target_indices = target_indices.long()
        for part_name in self.part_names:
            token_slice = self.layout.part_token_slices[part_name]
            offset = int(self.layout.part_codebook_offsets[part_name])
            k_local = int(self.layout.part_codebook_size[part_name])
            if offset < 0 or k_local <= 0 or offset + k_local > int(pred_logits.shape[-1]):
                raise ValueError(
                    f"Invalid codebook slice for {part_name}: offset={offset}, size={k_local}, "
                    f"logits_dim={int(pred_logits.shape[-1])}"
                )

            logits_local = pred_logits[:, token_slice, offset : offset + k_local]
            target_local = target_indices[:, token_slice] - offset
            valid_mask = (target_local >= 0) & (target_local < k_local)
            if not bool(valid_mask.any().item()):
                continue

            count = int(valid_mask.sum().item())
            self.logit_count += count
            self.logit_part_count[part_name] += count

            max_eff_k = min(max(self.topk), k_local)
            top_idx = torch.topk(logits_local.float(), k=max_eff_k, dim=-1).indices
            slot_valid_counts: list[int] = []
            if self.include_slots:
                for slot_idx in range(int(self.layout.part_token_num[part_name])):
                    slot_count = int(valid_mask[:, slot_idx].sum().item())
                    slot_valid_counts.append(slot_count)
                    if slot_count > 0:
                        self.logit_slot_count[part_name][slot_idx] += slot_count
            for k in self.topk:
                key = _topk_key(k)
                eff_k = min(k, k_local)
                correct_mask = (top_idx[..., :eff_k] == target_local.unsqueeze(-1)).any(dim=-1) & valid_mask
                correct = int(correct_mask.sum().item())
                self.logit_correct[key] += correct
                self.logit_part_correct[part_name][key] += correct

                if self.include_slots:
                    for slot_idx in range(int(self.layout.part_token_num[part_name])):
                        if slot_valid_counts[slot_idx] <= 0:
                            continue
                        slot_correct = int(correct_mask[:, slot_idx].sum().item())
                        self.logit_slot_correct[part_name][slot_idx][key] += slot_correct

    def update_quantized(self, pred_indices: torch.Tensor, target_indices: torch.Tensor) -> None:
        """Accumulate existing quantized-token exact accuracy."""
        if pred_indices is None:
            raise ValueError("pred_indices is required for quantized token accuracy")
        if pred_indices.dim() != 2 or target_indices.dim() != 2:
            raise ValueError(
                f"pred_indices and target_indices must be [B,T], got {tuple(pred_indices.shape)} and "
                f"{tuple(target_indices.shape)}"
            )
        if pred_indices.shape != target_indices.shape:
            raise ValueError(f"pred/target index shape mismatch: {tuple(pred_indices.shape)} vs {tuple(target_indices.shape)}")

        pred_indices = pred_indices.long()
        target_indices = target_indices.long()
        for part_name in self.part_names:
            token_slice = self.layout.part_token_slices[part_name]
            offset = int(self.layout.part_codebook_offsets[part_name])
            k_local = int(self.layout.part_codebook_size[part_name])
            pred_part = pred_indices[:, token_slice]
            target_part = target_indices[:, token_slice]
            valid_mask = (target_part >= offset) & (target_part < offset + k_local)
            if not bool(valid_mask.any().item()):
                continue

            correct_mask = (pred_part == target_part) & valid_mask
            count = int(valid_mask.sum().item())
            correct = int(correct_mask.sum().item())
            self.quantized_count += count
            self.quantized_correct["top1"] += correct
            self.quantized_part_count[part_name] += count
            self.quantized_part_correct[part_name]["top1"] += correct

            if self.include_slots:
                for slot_idx in range(int(self.layout.part_token_num[part_name])):
                    slot_valid = valid_mask[:, slot_idx]
                    slot_count = int(slot_valid.sum().item())
                    if slot_count <= 0:
                        continue
                    self.quantized_slot_count[part_name][slot_idx] += slot_count
                    self.quantized_slot_correct[part_name][slot_idx]["top1"] += int(
                        correct_mask[:, slot_idx].sum().item()
                    )

    def _topk_summary(
        self,
        *,
        correct: dict[str, int],
        count: int,
        part_correct: dict[str, dict[str, int]],
        part_count: dict[str, int],
        slot_correct: dict[str, list[dict[str, int]]],
        slot_count: dict[str, list[int]],
        keys: tuple[str, ...],
    ) -> dict[str, object]:
        overall = {"count": int(count)}
        overall.update({key: _ratio(int(correct[key]), int(count)) for key in keys})

        by_part: dict[str, dict[str, float | int]] = {}
        by_slot: dict[str, list[dict[str, float | int]]] = {}
        for part_name in self.part_names:
            p_count = int(part_count[part_name])
            item: dict[str, float | int] = {"count": p_count}
            item.update({key: _ratio(int(part_correct[part_name][key]), p_count) for key in keys})
            by_part[part_name] = item

            if self.include_slots:
                slots = []
                for slot_idx, s_count in enumerate(slot_count[part_name]):
                    s_count = int(s_count)
                    slot_item: dict[str, float | int] = {"slot": int(slot_idx), "count": s_count}
                    slot_item.update(
                        {key: _ratio(int(slot_correct[part_name][slot_idx][key]), s_count) for key in keys}
                    )
                    slots.append(slot_item)
                by_slot[part_name] = slots

        result: dict[str, object] = {"overall": overall, "by_part": by_part}
        if self.include_slots:
            result["by_slot"] = by_slot
        return result

    def summary(self) -> dict[str, object]:
        """Return JSON-serializable accumulated metrics."""
        logit_keys = tuple(_topk_key(k) for k in self.topk)
        return {
            "topk": [int(k) for k in self.topk],
            "logit": self._topk_summary(
                correct=self.logit_correct,
                count=self.logit_count,
                part_correct=self.logit_part_correct,
                part_count=self.logit_part_count,
                slot_correct=self.logit_slot_correct,
                slot_count=self.logit_slot_count,
                keys=logit_keys,
            ),
            "quantized": self._topk_summary(
                correct=self.quantized_correct,
                count=self.quantized_count,
                part_correct=self.quantized_part_correct,
                part_count=self.quantized_part_count,
                slot_correct=self.quantized_slot_correct,
                slot_count=self.quantized_slot_count,
                keys=("top1",),
            ),
        }


__all__ = ["PartTokenTopKAccumulator", "parse_topk_list"]
