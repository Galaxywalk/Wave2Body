# -*- coding: utf-8 -*-
"""
Part schema and layout utilities for part-level VQ-VAE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple, Union

import torch


DEFAULT_PART_NAMES: Tuple[str, ...] = (
    'L_Leg',
    'R_Leg',
    'SpineCore',
    'L_Arm',
    'R_Arm',
)

DEFAULT_PART_JOINTS: Dict[str, Tuple[int, ...]] = {
    'L_Leg': (1, 4, 7, 10),
    'R_Leg': (2, 5, 8, 11),
    'SpineCore': (3, 6, 9, 12, 15),
    'L_Arm': (9, 13, 16, 18, 20),
    'R_Arm': (9, 14, 17, 19, 21),
}

DEFAULT_PART_TOKEN_NUM: Dict[str, int] = {
    'L_Leg': 3,
    'R_Leg': 3,
    'SpineCore': 4,
    'L_Arm': 3,
    'R_Arm': 3,
}


def _normalize_part_value(
    value: Union[int, Mapping[str, int]],
    part_names: Iterable[str],
    field_name: str,
) -> Dict[str, int]:
    names = list(part_names)
    if isinstance(value, int):
        return {name: value for name in names}

    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be int or mapping[str, int], got {type(value)}")

    out: Dict[str, int] = {}
    for name in names:
        if name not in value:
            raise KeyError(f"Missing key '{name}' in {field_name}")
        part_val = int(value[name])
        if part_val <= 0:
            raise ValueError(f"{field_name}['{name}'] must be > 0, got {part_val}")
        out[name] = part_val
    return out


def _normalize_part_joints(
    part_joints: Mapping[str, Iterable[int]],
    part_names: Iterable[str],
) -> Dict[str, Tuple[int, ...]]:
    names = list(part_names)
    out: Dict[str, Tuple[int, ...]] = {}
    for name in names:
        if name not in part_joints:
            raise KeyError(f"Missing key '{name}' in part_joints")
        joints_tuple = tuple(int(i) for i in part_joints[name])
        if len(joints_tuple) == 0:
            raise ValueError(f"part_joints['{name}'] must be non-empty")
        out[name] = joints_tuple
    return out


def _build_slices(part_names: Iterable[str], part_token_num: Mapping[str, int]) -> Dict[str, slice]:
    start = 0
    slices: Dict[str, slice] = {}
    for name in part_names:
        n = int(part_token_num[name])
        slices[name] = slice(start, start + n)
        start += n
    return slices


def _build_offsets(part_names: Iterable[str], part_codebook_size: Mapping[str, int]) -> Dict[str, int]:
    start = 0
    offsets: Dict[str, int] = {}
    for name in part_names:
        offsets[name] = start
        start += int(part_codebook_size[name])
    return offsets


@dataclass(frozen=True)
class PartLayout:
    num_joints: int
    root_index: int
    part_names: Tuple[str, ...]
    part_joints: Dict[str, Tuple[int, ...]]
    part_token_num: Dict[str, int]
    part_codebook_size: Dict[str, int]
    part_token_slices: Dict[str, slice]
    part_codebook_offsets: Dict[str, int]
    total_tokens: int
    total_codebook_size: int

    def split_joints(self, joints: torch.Tensor) -> Dict[str, torch.Tensor]:
        if joints.dim() != 3:
            raise ValueError(f"Expected joints shape [B, J, C], got {tuple(joints.shape)}")
        if joints.shape[1] != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joints, got {joints.shape[1]}")

        out: Dict[str, torch.Tensor] = {}
        for name in self.part_names:
            idx = torch.as_tensor(self.part_joints[name], device=joints.device, dtype=torch.long)
            out[name] = joints.index_select(1, idx)
        return out

    def merge_joints(self, part_outputs: Mapping[str, torch.Tensor], force_root_zero: bool = True) -> torch.Tensor:
        if not part_outputs:
            raise ValueError("part_outputs cannot be empty")

        first_name = self.part_names[0]
        if first_name not in part_outputs:
            raise KeyError(f"Missing part output for '{first_name}'")
        ref = part_outputs[first_name]
        if ref.dim() != 3:
            raise ValueError(f"Part output must be [B, J_part, C], got {tuple(ref.shape)}")

        bsz = ref.shape[0]
        cdim = ref.shape[2]
        merged = torch.zeros(bsz, self.num_joints, cdim, device=ref.device, dtype=ref.dtype)
        counts = torch.zeros(bsz, self.num_joints, 1, device=ref.device, dtype=ref.dtype)

        for name in self.part_names:
            if name not in part_outputs:
                raise KeyError(f"Missing part output for '{name}'")
            part_tensor = part_outputs[name]
            expected_j = len(self.part_joints[name])
            if part_tensor.shape[:2] != (bsz, expected_j):
                raise ValueError(
                    f"Part '{name}' shape mismatch, expected [B,{expected_j},C], got {tuple(part_tensor.shape)}"
                )
            idx = torch.as_tensor(self.part_joints[name], device=part_tensor.device, dtype=torch.long)
            merged[:, idx, :] += part_tensor
            counts[:, idx, :] += 1.0

        counts_safe = torch.clamp(counts, min=1.0)
        merged = merged / counts_safe

        if force_root_zero:
            merged[:, self.root_index, :] = 0.0
        return merged

    def split_part_tokens(self, token_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        if token_tensor.dim() != 3:
            raise ValueError(f"Expected token tensor [B, N, D], got {tuple(token_tensor.shape)}")
        if token_tensor.shape[1] != self.total_tokens:
            raise ValueError(f"Expected total tokens {self.total_tokens}, got {token_tensor.shape[1]}")
        return {name: token_tensor[:, self.part_token_slices[name], :] for name in self.part_names}

    def concat_part_tokens(self, part_tokens: Mapping[str, torch.Tensor]) -> torch.Tensor:
        chunks: List[torch.Tensor] = []
        for name in self.part_names:
            if name not in part_tokens:
                raise KeyError(f"Missing part token chunk for '{name}'")
            x = part_tokens[name]
            expected = self.part_token_num[name]
            if x.dim() != 3:
                raise ValueError(f"Part token chunk '{name}' must be [B,N,D], got {tuple(x.shape)}")
            if x.shape[1] != expected:
                raise ValueError(f"Part token count mismatch for '{name}', expected {expected}, got {x.shape[1]}")
            chunks.append(x)
        return torch.cat(chunks, dim=1)

    def local_to_global_indices(self, local_idx: torch.Tensor, part_name: str) -> torch.Tensor:
        if part_name not in self.part_codebook_offsets:
            raise KeyError(f"Unknown part name '{part_name}'")
        return local_idx + self.part_codebook_offsets[part_name]

    def global_to_local_indices(self, global_idx: torch.Tensor, part_name: str) -> torch.Tensor:
        if part_name not in self.part_codebook_offsets:
            raise KeyError(f"Unknown part name '{part_name}'")
        return global_idx - self.part_codebook_offsets[part_name]

    def split_global_indices(self, indices_global: torch.Tensor) -> Dict[str, torch.Tensor]:
        if indices_global.dim() != 2:
            raise ValueError(f"Expected indices [B, N], got {tuple(indices_global.shape)}")
        if indices_global.shape[1] != self.total_tokens:
            raise ValueError(f"Expected total tokens {self.total_tokens}, got {indices_global.shape[1]}")
        out: Dict[str, torch.Tensor] = {}
        for name in self.part_names:
            sl = self.part_token_slices[name]
            out[name] = self.global_to_local_indices(indices_global[:, sl], name)
        return out

    def merge_local_indices_to_global(self, local_indices_by_part: Mapping[str, torch.Tensor]) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        for name in self.part_names:
            if name not in local_indices_by_part:
                raise KeyError(f"Missing local indices for part '{name}'")
            local_idx = local_indices_by_part[name]
            if local_idx.dim() != 2:
                raise ValueError(f"Local indices for '{name}' must be [B, N_part], got {tuple(local_idx.shape)}")
            expected = self.part_token_num[name]
            if local_idx.shape[1] != expected:
                raise ValueError(f"Local indices token count mismatch for '{name}', expected {expected}, got {local_idx.shape[1]}")
            parts.append(self.local_to_global_indices(local_idx, name))
        return torch.cat(parts, dim=1)

    def build_part_ids(self, device: torch.device) -> torch.Tensor:
        chunks: List[torch.Tensor] = []
        for pid, name in enumerate(self.part_names):
            chunks.append(torch.full((self.part_token_num[name],), pid, dtype=torch.long, device=device))
        return torch.cat(chunks, dim=0)


def create_part_layout(
    num_joints: int = 22,
    root_index: int = 0,
    part_names: Iterable[str] = DEFAULT_PART_NAMES,
    part_joints: Mapping[str, Iterable[int]] = DEFAULT_PART_JOINTS,
    part_token_num: Union[int, Mapping[str, int]] = DEFAULT_PART_TOKEN_NUM,
    part_codebook_size: Union[int, Mapping[str, int]] = 64,
) -> PartLayout:
    part_names_tuple = tuple(part_names)
    normalized_part_joints = _normalize_part_joints(part_joints, part_names_tuple)
    normalized_part_token_num = _normalize_part_value(part_token_num, part_names_tuple, 'part_token_num')
    normalized_part_codebook_size = _normalize_part_value(part_codebook_size, part_names_tuple, 'part_codebook_size')

    part_token_slices = _build_slices(part_names_tuple, normalized_part_token_num)
    part_codebook_offsets = _build_offsets(part_names_tuple, normalized_part_codebook_size)

    total_tokens = sum(normalized_part_token_num[name] for name in part_names_tuple)
    total_codebook_size = sum(normalized_part_codebook_size[name] for name in part_names_tuple)

    return PartLayout(
        num_joints=num_joints,
        root_index=root_index,
        part_names=part_names_tuple,
        part_joints=normalized_part_joints,
        part_token_num=normalized_part_token_num,
        part_codebook_size=normalized_part_codebook_size,
        part_token_slices=part_token_slices,
        part_codebook_offsets=part_codebook_offsets,
        total_tokens=total_tokens,
        total_codebook_size=total_codebook_size,
    )
