# -*- coding: utf-8 -*-
"""Evidence-grounded token translator components.

The residual-gated translator separates a context-only posterior from a local
evidence correction:

    logits_full = logits_ctx + gate_part * logits_evidence_delta

The context branch uses per-pose-token attention masks so the token(s) for part
``p`` cannot attend to mmWave tokens from part ``p``. This makes qctx a
context-conditioned posterior instead of a synthetic all-zero-input posterior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...human_body_tokenizer.part_schema import PartLayout, create_part_layout
from ..cross_attention_translator import CrossAttentionLayer, CrossAttentionTokenTranslator, RMSNorm, SelfAttentionLayer


def _as_tuple_names(value: Iterable[str] | None, fallback: Iterable[str]) -> tuple[str, ...]:
    if value is None:
        return tuple(str(x) for x in fallback)
    out = tuple(str(x) for x in value)
    if not out:
        raise ValueError("part_names must not be empty")
    return out


def _normalize_part_codebook_size(
    value: int | Mapping[str, int] | None,
    *,
    total_codebook_size: int,
    part_names: tuple[str, ...],
) -> int | dict[str, int]:
    if value is not None:
        return dict(value) if isinstance(value, Mapping) else int(value)
    if total_codebook_size <= 0:
        raise ValueError(f"codebook_size must be positive, got {total_codebook_size}")
    if total_codebook_size % len(part_names) != 0:
        raise ValueError(
            "part_codebook_size is required when global codebook_size is not divisible by num parts: "
            f"{total_codebook_size} / {len(part_names)}"
        )
    return int(total_codebook_size // len(part_names))


def _make_norm(dim: int, use_rms_norm: bool) -> nn.Module:
    return RMSNorm(dim) if use_rms_norm else nn.LayerNorm(dim)


def _masked_mean(tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool ``tokens`` with a bool mask over the token axis."""
    if valid_mask.dtype != torch.bool:
        valid_mask = valid_mask != 0
    weights = valid_mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (tokens * weights).sum(dim=1) / denom


def _part_slices(layout: PartLayout, logits: torch.Tensor, part_name: str) -> tuple[slice, int, int]:
    token_slice = layout.part_token_slices[part_name]
    offset = int(layout.part_codebook_offsets[part_name])
    size = int(layout.part_codebook_size[part_name])
    end = offset + size
    if end > int(logits.shape[-1]):
        raise ValueError(
            f"Invalid codebook slice for {part_name}: offset={offset}, size={size}, "
            f"logits_dim={int(logits.shape[-1])}"
        )
    return token_slice, offset, size


def part_local_js_divergence(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    *,
    layout: PartLayout,
    include_part_ids: Iterable[int] | None = None,
    exclude_part_ids: Iterable[int] | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean JS divergence over legal part-local codebook slices.

    Returns a scalar. Both inputs are global logits ``[B,T,C]``; invalid global
    ranges are ignored by slicing each body part's legal code range.
    """
    if logits_a.shape != logits_b.shape:
        raise ValueError(f"logits shape mismatch: {tuple(logits_a.shape)} vs {tuple(logits_b.shape)}")
    include = None if include_part_ids is None else {int(x) for x in include_part_ids}
    exclude = set() if exclude_part_ids is None else {int(x) for x in exclude_part_ids}

    total = logits_a.new_zeros((), dtype=torch.float32)
    count = logits_a.new_zeros((), dtype=torch.float32)
    for pid, part_name in enumerate(layout.part_names):
        if include is not None and pid not in include:
            continue
        if pid in exclude:
            continue
        token_slice, offset, size = _part_slices(layout, logits_a, part_name)
        a_logits = logits_a[:, token_slice, offset : offset + size].float()
        b_logits = logits_b[:, token_slice, offset : offset + size].float()
        a_logp = F.log_softmax(a_logits, dim=-1)
        b_logp = F.log_softmax(b_logits, dim=-1)
        a_prob = a_logp.exp()
        b_prob = b_logp.exp()
        mix = 0.5 * (a_prob + b_prob)
        mix_log = torch.log(mix.clamp_min(float(eps)))
        js = 0.5 * (
            (a_prob * (a_logp - mix_log)).sum(dim=-1)
            + (b_prob * (b_logp - mix_log)).sum(dim=-1)
        )
        total = total + js.sum()
        count = count + torch.as_tensor(float(js.numel()), device=logits_a.device, dtype=torch.float32)
    return total / count.clamp_min(1.0)


def part_specificity_from_logits(
    logits: torch.Tensor,
    *,
    layout: PartLayout,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return entropy-based specificity per sample and part ``[B,P]``.

    ``0`` is maximally uncertain under a uniform local posterior, and ``1`` is
    maximally specific. This intentionally uses plain normalized entropy; D-aware
    uncertainty remains an evaluation/ablation extension.
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be [B,T,C], got {tuple(logits.shape)}")
    part_values: list[torch.Tensor] = []
    for part_name in layout.part_names:
        token_slice, offset, size = _part_slices(layout, logits, part_name)
        local_logits = logits[:, token_slice, offset : offset + size].float()
        probs = torch.softmax(local_logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(float(eps)))).sum(dim=-1)
        max_entropy = torch.log(torch.as_tensor(float(size), device=logits.device, dtype=entropy.dtype))
        specificity = 1.0 - entropy / max_entropy.clamp_min(float(eps))
        part_values.append(specificity.mean(dim=1))
    return torch.stack(part_values, dim=1)


def part_local_delta_l2(
    residual_logits: torch.Tensor,
    *,
    layout: PartLayout,
) -> torch.Tensor:
    """Mean squared residual logit scale over legal part-local code ranges."""
    total = residual_logits.new_zeros((), dtype=torch.float32)
    count = residual_logits.new_zeros((), dtype=torch.float32)
    for part_name in layout.part_names:
        token_slice, offset, size = _part_slices(layout, residual_logits, part_name)
        local = residual_logits[:, token_slice, offset : offset + size].float()
        total = total + (local * local).sum()
        count = count + torch.as_tensor(float(local.numel()), device=residual_logits.device, dtype=torch.float32)
    return total / count.clamp_min(1.0)


def _broadcast_force_gate(
    force_gate: float | torch.Tensor,
    *,
    gate_part: torch.Tensor,
    num_parts: int,
    batch_size: int,
) -> torch.Tensor:
    if torch.is_tensor(force_gate):
        forced = force_gate.to(device=gate_part.device, dtype=gate_part.dtype)
        if forced.dim() == 0:
            return forced.reshape(1, 1).expand_as(gate_part)
        if forced.shape == gate_part.shape:
            return forced
        if forced.dim() == 1 and int(forced.shape[0]) == int(num_parts):
            return forced.unsqueeze(0).expand_as(gate_part)
        if forced.dim() == 1 and int(forced.shape[0]) == int(batch_size):
            return forced.unsqueeze(1).expand_as(gate_part)
        try:
            return torch.broadcast_to(forced, gate_part.shape)
        except RuntimeError as exc:
            raise ValueError(
                "force_gate tensor must be broadcastable to "
                f"[batch,num_parts]={tuple(gate_part.shape)}, got {tuple(forced.shape)}"
            ) from exc
    return torch.full_like(gate_part, float(force_gate))


@dataclass
class ContextEvidenceOutput:
    pred_z_e: torch.Tensor
    pred_logits: torch.Tensor
    aux: dict[str, torch.Tensor]


class ContextEvidenceResidualGatedTranslator(nn.Module):
    """Context/evidence residual-gated token translator.

    The public forward interface mirrors ``CrossAttentionTokenTranslator``:
    ``forward(..., return_logits=True)`` returns ``(pred_z_e, pred_logits)``.
    Passing ``return_aux=True`` returns ``(pred_z_e, pred_logits, aux)`` where
    aux exposes q0/qctx/evidence-residual tensors for grounding losses.
    """

    def __init__(
        self,
        mmwave_token_dim: int = 32,
        pose_token_dim: int = 256,
        num_pose_tokens: int = 24,
        num_mmwave_tokens: int = 22,
        hidden_dim: int = 256,
        num_cross_attn_layers: int = 4,
        num_self_attn_layers: int = 2,
        heads: int = 8,
        dim_head: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_rms_norm: bool = True,
        use_sdpa: bool = False,
        codebook_size: int | None = None,
        part_names: Iterable[str] | None = None,
        part_joints: Mapping[str, Iterable[int]] | None = None,
        part_token_num: int | Mapping[str, int] | None = None,
        part_codebook_size: int | Mapping[str, int] | None = None,
        root_index: int = 0,
        evidence_hidden_dim: int | None = None,
        gate_hidden_dim: int | None = None,
        gate_init_bias: float = 1.0,
    ):
        super().__init__()
        if codebook_size is None:
            raise ValueError("codebook_size is required for evidence-grounded logits")
        if part_joints is None:
            raise ValueError("part_joints is required for context/evidence masking")
        if part_token_num is None:
            raise ValueError("part_token_num is required for context/evidence masking")

        names = _as_tuple_names(part_names, part_joints.keys())
        local_code_size = _normalize_part_codebook_size(
            part_codebook_size,
            total_codebook_size=int(codebook_size),
            part_names=names,
        )
        self.layout = create_part_layout(
            num_joints=int(num_mmwave_tokens),
            root_index=int(root_index),
            part_names=names,
            part_joints=part_joints,
            part_token_num=part_token_num,
            part_codebook_size=local_code_size,
        )
        if int(self.layout.total_tokens) != int(num_pose_tokens):
            raise ValueError(f"num_pose_tokens={num_pose_tokens} does not match layout={self.layout.total_tokens}")
        if int(self.layout.total_codebook_size) != int(codebook_size):
            raise ValueError(
                f"codebook_size={codebook_size} does not match layout={self.layout.total_codebook_size}"
            )

        self.mmwave_token_dim = int(mmwave_token_dim)
        self.pose_token_dim = int(pose_token_dim)
        self.num_pose_tokens = int(num_pose_tokens)
        self.num_mmwave_tokens = int(num_mmwave_tokens)
        self.hidden_dim = int(hidden_dim)
        self.codebook_size = int(codebook_size)
        self.num_parts = len(self.layout.part_names)
        self.supports_logits = True

        part_valid = torch.zeros((self.num_parts, self.num_mmwave_tokens), dtype=torch.bool)
        for pid, part_name in enumerate(self.layout.part_names):
            idx = torch.as_tensor(self.layout.part_joints[part_name], dtype=torch.long)
            if bool((idx < 0).any().item()) or bool((idx >= self.num_mmwave_tokens).any().item()):
                raise ValueError(f"part_joints[{part_name}] contains out-of-range mmWave token ids")
            part_valid[pid, idx] = True
        self.register_buffer("part_mmwave_mask", part_valid, persistent=False)
        self.register_buffer("token_part_ids", self.layout.build_part_ids(torch.device("cpu")), persistent=False)

        self.mmwave_proj = nn.Linear(self.mmwave_token_dim, self.hidden_dim)
        self.pose_token_embed = nn.Parameter(torch.empty(1, self.num_pose_tokens, self.hidden_dim))
        self.pose_pos_embed = nn.Parameter(torch.empty(1, self.num_pose_tokens, self.hidden_dim))
        nn.init.trunc_normal_(self.pose_token_embed, std=0.02)
        nn.init.trunc_normal_(self.pose_pos_embed, std=0.02)

        self.context_cross_layers = nn.ModuleList(
            [
                CrossAttentionLayer(
                    dim=self.hidden_dim,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    use_rms_norm=use_rms_norm,
                    use_sdpa=use_sdpa,
                )
                for _ in range(int(num_cross_attn_layers))
            ]
        )
        self.context_self_layers = nn.ModuleList(
            [
                SelfAttentionLayer(
                    dim=self.hidden_dim,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    use_rms_norm=use_rms_norm,
                    use_sdpa=use_sdpa,
                )
                for _ in range(int(num_self_attn_layers))
            ]
        )

        self.ctx_output_proj = nn.Sequential(
            _make_norm(self.hidden_dim, use_rms_norm),
            nn.Linear(self.hidden_dim, self.pose_token_dim),
            nn.Dropout(dropout),
        )
        self.ctx_logit_head = nn.Sequential(
            _make_norm(self.pose_token_dim, use_rms_norm),
            nn.Linear(self.pose_token_dim, self.codebook_size),
        )
        self.global_prior_logits = nn.Parameter(torch.zeros(1, self.num_pose_tokens, self.codebook_size))

        evi_h = int(evidence_hidden_dim or self.hidden_dim)
        self.evidence_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, evi_h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(evi_h, self.hidden_dim),
            nn.GELU(),
        )
        self.evidence_z_delta = nn.Sequential(
            _make_norm(self.hidden_dim, use_rms_norm),
            nn.Linear(self.hidden_dim, self.pose_token_dim),
            nn.Dropout(dropout),
        )
        self.evidence_logit_head = nn.Sequential(
            _make_norm(self.pose_token_dim, use_rms_norm),
            nn.Linear(self.pose_token_dim, self.codebook_size),
        )

        gate_h = int(gate_hidden_dim or self.hidden_dim)
        self.gate_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, gate_h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_h, 1),
        )
        final_gate = self.gate_head[-1]
        if isinstance(final_gate, nn.Linear):
            nn.init.zeros_(final_gate.weight)
            nn.init.constant_(final_gate.bias, float(gate_init_bias))

        self.learned_mask_token = nn.Parameter(torch.zeros(1, 1, self.mmwave_token_dim))
        self.input_dropout = nn.Dropout(dropout)

    def _base_valid_mask(self, mmwave_tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        bsz, n_tokens, _ = mmwave_tokens.shape
        if n_tokens != self.num_mmwave_tokens:
            raise ValueError(f"Expected {self.num_mmwave_tokens} mmWave tokens, got {n_tokens}")
        if mask is None:
            return torch.ones((bsz, n_tokens), device=mmwave_tokens.device, dtype=torch.bool)
        valid = mask.to(device=mmwave_tokens.device)
        if valid.dtype != torch.bool:
            valid = valid != 0
        if valid.shape != (bsz, n_tokens):
            raise ValueError(f"mask must be [B,{n_tokens}], got {tuple(mask.shape)}")
        return valid

    def _token_masks(self, base_valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        part_mask = self.part_mmwave_mask.to(device=base_valid.device)
        token_part = self.token_part_ids.to(device=base_valid.device)
        token_local = part_mask.index_select(0, token_part)  # [T,J]
        local = base_valid.unsqueeze(1) & token_local.unsqueeze(0)
        context = base_valid.unsqueeze(1) & ~token_local.unsqueeze(0)
        return context, local

    def _context_self_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        token_part = self.token_part_ids.to(device=device)
        same_part = token_part.unsqueeze(0) == token_part.unsqueeze(1)
        return same_part.unsqueeze(0).expand(int(batch_size), -1, -1)

    def _part_gate(self, mmwave_proj: torch.Tensor, base_valid: torch.Tensor) -> torch.Tensor:
        gates = []
        part_mask = self.part_mmwave_mask.to(device=mmwave_proj.device)
        for pid in range(self.num_parts):
            local_valid = base_valid & part_mask[pid].unsqueeze(0)
            context_valid = base_valid & ~part_mask[pid].unsqueeze(0)
            local_mean = _masked_mean(mmwave_proj, local_valid)
            context_mean = _masked_mean(mmwave_proj, context_valid)
            gates.append(self.gate_head(torch.cat([local_mean, context_mean], dim=-1)).squeeze(-1))
        return torch.sigmoid(torch.stack(gates, dim=1))

    def _local_summary(self, mmwave_proj: torch.Tensor, local_mask: torch.Tensor) -> torch.Tensor:
        weights = local_mask.to(device=mmwave_proj.device, dtype=mmwave_proj.dtype)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.matmul(weights, mmwave_proj) / denom

    def forward(
        self,
        mmwave_tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_logits: bool = False,
        return_aux: bool = False,
        force_gate: float | torch.Tensor | None = None,
    ):
        bsz = int(mmwave_tokens.shape[0])
        base_valid = self._base_valid_mask(mmwave_tokens, mask)
        context_mask, local_mask = self._token_masks(base_valid)

        mmwave_proj = self.input_dropout(self.mmwave_proj(mmwave_tokens))
        ctx_hidden = self.pose_token_embed.expand(bsz, -1, -1) + self.pose_pos_embed.expand(bsz, -1, -1)
        for layer in self.context_cross_layers:
            ctx_hidden = layer(ctx_hidden, mmwave_proj, context_mask)
        context_self_mask = self._context_self_mask(bsz, mmwave_tokens.device)
        for layer in self.context_self_layers:
            ctx_hidden = layer(ctx_hidden, context_self_mask)

        ctx_z_e = self.ctx_output_proj(ctx_hidden)
        prior_logits = self.global_prior_logits.expand(bsz, -1, -1)
        ctx_delta_logits = self.ctx_logit_head(ctx_z_e)
        ctx_logits = prior_logits + ctx_delta_logits

        local_summary = self._local_summary(mmwave_proj, local_mask)
        evi_hidden = self.evidence_mlp(torch.cat([ctx_hidden, local_summary, ctx_hidden * local_summary], dim=-1))
        evi_z_delta = self.evidence_z_delta(evi_hidden)
        evi_residual_logits = self.evidence_logit_head(evi_z_delta)

        gate_part = self._part_gate(mmwave_proj, base_valid)
        if force_gate is not None:
            gate_part = _broadcast_force_gate(
                force_gate,
                gate_part=gate_part,
                num_parts=self.num_parts,
                batch_size=bsz,
            )
        token_part = self.token_part_ids.to(device=mmwave_tokens.device)
        gate_token = gate_part.index_select(1, token_part).unsqueeze(-1)

        pred_z_e = ctx_z_e + gate_token * evi_z_delta
        pred_logits = ctx_logits + gate_token * evi_residual_logits
        if not return_logits and not return_aux:
            return pred_z_e

        aux = {
            "prior_logits": prior_logits,
            "ctx_logits": ctx_logits,
            "ctx_delta_logits": ctx_delta_logits,
            "ctx_z_e": ctx_z_e,
            "evidence_residual_logits": evi_residual_logits,
            "evidence_z_delta": evi_z_delta,
            "gate_part": gate_part,
            "gate_token": gate_token.squeeze(-1),
            "context_attention_mask": context_mask,
            "context_self_attention_mask": context_self_mask,
            "local_attention_mask": local_mask,
        }
        if return_aux:
            return pred_z_e, pred_logits, aux
        return pred_z_e, pred_logits

    def _part_mask_1d(self, part: int | str, *, device: torch.device) -> torch.Tensor:
        if isinstance(part, str):
            if part not in self.layout.part_names:
                raise KeyError(f"Unknown part name: {part}")
            part_id = self.layout.part_names.index(part)
        else:
            part_id = int(part)
        if part_id < 0 or part_id >= self.num_parts:
            raise ValueError(f"part id out of range: {part_id}")
        return self.part_mmwave_mask.to(device=device)[part_id]

    def replace_part_evidence(
        self,
        mmwave_tokens: torch.Tensor,
        part: int | str,
        *,
        strategy: str = "zero",
        noise_std: float = 0.05,
        dropout_prob: float = 1.0,
    ) -> torch.Tensor:
        """Return tokens with one part's local evidence replaced."""
        strategy = str(strategy).strip().lower()
        out = mmwave_tokens.clone()
        part_mask = self._part_mask_1d(part, device=mmwave_tokens.device)
        keep_mask = ~part_mask
        if strategy == "zero":
            replacement = torch.zeros_like(out[:, part_mask, :])
        elif strategy == "mean":
            mean = _masked_mean(mmwave_tokens, keep_mask.unsqueeze(0).expand(mmwave_tokens.shape[0], -1))
            replacement = mean.unsqueeze(1).expand(-1, int(part_mask.sum().item()), -1)
        elif strategy == "learned":
            replacement = self.learned_mask_token.to(device=out.device, dtype=out.dtype).expand(
                out.shape[0],
                int(part_mask.sum().item()),
                -1,
            )
        elif strategy in {"noise", "gaussian", "gaussian_noise", "noisy_mean"}:
            mean = _masked_mean(mmwave_tokens, keep_mask.unsqueeze(0).expand(mmwave_tokens.shape[0], -1))
            centered = mmwave_tokens[:, keep_mask, :] - mean.unsqueeze(1)
            std = centered.float().std(dim=1, unbiased=False).to(dtype=out.dtype).clamp_min(1e-6)
            replacement = mean.unsqueeze(1) + float(noise_std) * std.unsqueeze(1) * torch.randn_like(out[:, part_mask, :])
        elif strategy in {"replacement", "swap", "batch_replacement"}:
            replacement = torch.roll(mmwave_tokens[:, part_mask, :], shifts=1, dims=0)
        elif strategy in {"dropout", "token_dropout"}:
            replacement = out[:, part_mask, :]
            drop = torch.rand(replacement.shape[:2], device=out.device).unsqueeze(-1) < float(dropout_prob)
            replacement = torch.where(drop, torch.zeros_like(replacement), replacement)
        else:
            raise ValueError(f"Unsupported part evidence replacement strategy: {strategy}")
        out[:, part_mask, :] = replacement.to(dtype=out.dtype)
        return out

    def corrupt_tokens(
        self,
        mmwave_tokens: torch.Tensor,
        *,
        strategy: str = "heavy_dropout",
        noise_std: float = 0.1,
        dropout_prob: float = 0.8,
    ) -> torch.Tensor:
        """Return a corrupted-input counterfactual for specificity suppression."""
        strategy = str(strategy).strip().lower()
        if strategy == "zero":
            return torch.zeros_like(mmwave_tokens)
        if strategy == "mean":
            mean = mmwave_tokens.mean(dim=1, keepdim=True)
            return mean.expand_as(mmwave_tokens).clone()
        if strategy in {"noise", "gaussian", "gaussian_noise", "noisy_mean"}:
            mean = mmwave_tokens.mean(dim=1, keepdim=True)
            std = (mmwave_tokens - mean).float().std(dim=1, keepdim=True, unbiased=False).to(dtype=mmwave_tokens.dtype)
            return mean + float(noise_std) * std.clamp_min(1e-6) * torch.randn_like(mmwave_tokens)
        if strategy in {"shuffle", "replacement", "batch_replacement"}:
            return torch.roll(mmwave_tokens, shifts=1, dims=0)
        if strategy in {"heavy_dropout", "dropout", "token_dropout"}:
            keep = torch.rand(mmwave_tokens.shape[:2], device=mmwave_tokens.device).unsqueeze(-1) >= float(dropout_prob)
            return torch.where(keep, mmwave_tokens, torch.zeros_like(mmwave_tokens))
        if strategy == "learned":
            return self.learned_mask_token.to(device=mmwave_tokens.device, dtype=mmwave_tokens.dtype).expand_as(mmwave_tokens)
        raise ValueError(f"Unsupported corrupted token strategy: {strategy}")


class EvidenceGroundedAdapterTranslator(nn.Module):
    """Frozen base translator plus trainable evidence correction adapter.

    This wrapper is intended for evidence-grounded fine-tuning from an accepted
    learned posterior checkpoint. The base translator supplies both the full
    posterior and a context-only posterior; the adapter learns a small residual
    correction without changing the frozen base path.
    """

    def __init__(
        self,
        mmwave_token_dim: int = 32,
        pose_token_dim: int = 256,
        num_pose_tokens: int = 24,
        num_mmwave_tokens: int = 22,
        hidden_dim: int = 256,
        num_cross_attn_layers: int = 4,
        num_self_attn_layers: int = 2,
        heads: int = 8,
        dim_head: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_rms_norm: bool = True,
        use_sdpa: bool = False,
        codebook_size: int | None = None,
        part_names: Iterable[str] | None = None,
        part_joints: Mapping[str, Iterable[int]] | None = None,
        part_token_num: int | Mapping[str, int] | None = None,
        part_codebook_size: int | Mapping[str, int] | None = None,
        root_index: int = 0,
        adapter_hidden_dim: int | None = None,
        gate_hidden_dim: int | None = None,
        gate_init_bias: float = 5.0,
        freeze_base: bool = True,
    ):
        super().__init__()
        if codebook_size is None:
            raise ValueError("codebook_size is required for evidence-grounded adapter logits")
        if part_joints is None:
            raise ValueError("part_joints is required for evidence-grounded adapter masking")
        if part_token_num is None:
            raise ValueError("part_token_num is required for evidence-grounded adapter masking")

        names = _as_tuple_names(part_names, part_joints.keys())
        local_code_size = _normalize_part_codebook_size(
            part_codebook_size,
            total_codebook_size=int(codebook_size),
            part_names=names,
        )
        self.layout = create_part_layout(
            num_joints=int(num_mmwave_tokens),
            root_index=int(root_index),
            part_names=names,
            part_joints=part_joints,
            part_token_num=part_token_num,
            part_codebook_size=local_code_size,
        )
        if int(self.layout.total_tokens) != int(num_pose_tokens):
            raise ValueError(f"num_pose_tokens={num_pose_tokens} does not match layout={self.layout.total_tokens}")
        if int(self.layout.total_codebook_size) != int(codebook_size):
            raise ValueError(
                f"codebook_size={codebook_size} does not match layout={self.layout.total_codebook_size}"
            )

        self.mmwave_token_dim = int(mmwave_token_dim)
        self.pose_token_dim = int(pose_token_dim)
        self.num_pose_tokens = int(num_pose_tokens)
        self.num_mmwave_tokens = int(num_mmwave_tokens)
        self.hidden_dim = int(hidden_dim)
        self.codebook_size = int(codebook_size)
        self.num_parts = len(self.layout.part_names)
        self.freeze_base = bool(freeze_base)
        self.supports_logits = True

        self.base_translator = CrossAttentionTokenTranslator(
            mmwave_token_dim=mmwave_token_dim,
            pose_token_dim=pose_token_dim,
            num_pose_tokens=num_pose_tokens,
            num_mmwave_tokens=num_mmwave_tokens,
            hidden_dim=hidden_dim,
            num_cross_attn_layers=num_cross_attn_layers,
            num_self_attn_layers=num_self_attn_layers,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_rms_norm=use_rms_norm,
            use_sdpa=use_sdpa,
            codebook_size=codebook_size,
        )
        if self.freeze_base:
            for param in self.base_translator.parameters():
                param.requires_grad = False

        part_valid = torch.zeros((self.num_parts, self.num_mmwave_tokens), dtype=torch.bool)
        for pid, part_name in enumerate(self.layout.part_names):
            idx = torch.as_tensor(self.layout.part_joints[part_name], dtype=torch.long)
            if bool((idx < 0).any().item()) or bool((idx >= self.num_mmwave_tokens).any().item()):
                raise ValueError(f"part_joints[{part_name}] contains out-of-range mmWave token ids")
            part_valid[pid, idx] = True
        self.register_buffer("part_mmwave_mask", part_valid, persistent=False)
        self.register_buffer("token_part_ids", self.layout.build_part_ids(torch.device("cpu")), persistent=False)

        adapter_h = int(adapter_hidden_dim or hidden_dim)
        gate_h = int(gate_hidden_dim or hidden_dim)
        self.adapter_mmwave_proj = nn.Linear(self.mmwave_token_dim, adapter_h)
        self.adapter_ctx_proj = nn.Linear(self.pose_token_dim, adapter_h)
        self.adapter_mlp = nn.Sequential(
            nn.Linear(adapter_h * 3, adapter_h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_h, adapter_h),
            nn.GELU(),
        )
        self.adapter_z_delta = nn.Sequential(
            _make_norm(adapter_h, use_rms_norm),
            nn.Linear(adapter_h, self.pose_token_dim),
        )
        self.adapter_logit_delta = nn.Sequential(
            _make_norm(adapter_h, use_rms_norm),
            nn.Linear(adapter_h, self.codebook_size),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(adapter_h * 2, gate_h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_h, 1),
        )
        final_gate = self.gate_head[-1]
        if isinstance(final_gate, nn.Linear):
            nn.init.zeros_(final_gate.weight)
            nn.init.constant_(final_gate.bias, float(gate_init_bias))
        final_logit = self.adapter_logit_delta[-1]
        if isinstance(final_logit, nn.Linear):
            nn.init.zeros_(final_logit.weight)
            if final_logit.bias is not None:
                nn.init.zeros_(final_logit.bias)
        final_z = self.adapter_z_delta[-1]
        if isinstance(final_z, nn.Linear):
            nn.init.zeros_(final_z.weight)
            if final_z.bias is not None:
                nn.init.zeros_(final_z.bias)

        self.learned_mask_token = nn.Parameter(torch.zeros(1, 1, self.mmwave_token_dim))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_base:
            self.base_translator.eval()
        return self

    def _base_valid_mask(self, mmwave_tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        bsz, n_tokens, _ = mmwave_tokens.shape
        if n_tokens != self.num_mmwave_tokens:
            raise ValueError(f"Expected {self.num_mmwave_tokens} mmWave tokens, got {n_tokens}")
        if mask is None:
            return torch.ones((bsz, n_tokens), device=mmwave_tokens.device, dtype=torch.bool)
        valid = mask.to(device=mmwave_tokens.device)
        if valid.dtype != torch.bool:
            valid = valid != 0
        if valid.shape != (bsz, n_tokens):
            raise ValueError(f"mask must be [B,{n_tokens}], got {tuple(mask.shape)}")
        return valid

    def _token_masks(self, base_valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        part_mask = self.part_mmwave_mask.to(device=base_valid.device)
        token_part = self.token_part_ids.to(device=base_valid.device)
        token_local = part_mask.index_select(0, token_part)
        local = base_valid.unsqueeze(1) & token_local.unsqueeze(0)
        context = base_valid.unsqueeze(1) & ~token_local.unsqueeze(0)
        return context, local

    def _pool_by_mask(self, projected_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(device=projected_tokens.device, dtype=projected_tokens.dtype)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.matmul(weights, projected_tokens) / denom

    def _part_gate(self, adapter_proj: torch.Tensor, base_valid: torch.Tensor) -> torch.Tensor:
        gates = []
        part_mask = self.part_mmwave_mask.to(device=adapter_proj.device)
        for pid in range(self.num_parts):
            local_valid = base_valid & part_mask[pid].unsqueeze(0)
            context_valid = base_valid & ~part_mask[pid].unsqueeze(0)
            local_mean = _masked_mean(adapter_proj, local_valid)
            context_mean = _masked_mean(adapter_proj, context_valid)
            gates.append(self.gate_head(torch.cat([local_mean, context_mean], dim=-1)).squeeze(-1))
        return torch.sigmoid(torch.stack(gates, dim=1))

    def forward(
        self,
        mmwave_tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_logits: bool = False,
        return_aux: bool = False,
        force_gate: float | torch.Tensor | None = None,
    ):
        bsz = int(mmwave_tokens.shape[0])
        base_valid = self._base_valid_mask(mmwave_tokens, mask)
        context_mask, local_mask = self._token_masks(base_valid)

        base_context = torch.no_grad() if self.freeze_base else torch.enable_grad()
        with base_context:
            base_z_e, base_logits = self.base_translator(mmwave_tokens, mask=base_valid, return_logits=True)
            ctx_z_e, ctx_logits = self.base_translator(mmwave_tokens, mask=context_mask, return_logits=True)
        if self.freeze_base:
            base_z_e = base_z_e.detach()
            base_logits = base_logits.detach()
            ctx_z_e = ctx_z_e.detach()
            ctx_logits = ctx_logits.detach()

        adapter_proj = self.adapter_mmwave_proj(mmwave_tokens)
        local_summary = self._pool_by_mask(adapter_proj, local_mask)
        ctx_summary = self.adapter_ctx_proj(ctx_z_e)
        hidden = self.adapter_mlp(torch.cat([ctx_summary, local_summary, ctx_summary * local_summary], dim=-1))
        adapter_z_delta = self.adapter_z_delta(hidden)
        adapter_logit_delta = self.adapter_logit_delta(hidden)

        gate_part = self._part_gate(adapter_proj, base_valid)
        if force_gate is not None:
            gate_part = _broadcast_force_gate(
                force_gate,
                gate_part=gate_part,
                num_parts=self.num_parts,
                batch_size=bsz,
            )
        token_part = self.token_part_ids.to(device=mmwave_tokens.device)
        gate_token = gate_part.index_select(1, token_part).unsqueeze(-1)

        base_z_delta = base_z_e - ctx_z_e
        base_logit_delta = base_logits - ctx_logits
        evidence_z_delta = base_z_delta + adapter_z_delta
        evidence_logit_delta = base_logit_delta + adapter_logit_delta
        pred_z_e = ctx_z_e + gate_token * evidence_z_delta
        pred_logits = ctx_logits + gate_token * evidence_logit_delta
        if not return_logits and not return_aux:
            return pred_z_e

        aux = {
            "base_z_e": base_z_e,
            "base_logits": base_logits,
            "ctx_z_e": ctx_z_e,
            "ctx_logits": ctx_logits,
            "base_z_delta": base_z_delta,
            "base_logit_delta": base_logit_delta,
            "adapter_z_delta": adapter_z_delta,
            "adapter_logit_delta": adapter_logit_delta,
            "evidence_z_delta": evidence_z_delta,
            "evidence_residual_logits": evidence_logit_delta,
            "gate_part": gate_part,
            "gate_token": gate_token.squeeze(-1),
            "context_attention_mask": context_mask,
            "local_attention_mask": local_mask,
        }
        if return_aux:
            return pred_z_e, pred_logits, aux
        return pred_z_e, pred_logits

    def _part_mask_1d(self, part: int | str, *, device: torch.device) -> torch.Tensor:
        if isinstance(part, str):
            if part not in self.layout.part_names:
                raise KeyError(f"Unknown part name: {part}")
            part_id = self.layout.part_names.index(part)
        else:
            part_id = int(part)
        if part_id < 0 or part_id >= self.num_parts:
            raise ValueError(f"part id out of range: {part_id}")
        return self.part_mmwave_mask.to(device=device)[part_id]

    def replace_part_evidence(
        self,
        mmwave_tokens: torch.Tensor,
        part: int | str,
        *,
        strategy: str = "zero",
        noise_std: float = 0.05,
        dropout_prob: float = 1.0,
    ) -> torch.Tensor:
        return ContextEvidenceResidualGatedTranslator.replace_part_evidence(
            self,
            mmwave_tokens,
            part,
            strategy=strategy,
            noise_std=noise_std,
            dropout_prob=dropout_prob,
        )

    def corrupt_tokens(
        self,
        mmwave_tokens: torch.Tensor,
        *,
        strategy: str = "heavy_dropout",
        noise_std: float = 0.1,
        dropout_prob: float = 0.8,
    ) -> torch.Tensor:
        return ContextEvidenceResidualGatedTranslator.corrupt_tokens(
            self,
            mmwave_tokens,
            strategy=strategy,
            noise_std=noise_std,
            dropout_prob=dropout_prob,
        )


__all__ = [
    "ContextEvidenceOutput",
    "ContextEvidenceResidualGatedTranslator",
    "EvidenceGroundedAdapterTranslator",
    "part_local_delta_l2",
    "part_local_js_divergence",
    "part_specificity_from_logits",
]
