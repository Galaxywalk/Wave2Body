# -*- coding: utf-8 -*-
"""
Part-level VQ-VAE for single-frame joints.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .part_schema import (
    DEFAULT_PART_JOINTS,
    DEFAULT_PART_NAMES,
    DEFAULT_PART_TOKEN_NUM,
    PartLayout,
    create_part_layout,
)


def _as_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ('1', 'true', 'yes', 'y', 'on'):
            return True
        if lowered in ('0', 'false', 'no', 'n', 'off'):
            return False
    return bool(value)


class PartMLPEncoder(nn.Module):
    """Encode one body part from joint xyz to part tokens."""

    def __init__(
        self,
        input_dim: int,
        token_num: int,
        token_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_num = token_num
        self.token_dim = token_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_num * token_dim),
        )

    def forward(self, part_joints: torch.Tensor) -> torch.Tensor:
        # part_joints: [B, J_part, 3]
        bsz = part_joints.shape[0]
        flat = part_joints.reshape(bsz, -1)
        out = self.net(flat)
        return out.view(bsz, self.token_num, self.token_dim)


class PartMLPDecoder(nn.Module):
    """Decode one body part from part tokens back to joint xyz."""

    def __init__(
        self,
        token_num: int,
        token_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_num = token_num
        self.token_dim = token_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(token_num * token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, part_tokens: torch.Tensor) -> torch.Tensor:
        # part_tokens: [B, T_part, D]
        bsz = part_tokens.shape[0]
        flat = part_tokens.reshape(bsz, -1)
        out = self.net(flat)
        if self.output_dim % 3 != 0:
            raise ValueError(f"Decoder output_dim must be divisible by 3, got {self.output_dim}")
        return out.view(bsz, self.output_dim // 3, 3)


class PartEMAVectorQuantizer(nn.Module):
    """
    EMA-based vector quantizer with optional stale-code reset.
    """

    def __init__(
        self,
        num_codes: int,
        code_dim: int,
        quantizer_type: str = 'ema_reset',
        ema_decay: float = 0.99,
        reset_threshold: float = 1.0,
        reset_noise_std: float = 1e-2,
        reset_min_active_frac: float = 0.0,
        init_from_data: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        if quantizer_type not in ('ema', 'ema_reset'):
            raise ValueError(f"Unsupported quantizer_type: {quantizer_type}")
        self.num_codes = int(num_codes)
        self.code_dim = int(code_dim)
        self.quantizer_type = quantizer_type
        self.ema_decay = float(ema_decay)
        self.reset_threshold = float(reset_threshold)
        self.reset_noise_std = float(reset_noise_std)
        self.reset_min_active_frac = float(reset_min_active_frac)
        self.init_from_data = bool(init_from_data)
        self.eps = float(eps)

        self.register_buffer('codebook', torch.empty(self.num_codes, self.code_dim))
        self.register_buffer('ema_weight', torch.empty(self.num_codes, self.code_dim))
        self.register_buffer('ema_count_raw', torch.zeros(self.num_codes))
        self.register_buffer('initialized', torch.zeros(1, dtype=torch.bool))

        self.codebook.data.normal_()
        self.ema_weight.data.copy_(self.codebook.data)
        self.ema_count_raw.data.fill_(1.0)

        self.last_stats: Dict[str, float] = {}

    def _tile_with_noise(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, D]
        if x.numel() == 0:
            return torch.randn(self.num_codes, self.code_dim, device=self.codebook.device, dtype=self.codebook.dtype) * 0.01
        n = x.shape[0]
        if n >= self.num_codes:
            return x
        repeats = (self.num_codes + n - 1) // n
        out = x.repeat(repeats, 1)
        noise_std = 0.01 / (self.code_dim ** 0.5)
        out = out + torch.randn_like(out) * noise_std
        return out

    @torch.no_grad()
    def init_codebook_from_data(self, x: torch.Tensor):
        tiled = self._tile_with_noise(x)
        init_codes = tiled[:self.num_codes]
        self.codebook.copy_(init_codes)
        self.ema_weight.copy_(init_codes)
        self.ema_count_raw.fill_(1.0)
        self.initialized.fill_(True)

    @torch.no_grad()
    def _ensure_initialized(self, x: torch.Tensor):
        if bool(self.initialized.item()):
            return
        if self.init_from_data:
            self.init_codebook_from_data(x)
        else:
            self.ema_weight.copy_(self.codebook)
            self.ema_count_raw.fill_(1.0)
            self.initialized.fill_(True)

    def quantize(self, flat_latents: torch.Tensor) -> torch.Tensor:
        # flat_latents: [N, D]
        distances = (
            torch.sum(flat_latents ** 2, dim=1, keepdim=True)
            + torch.sum(self.codebook ** 2, dim=1)
            - 2 * torch.matmul(flat_latents, self.codebook.t())
        )
        return torch.argmin(distances, dim=1)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.codebook)

    def _compute_perplexity(self, counts: torch.Tensor) -> torch.Tensor:
        probs = counts / torch.clamp(counts.sum(), min=1.0)
        return torch.exp(-torch.sum(probs * torch.log(probs + 1e-7)))

    @torch.no_grad()
    def _sample_reset_codes(self, x: torch.Tensor, n_codes: int) -> torch.Tensor:
        tiled = self._tile_with_noise(x.detach())
        sampled = tiled[:n_codes]
        if self.reset_noise_std > 0:
            sampled = sampled + torch.randn_like(sampled) * self.reset_noise_std
        return sampled

    @torch.no_grad()
    def _update_codebook(self, flat_latents: torch.Tensor, indices: torch.Tensor) -> Dict[str, float]:
        flat_latents = flat_latents.to(dtype=self.codebook.dtype)
        counts = torch.bincount(indices, minlength=self.num_codes).to(flat_latents.dtype)
        sums = torch.zeros(self.num_codes, self.code_dim, device=flat_latents.device, dtype=flat_latents.dtype)
        sums.index_add_(0, indices, flat_latents)

        self.ema_count_raw.mul_(self.ema_decay).add_(counts, alpha=1.0 - self.ema_decay)
        self.ema_weight.mul_(self.ema_decay).add_(sums, alpha=1.0 - self.ema_decay)

        denom = torch.clamp(self.ema_count_raw, min=self.eps).unsqueeze(1)
        self.codebook.copy_(self.ema_weight / denom)

        usage_mask = self.ema_count_raw >= self.reset_threshold
        active_frac = usage_mask.float().mean()
        usage_ratio_pre = float(active_frac.item())
        active_code_count_pre = float(usage_mask.sum().item())
        reset_count = 0

        if self.quantizer_type == 'ema_reset':
            allow_reset = True
            if self.reset_min_active_frac > 0 and float(active_frac.item()) < self.reset_min_active_frac:
                allow_reset = False

            stale_mask = ~usage_mask
            stale_idx = torch.nonzero(stale_mask, as_tuple=False).view(-1)
            if allow_reset and stale_idx.numel() > 0:
                refreshed = self._sample_reset_codes(flat_latents, int(stale_idx.numel())).to(self.codebook.dtype)
                self.codebook[stale_idx] = refreshed
                self.ema_weight[stale_idx] = refreshed
                self.ema_count_raw[stale_idx] = 1.0
                reset_count = int(stale_idx.numel())

        perplexity = self._compute_perplexity(counts)
        stats = {
            'perplexity': float(perplexity.item()),
            'usage_ratio': usage_ratio_pre,
            'active_code_count': active_code_count_pre,
            'reset_count': float(reset_count),
            'reset_rate': float(reset_count / max(1, self.num_codes)),
        }
        self.last_stats = stats
        return stats

    @torch.no_grad()
    def _eval_stats(self, indices: torch.Tensor) -> Dict[str, float]:
        counts = torch.bincount(indices, minlength=self.num_codes).to(self.codebook.dtype)
        perplexity = self._compute_perplexity(counts)
        stats = {
            'perplexity': float(perplexity.item()),
            'usage_ratio': float((self.ema_count_raw >= self.reset_threshold).float().mean().item()),
            'active_code_count': float((self.ema_count_raw >= self.reset_threshold).sum().item()),
            'reset_count': 0.0,
            'reset_rate': 0.0,
        }
        self.last_stats = stats
        return stats

    def forward(self, z_e: torch.Tensor, train: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
        # z_e: [B, N, D]
        bsz, n_tokens, token_dim = z_e.shape
        if token_dim != self.code_dim:
            raise ValueError(f"Token dim mismatch, expected {self.code_dim}, got {token_dim}")

        flat = z_e.reshape(-1, token_dim)
        if train:
            self._ensure_initialized(flat.detach())

        indices = self.quantize(flat)
        z_q = self.dequantize(indices).view(bsz, n_tokens, token_dim)

        if train:
            stats = self._update_codebook(flat.detach(), indices)
            commit_loss = F.mse_loss(z_e, z_q.detach())
            z_q = z_e + (z_q - z_e).detach()
        else:
            stats = self._eval_stats(indices)
            commit_loss = None

        return indices.view(bsz, n_tokens), z_q, commit_loss, stats


class PartJointTokenizer(nn.Module):
    """
    Part-level joint tokenizer with per-part encoders/decoders and codebooks.
    """

    def __init__(
        self,
        num_joints: int = 22,
        input_dim: int = 3,
        token_dim: int = 256,
        part_hidden_dim: int = 512,
        part_dropout: float = 0.1,
        quantizer_type: str = 'ema_reset',
        ema_decay: float = 0.99,
        reset_threshold: float = 1.0,
        reset_noise_std: float = 1e-2,
        reset_min_active_frac: float = 0.0,
        init_from_data: bool = True,
        root_index: int = 0,
        part_names: Tuple[str, ...] = DEFAULT_PART_NAMES,
        part_joints: Mapping[str, Tuple[int, ...]] = DEFAULT_PART_JOINTS,
        part_token_num: Union[int, Mapping[str, int]] = DEFAULT_PART_TOKEN_NUM,
        part_codebook_size: Union[int, Mapping[str, int]] = 64,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.input_dim = input_dim
        self.token_dim = token_dim
        self.drop_rate = drop_rate
        self.quantizer_type = quantizer_type

        self.layout: PartLayout = create_part_layout(
            num_joints=num_joints,
            root_index=root_index,
            part_names=part_names,
            part_joints=part_joints,
            part_token_num=part_token_num,
            part_codebook_size=part_codebook_size,
        )

        # Keep compatibility with existing usage.
        self.token_num = self.layout.total_tokens
        self.token_class_num = self.layout.total_codebook_size

        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        self.quantizers = nn.ModuleDict()

        for name in self.layout.part_names:
            n_joint = len(self.layout.part_joints[name])
            n_token = self.layout.part_token_num[name]
            n_code = self.layout.part_codebook_size[name]
            in_dim = n_joint * input_dim

            self.encoders[name] = PartMLPEncoder(
                input_dim=in_dim,
                token_num=n_token,
                token_dim=token_dim,
                hidden_dim=part_hidden_dim,
                dropout=part_dropout,
            )
            self.decoders[name] = PartMLPDecoder(
                token_num=n_token,
                token_dim=token_dim,
                output_dim=in_dim,
                hidden_dim=part_hidden_dim,
                dropout=part_dropout,
            )
            self.quantizers[name] = PartEMAVectorQuantizer(
                num_codes=n_code,
                code_dim=token_dim,
                quantizer_type=quantizer_type,
                ema_decay=ema_decay,
                reset_threshold=reset_threshold,
                reset_noise_std=reset_noise_std,
                reset_min_active_frac=reset_min_active_frac,
                init_from_data=init_from_data,
            )

        self.last_quantizer_stats: Dict[str, float] = {}

    @property
    def codebook(self) -> torch.Tensor:
        return torch.cat([self.quantizers[name].codebook for name in self.layout.part_names], dim=0)

    def _apply_joint_visibility(
        self,
        joints: torch.Tensor,
        joints_visible: Optional[torch.Tensor],
        train: bool,
    ) -> torch.Tensor:
        if joints_visible is None:
            return joints

        vis = joints_visible
        if train and self.drop_rate > 0:
            rand_keep = torch.rand_like(vis.float()) > self.drop_rate
            vis = torch.logical_and(vis.bool(), rand_keep.bool())
        vis = vis.unsqueeze(-1).to(joints.dtype)
        return joints * vis

    def _quantizer_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for name in self.layout.part_names:
            stats = self.quantizers[name].last_stats
            for key, value in stats.items():
                metrics[f"{key}_{name}"] = float(value)
        return metrics

    def get_quantizer_metrics(self) -> Dict[str, float]:
        return dict(self.last_quantizer_stats)

    def encode(
        self,
        joints: torch.Tensor,
        joints_visible: Optional[torch.Tensor] = None,
        train: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        joints_masked = self._apply_joint_visibility(joints, joints_visible, train=train)
        part_inputs = self.layout.split_joints(joints_masked)

        part_tokens_q: Dict[str, torch.Tensor] = {}
        part_indices_local: Dict[str, torch.Tensor] = {}
        commit_losses = []

        for name in self.layout.part_names:
            z_e = self.encoders[name](part_inputs[name])
            idx_local, z_q, commit_loss, _stats = self.quantizers[name](z_e, train=train)
            part_tokens_q[name] = z_q
            part_indices_local[name] = idx_local
            if commit_loss is not None:
                commit_losses.append(commit_loss)

        quantized_features = self.layout.concat_part_tokens(part_tokens_q)
        encoding_indices = self.layout.merge_local_indices_to_global(part_indices_local)
        e_latent_loss = torch.stack(commit_losses).mean() if commit_losses else None
        self.last_quantizer_stats = self._quantizer_metrics()
        return encoding_indices, quantized_features, e_latent_loss

    def encode_with_intermediates(
        self,
        joints: torch.Tensor,
        joints_visible: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        joints_masked = self._apply_joint_visibility(joints, joints_visible, train=False)
        part_inputs = self.layout.split_joints(joints_masked)
        bsz = joints.shape[0]

        part_z_e: Dict[str, torch.Tensor] = {}
        part_z_q: Dict[str, torch.Tensor] = {}
        part_idx_local: Dict[str, torch.Tensor] = {}

        for name in self.layout.part_names:
            z_e = self.encoders[name](part_inputs[name])
            flat = z_e.reshape(-1, self.token_dim)
            idx_local = self.quantizers[name].quantize(flat).view(bsz, -1)
            z_q = self.quantizers[name].dequantize(idx_local.reshape(-1)).view(bsz, -1, self.token_dim)

            part_z_e[name] = z_e
            part_z_q[name] = z_q
            part_idx_local[name] = idx_local

        z_e_cat = self.layout.concat_part_tokens(part_z_e)
        z_q_cat = self.layout.concat_part_tokens(part_z_q)
        idx_global = self.layout.merge_local_indices_to_global(part_idx_local)
        idx_local_cat = torch.cat([part_idx_local[name] for name in self.layout.part_names], dim=1)
        part_ids = self.layout.build_part_ids(joints.device)

        return {
            'z_e': z_e_cat,
            'z_q': z_q_cat,
            'indices': idx_global,
            'indices_local': idx_local_cat,
            'part_ids': part_ids,
        }

    def quantize_tokens(
        self,
        continuous_tokens: torch.Tensor,
        apply_straight_through: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, n_tokens, token_dim = continuous_tokens.shape
        if n_tokens != self.token_num:
            raise ValueError(f"Expected token_num={self.token_num}, got {n_tokens}")
        if token_dim != self.token_dim:
            raise ValueError(f"Expected token_dim={self.token_dim}, got {token_dim}")

        part_tokens = self.layout.split_part_tokens(continuous_tokens)
        part_tokens_q: Dict[str, torch.Tensor] = {}
        part_idx_local: Dict[str, torch.Tensor] = {}

        for name in self.layout.part_names:
            x = part_tokens[name]
            flat = x.reshape(-1, self.token_dim)
            idx_local = self.quantizers[name].quantize(flat).view(bsz, -1)
            z_q = self.quantizers[name].dequantize(idx_local.reshape(-1)).view(bsz, -1, self.token_dim)
            if apply_straight_through:
                z_q = x + (z_q - x).detach()
            part_tokens_q[name] = z_q
            part_idx_local[name] = idx_local

        tokens_q = self.layout.concat_part_tokens(part_tokens_q)
        idx_global = self.layout.merge_local_indices_to_global(part_idx_local)
        return tokens_q, idx_global

    def decode(self, quantized_features: torch.Tensor) -> torch.Tensor:
        if quantized_features.dim() != 3:
            raise ValueError(f"Expected quantized features [B, N, D], got {tuple(quantized_features.shape)}")
        if quantized_features.shape[1] != self.token_num:
            raise ValueError(f"Expected token_num={self.token_num}, got {quantized_features.shape[1]}")

        part_tokens = self.layout.split_part_tokens(quantized_features)
        part_outputs: Dict[str, torch.Tensor] = {}
        for name in self.layout.part_names:
            part_outputs[name] = self.decoders[name](part_tokens[name])
        return self.layout.merge_joints(part_outputs, force_root_zero=True)

    def forward(
        self,
        joints: torch.Tensor,
        joints_visible: Optional[torch.Tensor] = None,
        cls_logits: Optional[torch.Tensor] = None,
        train: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if cls_logits is not None:
            if cls_logits.dim() == 2:
                if cls_logits.shape[0] % self.token_num != 0:
                    raise ValueError(f"Cannot infer batch from cls_logits shape {tuple(cls_logits.shape)}")
                bsz = cls_logits.shape[0] // self.token_num
                cls_logits = cls_logits.view(bsz, self.token_num, -1)
            if cls_logits.dim() != 3:
                raise ValueError(f"Expected cls_logits [B, N, C], got {tuple(cls_logits.shape)}")
            if cls_logits.shape[1] != self.token_num:
                raise ValueError(f"Expected {self.token_num} tokens in cls_logits, got {cls_logits.shape[1]}")
            if cls_logits.shape[2] != self.token_class_num:
                raise ValueError(f"Expected codebook logits dim {self.token_class_num}, got {cls_logits.shape[2]}")

            quantized_features = torch.matmul(cls_logits, self.codebook)
            encoding_indices = torch.argmax(cls_logits, dim=-1)
            e_latent_loss = None
        else:
            encoding_indices, quantized_features, e_latent_loss = self.encode(
                joints, joints_visible=joints_visible, train=train
            )

        recovered_joints = self.decode(quantized_features)
        return recovered_joints, encoding_indices, e_latent_loss


def create_part_joint_tokenizer(config: Mapping[str, object]) -> PartJointTokenizer:
    return PartJointTokenizer(
        num_joints=int(config.get('num_joints', 22)),
        input_dim=int(config.get('input_dim', 3)),
        token_dim=int(config.get('token_dim', 256)),
        part_hidden_dim=int(config.get('part_hidden_dim', 512)),
        part_dropout=float(config.get('part_dropout', config.get('enc_dropout', 0.1))),
        quantizer_type=str(config.get('quantizer_type', config.get('quantizer', 'ema_reset'))),
        ema_decay=float(config.get('ema_decay', 0.99)),
        reset_threshold=float(config.get('reset_threshold', 1.0)),
        reset_noise_std=float(config.get('reset_noise_std', 1e-2)),
        reset_min_active_frac=float(config.get('reset_min_active_frac', 0.0)),
        init_from_data=_as_bool(config.get('init_from_data'), True),
        root_index=int(config.get('root_index', 0)),
        part_names=tuple(config.get('part_token_layout_order', DEFAULT_PART_NAMES)),
        part_joints=config.get('part_joints', DEFAULT_PART_JOINTS),
        part_token_num=config.get('part_token_num', DEFAULT_PART_TOKEN_NUM),
        part_codebook_size=config.get('part_codebook_size', config.get('token_class_num', 64)),
        drop_rate=float(config.get('drop_rate', 0.0)),
    )
