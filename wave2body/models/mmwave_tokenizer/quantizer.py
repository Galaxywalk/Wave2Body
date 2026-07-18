# -*- coding: utf-8 -*-
"""EMA vector quantizer for mmWave evidence tokens."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class EMAVectorQuantizer(nn.Module):
    """EMA VQ codebook with optional stale-code reset and masked tokens."""

    def __init__(
        self,
        num_codes: int,
        code_dim: int,
        *,
        ema_decay: float = 0.99,
        reset_threshold: float = 1.0,
        reset_noise_std: float = 1e-2,
        init_from_data: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.num_codes = int(num_codes)
        self.code_dim = int(code_dim)
        self.ema_decay = float(ema_decay)
        self.reset_threshold = float(reset_threshold)
        self.reset_noise_std = float(reset_noise_std)
        self.init_from_data = bool(init_from_data)
        self.eps = float(eps)

        self.register_buffer("codebook", torch.empty(self.num_codes, self.code_dim))
        self.register_buffer("ema_weight", torch.empty(self.num_codes, self.code_dim))
        self.register_buffer("ema_count_raw", torch.zeros(self.num_codes))
        self.register_buffer("initialized", torch.zeros(1, dtype=torch.bool))
        self.codebook.data.normal_()
        self.ema_weight.data.copy_(self.codebook.data)
        self.ema_count_raw.data.fill_(1.0)
        self.last_stats: dict[str, float] = {}

    def _tile_with_noise(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return torch.randn(self.num_codes, self.code_dim, device=self.codebook.device, dtype=self.codebook.dtype)
        if x.shape[0] >= self.num_codes:
            return x[: self.num_codes]
        repeats = (self.num_codes + x.shape[0] - 1) // x.shape[0]
        out = x.repeat(repeats, 1)[: self.num_codes]
        return out + torch.randn_like(out) * (0.01 / (self.code_dim ** 0.5))

    @torch.no_grad()
    def _ensure_initialized(self, x: torch.Tensor) -> None:
        if bool(self.initialized.item()):
            return
        if self.init_from_data and x.numel() > 0:
            init_codes = self._tile_with_noise(x.detach()).to(dtype=self.codebook.dtype)
            self.codebook.copy_(init_codes)
            self.ema_weight.copy_(init_codes)
        else:
            self.ema_weight.copy_(self.codebook)
        self.ema_count_raw.fill_(1.0)
        self.initialized.fill_(True)

    def distances(self, flat_latents: torch.Tensor) -> torch.Tensor:
        return (
            torch.sum(flat_latents ** 2, dim=1, keepdim=True)
            + torch.sum(self.codebook.float() ** 2, dim=1).unsqueeze(0)
            - 2 * torch.matmul(flat_latents.float(), self.codebook.float().t())
        )

    @torch.no_grad()
    def _update_codebook(self, flat_latents: torch.Tensor, indices: torch.Tensor) -> dict[str, float]:
        flat_latents = flat_latents.to(dtype=self.codebook.dtype)
        counts = torch.bincount(indices, minlength=self.num_codes).to(flat_latents.dtype)
        sums = torch.zeros(self.num_codes, self.code_dim, device=flat_latents.device, dtype=flat_latents.dtype)
        sums.index_add_(0, indices, flat_latents)

        self.ema_count_raw.mul_(self.ema_decay).add_(counts, alpha=1.0 - self.ema_decay)
        self.ema_weight.mul_(self.ema_decay).add_(sums, alpha=1.0 - self.ema_decay)
        self.codebook.copy_(self.ema_weight / torch.clamp(self.ema_count_raw, min=self.eps).unsqueeze(1))

        usage_mask = self.ema_count_raw >= self.reset_threshold
        stale_idx = torch.nonzero(~usage_mask, as_tuple=False).squeeze(-1)
        reset_count = 0
        if stale_idx.numel() > 0:
            refreshed = self._tile_with_noise(flat_latents.detach())[: stale_idx.numel()].to(dtype=self.codebook.dtype)
            if self.reset_noise_std > 0:
                refreshed = refreshed + torch.randn_like(refreshed) * self.reset_noise_std
            self.codebook[stale_idx] = refreshed
            self.ema_weight[stale_idx] = refreshed
            self.ema_count_raw[stale_idx] = 1.0
            reset_count = int(stale_idx.numel())

        probs = counts / torch.clamp(counts.sum(), min=1.0)
        perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-7)))
        stats = {
            "perplexity": float(perplexity.item()),
            "usage_ratio": float(usage_mask.float().mean().item()),
            "active_code_count": float(usage_mask.sum().item()),
            "reset_count": float(reset_count),
            "reset_rate": float(reset_count / max(1, self.num_codes)),
        }
        self.last_stats = stats
        return stats

    @torch.no_grad()
    def _eval_stats(self, indices: torch.Tensor) -> dict[str, float]:
        counts = torch.bincount(indices, minlength=self.num_codes).to(self.codebook.dtype)
        probs = counts / torch.clamp(counts.sum(), min=1.0)
        perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-7)))
        stats = {
            "perplexity": float(perplexity.item()),
            "usage_ratio": float((self.ema_count_raw >= self.reset_threshold).float().mean().item()),
            "active_code_count": float((self.ema_count_raw >= self.reset_threshold).sum().item()),
            "reset_count": 0.0,
            "reset_rate": 0.0,
        }
        self.last_stats = stats
        return stats

    def usage_kl_to_uniform(self, z_e: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        flat = z_e.reshape(-1, z_e.shape[-1])
        if token_mask is not None:
            flat = flat[token_mask.reshape(-1)]
        if flat.numel() == 0:
            return z_e.sum() * 0.0
        probs = F.softmax(-self.distances(flat), dim=-1)
        mean_probs = probs.mean(dim=0)
        return torch.sum(mean_probs * (torch.log(mean_probs + 1e-7) + torch.log(torch.tensor(float(self.num_codes), device=z_e.device))))

    def forward(
        self,
        z_e: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        *,
        train: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        if z_e.dim() != 3 or z_e.shape[-1] != self.code_dim:
            raise ValueError(f"Expected z_e [B,T,{self.code_dim}], got {tuple(z_e.shape)}")
        flat = z_e.reshape(-1, self.code_dim)
        valid = torch.ones(flat.shape[0], device=z_e.device, dtype=torch.bool)
        if token_mask is not None:
            valid = token_mask.reshape(-1).to(device=z_e.device, dtype=torch.bool)
        valid_flat = flat[valid]
        if train:
            self._ensure_initialized(valid_flat.detach())

        all_indices = torch.zeros(flat.shape[0], device=z_e.device, dtype=torch.long)
        z_q_flat = flat.clone()
        if valid_flat.numel() > 0:
            indices = torch.argmin(self.distances(valid_flat), dim=1)
            all_indices[valid] = indices
            z_q_valid = F.embedding(indices, self.codebook.float()).to(dtype=z_e.dtype)
            z_q_flat[valid] = z_q_valid
            stats = self._update_codebook(valid_flat.detach(), indices) if train else self._eval_stats(indices)
            commit_loss = F.mse_loss(valid_flat, z_q_valid.detach())
        else:
            stats = {"perplexity": 0.0, "usage_ratio": 0.0, "active_code_count": 0.0, "reset_count": 0.0, "reset_rate": 0.0}
            commit_loss = z_e.sum() * 0.0

        z_q = z_q_flat.view_as(z_e)
        if train:
            z_q = z_e + (z_q - z_e).detach()
        return all_indices.view(z_e.shape[0], z_e.shape[1]), z_q, commit_loss, stats

    def extra_repr(self) -> str:
        return f"num_codes={self.num_codes}, code_dim={self.code_dim}, ema_decay={self.ema_decay}"
