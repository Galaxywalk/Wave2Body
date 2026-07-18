# -*- coding: utf-8 -*-
"""
Cross-Attention Token Translator

This module implements a cross-attention based architecture to translate
mmWave tokens (from Point Transformer) to pose tokens (for VQ-VAE).

Architecture:
    mmwave_tokens -> Token Alignment -> Cross-Attention Layers -> pose_tokens

Reference designs:
    - Perceiver IO: Cross-attention between latent queries and input tokens
    - Flamingo: Gated cross-attention for multimodal fusion
    - CLIP: Contrastive learning with cross-attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RMSNorm(nn.Module):
    """RMSNorm (no bias), stable for deeper networks."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms
        return x * self.weight


def _attention_masks(
    mask: torch.Tensor,
    *,
    heads: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a 0/1 or bool valid-token mask into attention masks.

    The public mask convention is: nonzero/True means valid and zero/False means
    masked. Rows with no valid keys are handled as zero attention output instead
    of producing NaNs from softmax over all -inf logits.
    """
    valid = mask.to(device=device)
    if valid.dtype != torch.bool:
        valid = valid != 0
    if valid.dim() == 3:
        valid = valid.unsqueeze(1)
    elif valid.dim() != 4:
        raise ValueError(f"Attention mask must be [B,Q,K] or [B,H,Q,K], got {tuple(mask.shape)}")
    if valid.shape[1] == 1:
        valid = valid.expand(-1, heads, -1, -1)
    elif valid.shape[1] != heads:
        raise ValueError(f"Attention mask head dimension must be 1 or {heads}, got {valid.shape[1]}")

    all_invalid = ~valid.any(dim=-1, keepdim=True)
    safe_valid = torch.where(all_invalid, torch.ones_like(valid), valid)
    additive = torch.zeros(safe_valid.shape, device=device, dtype=dtype)
    additive = additive.masked_fill(~safe_valid, float("-inf"))
    return safe_valid, additive, all_invalid


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer for token translation.

    Uses mmWave tokens as keys/values and pose tokens as queries.
    This allows pose tokens to attend to relevant mmWave features.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_rms_norm: bool = True,
        use_sdpa: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.use_sdpa = bool(use_sdpa) and hasattr(F, "scaled_dot_product_attention")

        inner_dim = heads * dim_head

        # Cross-attention: pose tokens query mmWave tokens
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        # MLP
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

        # Normalization
        if use_rms_norm:
            self.norm1 = RMSNorm(dim)
            self.norm2 = RMSNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        pose_tokens: torch.Tensor,
        mmwave_tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            pose_tokens: Pose token queries [B, N_pose, dim]
            mmwave_tokens: mmWave token keys/values [B, N_mmwave, dim]
            mask: Optional attention mask [B, N_pose, N_mmwave]

        Returns:
            Updated pose tokens [B, N_pose, dim]
        """
        # Cross-attention: pose tokens attend to mmWave tokens
        q = self.to_q(self.norm1(pose_tokens))
        k = self.to_k(mmwave_tokens)
        v = self.to_v(mmwave_tokens)

        # Reshape for multi-head attention
        B, N_pose, _ = pose_tokens.shape
        _, N_mmwave, _ = mmwave_tokens.shape

        q = q.view(B, N_pose, self.heads, -1).transpose(1, 2)  # [B, heads, N_pose, dim_head]
        k = k.view(B, N_mmwave, self.heads, -1).transpose(1, 2)  # [B, heads, N_mmwave, dim_head]
        v = v.view(B, N_mmwave, self.heads, -1).transpose(1, 2)  # [B, heads, N_mmwave, dim_head]

        # Attention
        if self.use_sdpa:
            attn_mask = None
            all_invalid = None
            if mask is not None:
                _, attn_mask, all_invalid = _attention_masks(
                    mask,
                    heads=self.heads,
                    device=q.device,
                    dtype=q.dtype,
                )
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )
            if all_invalid is not None:
                out = torch.where(all_invalid, torch.zeros_like(out), out)
        else:
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, heads, N_pose, N_mmwave]

            # Apply mask if provided
            all_invalid = None
            if mask is not None:
                valid_mask, _, all_invalid = _attention_masks(
                    mask,
                    heads=self.heads,
                    device=q.device,
                    dtype=q.dtype,
                )
                attn = attn.masked_fill(~valid_mask, float('-inf'))

            attn = F.softmax(attn, dim=-1)
            if all_invalid is not None:
                attn = torch.where(all_invalid, torch.zeros_like(attn), attn)
            attn = self.dropout(attn)

            # Apply attention to values
            out = torch.matmul(attn, v)  # [B, heads, N_pose, dim_head]
        out = out.transpose(1, 2).contiguous().view(B, N_pose, -1)  # [B, N_pose, heads*dim_head]

        # Output projection with residual
        pose_tokens = pose_tokens + self.to_out(out)

        # MLP with residual
        pose_tokens = pose_tokens + self.mlp(self.norm2(pose_tokens))

        return pose_tokens


class SelfAttentionLayer(nn.Module):
    """
    Self-attention layer for pose tokens.
    Allows pose tokens to interact with each other.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_rms_norm: bool = True,
        use_sdpa: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.use_sdpa = bool(use_sdpa) and hasattr(F, "scaled_dot_product_attention")

        inner_dim = heads * dim_head

        # Self-attention
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        # MLP
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

        # Normalization
        if use_rms_norm:
            self.norm1 = RMSNorm(dim)
            self.norm2 = RMSNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tokens [B, N, dim]
            mask: Optional attention mask [B, N, N]

        Returns:
            Updated tokens [B, N, dim]
        """
        B, N, _ = x.shape

        # Self-attention
        qkv = self.to_qkv(self.norm1(x)).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, N, self.heads, -1).transpose(1, 2), qkv)

        # Attention
        if self.use_sdpa:
            attn_mask = None
            all_invalid = None
            if mask is not None:
                _, attn_mask, all_invalid = _attention_masks(
                    mask,
                    heads=self.heads,
                    device=q.device,
                    dtype=q.dtype,
                )
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )
            if all_invalid is not None:
                out = torch.where(all_invalid, torch.zeros_like(out), out)
        else:
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # Apply mask if provided
            all_invalid = None
            if mask is not None:
                valid_mask, _, all_invalid = _attention_masks(
                    mask,
                    heads=self.heads,
                    device=q.device,
                    dtype=q.dtype,
                )
                attn = attn.masked_fill(~valid_mask, float('-inf'))

            attn = F.softmax(attn, dim=-1)
            if all_invalid is not None:
                attn = torch.where(all_invalid, torch.zeros_like(attn), attn)
            attn = self.dropout(attn)

            # Apply attention to values
            out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)

        # Output projection with residual
        x = x + self.to_out(out)

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x


class CrossAttentionTokenTranslator(nn.Module):
    """
    Cross-Attention Token Translator

    Translates mmWave tokens (from Point Transformer) to pose tokens (for VQ-VAE).

    Architecture:
        1. Token Alignment: Project mmWave tokens and initialize pose tokens
        2. Cross-Attention Layers: Pose tokens attend to mmWave tokens
        3. Self-Attention Layers: Pose tokens interact with each other
        4. Output Projection: Final pose tokens ready for VQ-VAE

    Args:
        mmwave_token_dim: Dimension of mmWave tokens (from Point Transformer feat_dim)
        pose_token_dim: Dimension of pose tokens (VQ-VAE token_dim)
        num_pose_tokens: Number of pose tokens (VQ-VAE token_num)
        num_mmwave_tokens: Number of mmWave tokens (typically num_joints)
        hidden_dim: Hidden dimension for transformer layers
        num_cross_attn_layers: Number of cross-attention layers
        num_self_attn_layers: Number of self-attention layers
        heads: Number of attention heads
        dim_head: Dimension per attention head
        mlp_ratio: MLP expansion ratio
        dropout: Dropout rate
        use_rms_norm: Whether to use RMSNorm instead of LayerNorm
    """

    def __init__(
        self,
        mmwave_token_dim: int = 32,  # Point Transformer feat_dim
        pose_token_dim: int = 256,   # VQ-VAE token_dim
        num_pose_tokens: int = 64,   # VQ-VAE token_num
        num_mmwave_tokens: int = 22, # num_joints
        hidden_dim: int = 256,
        num_cross_attn_layers: int = 4,
        num_self_attn_layers: int = 2,
        heads: int = 8,
        dim_head: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_rms_norm: bool = True,
        use_sdpa: bool = False,
        codebook_size: Optional[int] = None,  # For token classification head
    ):
        super().__init__()

        self.mmwave_token_dim = mmwave_token_dim
        self.pose_token_dim = pose_token_dim
        self.num_pose_tokens = num_pose_tokens
        self.num_mmwave_tokens = num_mmwave_tokens
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size

        # Token Alignment: Project mmWave tokens to hidden dimension
        self.mmwave_proj = nn.Linear(mmwave_token_dim, hidden_dim)

        # Initialize pose tokens (learnable queries)
        self.pose_token_embed = nn.Parameter(
            torch.randn(1, num_pose_tokens, hidden_dim)
        )
        nn.init.trunc_normal_(self.pose_token_embed, std=0.02)

        # Positional embeddings (optional, for sequence order)
        self.pose_pos_embed = nn.Parameter(
            torch.randn(1, num_pose_tokens, hidden_dim)
        )
        nn.init.trunc_normal_(self.pose_pos_embed, std=0.02)

        # Cross-attention layers: pose tokens attend to mmWave tokens
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(
                dim=hidden_dim,
                heads=heads,
                dim_head=dim_head,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_rms_norm=use_rms_norm,
                use_sdpa=use_sdpa,
            )
            for _ in range(num_cross_attn_layers)
        ])

        # Self-attention layers: pose tokens interact with each other
        self.self_attn_layers = nn.ModuleList([
            SelfAttentionLayer(
                dim=hidden_dim,
                heads=heads,
                dim_head=dim_head,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_rms_norm=use_rms_norm,
                use_sdpa=use_sdpa,
            )
            for _ in range(num_self_attn_layers)
        ])

        # Output projection: hidden_dim -> pose_token_dim
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim) if not use_rms_norm else RMSNorm(hidden_dim),
            nn.Linear(hidden_dim, pose_token_dim),
            nn.Dropout(dropout)
        )

        # Token classification head (for L_token loss)
        # Maps pose tokens to codebook logits for direct token supervision
        if codebook_size is not None:
            self.token_classifier = nn.Sequential(
                nn.LayerNorm(pose_token_dim) if not use_rms_norm else RMSNorm(pose_token_dim),
                nn.Linear(pose_token_dim, codebook_size)
            )
        else:
            self.token_classifier = None

        # Dropout for input
        self.input_dropout = nn.Dropout(dropout)

    def forward(
        self,
        mmwave_tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_logits: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            mmwave_tokens: mmWave tokens from Point Transformer [B, num_joints, mmwave_token_dim]
            mask: Optional mask for mmWave tokens [B, num_joints]
            return_logits: If True and token_classifier exists, also return token logits

        Returns:
            If return_logits=False:
                pose_tokens: Pose tokens ready for VQ-VAE [B, num_pose_tokens, pose_token_dim]
            If return_logits=True and token_classifier exists:
                tuple of (pose_tokens, logits) where logits is [B, num_pose_tokens, codebook_size]
        """
        B = mmwave_tokens.shape[0]

        # Project mmWave tokens to hidden dimension
        mmwave_proj = self.mmwave_proj(mmwave_tokens)  # [B, num_joints, hidden_dim]
        mmwave_proj = self.input_dropout(mmwave_proj)

        # Initialize pose tokens (learnable queries)
        pose_tokens = self.pose_token_embed.expand(B, -1, -1)  # [B, num_pose_tokens, hidden_dim]
        pose_tokens = pose_tokens + self.pose_pos_embed.expand(B, -1, -1)

        # Expand mask for cross-attention if provided. A 2-D mask applies the
        # same valid mmWave tokens to all pose queries; a 3-D mask allows
        # per-query context/evidence isolation.
        cross_attn_mask = None
        if mask is not None:
            # Public convention: nonzero/True means valid, zero/False means masked.
            if mask.dim() == 2:
                # mask: [B, num_joints] -> [B, num_pose_tokens, num_joints].
                cross_attn_mask = mask.unsqueeze(1).expand(-1, self.num_pose_tokens, -1)
            elif mask.dim() == 3:
                if int(mask.shape[1]) != int(self.num_pose_tokens):
                    raise ValueError(
                        f"3-D mask query dimension must be {self.num_pose_tokens}, got {int(mask.shape[1])}"
                    )
                cross_attn_mask = mask
            else:
                raise ValueError(f"mask must be [B,J] or [B,T,J], got {tuple(mask.shape)}")

        # Cross-attention: pose tokens attend to mmWave tokens
        for layer in self.cross_attn_layers:
            pose_tokens = layer(pose_tokens, mmwave_proj, cross_attn_mask)

        # Self-attention: pose tokens interact with each other
        for layer in self.self_attn_layers:
            pose_tokens = layer(pose_tokens)

        # Output projection
        pose_tokens = self.output_proj(pose_tokens)  # [B, num_pose_tokens, pose_token_dim]

        # Optionally compute token classification logits for L_token loss
        if return_logits and self.token_classifier is not None:
            logits = self.token_classifier(pose_tokens)  # [B, num_pose_tokens, codebook_size]
            return pose_tokens, logits

        return pose_tokens

    def get_attention_maps(
        self,
        mmwave_tokens: torch.Tensor,
        layer_idx: int = -1
    ) -> torch.Tensor:
        """
        Get attention maps from a specific cross-attention layer.

        Args:
            mmwave_tokens: mmWave tokens [B, num_joints, mmwave_token_dim]
            layer_idx: Layer index (negative for counting from end)

        Returns:
            Attention maps [B, heads, num_pose_tokens, num_joints]
        """
        self.eval()
        B = mmwave_tokens.shape[0]

        # Project mmWave tokens
        mmwave_proj = self.mmwave_proj(mmwave_tokens)

        # Initialize pose tokens
        pose_tokens = self.pose_token_embed.expand(B, -1, -1)
        pose_tokens = pose_tokens + self.pose_pos_embed.expand(B, -1, -1)

        # Forward through layers up to layer_idx
        for i, layer in enumerate(self.cross_attn_layers):
            if i == len(self.cross_attn_layers) + layer_idx:
                # Extract attention from this layer
                q = layer.to_q(layer.norm1(pose_tokens))
                k = layer.to_k(mmwave_proj)

                q = q.view(B, self.num_pose_tokens, layer.heads, -1).transpose(1, 2)
                k = k.view(B, self.num_mmwave_tokens, layer.heads, -1).transpose(1, 2)

                attn = torch.matmul(q, k.transpose(-2, -1)) * layer.scale
                attn = F.softmax(attn, dim=-1)

                return attn

            pose_tokens = layer(pose_tokens, mmwave_proj)

        return None
