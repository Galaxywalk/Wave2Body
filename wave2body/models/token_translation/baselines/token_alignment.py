# -*- coding: utf-8 -*-
"""
Token Alignment Module

Provides utilities for aligning mmWave tokens and pose tokens,
including dimension matching and feature normalization.
"""

import torch
import torch.nn as nn


class TokenAlignmentModule(nn.Module):
    """
    Token alignment module for matching dimensions and normalizing features.

    This module can be used as a preprocessing step before cross-attention,
    or as a standalone alignment layer.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_layer_norm: bool = True,
        use_dropout: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.proj = nn.Linear(input_dim, output_dim)

        if use_layer_norm:
            self.norm = nn.LayerNorm(output_dim)
        else:
            self.norm = nn.Identity()

        if use_dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tokens [B, N, input_dim]

        Returns:
            Aligned tokens [B, N, output_dim]
        """
        x = self.proj(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class AdaptiveTokenPooling(nn.Module):
    """
    Adaptive token pooling for handling variable number of tokens.

    Can be used to pool mmWave tokens if their number doesn't match
    the expected number of pose tokens.
    """

    def __init__(
        self,
        input_dim: int,
        output_tokens: int,
        pooling_type: str = 'attention',  # 'attention', 'mean', 'max', 'learned'
    ):
        super().__init__()

        self.output_tokens = output_tokens
        self.pooling_type = pooling_type

        if pooling_type == 'attention':
            # Learnable query tokens for attention pooling
            self.query_tokens = nn.Parameter(
                torch.randn(1, output_tokens, input_dim)
            )
            nn.init.trunc_normal_(self.query_tokens, std=0.02)

            self.attention = nn.MultiheadAttention(
                input_dim, num_heads=8, batch_first=True
            )
        elif pooling_type == 'learned':
            # Learned linear projection
            self.pool = nn.Linear(input_dim, output_tokens)
        else:
            # 'mean' or 'max' pooling
            self.pool = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tokens [B, N, input_dim]

        Returns:
            Pooled tokens [B, output_tokens, input_dim]
        """
        B = x.shape[0]

        if self.pooling_type == 'attention':
            # Attention pooling
            query = self.query_tokens.expand(B, -1, -1)
            pooled, _ = self.attention(query, x, x)
            return pooled

        elif self.pooling_type == 'learned':
            # Learned pooling
            x = x.transpose(1, 2)  # [B, input_dim, N]
            pooled = self.pool(x)  # [B, input_dim, output_tokens]
            return pooled.transpose(1, 2)  # [B, output_tokens, input_dim]

        elif self.pooling_type == 'mean':
            # Mean pooling with interpolation
            if x.shape[1] == self.output_tokens:
                return x
            # Interpolate to output_tokens
            x = x.transpose(1, 2)  # [B, input_dim, N]
            x = torch.nn.functional.adaptive_avg_pool1d(
                x, self.output_tokens
            )
            return x.transpose(1, 2)  # [B, output_tokens, input_dim]

        elif self.pooling_type == 'max':
            # Max pooling with interpolation
            if x.shape[1] == self.output_tokens:
                return x
            x = x.transpose(1, 2)  # [B, input_dim, N]
            x = torch.nn.functional.adaptive_max_pool1d(
                x, self.output_tokens
            )
            return x.transpose(1, 2)  # [B, output_tokens, input_dim]

        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
