# -*- coding: utf-8 -*-
"""
Loss functions for token translation training.

This module implements multi-loss training for Token Translator to address
the weak gradient problem when training through VQ-VAE.

Key insight: Instead of only using pose-level MSE loss (which requires gradients
to flow through VQ-VAE decoder and STE quantization), we add direct supervision:
- L_latent: Direct latent space alignment (bypasses quantization entirely)
- L_token: Token classification loss (discrete supervision)
- L_pose: Final pose reconstruction loss (end-to-end supervision)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DirectLatentLoss(nn.Module):
    """
    Direct latent alignment loss (L_latent).

    Computes MSE between predicted latents (from Token Translator) and
    target latents (from VQ-VAE encoder on ground truth poses).

    This loss provides direct gradient to Token Translator output,
    bypassing the VQ quantization bottleneck entirely.
    """

    def __init__(self, normalize: bool = False):
        super().__init__()
        self.normalize = normalize

    def forward(
        self,
        pred_z_e: torch.Tensor,
        target_z_e: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_z_e: Predicted latent from Token Translator [B, token_num, token_dim]
            target_z_e: Target latent from VQ-VAE encoder [B, token_num, token_dim]

        Returns:
            MSE loss value
        """
        if self.normalize:
            pred_z_e = F.normalize(pred_z_e, p=2, dim=-1)
            target_z_e = F.normalize(target_z_e, p=2, dim=-1)

        # Target should be detached to prevent gradients flowing to VQ-VAE encoder.
        # Compute the scalar loss in fp32 so bf16/fp16 autocast forwards keep a
        # stable backward path.
        return F.mse_loss(pred_z_e.float(), target_z_e.detach().float())


class TokenClassificationLoss(nn.Module):
    """
    Token classification loss (L_token).

    Computes cross-entropy loss between predicted token logits
    and target token indices from VQ-VAE codebook.

    This provides discrete token-level supervision, teaching the
    Token Translator to predict the correct codebook entries.
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_logits: Predicted logits [B, token_num, codebook_size]
            target_indices: Target token indices [B, token_num]

        Returns:
            Cross-entropy loss value
        """
        B, N, C = pred_logits.shape

        # Reshape for cross-entropy: [B*N, C] and [B*N]
        pred_logits = pred_logits.view(B * N, C).float()
        target_indices = target_indices.view(B * N)

        return self.ce_loss(pred_logits, target_indices)


class TokenTranslatorLoss(nn.Module):
    """
    Combined loss function for Token Translator training.

    Combines three loss terms to address the weak gradient problem:

    1. L_latent (latent_loss_weight): Direct latent alignment
       - Bypasses VQ quantization entirely
       - Provides strong, direct gradients to Token Translator

    2. L_token (token_loss_weight): Token classification
       - Discrete codebook supervision
       - Teaches correct token mapping

    3. L_pose (pose_loss_weight): Pose reconstruction
       - End-to-end supervision through VQ-VAE decoder
       - Ensures final pose accuracy

    Training Strategy:
    - Stage 1: High L_latent + L_token, low L_pose
      (Focus on learning correct token representations)
    - Stage 2: Balanced weights
      (Fine-tune for end-to-end accuracy)
    """

    def __init__(
        self,
        latent_loss_weight: float = 1.0,
        token_loss_weight: float = 0.5,
        pose_loss_weight: float = 0.1,
        label_smoothing: float = 0.0,
        normalize_latent: bool = False,
    ):
        super().__init__()

        self.latent_loss_weight = latent_loss_weight
        self.token_loss_weight = token_loss_weight
        self.pose_loss_weight = pose_loss_weight

        # Individual loss functions
        self.latent_loss_fn = DirectLatentLoss(normalize=normalize_latent)
        self.token_loss_fn = TokenClassificationLoss(label_smoothing=label_smoothing)
        self.pose_loss_fn = nn.MSELoss()

    def forward(
        self,
        pred_z_e: Optional[torch.Tensor] = None,
        target_z_e: Optional[torch.Tensor] = None,
        pred_logits: Optional[torch.Tensor] = None,
        target_indices: Optional[torch.Tensor] = None,
        pred_pose: Optional[torch.Tensor] = None,
        target_pose: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            pred_z_e: Predicted latent from Token Translator [B, token_num, token_dim]
            target_z_e: Target latent from VQ-VAE encoder [B, token_num, token_dim]
            pred_logits: Predicted token logits [B, token_num, codebook_size]
            target_indices: Target token indices [B, token_num]
            pred_pose: Predicted pose [B, num_joints, 3]
            target_pose: Target pose [B, num_joints, 3]

        Returns:
            Dictionary containing:
                - latent_loss: L_latent value
                - token_loss: L_token value
                - pose_loss: L_pose value
                - total_loss: Weighted sum of all losses
        """
        losses = {}
        total_loss = 0.0
        device = None

        # L_latent: Direct latent alignment (primary for Stage 1)
        if pred_z_e is not None and target_z_e is not None:
            device = pred_z_e.device
            latent_loss = self.latent_loss_fn(pred_z_e, target_z_e)
            losses['latent_loss'] = latent_loss
            total_loss = total_loss + self.latent_loss_weight * latent_loss

        # L_token: Token classification (secondary for Stage 1)
        if pred_logits is not None and target_indices is not None:
            device = pred_logits.device
            token_loss = self.token_loss_fn(pred_logits, target_indices)
            losses['token_loss'] = token_loss
            total_loss = total_loss + self.token_loss_weight * token_loss

        # L_pose: Pose reconstruction (auxiliary for Stage 1, primary for Stage 2)
        if pred_pose is not None and target_pose is not None:
            device = pred_pose.device
            pose_loss = self.pose_loss_fn(pred_pose.float(), target_pose.float())
            losses['pose_loss'] = pose_loss
            total_loss = total_loss + self.pose_loss_weight * pose_loss

        # Handle case where no losses were computed
        if device is None:
            raise ValueError("At least one pair of (pred, target) must be provided")

        losses['total_loss'] = total_loss

        return losses


class TokenTranslationLoss(nn.Module):
    """
    Legacy loss function for token translation (kept for backward compatibility).

    Combines multiple loss terms:
    1. Token prediction loss (MSE/CE for token indices)
    2. Feature reconstruction loss (MSE for token features)
    3. Contrastive loss (optional, for better alignment)
    """

    def __init__(
        self,
        token_pred_weight: float = 1.0,
        feature_recon_weight: float = 1.0,
        contrastive_weight: float = 0.1,
        temperature: float = 0.07,
        use_contrastive: bool = True,
    ):
        super().__init__()

        self.token_pred_weight = token_pred_weight
        self.feature_recon_weight = feature_recon_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.use_contrastive = use_contrastive

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        pred_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        pred_features: Optional[torch.Tensor] = None,
        target_features: Optional[torch.Tensor] = None,
        target_token_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.

        Args:
            pred_tokens: Predicted pose tokens [B, num_tokens, token_dim]
            target_tokens: Target pose tokens [B, num_tokens, token_dim]
            pred_features: Predicted features (optional)
            target_features: Target features (optional)
            target_token_indices: Target token indices for classification loss [B, num_tokens]

        Returns:
            Dictionary of losses
        """
        losses = {}

        # Feature reconstruction loss (MSE)
        if pred_features is not None and target_features is not None:
            feature_loss = self.mse_loss(pred_features, target_features)
            losses['feature_recon'] = feature_loss
        else:
            # Fallback: use token features directly
            feature_loss = self.mse_loss(pred_tokens, target_tokens)
            losses['feature_recon'] = feature_loss

        # Token prediction loss (if token indices provided)
        token_pred_loss = None
        if target_token_indices is not None:
            # Compute logits from predicted tokens
            # This assumes we have access to codebook for computing logits
            # For now, we'll use a simple approach: predict token indices
            # In practice, you'd compute distances to codebook entries
            token_pred_loss = torch.tensor(0.0, device=pred_tokens.device)
            losses['token_pred'] = token_pred_loss

        # Contrastive loss (optional)
        contrastive_loss = None
        if self.use_contrastive:
            # Normalize features for contrastive learning
            pred_norm = F.normalize(pred_tokens, p=2, dim=-1)
            target_norm = F.normalize(target_tokens, p=2, dim=-1)

            # Compute similarity matrix
            # [B, num_tokens, token_dim] -> [B, num_tokens, num_tokens]
            similarity = torch.bmm(
                pred_norm, target_norm.transpose(1, 2)
            ) / self.temperature

            # Positive pairs: diagonal elements
            B, N, _ = pred_tokens.shape
            labels = torch.arange(N, device=pred_tokens.device).unsqueeze(0).expand(B, -1)

            # Reshape for cross-entropy
            similarity = similarity.view(B * N, N)
            labels = labels.view(B * N)

            contrastive_loss = self.ce_loss(similarity, labels)
            losses['contrastive'] = contrastive_loss

        # Total loss
        total_loss = (
            self.feature_recon_weight * feature_loss +
            (self.token_pred_weight * token_pred_loss if token_pred_loss is not None else 0.0) +
            (self.contrastive_weight * contrastive_loss if contrastive_loss is not None else 0.0)
        )
        losses['total'] = total_loss

        return losses



class ContrastiveTokenLoss(nn.Module):
    """
    Contrastive loss for token alignment.

    Encourages similar mmWave tokens and pose tokens to be close
    in the embedding space.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        mmwave_features: torch.Tensor,
        pose_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            mmwave_features: mmWave token features [B, N_mmwave, dim]
            pose_features: Pose token features [B, N_pose, dim]

        Returns:
            Contrastive loss
        """
        # Normalize features
        mmwave_norm = F.normalize(mmwave_features, p=2, dim=-1)
        pose_norm = F.normalize(pose_features, p=2, dim=-1)

        # Compute similarity matrix
        # [B, N_mmwave, dim] x [B, dim, N_pose] -> [B, N_mmwave, N_pose]
        similarity = torch.bmm(
            mmwave_norm, pose_norm.transpose(1, 2)
        ) / self.temperature

        # InfoNCE loss: maximize diagonal (positive pairs)
        # For simplicity, we use mean of diagonal as positive similarity
        B, N_mmwave, N_pose = similarity.shape

        # Positive pairs: assume first N_pose mmWave tokens correspond to N_pose pose tokens
        min_tokens = min(N_mmwave, N_pose)
        positive_sim = torch.diagonal(
            similarity[:, :min_tokens, :min_tokens], dim1=1, dim2=2
        ).mean()

        # Negative pairs: off-diagonal elements
        # For simplicity, use all off-diagonal elements
        mask = torch.eye(min_tokens, device=similarity.device).bool()
        negative_sim = similarity[:, :min_tokens, :min_tokens].masked_fill(
            mask.unsqueeze(0), float('-inf')
        ).exp().sum(dim=-1).mean()

        # InfoNCE loss
        loss = -torch.log(positive_sim.exp() / (positive_sim.exp() + negative_sim))

        return loss
