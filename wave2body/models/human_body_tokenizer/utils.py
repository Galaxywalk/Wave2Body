# -*- coding: utf-8 -*-
"""Utility functions for human-body tokenizer checkpoints."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, Dict, Optional

from .model_vqvae import create_joint_tokenizer


def load_pretrained_vqvae(
    checkpoint_path: Union[str, Path],
    config: Optional[Dict] = None,
    device: str = 'cpu',
    strict: bool = True,
) -> nn.Module:
    """
    Load a pretrained VQ-VAE model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file (.pth or .pt)
        config: Model configuration dict. If None, will try to load from checkpoint.
        device: Device to load model on
        strict: Whether to strictly enforce state_dict keys match

    Returns:
        Loaded VQ-VAE model

    Example:
        >>> vqvae = load_pretrained_vqvae('checkpoints/vqvae_best.pth')
        >>> vqvae.eval()
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config if not provided
    if config is None:
        if 'config' in checkpoint:
            config = checkpoint['config']
        elif 'model_config' in checkpoint:
            config = checkpoint['model_config']
        else:
            raise ValueError(
                "Config not provided and not found in checkpoint. "
                "Please provide config argument."
            )

    # Create model
    model = create_joint_tokenizer(config)

    # Load state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # Assume the checkpoint is the state dict itself
        state_dict = checkpoint

    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

    if missing_keys:
        print(f"Warning: Missing keys in state dict: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in state dict: {unexpected_keys}")

    model = model.to(device)

    # Print info
    if 'epoch' in checkpoint:
        print(f"Loaded VQ-VAE from epoch {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"Checkpoint loss: {checkpoint['loss']:.4f}")

    return model


def save_vqvae_checkpoint(
    model: nn.Module,
    save_path: Union[str, Path],
    epoch: Optional[int] = None,
    optimizer_state: Optional[Dict] = None,
    loss: Optional[float] = None,
    config: Optional[Dict] = None,
    **kwargs
):
    """
    Save VQ-VAE checkpoint with all necessary information.

    Args:
        model: JointTokenizer model to save
        save_path: Path to save checkpoint
        epoch: Current epoch number
        optimizer_state: Optimizer state dict
        loss: Current loss value
        config: Model configuration
        **kwargs: Additional items to save
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config or {},
    }

    if epoch is not None:
        checkpoint['epoch'] = epoch
    if optimizer_state is not None:
        checkpoint['optimizer_state_dict'] = optimizer_state
    if loss is not None:
        checkpoint['loss'] = loss

    # Add any additional items
    checkpoint.update(kwargs)

    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


def get_vqvae_config_from_checkpoint(checkpoint_path: Union[str, Path]) -> Dict:
    """
    Extract VQ-VAE configuration from checkpoint without loading the model.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Configuration dictionary
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'config' in checkpoint:
        return checkpoint['config']
    elif 'model_config' in checkpoint:
        return checkpoint['model_config']
    else:
        raise ValueError("Config not found in checkpoint")
