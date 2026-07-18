# -*- coding: utf-8 -*-
"""
Configuration validation and utilities.
"""

from typing import Dict, List, Any
from pathlib import Path


def _is_part_vqvae_config(vqvae_config: Dict) -> bool:
    model_type = str(vqvae_config.get('model_type', 'joint'))
    return (
        model_type in ('part', 'part_vqvae')
        or 'part_codebook_size' in vqvae_config
        or 'part_token_num' in vqvae_config
    )


def _infer_num_parts(vqvae_config: Dict) -> int:
    if isinstance(vqvae_config.get('part_token_layout_order'), list):
        return len(vqvae_config['part_token_layout_order'])
    if isinstance(vqvae_config.get('part_token_num'), dict):
        return len(vqvae_config['part_token_num'])
    if isinstance(vqvae_config.get('part_codebook_size'), dict):
        return len(vqvae_config['part_codebook_size'])
    return 5


def _sum_part_values(value: Any, num_parts: int) -> int:
    if isinstance(value, int):
        return int(value) * int(num_parts)
    if isinstance(value, dict):
        return int(sum(int(v) for v in value.values()))
    raise TypeError(f"Unsupported part config value type: {type(value)}")


def validate_config(config: Dict) -> List[str]:
    """
    Validate configuration and return list of errors.

    Args:
        config: Configuration dictionary

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Check required top-level keys
    required_keys = ['point_transformer', 'token_translator', 'vqvae', 'training', 'dataset']
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required config section: {key}")

    # Validate Point Transformer config
    if 'point_transformer' in config:
        pt_config = config['point_transformer']
        if 'data_format' not in pt_config:
            errors.append("Point Transformer: missing 'data_format'")
        if 'model' not in pt_config:
            errors.append("Point Transformer: missing 'model'")

    # Validate Token Translator config
    if 'token_translator' in config:
        tt_config = config['token_translator']
        required_tt = ['mmwave_token_dim', 'pose_token_dim', 'num_pose_tokens', 'num_mmwave_tokens']
        for key in required_tt:
            if key not in tt_config:
                errors.append(f"Token Translator: missing '{key}'")

    # Validate VQ-VAE config
    if 'vqvae' in config:
        vqvae_config = config['vqvae']
        if _is_part_vqvae_config(vqvae_config):
            required_vqvae = ['token_dim', 'part_token_num', 'part_codebook_size']
        else:
            required_vqvae = ['token_num', 'token_dim', 'token_class_num']
        for key in required_vqvae:
            if key not in vqvae_config:
                errors.append(f"VQ-VAE: missing '{key}'")

    # Validate dimension consistency
    if 'token_translator' in config and 'vqvae' in config:
        tt_pose_tokens = config['token_translator'].get('num_pose_tokens')
        vqvae_cfg = config['vqvae']
        if _is_part_vqvae_config(vqvae_cfg):
            num_parts = _infer_num_parts(vqvae_cfg)
            try:
                vqvae_token_num = _sum_part_values(vqvae_cfg.get('part_token_num'), num_parts)
            except Exception as exc:
                errors.append(f"VQ-VAE: invalid part_token_num ({exc})")
                vqvae_token_num = None
        else:
            vqvae_token_num = vqvae_cfg.get('token_num')
        if tt_pose_tokens != vqvae_token_num:
            errors.append(
                f"Dimension mismatch: token_translator.num_pose_tokens ({tt_pose_tokens}) "
                f"!= vqvae.token_num ({vqvae_token_num})"
            )

        tt_pose_dim = config['token_translator'].get('pose_token_dim')
        vqvae_token_dim = vqvae_cfg.get('token_dim')
        if tt_pose_dim != vqvae_token_dim:
            errors.append(
                f"Dimension mismatch: token_translator.pose_token_dim ({tt_pose_dim}) "
                f"!= vqvae.token_dim ({vqvae_token_dim})"
            )

        tt_codebook_size = config['token_translator'].get('codebook_size')
        if tt_codebook_size is not None:
            if _is_part_vqvae_config(vqvae_cfg):
                num_parts = _infer_num_parts(vqvae_cfg)
                try:
                    vqvae_codebook_size = _sum_part_values(vqvae_cfg.get('part_codebook_size'), num_parts)
                except Exception as exc:
                    errors.append(f"VQ-VAE: invalid part_codebook_size ({exc})")
                    vqvae_codebook_size = None
            else:
                vqvae_codebook_size = vqvae_cfg.get('token_class_num')
            if vqvae_codebook_size is not None and int(tt_codebook_size) != int(vqvae_codebook_size):
                errors.append(
                    f"Dimension mismatch: token_translator.codebook_size ({tt_codebook_size}) "
                    f"!= vqvae.codebook_size ({vqvae_codebook_size})"
                )

    # Validate Training config
    if 'training' in config:
        train_config = config['training']
        required_train = ['batch_size', 'epochs', 'lr']
        for key in required_train:
            if key not in train_config:
                errors.append(f"Training: missing '{key}'")

    # Validate Dataset config
    if 'dataset' in config:
        dataset_config = config['dataset']
        if 'name' not in dataset_config:
            errors.append("Dataset: missing 'name'")
        if 'dataset_path' not in config:
            errors.append("Dataset: missing 'dataset_path'")

    return errors


def print_config_summary(config: Dict):
    """
    Print a formatted summary of the configuration.

    Args:
        config: Configuration dictionary
    """
    print("\n" + "="*60)
    print("Configuration Summary")
    print("="*60)

    # Point Transformer
    if 'point_transformer' in config:
        pt = config['point_transformer']
        print("\n[Point Transformer]")
        print(f"  Scale: {pt.get('model', {}).get('scale', 'N/A')}")
        print(f"  Input channels: {pt.get('data_format', {}).get('radar_input_c', 'N/A')}")
        print(f"  Num joints: {pt.get('data_format', {}).get('num_joints', 'N/A')}")

    # Token Translator
    if 'token_translator' in config:
        tt = config['token_translator']
        print("\n[Token Translator]")
        print(f"  mmWave tokens: {tt.get('num_mmwave_tokens', 'N/A')} x {tt.get('mmwave_token_dim', 'N/A')}")
        print(f"  Pose tokens: {tt.get('num_pose_tokens', 'N/A')} x {tt.get('pose_token_dim', 'N/A')}")
        print(f"  Cross-attn layers: {tt.get('num_cross_attn_layers', 'N/A')}")
        print(f"  Self-attn layers: {tt.get('num_self_attn_layers', 'N/A')}")

    # VQ-VAE
    if 'vqvae' in config:
        vqvae = config['vqvae']
        if _is_part_vqvae_config(vqvae):
            num_parts = _infer_num_parts(vqvae)
            codebook_size = _sum_part_values(vqvae.get('part_codebook_size', {}), num_parts)
            token_num = _sum_part_values(vqvae.get('part_token_num', {}), num_parts)
        else:
            codebook_size = vqvae.get('token_class_num', 'N/A')
            token_num = vqvae.get('token_num', 'N/A')
        print("\n[VQ-VAE]")
        print(f"  Token num: {token_num}")
        print(f"  Token dim: {vqvae.get('token_dim', 'N/A')}")
        print(f"  Codebook size: {codebook_size}")
        print(f"  Model type: {'part' if _is_part_vqvae_config(vqvae) else 'joint'}")
        print(f"  Encoder blocks: {vqvae.get('enc_num_blocks', 'N/A')}")
        print(f"  Decoder blocks: {vqvae.get('dec_num_blocks', 'N/A')}")

    # Training
    if 'training' in config:
        train = config['training']
        print("\n[Training]")
        print(f"  Batch size: {train.get('batch_size', 'N/A')}")
        print(f"  Epochs: {train.get('epochs', 'N/A')}")
        print(f"  Learning rate: {train.get('lr', 'N/A')}")
        print(f"  Weight decay: {train.get('weight_decay', 'N/A')}")
        print(f"  Freeze PT: {train.get('freeze_point_transformer', False)}")
        print(f"  Freeze VQ-VAE: {train.get('freeze_vqvae', True)}")

    # Dataset
    if 'dataset' in config:
        dataset = config['dataset']
        print("\n[Dataset]")
        print(f"  Name: {dataset.get('name', 'N/A')}")
        print(f"  Path: {config.get('dataset_path', 'N/A')}")

    print("\n" + "="*60 + "\n")


def check_pretrained_models(config: Dict) -> Dict[str, bool]:
    """
    Check if pretrained model paths exist.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary indicating which models are available
    """
    available = {}

    # Check Point Transformer
    pt_path = config.get('pretrained', {}).get('point_transformer', None)
    if pt_path:
        available['point_transformer'] = Path(pt_path).exists()
    else:
        available['point_transformer'] = False

    # Check VQ-VAE
    vqvae_path = config.get('pretrained', {}).get('vqvae', None)
    if vqvae_path:
        available['vqvae'] = Path(vqvae_path).exists()
    else:
        available['vqvae'] = False

    # Check Token Translator
    tt_path = config.get('pretrained', {}).get('token_translator', None)
    if tt_path:
        available['token_translator'] = Path(tt_path).exists()
    else:
        available['token_translator'] = False

    return available


def update_config_from_args(config: Dict, args) -> Dict:
    """
    Update configuration from command line arguments.

    Args:
        config: Base configuration
        args: Parsed command line arguments

    Returns:
        Updated configuration
    """
    # Update training params
    if hasattr(args, 'batch_size') and args.batch_size:
        config['training']['batch_size'] = args.batch_size

    if hasattr(args, 'lr') and args.lr:
        config['training']['lr'] = args.lr

    if hasattr(args, 'epochs') and args.epochs:
        config['training']['epochs'] = args.epochs

    # Update device
    if hasattr(args, 'device') and args.device:
        config['training']['device'] = args.device

    # Update pretrained paths
    if hasattr(args, 'pt_checkpoint') and args.pt_checkpoint:
        if 'pretrained' not in config:
            config['pretrained'] = {}
        config['pretrained']['point_transformer'] = args.pt_checkpoint

    if hasattr(args, 'vqvae_checkpoint') and args.vqvae_checkpoint:
        if 'pretrained' not in config:
            config['pretrained'] = {}
        config['pretrained']['vqvae'] = args.vqvae_checkpoint

    return config
