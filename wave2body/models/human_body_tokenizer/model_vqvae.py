#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Joint VQ-VAE Model inspired by PCT architecture
基于PCT架构的关节VQ-VAE模型，保持PCT的结构特点
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Dict


class MixerLayer(nn.Module):
    """MLP-Mixer层，完全参考PCT的实现"""
    def __init__(self, hidden_dim: int, hidden_inter_dim: int,
                 token_num: int, token_inter_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.token_num = token_num

        # Token mixing (spatial) - 完全按照PCT的方式
        self.token_mixer = nn.Sequential(
            nn.Linear(token_num, token_inter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_inter_dim, token_num),
            nn.Dropout(dropout)
        )

        # Channel mixing (feature) - 完全按照PCT的方式
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_inter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_inter_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, token_num, hidden_dim) - 与PCT保持一致
        # Token mixing - 按照PCT的residual connection方式
        x = x + self.token_mixer(x.transpose(1, 2)).transpose(1, 2)
        # Channel mixing - 按照PCT的residual connection方式
        x = x + self.channel_mixer(x)
        return x


class JointTokenizer(nn.Module):
    """
    关节Tokenizer，基于PCT的tokenizer架构
    包含编码器、codebook和解码器
    """
    def __init__(self,
                 # 基础参数
                 num_joints: int = 22,
                 input_dim: int = 3,  # 每个关节的维度 (x, y, z)

                 # 编码器参数
                 enc_num_blocks: int = 4,
                 enc_hidden_dim: int = 256,
                 enc_token_inter_dim: int = 512,  # token mixing的中间维度
                 enc_hidden_inter_dim: int = 1024,  # channel mixing的中间维度
                 enc_dropout: float = 0.1,

                 # Codebook参数
                 token_num: int = 64,  # 每个样本的token数量
                 token_class_num: int = 512,  # codebook大小
                 token_dim: int = 256,  # token维度
                 ema_decay: float = 0.99,

                 # 解码器参数
                 dec_num_blocks: int = 4,
                 dec_hidden_dim: int = 256,
                 dec_token_inter_dim: int = 512,
                 dec_hidden_inter_dim: int = 1024,
                 dec_dropout: float = 0.1,

                 # 训练参数
                 drop_rate: float = 0.0):  # random masking rate

        super().__init__()

        # 存储所有配置参数
        self.num_joints = num_joints
        self.input_dim = input_dim

        # 编码器配置
        self.enc_num_blocks = enc_num_blocks
        self.enc_hidden_dim = enc_hidden_dim
        self.enc_token_inter_dim = enc_token_inter_dim
        self.enc_hidden_inter_dim = enc_hidden_inter_dim
        self.enc_dropout = enc_dropout

        # Codebook配置
        self.token_num = token_num
        self.token_class_num = token_class_num
        self.token_dim = token_dim
        self.ema_decay = ema_decay

        # 解码器配置
        self.dec_num_blocks = dec_num_blocks
        self.dec_hidden_dim = dec_hidden_dim
        self.dec_token_inter_dim = dec_token_inter_dim
        self.dec_hidden_inter_dim = dec_hidden_inter_dim
        self.dec_dropout = dec_dropout

        # 训练配置
        self.drop_rate = drop_rate

        # === 编码器部分 ===
        # 输入嵌入 - 将3D关节坐标映射到隐藏维度
        self.start_embed = nn.Linear(input_dim, enc_hidden_dim)

        # 不可见token（用于masking，类似PCT）
        self.invisible_token = nn.Parameter(torch.zeros(1, 1, enc_hidden_dim))
        nn.init.trunc_normal_(self.invisible_token, mean=0., std=0.02, a=-0.02, b=0.02)

        # 编码器Mixer层
        self.encoder = nn.ModuleList([
            MixerLayer(enc_hidden_dim, enc_hidden_inter_dim,
                      num_joints, enc_token_inter_dim, enc_dropout)
            for _ in range(enc_num_blocks)
        ])
        self.encoder_layer_norm = nn.LayerNorm(enc_hidden_dim)

        # Token生成层
        self.token_mlp = nn.Linear(num_joints, token_num)
        self.feature_embed = nn.Linear(enc_hidden_dim, token_dim)

        # === Codebook部分（VQ-VAE的核心）===
        # 注册codebook相关的buffer（不会被梯度更新）
        self.register_buffer('codebook', torch.empty(token_class_num, token_dim))
        self.codebook.data.normal_()

        # EMA更新相关的buffer
        self.register_buffer('ema_cluster_size', torch.zeros(token_class_num))
        self.register_buffer('ema_w', torch.empty(token_class_num, token_dim))
        self.ema_w.data.normal_()

        # === 解码器部分 ===
        # Token恢复层
        self.decoder_token_mlp = nn.Linear(token_num, num_joints)
        self.decoder_start = nn.Linear(token_dim, dec_hidden_dim)

        # 解码器Mixer层
        self.decoder = nn.ModuleList([
            MixerLayer(dec_hidden_dim, dec_hidden_inter_dim,
                      num_joints, dec_token_inter_dim, dec_dropout)
            for _ in range(dec_num_blocks)
        ])
        self.decoder_layer_norm = nn.LayerNorm(dec_hidden_dim)

        # 输出层 - 恢复到原始关节维度
        self.recover_embed = nn.Linear(dec_hidden_dim, input_dim)

    def encode(self, joints, joints_visible=None, train=True):
        """
        编码阶段：将关节位置编码为离散tokens

        Args:
            joints: (B, num_joints, input_dim) 关节位置
            joints_visible: (B, num_joints) 关节可见性掩码
            train: 是否训练模式

        Returns:
            encoding_indices: 量化后的token索引
            quantized_features: 量化后的特征
            e_latent_loss: VQ损失
        """
        B = joints.shape[0]

        # 输入嵌入
        encode_feat = self.start_embed(joints)  # (B, num_joints, enc_hidden_dim)

        # Random masking during training (like PCT)
        if train and joints_visible is not None and self.drop_rate > 0:
            rand_mask_ind = torch.rand(joints_visible.shape, device=joints.device) > self.drop_rate
            joints_visible = torch.logical_and(rand_mask_ind, joints_visible)

        # Apply visibility mask
        if joints_visible is not None:
            mask_tokens = self.invisible_token.expand(B, self.num_joints, -1)
            w = joints_visible.unsqueeze(-1).type_as(mask_tokens)
            encode_feat = encode_feat * w + mask_tokens * (1 - w)

        # 通过编码器
        for layer in self.encoder:
            encode_feat = layer(encode_feat)
        encode_feat = self.encoder_layer_norm(encode_feat)

        # 生成tokens
        encode_feat = encode_feat.transpose(2, 1)  # (B, enc_hidden_dim, num_joints)
        encode_feat = self.token_mlp(encode_feat).transpose(2, 1)  # (B, token_num, enc_hidden_dim)
        encode_feat = self.feature_embed(encode_feat)  # (B, token_num, token_dim)
        encode_feat_flat = encode_feat.flatten(0, 1)  # (B*token_num, token_dim)

        # Vector Quantization
        distances = torch.sum(encode_feat_flat**2, dim=1, keepdim=True) \
                    + torch.sum(self.codebook**2, dim=1) \
                    - 2 * torch.matmul(encode_feat_flat, self.codebook.t())

        encoding_indices = torch.argmin(distances, dim=1)  # (B*token_num,)
        encodings = torch.zeros(encoding_indices.shape[0], self.token_class_num,
                               device=joints.device, dtype=encode_feat.dtype)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # 获取量化后的特征
        quantized_features = torch.matmul(encodings, self.codebook)  # (B*token_num, token_dim)
        quantized_features = quantized_features.view(B, self.token_num, self.token_dim)

        # VQ损失和straight-through estimator
        if train:
            # EMA更新codebook
            if dist.is_available() and dist.is_initialized():
                # 分布式训练时同步
                dw = torch.matmul(encodings.t(), encode_feat_flat.detach())
                n_encodings, n_dw = encodings.numel(), dw.numel()
                encodings_shape, dw_shape = encodings.shape, dw.shape
                combined = torch.cat((encodings.flatten(), dw.flatten()))
                dist.all_reduce(combined)
                sync_encodings, sync_dw = torch.split(combined, [n_encodings, n_dw])
                sync_encodings = sync_encodings.view(encodings_shape)
                sync_dw = sync_dw.view(dw_shape)
            else:
                sync_encodings = encodings
                sync_dw = torch.matmul(encodings.t(), encode_feat_flat.detach())

            # 更新EMA参数
            self.ema_cluster_size.data.mul_(self.ema_decay).add_(
                torch.sum(sync_encodings, 0), alpha=1 - self.ema_decay)

            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size.data.add_(1e-5).div_(
                n + self.token_class_num * 1e-5).mul_(n)

            self.ema_w.data.mul_(self.ema_decay).add_(sync_dw, alpha=1 - self.ema_decay)
            self.codebook.data.copy_(self.ema_w / self.ema_cluster_size.unsqueeze(1))

            # VQ损失 (Commitment Loss) - 让编码器输出向量量化结果靠近
            e_latent_loss = F.mse_loss(quantized_features.detach(), encode_feat)

            # Straight-through estimator - 前向传播用量化结果，反向传播用编码结果
            quantized_features = encode_feat + (quantized_features - encode_feat).detach()
        else:
            e_latent_loss = None

        return encoding_indices.view(B, self.token_num), quantized_features, e_latent_loss

    def encode_with_intermediates(self, joints, joints_visible=None):
        """
        Encode joints and return intermediate representations for direct supervision.

        This method is useful for training Token Translator with direct latent alignment,
        bypassing the quantization bottleneck for gradient flow.

        Args:
            joints: (B, num_joints, input_dim) Joint positions
            joints_visible: (B, num_joints) Joint visibility mask (optional)

        Returns:
            dict containing:
                z_e: Pre-quantization latent [B, token_num, token_dim]
                z_q: Post-quantization latent [B, token_num, token_dim]
                indices: Codebook indices [B, token_num]
        """
        B = joints.shape[0]

        # Input embedding
        encode_feat = self.start_embed(joints)  # (B, num_joints, enc_hidden_dim)

        # Apply visibility mask if provided
        if joints_visible is not None:
            mask_tokens = self.invisible_token.expand(B, self.num_joints, -1)
            w = joints_visible.unsqueeze(-1).type_as(mask_tokens)
            encode_feat = encode_feat * w + mask_tokens * (1 - w)

        # Through encoder
        for layer in self.encoder:
            encode_feat = layer(encode_feat)
        encode_feat = self.encoder_layer_norm(encode_feat)

        # Generate tokens
        encode_feat = encode_feat.transpose(2, 1)  # (B, enc_hidden_dim, num_joints)
        encode_feat = self.token_mlp(encode_feat).transpose(2, 1)  # (B, token_num, enc_hidden_dim)
        z_e = self.feature_embed(encode_feat)  # (B, token_num, token_dim) - Pre-quantization

        # Vector Quantization
        z_e_flat = z_e.flatten(0, 1)  # (B*token_num, token_dim)
        distances = torch.sum(z_e_flat**2, dim=1, keepdim=True) \
                    + torch.sum(self.codebook**2, dim=1) \
                    - 2 * torch.matmul(z_e_flat, self.codebook.t())

        indices = torch.argmin(distances, dim=1)  # (B*token_num,)

        # Get quantized features
        z_q = self.codebook[indices]  # (B*token_num, token_dim)
        z_q = z_q.view(B, self.token_num, self.token_dim)
        indices = indices.view(B, self.token_num)

        return {
            'z_e': z_e,       # Pre-quantization latent for L_latent loss
            'z_q': z_q,       # Post-quantization latent
            'indices': indices  # Codebook indices for L_token loss
        }

    def decode(self, quantized_features):
        """
        解码阶段：将量化特征解码为关节位置

        Args:
            quantized_features: (B, token_num, token_dim) 量化后的特征

        Returns:
            recovered_joints: (B, num_joints, input_dim) 恢复的关节位置
        """
        B = quantized_features.shape[0]

        # Token维度变换
        part_token_feat = quantized_features.transpose(2, 1)  # (B, token_dim, token_num)
        part_token_feat = self.decoder_token_mlp(part_token_feat).transpose(2, 1)  # (B, num_joints, token_dim)

        # 解码器输入投影
        decode_feat = self.decoder_start(part_token_feat)  # (B, num_joints, dec_hidden_dim)

        # 通过解码器
        for layer in self.decoder:
            decode_feat = layer(decode_feat)
        decode_feat = self.decoder_layer_norm(decode_feat)

        # 输出投影
        recovered_joints = self.recover_embed(decode_feat)  # (B, num_joints, input_dim)

        return recovered_joints

    def forward(self, joints, joints_visible=None, cls_logits=None, train=True):
        """
        前向传播

        Args:
            joints: (B, num_joints, input_dim) 关节位置
            joints_visible: (B, num_joints) 关节可见性（可选）
            cls_logits: 分类器输出（用于classifier阶段）
            train: 是否训练模式

        Returns:
            recovered_joints: 重构的关节位置
            encoding_indices: 编码索引
            e_latent_loss: VQ损失
        """
        if cls_logits is not None:
            # Classifier阶段：直接使用分类器输出
            B = cls_logits.shape[0] // self.token_num
            quantized_features = torch.matmul(cls_logits, self.codebook)
            quantized_features = quantized_features.view(B, self.token_num, self.token_dim)
            encoding_indices = None
            e_latent_loss = None
        else:
            # Tokenizer阶段：编码-量化-解码
            encoding_indices, quantized_features, e_latent_loss = self.encode(
                joints, joints_visible, train)

        # 解码
        recovered_joints = self.decode(quantized_features)

        return recovered_joints, encoding_indices, e_latent_loss


class JointVQVAELoss(nn.Module):
    """Joint VQ-VAE损失函数"""
    def __init__(self,
                 recon_weight: float = 1.0,
                 vq_weight: float = 1.0,
                 commitment_weight: float = 0.25,
                 recon_loss_type: str = 'mse'):
        super().__init__()
        self.recon_weight = recon_weight
        self.vq_weight = vq_weight
        self.commitment_weight = commitment_weight

        if recon_loss_type == 'mse':
            self.recon_loss_fn = nn.MSELoss()
        elif recon_loss_type == 'l1':
            self.recon_loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unknown reconstruction loss type: {recon_loss_type}")

    def forward(self, recovered_joints, target_joints, e_latent_loss=None):
        """
        计算VQ-VAE损失

        Args:
            recovered_joints: (B, num_joints, 3) 重构的关节位置
            target_joints: (B, num_joints, 3) 目标关节位置
            e_latent_loss: VQ损失（训练时）

        Returns:
            loss_dict: 包含各项损失的字典
        """
        # 重构损失
        recon_loss = self.recon_loss_fn(recovered_joints, target_joints)

        # 确保损失是标量，避免DataParallel警告
        if recon_loss.dim() > 0:
            recon_loss = recon_loss.mean()

        losses = {
            'recon_loss': recon_loss,
            'total_loss': self.recon_weight * recon_loss
        }

        # VQ损失（仅训练时）- 这是commitment loss
        if e_latent_loss is not None:
            # 确保VQ损失也是标量
            if hasattr(e_latent_loss, 'dim') and e_latent_loss.dim() > 0:
                e_latent_loss = e_latent_loss.mean()

            # 应用commitment权重
            commitment_loss = self.commitment_weight * e_latent_loss
            losses['vq_loss'] = e_latent_loss  # 原始commitment loss用于监控
            losses['commitment_loss'] = e_latent_loss  # 添加commitment_loss字段用于训练代码兼容
            losses['total_loss'] = losses['total_loss'] + commitment_loss

        return losses


def create_joint_tokenizer(config: Dict) -> nn.Module:
    """
    根据配置创建Joint Tokenizer模型

    Args:
        config: 模型配置字典，包含所有必要参数

    Returns:
        JointTokenizer模型实例
    """
    # Part-level model dispatch (multi-codebook).
    model_type = str(config.get('model_type', 'joint'))
    is_part_model = (
        model_type in ('part', 'part_vqvae')
        or 'part_codebook_size' in config
        or 'part_token_num' in config
    )
    if is_part_model:
        from .model_part_vqvae import create_part_joint_tokenizer
        return create_part_joint_tokenizer(config)

    return JointTokenizer(
        # 基础参数
        num_joints=config.get('num_joints', 22),
        input_dim=config.get('input_dim', 3),

        # 编码器参数
        enc_num_blocks=config.get('enc_num_blocks', 4),
        enc_hidden_dim=config.get('enc_hidden_dim', 256),
        enc_token_inter_dim=config.get('enc_token_inter_dim', 512),
        enc_hidden_inter_dim=config.get('enc_hidden_inter_dim', 1024),
        enc_dropout=config.get('enc_dropout', 0.1),

        # Codebook参数
        token_num=config.get('token_num', 64),
        token_class_num=config.get('token_class_num', 512),
        token_dim=config.get('token_dim', 256),
        ema_decay=config.get('ema_decay', 0.99),

        # 解码器参数
        dec_num_blocks=config.get('dec_num_blocks', 4),
        dec_hidden_dim=config.get('dec_hidden_dim', 256),
        dec_token_inter_dim=config.get('dec_token_inter_dim', 512),
        dec_hidden_inter_dim=config.get('dec_hidden_inter_dim', 1024),
        dec_dropout=config.get('dec_dropout', 0.1),

        # 训练参数
        drop_rate=config.get('drop_rate', 0.0)
    )


# 使用示例
if __name__ == "__main__":
    # 测试模型
    config = {
        'num_joints': 22,
        'input_dim': 3,

        # 编码器配置
        'enc_num_blocks': 4,
        'enc_hidden_dim': 256,
        'enc_token_inter_dim': 512,
        'enc_hidden_inter_dim': 1024,
        'enc_dropout': 0.1,

        # Codebook配置
        'token_num': 64,
        'token_class_num': 512,
        'token_dim': 256,
        'ema_decay': 0.99,

        # 解码器配置
        'dec_num_blocks': 4,
        'dec_hidden_dim': 256,
        'dec_token_inter_dim': 512,
        'dec_hidden_inter_dim': 1024,
        'dec_dropout': 0.1,

        # 训练配置
        'drop_rate': 0.1
    }

    model = create_joint_tokenizer(config)

    # 测试输入
    batch_size = 8
    joints = torch.randn(batch_size, 22, 3)
    joints_visible = torch.ones(batch_size, 22).bool()

    # 前向传播
    recovered_joints, encoding_indices, e_latent_loss = model(
        joints, joints_visible, train=True)

    print(f"输入形状: {joints.shape}")
    print(f"重构形状: {recovered_joints.shape}")
    print(f"编码索引形状: {encoding_indices.shape if encoding_indices is not None else None}")
    print(f"VQ损失: {e_latent_loss.item() if e_latent_loss is not None else None}")

    # 测试损失
    criterion = JointVQVAELoss()
    loss_dict = criterion(recovered_joints, joints, e_latent_loss)
    print(f"损失字典: {loss_dict}")

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试配置的所有参数
    print("\n模型配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
