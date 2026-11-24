# -*- coding: utf-8 -*-
"""Deployable inference optimizations for Wave2Body HPE-token models.

Provides bitwise-equivalent inference utilities without altering trained weights:
- ``prune_body_encoder`` strips the training-only Stage-2 body encoder (``E_body``).
- ``BatchedDeployablePartVQVAE`` batches the five regional VQ lookups (and optionally
  the regional MLP decoders) into single ``torch.bmm`` / ``torch.baddbmm`` calls.
- ``DeployableWave2BodyInferenceWrapper`` exposes a thin tensor-in / tensor-out
  deployment module suitable for eager execution or ``torch.compile``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .absolute_root import AbsoluteRootHeadEstimatorModel, masked_point_root_stats
from ..mmwave_tokenizer.grouping import triton_fps_available


def prune_body_encoder(model: nn.Module) -> bool:
    """Strip the training-only body encoder ``E_body`` from a Wave2Body model in place.

    Returns True when ``vqvae.encoders`` was present and removed.
    """
    vqvae = getattr(model, "vqvae", None)
    if vqvae is not None and hasattr(vqvae, "encoders"):
        del vqvae.encoders
        return True
    return False


class BatchedDeployablePartVQVAE(nn.Module):
    """Batched inference wrapper around ``PartJointTokenizer`` (with ``E_body`` pruned).

    Replaces sequential per-part codebook lookups with a single batched matrix
    multiplication, and optionally fuses the per-part 3-layer MLP decoders into
    three batched ``torch.baddbmm`` operations. Original ``decoders`` and
    ``quantizers`` modules are retained so parameter counts stay unchanged.
    """

    def __init__(self, base_vqvae: nn.Module, *, fuse_mlp_decoder: bool = True) -> None:
        super().__init__()
        if hasattr(base_vqvae, "encoders"):
            del base_vqvae.encoders
        if not hasattr(base_vqvae, "layout") or not hasattr(base_vqvae, "quantizers"):
            raise TypeError("BatchedDeployablePartVQVAE requires a PartJointTokenizer instance.")

        self.layout = base_vqvae.layout
        self.num_joints = int(base_vqvae.num_joints)
        self.input_dim = int(base_vqvae.input_dim)
        self.token_dim = int(base_vqvae.token_dim)
        self.token_num = int(base_vqvae.token_num)
        self.token_class_num = int(base_vqvae.token_class_num)
        self.fuse_mlp_decoder = bool(fuse_mlp_decoder)

        # Retain original ModuleDicts so deployable parameter count is unchanged.
        self.decoders = base_vqvae.decoders
        self.quantizers = base_vqvae.quantizers

        part_names = list(self.layout.part_names)
        num_parts = len(part_names)
        if num_parts == 0:
            raise ValueError("PartLayout must define at least one body part.")
        max_tok = max(int(self.layout.part_token_num[p]) for p in part_names)
        code_dim = self.token_dim
        n_codes = int(self.layout.part_codebook_size[part_names[0]])
        if not all(int(self.layout.part_codebook_size[p]) == n_codes for p in part_names):
            raise ValueError("BatchedDeployablePartVQVAE requires uniform per-part codebook size.")

        self._num_parts = num_parts
        self._max_tok = max_tok

        cb_stack = torch.stack([self.quantizers[p].codebook.detach().float() for p in part_names], dim=0)
        cb_t = cb_stack.transpose(1, 2).contiguous()
        cb_sq = (cb_stack ** 2).sum(dim=-1).unsqueeze(1).contiguous()
        cb_global = torch.cat([self.quantizers[p].codebook.detach().float() for p in part_names], dim=0).contiguous()

        pad_idx_list: list[int] = []
        unpad_idx_list: list[int] = []
        offsets_list: list[int] = []
        pad_valid_mask_list: list[float] = []
        for p_idx, p_name in enumerate(part_names):
            sl = self.layout.part_token_slices[p_name]
            t_cnt = int(self.layout.part_token_num[p_name])
            off = int(self.layout.part_codebook_offsets[p_name])
            for s in range(max_tok):
                if s < t_cnt:
                    pad_idx_list.append(int(sl.start) + s)
                    unpad_idx_list.append(p_idx * max_tok + s)
                    offsets_list.append(off)
                    pad_valid_mask_list.append(1.0)
                else:
                    pad_idx_list.append(int(sl.stop) - 1)
                    pad_valid_mask_list.append(0.0)

        self.register_buffer("_fast_cb_t", cb_t, persistent=False)
        self.register_buffer("_fast_cb_sq", cb_sq, persistent=False)
        self.register_buffer("_fast_cb_global", cb_global, persistent=False)
        self.register_buffer("_fast_pad_idx", torch.tensor(pad_idx_list, dtype=torch.long), persistent=False)
        self.register_buffer("_fast_unpad_idx", torch.tensor(unpad_idx_list, dtype=torch.long), persistent=False)
        self.register_buffer("_fast_code_offsets", torch.tensor(offsets_list, dtype=torch.long).unsqueeze(0), persistent=False)
        self.register_buffer(
            "_fast_pad_valid",
            torch.tensor(pad_valid_mask_list, dtype=torch.float32).view(1, num_parts, max_tok, 1),
            persistent=False,
        )

        # Precompute stacked 3-layer MLP decoder weights when compatible (PartMLPDecoder).
        if self.fuse_mlp_decoder and self._supports_fused_mlp(part_names):
            self._init_fused_mlp_buffers(part_names, max_tok=max_tok, code_dim=code_dim)
        else:
            self.fuse_mlp_decoder = False

    def _supports_fused_mlp(self, part_names: list[str]) -> bool:
        for p_name in part_names:
            dec = self.decoders[p_name]
            net = getattr(dec, "net", None)
            if not isinstance(net, nn.Sequential) or len(net) < 7:
                return False
            if not (isinstance(net[0], nn.Linear) and isinstance(net[3], nn.Linear) and isinstance(net[6], nn.Linear)):
                return False
        hidden_dims = {int(self.decoders[p].net[0].out_features) for p in part_names}
        return len(hidden_dims) == 1

    def _init_fused_mlp_buffers(self, part_names: list[str], *, max_tok: int, code_dim: int) -> None:
        num_parts = len(part_names)
        max_out_dim = max(len(self.layout.part_joints[p]) * 3 for p in part_names)
        max_part_joints = max_out_dim // 3
        hidden_dim = int(self.decoders[part_names[0]].net[0].out_features)
        in_padded = max_tok * code_dim

        w1_list, b1_list, w2_list, b2_list, w3_list, b3_list = [], [], [], [], [], []
        merge_mat = torch.zeros((num_parts * max_part_joints, self.num_joints), dtype=torch.float32)
        joint_counts = torch.zeros((self.num_joints,), dtype=torch.float32)
        for p_name in part_names:
            for j_id in self.layout.part_joints[p_name]:
                joint_counts[int(j_id)] += 1.0

        for p_idx, p_name in enumerate(part_names):
            dec = self.decoders[p_name]
            l0, l3, l6 = dec.net[0], dec.net[3], dec.net[6]
            w0_pad = torch.zeros((hidden_dim, in_padded), dtype=l0.weight.dtype, device=l0.weight.device)
            w0_pad[:, : l0.weight.shape[1]] = l0.weight.detach()
            w1_list.append(w0_pad.t().contiguous())
            b1_list.append(l0.bias.detach().view(1, hidden_dim))

            w2_list.append(l3.weight.detach().t().contiguous())
            b2_list.append(l3.bias.detach().view(1, hidden_dim))

            w6_pad = torch.zeros((max_out_dim, hidden_dim), dtype=l6.weight.dtype, device=l6.weight.device)
            b6_pad = torch.zeros((max_out_dim,), dtype=l6.bias.dtype, device=l6.bias.device)
            w6_pad[: l6.weight.shape[0], :] = l6.weight.detach()
            b6_pad[: l6.bias.shape[0]] = l6.bias.detach()
            w3_list.append(w6_pad.t().contiguous())
            b3_list.append(b6_pad.view(1, max_out_dim))

            for local_j, global_j in enumerate(self.layout.part_joints[p_name]):
                if int(global_j) != int(self.layout.root_index):
                    row = p_idx * max_part_joints + local_j
                    merge_mat[row, int(global_j)] = 1.0 / max(float(joint_counts[int(global_j)]), 1.0)

        self._max_part_joints = max_part_joints
        self.register_buffer("_dec_w1", torch.stack(w1_list, dim=0).contiguous(), persistent=False)
        self.register_buffer("_dec_b1", torch.stack(b1_list, dim=0).contiguous(), persistent=False)
        self.register_buffer("_dec_w2", torch.stack(w2_list, dim=0).contiguous(), persistent=False)
        self.register_buffer("_dec_b2", torch.stack(b2_list, dim=0).contiguous(), persistent=False)
        self.register_buffer("_dec_w3", torch.stack(w3_list, dim=0).contiguous(), persistent=False)
        self.register_buffer("_dec_b3", torch.stack(b3_list, dim=0).contiguous(), persistent=False)
        self.register_buffer("_merge_mat", merge_mat.contiguous(), persistent=False)

    def quantize_tokens(
        self,
        continuous_tokens: torch.Tensor,
        apply_straight_through: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, _, token_dim = continuous_tokens.shape
        padded = continuous_tokens.index_select(1, self._fast_pad_idx)
        grouped = (
            padded.view(bsz, self._num_parts, self._max_tok, token_dim)
            .permute(1, 0, 2, 3)
            .reshape(self._num_parts, bsz * self._max_tok, token_dim)
        )

        x_sq = torch.sum(grouped ** 2, dim=-1, keepdim=True)
        if torch.is_autocast_enabled():
            ac_dtype = (
                torch.get_autocast_gpu_dtype()
                if grouped.is_cuda
                else torch.get_autocast_cpu_dtype()
            )
            prod = torch.bmm(grouped.to(ac_dtype), self._fast_cb_t.to(ac_dtype))
        else:
            prod = torch.bmm(grouped.float(), self._fast_cb_t)
        distances = x_sq + self._fast_cb_sq - 2.0 * prod
        idx_local = torch.argmin(distances, dim=-1)

        idx_padded = (
            idx_local.view(self._num_parts, bsz, self._max_tok)
            .permute(1, 0, 2)
            .reshape(bsz, self._num_parts * self._max_tok)
        )
        idx_unpadded = idx_padded.index_select(1, self._fast_unpad_idx)
        idx_global = idx_unpadded + self._fast_code_offsets

        tokens_q = F.embedding(idx_global, self._fast_cb_global)
        if apply_straight_through:
            tokens_q = continuous_tokens + (tokens_q - continuous_tokens).detach()
        return tokens_q, idx_global

    def decode(self, quantized_features: torch.Tensor) -> torch.Tensor:
        if not self.fuse_mlp_decoder:
            part_tokens = self.layout.split_part_tokens(quantized_features)
            part_outputs = {name: self.decoders[name](part_tokens[name]) for name in self.layout.part_names}
            return self.layout.merge_joints(part_outputs, force_root_zero=True)

        bsz, _, token_dim = quantized_features.shape
        padded = quantized_features.index_select(1, self._fast_pad_idx).view(
            bsz, self._num_parts, self._max_tok, token_dim
        )
        padded = padded * self._fast_pad_valid.to(dtype=padded.dtype)
        x = padded.permute(1, 0, 2, 3).reshape(self._num_parts, bsz, self._max_tok * token_dim)

        if torch.is_autocast_enabled():
            dt = torch.get_autocast_gpu_dtype() if x.is_cuda else torch.get_autocast_cpu_dtype()
            h1 = F.gelu(torch.baddbmm(self._dec_b1.to(dt), x.to(dt), self._dec_w1.to(dt)))
            h2 = F.gelu(torch.baddbmm(self._dec_b2.to(dt), h1, self._dec_w2.to(dt)))
            out_parts = torch.baddbmm(self._dec_b3.to(dt), h2, self._dec_w3.to(dt))
        else:
            x_f = x.float()
            h1 = F.gelu(torch.baddbmm(self._dec_b1, x_f, self._dec_w1))
            h2 = F.gelu(torch.baddbmm(self._dec_b2, h1, self._dec_w2))
            out_parts = torch.baddbmm(self._dec_b3, h2, self._dec_w3)

        joints_padded = out_parts.permute(1, 0, 2).reshape(bsz, self._num_parts * self._max_part_joints, 3)
        merged = torch.matmul(joints_padded.transpose(1, 2).float(), self._merge_mat).transpose(1, 2)
        return merged.to(dtype=out_parts.dtype)


class DeployableWave2BodyInferenceWrapper(nn.Module):
    """Thin tensor-in -> tensor-out deployable wrapper preserving exact end-to-end forward."""

    def __init__(self, model: AbsoluteRootHeadEstimatorModel) -> None:
        super().__init__()
        self.model = model
        self._skip_unused_logits = not model._uses_learned_logit_decode()

    def forward(self, mmwave_points: torch.Tensor) -> torch.Tensor:
        encoded = self.model.encode_mmwave_tokens(mmwave_points)
        mmwave_tokens = encoded["tokens"]
        mmwave_token_mask = encoded["mask"]
        centers = encoded.get("centers")

        if self._skip_unused_logits:
            pred_z_e = self.model.token_translator(mmwave_tokens, mask=mmwave_token_mask, return_logits=False)
            pred_logits = None
        else:
            pred_z_e, pred_logits = self.model._translate(mmwave_tokens, mmwave_token_mask)

        _, _, pred_pose_relative = self.model._decode_relative_pose(pred_z_e, pred_logits)
        point_root_stats = masked_point_root_stats(mmwave_points)
        pred_root, _, _ = self.model.root_head.forward_with_components(
            mmwave_tokens, mmwave_token_mask, centers, point_root_stats, None
        )
        return pred_pose_relative + pred_root.unsqueeze(1)


def prepare_deployable_model(
    model: AbsoluteRootHeadEstimatorModel,
    *,
    prune_encoder: bool = True,
    batch_part_vq: bool = True,
    fuse_mlp_decoder: bool = True,
    use_triton_fps: bool = True,
) -> DeployableWave2BodyInferenceWrapper:
    """Optimize a trained ``AbsoluteRootHeadEstimatorModel`` for deployment in place."""
    if prune_encoder:
        prune_body_encoder(model)

    if batch_part_vq and hasattr(model.vqvae, "layout") and not isinstance(model.vqvae, BatchedDeployablePartVQVAE):
        device = next(model.parameters()).device
        model.vqvae = BatchedDeployablePartVQVAE(
            model.vqvae,
            fuse_mlp_decoder=fuse_mlp_decoder,
        ).to(device)

    if hasattr(model.mmwave_tokenizer, "fps_backend"):
        if use_triton_fps and triton_fps_available():
            model.mmwave_tokenizer.fps_backend = "auto"
        else:
            model.mmwave_tokenizer.fps_backend = "torch"

    model.eval()
    return DeployableWave2BodyInferenceWrapper(model).eval()


__all__ = [
    "BatchedDeployablePartVQVAE",
    "DeployableWave2BodyInferenceWrapper",
    "prepare_deployable_model",
    "prune_body_encoder",
]
