# -*- coding: utf-8 -*-
"""Historical HPE-token ablation losses.

These are kept for old top-k and candidate-reranker experiments. The canonical
two-stage HPE-token pipeline uses the simpler losses in hpe_token_training.py.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


ABLATION_LOSS_METRIC_KEYS = (
    "topk_token_loss",
    "topk_token_weighted_loss",
    "topk_token_valid_count",
    "topk_token_total_count",
    "topk_token_valid_fraction",
    "topk_token_top1_fraction",
    "candidate_reranker_loss",
    "candidate_reranker_weighted_loss",
    "candidate_reranker_valid_count",
    "candidate_reranker_total_count",
    "candidate_reranker_valid_fraction",
    "candidate_reranker_top1_fraction",
    "candidate_reranker_oracle_local_mpjpe_mm",
    "candidate_reranker_selected_local_mpjpe_mm",
)


ABLATION_LOSS_COUNT_KEYS = (
    "topk_token_valid_count",
    "topk_token_total_count",
    "candidate_reranker_valid_count",
    "candidate_reranker_total_count",
)


def _ablation_loss_config_from_training_config(training_cfg: dict) -> dict[str, Any]:
    return {
        "topk_token": _topk_token_ce_from_config(training_cfg),
        "candidate_reranker": _candidate_reranker_ce_from_config(training_cfg),
    }


def _topk_token_ce_from_config(training_cfg: dict) -> dict[str, Any] | None:
    cfg = training_cfg.get("topk_token_ce")
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return None
    weight = float(cfg.get("weight", 1.0))
    if weight <= 0:
        return None
    topk = int(cfg.get("topk", 10))
    if topk <= 0:
        raise ValueError(f"training.topk_token_ce.topk must be positive, got {topk}")
    temperature = float(cfg.get("temperature", 1.0))
    if temperature <= 0:
        raise ValueError(f"training.topk_token_ce.temperature must be positive, got {temperature}")
    return {
        "weight": weight,
        "topk": topk,
        "temperature": temperature,
        "min_valid_fraction": float(cfg.get("min_valid_fraction", 0.0)),
    }


def _part_local_topk_token_ce(
    *,
    pred_logits: torch.Tensor,
    target_indices: torch.Tensor,
    layout,
    topk: int,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    if pred_logits is None:
        raise ValueError("pred_logits is required for training.topk_token_ce")
    if pred_logits.dim() != 3:
        raise ValueError(f"pred_logits must be [B,T,C], got {tuple(pred_logits.shape)}")
    if target_indices.dim() != 2:
        raise ValueError(f"target_indices must be [B,T], got {tuple(target_indices.shape)}")
    if pred_logits.shape[:2] != target_indices.shape:
        raise ValueError(
            f"logit prefix {tuple(pred_logits.shape[:2])} does not match target shape {tuple(target_indices.shape)}"
        )

    target_indices = target_indices.long()
    losses: list[torch.Tensor] = []
    valid_total = pred_logits.new_zeros((), dtype=torch.float32)
    target_total = pred_logits.new_zeros((), dtype=torch.float32)
    top1_total = pred_logits.new_zeros((), dtype=torch.float32)
    tau = max(float(temperature), 1e-6)

    for part_name in layout.part_names:
        token_slice = layout.part_token_slices[part_name]
        offset = int(layout.part_codebook_offsets[part_name])
        codebook_size = int(layout.part_codebook_size[part_name])
        end = offset + codebook_size
        if offset < 0 or codebook_size <= 0 or end > int(pred_logits.shape[-1]):
            raise ValueError(
                f"Invalid codebook slice for {part_name}: offset={offset}, size={codebook_size}, "
                f"logits_dim={int(pred_logits.shape[-1])}"
            )

        logits_local = pred_logits[:, token_slice, offset:end].float()
        target_local = target_indices[:, token_slice] - offset
        target_valid = (target_local >= 0) & (target_local < codebook_size)
        if not bool(target_valid.any().item()):
            continue

        effective_k = min(int(topk), codebook_size)
        top_idx = torch.topk(logits_local, k=effective_k, dim=-1).indices
        target_in_topk = (top_idx == target_local.unsqueeze(-1)).any(dim=-1) & target_valid
        target_total = target_total + target_valid.float().sum()
        valid_total = valid_total + target_in_topk.float().sum()
        top1_total = top1_total + ((top_idx[..., 0] == target_local) & target_valid).float().sum()
        if not bool(target_in_topk.any().item()):
            continue

        top_logits = logits_local.gather(-1, top_idx) / tau
        target_pos = (top_idx == target_local.unsqueeze(-1)).float().argmax(dim=-1)
        losses.append(F.cross_entropy(top_logits[target_in_topk], target_pos[target_in_topk].long()))

    if losses:
        loss = torch.stack(losses).mean()
    else:
        loss = pred_logits.float().sum() * 0.0
    coverage = valid_total / torch.clamp(target_total, min=1.0)
    top1 = top1_total / torch.clamp(target_total, min=1.0)
    return {
        "topk_token_loss": loss,
        "topk_token_valid_count": valid_total,
        "topk_token_total_count": target_total,
        "topk_token_valid_fraction": coverage,
        "topk_token_top1_fraction": top1,
    }


def _apply_optional_topk_token_loss(
    loss_dict: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    *,
    layout,
    topk_token_cfg: dict[str, Any] | None,
) -> dict[str, torch.Tensor]:
    if topk_token_cfg is None:
        return loss_dict
    topk = _part_local_topk_token_ce(
        pred_logits=outputs["pred_logits"],
        target_indices=outputs["target_indices"],
        layout=layout,
        topk=int(topk_token_cfg["topk"]),
        temperature=float(topk_token_cfg["temperature"]),
    )
    merged = dict(loss_dict)
    weighted = float(topk_token_cfg["weight"]) * topk["topk_token_loss"]
    merged["total_loss"] = merged["total_loss"] + weighted
    merged["topk_token_weighted_loss"] = weighted
    merged.update(topk)
    return merged


def _candidate_reranker_ce_from_config(training_cfg: dict) -> dict[str, Any] | None:
    cfg = training_cfg.get("candidate_reranker_ce")
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return None
    weight = float(cfg.get("weight", 1.0))
    if weight <= 0:
        return None
    candidate_topk = int(cfg.get("candidate_topk", cfg.get("topk", 10)))
    if candidate_topk <= 0:
        raise ValueError(f"training.candidate_reranker_ce.candidate_topk must be positive, got {candidate_topk}")
    distance_scale = float(cfg.get("distance_scale", cfg.get("scale", 1.0)))
    if distance_scale <= 0:
        raise ValueError(f"training.candidate_reranker_ce.distance_scale must be > 0, got {distance_scale}")
    label_source = str(cfg.get("label_source", "target_token")).strip().lower()
    if label_source not in {"target_token", "token", "local_pose_oracle", "pose_oracle"}:
        raise ValueError(f"Unsupported training.candidate_reranker_ce.label_source: {label_source!r}")
    return {
        "weight": weight,
        "candidate_topk": candidate_topk,
        "distance_scale": distance_scale,
        "label_source": label_source,
    }


def _part_local_masked_token_ce(
    *,
    pred_logits: torch.Tensor,
    target_indices: torch.Tensor,
    layout,
    prefix: str,
) -> dict[str, torch.Tensor]:
    if pred_logits is None:
        raise ValueError("pred_logits is required for masked token CE")
    if pred_logits.dim() != 3:
        raise ValueError(f"pred_logits must be [B,T,C], got {tuple(pred_logits.shape)}")
    if target_indices.dim() != 2:
        raise ValueError(f"target_indices must be [B,T], got {tuple(target_indices.shape)}")
    if pred_logits.shape[:2] != target_indices.shape:
        raise ValueError(
            f"logit prefix {tuple(pred_logits.shape[:2])} does not match target shape {tuple(target_indices.shape)}"
        )

    target_indices = target_indices.long()
    rows: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    valid_total = pred_logits.new_zeros((), dtype=torch.float32)
    target_total = pred_logits.new_zeros((), dtype=torch.float32)
    top1_total = pred_logits.new_zeros((), dtype=torch.float32)

    for part_name in layout.part_names:
        token_slice = layout.part_token_slices[part_name]
        offset = int(layout.part_codebook_offsets[part_name])
        codebook_size = int(layout.part_codebook_size[part_name])
        end = offset + codebook_size
        if offset < 0 or codebook_size <= 0 or end > int(pred_logits.shape[-1]):
            raise ValueError(
                f"Invalid codebook slice for {part_name}: offset={offset}, size={codebook_size}, "
                f"logits_dim={int(pred_logits.shape[-1])}"
            )
        logits_local = pred_logits[:, token_slice, offset:end].float()
        target_local = target_indices[:, token_slice] - offset
        target_valid = (target_local >= 0) & (target_local < codebook_size)
        finite = torch.isfinite(logits_local)
        target_score_valid = finite.gather(-1, target_local.clamp(min=0, max=codebook_size - 1).unsqueeze(-1)).squeeze(-1)
        valid = target_valid & target_score_valid
        if bool(target_valid.any().item()):
            target_total = target_total + target_valid.float().sum()
        if bool(valid.any().item()):
            valid_total = valid_total + valid.float().sum()
            masked_logits = logits_local.masked_fill(~finite, -1.0e9)
            top1_total = top1_total + ((masked_logits.argmax(dim=-1) == target_local) & target_valid).float().sum()
            rows.append(masked_logits[valid])
            targets.append(target_local[valid].long())

    if rows:
        loss = F.cross_entropy(torch.cat(rows, dim=0), torch.cat(targets, dim=0))
    else:
        loss = pred_logits.float().sum() * 0.0
    coverage = valid_total / torch.clamp(target_total, min=1.0)
    top1 = top1_total / torch.clamp(target_total, min=1.0)
    return {
        f"{prefix}_loss": loss,
        f"{prefix}_valid_count": valid_total,
        f"{prefix}_total_count": target_total,
        f"{prefix}_valid_fraction": coverage,
        f"{prefix}_top1_fraction": top1,
    }


def _part_local_pose_oracle_candidate_ce(
    *,
    reranker_logits: torch.Tensor,
    proposal_logits: torch.Tensor,
    baseline_indices: torch.Tensor,
    target_pose: torch.Tensor,
    vqvae,
    candidate_topk: int,
    prefix: str,
) -> dict[str, torch.Tensor]:
    if reranker_logits.dim() != 3 or proposal_logits.dim() != 3:
        raise ValueError(
            f"reranker/proposal logits must be [B,T,C], got {tuple(reranker_logits.shape)} and "
            f"{tuple(proposal_logits.shape)}"
        )
    if reranker_logits.shape != proposal_logits.shape:
        raise ValueError(
            f"reranker/proposal logit shape mismatch: {tuple(reranker_logits.shape)} vs "
            f"{tuple(proposal_logits.shape)}"
        )
    if baseline_indices.dim() != 2:
        raise ValueError(f"baseline_indices must be [B,T], got {tuple(baseline_indices.shape)}")
    if baseline_indices.shape != reranker_logits.shape[:2]:
        raise ValueError(
            f"baseline index shape {tuple(baseline_indices.shape)} does not match logits "
            f"{tuple(reranker_logits.shape[:2])}"
        )
    if target_pose.dim() != 3:
        raise ValueError(f"target_pose must be [B,J,3], got {tuple(target_pose.shape)}")
    if candidate_topk <= 0:
        raise ValueError(f"candidate_topk must be positive, got {candidate_topk}")
    if not hasattr(vqvae, "layout") or not hasattr(vqvae, "quantizers") or not hasattr(vqvae, "decoders"):
        raise RuntimeError("Expected a part-level VQ-VAE with layout, quantizers, and decoders.")

    layout = vqvae.layout
    rows: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    valid_total = reranker_logits.new_zeros((), dtype=torch.float32)
    target_total = reranker_logits.new_zeros((), dtype=torch.float32)
    top1_total = reranker_logits.new_zeros((), dtype=torch.float32)
    oracle_error_total = reranker_logits.new_zeros((), dtype=torch.float32)
    selected_error_total = reranker_logits.new_zeros((), dtype=torch.float32)

    for part_name in layout.part_names:
        token_slice = layout.part_token_slices[part_name]
        offset = int(layout.part_codebook_offsets[part_name])
        codebook_size = int(layout.part_codebook_size[part_name])
        end = offset + codebook_size
        if offset < 0 or codebook_size <= 0 or end > int(reranker_logits.shape[-1]):
            raise ValueError(
                f"Invalid codebook slice for {part_name}: offset={offset}, size={codebook_size}, "
                f"logits_dim={int(reranker_logits.shape[-1])}"
            )
        effective_k = min(int(candidate_topk), codebook_size)
        proposal_local = proposal_logits[:, token_slice, offset:end].float().detach()
        score_local = reranker_logits[:, token_slice, offset:end].float()

        with torch.no_grad():
            top_idx = torch.topk(proposal_local, k=effective_k, dim=-1).indices
            baseline_local = (baseline_indices[:, token_slice].long() - offset).clamp(min=0, max=codebook_size - 1)
            codebook = vqvae.quantizers[part_name].codebook.to(device=reranker_logits.device, dtype=torch.float32)
            baseline_tokens = codebook[baseline_local]
            candidate_code = codebook[top_idx]
            bsz, part_tokens, topk = top_idx.shape
            token_dim = int(codebook.shape[-1])
            candidate_tokens = (
                baseline_tokens[:, None, None, :, :]
                .expand(bsz, part_tokens, topk, part_tokens, token_dim)
                .clone()
            )
            for slot_idx in range(part_tokens):
                candidate_tokens[:, slot_idx, :, slot_idx, :] = candidate_code[:, slot_idx, :, :]
            decoded = vqvae.decoders[part_name](
                candidate_tokens.reshape(bsz * part_tokens * topk, part_tokens, token_dim)
            )
            joint_idx = torch.as_tensor(layout.part_joints[part_name], device=target_pose.device, dtype=torch.long)
            target_part = target_pose.float().index_select(1, joint_idx)
            decoded = decoded.view(bsz, part_tokens, topk, target_part.shape[1], target_part.shape[2])
            local_error = torch.linalg.norm(
                decoded.float() - target_part[:, None, None, :, :],
                dim=-1,
            ).mean(dim=-1)
            labels = local_error.argmin(dim=-1)

        candidate_scores = score_local.gather(-1, top_idx)
        finite = torch.isfinite(candidate_scores)
        valid = finite.any(dim=-1)
        target_total = target_total + valid.numel()
        if bool(valid.any().item()):
            safe_scores = candidate_scores.masked_fill(~finite, -1.0e9)
            pred_label = safe_scores.argmax(dim=-1)
            valid_scores = safe_scores[valid]
            valid_labels = labels[valid].long()
            rows.append(valid_scores)
            targets.append(valid_labels)
            valid_total = valid_total + valid.float().sum()
            top1_total = top1_total + (pred_label[valid] == valid_labels).float().sum()
            oracle_error_total = oracle_error_total + local_error.gather(-1, labels.unsqueeze(-1)).squeeze(-1)[
                valid
            ].sum()
            selected_error_total = selected_error_total + local_error.gather(-1, pred_label.unsqueeze(-1)).squeeze(-1)[
                valid
            ].sum()

    if rows:
        loss = F.cross_entropy(torch.cat(rows, dim=0), torch.cat(targets, dim=0))
    else:
        loss = reranker_logits.float().sum() * 0.0
    coverage = valid_total / torch.clamp(target_total, min=1.0)
    top1 = top1_total / torch.clamp(valid_total, min=1.0)
    oracle_error = oracle_error_total / torch.clamp(valid_total, min=1.0)
    selected_error = selected_error_total / torch.clamp(valid_total, min=1.0)
    return {
        f"{prefix}_loss": loss,
        f"{prefix}_valid_count": valid_total,
        f"{prefix}_total_count": target_total,
        f"{prefix}_valid_fraction": coverage,
        f"{prefix}_top1_fraction": top1,
        f"{prefix}_oracle_local_mpjpe_mm": oracle_error * 1000.0,
        f"{prefix}_selected_local_mpjpe_mm": selected_error * 1000.0,
    }


def _apply_optional_candidate_reranker_loss(
    loss_dict: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    *,
    model,
    candidate_reranker_cfg: dict[str, Any] | None,
    target_poses: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if candidate_reranker_cfg is None:
        return loss_dict
    reranker_logits = model.candidate_rerank_logits(
        pred_z_e=outputs["pred_z_e"],
        pred_logits=outputs["pred_logits"],
        candidate_topk=int(candidate_reranker_cfg["candidate_topk"]),
        distance_scale=float(candidate_reranker_cfg["distance_scale"]),
    )
    label_source = str(candidate_reranker_cfg.get("label_source", "target_token")).strip().lower()
    if label_source in {"local_pose_oracle", "pose_oracle"}:
        if target_poses is None:
            raise ValueError("target_poses is required for local-pose-oracle candidate reranker CE")
        rerank = _part_local_pose_oracle_candidate_ce(
            reranker_logits=reranker_logits,
            proposal_logits=outputs["pred_logits"],
            baseline_indices=outputs["pred_indices"],
            target_pose=target_poses,
            vqvae=model.vqvae,
            candidate_topk=int(candidate_reranker_cfg["candidate_topk"]),
            prefix="candidate_reranker",
        )
    else:
        rerank = _part_local_masked_token_ce(
            pred_logits=reranker_logits,
            target_indices=outputs["target_indices"],
            layout=model.vqvae.layout,
            prefix="candidate_reranker",
        )
    merged = dict(loss_dict)
    weighted = float(candidate_reranker_cfg["weight"]) * rerank["candidate_reranker_loss"]
    merged["total_loss"] = merged["total_loss"] + weighted
    merged["candidate_reranker_weighted_loss"] = weighted
    merged.update(rerank)
    return merged


def _apply_optional_ablation_losses(
    loss_dict: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    *,
    model,
    layout,
    ablation_cfg: dict[str, Any],
    target_poses: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    loss_dict = _apply_optional_topk_token_loss(
        loss_dict,
        outputs,
        layout=layout,
        topk_token_cfg=ablation_cfg.get("topk_token"),
    )
    return _apply_optional_candidate_reranker_loss(
        loss_dict,
        outputs,
        model=model,
        candidate_reranker_cfg=ablation_cfg.get("candidate_reranker"),
        target_poses=target_poses,
    )
