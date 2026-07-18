# -*- coding: utf-8 -*-
"""Reconstruction evaluation for human pose tokenizers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .sample_metrics import per_sample_p_mpjpe_mm
from ..dataset.human_body import create_dataset
from ..models.human_body_tokenizer import create_joint_tokenizer
from ..train.common import load_config


def load_model(config_path: str | Path, checkpoint_path: str | Path, device: torch.device):
    """Load a human pose tokenizer from a training config and checkpoint."""
    cfg = load_config(config_path)
    model = create_joint_tokenizer(cfg["model"]).float().to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def load_eval_config(eval_config_path: str | Path) -> dict[str, Any]:
    """Load reconstruction eval config with dataset URI expansion."""
    cfg = load_config(eval_config_path)
    cfg.setdefault("evaluation", {})
    return cfg


def _empty_part_accumulators(layout):
    return {
        name: {
            "l2_sum": 0.0,
            "l1_sum": 0.0,
            "joint_count": 0,
            "coord_count": 0,
            "used_codes": torch.zeros(layout.part_codebook_size[name], dtype=torch.bool),
        }
        for name in layout.part_names
    }


def _commitment_loss(model, joints: torch.Tensor) -> torch.Tensor:
    outputs = model.encode_with_intermediates(joints)
    return torch.mean((outputs["z_e"] - outputs["z_q"].detach()) ** 2)


def _batch_recon_loss(recon_loss_type: str, diff: torch.Tensor, abs_diff: torch.Tensor) -> torch.Tensor:
    if recon_loss_type == "mse":
        return torch.mean(diff * diff)
    if recon_loss_type == "l1":
        return torch.mean(abs_diff)
    raise ValueError(f"Unknown reconstruction loss type: {recon_loss_type}")


@torch.inference_mode()
def evaluate_dataset(
    *,
    model,
    dataset_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    cache_dir: str | Path,
    cache_build_workers: int | None,
    cache_build_chunk_size: int | None,
    compute_p_mpjpe: bool,
) -> dict[str, Any]:
    """Evaluate one joint-only dataset against a loaded human pose tokenizer."""
    dataset, _ = create_dataset(
        [dict(dataset_config)],
        int(training_config["model"]["num_joints"]),
        "cpu",
        use_cache=True,
        cache_dir=str(cache_dir),
        cache_build_workers=cache_build_workers,
        cache_build_chunk_size=cache_build_chunk_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(int(num_workers) > 0),
    )

    layout = model.layout
    num_joints = int(training_config["model"]["num_joints"])
    global_used = torch.zeros(int(model.token_class_num), dtype=torch.bool)
    part_acc = _empty_part_accumulators(layout)
    total_l2 = 0.0
    total_l1 = 0.0
    total_sq = 0.0
    total_joint_count = 0
    total_coord_count = 0
    recon_loss_sum = 0.0
    total_loss_sum = 0.0
    commitment_loss_sum = 0.0
    commitment_loss_count = 0
    loss_cfg = dict(training_config.get("loss", {}))
    recon_loss_type = str(loss_cfg.get("recon_loss_type", "mse")).strip().lower()
    recon_weight = float(loss_cfg.get("recon_weight", 1.0))
    commitment_weight = float(loss_cfg.get("commitment_weight", 0.25))
    p_mpjpe_sum = 0.0
    p_mpjpe_count = 0
    t0 = time.perf_counter()

    for joints, _label in tqdm(loader, desc=f"Evaluating {dataset_config['name']}"):
        joints = joints.to(device, non_blocking=True).float()
        recovered, indices_global, _ = model(joints=joints, train=False)
        recovered = recovered.float()
        diff = recovered - joints
        abs_diff = diff.abs()
        l2 = torch.norm(diff, dim=-1)

        bsz = int(joints.shape[0])
        total_l2 += float(l2.sum().item())
        total_l1 += float(abs_diff.sum().item())
        total_sq += float((diff * diff).sum().item())
        total_joint_count += bsz * num_joints
        total_coord_count += bsz * num_joints * 3
        batch_recon_loss = _batch_recon_loss(recon_loss_type, diff, abs_diff)
        batch_commitment = _commitment_loss(model, joints)
        batch_total_loss = recon_weight * batch_recon_loss + commitment_weight * batch_commitment
        commitment_loss_sum += float(batch_commitment.item()) * bsz
        commitment_loss_count += bsz
        recon_loss_sum += float(batch_recon_loss.item()) * bsz
        total_loss_sum += float(batch_total_loss.item()) * bsz

        idx_cpu = indices_global.detach().cpu()
        global_used[idx_cpu.reshape(-1)] = True
        local_by_part = layout.split_global_indices(idx_cpu)
        for name in layout.part_names:
            part_idx = torch.as_tensor(layout.part_joints[name], device=device, dtype=torch.long)
            part_l2 = l2.index_select(1, part_idx)
            part_abs = abs_diff.index_select(1, part_idx)
            part_acc[name]["l2_sum"] += float(part_l2.sum().item())
            part_acc[name]["l1_sum"] += float(part_abs.sum().item())
            part_acc[name]["joint_count"] += bsz * len(layout.part_joints[name])
            part_acc[name]["coord_count"] += bsz * len(layout.part_joints[name]) * 3
            part_acc[name]["used_codes"][local_by_part[name].reshape(-1)] = True

        if compute_p_mpjpe:
            p_vals = per_sample_p_mpjpe_mm(recovered.detach().cpu(), joints.detach().cpu())
            p_mpjpe_sum += float(p_vals.sum())
            p_mpjpe_count += int(p_vals.shape[0])

    num_samples = int(total_joint_count // num_joints)
    if num_samples == 0:
        raise RuntimeError(f"No samples processed for dataset {dataset_config['name']}")

    part_metrics: dict[str, dict[str, Any]] = {}
    for name, acc in part_acc.items():
        part_metrics[name] = {
            "mpjpe_l2_mm": acc["l2_sum"] / acc["joint_count"] * 1000.0,
            "l1_coord_mae_mm": acc["l1_sum"] / acc["coord_count"] * 1000.0,
            "l1_xyzsum_per_joint_mm": acc["l1_sum"] / acc["joint_count"] * 1000.0,
            "used_codes": int(acc["used_codes"].sum().item()),
            "codebook_size": int(layout.part_codebook_size[name]),
            "usage_ratio": float(acc["used_codes"].float().mean().item()),
        }

    return {
        "name": dataset_config["name"],
        "type": dataset_config["type"],
        "num_samples": num_samples,
        "mpjpe_l2_mm": total_l2 / total_joint_count * 1000.0,
        "l1_coord_mae_mm": total_l1 / total_coord_count * 1000.0,
        "l1_xyzsum_per_joint_mm": total_l1 / total_joint_count * 1000.0,
        "mse_m2": total_sq / total_coord_count,
        "rmse_coord_mm": (total_sq / total_coord_count) ** 0.5 * 1000.0,
        "recon_loss": recon_loss_sum / num_samples,
        "commitment_loss": commitment_loss_sum / commitment_loss_count,
        "total_loss": total_loss_sum / num_samples,
        "loss_reduction": "sample_weighted_batch_mean",
        "p_mpjpe_mm": (p_mpjpe_sum / p_mpjpe_count) if p_mpjpe_count else None,
        "used_codes": int(global_used.sum().item()),
        "codebook_size": int(model.token_class_num),
        "usage_ratio": float(global_used.float().mean().item()),
        "part_metrics": part_metrics,
        "elapsed_sec": time.perf_counter() - t0,
    }


def evaluate_checkpoint(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    eval_config_path: str | Path,
    device: torch.device,
    batch_size: int | None = None,
    num_workers: int | None = None,
    cache_dir: str | Path | None = None,
    compute_p_mpjpe: bool | None = None,
) -> dict[str, Any]:
    """Evaluate a checkpoint on all datasets from an eval config."""
    model, training_config = load_model(config_path, checkpoint_path, device)
    eval_config = load_eval_config(eval_config_path)
    eval_cfg = dict(eval_config.get("evaluation", {}))
    batch_size = int(batch_size if batch_size is not None else eval_cfg.get("batch_size", 8192))
    num_workers = int(num_workers if num_workers is not None else eval_cfg.get("num_workers", 0))
    cache_dir = Path(cache_dir if cache_dir is not None else eval_cfg.get("cache_dir", "./human_pose_tokenizer_eval_cache"))
    compute_p_mpjpe = bool(compute_p_mpjpe if compute_p_mpjpe is not None else eval_cfg.get("compute_p_mpjpe", True))
    cache_dir.mkdir(parents=True, exist_ok=True)

    results = [
        evaluate_dataset(
            model=model,
            dataset_config=dataset_cfg,
            training_config=training_config,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            cache_dir=cache_dir,
            cache_build_workers=eval_cfg.get("cache_build_workers"),
            cache_build_chunk_size=eval_cfg.get("cache_build_chunk_size"),
            compute_p_mpjpe=compute_p_mpjpe,
        )
        for dataset_cfg in eval_config["datasets"]
    ]
    return {
        "checkpoint": str(Path(checkpoint_path).expanduser()),
        "config": str(Path(config_path).expanduser()),
        "eval_config": str(Path(eval_config_path).expanduser()),
        "cache_dir": str(cache_dir),
        "datasets": results,
    }


def save_evaluation(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")
    return path


def print_summary(payload: Mapping[str, Any]) -> None:
    print("\nSample-Level Metrics")
    print("| Dataset | Samples | Total loss | Recon loss | Commit loss | MPJPE L2 mm | P-MPJPE mm | L1 coord mm | Codebook |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for item in payload["datasets"]:
        p = item["p_mpjpe_mm"]
        p_text = "-" if p is None else f"{p:.2f}"
        print(
            f"| {item['name']} | {item['num_samples']} | {item['total_loss']:.6f} | "
            f"{item['recon_loss']:.6f} | {item['commitment_loss']:.6f} | {item['mpjpe_l2_mm']:.2f} | "
            f"{p_text} | {item['l1_coord_mae_mm']:.2f} | "
            f"{item['used_codes']}/{item['codebook_size']} |"
        )

    print("\nPart-Level Metrics")
    print("| Dataset | Part | MPJPE L2 mm | L1 coord mm | Codebook |")
    print("|---|---|---:|---:|---:|")
    for item in payload["datasets"]:
        for part_name, part in item["part_metrics"].items():
            print(
                f"| {item['name']} | {part_name} | {part['mpjpe_l2_mm']:.2f} | "
                f"{part['l1_coord_mae_mm']:.2f} | {part['used_codes']}/{part['codebook_size']} |"
            )


__all__ = [
    "evaluate_dataset",
    "evaluate_checkpoint",
    "load_eval_config",
    "load_model",
    "print_summary",
    "save_evaluation",
]
