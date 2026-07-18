#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate a human pose tokenizer on the training config's held-out val split."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from scripts.human_pose_tokenizer._configs import resolve_config  # noqa: E402
from wave2body.dataset.human_body import create_dataset  # noqa: E402
from wave2body.evaluate.sample_metrics import per_sample_p_mpjpe_mm  # noqa: E402
from wave2body.models.human_body_tokenizer import JointVQVAELoss, create_joint_tokenizer  # noqa: E402
from wave2body.train.common import load_config, set_global_seed  # noqa: E402
from wave2body.train.human_pose_tokenizer import resolve_enabled_datasets  # noqa: E402
from wave2body.train.joint_batch_loader import JointTensorBatchLoader, pack_joints_dataset  # noqa: E402


def _label_names(dataset) -> dict[int, str]:
    datasets = list(getattr(dataset, "datasets", [dataset]))
    return {int(getattr(ds, "label", idx)): str(getattr(ds, "name", f"label_{idx}")) for idx, ds in enumerate(datasets)}


def _build_training_val_loader(config: dict[str, Any], *, batch_size: int | None):
    data_config = config.get("data", {})
    dataset_device = str(data_config.get("dataset_device", "cpu"))
    dataset, _ = create_dataset(
        resolve_enabled_datasets(config),
        int(config["model"]["num_joints"]),
        dataset_device,
        use_cache=bool(data_config.get("use_cache", True)),
        cache_dir=data_config.get("cache_dir"),
        cache_build_workers=data_config.get("cache_build_workers"),
        cache_build_chunk_size=data_config.get("cache_build_chunk_size"),
    )
    train_size = int(float(data_config["train_ratio"]) * len(dataset))
    val_size = len(dataset) - train_size
    if val_size <= 0:
        raise RuntimeError(f"Training config produced empty val split: len={len(dataset)}, train_size={train_size}")

    effective_batch_size = int(batch_size or config["training"]["batch_size"])
    pin_memory = bool(config.get("training", {}).get("pin_memory", True))
    batch_loader_mode = str(data_config.get("batch_loader", "dataloader")).strip().lower()
    names = _label_names(dataset)

    if batch_loader_mode in {"tensor", "tensor_batch", "joint_tensor"}:
        packed = pack_joints_dataset(dataset)
        if packed is None:
            raise RuntimeError("data.batch_loader=tensor is configured, but the dataset could not be packed.")
        split_indices = torch.randperm(len(packed))
        val_indices = split_indices[train_size:]
        loader = JointTensorBatchLoader(
            packed,
            val_indices,
            batch_size=effective_batch_size,
            shuffle=False,
            pin_memory=pin_memory,
        )
        return loader, names, {
            "num_samples_total": len(packed),
            "train_size": train_size,
            "val_size": val_size,
            "batch_loader": "joint_tensor_batch",
        }

    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    del train_ds
    num_workers = int(config.get("training", {}).get("num_workers", 0))
    loader = DataLoader(
        val_ds,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
    return loader, names, {
        "num_samples_total": len(dataset),
        "train_size": train_size,
        "val_size": val_size,
        "batch_loader": "dataloader",
    }


def _empty_acc() -> dict[str, float | int]:
    return {"l2_sum": 0.0, "l1_sum": 0.0, "sq_sum": 0.0, "joint_count": 0, "coord_count": 0, "sample_count": 0}


def _add_batch(acc: dict[str, float | int], diff: torch.Tensor, labels: torch.Tensor | None = None, label: int | None = None):
    if labels is not None and label is not None:
        mask = labels == label
        if not bool(mask.any()):
            return
        diff = diff[mask]
    l2 = torch.linalg.vector_norm(diff, dim=-1)
    abs_diff = diff.abs()
    acc["l2_sum"] = float(acc["l2_sum"]) + float(l2.sum().item())
    acc["l1_sum"] = float(acc["l1_sum"]) + float(abs_diff.sum().item())
    acc["sq_sum"] = float(acc["sq_sum"]) + float((diff * diff).sum().item())
    acc["joint_count"] = int(acc["joint_count"]) + int(diff.shape[0] * diff.shape[1])
    acc["coord_count"] = int(acc["coord_count"]) + int(diff.numel())
    acc["sample_count"] = int(acc["sample_count"]) + int(diff.shape[0])


def _finalize_acc(acc: dict[str, float | int]) -> dict[str, Any]:
    joint_count = max(1, int(acc["joint_count"]))
    coord_count = max(1, int(acc["coord_count"]))
    return {
        "num_samples": int(acc["sample_count"]),
        "mpjpe_l2_mm": float(acc["l2_sum"]) / joint_count * 1000.0,
        "l1_coord_mae_mm": float(acc["l1_sum"]) / coord_count * 1000.0,
        "l1_xyzsum_per_joint_mm": float(acc["l1_sum"]) / joint_count * 1000.0,
        "mse_m2": float(acc["sq_sum"]) / coord_count,
        "rmse_coord_mm": (float(acc["sq_sum"]) / coord_count) ** 0.5 * 1000.0,
    }


@torch.inference_mode()
def evaluate_training_val(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
    batch_size: int | None = None,
    compute_p_mpjpe: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    set_global_seed(int(config.get("seed", 42)), deterministic=bool(config.get("training", {}).get("deterministic", False)))
    loader, label_to_name, split_info = _build_training_val_loader(config, batch_size=batch_size)

    model = create_joint_tokenizer(config["model"]).float().to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    criterion = JointVQVAELoss(**config["loss"])

    overall = _empty_acc()
    by_label = {int(label): _empty_acc() for label in label_to_name}
    loss_sum = 0.0
    recon_loss_sum = 0.0
    commitment_loss_sum = 0.0
    loss_weight = 0
    p_mpjpe_sum = 0.0
    p_mpjpe_count = 0
    t0 = time.perf_counter()

    for joints, labels in loader:
        joints = joints.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()
        visible = torch.ones(joints.shape[:2], device=device, dtype=torch.bool)
        recovered, _indices, e_loss = model(joints=joints, joints_visible=visible, train=False)
        recovered = recovered.float()
        loss = criterion(recovered, joints, e_loss)
        bsz = int(joints.shape[0])
        loss_sum += float(loss["total_loss"].item()) * bsz
        recon_loss_sum += float(loss["recon_loss"].item()) * bsz
        commitment_loss_sum += float(loss.get("commitment_loss", torch.zeros((), device=device)).item()) * bsz
        loss_weight += bsz

        diff = recovered - joints
        _add_batch(overall, diff)
        for label in by_label:
            _add_batch(by_label[label], diff, labels=labels, label=label)
        if compute_p_mpjpe:
            vals = per_sample_p_mpjpe_mm(recovered.detach().cpu(), joints.detach().cpu())
            p_mpjpe_sum += float(vals.sum())
            p_mpjpe_count += int(vals.shape[0])

    denom = max(1, loss_weight)
    metrics = _finalize_acc(overall)
    metrics.update(
        {
            "total_loss": loss_sum / denom,
            "recon_loss": recon_loss_sum / denom,
            "commitment_loss": commitment_loss_sum / denom,
            "p_mpjpe_mm": p_mpjpe_sum / p_mpjpe_count if p_mpjpe_count else None,
            "elapsed_sec": time.perf_counter() - t0,
            "by_dataset": {
                label_to_name[label]: _finalize_acc(acc)
                for label, acc in sorted(by_label.items())
                if int(acc["sample_count"]) > 0
            },
        }
    )
    return {
        "config": str(Path(config_path).expanduser()),
        "checkpoint": str(Path(checkpoint_path).expanduser()),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_best_val_loss": ckpt.get("best_val_loss"),
        "split": split_info,
        "metrics": metrics,
    }


def _print_summary(payload: dict[str, Any]) -> None:
    metrics = payload["metrics"]
    print("\nTraining-Val Reconstruction")
    print(
        f"overall | samples={metrics['num_samples']} | loss={metrics['total_loss']:.6f} | "
        f"mpjpe={metrics['mpjpe_l2_mm']:.2f}mm | l1_coord={metrics['l1_coord_mae_mm']:.2f}mm"
    )
    print("\nBy Dataset")
    print("| Dataset | Samples | MPJPE L2 mm | L1 coord mm | RMSE coord mm |")
    print("|---|---:|---:|---:|---:|")
    for name, item in metrics["by_dataset"].items():
        print(
            f"| {name} | {item['num_samples']} | {item['mpjpe_l2_mm']:.2f} | "
            f"{item['l1_coord_mae_mm']:.2f} | {item['rmse_coord_mm']:.2f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Training config alias or YAML path.")
    parser.add_argument("--checkpoint", required=True, help="Tokenizer checkpoint.")
    parser.add_argument("--output", required=True, help="JSON output path.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override eval batch size.")
    parser.add_argument("--gpus", default="0", help='CUDA_VISIBLE_DEVICES value. Use "" for current environment.')
    parser.add_argument("--compute-p-mpjpe", action="store_true", help="Also compute Procrustes MPJPE.")
    args = parser.parse_args()

    if args.gpus is not None and args.gpus != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    payload = evaluate_training_val(
        config_path=resolve_config(args.config),
        checkpoint_path=Path(args.checkpoint).expanduser(),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size=args.batch_size,
        compute_p_mpjpe=bool(args.compute_p_mpjpe),
    )
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _print_summary(payload)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
