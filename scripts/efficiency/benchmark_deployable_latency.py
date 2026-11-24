#!/usr/bin/env python3
"""Benchmark deployable multi-batch inference latency and numerical equivalence.

Compares deployable Wave2Body (training-only ``E_body`` pruned, Triton valid-only FPS,
and batched 5-part VQ lookup) against an optional PointTransformer/WiCompass baseline
across configurable batch sizes under both Eager and ``torch.compile(mode="reduce-overhead")``
execution modes, gated by a real-data numerical equivalence check.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wave2body.models.hpe_token.absolute_root import AbsoluteRootHeadEstimatorModel
from wave2body.models.hpe_token.deploy import (
    prepare_deployable_model,
    prune_body_encoder,
)
from wave2body.models.mmwave_tokenizer.grouping import (
    _batched_farthest_point_indices,
    triton_batched_farthest_point_indices,
    triton_fps_available,
    valid_point_mask,
)
from wave2body.models.point_transformer.pose_baseline import PTPoseModel
from wave2body.train.checkpoint import load_checkpoint, load_state_dict_shape_safe
from wave2body.train.pipelines.absolute_root_head import _load_absolute_component_states
from wave2body.train.pt_pose_utils import materialize_lazy_modules
from wave2body.train.radar_pose_data import _build_loader, create_radar_pose_dataset
from wave2body.utils.paths import resolve_dataset_paths
from wave2body.utils.runtime_utils import autocast_context, configure_runtime


class _WiCompassInferenceWrapper(nn.Module):
    def __init__(self, model: PTPoseModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, mmwave_points: torch.Tensor) -> torch.Tensor:
        return self.model(mmwave_points)["pred_pose"]


def _load_run_config(ckpt_path: Path, cfg_override: Path | None) -> dict[str, Any]:
    if cfg_override is not None and cfg_override.is_file():
        cfg = yaml.safe_load(cfg_override.read_text(encoding="utf-8"))
    else:
        eff = ckpt_path.parent.parent / "effective_config.yaml"
        if eff.is_file():
            cfg = yaml.safe_load(eff.read_text(encoding="utf-8"))
        else:
            ck = load_checkpoint(ckpt_path, map_location="cpu")
            cfg = copy.deepcopy(ck["config"])
    cfg = resolve_dataset_paths(cfg)
    cfg.setdefault("training", {})["compile"] = {"enabled": False}
    cfg["training"]["num_workers"] = 0
    return cfg


def _build_wave2body_deployable(
    ckpt_path: Path,
    cfg_override: Path | None,
    device: torch.device,
) -> tuple[AbsoluteRootHeadEstimatorModel, dict[str, Any], Any, int]:
    cfg = _load_run_config(ckpt_path, cfg_override)
    runtime = configure_runtime(cfg.get("training", {}), device, default_amp=True, log_prefix="[Latency W2B]")
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    tcfg = cfg.get("training", {})
    model = AbsoluteRootHeadEstimatorModel(
        mmwave_tokenizer_config=cfg["mmwave_tokenizer"],
        sensor_token_config=cfg.get("sensor_tokens", {}),
        token_translator_config=cfg["token_translator"],
        vqvae_config=cfg["vqvae"],
        root_head_config=cfg.get("root_head", {}),
        device=str(device),
        freeze_mmwave_tokenizer=True,
        freeze_token_translator=True,
        freeze_vqvae=True,
        bypass_vq_quantization=bool(tcfg.get("bypass_vq_quantization", cfg["vqvae"].get("bypass_quantization", False))),
        pose_decode_cfg=tcfg.get("pose_decode"),
    )
    pretrained = cfg.get("pretrained", {})
    model.load_pretrained_weights(
        mmwave_tokenizer_path=pretrained.get("mmwave_tokenizer"),
        vqvae_path=pretrained.get("vqvae"),
        hpe_token_path=pretrained.get("hpe_token"),
        device=str(device),
    )
    _load_absolute_component_states(model, ckpt, prefix="[Latency W2B]")
    model = model.to(device).eval()
    full_params = int(sum(p.numel() for p in model.parameters()))
    prune_body_encoder(model)
    return model, cfg, runtime, full_params


def _build_wicompass(
    ckpt_path: Path,
    cfg_override: Path | None,
    device: torch.device,
) -> tuple[PTPoseModel, dict[str, Any], Any]:
    cfg = _load_run_config(ckpt_path, cfg_override)
    runtime = configure_runtime(cfg.get("training", {}), device, default_amp=True, log_prefix="[Latency PT]")
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    model = PTPoseModel(cfg["point_transformer"], device=str(device)).to(device)
    materialize_lazy_modules(model, config=cfg, device=device, runtime=runtime)
    load_state_dict_shape_safe(model, ckpt.get("model_state_dict", ckpt), module_name="LatencyPT", strict=False)
    model.eval()
    return model, cfg, runtime


def _load_gpu_sample_pool(cfg: dict[str, Any], device: torch.device, pool_samples: int = 1536) -> torch.Tensor:
    ds = create_radar_pose_dataset(cfg, split="test", device="cpu")
    loader = _build_loader(
        ds,
        training_cfg={**cfg.get("training", {}), "batch_size": 256, "num_workers": 0},
        shuffle=False,
        drop_last=False,
        generator=None,
        worker_init_fn=None,
    )
    chunks = []
    total = 0
    try:
        for batch in loader:
            chunks.append(batch["mmwave_points"].float())
            total += int(batch["mmwave_points"].shape[0])
            if total >= pool_samples:
                break
    finally:
        if hasattr(ds, "close"):
            ds.close()
    return torch.cat(chunks, dim=0)[:pool_samples].contiguous().to(device)


@torch.no_grad()
def verify_numerical_equivalence(
    ckpt_path: Path,
    cfg_override: Path | None,
    device: torch.device,
    *,
    num_samples: int = 128,
) -> tuple[dict[str, Any], bool]:
    model_ref, cfg, runtime, _ = _build_wave2body_deployable(ckpt_path, cfg_override, device)
    ds = create_radar_pose_dataset(cfg, split="test", device="cpu")
    loader = _build_loader(
        ds,
        training_cfg={**cfg.get("training", {}), "batch_size": num_samples, "num_workers": 0},
        shuffle=False,
        drop_last=False,
        generator=None,
        worker_init_fn=None,
    )
    try:
        batch = next(iter(loader))
    finally:
        if hasattr(ds, "close"):
            ds.close()

    pts = batch["mmwave_points"].to(device)
    gt = batch["target_poses_absolute"].to(device).float()

    vm = valid_point_mask(pts)
    xyz = pts[:, :, :3]
    num_groups = int(cfg["mmwave_tokenizer"].get("num_groups", 96))
    ref_fps_idx, ref_gm = _batched_farthest_point_indices(xyz, vm, num_groups)
    if triton_fps_available() and device.type == "cuda":
        opt_fps_idx, opt_gm = triton_batched_farthest_point_indices(xyz, vm, num_groups)
    else:
        opt_fps_idx, opt_gm = ref_fps_idx, ref_gm
    fps_match_rate = float((ref_fps_idx == opt_fps_idx).float().mean().item())
    gm_match_rate = float((ref_gm == opt_gm).float().mean().item())

    with autocast_context(runtime):
        ref_out_amp = model_ref(pts, None)
    ref_pose_amp = ref_out_amp["pred_pose_absolute"].float()
    ref_vq_amp = ref_out_amp["pred_indices"]

    ref_out_fp32 = model_ref(pts.float(), None)
    ref_pose_fp32 = ref_out_fp32["pred_pose_absolute"].float()
    ref_vq_fp32 = ref_out_fp32["pred_indices"]

    candidates: dict[str, Any] = {}
    for fuse_mlp in (True, False):
        model_opt, _, _, _ = _build_wave2body_deployable(ckpt_path, cfg_override, device)
        wrapper_opt = prepare_deployable_model(
            model_opt,
            prune_encoder=True,
            batch_part_vq=True,
            fuse_mlp_decoder=fuse_mlp,
            use_triton_fps=True,
        ).to(device)

        with autocast_context(runtime):
            _, opt_vq_amp = model_opt.quantize_tokens(ref_out_amp["pred_z_e"])
            opt_pose_amp = wrapper_opt(pts).float()
        _, opt_vq_fp32 = model_opt.quantize_tokens(ref_out_fp32["pred_z_e"])
        opt_pose_fp32 = wrapper_opt(pts.float()).float()

        diff_amp = (opt_pose_amp - ref_pose_amp).abs()
        diff_fp32 = (opt_pose_fp32 - ref_pose_fp32).abs()
        candidates[f"fuse_mlp_{fuse_mlp}"] = {
            "fuse_mlp_decoder": fuse_mlp,
            "params_deployable": int(sum(p.numel() for p in model_opt.parameters())),
            "fps_index_exact_match_rate": fps_match_rate,
            "fps_group_mask_exact_match_rate": gm_match_rate,
            "vq_index_exact_match_rate_amp": float((ref_vq_amp == opt_vq_amp).float().mean().item()),
            "vq_index_exact_match_rate_fp32": float((ref_vq_fp32 == opt_vq_fp32).float().mean().item()),
            "amp": {
                "max_abs_diff_mm": float(diff_amp.max().item() * 1000.0),
                "mean_abs_diff_mm": float(diff_amp.mean().item() * 1000.0),
                "ref_mpjpe_mm": float((torch.linalg.norm(ref_pose_amp - gt, dim=-1).mean() * 1000.0).item()),
                "opt_mpjpe_mm": float((torch.linalg.norm(opt_pose_amp - gt, dim=-1).mean() * 1000.0).item()),
            },
            "fp32": {
                "max_abs_diff_mm": float(diff_fp32.max().item() * 1000.0),
                "mean_abs_diff_mm": float(diff_fp32.mean().item() * 1000.0),
                "ref_mpjpe_mm": float((torch.linalg.norm(ref_pose_fp32 - gt, dim=-1).mean() * 1000.0).item()),
                "opt_mpjpe_mm": float((torch.linalg.norm(opt_pose_fp32 - gt, dim=-1).mean() * 1000.0).item()),
            },
        }
        del model_opt, wrapper_opt

    del model_ref
    if device.type == "cuda":
        torch.cuda.empty_cache()

    c_true = candidates["fuse_mlp_True"]
    chosen_fuse_mlp = (
        c_true["amp"]["max_abs_diff_mm"] < 0.1
        and abs(c_true["amp"]["opt_mpjpe_mm"] - c_true["amp"]["ref_mpjpe_mm"]) < 0.01
    )
    selected = candidates[f"fuse_mlp_{chosen_fuse_mlp}"]
    if selected["amp"]["max_abs_diff_mm"] >= 0.1:
        raise AssertionError(f"Numerical equivalence gate failed: {selected}")
    return {"chosen_fuse_mlp_decoder": chosen_fuse_mlp, "selected": selected, "candidates": candidates}, chosen_fuse_mlp


@torch.no_grad()
def measure_latency_and_memory(
    fn,
    pts_pool: torch.Tensor,
    batch_size: int,
    runtime: Any,
    device: torch.device,
    *,
    warmup_iters: int = 50,
    timed_iters: int = 200,
    use_cudagraph_step_mark: bool = False,
) -> dict[str, float]:
    n_pool = int(pts_pool.shape[0])
    num_slices = max(1, n_pool // batch_size)
    slices = [pts_pool[i * batch_size : (i + 1) * batch_size].contiguous() for i in range(num_slices)]

    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with autocast_context(runtime):
        for step in range(warmup_iters):
            if use_cudagraph_step_mark and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            _ = fn(slices[step % num_slices]).shape
    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    step_ms: list[float] = []
    with autocast_context(runtime):
        for step in range(timed_iters):
            x = slices[step % num_slices]
            if is_cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            if use_cudagraph_step_mark and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            _ = fn(x)
            if is_cuda:
                torch.cuda.synchronize(device)
            step_ms.append((time.perf_counter() - t0) * 1000.0)

    peak_alloc_mib = float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)) if is_cuda else 0.0
    peak_reserved_mib = float(torch.cuda.max_memory_reserved(device) / (1024.0 ** 2)) if is_cuda else 0.0
    arr = np.asarray(step_ms, dtype=np.float64)
    mean_batch_ms = float(arr.mean())
    return {
        "batch_latency_ms": mean_batch_ms,
        "batch_latency_std_ms": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "batch_latency_p50_ms": float(np.percentile(arr, 50)),
        "batch_latency_p95_ms": float(np.percentile(arr, 95)),
        "ms_per_sample": mean_batch_ms / float(batch_size),
        "throughput_fps": float(batch_size) * 1000.0 / mean_batch_ms,
        "peak_alloc_mib": peak_alloc_mib,
        "peak_reserved_mib": peak_reserved_mib,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wave2body-checkpoint", required=True, help="Wave2Body Stage-3 Phase-2 checkpoint (.pth)")
    ap.add_argument("--wave2body-config", default=None, help="Optional YAML override for Wave2Body")
    ap.add_argument("--wicompass-checkpoint", default=None, help="Optional WiCompass/PointTransformer checkpoint (.pth)")
    ap.add_argument("--wicompass-config", default=None, help="Optional YAML override for WiCompass")
    ap.add_argument("--output-dir", required=True, help="Directory for CSV and JSON benchmark outputs")
    ap.add_argument("--batch-sizes", default="1,8,32,192", help="Comma-separated batch sizes")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--pool-samples", type=int, default=1536)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-compile", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]

    w2b_ckpt = Path(args.wave2body_checkpoint).expanduser()
    w2b_cfg_override = Path(args.wave2body_config).expanduser() if args.wave2body_config else None

    equiv_report, chosen_fuse_mlp = verify_numerical_equivalence(w2b_ckpt, w2b_cfg_override, device)
    w2b_model, w2b_cfg, w2b_runtime, w2b_full_params = _build_wave2body_deployable(w2b_ckpt, w2b_cfg_override, device)
    w2b_deploy_params = int(sum(p.numel() for p in w2b_model.parameters()))
    pts_pool_w2b = _load_gpu_sample_pool(w2b_cfg, device, pool_samples=max(args.pool_samples, max(batch_sizes)))

    w2b_eager = prepare_deployable_model(
        w2b_model,
        prune_encoder=True,
        batch_part_vq=True,
        fuse_mlp_decoder=chosen_fuse_mlp,
        use_triton_fps=True,
    ).to(device)

    csv_rows: list[dict[str, Any]] = []
    for bsz in batch_sizes:
        m = measure_latency_and_memory(
            w2b_eager, pts_pool_w2b, bsz, w2b_runtime, device, warmup_iters=args.warmup, timed_iters=args.iters
        )
        csv_rows.append(
            {
                "device_name": device_name,
                "execution_mode": "Eager + Native GPU Ops",
                "method": "Wave2Body (deployable)",
                "params_total": w2b_deploy_params,
                "batch_size": bsz,
                **m,
            }
        )
        print(f"[Eager W2B] B={bsz}: {m['batch_latency_ms']:.3f} ms ({m['throughput_fps']:.1f} FPS, {m['peak_alloc_mib']:.1f} MiB)")

    pt_eager = None
    pt_runtime = None
    pts_pool_pt = None
    pt_params = 0
    if args.wicompass_checkpoint:
        pt_ckpt = Path(args.wicompass_checkpoint).expanduser()
        pt_cfg_override = Path(args.wicompass_config).expanduser() if args.wicompass_config else None
        pt_model, pt_cfg, pt_runtime = _build_wicompass(pt_ckpt, pt_cfg_override, device)
        pt_params = int(sum(p.numel() for p in pt_model.parameters()))
        pts_pool_pt = _load_gpu_sample_pool(pt_cfg, device, pool_samples=max(args.pool_samples, max(batch_sizes)))
        pt_eager = _WiCompassInferenceWrapper(pt_model).to(device).eval()
        for bsz in batch_sizes:
            m = measure_latency_and_memory(
                pt_eager, pts_pool_pt, bsz, pt_runtime, device, warmup_iters=args.warmup, timed_iters=args.iters
            )
            csv_rows.append(
                {
                    "device_name": device_name,
                    "execution_mode": "Eager + Native GPU Ops",
                    "method": "WiCompass (PT)",
                    "params_total": pt_params,
                    "batch_size": bsz,
                    **m,
                }
            )
            print(f"[Eager PT ] B={bsz}: {m['batch_latency_ms']:.3f} ms ({m['throughput_fps']:.1f} FPS, {m['peak_alloc_mib']:.1f} MiB)")

    if not args.skip_compile and hasattr(torch, "compile") and device.type == "cuda":
        for bsz in batch_sizes:
            torch._dynamo.reset()
            w2b_comp = torch.compile(w2b_eager, mode="reduce-overhead")
            m = measure_latency_and_memory(
                w2b_comp,
                pts_pool_w2b,
                bsz,
                w2b_runtime,
                device,
                warmup_iters=args.warmup,
                timed_iters=args.iters,
                use_cudagraph_step_mark=True,
            )
            csv_rows.append(
                {
                    "device_name": device_name,
                    "execution_mode": "torch.compile(reduce-overhead) + Native GPU Ops",
                    "method": "Wave2Body (deployable)",
                    "params_total": w2b_deploy_params,
                    "batch_size": bsz,
                    **m,
                }
            )
            print(f"[Comp  W2B] B={bsz}: {m['batch_latency_ms']:.3f} ms ({m['throughput_fps']:.1f} FPS, {m['peak_alloc_mib']:.1f} MiB)")
            del w2b_comp
            torch.cuda.empty_cache()

        if pt_eager is not None and pts_pool_pt is not None and pt_runtime is not None:
            for bsz in batch_sizes:
                torch._dynamo.reset()
                pt_comp = torch.compile(pt_eager, mode="reduce-overhead")
                m = measure_latency_and_memory(
                    pt_comp,
                    pts_pool_pt,
                    bsz,
                    pt_runtime,
                    device,
                    warmup_iters=args.warmup,
                    timed_iters=args.iters,
                    use_cudagraph_step_mark=True,
                )
                csv_rows.append(
                    {
                        "device_name": device_name,
                        "execution_mode": "torch.compile(reduce-overhead) + Native GPU Ops",
                        "method": "WiCompass (PT)",
                        "params_total": pt_params,
                        "batch_size": bsz,
                        **m,
                    }
                )
                print(f"[Comp  PT ] B={bsz}: {m['batch_latency_ms']:.3f} ms ({m['throughput_fps']:.1f} FPS, {m['peak_alloc_mib']:.1f} MiB)")
                del pt_comp
                torch.cuda.empty_cache()

    csv_path = out_dir / "deployable_latency.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    audit_path = out_dir / "deployable_latency_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema": "wave2body_deployable_latency_v1",
                "device_name": device_name,
                "torch_version": torch.__version__,
                "triton_fps_available": triton_fps_available(),
                "parameter_audit": {
                    "wave2body_full_params": w2b_full_params,
                    "wave2body_deployable_params": w2b_deploy_params,
                    "wicompass_params": pt_params,
                },
                "numerical_equivalence": equiv_report,
                "rows": csv_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[Latency DONE] Wrote {csv_path} and {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
