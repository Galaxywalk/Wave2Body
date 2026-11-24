#!/usr/bin/env python3
"""Unified diagnostic evaluator for Wave2Body and baseline checkpoints.

Computes:
- Core 4 metrics: abs_mpjpe_mm, root_error_mm, root_aligned_mpjpe_mm, pa_mpjpe_mm
- Anatomical geometry: mble_all_mm, mble_limbs_mm, mble_distal_mm,
  distal_bias_mm / distal_bias_rel_pct, lr_asym_err_mm
- Benchmark slices:
    * M4Human: In-Place, Sit-In-Place, Non-In-Place, Micro Average, Sparse Q20/Q10, Dense Q20
    * mmBody: All, 7 official scenarios, Sparse Q20/Q10, Dense Q20
- Quantization decoding modes (Wave2Body): post_vq_hard, pre_vq_continuous, soft_codebook_tau1
- Strictly causal & windowed temporal smoothness filters: causal_median3, causal_ema_0.7, etc.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wave2body.dataset.radar_pose.m4human_protocols import M4HUMAN_ACTION_GROUP_ORDER, action_indices_for_group
from wave2body.evaluate.diagnostics import (
    MMBODY_SCENARIOS,
    apply_temporal_filter,
    compute_anatomical_pose_metrics,
    compute_pre_cap_point_counts,
    compute_temporal_smoothness_metrics,
    extract_sequence_metadata,
    summarize_metric_vectors,
)
from wave2body.models.ablations import (
    AblationAbsoluteRootHeadEstimatorModel,
    NoBodyTokenPooledMLP,
    NoBodyTokenRelativeRootHeadMLP,
)
from wave2body.models.baselines import build_pose_baseline, forward_pose_baseline, move_pose_baseline_batch
from wave2body.models.hpe_token.absolute_root import AbsoluteRootHeadEstimatorModel
from wave2body.models.hpe_token.decoder import decode_part_token_logits, part_codebook_distance_logits
from wave2body.models.point_transformer.pose_baseline import PTDecoupledRootPoseModel, PTPoseModel
from wave2body.train.checkpoint import load_checkpoint, load_optional_component_states, load_state_dict_shape_safe
from wave2body.train.pipelines.absolute_root_head import _load_absolute_component_states
from wave2body.train.pose_baseline_utils import _decoded_joints_from_outputs
from wave2body.train.pt_pose_utils import materialize_lazy_modules
from wave2body.train.radar_pose_data import _build_loader, create_radar_pose_dataset
from wave2body.utils.paths import resolve_dataset_paths
from wave2body.utils.runtime_utils import autocast_context, configure_runtime


def load_eval_config(
    ckpt_path: Path,
    cfg_path: Path | None,
    *,
    num_workers: int,
    batch_size: int | None,
) -> dict[str, Any]:
    if cfg_path is not None and cfg_path.is_file():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    else:
        eff = ckpt_path.parent.parent / "effective_config.yaml"
        if eff.is_file():
            cfg = yaml.safe_load(eff.read_text(encoding="utf-8"))
        else:
            ck = load_checkpoint(ckpt_path, map_location="cpu")
            cfg = copy.deepcopy(ck["config"])
    cfg = resolve_dataset_paths(cfg)
    cfg.setdefault("m4human", {}).pop("scale", None)
    tcfg = cfg.setdefault("training", {})
    tcfg["compile"] = {"enabled": False}
    tcfg["num_workers"] = int(num_workers)
    tcfg["persistent_workers"] = False
    tcfg["prefetch_factor"] = 2
    if batch_size is not None:
        tcfg["batch_size"] = int(batch_size)
    if isinstance(cfg.get("data"), dict) and isinstance(cfg["data"].get("cache"), dict):
        cpath = Path(str(cfg["data"]["cache"].get("path", "")))
        if not (cpath.is_dir() and any(cpath.glob("*test*.pt"))):
            cfg["data"]["cache"]["mode"] = "none"
    return cfg


def detect_model_type(cfg: dict[str, Any], ckpt: dict[str, Any]) -> str:
    bname = str((cfg.get("baseline") or {}).get("name", "")).strip().lower()
    if bname in {"mmdiff", "mm_diff"}:
        return "mmdiff"
    if bname in {"mmmesh", "mm_mesh"}:
        return "mmmesh"
    pt_cfg = cfg.get("point_transformer")
    if isinstance(pt_cfg, dict) and ("model" in pt_cfg or bname in {"point_transformer", "pt"}):
        dec = pt_cfg.get("decoupled_root_head") or {}
        return "wicompass_decoupled" if bool(dec.get("enabled", False)) else "wicompass"
    stage = str(cfg.get("training", {}).get("stage", "")).lower()
    if "no_body_token" in stage or "regressor_state_dict" in ckpt:
        return "no_body_token"
    vq_sd = ckpt.get("vqvae_state_dict") or {}
    if any(k.endswith(".net.weight") for k in vq_sd):
        return "query_coord_head"
    return "wave2body"


def _close_dataset(ds: Any) -> None:
    if hasattr(ds, "close"):
        ds.close()
    elif getattr(ds, "_compact_store", None) is not None:
        ds._compact_store.close()


@torch.no_grad()
def run_inference(
    ckpt_path: Path,
    cfg: dict[str, Any],
    model_type: str,
    modes: list[str],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, Any]]:
    ds = create_radar_pose_dataset(cfg, split="test", device="cpu")
    seq_meta = extract_sequence_metadata(ds)
    loader = _build_loader(
        ds,
        training_cfg=cfg.get("training", {}),
        shuffle=False,
        drop_last=False,
        generator=None,
        worker_init_fn=None,
    )
    runtime = configure_runtime(cfg.get("training", {}), device, default_amp=True, log_prefix="[supp_eval]")
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    tcfg = cfg.get("training", {})

    preds: dict[str, list[torch.Tensor]] = {m: [] for m in modes}
    gts: list[torch.Tensor] = []

    try:
        if model_type in {"wave2body", "query_coord_head"}:
            vq_cfg = copy.deepcopy(cfg["vqvae"])
            if model_type == "query_coord_head":
                vq_cfg["decoder_type"] = "linear"
            bypass_vq = (model_type == "query_coord_head") or bool(
                tcfg.get("bypass_vq_quantization", vq_cfg.get("bypass_quantization", False))
            )
            mm_method = str(cfg["mmwave_tokenizer"].get("method", "mae")).strip().lower()
            mm_policy = str(cfg["mmwave_tokenizer"].get("patch_policy", "spatial_fps_knn")).strip().lower()
            if mm_method != "mae" or mm_policy != "spatial_fps_knn":
                model = AblationAbsoluteRootHeadEstimatorModel(
                    mmwave_tokenizer_config=cfg["mmwave_tokenizer"],
                    sensor_token_config=cfg.get("sensor_tokens", {}),
                    token_translator_config=cfg["token_translator"],
                    vqvae_config=vq_cfg,
                    root_head_config=cfg.get("root_head", {}),
                    device=str(device),
                    freeze_mmwave_tokenizer=True,
                    freeze_token_translator=True,
                    freeze_vqvae=True,
                )
            else:
                model = AbsoluteRootHeadEstimatorModel(
                    mmwave_tokenizer_config=cfg["mmwave_tokenizer"],
                    sensor_token_config=cfg.get("sensor_tokens", {}),
                    token_translator_config=cfg["token_translator"],
                    vqvae_config=vq_cfg,
                    root_head_config=cfg.get("root_head", {}),
                    device=str(device),
                    freeze_mmwave_tokenizer=True,
                    freeze_token_translator=True,
                    freeze_vqvae=True,
                    bypass_vq_quantization=bypass_vq,
                    pose_decode_cfg=tcfg.get("pose_decode"),
                )
            pretrained = cfg.get("pretrained", {})
            model.load_pretrained_weights(
                mmwave_tokenizer_path=pretrained.get("mmwave_tokenizer"),
                vqvae_path=pretrained.get("vqvae"),
                hpe_token_path=pretrained.get("hpe_token"),
                device=str(device),
            )
            _load_absolute_component_states(model, ckpt, prefix="[supp_eval]")
            model = model.to(device).eval()

            for batch in loader:
                pts = batch["mmwave_points"].to(device, non_blocking=True)
                gts.append(batch["target_poses_absolute"].float().cpu())
                with autocast_context(runtime):
                    outputs = model(pts, None)
                    pred_root = outputs["pred_root"].float().unsqueeze(1)
                    for mode in modes:
                        if mode in {"post_vq_hard", "default"}:
                            preds[mode].append((outputs["pred_pose_relative"].float() + pred_root).cpu())
                        elif mode == "pre_vq_continuous":
                            rel = model.vqvae.decode(outputs["pred_z_e"]).float()
                            preds[mode].append((rel + pred_root).cpu())
                        elif mode == "soft_codebook_tau1":
                            dist_logits = part_codebook_distance_logits(
                                vqvae=model.vqvae, pred_z_e=outputs["pred_z_e"], scale=1.0
                            )
                            soft_out = decode_part_token_logits(
                                vqvae=model.vqvae,
                                pred_logits=dist_logits,
                                mode="soft_expectation",
                                temperature=1.0,
                            )
                            preds[mode].append((soft_out.pose.float() + pred_root).cpu())
                        else:
                            raise ValueError(f"Unsupported mode={mode!r}")
            del model
        elif model_type == "no_body_token":
            model_cls = NoBodyTokenRelativeRootHeadMLP if "root_head_state_dict" in ckpt else NoBodyTokenPooledMLP
            model = model_cls(cfg).to(device)
            load_optional_component_states(model, ckpt, prefix="[supp_eval NoBody]")
            if "regressor_state_dict" in ckpt:
                model.regressor.load_state_dict(ckpt["regressor_state_dict"], strict=False)
            if "root_head_state_dict" in ckpt and hasattr(model, "root_head"):
                model.root_head.load_state_dict(ckpt["root_head_state_dict"], strict=False)
            model.eval()
            for batch in loader:
                pts = batch["mmwave_points"].to(device, non_blocking=True)
                gts.append(batch["target_poses_absolute"].float().cpu())
                with autocast_context(runtime):
                    out = model(pts)
                preds[modes[0]].append(out["pred_pose_absolute"].float().cpu())
            del model
        elif model_type in {"wicompass", "wicompass_decoupled"}:
            if model_type == "wicompass_decoupled":
                dec_cfg = dict(cfg["point_transformer"].get("decoupled_root_head") or {})
                model = PTDecoupledRootPoseModel(
                    cfg["point_transformer"], root_head_config=dec_cfg, device=str(device)
                ).to(device)
            else:
                model = PTPoseModel(cfg["point_transformer"], device=str(device)).to(device)
            materialize_lazy_modules(model, config=cfg, device=device, runtime=runtime)
            state = ckpt.get("model_state_dict", ckpt)
            load_state_dict_shape_safe(model, state, module_name="supp_eval_pt", strict=False)
            model.eval()
            for batch in loader:
                pts = batch["mmwave_points"].to(device, non_blocking=True)
                gts.append(batch.get("target_poses_absolute", batch["target_poses"]).float().cpu())
                with autocast_context(runtime):
                    out = model(pts)
                preds[modes[0]].append(out["pred_pose"].float().cpu())
            del model
        elif model_type in {"mmmesh", "mmdiff"}:
            bundle = build_pose_baseline(cfg, device=str(device))
            bundle.model.to(device)
            if model_type == "mmdiff":
                fmt = cfg.get("data_format", {})
                dummy = {
                    "mmwave_points": torch.zeros(
                        (1, int(fmt.get("n_points", 256)), int(fmt.get("radar_input_c", 4))), device=device
                    ),
                    "timesteps": torch.zeros((1,), device=device, dtype=torch.long),
                }
                forward_pose_baseline(bundle, dummy)
            state = ckpt.get("model_state_dict", ckpt)
            load_state_dict_shape_safe(bundle.model, state, module_name=f"supp_eval_{model_type}", strict=False)
            bundle.model.eval()
            for batch in loader:
                dev_b = move_pose_baseline_batch(batch, device)
                gts.append(batch["target_poses_absolute"].float().cpu())
                with autocast_context(runtime):
                    out = forward_pose_baseline(bundle, dev_b)
                pred = _decoded_joints_from_outputs(out)
                if pred is None:
                    pred = out["pred_pose"]
                preds[modes[0]].append(pred.float().cpu())
            del bundle
        else:
            raise ValueError(f"Unsupported model_type={model_type!r}")
    finally:
        _close_dataset(ds)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {m: torch.cat(v, dim=0) for m, v in preds.items()}, torch.cat(gts, dim=0), seq_meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    ap.add_argument("--output", required=True, help="Output JSON metrics path")
    ap.add_argument("--config", default=None, help="Optional YAML config override")
    ap.add_argument("--model-type", default="auto", help="Model type or 'auto'")
    ap.add_argument(
        "--modes",
        default="post_vq_hard",
        help="Comma-separated Wave2Body decode modes: post_vq_hard,pre_vq_continuous,soft_codebook_tau1",
    )
    ap.add_argument(
        "--temporal-filters",
        default="",
        help="Comma-separated filters on primary mode: causal_median3,causal_ema_0.7,median3,ema_0.7",
    )
    ap.add_argument("--pre-cap-cache", default=None, help="Optional .npy cache path for pre-cap valid point counts")
    ap.add_argument("--save-preds-npz", default=None, help="Optional .npz path to save predictions and metadata")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint).expanduser()
    cfg_path = Path(args.config).expanduser() if args.config else None
    cfg = load_eval_config(ckpt_path, cfg_path, num_workers=args.num_workers, batch_size=args.batch_size)
    ckpt_cpu = load_checkpoint(ckpt_path, map_location="cpu")
    model_type = detect_model_type(cfg, ckpt_cpu) if args.model_type == "auto" else args.model_type
    del ckpt_cpu

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if model_type not in {"wave2body", "query_coord_head"}:
        modes = [modes[0]]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    preds_by_mode, gt_abs, seq_meta = run_inference(ckpt_path, cfg, model_type, modes, device)

    pre_cap_cache = Path(args.pre_cap_cache).expanduser() if args.pre_cap_cache else None
    if pre_cap_cache is not None and pre_cap_cache.is_file():
        pre_cap_counts = np.load(pre_cap_cache)
    else:
        ds_cnt = create_radar_pose_dataset(copy.deepcopy(cfg), split="test", device="cpu")
        try:
            pre_cap_counts = compute_pre_cap_point_counts(ds_cnt, canon_gt=gt_abs, cache_path=pre_cap_cache)
        finally:
            _close_dataset(ds_cnt)

    q10 = float(np.quantile(pre_cap_counts, 0.10))
    q20 = float(np.quantile(pre_cap_counts, 0.20))
    q80 = float(np.quantile(pre_cap_counts, 0.80))
    mask_q10 = pre_cap_counts <= q10
    mask_q20 = pre_cap_counts <= q20
    mask_d20 = pre_cap_counts >= q80
    n = int(gt_abs.shape[0])
    all_mask = np.ones(n, dtype=bool)

    mode_results: dict[str, Any] = {}
    for m_name, pred_t in preds_by_mode.items():
        vecs = compute_anatomical_pose_metrics(pred_t, gt_abs, root_index=0)
        temp = compute_temporal_smoothness_metrics(pred_t, gt_abs, seq_meta, root_index=0)
        overall = summarize_metric_vectors(vecs, all_mask)
        overall.update(temp)
        slices: dict[str, Any] = {}
        if seq_meta["kind"] == "m4human":
            act_ids = seq_meta["action_ids"]
            slices["Micro Average"] = overall
            for grp_name in M4HUMAN_ACTION_GROUP_ORDER:
                gmask = np.isin(act_ids, sorted(action_indices_for_group(grp_name)))
                slices[grp_name] = summarize_metric_vectors(vecs, gmask)
                slices[f"{grp_name}__Sparse_Q20"] = summarize_metric_vectors(vecs, gmask & mask_q20)
        else:
            scenarios = seq_meta["scenarios"]
            slices["All"] = overall
            for sc in MMBODY_SCENARIOS:
                slices[sc] = summarize_metric_vectors(vecs, scenarios == sc)
        slices["Sparse Q20"] = summarize_metric_vectors(vecs, mask_q20)
        slices["Sparse Q10"] = summarize_metric_vectors(vecs, mask_q10)
        slices["Dense Q20"] = summarize_metric_vectors(vecs, mask_d20)
        mode_results[m_name] = {"overall": overall, "slices": slices}

    filter_results: dict[str, Any] = {}
    if args.temporal_filters:
        base_pred = preds_by_mode[modes[0]]
        for fname in [x.strip() for x in args.temporal_filters.split(",") if x.strip()]:
            fpred = apply_temporal_filter(base_pred, seq_meta, fname)
            fvecs = compute_anatomical_pose_metrics(fpred, gt_abs, root_index=0)
            ftemp = compute_temporal_smoothness_metrics(fpred, gt_abs, seq_meta, root_index=0)
            fover = summarize_metric_vectors(fvecs, all_mask)
            fover.update(ftemp)
            filter_results[fname] = fover

    if args.save_preds_npz:
        npz_path = Path(args.save_preds_npz).expanduser()
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict: dict[str, Any] = {
            "pred_abs": preds_by_mode[modes[0]].numpy(),
            "gt_abs": gt_abs.numpy(),
            "pre_cap_counts": pre_cap_counts,
        }
        for key, val in seq_meta.items():
            if key != "kind":
                save_dict[key] = val
        np.savez(npz_path, **save_dict)

    primary = mode_results[modes[0]]
    payload: dict[str, Any] = {
        "schema": "wave2body_diagnostics_v1",
        "checkpoint": str(ckpt_path),
        "model_type": model_type,
        "dataset": cfg.get("dataset", {}).get("name"),
        "split": cfg.get("m4human", {}).get("split") if seq_meta["kind"] == "m4human" else "official_test",
        "num_samples": n,
        "sparsity_quantiles": {"q10": q10, "q20": q20, "q80": q80},
        "eval_elapsed_sec": time.perf_counter() - t0,
        "primary_mode": modes[0],
        "overall": primary["overall"],
        "slices": primary["slices"],
        "modes": mode_results,
    }
    if filter_results:
        payload["temporal_filters"] = filter_results

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    ov = primary["overall"]
    print(
        f"[supp_eval] {model_type} ({payload['dataset']}/{payload['split']}, N={n}) | "
        f"abs={ov['abs_mpjpe_mm']:.2f}mm root={ov['root_error_mm']:.2f}mm "
        f"ra={ov['root_aligned_mpjpe_mm']:.2f}mm pa={ov['pa_mpjpe_mm']:.2f}mm "
        f"mble_dist={ov['mble_distal_mm']:.2f}mm mpjve={ov['mpjve_abs_mm_per_frame']:.2f}mm/f -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
