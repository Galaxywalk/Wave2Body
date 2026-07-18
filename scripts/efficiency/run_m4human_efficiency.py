#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Orchestrate the current M4Human s2 efficiency reproduction pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wave2body.utils.paths import artifact_root  # noqa: E402


EFFICIENCY_DIR = REPO_ROOT / "scripts" / "efficiency"
PROFILE_TRAINING = EFFICIENCY_DIR / "profile_training_flops.py"
PROFILE_INFERENCE = EFFICIENCY_DIR / "profile_inference.py"

MMWAVE_CONFIG = REPO_ROOT / "wave2body/configs/mmwave_tokenizer/variants/m4human_s2_trainonly_mae_spatial_k96_fast.yaml"
BODY_TOKENIZER_CONFIG = (
    REPO_ROOT / "wave2body/configs/human_pose_tokenizer/stage_a_part_k96_amass_m4human_s2_trainonly_param_joints.yaml"
)
HPE_TOKEN_CONFIG = REPO_ROOT / "wave2body/configs/hpe_token/m4human_s2_k96_xattn.yaml"
ABSOLUTE_ROOT_CONFIG = (
    REPO_ROOT
    / "wave2body/configs/hpe_token/"
    "m4human_s2_k96_queryroot_pointiwmean_absolute_trainable_translator.yaml"
)
POINT_TRANSFORMER_CONFIG = (
    REPO_ROOT / "wave2body/configs/hpe_point_transformer/m4human_absolute_hist4_no_rootnorm_a100.yaml"
)

TRAINING_STAGE_ORDER = (
    "point_transformer",
    "mmwave_tokenizer",
    "human_pose_tokenizer_stage_a",
    "hpe_token",
    "absolute_root_head",
)
WAVE2BODY_STAGE_ORDER = (
    "mmwave_tokenizer",
    "human_pose_tokenizer_stage_a",
    "hpe_token",
    "absolute_root_head",
)
DISPLAY_NAMES = {
    "point_transformer": "PT/WiCompass",
    "mmwave_tokenizer": "mmWave Tokenizer",
    "human_pose_tokenizer_stage_a": "Body tokenizer",
    "hpe_token": "Radar-to-body translator",
    "absolute_root_head": "Absolute root head",
}


@dataclass(frozen=True)
class EfficiencyJob:
    name: str
    gpu: str
    command: list[str]
    output_json: Path
    log_path: Path
    batch_arg: str | None = None
    batch_candidates: tuple[int, ...] = ()


@dataclass(frozen=True)
class JobGroup:
    gpu: str
    jobs: tuple[EfficiencyJob, ...]


def parse_gpu_list(value: str) -> list[str]:
    gpus = [item.strip() for item in str(value).split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"--gpus contains duplicates: {value}")
    return gpus


def parse_batch_candidates(value: str) -> tuple[int, ...]:
    out = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not out or any(item <= 0 for item in out):
        raise ValueError("Batch candidates must be positive integers")
    return out


def parse_gpu_map(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    mapping: dict[str, str] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        key, sep, gpu = item.partition("=")
        if not sep or not key.strip() or not gpu.strip():
            raise ValueError("--gpu-map entries must look like key=gpu")
        mapping[key.strip()] = gpu.strip()
    return mapping


def _default_run_name() -> str:
    return f"m4human_s2_current_4090_profile128_gpu013_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def resolve_work_root(args: argparse.Namespace) -> Path:
    if args.work_dir:
        return Path(args.work_dir).expanduser()
    return artifact_root() / "experiments" / "efficiency" / "results" / args.run_name


def _python_cmd(script: Path) -> list[str]:
    return [sys.executable, str(script)]


def _common_training_args(args: argparse.Namespace, output_json: Path, work_dir: Path) -> list[str]:
    cmd = [
        "--device",
        args.device,
        "--profile-batches",
        str(args.profile_batches),
        "--warmup-batches",
        str(args.warmup_batches),
        "--num-workers",
        str(args.num_workers),
        "--work-dir",
        str(work_dir),
        "--output-json",
        str(output_json),
    ]
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.smoke:
        cmd.extend(["--epochs", "1", "--max-train-batches", "2", "--warmup-batches", "0"])
    if args.no_pretrained:
        cmd.append("--no-pretrained")
    return cmd


def _training_job(
    *,
    name: str,
    gpu: str,
    args: argparse.Namespace,
    work_root: Path,
    stage_args: list[str],
    batch_arg: str | None = None,
    batch_candidates: tuple[int, ...] = (),
) -> EfficiencyJob:
    stage_root = work_root / name
    output_json = stage_root / "training_flops_summary.json"
    command = _python_cmd(PROFILE_TRAINING) + stage_args + _common_training_args(args, output_json, stage_root)
    log_name = f"{name}.log" if not batch_candidates else f"{name}_batch_fallback.log"
    return EfficiencyJob(
        name=name,
        gpu=gpu,
        command=command,
        output_json=output_json,
        log_path=work_root / "logs" / log_name,
        batch_arg=batch_arg,
        batch_candidates=batch_candidates,
    )


def _inference_job(*, gpu: str, args: argparse.Namespace, work_root: Path) -> EfficiencyJob:
    stage_root = work_root / "inference"
    output_json = stage_root / "inference_efficiency_summary.json"
    command = _python_cmd(PROFILE_INFERENCE) + [
        "--methods",
        "both",
        "--point-transformer-config",
        str(POINT_TRANSFORMER_CONFIG),
        "--wave2body-config",
        str(ABSOLUTE_ROOT_CONFIG),
        "--split",
        "test",
        "--device",
        args.device,
        "--batch-size",
        str(args.inference_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--flops-batches",
        str(args.flops_batches),
        "--latency-batches",
        str(args.latency_batches),
        "--latency-repeats",
        str(args.latency_repeats),
        "--warmup-batches",
        str(args.latency_warmup_batches),
        "--flops-warmup-batches",
        str(args.flops_warmup_batches),
        "--single-sample-latency-batches",
        str(args.single_sample_latency_batches),
        "--single-sample-warmup-batches",
        str(args.single_sample_warmup_batches),
        "--single-sample-latency-repeats",
        str(args.single_sample_latency_repeats),
        "--work-dir",
        str(stage_root),
        "--output-json",
        str(output_json),
    ]
    if args.no_pretrained:
        command.append("--no-pretrained")
    if args.smoke:
        command[command.index("--flops-batches") + 1] = "1"
        command[command.index("--latency-batches") + 1] = "1"
        command[command.index("--latency-repeats") + 1] = "1"
        command[command.index("--warmup-batches") + 1] = "0"
        command[command.index("--single-sample-latency-batches") + 1] = "1"
        command[command.index("--single-sample-warmup-batches") + 1] = "0"
        command[command.index("--single-sample-latency-repeats") + 1] = "1"
    return EfficiencyJob(
        name="inference",
        gpu=gpu,
        command=command,
        output_json=output_json,
        log_path=work_root / "logs" / "inference.log",
    )


def build_job_groups(args: argparse.Namespace, work_root: Path) -> list[JobGroup]:
    gpus = parse_gpu_list(args.gpus)
    gpu_map = parse_gpu_map(args.gpu_map)

    def gpu_for(key: str, default_index: int) -> str:
        return gpu_map.get(key, gpus[min(default_index, len(gpus) - 1)])

    pt_gpu = gpu_for("pt", 0)
    radar_gpu = gpu_for("radar", 1)
    body_gpu = gpu_for("body", 2)
    inference_gpu = gpu_map.get("inference", body_gpu)
    candidates = parse_batch_candidates(args.pt_train_batch_candidates)

    pt_job = _training_job(
        name="pt_training_flops",
        gpu=pt_gpu,
        args=args,
        work_root=work_root,
        stage_args=[
            "--stages",
            "point_transformer",
            "--point-transformer-config",
            str(POINT_TRANSFORMER_CONFIG),
        ],
        batch_arg="--point-transformer-batch-size",
        batch_candidates=candidates,
    )
    mmwave_job = _training_job(
        name="mmwave_tokenizer_training_flops",
        gpu=radar_gpu,
        args=args,
        work_root=work_root,
        stage_args=[
            "--stages",
            "mmwave_tokenizer",
            "--mmwave-config",
            str(MMWAVE_CONFIG),
            "--mmwave-batch-size",
            str(args.wave2body_batch_size),
        ],
    )
    hpe_job = _training_job(
        name="hpe_token_training_flops",
        gpu=radar_gpu,
        args=args,
        work_root=work_root,
        stage_args=[
            "--stages",
            "hpe_token",
            "--hpe-token-config",
            str(HPE_TOKEN_CONFIG),
            "--hpe-token-batch-size",
            str(args.wave2body_batch_size),
        ],
    )
    body_job = _training_job(
        name="human_pose_tokenizer_stage_a_training_flops",
        gpu=body_gpu,
        args=args,
        work_root=work_root,
        stage_args=[
            "--stages",
            "human_pose_tokenizer_stage_a",
            "--human-pose-stage-a-config",
            str(BODY_TOKENIZER_CONFIG),
            "--human-pose-stage-a-batch-size",
            str(args.wave2body_batch_size),
        ],
    )
    absolute_job = _training_job(
        name="absolute_root_training_flops",
        gpu=body_gpu,
        args=args,
        work_root=work_root,
        stage_args=[
            "--stages",
            "absolute_root_head",
            "--absolute-root-config",
            str(ABSOLUTE_ROOT_CONFIG),
            "--absolute-root-batch-size",
            str(args.wave2body_batch_size),
        ],
    )
    inference_job = _inference_job(gpu=inference_gpu, args=args, work_root=work_root)

    groups_by_gpu: dict[str, list[EfficiencyJob]] = {}
    for job in (pt_job, mmwave_job, hpe_job, body_job, absolute_job, inference_job):
        groups_by_gpu.setdefault(job.gpu, []).append(job)
    return [JobGroup(gpu=gpu, jobs=tuple(jobs)) for gpu, jobs in groups_by_gpu.items()]


def _command_with_candidate(job: EfficiencyJob, batch_size: int | None) -> list[str]:
    command = list(job.command)
    if job.batch_arg and batch_size is not None:
        command.extend([job.batch_arg, str(batch_size)])
    return command


def _is_oom_log(text: str) -> bool:
    lowered = text.lower()
    return "out of memory" in lowered or "cublas_status_alloc_failed" in lowered or "cuda oom" in lowered


def run_job(job: EfficiencyJob, *, skip_existing: bool) -> dict[str, Any]:
    if skip_existing and job.output_json.exists():
        return {"job": job.name, "gpu": job.gpu, "status": "skipped", "output_json": str(job.output_json)}

    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    job.output_json.parent.mkdir(parents=True, exist_ok=True)
    candidates: tuple[int | None, ...] = job.batch_candidates or (None,)
    attempts = []
    for candidate in candidates:
        command = _command_with_candidate(job, candidate)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
        env.setdefault("PYTHONUNBUFFERED", "1")
        previous_log_size = job.log_path.stat().st_size if job.log_path.exists() else 0
        with job.log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] START {job.name} gpu={job.gpu}")
            if candidate is not None:
                log.write(f" batch={candidate}")
            log.write("\n")
            log.write(" ".join(command) + "\n")
            result = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            log.write(f"[{datetime.now(timezone.utc).isoformat()}] END rc={result.returncode}\n")
        attempts.append({"batch_size": candidate, "returncode": result.returncode})
        if result.returncode == 0:
            payload = {"job": job.name, "gpu": job.gpu, "status": "ok", "output_json": str(job.output_json), "attempts": attempts}
            if candidate is not None:
                payload["selected_batch_size"] = candidate
            return payload
        with job.log_path.open("r", encoding="utf-8", errors="replace") as log:
            log.seek(previous_log_size)
            log_text = log.read()
        if candidate is None or not _is_oom_log(log_text):
            break
    return {
        "job": job.name,
        "gpu": job.gpu,
        "status": "failed",
        "output_json": str(job.output_json),
        "log_path": str(job.log_path),
        "attempts": attempts,
    }


def run_group(group: JobGroup, *, skip_existing: bool) -> list[dict[str, Any]]:
    results = []
    for job in group.jobs:
        print(f"[Efficiency Pipeline] {job.name} on physical GPU {job.gpu}")
        result = run_job(job, skip_existing=skip_existing)
        results.append(result)
        if result["status"] == "failed":
            break
    return results


def _run_git(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _gpu_snapshot() -> list[dict[str, str]]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4:
            rows.append({"index": parts[0], "name": parts[1], "memory_total": parts[2], "driver": parts[3]})
    return rows


def write_preflight(args: argparse.Namespace, work_root: Path, groups: list[JobGroup]) -> Path:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preset": args.preset,
        "run_name": args.run_name,
        "work_root": str(work_root),
        "command": sys.argv,
        "git": {
            "commit": _run_git(["rev-parse", "HEAD"]),
            "status_porcelain": _run_git(["status", "--porcelain"]),
        },
        "environment": {
            "python": sys.version.replace("\n", " "),
            "python_executable": sys.executable,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "wave2body_artifact_root": os.environ.get("WAVE2BODY_ARTIFACT_ROOT"),
            "wave2body_workspace_root": os.environ.get("WAVE2BODY_WORKSPACE_ROOT"),
            "wave2body_dataset_paths": os.environ.get("WAVE2BODY_DATASET_PATHS"),
        },
        "gpus": _gpu_snapshot(),
        "job_groups": [
            {
                "gpu": group.gpu,
                "jobs": [
                    {
                        "name": job.name,
                        "output_json": str(job.output_json),
                        "log_path": str(job.log_path),
                        "batch_candidates": list(job.batch_candidates),
                        "command": job.command,
                    }
                    for job in group.jobs
                ],
            }
            for group in groups
        ],
        "configs": {
            "mmwave": str(MMWAVE_CONFIG),
            "body_tokenizer": str(BODY_TOKENIZER_CONFIG),
            "hpe_token": str(HPE_TOKEN_CONFIG),
            "absolute_root": str(ABSOLUTE_ROOT_CONFIG),
            "point_transformer": str(POINT_TRANSFORMER_CONFIG),
        },
    }
    path = work_root / "preflight.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_flops(value: float) -> str:
    abs_value = abs(float(value))
    for suffix, scale in (("EFLOPs", 1e18), ("PFLOPs", 1e15), ("TFLOPs", 1e12), ("GFLOPs", 1e9), ("MFLOPs", 1e6)):
        if abs_value >= scale:
            return f"{float(value) / scale:.3f} {suffix}"
    return f"{float(value):.3f} FLOPs"


def _training_row_from_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not payload.get("stages"):
        raise KeyError(f"{path} does not contain stage reports")
    stage_report = payload["stages"][0]
    table_row = payload["comparison_table"][0]
    flops = stage_report["flops"]
    params = stage_report["params"]
    stage = str(stage_report["stage"])
    total = float(table_row["Total Training FLOPs"])
    return {
        "stage": stage,
        "name": DISPLAY_NAMES.get(stage, stage),
        "kind": "baseline" if stage == "point_transformer" else "stage",
        "total_flops": total,
        "display_flops": _format_flops(total),
        "profile_batch": int(round(float(flops.get("avg_batch_samples", 0)))),
        "trainable_params": int(params["trainable"]),
        "total_params": int(params["total"]),
        "epochs": int(flops["epochs"]),
        "profile_samples": int(flops["profile_samples"]),
        "profile_batches": int(flops["profile_batches"]),
        "profile_coverage": flops.get("profile_train_batch_coverage"),
        "config_path": stage_report.get("config_path"),
        "summary_path": str(path),
    }


def _load_training_rows(work_root: Path) -> list[dict[str, Any]]:
    paths = {
        "point_transformer": work_root / "pt_training_flops" / "training_flops_summary.json",
        "mmwave_tokenizer": work_root / "mmwave_tokenizer_training_flops" / "training_flops_summary.json",
        "human_pose_tokenizer_stage_a": work_root
        / "human_pose_tokenizer_stage_a_training_flops"
        / "training_flops_summary.json",
        "hpe_token": work_root / "hpe_token_training_flops" / "training_flops_summary.json",
        "absolute_root_head": work_root / "absolute_root_training_flops" / "training_flops_summary.json",
    }
    rows = []
    for stage in TRAINING_STAGE_ORDER:
        path = paths[stage]
        if path.exists():
            rows.append(_training_row_from_summary(path))
    return rows


def _load_inference_rows(work_root: Path) -> list[dict[str, Any]]:
    path = work_root / "inference" / "inference_efficiency_summary.json"
    if not path.exists():
        return []
    payload = _read_json(path)
    rows = []
    for row in payload.get("comparison_table", []):
        rows.append({**row, "summary_path": str(path)})
    return rows


def write_training_csv(rows: list[dict[str, Any]], path: Path) -> None:
    wave_total = sum(float(row["total_flops"]) for row in rows if row["stage"] in WAVE2BODY_STAGE_ORDER)
    fieldnames = [
        "name",
        "kind",
        "total_flops_pflops",
        "display_flops",
        "profile_batch",
        "trainable_params_m",
        "total_params_m",
        "epochs",
        "profile_samples",
        "route_group",
        "note",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "name": row["name"],
                    "kind": row["kind"],
                    "total_flops_pflops": f"{float(row['total_flops']) / 1e15:.3f}",
                    "display_flops": row["display_flops"],
                    "profile_batch": row["profile_batch"],
                    "trainable_params_m": f"{int(row['trainable_params']) / 1e6:.2f}",
                    "total_params_m": f"{int(row['total_params']) / 1e6:.2f}",
                    "epochs": row["epochs"],
                    "profile_samples": row["profile_samples"],
                    "route_group": "profiled" if row["stage"] == "point_transformer" else "wave2body",
                    "note": "RTX4090 reproduction",
                }
            )
        if wave_total > 0:
            writer.writerow(
                {
                    "name": "Wave2Body",
                    "kind": "summary",
                    "total_flops_pflops": f"{wave_total / 1e15:.3f}",
                    "display_flops": _format_flops(wave_total),
                    "profile_batch": "",
                    "trainable_params_m": "",
                    "total_params_m": "",
                    "epochs": "",
                    "profile_samples": "",
                    "route_group": "complete",
                    "note": "current full-bodytoken route",
                }
            )


def write_inference_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "method",
        "params_m",
        "flops_per_sample",
        "latency_ms_per_sample",
        "fps",
        "latency_b1_ms_per_sample",
        "throughput_b192_fps",
        "peak_memory_b192_mib",
        "peak_reserved_memory_b192_mib",
        "batch_size",
        "split",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row["Method"],
                    "params_m": f"{int(row['Total Params']) / 1e6:.2f}",
                    "flops_per_sample": _format_flops(float(row["FLOPs / Sample"])),
                    "latency_ms_per_sample": f"{float(row['Latency ms / Sample']):.4f}",
                    "fps": f"{float(row['FPS']):.2f}",
                    "latency_b1_ms_per_sample": f"{float(row.get('Latency B1 ms / Sample', row['Latency ms / Sample'])):.4f}",
                    "throughput_b192_fps": f"{float(row.get('Throughput B192 FPS', row['FPS'])):.2f}",
                    "peak_memory_b192_mib": f"{float(row.get('Peak Memory B192 MiB', 0.0)):.1f}",
                    "peak_reserved_memory_b192_mib": f"{float(row.get('Peak Reserved Memory B192 MiB', 0.0)):.1f}",
                    "batch_size": row["Batch Size"],
                    "split": row["Split"],
                }
            )


def consolidate_results(work_root: Path, *, run_status: list[dict[str, Any]]) -> dict[str, Any]:
    training_rows = _load_training_rows(work_root)
    inference_rows = _load_inference_rows(work_root)
    pt_total = next((float(row["total_flops"]) for row in training_rows if row["stage"] == "point_transformer"), None)
    wave_total = sum(float(row["total_flops"]) for row in training_rows if row["stage"] in WAVE2BODY_STAGE_ORDER)
    supervised_total = sum(float(row["total_flops"]) for row in training_rows if row["stage"] in {"hpe_token", "absolute_root_head"})

    ratios = {
        "pt_over_wave2body_training_flops": None if not pt_total or not wave_total else float(pt_total / wave_total),
        "pt_over_wave2body_supervised_training_flops": None
        if not pt_total or not supervised_total
        else float(pt_total / supervised_total),
    }
    report = {
        "benchmark": "m4human_s2_current_efficiency",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "work_root": str(work_root),
        "run_status": run_status,
        "training_rows": training_rows,
        "inference_rows": inference_rows,
        "wave2body_total_training_flops": float(wave_total),
        "wave2body_total_training_flops_text": _format_flops(wave_total) if wave_total else None,
        "wave2body_supervised_training_flops": float(supervised_total),
        "wave2body_supervised_training_flops_text": _format_flops(supervised_total) if supervised_total else None,
        "point_transformer_training_flops": pt_total,
        "ratios": ratios,
    }
    out_json = work_root / "efficiency_consolidated_summary.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_training_csv(training_rows, work_root / "training_figure_source.csv")
    write_inference_csv(inference_rows, work_root / "inference_table_source.csv")
    write_markdown_summary(report, work_root / "efficiency_consolidated_summary.md")
    report["summary_path"] = str(out_json)
    return report


def write_markdown_summary(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# M4Human s2 Efficiency Reproduction",
        "",
        f"- Work root: `{report['work_root']}`",
        f"- Wave2Body total training FLOPs: `{report.get('wave2body_total_training_flops_text')}`",
    ]
    ratio = report.get("ratios", {}).get("pt_over_wave2body_training_flops")
    if ratio is not None:
        lines.append(f"- PT/WiCompass over Wave2Body training FLOPs: `{float(ratio):.2f}x`")
    supervised_ratio = report.get("ratios", {}).get("pt_over_wave2body_supervised_training_flops")
    if supervised_ratio is not None:
        lines.append(f"- PT/WiCompass over supervised Wave2Body stages: `{float(supervised_ratio):.2f}x`")
    lines.extend(["", "## Training FLOPs", "", "| Stage | FLOPs | Params | Epochs | Profile samples |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in report.get("training_rows", []):
        lines.append(
            f"| {row['name']} | {row['display_flops']} | "
            f"{int(row['trainable_params']):,}/{int(row['total_params']):,} | "
            f"{row['epochs']} | {row['profile_samples']:,} |"
        )
    if report.get("inference_rows"):
        lines.extend(
            [
                "",
                "## Inference",
                "",
                "| Method | FLOPs/sample | Latency B=1 | Throughput B=192 | Peak mem B=192 |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in report["inference_rows"]:
            lines.append(
                f"| {row['Method']} | {_format_flops(float(row['FLOPs / Sample']))} | "
                f"{float(row.get('Latency B1 ms / Sample', row['Latency ms / Sample'])):.4f} ms | "
                f"{float(row.get('Throughput B192 FPS', row['FPS'])):.2f} samples/s | "
                f"{float(row.get('Peak Memory B192 MiB', 0.0)):.1f} MiB |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="m4human_s2_current", choices=("m4human_s2_current",))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpus", default="0,1,3", help="Physical GPU ids available to the orchestrator.")
    parser.add_argument("--gpu-map", help="Optional mapping, e.g. pt=0,radar=1,body=3,inference=3.")
    parser.add_argument("--parallel", action="store_true", help="Run GPU job groups concurrently.")
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--work-dir", help="Explicit output root.")
    parser.add_argument("--profile-batches", default="128")
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--wave2body-batch-size", type=int, default=192)
    parser.add_argument("--pt-train-batch-candidates", default="96,64,48,32")
    parser.add_argument("--inference-batch-size", type=int, default=192)
    parser.add_argument("--flops-batches", default="128")
    parser.add_argument("--latency-batches", default="all")
    parser.add_argument("--latency-repeats", type=int, default=3)
    parser.add_argument("--latency-warmup-batches", type=int, default=5)
    parser.add_argument("--flops-warmup-batches", type=int, default=0)
    parser.add_argument("--single-sample-latency-batches", default="256")
    parser.add_argument("--single-sample-warmup-batches", type=int, default=32)
    parser.add_argument("--single-sample-latency-repeats", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing per-job summary JSON files.")
    parser.add_argument("--dry-run", action="store_true", help="Write preflight and planned commands without running jobs.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.max_parallel <= 0:
        raise ValueError("--max-parallel must be positive")
    if args.run_name is None:
        args.run_name = _default_run_name()
    work_root = resolve_work_root(args)
    (work_root / "logs").mkdir(parents=True, exist_ok=True)
    groups = build_job_groups(args, work_root)
    preflight = write_preflight(args, work_root, groups)
    print(f"[Efficiency Pipeline] Wrote preflight: {preflight}")
    if args.dry_run:
        print(f"[Efficiency Pipeline] Dry run only. Work root: {work_root}")
        return

    run_status: list[dict[str, Any]] = []
    if args.parallel:
        max_workers = min(int(args.max_parallel), len(groups))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(run_group, group, skip_existing=bool(args.skip_existing)) for group in groups]
            for future in as_completed(futures):
                run_status.extend(future.result())
    else:
        for group in groups:
            run_status.extend(run_group(group, skip_existing=bool(args.skip_existing)))

    status_path = work_root / "run_status.json"
    status_path.write_text(json.dumps(run_status, indent=2, ensure_ascii=False), encoding="utf-8")
    failures = [item for item in run_status if item["status"] == "failed"]
    if failures:
        print(f"[Efficiency Pipeline] Failures recorded in {status_path}")
        raise SystemExit(1)

    report = consolidate_results(work_root, run_status=run_status)
    print(f"[Efficiency Pipeline] Wrote consolidated summary: {report['summary_path']}")


if __name__ == "__main__":
    main()
