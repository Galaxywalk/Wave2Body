#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run 3-epoch efficiency benchmarks for Wave2Body vs PointTransformer."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wave2body.train.common import load_config  # noqa: E402
from wave2body.train.pipelines.absolute_root_head import run_absolute_root_head  # noqa: E402
from wave2body.train.pipelines.hpe_token import run_hpe_token  # noqa: E402
from wave2body.train.pipelines.pt_pose import run_pt_pose  # noqa: E402
from wave2body.utils.paths import artifact_root  # noqa: E402


DEFAULT_WAVE2BODY_CONFIG = (
    REPO_ROOT
    / "wave2body"
    / "configs"
    / "hpe_token"
    / "m4human_s2_k96_queryroot_pointiwmean_absolute_trainable_translator.yaml"
)
DEFAULT_POINT_TRANSFORMER_CONFIG = (
    REPO_ROOT / "wave2body" / "configs" / "hpe_point_transformer" / "m4human_absolute_hist4_s2_bf16.yaml"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave2body-config", default=str(DEFAULT_WAVE2BODY_CONFIG), help="HPE-token YAML config.")
    parser.add_argument(
        "--point-transformer-config",
        default=str(DEFAULT_POINT_TRANSFORMER_CONFIG),
        help="PointTransformer YAML config.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Benchmark epochs. Defaults to the requested 3.")
    parser.add_argument("--device", default="cuda", help="cuda/cpu; training code falls back to CPU if CUDA is absent.")
    parser.add_argument("--only", choices=("both", "wave2body", "point_transformer"), default="both")
    parser.add_argument("--run-name", help="Benchmark run name. Defaults to a timestamped name.")
    parser.add_argument("--work-dir", help="Output directory. Defaults under WAVE2BODY artifact root.")
    parser.add_argument("--output-json", help="Write the aggregate report to this path.")
    parser.add_argument("--batch-size", type=int, help="Override both configs' training.batch_size.")
    parser.add_argument("--wave2body-batch-size", type=int, help="Override only Wave2Body batch size.")
    parser.add_argument("--point-transformer-batch-size", type=int, help="Override only PointTransformer batch size.")
    parser.add_argument("--num-workers", type=int, help="Override both configs' training.num_workers.")
    parser.add_argument("--max-train-batches", type=int, help="Limit train batches per epoch for shorter benchmark runs.")
    parser.add_argument("--max-val-batches", type=int, help="Limit val batches per epoch.")
    parser.add_argument(
        "--max-test-batches",
        type=int,
        default=1,
        help="Limit PointTransformer final test batches. Defaults to 1 because val timing is the inference comparison.",
    )
    parser.add_argument("--full-test", action="store_true", help="Run the full PointTransformer final test split.")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not load pretrained tokenizer/VQ-VAE/PT checkpoints. Compute path is unchanged.",
    )
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        default=True,
        help="Keep torch.compile disabled to avoid compile warmup contaminating 3-epoch averages.",
    )
    parser.add_argument(
        "--enable-compile",
        dest="disable_compile",
        action="store_false",
        help="Honor each config's compile setting.",
    )
    return parser


def _default_run_name() -> str:
    return f"efficiency_3epoch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _resolve_work_root(args: argparse.Namespace, run_name: str) -> Path:
    if args.work_dir:
        return Path(args.work_dir).expanduser()
    return artifact_root() / "experiments" / "efficiency" / "results" / run_name


def _set_compile_disabled(training: dict[str, Any]) -> None:
    current = training.get("compile", {})
    if isinstance(current, dict):
        current = dict(current)
        current["enabled"] = False
        training["compile"] = current
    else:
        training["compile"] = {"enabled": False}


def _prepare_config(
    config: dict[str, Any],
    *,
    stage_name: str,
    run_name: str,
    epochs: int,
    batch_size: int | None,
    num_workers: int | None,
    no_pretrained: bool,
    disable_compile: bool,
) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    training = cfg.setdefault("training", {})
    training["epochs"] = int(epochs)
    training["run_name"] = f"{run_name}_{stage_name}"
    if batch_size is not None:
        training["batch_size"] = int(batch_size)
    if num_workers is not None:
        workers = int(num_workers)
        training["num_workers"] = workers
        if workers <= 0:
            training["pin_memory"] = False
            training["persistent_workers"] = False
    if disable_compile:
        _set_compile_disabled(training)
    if no_pretrained:
        pretrained = cfg.setdefault("pretrained", {})
        for key in list(pretrained):
            pretrained[key] = None
    return cfg


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mean_existing(rows: list[dict[str, Any]], phase: str, key: str) -> float | None:
    values = []
    for row in rows:
        phase_metrics = row.get(phase)
        if isinstance(phase_metrics, dict) and key in phase_metrics and phase_metrics[key] is not None:
            values.append(float(phase_metrics[key]))
    return mean(values) if values else None


def _load_split_manifest(output_root: str | Path | None) -> dict[str, Any]:
    if not output_root:
        return {}
    path = Path(output_root) / "split_manifest.json"
    if not path.exists():
        return {}
    return _load_json(path)


def _estimated_batches_samples(
    *,
    split_manifest: dict[str, Any],
    batch_size: int,
    max_train_batches: int | None,
    max_val_batches: int | None,
    max_test_batches: int | None,
) -> dict[str, dict[str, int | None]]:
    def bounded_batches(total_samples: int | None, limit: int | None, *, drop_last: bool) -> int | None:
        if total_samples is None:
            return limit
        batches = int(total_samples) // int(batch_size) if drop_last else (int(total_samples) + int(batch_size) - 1) // int(batch_size)
        return min(batches, int(limit)) if limit is not None else batches

    def payload(sample_key: str, limit: int | None, *, drop_last: bool) -> dict[str, int | None]:
        total = split_manifest.get(sample_key)
        total_int = int(total) if total is not None else None
        batches = bounded_batches(total_int, limit, drop_last=drop_last)
        samples = None if batches is None else int(batches) * int(batch_size)
        if total_int is not None and not drop_last and samples is not None:
            samples = min(samples, total_int)
        return {"configured_samples": total_int, "batches_per_epoch": batches, "samples_per_epoch": samples}

    return {
        "train": payload("train_num_samples", max_train_batches, drop_last=True),
        "val": payload("val_num_samples", max_val_batches, drop_last=False),
        "test": payload("test_num_samples", max_test_batches, drop_last=False),
    }


def _phase_summary(
    history: list[dict[str, Any]],
    *,
    phase: str,
    samples_per_epoch: int | None,
    batches_per_epoch: int | None,
) -> dict[str, Any]:
    samples_per_sec = _mean_existing(history, phase, "samples_per_sec")
    step_time_sec = _mean_existing(history, phase, "step_time_sec")
    data_time_sec = _mean_existing(history, phase, "data_time_sec")
    max_memory_mb = _mean_existing(history, phase, "max_memory_mb")
    approx_epoch_wall_sec = None
    if samples_per_sec and samples_per_epoch:
        approx_epoch_wall_sec = float(samples_per_epoch) / float(samples_per_sec)
    step_time_per_batch_sec = None
    data_time_per_batch_sec = None
    if batches_per_epoch:
        if step_time_sec is not None:
            step_time_per_batch_sec = float(step_time_sec) / float(batches_per_epoch)
        if data_time_sec is not None:
            data_time_per_batch_sec = float(data_time_sec) / float(batches_per_epoch)
    return {
        "avg_samples_per_sec": samples_per_sec,
        "avg_step_time_sec": step_time_sec,
        "avg_data_time_sec": data_time_sec,
        "avg_step_time_per_batch_sec": step_time_per_batch_sec,
        "avg_data_time_per_batch_sec": data_time_per_batch_sec,
        "avg_max_memory_mb": max_memory_mb,
        "approx_epoch_wall_sec": approx_epoch_wall_sec,
    }


def _summarize_run(
    *,
    stage: str,
    output_root: str | Path,
    history: list[dict[str, Any]],
    config: dict[str, Any],
    max_train_batches: int | None,
    max_val_batches: int | None,
    max_test_batches: int | None,
) -> dict[str, Any]:
    split_manifest = _load_split_manifest(output_root)
    batch_size = int(config.get("training", {}).get("batch_size", 0))
    size_estimates = _estimated_batches_samples(
        split_manifest=split_manifest,
        batch_size=batch_size,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        max_test_batches=max_test_batches,
    )
    train_samples = size_estimates["train"]["samples_per_epoch"]
    val_samples = size_estimates["val"]["samples_per_epoch"]
    train_batches = size_estimates["train"]["batches_per_epoch"]
    val_batches = size_estimates["val"]["batches_per_epoch"]
    return {
        "stage": stage,
        "output_root": str(output_root),
        "batch_size": batch_size,
        "epochs_completed": len(history),
        "dataset": config.get("dataset", {}).get("name"),
        "split_manifest": split_manifest,
        "epoch_size_estimates": size_estimates,
        "train": _phase_summary(
            history,
            phase="train",
            samples_per_epoch=train_samples,
            batches_per_epoch=train_batches,
        ),
        "inference_val": _phase_summary(
            history,
            phase="val",
            samples_per_epoch=val_samples,
            batches_per_epoch=val_batches,
        ),
        "history": history,
    }


def _read_pt_history(summary: dict[str, Any]) -> list[dict[str, Any]]:
    results_path = Path(summary["output_root"]) / "training_results.json"
    if not results_path.exists():
        return []
    payload = _load_json(results_path)
    history = payload.get("history", [])
    return history if isinstance(history, list) else []


def _is_absolute_root_config(config: dict[str, Any]) -> bool:
    stage = str(config.get("training", {}).get("stage", "")).strip().lower()
    return "root_head" in config or stage.startswith("absolute")


def _run_wave2body_stage(
    config: dict[str, Any],
    *,
    config_path: Path,
    work_dir: Path,
    device: str,
    max_train_batches: int | None,
    max_val_batches: int | None,
    max_test_batches: int | None,
) -> dict[str, Any]:
    if _is_absolute_root_config(config):
        return run_absolute_root_head(
            config,
            config_path=config_path,
            work_dir=work_dir,
            device=device,
            max_train_batches=max_train_batches,
            max_val_batches=max_val_batches,
            max_test_batches=max_test_batches,
        )
    return run_hpe_token(
        config,
        config_path=config_path,
        work_dir=work_dir,
        device=device,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
    )


def _print_report(report: dict[str, Any]) -> None:
    def fmt(value: float | None, suffix: str = "") -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.3f}{suffix}"

    print("\n[Efficiency] 3-epoch average summary")
    for key in ("wave2body", "point_transformer"):
        item = report.get(key)
        if not item:
            continue
        train = item["train"]
        infer = item["inference_val"]
        print(
            f"- {key}: train={fmt(train['avg_samples_per_sec'])} samples/s, "
            f"infer_val={fmt(infer['avg_samples_per_sec'])} samples/s, "
            f"train_step={fmt(train['avg_step_time_sec'], 's')}/epoch, "
            f"data={fmt(train['avg_data_time_sec'], 's')}/epoch, "
            f"peak_mem={fmt(train['avg_max_memory_mb'])} MB, "
            f"output={item['output_root']}"
        )


def main() -> None:
    args = _build_parser().parse_args()
    if int(args.epochs) <= 0:
        raise ValueError("--epochs must be positive")
    if args.full_test:
        args.max_test_batches = None
    run_name = args.run_name or _default_run_name()
    work_root = _resolve_work_root(args, run_name)
    work_root.mkdir(parents=True, exist_ok=True)

    wave_batch = args.wave2body_batch_size if args.wave2body_batch_size is not None else args.batch_size
    pt_batch = args.point_transformer_batch_size if args.point_transformer_batch_size is not None else args.batch_size

    report: dict[str, Any] = {
        "benchmark": "wave2body_vs_point_transformer_3epoch",
        "run_name": run_name,
        "work_root": str(work_root),
        "epochs": int(args.epochs),
        "device": args.device,
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
        "max_test_batches": args.max_test_batches,
    }

    if args.only in {"both", "wave2body"}:
        wave_config_path = Path(args.wave2body_config).expanduser()
        wave_config = _prepare_config(
            load_config(wave_config_path),
            stage_name="wave2body",
            run_name=run_name,
            epochs=args.epochs,
            batch_size=wave_batch,
            num_workers=args.num_workers,
            no_pretrained=args.no_pretrained,
            disable_compile=args.disable_compile,
        )
        wave_output = work_root / "wave2body"
        wave_summary = _run_wave2body_stage(
            wave_config,
            config_path=wave_config_path,
            work_dir=wave_output,
            device=args.device,
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
            max_test_batches=args.max_test_batches,
        )
        report["wave2body"] = _summarize_run(
            stage="wave2body_absolute_root_head" if _is_absolute_root_config(wave_config) else "wave2body_hpe_token",
            output_root=wave_summary["output_root"],
            history=list(wave_summary.get("history", [])),
            config=wave_config,
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
            max_test_batches=args.max_test_batches if _is_absolute_root_config(wave_config) else None,
        )

    if args.only in {"both", "point_transformer"}:
        pt_config_path = Path(args.point_transformer_config).expanduser()
        pt_config = _prepare_config(
            load_config(pt_config_path),
            stage_name="point_transformer",
            run_name=run_name,
            epochs=args.epochs,
            batch_size=pt_batch,
            num_workers=args.num_workers,
            no_pretrained=args.no_pretrained,
            disable_compile=args.disable_compile,
        )
        pt_output = work_root / "point_transformer"
        pt_summary = run_pt_pose(
            pt_config,
            config_path=pt_config_path,
            work_dir=pt_output,
            device=args.device,
            experiment_name="efficiency",
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
            max_test_batches=args.max_test_batches,
        )
        report["point_transformer"] = _summarize_run(
            stage="hpe_point_transformer",
            output_root=pt_summary["output_root"],
            history=_read_pt_history(pt_summary),
            config=pt_config,
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
            max_test_batches=args.max_test_batches,
        )

    output_json = Path(args.output_json).expanduser() if args.output_json else work_root / "efficiency_summary.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    report["summary_path"] = str(output_json)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_report(report)
    print(f"[Efficiency] Wrote summary: {output_json}")


if __name__ == "__main__":
    main()
