# -*- coding: utf-8 -*-
"""Config-driven CLI helpers for self-supervised mmWave tokenizer training."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Sequence

from ..utils.paths import artifact_root, repo_root
from .common import load_config, resolve_path
from .pipelines.mmwave_tokenizer import export_mmwave_tokens, run_mmwave_tokenizer_training


DEFAULT_MMWAVE_TOKENIZER_EXPERIMENT = "mmwave_tokenizer"
DEFAULT_STAGE_NAME = "mmwave_tokenizer"
SOTA_MMWAVE_TOKENIZER_SCHEME = "mae_spatial_k96"
PRODUCTION_MMWAVE_TOKENIZER_METHOD = "mae"
PRODUCTION_MMWAVE_PATCH_POLICY = "spatial_fps_knn"
SOTA_MMWAVE_TOKENIZER_SUMMARY = (
    "Current mmWave tokenizer SOTA: spatial MAE with K=96 sensor tokens. "
    "K512/K1024 improve reconstruction/coverage but do not improve held-out downstream pose."
)
DEFAULT_CONFIG_ALIAS = "sota"

CONFIG_DIR = repo_root() / "wave2body" / "configs" / "mmwave_tokenizer"
MMWAVE_TOKENIZER_SOTA_CONFIGS = {
    "m4human_s2": CONFIG_DIR / "variants" / "m4human_s2_trainonly_mae_spatial_k96_fast.yaml",
    "m4human_s3": CONFIG_DIR / "variants" / "m4human_s3_trainonly_mae_spatial_k96_fast.yaml",
    "mmbody": CONFIG_DIR / "mmbody_mae_spatial_k96_fast.yaml",
}
MMWAVE_TOKENIZER_NAMED_CONFIGS = {
    "sota": MMWAVE_TOKENIZER_SOTA_CONFIGS["m4human_s2"],
    "m4human_sota": MMWAVE_TOKENIZER_SOTA_CONFIGS["m4human_s2"],
    "m4human_s2_trainonly": MMWAVE_TOKENIZER_SOTA_CONFIGS["m4human_s2"],
    "m4human_s3_trainonly": MMWAVE_TOKENIZER_SOTA_CONFIGS["m4human_s3"],
    "mmbody_sota": MMWAVE_TOKENIZER_SOTA_CONFIGS["mmbody"],
}


def resolve_mmwave_tokenizer_config(value: str | Path, *, base_dir: str | Path | None = None) -> Path:
    """Resolve a named config, YAML stem, or filesystem path."""
    key = str(value)
    if key in MMWAVE_TOKENIZER_NAMED_CONFIGS:
        return MMWAVE_TOKENIZER_NAMED_CONFIGS[key]
    package_path = CONFIG_DIR / key
    if package_path.exists():
        return package_path
    if not key.endswith((".yaml", ".yml", ".json")):
        package_yaml = CONFIG_DIR / f"{key}.yaml"
        if package_yaml.exists():
            return package_yaml
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if base_dir is not None:
        resolved = resolve_path(base_dir, path)
        if resolved is not None:
            return resolved
    return (Path.cwd() / path).resolve()


def print_mmwave_tokenizer_configs() -> None:
    print(SOTA_MMWAVE_TOKENIZER_SUMMARY)
    print("SOTA configs:")
    for dataset, path in MMWAVE_TOKENIZER_SOTA_CONFIGS.items():
        print(f"  {dataset}: {path}")
    print("Named configs:")
    for name, path in MMWAVE_TOKENIZER_NAMED_CONFIGS.items():
        print(f"  {name}: {path}")
    print("YAML stems:")
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        print(f"  {path.stem}: {path}")
    variant_dir = CONFIG_DIR / "variants"
    if variant_dir.exists():
        print("Variant YAML stems:")
        for path in sorted(variant_dir.glob("*.yaml")):
            print(f"  variants/{path.stem}: {path}")
    baseline_dir = CONFIG_DIR / "baselines"
    if baseline_dir.exists():
        print("Baseline YAML stems (not supported by this production CLI):")
        for path in sorted(baseline_dir.glob("*.yaml")):
            print(f"  baselines/{path.stem}: {path}")


def planned_mmwave_tokenizer_output_root(
    config: dict[str, Any],
    work_dir: str | None,
    *,
    experiment_name: str = DEFAULT_MMWAVE_TOKENIZER_EXPERIMENT,
) -> Path:
    if work_dir:
        return Path(work_dir).expanduser()
    return (
        artifact_root()
        / "experiments"
        / experiment_name
        / "results"
        / str(config.get("training", {}).get("run_name", "m4human_mmwave_tokenizer"))
    )


def _short_patch_policy(policy: str) -> str:
    key = str(policy).lower()
    if key == "weak_body_part_knn":
        return "body_part"
    if key == "spatial_fps_knn":
        return "spatial"
    return key


def _uses_pose(method: str, patch_policy: str) -> bool:
    del method
    return _short_patch_policy(patch_policy) != "spatial"


def _is_m4human_config(config: dict[str, Any]) -> bool:
    name = str(config.get("dataset", {}).get("name", "")).lower()
    return name == "m4human"


def _default_run_name(config: dict[str, Any]) -> str:
    training = config.get("training", {})
    tokenizer = config.get("mmwave_tokenizer", {})
    m4human = config.get("m4human", {})
    data_format = config.get("data_format", {})
    dataset_tag = str(config.get("dataset", {}).get("name", "dataset")).lower()
    method = str(tokenizer.get("method", PRODUCTION_MMWAVE_TOKENIZER_METHOD))
    patch = _short_patch_policy(str(tokenizer.get("patch_policy", "spatial")))
    split = str(m4human.get("split", "s2"))
    train_subsets = list(training.get("train_subsets", ["train"]))
    precision = str(training.get("precision", "fp32"))
    batch_size = int(training.get("batch_size", 8))
    num_groups = int(tokenizer.get("num_groups", data_format.get("num_joints", 22)))
    k_tag = f"_k{num_groups}" if num_groups != int(data_format.get("num_joints", 22)) else ""
    if not _is_m4human_config(config):
        return f"{dataset_tag}_{method}_{patch}_{precision}_bs{batch_size}{k_tag}"
    if train_subsets == ["train", "val", "test"]:
        return f"m4human_all_{method}_{patch}_{precision}_bs{batch_size}{k_tag}"
    return f"m4human_{split}_{method}_{patch}_{precision}_bs{batch_size}{k_tag}"


def _normalize_mmwave_tokenizer_config(config: dict[str, Any]) -> None:
    tokenizer = config.setdefault("mmwave_tokenizer", {})
    data_format = config.setdefault("data_format", {})
    m4human = config.setdefault("m4human", {})
    tokenizer.setdefault("input_channels", int(data_format.get("radar_input_c", 4)))
    tokenizer.setdefault("num_groups", int(data_format.get("num_joints", 22)))
    if _is_m4human_config(config):
        if _uses_pose(str(tokenizer.get("method", "")), str(tokenizer.get("patch_policy", "spatial"))):
            m4human["load_pose"] = True
            m4human.setdefault("pose_source", "param_joints")
        else:
            m4human["load_pose"] = False
    config.setdefault("training", {}).setdefault("run_name", _default_run_name(config))


def validate_production_mmwave_tokenizer_config(config: dict[str, Any]) -> None:
    tokenizer = config.get("mmwave_tokenizer", {})
    method = str(tokenizer.get("method", PRODUCTION_MMWAVE_TOKENIZER_METHOD)).strip().lower()
    if method != PRODUCTION_MMWAVE_TOKENIZER_METHOD:
        raise ValueError(
            "scripts/mmwave_tokenizer is the production Point-MAE tokenizer CLI and only supports "
            "mmwave_tokenizer.method='mae'. Historical tokenizer methods were moved to "
            "wave2body.models.mmwave_tokenizer.baselines and baseline configs under "
            "wave2body/configs/mmwave_tokenizer/baselines/."
        )
    patch_policy = str(tokenizer.get("patch_policy", PRODUCTION_MMWAVE_PATCH_POLICY)).strip().lower()
    if patch_policy != PRODUCTION_MMWAVE_PATCH_POLICY:
        raise ValueError(
            "scripts/mmwave_tokenizer only supports patch_policy='spatial_fps_knn'. "
            "Body-part-assisted grouping belongs to historical baseline experiments."
        )


def apply_mmwave_tokenizer_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    training = cfg.setdefault("training", {})

    if getattr(args, "run_name", None):
        training["run_name"] = str(args.run_name)
    if getattr(args, "epochs", None) is not None:
        training["epochs"] = int(args.epochs)
    if getattr(args, "batch_size", None) is not None:
        training["batch_size"] = int(args.batch_size)
    if getattr(args, "num_workers", None) is not None:
        training["num_workers"] = int(args.num_workers)
        if int(args.num_workers) <= 0:
            training["pin_memory"] = False
            training["persistent_workers"] = False
    if getattr(args, "prefetch_factor", None) is not None:
        training["prefetch_factor"] = int(args.prefetch_factor)
    if getattr(args, "log_every", None) is not None:
        training["log_every"] = int(args.log_every)
    if getattr(args, "sync_timing", False):
        training["sync_timing"] = True
    if getattr(args, "precision", None) is not None:
        training["precision"] = str(args.precision)
        training["amp"] = str(args.precision) != "fp32"
    if getattr(args, "smoke", False):
        training["epochs"] = 1
        training["batch_size"] = min(int(training.get("batch_size", 8)), 2)
        training["num_workers"] = 0
        training["pin_memory"] = False
        training["persistent_workers"] = False
        args.max_train_batches = 1 if getattr(args, "max_train_batches", None) is None else args.max_train_batches
        args.max_val_batches = 1 if getattr(args, "max_val_batches", None) is None else args.max_val_batches

    _normalize_mmwave_tokenizer_config(cfg)
    return cfg


def build_mmwave_tokenizer_parser(
    *,
    default_config: str | Path = DEFAULT_CONFIG_ALIAS,
    default_mode: str = "train",
    description: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("--config", default=str(default_config), help="Named config, YAML stem, or YAML path.")
    parser.add_argument("--list-configs", action="store_true", help="List named configs and packaged YAML stems, then exit.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpus", default=None, help='CUDA_VISIBLE_DEVICES value, e.g. "0".')
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config/output plan")

    if default_mode == "train":
        parser.add_argument("--work-dir", default=None)
        parser.add_argument("--run-name", default=None)
        parser.add_argument("--resume", default=None, help="Resume/train from checkpoint")
        parser.add_argument("--epochs", type=int, default=None)
        parser.add_argument("--prefetch-factor", type=int, default=None)
        parser.add_argument("--log-every", type=int, default=None)
        parser.add_argument("--sync-timing", action="store_true", help="Synchronize CUDA each step for profiling metrics.")
        parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default=None)
        parser.add_argument("--max-train-batches", type=int, default=None)
        parser.add_argument("--max-val-batches", type=int, default=None)
        parser.add_argument("--smoke", action="store_true", help="One epoch, batch<=2, one train/val batch")
    elif default_mode == "export":
        parser.add_argument("--checkpoint", required=False, default=None)
        parser.add_argument("--export-split", choices=["train", "val", "test"], default="val")
        parser.add_argument("--export-output", default=None)
        parser.add_argument("--max-export-batches", type=int, default=None)
    else:
        raise ValueError(f"Unsupported mmWave tokenizer CLI mode: {default_mode}")
    return parser


def _dry_run_plan(
    *,
    config: dict[str, Any],
    config_path: Path,
    mode: str,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    training = config.get("training", {})
    tokenizer = config.get("mmwave_tokenizer", {})
    m4human = config.get("m4human", {})
    return {
        "config": str(config_path),
        "mode": mode,
        "dataset": config.get("dataset", {}).get("name"),
        "run_name": training.get("run_name"),
        "method": tokenizer.get("method"),
        "patch_policy": tokenizer.get("patch_policy"),
        "split": m4human.get("split"),
        "train_subsets": training.get("train_subsets"),
        "load_pose": m4human.get("load_pose"),
        "pose_source": m4human.get("pose_source"),
        "teacher_checkpoint": tokenizer.get("teacher_checkpoint"),
        "num_groups": tokenizer.get("num_groups"),
        "output_root": str(output_root),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dataloader": {
            "batch_size": training.get("batch_size"),
            "num_workers": training.get("num_workers"),
            "pin_memory": training.get("pin_memory"),
            "persistent_workers": training.get("persistent_workers"),
            "prefetch_factor": training.get("prefetch_factor"),
            "sync_timing": training.get("sync_timing", False),
        },
        "max_batches": {
            "train": getattr(args, "max_train_batches", None),
            "val": getattr(args, "max_val_batches", None),
            "export": getattr(args, "max_export_batches", None),
        },
    }


def run_mmwave_tokenizer_cli(
    argv: Sequence[str] | None = None,
    *,
    default_config: str | Path = DEFAULT_CONFIG_ALIAS,
    default_mode: str = "train",
    base_dir: str | Path | None = None,
    experiment_name: str = DEFAULT_MMWAVE_TOKENIZER_EXPERIMENT,
    stage_name: str = DEFAULT_STAGE_NAME,
    log_prefix: str = "[MMWaveTokenizer Runtime]",
    tensorboard_namespace: str = "mmwave_tokenizer",
    description: str | None = None,
) -> None:
    parser = build_mmwave_tokenizer_parser(
        default_config=default_config,
        default_mode=default_mode,
        description=description,
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    if args.list_configs:
        print_mmwave_tokenizer_configs()
        return

    config_path = resolve_mmwave_tokenizer_config(args.config, base_dir=base_dir)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = apply_mmwave_tokenizer_overrides(load_config(config_path), args)
    validate_production_mmwave_tokenizer_config(config)
    output_root = planned_mmwave_tokenizer_output_root(
        config,
        getattr(args, "work_dir", None),
        experiment_name=experiment_name,
    )
    if args.dry_run:
        print(
            json.dumps(
                _dry_run_plan(
                    config=config,
                    config_path=config_path,
                    mode=default_mode,
                    output_root=output_root,
                    args=args,
                ),
                indent=2,
            )
        )
        return

    if default_mode == "train":
        run_mmwave_tokenizer_training(
            config,
            config_path=config_path,
            resume=getattr(args, "resume", None),
            work_dir=getattr(args, "work_dir", None),
            device=args.device,
            max_train_batches=getattr(args, "max_train_batches", None),
            max_val_batches=getattr(args, "max_val_batches", None),
            experiment_name=experiment_name,
            stage_name=stage_name,
            log_prefix=log_prefix,
            tensorboard_namespace=tensorboard_namespace,
        )
        return

    export_mmwave_tokens(
        config,
        checkpoint=str(getattr(args, "checkpoint", "") or ""),
        split=str(getattr(args, "export_split", "val")),
        output_path=getattr(args, "export_output", None),
        max_batches=getattr(args, "max_export_batches", None),
        device=args.device,
    )


__all__ = [
    "CONFIG_DIR",
    "MMWAVE_TOKENIZER_NAMED_CONFIGS",
    "MMWAVE_TOKENIZER_SOTA_CONFIGS",
    "SOTA_MMWAVE_TOKENIZER_SCHEME",
    "SOTA_MMWAVE_TOKENIZER_SUMMARY",
    "DEFAULT_CONFIG_ALIAS",
    "DEFAULT_MMWAVE_TOKENIZER_EXPERIMENT",
    "apply_mmwave_tokenizer_overrides",
    "build_mmwave_tokenizer_parser",
    "planned_mmwave_tokenizer_output_root",
    "print_mmwave_tokenizer_configs",
    "resolve_mmwave_tokenizer_config",
    "run_mmwave_tokenizer_cli",
    "validate_production_mmwave_tokenizer_config",
]
