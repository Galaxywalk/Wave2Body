# -*- coding: utf-8 -*-
"""Path helpers for training pipeline entry points."""

from __future__ import annotations

from pathlib import Path

from ...utils.paths import artifact_root


def resolve_stage_dirs(
    config: dict,
    *,
    work_dir: str | Path | None,
    default_name: str,
    tensorboard_name: str = "tensorboard",
) -> tuple[Path, Path]:
    training_cfg = config.setdefault("training", {})
    if work_dir:
        ckpt_dir = Path(work_dir).expanduser()
        log_dir = ckpt_dir / tensorboard_name
    else:
        run_name = str(training_cfg.get("run_name") or default_name)
        ckpt_dir = Path(training_cfg.get("checkpoint_dir", artifact_root() / "training" / run_name / "checkpoints"))
        log_dir = Path(training_cfg.get("log_dir", ckpt_dir / tensorboard_name))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    training_cfg["checkpoint_dir"] = str(ckpt_dir)
    training_cfg["log_dir"] = str(log_dir)
    return ckpt_dir, log_dir


def resolve_run_root(
    config: dict,
    *,
    work_dir: str | Path | None,
    default_name: str,
    experiment_name: str,
) -> tuple[Path, Path, Path]:
    training_cfg = config.setdefault("training", {})
    run_name = str(training_cfg.get("run_name") or default_name)
    output_root = (
        Path(work_dir).expanduser()
        if work_dir
        else artifact_root() / "experiments" / experiment_name / "results" / run_name
    )
    checkpoint_dir = output_root / "checkpoints"
    log_dir = output_root / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return output_root, checkpoint_dir, log_dir


def write_effective_config(config: dict, output_dir: str | Path) -> Path:
    import yaml

    path = Path(output_dir) / "effective_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path
