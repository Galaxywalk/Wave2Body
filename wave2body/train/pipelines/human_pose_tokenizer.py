# -*- coding: utf-8 -*-
"""Pipeline entrypoint for human pose tokenizer training."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run_human_pose_tokenizer(
    *,
    config_path: str | Path,
    work_dir: str | Path,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Run the package human pose tokenizer trainer and return checkpoint paths."""
    from .. import human_pose_tokenizer

    output_dir = Path(work_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        "--config",
        str(config_path),
        "--work-dir",
        str(output_dir),
    ]
    if extra_args:
        argv.extend(str(x) for x in extra_args)

    result = human_pose_tokenizer.main(argv) or {}
    return {
        "stage": "human_pose_tokenizer",
        "output_root": str(output_dir),
        "best_checkpoint": str(output_dir / "best_model.pth"),
        "final_checkpoint": str(output_dir / "final_model.pth"),
        "latest_checkpoint": str(output_dir / "final_model.pth"),
        **result,
    }
