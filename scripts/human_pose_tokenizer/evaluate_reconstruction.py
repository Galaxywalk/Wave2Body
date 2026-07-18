#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate human pose tokenizer reconstruction from package configs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from wave2body.evaluate.human_pose_tokenizer_reconstruction import (  # noqa: E402
    evaluate_checkpoint,
    print_summary,
    save_evaluation,
)
from scripts.human_pose_tokenizer._configs import HUMAN_POSE_TOKENIZER_WORKSPACE, resolve_config  # noqa: E402


EVAL_CONFIG = (
    REPO_ROOT
    / "wave2body"
    / "configs"
    / "human_pose_tokenizer"
    / "eval_part_k96_mmbody_m4human.yaml"
)


def default_checkpoint() -> Path:
    return HUMAN_POSE_TOKENIZER_WORKSPACE / "checkpoints" / "part_k96_stage_b_mmbody_m4human" / "best_model.pth"


def default_output() -> Path:
    return HUMAN_POSE_TOKENIZER_WORKSPACE / "results" / "reconstruction_eval_mmbody_m4human.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate human pose tokenizer reconstruction on configured pose datasets.")
    parser.add_argument("--config", default="adapt", help="Training config alias or YAML path used to build the model.")
    parser.add_argument("--eval-config", default=str(EVAL_CONFIG), help="Evaluation dataset/metric config YAML.")
    parser.add_argument("--checkpoint", default=str(default_checkpoint()), help="VQ-VAE checkpoint.")
    parser.add_argument("--output", default=str(default_output()), help="JSON output path.")
    parser.add_argument("--cache-dir", help="Override eval joint cache directory.")
    parser.add_argument("--batch-size", type=int, help="Override eval batch size.")
    parser.add_argument("--num-workers", type=int, help="Override eval DataLoader workers.")
    parser.add_argument("--gpus", default="0", help='CUDA_VISIBLE_DEVICES value. Use "" for current environment.')
    parser.add_argument("--skip-p-mpjpe", action="store_true", help="Skip Procrustes MPJPE.")
    args = parser.parse_args()

    if args.gpus is not None and args.gpus != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    config_path = resolve_config(args.config)
    eval_config_path = Path(args.eval_config).expanduser()
    checkpoint_path = Path(args.checkpoint).expanduser()
    output_path = Path(args.output).expanduser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = evaluate_checkpoint(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        eval_config_path=eval_config_path,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        compute_p_mpjpe=False if args.skip_p_mpjpe else None,
    )
    save_evaluation(payload, output_path)
    print_summary(payload)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
