#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train the two-stage HPE-token practical absolute pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
VALID_SPLIT_IDS = ("s1", "s2", "s3")
EXPERIMENT_NAME = "hpe_token"


def _preparse_gpus(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpus")
    args, _ = parser.parse_known_args(argv)
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)


_preparse_gpus(sys.argv[1:])

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.hpe_token._configs import print_configs, resolve_config  # noqa: E402
from wave2body.train.common import load_config  # noqa: E402
from wave2body.dataset.radar_pose.point_sampling import VALID_POINT_SAMPLING_STRATEGIES  # noqa: E402
from wave2body.train.pipelines.absolute_root_head import run_absolute_root_head  # noqa: E402
from wave2body.train.pipelines.hpe_token import run_hpe_token  # noqa: E402
from wave2body.train.pipelines.hpe_token_plan import (  # noqa: E402
    dry_run_payload as build_hpe_token_dry_run_payload,
    planned_output_root as planned_hpe_token_output_root,
)
from wave2body.utils.paths import resolve_runtime_uri  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="mmbody_abs", help="Config alias, YAML stem, or YAML path.")
    parser.add_argument("--list-configs", action="store_true", help="List config aliases and YAML stems, then exit.")
    parser.add_argument("--split-id", choices=VALID_SPLIT_IDS, help="M4Human split id. Defaults to config value.")
    parser.add_argument("--device", default="cuda", help="cuda/cpu")
    parser.add_argument("--gpus", help='CUDA_VISIBLE_DEVICES value, e.g. "0" or "0,1".')
    parser.add_argument("--resume", help="Resume a full HPE token checkpoint.")
    parser.add_argument("--reset-epoch", action="store_true", help="Load model weights from --resume but restart optimizer/scheduler.")
    parser.add_argument("--work-dir", help="Explicit output directory.")
    parser.add_argument("--run-name", help="Override run name.")
    parser.add_argument("--run-tag", default="", help="Optional suffix appended to generated run name.")
    parser.add_argument("--seed", type=int, help="Override training seed and append it to the run name.")
    parser.add_argument("--epochs", type=int, help="Runtime epoch override.")
    parser.add_argument("--lr", type=float, help="Runtime optimizer learning-rate override.")
    parser.add_argument("--min-lr", type=float, help="Runtime scheduler minimum learning-rate override.")
    parser.add_argument("--batch-size", type=int, help="Runtime batch-size override.")
    parser.add_argument("--num-workers", type=int, help="Runtime dataloader worker override.")
    parser.add_argument("--n-points", type=int, help="Override data_format.n_points for point-cloud inputs.")
    parser.add_argument("--merge-nframes", type=int, help="Override data_format.merge_nframes for point-cloud inputs.")
    parser.add_argument(
        "--point-sampling",
        choices=sorted(VALID_POINT_SAMPLING_STRATEGIES),
        help="Override data_format.point_sampling for point-cloud downsampling.",
    )
    parser.add_argument("--mmwave-tokenizer-checkpoint", help="Override the K96 mmWave tokenizer checkpoint.")
    parser.add_argument("--body-vqvae-checkpoint", help="Override the K96 human-body VQ-VAE checkpoint.")
    parser.add_argument("--hpe-token-checkpoint", help="Override the frozen HPE token translator checkpoint for absolute-root training.")
    freeze_group = parser.add_mutually_exclusive_group()
    freeze_group.add_argument(
        "--finetune-mmwave-tokenizer",
        action="store_true",
        help="Unfreeze the mmWave tokenizer and optimize it with a lower LR group.",
    )
    freeze_group.add_argument(
        "--freeze-mmwave-tokenizer",
        action="store_true",
        help="Force-freeze the mmWave tokenizer even if the config enables finetuning.",
    )
    parser.add_argument(
        "--mmwave-tokenizer-lr-mult",
        type=float,
        help="LR multiplier for the mmWave tokenizer when it is trainable.",
    )
    parser.add_argument("--max-train-batches", type=int, help="Limit train batches for short verification runs.")
    parser.add_argument("--max-val-batches", type=int, help="Limit val batches for short verification runs.")
    parser.add_argument("--max-test-batches", type=int, help="Limit test batches for short verification runs.")
    parser.add_argument("--smoke", action="store_true", help="One epoch and one train/val batch.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved run plan without training.")
    return parser


def _is_absolute_root_config(config: dict[str, Any]) -> bool:
    stage = str(config.get("training", {}).get("stage", "")).strip().lower()
    return "root_head" in config or stage.startswith("absolute")


def _dataset_name(config: dict[str, Any]) -> str:
    return str(config.get("dataset", {}).get("name", "")).strip().lower()


def _is_m4human_config(config: dict[str, Any]) -> bool:
    return _dataset_name(config) == "m4human"


def _sensor_label(config: dict[str, Any]) -> str:
    sensor_mode = str(config.get("sensor_tokens", {}).get("mode", "token_only")).strip().lower()
    if sensor_mode in {"token", "embedding", "token_only"}:
        return "tokenonly"
    if sensor_mode in {"token_center_concat", "token+center", "embedding_center_concat"}:
        return "centerconcat"
    return sensor_mode.replace("_", "")


def _root_prior_label(config: dict[str, Any]) -> str:
    prior_mode = str(config.get("root_head", {}).get("prior_mode", "none")).strip().lower()
    if prior_mode in {"point_intensity_weighted_mean", "point_iw_mean"}:
        return "pointiwmean"
    if prior_mode == "point_bbox_midpoint":
        return "bboxmid"
    return prior_mode.replace("_", "") or "none"


def _root_arch_label(config: dict[str, Any]) -> str:
    architecture = str(config.get("root_head", {}).get("architecture", "query_attn")).strip().lower()
    return "" if architecture in {"", "query_attn"} else architecture.replace("_", "")


def _train_split_label(config: dict[str, Any]) -> str:
    split_cfg = config.get("training", {}).get("train_val_split")
    if not isinstance(split_cfg, dict):
        return ""
    if bool(split_cfg.get("enabled", False)):
        protocol = str(split_cfg.get("protocol", "")).strip().lower()
        if protocol == "heldout_train_sequences":
            return "heldout"
        return protocol.replace("_", "") if protocol else ""
    val_split = str(split_cfg.get("val_split", "")).strip().lower()
    if val_split == "test":
        return "fulltrain"
    return ""


def _retarget_split_checkpoint(path: str | os.PathLike[str] | None, split_id: str) -> str | None:
    if not path:
        return None
    path_str = str(Path(resolve_runtime_uri(str(path))).expanduser())
    if f"_{split_id}_" in path_str:
        return path_str
    fallback_candidate: str | None = None
    for existing_split in VALID_SPLIT_IDS:
        replacements = (
            (f"_{existing_split}_", f"_{split_id}_"),
        )
        for old, new in replacements:
            if old not in path_str:
                continue
            candidate = path_str.replace(old, new)
            if Path(candidate).expanduser().exists():
                return candidate
            fallback_candidate = fallback_candidate or candidate
    return fallback_candidate or path_str


def _resolve_pretrained_override(path: str | os.PathLike[str] | None) -> str | None:
    if path in (None, "", "null"):
        return None
    return str(Path(resolve_runtime_uri(str(path))).expanduser())


def _apply_experiment_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    training = config.setdefault("training", {})
    data_format = config.setdefault("data_format", {})
    pretrained = config.setdefault("pretrained", {})
    absolute_root = _is_absolute_root_config(config)
    is_m4human = _is_m4human_config(config)
    finetune_tokenizer = bool(getattr(args, "finetune_mmwave_tokenizer", False))
    freeze_tokenizer = bool(getattr(args, "freeze_mmwave_tokenizer", False))
    tokenizer_lr_mult_arg = getattr(args, "mmwave_tokenizer_lr_mult", None)

    if args.seed is not None:
        seed = int(args.seed)
        config["random_seed"] = seed
        training["seed"] = seed

    split_id = None
    if is_m4human:
        m4human = config.setdefault("m4human", {})
        split_id = args.split_id or str(m4human.get("split", "s2"))
        m4human["split"] = split_id
        m4human["temporal_window"] = 4
        m4human["pose_source"] = "param_joints"
        m4human["normalize_z"] = True
        data_format["merge_nframes"] = 4
        data_format["normalized"] = True
    elif args.split_id:
        raise ValueError("--split-id is only valid for M4Human HPE token configs.")

    if absolute_root:
        training["stage"] = "absolute_root_head"
        training["stage_display_name"] = "Absolute Root Head"
        training["metric_namespace"] = "absolute_root_head"
    else:
        training["stage"] = EXPERIMENT_NAME
        training["stage_display_name"] = "HPE Token"
        training["metric_namespace"] = "hpe_token"
    if finetune_tokenizer:
        training["freeze_mmwave_tokenizer"] = False
    elif freeze_tokenizer:
        training["freeze_mmwave_tokenizer"] = True
    else:
        training.setdefault("freeze_mmwave_tokenizer", True)
    if absolute_root:
        training.setdefault("freeze_token_translator", True)
    training["freeze_vqvae"] = True
    if tokenizer_lr_mult_arg is not None:
        if float(tokenizer_lr_mult_arg) <= 0:
            raise ValueError("--mmwave-tokenizer-lr-mult must be > 0")
        training["mmwave_tokenizer_lr_mult"] = float(tokenizer_lr_mult_arg)
    if not bool(training.get("freeze_mmwave_tokenizer", True)):
        training["frozen_feature_cache"] = {"enabled": False}
    else:
        training.setdefault("frozen_feature_cache", {"enabled": False})

    hpe_token_checkpoint = getattr(args, "hpe_token_checkpoint", None)
    if args.mmwave_tokenizer_checkpoint:
        pretrained["mmwave_tokenizer"] = _resolve_pretrained_override(args.mmwave_tokenizer_checkpoint)
    if args.body_vqvae_checkpoint:
        pretrained["vqvae"] = _resolve_pretrained_override(args.body_vqvae_checkpoint)
    if hpe_token_checkpoint:
        if not absolute_root:
            raise ValueError("--hpe-token-checkpoint is only valid with an absolute-root hpe_token config.")
        pretrained["hpe_token"] = _resolve_pretrained_override(hpe_token_checkpoint)
    elif absolute_root:
        if is_m4human:
            assert split_id is not None
            retargeted = _retarget_split_checkpoint(pretrained.get("hpe_token"), split_id)
            if retargeted:
                pretrained["hpe_token"] = retargeted
    if args.epochs is not None:
        training["epochs"] = int(args.epochs)
    if args.lr is not None:
        if float(args.lr) <= 0.0:
            raise ValueError("--lr must be > 0")
        training["lr"] = float(args.lr)
    if args.min_lr is not None:
        if float(args.min_lr) < 0.0:
            raise ValueError("--min-lr must be >= 0")
        training["min_lr"] = float(args.min_lr)
    if args.batch_size is not None:
        training["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        workers = int(args.num_workers)
        training["num_workers"] = workers
        if workers <= 0:
            training["pin_memory"] = False
            training["persistent_workers"] = False
    if args.n_points is not None:
        n_points = int(args.n_points)
        if n_points <= 0:
            raise ValueError("--n-points must be > 0")
        data_format["n_points"] = n_points
    if args.merge_nframes is not None:
        merge_nframes = int(args.merge_nframes)
        if merge_nframes <= 0:
            raise ValueError("--merge-nframes must be > 0")
        data_format["merge_nframes"] = merge_nframes
    if args.point_sampling:
        data_format["point_sampling"] = str(args.point_sampling)

    if args.smoke:
        training["epochs"] = 1
        training["batch_size"] = min(int(training.get("batch_size", 2)), 2)
        training["num_workers"] = 0
        training["pin_memory"] = False
        training["persistent_workers"] = False
        training["compile"] = {"enabled": False}

    recipe_or_runtime_override = any(
        [
            finetune_tokenizer,
            freeze_tokenizer,
            tokenizer_lr_mult_arg is not None,
            args.epochs is not None,
            args.lr is not None,
            args.min_lr is not None,
            args.batch_size is not None,
            args.split_id is not None,
            args.n_points is not None,
            args.merge_nframes is not None,
            bool(args.point_sampling),
            bool(args.run_tag),
            bool(args.smoke),
        ]
    )
    config_run_name = str(training.get("run_name", "")).strip()
    if args.run_name:
        training["run_name"] = args.run_name
    elif config_run_name and not recipe_or_runtime_override:
        training["run_name"] = config_run_name
    else:
        precision = str(training.get("precision", "fp32"))
        batch_size = int(training.get("batch_size", 32))
        epochs = int(training.get("epochs", 25))
        sensor_k = int(config.get("mmwave_tokenizer", {}).get("num_groups", 96))
        sensor_label = _sensor_label(config)
        tag = str(args.run_tag or "")
        if tag and not tag.startswith("_"):
            tag = f"_{tag}"
        smoke_tag = "_smoke" if args.smoke and "_smoke" not in tag else ""
        tokenizer_ft_tag = ""
        if not bool(training.get("freeze_mmwave_tokenizer", True)):
            lr_mult = float(training.get("mmwave_tokenizer_lr_mult", 0.1))
            lr_label = f"{lr_mult:.3g}".replace(".", "p")
            tokenizer_ft_tag = f"_tokft_lrm{lr_label}"
        points_tag = f"_np{int(args.n_points)}" if args.n_points is not None else ""
        merge_tag = f"_mf{int(args.merge_nframes)}" if args.merge_nframes is not None else ""
        sampling_tag = ""
        if args.point_sampling:
            sampling_label = str(args.point_sampling).replace("_", "")
            sampling_tag = f"_pts{sampling_label}"
        if absolute_root:
            prior_label = _root_prior_label(config)
            arch_label = _root_arch_label(config)
            arch_tag = f"{arch_label}_" if arch_label else ""
            if is_m4human:
                arch_tag = f"{arch_label or 'queryattn'}_"
                training["run_name"] = (
                    f"m4human_hpe_token_p1_{split_id}_k{sensor_k}_{sensor_label}_{arch_tag}{prior_label}_absolute_"
                    f"trainable_e{epochs}_{precision}_bs{batch_size}_trainonly_tokenizers"
                    f"{tokenizer_ft_tag}{points_tag}{merge_tag}{sampling_tag}{tag}{smoke_tag}"
                )
            else:
                dataset_tag = _dataset_name(config).replace("dataset", "") or "dataset"
                split_label = _train_split_label(config)
                split_tag = f"_{split_label}" if split_label else ""
                training["run_name"] = (
                    f"{dataset_tag}_hpe_token_k{sensor_k}_{sensor_label}{split_tag}_{arch_tag}{prior_label}_absolute_"
                    f"e{epochs}_{precision}_bs{batch_size}{tokenizer_ft_tag}{points_tag}{merge_tag}{sampling_tag}{tag}{smoke_tag}"
                )
        elif is_m4human:
            assert split_id is not None
            training["run_name"] = (
                f"m4human_hpe_token_p1_{split_id}_k{sensor_k}_{sensor_label}_xattn_"
                f"e{epochs}_{precision}_bs{batch_size}_trainonly_tokenizers"
                f"{tokenizer_ft_tag}{points_tag}{merge_tag}{sampling_tag}{tag}{smoke_tag}"
            )
        else:
            dataset_tag = _dataset_name(config).replace("dataset", "") or "dataset"
            split_label = _train_split_label(config)
            split_tag = f"_{split_label}" if split_label else ""
            training["run_name"] = (
                f"{dataset_tag}_hpe_token_k{sensor_k}_{sensor_label}{split_tag}_xattn_"
                f"e{epochs}_{precision}_bs{batch_size}{tokenizer_ft_tag}{points_tag}{merge_tag}{sampling_tag}{tag}{smoke_tag}"
            )
    if args.seed is not None:
        seed_tag = f"_seed{int(args.seed)}"
        run_name = str(training["run_name"])
        if seed_tag not in run_name:
            training["run_name"] = f"{run_name}{seed_tag}"
    return config


def _batch_limits(args: argparse.Namespace) -> dict[str, int | None]:
    smoke_limit = 1 if args.smoke else None
    train_limit = getattr(args, "max_train_batches", None)
    val_limit = getattr(args, "max_val_batches", None)
    test_limit = getattr(args, "max_test_batches", None)
    return {
        "train": train_limit if train_limit is not None else smoke_limit,
        "val": val_limit if val_limit is not None else smoke_limit,
        "test": test_limit if test_limit is not None else smoke_limit,
    }


def _planned_output_root(config: dict[str, Any], work_dir: str | None) -> Path:
    return planned_hpe_token_output_root(config, work_dir)


def _dry_run_payload(
    config_path: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
    batch_limits: dict[str, int | None],
) -> dict[str, Any]:
    return build_hpe_token_dry_run_payload(
        config_path=config_path,
        config=config,
        work_dir=args.work_dir,
        resume=args.resume,
        reset_epoch=bool(args.reset_epoch),
        batch_limits=batch_limits,
    )


def _validate_production_mmwave_tokenizer(config: dict[str, Any]) -> None:
    tokenizer = config.get("mmwave_tokenizer", {})
    method = str(tokenizer.get("method", "mae")).strip().lower()
    if method != "mae":
        raise ValueError(
            "Production HPE-token training only supports mmwave_tokenizer.method='mae'. "
            "Raw-point and other historical sensor encoder ablations were moved out of the production path."
        )
    patch_policy = str(tokenizer.get("patch_policy", "spatial_fps_knn")).strip().lower()
    if patch_policy != "spatial_fps_knn":
        raise ValueError(
            "Production HPE-token training only supports mmwave_tokenizer.patch_policy='spatial_fps_knn'. "
            "Body-part-assisted tokenizer ablations were moved out of the production path."
        )


def _validate_production_hpe_token_config(config: dict[str, Any]) -> None:
    _validate_production_mmwave_tokenizer(config)

    translator = config.get("token_translator", {})
    architecture = str(translator.get("architecture", translator.get("type", "cross_attention"))).strip().lower()
    if architecture not in {"cross_attention", "cross_attention_token_translator", "default"}:
        raise ValueError(
            "Production HPE-token training only supports token_translator.architecture='cross_attention'. "
            "Historical translator variants live under wave2body.models.token_translation.baselines."
        )
    if translator.get("candidate_reranker") is not None:
        raise ValueError(
            "token_translator.candidate_reranker is a baseline-only HPE-token option. "
            "Use a baseline-specific reproduction path instead of scripts/hpe_token/train.py."
        )
    training = config.get("training", {})
    if training.get("topk_token_ce") is not None or training.get("candidate_reranker_ce") is not None:
        raise ValueError(
            "training.topk_token_ce and training.candidate_reranker_ce are baseline-only HPE-token losses. "
            "Use a baseline-specific reproduction path instead of scripts/hpe_token/train.py."
        )

    if _is_absolute_root_config(config):
        root_head = config.get("root_head", {})
        root_arch = str(root_head.get("architecture", "query_attn")).strip().lower()
        if root_arch != "query_attn":
            raise ValueError(
                "Production absolute HPE-token training only supports root_head.architecture='query_attn'. "
                "Historical root-head architectures live under wave2body.models.hpe_token.baselines."
            )


def main() -> None:
    args = _build_parser().parse_args()
    if args.list_configs:
        print_configs()
        return

    config_path = resolve_config(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = _apply_experiment_overrides(load_config(config_path), args)
    _validate_production_hpe_token_config(config)
    batch_limits = _batch_limits(args)
    if args.dry_run:
        print(json.dumps(_dry_run_payload(config_path, config, args, batch_limits), indent=2))
        return

    if _is_absolute_root_config(config):
        work_dir = args.work_dir or str(_planned_output_root(config, None))
        run_absolute_root_head(
            config,
            config_path=config_path,
            resume=args.resume,
            work_dir=work_dir,
            device=args.device,
            max_train_batches=batch_limits["train"],
            max_val_batches=batch_limits["val"],
            max_test_batches=batch_limits["test"],
        )
        return

    run_hpe_token(
        config,
        config_path=config_path,
        resume=args.resume,
        reset_epoch=bool(args.reset_epoch),
        work_dir=args.work_dir,
        device=args.device,
        max_train_batches=batch_limits["train"],
        max_val_batches=batch_limits["val"],
    )


if __name__ == "__main__":
    main()
