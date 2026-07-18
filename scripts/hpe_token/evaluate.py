#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate HPE-token checkpoints on configured radar-pose splits."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
VALID_SPLIT_IDS = ("s1", "s2", "s3")
VALID_EVAL_SPLITS = ("val", "test", "train")


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
from wave2body.train.pipelines.absolute_root_head import run_absolute_root_head_evaluation  # noqa: E402
from wave2body.train.pipelines.hpe_token import run_hpe_token_evaluation  # noqa: E402
from wave2body.dataset.radar_pose.point_sampling import VALID_POINT_SAMPLING_STRATEGIES  # noqa: E402


def _dataset_name(config: dict[str, Any]) -> str:
    return str(config.get("dataset", {}).get("name", "")).strip().lower()


def _is_m4human_config(config: dict[str, Any]) -> bool:
    return _dataset_name(config) == "m4human"


def _is_absolute_root_config(config: dict[str, Any]) -> bool:
    training = config.get("training", {})
    return str(training.get("stage", "")).strip().lower() == "absolute_root_head" or bool(config.get("root_head"))


def _validate_production_hpe_token_config(config: dict[str, Any]) -> None:
    tokenizer = config.get("mmwave_tokenizer", {})
    method = str(tokenizer.get("method", "mae")).strip().lower()
    if method != "mae":
        raise ValueError(
            "Production HPE-token evaluation only supports mmwave_tokenizer.method='mae'. "
            "Raw-point and other historical sensor encoder ablations were moved out of the production path."
        )
    patch_policy = str(tokenizer.get("patch_policy", "spatial_fps_knn")).strip().lower()
    if patch_policy != "spatial_fps_knn":
        raise ValueError(
            "Production HPE-token evaluation only supports mmwave_tokenizer.patch_policy='spatial_fps_knn'. "
            "Body-part-assisted tokenizer ablations were moved out of the production path."
        )

    translator = config.get("token_translator", {})
    architecture = str(translator.get("architecture", translator.get("type", "cross_attention"))).strip().lower()
    if architecture not in {"cross_attention", "cross_attention_token_translator", "default"}:
        raise ValueError(
            "Production HPE-token evaluation only supports token_translator.architecture='cross_attention'. "
            "Historical translator variants live under wave2body.models.token_translation.baselines."
        )
    if translator.get("candidate_reranker") is not None:
        raise ValueError(
            "token_translator.candidate_reranker is a baseline-only HPE-token option. "
            "Use a baseline-specific reproduction path instead of scripts/hpe_token/evaluate.py."
        )
    training = config.get("training", {})
    if training.get("topk_token_ce") is not None or training.get("candidate_reranker_ce") is not None:
        raise ValueError(
            "training.topk_token_ce and training.candidate_reranker_ce are baseline-only HPE-token losses. "
            "Use a baseline-specific reproduction path instead of scripts/hpe_token/evaluate.py."
        )

    if _is_absolute_root_config(config):
        root_head = config.get("root_head", {})
        root_arch = str(root_head.get("architecture", "query_attn")).strip().lower()
        if root_arch != "query_attn":
            raise ValueError(
                "Production absolute HPE-token evaluation only supports root_head.architecture='query_attn'. "
                "Historical root-head architectures live under wave2body.models.hpe_token.baselines."
            )


def _float_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _parse_float_csv(value: str) -> list[float]:
    out: list[float] = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    if not out:
        raise ValueError("Expected at least one float value.")
    return out


def _parse_int_csv(value: str) -> list[int]:
    out: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise ValueError("Expected at least one integer value.")
    if any(item <= 0 for item in out):
        raise ValueError(f"Expected positive integer values, got {out}")
    return out


def _parse_str_csv(value: str) -> list[str]:
    out: list[str] = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
            out.append(part)
    if not out:
        raise ValueError("Expected at least one non-empty value.")
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="mmbody_abs", help="Config alias, YAML stem, or YAML path.")
    parser.add_argument("--list-configs", action="store_true", help="List config aliases and YAML stems, then exit.")
    parser.add_argument("--checkpoint", required=False, help="HPE token checkpoint to evaluate.")
    parser.add_argument(
        "--checkpoint-ensemble",
        help=(
            "Comma-separated additional HPE token checkpoints to average with --checkpoint at eval time. "
            "The primary --checkpoint is always the first ensemble member."
        ),
    )
    parser.add_argument(
        "--checkpoint-ensemble-weights",
        help=(
            "Optional comma-separated model weights for --checkpoint followed by --checkpoint-ensemble entries. "
            "Defaults to equal weights."
        ),
    )
    parser.add_argument("--split-id", choices=VALID_SPLIT_IDS, help="M4Human split id.")
    parser.add_argument("--eval-split", default="test", choices=VALID_EVAL_SPLITS, help="Dataset subset to evaluate.")
    parser.add_argument("--device", default="cuda", help="cuda/cpu")
    parser.add_argument("--gpus", help='CUDA_VISIBLE_DEVICES value, e.g. "0" or "0,1".')
    parser.add_argument("--batch-size", type=int, help="Eval batch-size override.")
    parser.add_argument("--num-workers", type=int, help="Eval dataloader worker override.")
    parser.add_argument("--n-points", type=int, help="Override data_format.n_points for point-cloud inputs.")
    parser.add_argument("--merge-nframes", type=int, help="Override data_format.merge_nframes for point-cloud inputs.")
    parser.add_argument(
        "--point-ensemble-n-points",
        help=(
            "Comma-separated deterministic point-view sizes for eval-time ensembling, "
            "e.g. 200,160. Uses --point-sampling or the config sampling strategy for every view."
        ),
    )
    parser.add_argument(
        "--point-ensemble-weights",
        help="Optional comma-separated point-view weights matching --point-ensemble-n-points, e.g. 2,1,1.",
    )
    parser.add_argument(
        "--point-sampling",
        choices=sorted(VALID_POINT_SAMPLING_STRATEGIES),
        help="Override data_format.point_sampling for point-cloud downsampling diagnostics.",
    )
    parser.add_argument("--max-batches", type=int, help="Limit eval batches for smoke checks.")
    parser.add_argument("--output", help="Output metrics JSON path.")
    parser.add_argument("--mmwave-tokenizer-checkpoint", help="Override the K96 mmWave tokenizer checkpoint.")
    parser.add_argument("--body-vqvae-checkpoint", help="Override the K96 human-body VQ-VAE checkpoint.")
    parser.add_argument("--no-action-groups", action="store_true", help="Disable M4Human action-group metrics.")
    parser.add_argument("--per-action", action="store_true", help="Also write per-action MPJPE bins.")
    parser.add_argument(
        "--logit-decode-diagnostics",
        action="store_true",
        help="Also evaluate learned-logit and codebook-distance logit decoding variants.",
    )
    parser.add_argument(
        "--logit-decode-temperatures",
        default="0.5,1.0",
        help="Comma-separated temperatures for soft logit decode diagnostics.",
    )
    parser.add_argument(
        "--logit-blend-weights",
        default=None,
        help="Comma-separated learned-logit weights for learned+codebook blend softdecode diagnostics.",
    )
    parser.add_argument(
        "--token-topk",
        default=None,
        help="Comma-separated learned-logit body-token top-k values to report, e.g. 1,3,5,10.",
    )
    parser.add_argument(
        "--token-topk-no-slots",
        action="store_true",
        help="When --token-topk is set, omit per-token-slot top-k details.",
    )
    parser.add_argument(
        "--token-topk-oracle",
        action="store_true",
        help="Also decode target-aware top-k oracle poses for reranker upper-bound diagnostics.",
    )
    parser.add_argument(
        "--token-alignment",
        action="store_true",
        help=(
            "Analyze Stage-I quantized body-token alignment against same-action "
            "cross-subject and different-action same-subject controls."
        ),
    )
    parser.add_argument(
        "--token-alignment-shuffle-seed",
        type=int,
        default=0,
        help="Deterministic shuffle seed for --token-alignment (default: 0).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved eval plan without running.")
    return parser


def _apply_eval_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    training = config.setdefault("training", {})
    data_format = config.setdefault("data_format", {})
    pretrained = config.setdefault("pretrained", {})
    is_m4human = _is_m4human_config(config)
    is_absolute_root = _is_absolute_root_config(config)

    if is_m4human:
        m4human = config.setdefault("m4human", {})
        if "scale" in m4human:
            raise ValueError("M4Human scale is no longer configurable; remove m4human.scale from the config.")
        m4human["split"] = args.split_id or str(m4human.get("split", "s2"))
        m4human["temporal_window"] = 4
        m4human["pose_source"] = "param_joints"
        m4human["normalize_z"] = True
        data_format["merge_nframes"] = 4
        data_format["normalized"] = True
    elif args.split_id:
        raise ValueError("--split-id is only valid for M4Human HPE token configs.")

    if is_absolute_root:
        training["stage"] = "absolute_root_head"
        training["stage_display_name"] = "Absolute Root Head"
        training["metric_namespace"] = "absolute_root_head"
        training["freeze_token_translator"] = True
    else:
        training["stage"] = "hpe_token"
        training["stage_display_name"] = "HPE Token"
        training["metric_namespace"] = "hpe_token"
    training["freeze_mmwave_tokenizer"] = True
    training["freeze_vqvae"] = True
    training["frozen_feature_cache"] = {"enabled": False}
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
    if args.mmwave_tokenizer_checkpoint:
        pretrained["mmwave_tokenizer"] = args.mmwave_tokenizer_checkpoint
    if args.body_vqvae_checkpoint:
        pretrained["vqvae"] = args.body_vqvae_checkpoint
    if args.point_ensemble_n_points:
        ensemble_points = _parse_int_csv(str(args.point_ensemble_n_points))
        ensemble_weights = (
            _parse_float_csv(str(args.point_ensemble_weights))
            if args.point_ensemble_weights
            else [1.0] * len(ensemble_points)
        )
        if len(ensemble_weights) != len(ensemble_points):
            raise ValueError("--point-ensemble-weights must match --point-ensemble-n-points length.")
        if any(weight < 0.0 for weight in ensemble_weights) or sum(ensemble_weights) <= 0.0:
            raise ValueError("--point-ensemble-weights must be non-negative with positive sum.")
        point_sampling = str(args.point_sampling or data_format.get("point_sampling", "random"))
        config.setdefault("evaluation", {})["point_ensemble"] = {
            "enabled": True,
            "views": [
                {
                    "n_points": int(n_points),
                    "point_sampling": point_sampling,
                    "weight": float(weight),
                }
                for n_points, weight in zip(ensemble_points, ensemble_weights)
            ],
        }
    if args.checkpoint_ensemble:
        primary_checkpoint = str(getattr(args, "checkpoint", "") or "").strip()
        if not primary_checkpoint:
            raise ValueError("--checkpoint-ensemble requires --checkpoint.")
        extra_checkpoints = _parse_str_csv(str(args.checkpoint_ensemble))
        if primary_checkpoint in extra_checkpoints:
            raise ValueError("--checkpoint-ensemble should list additional checkpoints, not repeat --checkpoint.")
        checkpoints = [primary_checkpoint] + extra_checkpoints
        weights = (
            _parse_float_csv(str(args.checkpoint_ensemble_weights))
            if args.checkpoint_ensemble_weights
            else [1.0] * len(checkpoints)
        )
        if len(weights) != len(checkpoints):
            raise ValueError("--checkpoint-ensemble-weights must match --checkpoint plus ensemble entries.")
        if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
            raise ValueError("--checkpoint-ensemble-weights must be non-negative with positive sum.")
        config.setdefault("evaluation", {})["checkpoint_ensemble"] = {
            "enabled": True,
            "checkpoints": checkpoints,
            "weights": [float(weight) for weight in weights],
        }
    if bool(getattr(args, "logit_decode_diagnostics", False)):
        temperatures = _parse_float_csv(str(args.logit_decode_temperatures))
        first_temp = temperatures[0]
        specs: list[dict[str, Any]] = [
            {
                "name": f"learned_argmax_t{_float_tag(first_temp)}",
                "source": "learned",
                "mode": "argmax",
                "temperature": first_temp,
            },
            {
                "name": f"codebook_argmax_s1_t{_float_tag(first_temp)}",
                "source": "codebook_distance",
                "mode": "argmax",
                "temperature": first_temp,
                "scale": 1.0,
            },
        ]
        for temperature in temperatures:
            tag = _float_tag(temperature)
            specs.extend(
                [
                    {
                        "name": f"learned_soft_t{tag}",
                        "source": "learned",
                        "mode": "soft_expectation",
                        "temperature": temperature,
                    },
                    {
                        "name": f"codebook_soft_s1_t{tag}",
                        "source": "codebook_distance",
                        "mode": "soft_expectation",
                        "temperature": temperature,
                        "scale": 1.0,
                    },
                ]
            )
        config.setdefault("evaluation", {})["logit_decodes"] = specs
    if args.logit_blend_weights:
        temperatures = _parse_float_csv(str(args.logit_decode_temperatures))
        learned_weights = _parse_float_csv(str(args.logit_blend_weights))
        evaluation = config.setdefault("evaluation", {})
        specs = list(evaluation.get("logit_decodes", []))
        for learned_weight in learned_weights:
            weight_tag = _float_tag(learned_weight)
            for temperature in temperatures:
                temp_tag = _float_tag(temperature)
                specs.append(
                    {
                        "name": f"blend_lw{weight_tag}_cw1_t{temp_tag}",
                        "source": "blend",
                        "mode": "soft_expectation",
                        "temperature": temperature,
                        "scale": 1.0,
                        "learned_weight": learned_weight,
                        "codebook_weight": 1.0,
                    }
                )
        evaluation["logit_decodes"] = specs
    if args.token_topk:
        config.setdefault("evaluation", {})["token_topk"] = {
            "enabled": True,
            "topk": str(args.token_topk),
            "include_slots": not bool(args.token_topk_no_slots),
        }
    if bool(getattr(args, "token_topk_oracle", False)):
        config.setdefault("evaluation", {})["token_topk_oracle"] = {
            "enabled": True,
            "topk": str(args.token_topk or "1,3,5,10"),
            "source": "learned",
        }
    if bool(getattr(args, "token_alignment", False)):
        if not is_m4human:
            raise ValueError("--token-alignment currently supports only M4Human.")
        if is_absolute_root:
            raise ValueError(
                "--token-alignment requires a Stage-I HPE-token config without the absolute root head."
            )
        if str(config.get("training", {}).get("stage", "")).strip().lower() != "hpe_token":
            raise ValueError(
                "--token-alignment requires training.stage='hpe_token' (M4Human Stage-I)."
            )
        shuffle_seed = int(getattr(args, "token_alignment_shuffle_seed", 0))
        if shuffle_seed < 0:
            raise ValueError("--token-alignment-shuffle-seed must be non-negative.")
        config.setdefault("evaluation", {})["token_alignment"] = {
            "enabled": True,
            "shuffle_seed": shuffle_seed,
        }
    return config


def _dry_run_payload(config_path: Path, config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "config": str(config_path),
        "checkpoint": args.checkpoint,
        "dataset": config.get("dataset", {}).get("name"),
        "split_id": config.get("m4human", {}).get("split"),
        "eval_split": args.eval_split,
        "batch_size": config["training"].get("batch_size"),
        "num_workers": config["training"].get("num_workers"),
        "n_points": config.get("data_format", {}).get("n_points"),
        "merge_nframes": config.get("data_format", {}).get("merge_nframes"),
        "point_sampling": config.get("data_format", {}).get("point_sampling"),
        "point_ensemble": config.get("evaluation", {}).get("point_ensemble"),
        "checkpoint_ensemble": config.get("evaluation", {}).get("checkpoint_ensemble"),
        "max_batches": args.max_batches,
        "action_groups": not bool(args.no_action_groups),
        "logit_decode_diagnostics": bool(args.logit_decode_diagnostics),
        "logit_blend_weights": args.logit_blend_weights,
        "logit_decodes": config.get("evaluation", {}).get("logit_decodes"),
        "token_topk": config.get("evaluation", {}).get("token_topk"),
        "token_topk_oracle": config.get("evaluation", {}).get("token_topk_oracle"),
        "token_alignment": config.get("evaluation", {}).get("token_alignment"),
        "mmwave_tokenizer_checkpoint": config.get("pretrained", {}).get("mmwave_tokenizer"),
        "body_vqvae_checkpoint": config.get("pretrained", {}).get("vqvae"),
        "output": args.output,
    }


def main() -> None:
    args = _build_parser().parse_args()
    if args.list_configs:
        print_configs()
        return
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --list-configs is used")

    config_path = resolve_config(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = _apply_eval_overrides(load_config(config_path), args)
    _validate_production_hpe_token_config(config)
    if args.dry_run:
        print(json.dumps(_dry_run_payload(config_path, config, args), indent=2))
        return

    if _is_absolute_root_config(config):
        if args.checkpoint_ensemble:
            raise ValueError("--checkpoint-ensemble is only supported for relative HPE-token evaluation.")
        payload = run_absolute_root_head_evaluation(
            config,
            checkpoint=args.checkpoint,
            config_path=config_path,
            eval_split=args.eval_split,
            device=args.device,
            output_path=args.output,
            max_eval_batches=args.max_batches,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            include_per_action=bool(args.per_action),
        )
    else:
        payload = run_hpe_token_evaluation(
            config,
            checkpoint=args.checkpoint,
            config_path=config_path,
            eval_split=args.eval_split,
            device=args.device,
            output_path=args.output,
            max_eval_batches=args.max_batches,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            include_action_groups=not bool(args.no_action_groups),
            include_per_action=bool(args.per_action),
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
