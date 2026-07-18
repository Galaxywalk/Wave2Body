#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate Stage-I token-alignment metrics and render the paper figure."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SEEDS = (42, 43, 44)
EXPECTED_SAMPLES = {"s2": 133_211, "s3": 132_217}
EXPECTED_PART_TOKEN_COUNTS = {
    "L_Leg": 5,
    "R_Leg": 5,
    "SpineCore": 4,
    "L_Arm": 5,
    "R_Arm": 5,
}
CATEGORY_SPECS = (
    ("overall", "Overall"),
    ("L_Leg", "L Leg"),
    ("R_Leg", "R Leg"),
    ("SpineCore", "Spine"),
    ("L_Arm", "L Arm"),
    ("R_Arm", "R Arm"),
)
PAIRED_EDGE_COLOR = "#1E90FF"
PAIRED_FACE_COLOR = "#8EC7FF"
SAME_ACTION_OTHER_SUBJECT_EDGE_COLOR = "#FFBB00"
SAME_ACTION_OTHER_SUBJECT_FACE_COLOR = "#FFDD7F"
OTHER_ACTION_SAME_SUBJECT_EDGE_COLOR = "#FF5080"
OTHER_ACTION_SAME_SUBJECT_FACE_COLOR = "#FFA7BF"
TEXT_COLOR = "#282828"
GRID_COLOR = "#D6D6D6"
PDF_METADATA = {
    "Creator": "Wave2Body paper asset pipeline",
    "CreationDate": None,
    "ModDate": None,
}


def _finite_float(value: Any, *, label: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return out


def _agreement_value(alignment: dict[str, Any], metric: str, part: str) -> float:
    agreement_metrics = {
        "paired",
        "shuffled",
        "different_action_same_subject",
    }
    if metric in agreement_metrics:
        block = (
            alignment[metric]["overall"]
            if part == "overall"
            else alignment[metric]["by_part"][part]
        )
        return _finite_float(block["agreement"], label=f"{metric}.{part}")
    gain_metrics = {"gain", "different_action_same_subject_gain"}
    if metric in gain_metrics:
        value = (
            alignment[metric]["overall"]
            if part == "overall"
            else alignment[metric]["by_part"][part]
        )
        return _finite_float(value, label=f"{metric}.{part}")
    raise KeyError(metric)


def _stats(values_by_seed: dict[int, float | None]) -> dict[str, Any]:
    valid = [float(value) for value in values_by_seed.values() if value is not None]
    if not valid:
        mean = None
        sample_std = None
    else:
        mean = float(np.mean(valid, dtype=np.float64))
        sample_std = float(np.std(valid, ddof=1, dtype=np.float64)) if len(valid) > 1 else 0.0
    return {
        "mean": mean,
        "sample_std": sample_std,
        "num_valid_seeds": len(valid),
        "values_by_seed": {
            str(seed): (None if value is None else float(value))
            for seed, value in values_by_seed.items()
        },
    }


def _read_seed_result(
    path: Path,
    *,
    split: str,
    seed: int,
    validate_official: bool,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Stage-I token-alignment result: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{path}: missing metrics mapping")
    alignment = metrics.get("token_alignment")
    if not isinstance(alignment, dict):
        raise ValueError(f"{path}: missing metrics.token_alignment")

    paired_overall = _agreement_value(alignment, "paired", "overall")
    token_acc = _finite_float(metrics.get("token_acc"), label=f"{path}: metrics.token_acc")
    if not math.isclose(paired_overall, token_acc, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError(
            f"{path}: paired overall agreement {paired_overall} does not match "
            f"quantized token_acc {token_acc}"
        )

    for metric in (
        "paired",
        "shuffled",
        "gain",
        "different_action_same_subject",
        "different_action_same_subject_gain",
    ):
        for part, _ in CATEGORY_SPECS:
            _agreement_value(alignment, metric, part)

    protocol = alignment.get("negative_protocol", {})
    signature = str(protocol.get("permutation_signature", ""))
    if not signature.startswith("sha256:"):
        raise ValueError(f"{path}: invalid permutation signature {signature!r}")
    for key in ("verified_same_action", "verified_different_subject", "verified_bijection"):
        if protocol.get(key) is not True:
            raise ValueError(f"{path}: negative protocol did not verify {key}")
    different_action_protocol = alignment.get(
        "different_action_same_subject_protocol", {}
    )
    different_action_signature = str(
        different_action_protocol.get("permutation_signature", "")
    )
    if not different_action_signature.startswith("sha256:"):
        raise ValueError(
            f"{path}: invalid different-action permutation signature "
            f"{different_action_signature!r}"
        )
    for key in ("verified_same_subject", "verified_different_action", "verified_bijection"):
        if different_action_protocol.get(key) is not True:
            raise ValueError(
                f"{path}: different-action protocol did not verify {key}"
            )

    spearman_block = alignment.get("spearman", {})
    spearman_value = spearman_block.get("token_mismatch_vs_root_relative_mpjpe")
    if spearman_value is not None:
        spearman_value = _finite_float(spearman_value, label=f"{path}: Spearman rho")
        if not -1.0 <= spearman_value <= 1.0:
            raise ValueError(f"{path}: Spearman rho is outside [-1, 1]")

    part_token_counts = {
        str(key): int(value)
        for key, value in alignment.get("part_token_counts", {}).items()
    }
    num_samples = int(alignment.get("num_samples", -1))
    tokens_per_sample = int(alignment.get("tokens_per_sample", -1))

    if validate_official:
        if str(payload.get("eval_split")) != "test":
            raise ValueError(f"{path}: expected official test split")
        if str(payload.get("split_id")).lower() != split:
            raise ValueError(f"{path}: expected split_id={split!r}")
        if num_samples != EXPECTED_SAMPLES[split]:
            raise ValueError(
                f"{path}: expected {EXPECTED_SAMPLES[split]} samples, got {num_samples}"
            )
        if tokens_per_sample != 24:
            raise ValueError(f"{path}: expected 24 tokens per sample, got {tokens_per_sample}")
        if part_token_counts != EXPECTED_PART_TOKEN_COUNTS:
            raise ValueError(
                f"{path}: expected part-token layout {EXPECTED_PART_TOKEN_COUNTS}, "
                f"got {part_token_counts}"
            )
        if int(protocol.get("shuffle_seed", -1)) != 0:
            raise ValueError(f"{path}: expected shuffle seed 0")
        if int(different_action_protocol.get("shuffle_seed", -1)) != 0:
            raise ValueError(
                f"{path}: expected different-action shuffle seed 0"
            )
        if f"seed{seed}" not in str(payload.get("checkpoint", "")).lower():
            raise ValueError(f"{path}: checkpoint does not identify seed{seed}")

    return {
        "path": str(path),
        "checkpoint": str(payload.get("checkpoint")),
        "num_samples": num_samples,
        "tokens_per_sample": tokens_per_sample,
        "part_token_counts": part_token_counts,
        "signature": signature,
        "different_action_signature": different_action_signature,
        "spearman": spearman_value,
        "alignment": alignment,
    }


def aggregate_results(
    input_root: str | Path,
    *,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    validate_official: bool = True,
) -> dict[str, Any]:
    """Load the six per-seed JSON files and return a validated aggregate."""

    root = Path(input_root).expanduser()
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("Expected at least one model seed")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError(f"Model seeds must be unique, got {seed_values}")

    raw: dict[str, dict[int, dict[str, Any]]] = {}
    for split in ("s2", "s3"):
        raw[split] = {}
        for seed in seed_values:
            path = root / split / f"seed{seed}" / "metrics.json"
            raw[split][seed] = _read_seed_result(
                path,
                split=split,
                seed=seed,
                validate_official=validate_official,
            )

        signatures = {item["signature"] for item in raw[split].values()}
        if len(signatures) != 1:
            raise ValueError(
                f"{split}: permutation signatures differ across seeds: {sorted(signatures)}"
            )
        different_action_signatures = {
            item["different_action_signature"] for item in raw[split].values()
        }
        if len(different_action_signatures) != 1:
            raise ValueError(
                f"{split}: different-action permutation signatures differ across "
                f"seeds: {sorted(different_action_signatures)}"
            )
        sample_counts = {item["num_samples"] for item in raw[split].values()}
        if len(sample_counts) != 1:
            raise ValueError(f"{split}: sample counts differ across seeds: {sorted(sample_counts)}")
        token_counts = {item["tokens_per_sample"] for item in raw[split].values()}
        if len(token_counts) != 1:
            raise ValueError(
                f"{split}: tokens-per-sample differ across seeds: {sorted(token_counts)}"
            )

    aggregate: dict[str, Any] = {
        "schema_version": 2,
        "description": "Stage-I quantized body-token alignment on M4Human official tests",
        "model_seeds": list(seed_values),
        "control_protocols": {
            "same_action_other_subject": {
                "strategy": "same_action_cross_subject_permutation",
                "shuffle_seed": 0,
            },
            "other_action_same_subject": {
                "strategy": "different_action_same_subject_permutation",
                "shuffle_seed": 0,
            },
        },
        "category_order": [label for _, label in CATEGORY_SPECS],
        "splits": {},
    }

    for split in ("s2", "s3"):
        split_items = raw[split]
        categories: dict[str, Any] = {}
        metric_sources = (
            ("paired", "paired"),
            ("same_action_other_subject", "shuffled"),
            ("other_action_same_subject", "different_action_same_subject"),
            ("gain_over_same_action_other_subject", "gain"),
            (
                "gain_over_other_action_same_subject",
                "different_action_same_subject_gain",
            ),
        )
        for part, label in CATEGORY_SPECS:
            categories[label] = {
                output_name: _stats(
                    {
                        seed: _agreement_value(
                            item["alignment"], source_name, part
                        )
                        for seed, item in split_items.items()
                    }
                )
                for output_name, source_name in metric_sources
            }

        spearman = _stats(
            {seed: item["spearman"] for seed, item in split_items.items()}
        )
        overall = {
            name: categories["Overall"][name]
            for name, _ in metric_sources
        }
        overall["spearman"] = spearman
        first_item = split_items[seed_values[0]]
        aggregate["splits"][split] = {
            "num_samples": int(first_item["num_samples"]),
            "tokens_per_sample": int(first_item["tokens_per_sample"]),
            "num_tokens": int(first_item["num_samples"] * first_item["tokens_per_sample"]),
            "part_token_counts": first_item["part_token_counts"],
            "permutation_signatures": {
                "same_action_other_subject": first_item["signature"],
                "other_action_same_subject": first_item["different_action_signature"],
            },
            "categories": categories,
            "overall": overall,
            "source_results": {
                str(seed): {
                    "path": item["path"],
                    "checkpoint": item["checkpoint"],
                }
                for seed, item in split_items.items()
            },
        }

    return aggregate


def _summary_rows(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("s2", "s3"):
        split_summary = aggregate["splits"][split]
        for label in aggregate["category_order"]:
            category = split_summary["categories"][label]
            spearman = split_summary["overall"]["spearman"] if label == "Overall" else None
            rows.append(
                {
                    "split": split,
                    "category": label,
                    "paired_mean": category["paired"]["mean"],
                    "paired_sample_std": category["paired"]["sample_std"],
                    "same_action_other_subject_mean": category[
                        "same_action_other_subject"
                    ]["mean"],
                    "same_action_other_subject_sample_std": category[
                        "same_action_other_subject"
                    ]["sample_std"],
                    "other_action_same_subject_mean": category[
                        "other_action_same_subject"
                    ]["mean"],
                    "other_action_same_subject_sample_std": category[
                        "other_action_same_subject"
                    ]["sample_std"],
                    "gain_over_same_action_other_subject_mean": category[
                        "gain_over_same_action_other_subject"
                    ]["mean"],
                    "gain_over_same_action_other_subject_sample_std": category[
                        "gain_over_same_action_other_subject"
                    ]["sample_std"],
                    "gain_over_other_action_same_subject_mean": category[
                        "gain_over_other_action_same_subject"
                    ]["mean"],
                    "gain_over_other_action_same_subject_sample_std": category[
                        "gain_over_other_action_same_subject"
                    ]["sample_std"],
                    "spearman_mean": None if spearman is None else spearman["mean"],
                    "spearman_sample_std": None if spearman is None else spearman["sample_std"],
                    "num_model_seeds": len(aggregate["model_seeds"]),
                }
            )
    return rows


def write_aggregate_outputs(
    aggregate: dict[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "stage1_token_alignment_summary.json"
    csv_path = output_root / "stage1_token_alignment_summary.csv"
    json_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    rows = _summary_rows(aggregate)
    fieldnames = list(rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return csv_path, json_path


def plot_s2_figure(
    aggregate: dict[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    pdf_path = output_root / "stage1_token_alignment_s2.pdf"
    png_path = output_root / "stage1_token_alignment_s2.png"

    labels = list(aggregate["category_order"])
    categories = aggregate["splits"]["s2"]["categories"]
    series_specs = (
        (
            "Paired",
            "paired",
            PAIRED_EDGE_COLOR,
            PAIRED_FACE_COLOR,
            r"\\\\",
        ),
        (
            "Same action / other subject",
            "same_action_other_subject",
            SAME_ACTION_OTHER_SUBJECT_EDGE_COLOR,
            SAME_ACTION_OTHER_SUBJECT_FACE_COLOR,
            "//",
        ),
        (
            "Other action / same subject",
            "other_action_same_subject",
            OTHER_ACTION_SAME_SUBJECT_EDGE_COLOR,
            OTHER_ACTION_SAME_SUBJECT_FACE_COLOR,
            "xx",
        ),
    )
    series_values = {
        metric: (
            np.asarray([categories[label][metric]["mean"] for label in labels])
            * 100.0
        )
        for _, metric, _, _, _ in series_specs
    }
    series_stds = {
        metric: (
            np.asarray(
                [categories[label][metric]["sample_std"] for label in labels]
            )
            * 100.0
        )
        for _, metric, _, _, _ in series_specs
    }

    x = np.arange(len(labels), dtype=np.float64)
    total_width = 0.78
    bar_width = total_width / len(series_specs)
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "DejaVu Sans",
            "sans-serif",
        ],
        "font.size": 8,
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "axes.labelsize": 8,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 7.4,
        "legend.fontsize": 5.5,
        "axes.linewidth": 0.8,
        "hatch.linewidth": 0.85,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(3.35, 1.95))
        ax.axvspan(-0.5, 0.5, color="#F4F6F8", zorder=0)
        error_kw = {
            "ecolor": TEXT_COLOR,
            "elinewidth": 0.6,
            "capsize": 1.5,
            "capthick": 0.6,
        }
        handles = []
        for index, (legend_label, metric, edge_color, face_color, hatch) in enumerate(
            series_specs
        ):
            offset = (index - (len(series_specs) - 1) / 2) * bar_width
            bars = ax.bar(
                x + offset,
                series_values[metric],
                width=bar_width * 0.92,
                yerr=series_stds[metric],
                color=face_color,
                edgecolor=edge_color,
                hatch=hatch,
                linewidth=0.9,
                error_kw=error_kw,
                label=legend_label,
                zorder=3,
            )
            handles.append(bars)

        ax.set_xlim(-0.55, len(labels) - 0.45)
        ax.set_ylim(0.0, 30.0)
        ax.set_yticks(np.arange(0.0, 30.1, 5.0))
        ax.set_ylabel("Token Accuracy (%)")
        ax.set_xticks(x, labels)
        ax.grid(
            axis="y",
            color=GRID_COLOR,
            linewidth=0.65,
            alpha=0.75,
            linestyle="--",
            zorder=0,
        )
        ax.axvline(0.5, color="#B8B8B8", linewidth=0.85, zorder=1)
        ax.tick_params(width=0.8)
        fig.legend(
            handles=handles,
            labels=[item[0] for item in series_specs],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.91),
            ncol=3,
            borderaxespad=0.0,
            handlelength=1.0,
            handletextpad=0.3,
            columnspacing=0.35,
            labelspacing=0.2,
            frameon=False,
        )

        fig.subplots_adjust(left=0.16, right=0.99, bottom=0.22, top=0.82)
        fig.savefig(
            pdf_path,
            bbox_inches="tight",
            pad_inches=0.02,
            metadata=PDF_METADATA,
        )
        fig.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close(fig)

    return pdf_path, png_path


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("Expected a comma-separated seed list")
    return seeds


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        required=True,
        help="Artifact directory containing {s2,s3}/seed{42,43,44}/metrics.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/002_stage1_token_alignment",
        help="Repository directory for aggregate tables and the S2 figure.",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=DEFAULT_SEEDS,
        help="Comma-separated model seeds (default: 42,43,44).",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Skip official sample-count and checkpoint-identity checks for tests.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    aggregate = aggregate_results(
        args.input_root,
        seeds=args.seeds,
        validate_official=not bool(args.allow_synthetic),
    )
    csv_path, json_path = write_aggregate_outputs(aggregate, args.output_dir)
    pdf_path, png_path = plot_s2_figure(aggregate, args.output_dir)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
