# -*- coding: utf-8 -*-

import json

import pytest

from scripts.visualization.plot_stage1_token_alignment import (
    EXPECTED_PART_TOKEN_COUNTS,
    aggregate_results,
    plot_s2_figure,
    write_aggregate_outputs,
)


def _write_result(root, *, split: str, seed: int) -> None:
    split_offset = 0.00 if split == "s2" else 0.05
    seed_offset = (seed - 42) * 0.02
    paired_overall = 0.50 + split_offset + seed_offset
    shuffled_overall = 0.10 + split_offset / 5.0 + seed_offset / 4.0
    other_action_overall = 0.16 + split_offset / 4.0 + seed_offset / 5.0
    part_offsets = {
        "L_Leg": -0.04,
        "R_Leg": -0.02,
        "SpineCore": 0.06,
        "L_Arm": 0.01,
        "R_Arm": 0.03,
    }

    paired_parts = {
        name: {
            "count": 40,
            "matches": 20,
            "agreement": paired_overall + offset,
        }
        for name, offset in part_offsets.items()
    }
    shuffled_parts = {
        name: {
            "count": 40,
            "matches": 4,
            "agreement": shuffled_overall + offset / 10.0,
        }
        for name, offset in part_offsets.items()
    }
    other_action_parts = {
        name: {
            "count": 40,
            "matches": 6,
            "agreement": other_action_overall + offset / 8.0,
        }
        for name, offset in part_offsets.items()
    }
    gain_parts = {
        name: paired_parts[name]["agreement"] - shuffled_parts[name]["agreement"]
        for name in part_offsets
    }
    other_action_gain_parts = {
        name: paired_parts[name]["agreement"] - other_action_parts[name]["agreement"]
        for name in part_offsets
    }
    alignment = {
        "num_samples": 8,
        "tokens_per_sample": 24,
        "part_token_counts": EXPECTED_PART_TOKEN_COUNTS,
        "paired": {
            "overall": {
                "count": 192,
                "matches": 96,
                "agreement": paired_overall,
            },
            "by_part": paired_parts,
        },
        "shuffled": {
            "overall": {
                "count": 192,
                "matches": 19,
                "agreement": shuffled_overall,
            },
            "by_part": shuffled_parts,
        },
        "gain": {
            "overall": paired_overall - shuffled_overall,
            "by_part": gain_parts,
        },
        "different_action_same_subject": {
            "overall": {
                "count": 192,
                "matches": 31,
                "agreement": other_action_overall,
            },
            "by_part": other_action_parts,
        },
        "different_action_same_subject_gain": {
            "overall": paired_overall - other_action_overall,
            "by_part": other_action_gain_parts,
        },
        "spearman": {
            "token_mismatch_vs_root_relative_mpjpe": 0.40 + (seed - 42) * 0.01,
            "valid": True,
        },
        "negative_protocol": {
            "strategy": "same_action_cross_subject_permutation",
            "shuffle_seed": 0,
            "permutation_signature": f"sha256:{split * 32}",
            "verified_same_action": True,
            "verified_different_subject": True,
            "verified_bijection": True,
        },
        "different_action_same_subject_protocol": {
            "strategy": "different_action_same_subject_permutation",
            "shuffle_seed": 0,
            "permutation_signature": f"sha256:{split[::-1] * 32}",
            "verified_same_subject": True,
            "verified_different_action": True,
            "verified_bijection": True,
        },
    }
    payload = {
        "stage": "hpe_token",
        "dataset": "m4human",
        "eval_split": "test",
        "split_id": split,
        "checkpoint": f"/tmp/model_{split}_seed{seed}/best.pth",
        "metrics": {
            "token_acc": paired_overall,
            "token_alignment": alignment,
        },
    }
    path = root / split / f"seed{seed}" / "metrics.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_stage1_token_alignment_aggregate_and_render(tmp_path):
    input_root = tmp_path / "raw"
    output_dir = tmp_path / "paper"
    for split in ("s2", "s3"):
        for seed in (42, 43, 44):
            _write_result(input_root, split=split, seed=seed)

    aggregate = aggregate_results(
        input_root,
        seeds=(42, 43, 44),
        validate_official=False,
    )
    csv_path, json_path = write_aggregate_outputs(aggregate, output_dir)
    pdf_path, png_path = plot_s2_figure(aggregate, output_dir)

    assert aggregate["splits"]["s3"]["overall"]["paired"]["mean"] == pytest.approx(0.57)
    assert aggregate["splits"]["s3"]["overall"]["paired"]["sample_std"] == pytest.approx(0.02)
    assert aggregate["splits"]["s3"]["overall"]["same_action_other_subject"][
        "mean"
    ] == pytest.approx(0.115)
    assert aggregate["splits"]["s3"]["overall"]["other_action_same_subject"][
        "mean"
    ] == pytest.approx(0.1765)
    assert aggregate["splits"]["s3"]["overall"]["spearman"]["mean"] == pytest.approx(0.41)
    assert aggregate["splits"]["s3"]["overall"]["spearman"]["sample_std"] == pytest.approx(0.01)
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 13
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 2
    assert saved["model_seeds"] == [42, 43, 44]
    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf_path.stat().st_size > 1_000
    assert png_path.stat().st_size > 1_000
