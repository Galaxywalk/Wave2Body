# -*- coding: utf-8 -*-

import json

import pytest

from wave2body.train.mmwave_tokenizer_cli import resolve_mmwave_tokenizer_config, run_mmwave_tokenizer_cli


def test_mmwave_tokenizer_cli_rejects_body_part_grouping(tmp_path):
    cfg_path = tmp_path / "body_part_mae.json"
    cfg_path.write_text(
        json.dumps(
            {
                "base_config": str(resolve_mmwave_tokenizer_config("sota")),
                "mmwave_tokenizer": {
                    "method": "mae",
                    "patch_policy": "weak_body_part_knn",
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="spatial_fps_knn"):
        run_mmwave_tokenizer_cli(
            [
                "--dry-run",
                "--config",
                str(cfg_path),
                "--device",
                "cpu",
            ]
        )


def test_mmwave_tokenizer_cli_mmbody_yaml_stem_is_five_channel(capsys):
    run_mmwave_tokenizer_cli(
        [
            "--dry-run",
            "--config",
            "mmbody_sota",
            "--device",
            "cpu",
        ]
    )

    plan = json.loads(capsys.readouterr().out)

    assert plan["dataset"] == "mmbody"
    assert plan["method"] == "mae"
    assert plan["num_groups"] == 96
    assert plan["run_name"] == "mmbody_mae_spatial_k96_bf16_bs1024_w8_pf4"


def test_mmwave_tokenizer_cli_sota_alias_is_fast_m4human(capsys):
    run_mmwave_tokenizer_cli(["--dry-run", "--config", "sota", "--device", "cpu"])

    plan = json.loads(capsys.readouterr().out)

    assert plan["dataset"] == "m4human"
    assert plan["method"] == "mae"
    assert plan["patch_policy"] == "spatial_fps_knn"
    assert plan["num_groups"] == 96
    assert plan["run_name"] == "m4human_s2_trainonly_mae_spatial_k96_bf16_bs512_w16_pf4"
    assert plan["train_subsets"] == ["train"]
    assert plan["max_batches"]["train"] is None


def test_mmwave_tokenizer_cli_resolves_yaml_stem():
    path = resolve_mmwave_tokenizer_config("m4human_s2_trainonly")

    assert path.name == "m4human_s2_trainonly_mae_spatial_k96_fast.yaml"

