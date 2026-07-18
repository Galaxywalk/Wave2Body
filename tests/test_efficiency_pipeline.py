import json
from pathlib import Path

import pytest

from scripts.efficiency import run_m4human_efficiency as eff


def _args(*extra: str):
    return eff._build_parser().parse_args(
        [
            "--run-name",
            "unit",
            "--work-dir",
            "/tmp/unit_efficiency",
            "--gpus",
            "0,1,3",
            "--profile-batches",
            "1",
            "--flops-batches",
            "1",
            "--latency-batches",
            "1",
            "--latency-repeats",
            "1",
            *extra,
        ]
    )


def test_parse_gpu_list_rejects_empty_and_duplicates():
    assert eff.parse_gpu_list("0,1,3") == ["0", "1", "3"]
    with pytest.raises(ValueError):
        eff.parse_gpu_list("")
    with pytest.raises(ValueError):
        eff.parse_gpu_list("0,1,1")


def test_build_job_groups_assigns_default_gpu_schedule(tmp_path):
    args = _args("--work-dir", str(tmp_path))
    groups = eff.build_job_groups(args, tmp_path)
    by_gpu = {group.gpu: [job.name for job in group.jobs] for group in groups}

    assert by_gpu["0"] == ["pt_training_flops"]
    assert by_gpu["1"] == ["mmwave_tokenizer_training_flops", "hpe_token_training_flops"]
    assert by_gpu["3"] == [
        "human_pose_tokenizer_stage_a_training_flops",
        "absolute_root_training_flops",
        "inference",
    ]

    pt_job = next(job for group in groups for job in group.jobs if job.name == "pt_training_flops")
    assert pt_job.batch_arg == "--point-transformer-batch-size"
    assert pt_job.batch_candidates == (96, 64, 48, 32)
    assert str(eff.POINT_TRANSFORMER_CONFIG) in pt_job.command
    inference_job = next(job for group in groups for job in group.jobs if job.name == "inference")
    assert "--single-sample-latency-batches" in inference_job.command


def test_build_job_groups_honors_gpu_map(tmp_path):
    args = _args("--work-dir", str(tmp_path), "--gpu-map", "pt=3,radar=0,body=1,inference=0")
    groups = eff.build_job_groups(args, tmp_path)
    by_name = {job.name: job.gpu for group in groups for job in group.jobs}

    assert by_name["pt_training_flops"] == "3"
    assert by_name["mmwave_tokenizer_training_flops"] == "0"
    assert by_name["hpe_token_training_flops"] == "0"
    assert by_name["human_pose_tokenizer_stage_a_training_flops"] == "1"
    assert by_name["absolute_root_training_flops"] == "1"
    assert by_name["inference"] == "0"


def _write_training_summary(root: Path, dirname: str, stage: str, total_flops: float, params: int = 10) -> None:
    out = root / dirname
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "stages": [
            {
                "stage": stage,
                "config_path": f"{stage}.yaml",
                "params": {"trainable": params, "total": params},
                "flops": {
                    "avg_batch_samples": 4.0,
                    "epochs": 2,
                    "profile_samples": 8,
                    "profile_batches": 2,
                    "profile_train_batch_coverage": 0.1,
                },
            }
        ],
        "comparison_table": [{"Total Training FLOPs": total_flops}],
    }
    (out / "training_flops_summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_consolidate_results_writes_json_markdown_and_csv(tmp_path):
    _write_training_summary(tmp_path, "pt_training_flops", "point_transformer", 1000.0)
    _write_training_summary(tmp_path, "mmwave_tokenizer_training_flops", "mmwave_tokenizer", 100.0)
    _write_training_summary(
        tmp_path,
        "human_pose_tokenizer_stage_a_training_flops",
        "human_pose_tokenizer_stage_a",
        200.0,
    )
    _write_training_summary(tmp_path, "hpe_token_training_flops", "hpe_token", 300.0)
    _write_training_summary(tmp_path, "absolute_root_training_flops", "absolute_root_head", 400.0)

    inference_dir = tmp_path / "inference"
    inference_dir.mkdir()
    (inference_dir / "inference_efficiency_summary.json").write_text(
        json.dumps(
            {
                "comparison_table": [
                    {
                        "Method": "point_transformer",
                        "Total Params": 10,
                        "FLOPs / Sample": 100.0,
                        "Latency ms / Sample": 1.0,
                        "FPS": 1000.0,
                        "Latency B1 ms / Sample": 2.0,
                        "Throughput B192 FPS": 1000.0,
                        "Peak Memory B192 MiB": 123.4,
                        "Peak Reserved Memory B192 MiB": 256.0,
                        "Batch Size": 4,
                        "Split": "test",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = eff.consolidate_results(tmp_path, run_status=[{"job": "unit", "status": "ok"}])

    assert report["wave2body_total_training_flops"] == 1000.0
    assert report["wave2body_supervised_training_flops"] == 700.0
    assert report["ratios"]["pt_over_wave2body_training_flops"] == 1.0
    assert (tmp_path / "efficiency_consolidated_summary.json").exists()
    assert (tmp_path / "efficiency_consolidated_summary.md").exists()
    assert "Wave2Body" in (tmp_path / "training_figure_source.csv").read_text(encoding="utf-8")
    inference_csv = (tmp_path / "inference_table_source.csv").read_text(encoding="utf-8")
    assert "point_transformer" in inference_csv
    assert "latency_b1_ms_per_sample" in inference_csv
    assert "123.4" in inference_csv
