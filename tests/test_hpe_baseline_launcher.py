import json
import subprocess
import sys


def _run_launcher(*args: str):
    return subprocess.run(
        [sys.executable, "scripts/hpe_baselines/train.py", *args],
        check=True,
        text=True,
        capture_output=True,
    )


def test_hpe_baseline_launcher_lists_configs():
    result = _run_launcher("--list-configs")

    assert "mmdiff_mmbody" in result.stdout
    assert "mmmesh_mmbody" in result.stdout
    assert "mmdiff_m4human_s2" in result.stdout
    assert "mmmesh_m4human_s3" in result.stdout


def test_hpe_baseline_launcher_dry_run_resolves_mmdiff_smoke_plan():
    result = _run_launcher("--config", "mmdiff_mmbody", "--smoke", "--dry-run", "--device", "cpu")
    payload = json.loads(result.stdout)

    assert payload["baseline"] == "mmdiff"
    assert payload["dataset"] == "mmbody"
    assert payload["batch_size"] == 2
    assert payload["epochs"] == 1
    assert payload["max_batches"] == {"train": 1, "val": 1, "test": 1}
    assert payload["root_alignment"] == {
        "normalized": False,
        "input_root_normalized": False,
        "input_root_crop": True,
        "target_root_normalized": False,
    }


def test_hpe_baseline_launcher_dry_run_resolves_mmmesh_plan():
    result = _run_launcher("--config", "mmmesh_mmbody", "--dry-run", "--device", "cpu")
    payload = json.loads(result.stdout)

    assert payload["baseline"] == "mmmesh"
    assert payload["root_alignment"] == {
        "normalized": False,
        "input_root_normalized": False,
        "input_root_crop": True,
        "target_root_normalized": False,
    }
    assert "SMPL/body-target" in payload["notes"]


def test_hpe_baseline_launcher_dry_run_resolves_m4human_plan():
    result = _run_launcher("--config", "mmdiff_m4human_s3", "--dry-run", "--device", "cpu")
    payload = json.loads(result.stdout)

    assert payload["baseline"] == "mmdiff"
    assert payload["dataset"] == "m4human"
    assert payload["batch_size"] == 128
    assert payload["epochs"] == 25
    assert "m4human_mmdiff_s3" in payload["run_name"]
    assert payload["root_alignment"] == {
        "normalized": False,
        "input_root_normalized": False,
        "input_root_crop": False,
        "target_root_normalized": False,
    }
