# -*- coding: utf-8 -*-

import torch

from wave2body.train.checkpoint import extract_point_transformer_state_dict, load_state_dict_shape_safe, normalized_state_dict
from wave2body.train.pipelines._paths import resolve_run_root


def test_extract_point_transformer_from_pt_pose_checkpoint():
    ckpt = {
        "model_state_dict": {
            "pt.encoder.weight": torch.ones(2, 2),
            "pt.encoder.bias": torch.zeros(2),
            "conf_head.0.weight": torch.randn(1, 2),
        }
    }

    state = extract_point_transformer_state_dict(ckpt)

    assert sorted(state) == ["encoder.bias", "encoder.weight"]


def test_shape_safe_loader_skips_incompatible_tensors():
    module = torch.nn.Linear(2, 3)
    state = {
        "weight": torch.ones(4, 2),
        "bias": torch.ones(3),
        "extra": torch.ones(1),
    }

    report = load_state_dict_shape_safe(module, state, module_name="linear")

    assert report["loaded"] == 1
    assert report["shape_skipped"][0][0] == "weight"


def test_normalized_state_dict_strips_nested_compile_wrappers():
    state = {
        "pt._orig_mod.encoder.weight": torch.ones(2, 2),
        "pt._orig_mod.encoder.bias": torch.zeros(2),
    }

    normalized = normalized_state_dict(state)

    assert sorted(normalized) == ["pt.encoder.bias", "pt.encoder.weight"]


def test_resolve_run_root_uses_work_dir(tmp_path):
    config = {"training": {"run_name": "unit"}}

    output_root, checkpoint_dir, log_dir = resolve_run_root(
        config,
        work_dir=tmp_path / "out",
        default_name="default",
        experiment_name="unit_exp",
    )

    assert output_root == tmp_path / "out"
    assert checkpoint_dir == output_root / "checkpoints"
    assert log_dir == output_root / "logs"
    assert checkpoint_dir.exists()
    assert log_dir.exists()
