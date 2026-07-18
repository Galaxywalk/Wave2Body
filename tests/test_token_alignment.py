# -*- coding: utf-8 -*-

import numpy as np
import pytest
import torch

from scripts.hpe_token.evaluate import (
    _apply_eval_overrides,
    _build_parser,
    _dry_run_payload,
    resolve_config,
)
from wave2body.evaluate.token_alignment import (
    TokenAlignmentCollector,
    build_different_action_same_subject_permutation,
    build_same_action_cross_subject_permutation,
    spearman_rank_correlation,
    summarize_exact_agreement,
    summarize_topk_agreement,
)
from wave2body.models.human_body_tokenizer.part_schema import create_part_layout
from wave2body.train.common import load_config
from wave2body.train.pipelines.hpe_token import evaluate_hpe_token
from wave2body.utils.runtime_utils import RuntimeConfig


def _layout():
    return create_part_layout(
        num_joints=4,
        part_names=("A", "B"),
        part_joints={"A": (0, 1), "B": (2, 3)},
        part_token_num={"A": 2, "B": 1},
        part_codebook_size={"A": 3, "B": 2},
    )


def _runtime() -> RuntimeConfig:
    return RuntimeConfig(
        device=torch.device("cpu"),
        precision="fp32",
        amp_enabled=False,
        autocast_dtype=None,
        use_grad_scaler=False,
        deterministic=False,
        tf32=False,
        matmul_precision="high",
        compile_enabled=False,
        compile_mode=None,
        compile_dynamic=False,
        compile_fullgraph=False,
    )


def test_same_action_cross_subject_permutation_is_deterministic_bijection():
    indicators = np.asarray(
        [
            [1, 10, 0],
            [1, 10, 1],
            [1, 10, 2],
            [2, 10, 0],
            [2, 10, 1],
            [3, 10, 0],
            [4, 20, 0],
            [4, 20, 1],
            [5, 20, 0],
            [5, 20, 1],
        ],
        dtype=np.int64,
    )

    first = build_same_action_cross_subject_permutation(indicators, seed=0)
    second = build_same_action_cross_subject_permutation(indicators, seed=0)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.sort(first), np.arange(indicators.shape[0]))
    np.testing.assert_array_equal(indicators[:, 1], indicators[first, 1])
    assert np.all(indicators[:, 0] != indicators[first, 0])


@pytest.mark.parametrize(
    "indicators, message",
    [
        ([[1, 10, 0], [1, 10, 1]], "found one subject"),
        (
            [[1, 10, 0], [1, 10, 1], [1, 10, 2], [2, 10, 0]],
            "exceeds half",
        ),
    ],
)
def test_same_action_cross_subject_permutation_rejects_infeasible_groups(indicators, message):
    with pytest.raises(ValueError, match=message):
        build_same_action_cross_subject_permutation(np.asarray(indicators), seed=0)


def test_different_action_same_subject_permutation_is_deterministic_bijection():
    indicators = np.asarray(
        [
            [1, 10, 0],
            [1, 10, 1],
            [1, 10, 2],
            [1, 20, 0],
            [1, 20, 1],
            [1, 30, 0],
            [2, 10, 0],
            [2, 10, 1],
            [2, 20, 0],
            [2, 20, 1],
        ],
        dtype=np.int64,
    )

    first = build_different_action_same_subject_permutation(indicators, seed=0)
    second = build_different_action_same_subject_permutation(indicators, seed=0)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.sort(first), np.arange(indicators.shape[0]))
    np.testing.assert_array_equal(indicators[:, 0], indicators[first, 0])
    assert np.all(indicators[:, 1] != indicators[first, 1])


@pytest.mark.parametrize(
    "indicators, message",
    [
        ([[1, 10, 0], [1, 10, 1]], "found one action"),
        (
            [[1, 10, 0], [1, 10, 1], [1, 10, 2], [1, 20, 0]],
            "exceeds half",
        ),
    ],
)
def test_different_action_same_subject_permutation_rejects_infeasible_groups(
    indicators, message
):
    with pytest.raises(ValueError, match=message):
        build_different_action_same_subject_permutation(
            np.asarray(indicators), seed=0
        )


def test_exact_agreement_reports_overall_and_parts():
    target = np.asarray([[0, 1, 3], [2, 0, 4]])
    pred = np.asarray([[0, 2, 3], [1, 0, 3]])

    summary = summarize_exact_agreement(pred, target, layout=_layout())

    assert summary["overall"] == {"count": 6, "matches": 3, "agreement": 0.5}
    assert summary["by_part"]["A"] == {"count": 4, "matches": 2, "agreement": 0.5}
    assert summary["by_part"]["B"] == {"count": 2, "matches": 1, "agreement": 0.5}


def test_topk_agreement_reports_overall_and_parts():
    target = np.asarray([[0, 1, 3], [2, 0, 4]])
    ranked = np.asarray(
        [
            [[0, 1, 2], [2, 1, 0], [3, 4, 0]],
            [[1, 2, 0], [0, 2, 1], [3, 4, 0]],
        ]
    )

    summary = summarize_topk_agreement(
        ranked, target, layout=_layout(), topk=(1, 2, 3)
    )

    assert summary["overall"]["top1"] == pytest.approx(0.5)
    assert summary["overall"]["top2"] == pytest.approx(1.0)
    assert summary["overall"]["top3"] == pytest.approx(1.0)
    assert summary["by_part"]["A"]["top1"] == pytest.approx(0.5)
    assert summary["by_part"]["B"]["top1"] == pytest.approx(0.5)


def test_spearman_rank_correlation_is_tie_aware_and_handles_constants():
    rho = spearman_rank_correlation(
        np.asarray([0.0, 0.0, 1.0, 2.0]),
        np.asarray([0.0, 1.0, 1.0, 2.0]),
    )

    assert rho == pytest.approx(5.0 / 6.0)
    assert spearman_rank_correlation(np.ones(4), np.arange(4)) is None


def test_token_alignment_collector_reports_gain_and_spearman():
    collector = TokenAlignmentCollector(
        _layout(), shuffle_seed=0, topk=(1, 2, 3)
    )
    target = torch.tensor(
        [
            [0, 0, 3],
            [0, 0, 3],
            [1, 1, 4],
            [1, 1, 4],
        ]
    )
    pred = torch.tensor(
        [
            [0, 0, 3],
            [2, 0, 3],
            [0, 0, 4],
            [0, 0, 3],
        ]
    )
    ranked = torch.tensor(
        [
            [[0, 1, 2], [0, 1, 2], [3, 4, 0]],
            [[2, 0, 1], [0, 1, 2], [3, 4, 0]],
            [[0, 1, 2], [0, 1, 2], [4, 3, 0]],
            [[0, 1, 2], [0, 1, 2], [3, 4, 0]],
        ]
    )
    indicators = torch.tensor(
        [
            [1, 10, 0],
            [1, 20, 1],
            [2, 10, 0],
            [2, 20, 1],
        ]
    )
    collector.update(
        pred_indices=pred[:2],
        target_indices=target[:2],
        indicators=indicators[:2],
        root_relative_mpjpe=torch.tensor([0.1, 0.2]),
        ranked_indices=ranked[:2],
    )
    collector.update(
        pred_indices=pred[2:],
        target_indices=target[2:],
        indicators=indicators[2:],
        root_relative_mpjpe=torch.tensor([0.2, 0.4]),
        ranked_indices=ranked[2:],
    )

    summary = collector.summary()

    assert summary["num_samples"] == 4
    assert summary["tokens_per_sample"] == 3
    assert summary["part_token_counts"] == {"A": 2, "B": 1}
    assert summary["paired"]["overall"]["agreement"] == pytest.approx(0.5)
    assert summary["shuffled"]["overall"]["agreement"] == pytest.approx(5.0 / 12.0)
    assert summary["gain"]["overall"] == pytest.approx(1.0 / 12.0)
    assert summary["different_action_same_subject"]["overall"][
        "agreement"
    ] == pytest.approx(0.5)
    assert summary["different_action_same_subject_gain"]["overall"] == pytest.approx(0.0)
    assert summary["spearman"]["token_mismatch_vs_root_relative_mpjpe"] == pytest.approx(
        0.9486832980505139
    )
    assert summary["spearman"]["valid"] is True
    topk_spearman = summary["spearman"][
        "topk_miss_rate_vs_root_relative_mpjpe"
    ]
    assert topk_spearman["top1"] == pytest.approx(0.9486832980505139)
    assert topk_spearman["top2"] is None
    assert topk_spearman["top3"] is None
    assert summary["codebook_topk"]["paired"]["overall"]["top1"] == pytest.approx(0.5)
    assert summary["codebook_topk"]["paired"]["overall"]["top2"] == pytest.approx(1.0)
    assert summary["codebook_topk"]["paired"]["overall"]["top3"] == pytest.approx(1.0)
    assert summary["negative_protocol"]["verified_same_action"] is True
    assert summary["negative_protocol"]["verified_different_subject"] is True
    assert summary["negative_protocol"]["verified_bijection"] is True
    protocol = summary["different_action_same_subject_protocol"]
    assert protocol["verified_same_subject"] is True
    assert protocol["verified_different_action"] is True
    assert protocol["verified_bijection"] is True
    assert summary["negative_protocol"]["permutation_signature"].startswith("sha256:")


def test_token_alignment_cli_and_dry_run_expose_protocol():
    args = _build_parser().parse_args(
        [
            "--config",
            "m4human_stage1_s2",
            "--checkpoint",
            "/tmp/model.pth",
            "--token-alignment",
            "--token-alignment-shuffle-seed",
            "7",
            "--dry-run",
        ]
    )
    config_path = resolve_config(args.config)
    config = _apply_eval_overrides(
        load_config(config_path, resolve_datasets=False),
        args,
    )

    payload = _dry_run_payload(config_path, config, args)

    assert config["evaluation"]["token_alignment"] == {"enabled": True, "shuffle_seed": 7}
    assert payload["token_alignment"] == {"enabled": True, "shuffle_seed": 7}


@pytest.mark.parametrize(
    "config_name, shuffle_seed, message",
    [
        ("m4human_abs_s2", 0, "without the absolute root head"),
        ("m4human_stage1_s2", -1, "must be non-negative"),
    ],
)
def test_token_alignment_cli_rejects_unsupported_configs(config_name, shuffle_seed, message):
    args = _build_parser().parse_args(
        [
            "--config",
            config_name,
            "--checkpoint",
            "/tmp/model.pth",
            "--token-alignment",
            "--token-alignment-shuffle-seed",
            str(shuffle_seed),
        ]
    )

    with pytest.raises(ValueError, match=message):
        _apply_eval_overrides(
            load_config(resolve_config(config_name), resolve_datasets=False),
            args,
        )


def test_token_alignment_integrates_with_hpe_evaluator():
    layout = create_part_layout(
        num_joints=4,
        part_names=("A", "B"),
        part_joints={"A": (0, 1), "B": (2, 3)},
        part_token_num={"A": 2, "B": 1},
        part_codebook_size={"A": 5, "B": 5},
    )
    target_indices = torch.tensor(
        [
            [0, 0, 5],
            [0, 0, 5],
            [1, 1, 6],
            [1, 1, 6],
        ]
    )
    pred_indices = torch.tensor(
        [
            [0, 0, 5],
            [2, 0, 5],
            [0, 0, 6],
            [0, 0, 5],
        ]
    )

    class DummyVQVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layout = layout
            self.quantizers = {
                "A": type("Q", (), {"codebook": torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])})(),
                "B": type("Q", (), {"codebook": torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])})(),
            }

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vqvae = DummyVQVAE()

        def forward(self, mmwave_points, target_poses):
            batch_size = int(target_poses.shape[0])
            pred_pose = target_poses.clone()
            pred_pose[:, 1, 0] = torch.tensor([0.2, 0.4, 0.4, 0.8])
            zeros = torch.zeros(batch_size, 3, 2)
            return {
                "pred_z_e": zeros,
                "target_z_e": zeros.clone(),
                "pred_logits": torch.zeros(batch_size, 3, 10),
                "target_indices": target_indices,
                "pred_indices": pred_indices,
                "pred_pose": pred_pose,
            }

    def criterion(**kwargs):
        zero = kwargs["pred_z_e"].new_tensor(0.0)
        return {
            "total_loss": zero,
            "latent_loss": zero,
            "token_loss": zero,
            "pose_loss": zero,
        }

    batch = {
        "mmwave_points": torch.zeros(4, 4, 5),
        "target_poses": torch.zeros(4, 2, 3),
        "indicator": torch.tensor(
            [
                [1, 10, 0],
                [1, 20, 1],
                [2, 10, 0],
                [2, 20, 1],
            ]
        ),
    }
    metrics = evaluate_hpe_token(
        model=DummyModel(),
        dataloader=[batch],
        criterion=criterion,
        device=torch.device("cpu"),
        runtime=_runtime(),
        config={
            "training": {"stage": "hpe_token"},
            "vqvae": {"root_index": 0},
            "dataset": {"name": "M4Human"},
            "evaluation": {
                "token_alignment": {"enabled": True, "shuffle_seed": 0},
            },
        },
        split_name="test",
        include_action_groups=False,
    )

    assert metrics["token_acc"] == pytest.approx(0.5)
    assert metrics["token_alignment"]["paired"]["overall"]["agreement"] == pytest.approx(
        metrics["token_acc"]
    )
    assert metrics["token_alignment"]["num_samples"] == 4
    codebook_topk = metrics["token_alignment"]["codebook_topk"]["paired"]["overall"]
    assert codebook_topk["top1"] == pytest.approx(metrics["token_acc"])
    assert codebook_topk["top3"] == pytest.approx(1.0)
    assert codebook_topk["top5"] == pytest.approx(1.0)


def test_token_alignment_evaluator_requires_indicator_metadata():
    class DummyVQVAE:
        layout = _layout()

    class DummyModel(torch.nn.Module):
        vqvae = DummyVQVAE()

        def forward(self, mmwave_points, target_poses):
            zeros = torch.zeros(2, 3, 2)
            return {
                "pred_z_e": zeros,
                "target_z_e": zeros.clone(),
                "pred_logits": torch.zeros(2, 3, 5),
                "target_indices": torch.zeros(2, 3, dtype=torch.long),
                "pred_indices": torch.zeros(2, 3, dtype=torch.long),
                "pred_pose": target_poses.clone(),
            }

    def criterion(**kwargs):
        zero = kwargs["pred_z_e"].new_tensor(0.0)
        return {
            "total_loss": zero,
            "latent_loss": zero,
            "token_loss": zero,
            "pose_loss": zero,
        }

    with pytest.raises(ValueError, match="requires batch indicator metadata"):
        evaluate_hpe_token(
            model=DummyModel(),
            dataloader=[
                {
                    "mmwave_points": torch.zeros(2, 4, 5),
                    "target_poses": torch.zeros(2, 2, 3),
                }
            ],
            criterion=criterion,
            device=torch.device("cpu"),
            runtime=_runtime(),
            config={
                "training": {"stage": "hpe_token"},
                "vqvae": {"root_index": 0},
                "dataset": {"name": "M4Human"},
                "evaluation": {
                    "token_alignment": {"enabled": True, "shuffle_seed": 0},
                },
            },
            split_name="test",
            include_action_groups=False,
        )
