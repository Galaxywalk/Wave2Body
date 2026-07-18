# -*- coding: utf-8 -*-

import numpy as np
import torch

from wave2body.evaluate.token_accuracy import PartTokenTopKAccumulator, parse_topk_list
from wave2body.models.human_body_tokenizer.part_schema import create_part_layout


def _layout():
    return create_part_layout(
        num_joints=4,
        part_names=("A", "B"),
        part_joints={"A": (0, 1), "B": (2, 3)},
        part_token_num={"A": 2, "B": 1},
        part_codebook_size={"A": 3, "B": 2},
    )


def test_parse_topk_list_sorts_and_deduplicates_values():
    assert parse_topk_list("5,1,3,3") == (1, 3, 5)


def test_part_token_topk_uses_part_local_logits_and_reports_parts():
    layout = _layout()
    logits = torch.tensor(
        [
            [
                [0.0, 5.0, 1.0, -9.0, -9.0],  # A slot 0: target local 1 is rank 1.
                [4.0, 3.0, 2.0, -9.0, -9.0],  # A slot 1: target local 2 is rank 3.
                [-9.0, -9.0, -9.0, 3.0, 1.0],  # B slot 0: target local 1 is rank 2.
            ]
        ]
    )
    target = torch.tensor([[1, 2, 4]])
    pred_quantized = torch.tensor([[1, 0, 4]])

    acc = PartTokenTopKAccumulator(layout=layout, topk=(1, 2, 3))
    acc.update_logits(logits, target)
    acc.update_quantized(pred_quantized, target)
    out = acc.summary()

    np.testing.assert_allclose(out["logit"]["overall"]["top1"], 1.0 / 3.0)
    np.testing.assert_allclose(out["logit"]["overall"]["top2"], 2.0 / 3.0)
    np.testing.assert_allclose(out["logit"]["overall"]["top3"], 1.0)
    np.testing.assert_allclose(out["logit"]["by_part"]["A"]["top1"], 0.5)
    np.testing.assert_allclose(out["logit"]["by_part"]["A"]["top2"], 0.5)
    np.testing.assert_allclose(out["logit"]["by_part"]["A"]["top3"], 1.0)
    np.testing.assert_allclose(out["logit"]["by_part"]["B"]["top1"], 0.0)
    np.testing.assert_allclose(out["logit"]["by_part"]["B"]["top2"], 1.0)
    np.testing.assert_allclose(out["logit"]["by_part"]["B"]["top3"], 1.0)
    assert out["logit"]["by_slot"]["A"][0]["count"] == 1
    np.testing.assert_allclose(out["logit"]["by_slot"]["A"][1]["top3"], 1.0)

    np.testing.assert_allclose(out["quantized"]["overall"]["top1"], 2.0 / 3.0)
    np.testing.assert_allclose(out["quantized"]["by_part"]["A"]["top1"], 0.5)
    np.testing.assert_allclose(out["quantized"]["by_part"]["B"]["top1"], 1.0)
