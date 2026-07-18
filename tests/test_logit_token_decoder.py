# -*- coding: utf-8 -*-

import torch
from torch import nn

from wave2body.models.hpe_token.decoder import decode_part_token_logits, part_codebook_distance_logits
from wave2body.models.human_body_tokenizer.part_schema import create_part_layout


class _Quantizer(nn.Module):
    def __init__(self, codebook):
        super().__init__()
        self.register_buffer("codebook", torch.as_tensor(codebook, dtype=torch.float32))


class _FakeVQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.layout = create_part_layout(
            num_joints=3,
            part_names=("A", "B"),
            part_joints={"A": (0, 1), "B": (2,)},
            part_token_num={"A": 2, "B": 1},
            part_codebook_size={"A": 3, "B": 2},
        )
        self.quantizers = nn.ModuleDict(
            {
                "A": _Quantizer([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
                "B": _Quantizer([[0.0, 4.0], [0.0, 5.0]]),
            }
        )

    def decode(self, z_q):
        return z_q


def test_logits_decode_uses_only_part_local_code_ranges():
    vqvae = _FakeVQVAE()
    logits = torch.tensor(
        [
            [
                [0.0, 2.0, 0.0, 99.0, 99.0],
                [0.0, 0.0, 3.0, 99.0, 99.0],
                [99.0, 99.0, 99.0, 0.0, 4.0],
            ]
        ]
    )

    out = decode_part_token_logits(vqvae=vqvae, pred_logits=logits, mode="argmax", temperature=1.0)

    torch.testing.assert_close(out.indices, torch.tensor([[1, 2, 4]]))
    torch.testing.assert_close(out.z_q, torch.tensor([[[2.0, 0.0], [3.0, 0.0], [0.0, 5.0]]]))
    assert float(out.probs[:, :2, 3:].sum().item()) == 0.0
    assert float(out.probs[:, 2:, :3].sum().item()) == 0.0


def test_argmax_st_forward_is_hard_but_backward_reaches_logits():
    vqvae = _FakeVQVAE()
    logits = torch.tensor(
        [
            [
                [0.0, 3.0, 1.0, -5.0, -5.0],
                [2.0, 1.0, 0.0, -5.0, -5.0],
                [-5.0, -5.0, -5.0, 1.0, 2.0],
            ]
        ],
        requires_grad=True,
    )

    out = decode_part_token_logits(vqvae=vqvae, pred_logits=logits, mode="argmax_st", temperature=1.0)
    torch.testing.assert_close(out.z_q.detach(), torch.tensor([[[2.0, 0.0], [1.0, 0.0], [0.0, 5.0]]]))
    torch.testing.assert_close(out.probs.sum(dim=-1), torch.ones(1, 3))
    hard_probs = torch.nn.functional.one_hot(out.indices[:, :2], 5)[:, :, :3].float()
    assert not torch.equal(out.probs[:, :2, :3], hard_probs)

    loss = (out.pose[..., 0] ** 2).sum() + out.pose[..., 1].sum()
    loss.backward()

    assert logits.grad is not None
    assert float(logits.grad.abs().sum().item()) > 0.0
    assert float(logits.grad[:, :2, 3:].abs().sum().item()) == 0.0
    assert float(logits.grad[:, 2:, :3].abs().sum().item()) == 0.0


def test_codebook_distance_logits_match_nearest_part_codes():
    vqvae = _FakeVQVAE()
    pred_z_e = torch.tensor(
        [
            [
                [1.8, 0.0],
                [2.7, 0.0],
                [0.0, 4.6],
            ]
        ],
        requires_grad=True,
    )

    logits = part_codebook_distance_logits(vqvae=vqvae, pred_z_e=pred_z_e, scale=1.0)
    out = decode_part_token_logits(vqvae=vqvae, pred_logits=logits, mode="argmax", temperature=1.0)

    torch.testing.assert_close(out.indices, torch.tensor([[1, 2, 4]]))
    assert torch.isneginf(logits[:, :2, 3:]).all()
    assert torch.isneginf(logits[:, 2:, :3]).all()

    loss = -logits[:, :, :].masked_fill(torch.isneginf(logits), 0.0).sum()
    loss.backward()
    assert pred_z_e.grad is not None
    assert float(pred_z_e.grad.abs().sum().item()) > 0.0
