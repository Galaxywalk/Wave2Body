"""Regression coverage for local/global point fusion (GitHub issue #3)."""

import pytest
import torch

from wave2body.models.mmwave_tokenizer.modules import PointPatchEncoder


def test_fusion_distinguishes_patches_with_identical_channel_maxima():
    encoder = PointPatchEncoder(2, 2, hidden_dim=2).double().eval()
    with torch.no_grad():
        # Independent point channels make the first pooled descriptors equal.
        for layer in (encoder.point_mlp[0], encoder.point_mlp[3]):
            layer.weight.copy_(torch.eye(2))
            layer.bias.zero_()
        encoder.global_mlp[0].weight.copy_(torch.tensor([
            [1.0, -1.0, 0.25, 0.25],
            [-1.0, 1.0, 0.25, 0.25],
        ]))
        encoder.global_mlp[0].bias.zero_()
        encoder.global_mlp[3].weight.copy_(torch.eye(2))
        encoder.global_mlp[3].bias.zero_()

    patches = torch.tensor([
        [[[1.0, 0.0], [0.0, 1.0]]],
        [[[1.0, 1.0], [0.0, 0.0]]],
    ], dtype=torch.float64, requires_grad=True)
    features = encoder.point_mlp(patches)
    torch.testing.assert_close(features[0].amax(dim=1), features[1].amax(dim=1))

    tokens = encoder(patches)
    # Fusion sees whether the two responses belong to the same point. Pooling
    # before fusion would give identical (zero) tokens for these two patches.
    assert torch.all(tokens[0] > 0.99)
    torch.testing.assert_close(tokens[1], torch.zeros_like(tokens[1]))

    tokens[0, 0, 0].backward()
    # A coordinate below its channel maximum must still affect point fusion.
    assert patches.grad[0, 0, 0, 1].abs() > 1e-7
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())


@pytest.mark.parametrize("group_size", [1, 8, 32])
def test_patch_tokens_are_invariant_to_point_permutations(group_size):
    torch.manual_seed(3)
    encoder = PointPatchEncoder(4, 16, hidden_dim=16, dropout=0.1).eval()
    patches = torch.randn(2, 3, group_size, 4)
    tokens = encoder(patches)
    assert tokens.shape == (2, 3, 16)
    assert torch.isfinite(tokens).all()
    torch.testing.assert_close(tokens, encoder(patches[:, :, torch.randperm(group_size)]))


def test_point_fusion_backpropagates_with_training_dropout():
    torch.manual_seed(4)
    encoder = PointPatchEncoder(4, 16, hidden_dim=16, dropout=0.1).train()
    patches = torch.randn(2, 3, 8, 4, requires_grad=True)
    tokens = encoder(patches)
    (tokens * torch.randn_like(tokens)).sum().backward()
    assert torch.isfinite(tokens).all()
    assert torch.isfinite(patches.grad).all()
    assert patches.grad.abs().sum() > 0
    for parameter in encoder.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
