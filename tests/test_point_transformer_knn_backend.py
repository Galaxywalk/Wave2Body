# -*- coding: utf-8 -*-

import sys
import types

import pytest
import torch

from wave2body.models.point_transformer import transformer_helper as th


def _reset_torch_cluster_loader(monkeypatch):
    monkeypatch.setattr(th, "_TORCH_CLUSTER_KNN", None)
    monkeypatch.setattr(th, "_TORCH_CLUSTER_IMPORT_FAILED", False)
    monkeypatch.setattr(th, "_TORCH_CLUSTER_WARNED", False)
    monkeypatch.setattr(th, "_PYKEOPS_LAZY_TENSOR", None)
    monkeypatch.setattr(th, "_PYKEOPS_IMPORT_FAILED", False)
    monkeypatch.setattr(th, "_PYKEOPS_WARNED", False)


def test_point_transformer_knn_defaults_to_torch_cluster(monkeypatch):
    _reset_torch_cluster_loader(monkeypatch)
    monkeypatch.delenv("WAVE2BODY_PT_KNN_BACKEND", raising=False)

    calls = []
    fake = types.ModuleType("torch_cluster")

    def fake_knn(x, y, k, batch_x=None, batch_y=None):
        calls.append((x.shape, y.shape, k, batch_x is not None, batch_y is not None))
        rows = torch.arange(y.shape[0], device=y.device).repeat_interleave(k)
        cols = []
        for query_idx in range(y.shape[0]):
            same_batch = torch.nonzero(batch_x == batch_y[query_idx], as_tuple=False).flatten()
            dists = (x[same_batch] - y[query_idx]).square().sum(dim=-1)
            nearest = same_batch[dists.topk(k=k, largest=False, sorted=False).indices]
            cols.append(nearest)
        return torch.stack([rows, torch.stack(cols).reshape(-1)])

    fake.knn = fake_knn
    monkeypatch.setitem(sys.modules, "torch_cluster", fake)

    xyz = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [13.0, 0.0, 0.0]],
        ]
    )

    idx = th.knn_indices(xyz, 2)
    ref = th.knn_indices(xyz, 2, backend="torch")

    assert calls == [((6, 3), (6, 3), 2, True, True)]
    torch.testing.assert_close(idx.sort(dim=-1).values, ref.sort(dim=-1).values)


def test_point_transformer_knn_falls_back_when_torch_cluster_is_unavailable(monkeypatch):
    _reset_torch_cluster_loader(monkeypatch)
    monkeypatch.setenv("WAVE2BODY_PT_KNN_BACKEND", "torch_cluster")
    monkeypatch.setattr(th, "_TORCH_CLUSTER_IMPORT_FAILED", True)

    xyz = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])

    with pytest.warns(RuntimeWarning, match="falling back"):
        idx = th.knn_indices(xyz, 2)

    ref = th.knn_indices(xyz, 2, backend="torch")
    torch.testing.assert_close(idx.sort(dim=-1).values, ref.sort(dim=-1).values)


def test_point_transformer_knn_cdist_backend_matches_torch_backend():
    xyz = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [13.0, 0.0, 0.0]],
        ]
    )

    idx = th.knn_indices(xyz, 2, backend="cdist")
    ref = th.knn_indices(xyz, 2, backend="torch")

    torch.testing.assert_close(idx.sort(dim=-1).values, ref.sort(dim=-1).values)


def test_point_transformer_knn_keops_falls_back_when_unavailable(monkeypatch):
    _reset_torch_cluster_loader(monkeypatch)
    monkeypatch.setenv("WAVE2BODY_PT_KNN_BACKEND", "keops")
    monkeypatch.setattr(th, "_PYKEOPS_IMPORT_FAILED", True)

    xyz = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])

    with pytest.warns(RuntimeWarning, match="falling back"):
        idx = th.knn_indices(xyz, 2)

    ref = th.knn_indices(xyz, 2, backend="torch")
    torch.testing.assert_close(idx.sort(dim=-1).values, ref.sort(dim=-1).values)
