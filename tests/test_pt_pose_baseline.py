import torch

from wave2body.models.point_transformer.pose_baseline import PTPoseModel, PoseMPJPELoss


def test_pt_pose_model_has_no_confidence_head():
    model = PTPoseModel(
        {
            "data_format": {"radar_input_c": 4, "num_joints": 2},
            "model": {"scale": "zepto", "nneighbor": 2, "dropout": 0.0},
        },
        device="cpu",
    )
    points = torch.randn(2, 8, 4)

    outputs = model(points)

    assert not hasattr(model, "conf_head")
    assert sorted(outputs) == ["joint_feat", "pred_pose"]
    assert outputs["pred_pose"].shape == (2, 2, 3)


def test_pt_pose_model_clamps_neighbor_count_for_small_point_sets():
    model = PTPoseModel(
        {
            "data_format": {"radar_input_c": 4, "num_joints": 2},
            "model": {"scale": "zepto", "nneighbor": 16, "dropout": 0.0},
        },
        device="cpu",
    )
    model.eval()
    points = torch.randn(2, 4, 4)

    with torch.no_grad():
        outputs = model(points)

    assert outputs["pred_pose"].shape == (2, 2, 3)


def test_pose_mpjpe_loss_is_mean_joint_error():
    criterion = PoseMPJPELoss()
    pred = torch.tensor([[[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]])
    target = torch.zeros_like(pred)

    loss = criterion(pred, target)

    assert torch.isclose(loss["pose_loss"], torch.tensor(2.5))
    assert torch.isclose(loss["total_loss"], torch.tensor(2.5))
    torch.testing.assert_close(loss["per_joint_errors"], torch.tensor([[0.0, 5.0]]))
