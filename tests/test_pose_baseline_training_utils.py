import torch
from torch.utils.data import DataLoader, Dataset

from wave2body.models.baselines import build_pose_baseline
from wave2body.train.pose_baseline_utils import train_pose_baseline_one_epoch, validate_pose_baseline_one_epoch
from wave2body.utils.runtime_utils import build_grad_scaler, configure_runtime


class _RadarPoseDataset(Dataset):
    def __init__(self, *, length: int = 4, time: int = 4, points: int = 8, channels: int = 6, joints: int = 17):
        self.length = length
        self.time = time
        self.points = points
        self.channels = channels
        self.joints = joints

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        gen = torch.Generator().manual_seed(int(idx))
        return {
            "mmwave_points": torch.randn(self.time, self.points, self.channels, generator=gen),
            "target_poses": torch.randn(self.joints, 3, generator=gen),
            "timesteps": torch.tensor(idx % 2, dtype=torch.long),
        }


class _FakeEncoder(torch.nn.Module):
    def forward(self, radar):
        batch_size = radar.shape[0]
        return radar.new_zeros(batch_size, 17, 3), radar.new_ones(batch_size, 17, 8)


class _FakeMMMeshBodyDecoder(torch.nn.Module):
    def forward(self, rotmat, translation, betas, gender):
        joints = translation.unsqueeze(-2).expand(*translation.shape[:2], 22, 3)
        return {"joints": joints}


def _mmdiff_bundle():
    return build_pose_baseline(
        {
            "baseline": {"name": "mmdiff"},
            "mmdiff": {
                "num_joints": 17,
                "radar_input_dim": 6,
                "condition_dim": 8,
                "hidden_dim": 16,
                "history_frames": 3,
                "num_layers": 1,
                "num_heads": 2,
                "dropout": 0.0,
                "local_points": 4,
            },
            "mmdiff_loss": {"pose_weight": 1.0, "reduction": "mse"},
        },
        device="cpu",
        feature_encoder=_FakeEncoder(),
    )


def test_train_pose_baseline_one_epoch_updates_mmdiff_model():
    bundle = _mmdiff_bundle()
    loader = DataLoader(_RadarPoseDataset(length=4), batch_size=2)
    optimizer = torch.optim.AdamW(bundle.model.parameters(), lr=1.0e-3)
    runtime = configure_runtime({"precision": "fp32", "amp": False}, torch.device("cpu"), log_prefix="[test]")
    grad_scaler = build_grad_scaler(runtime)

    metrics, global_step = train_pose_baseline_one_epoch(
        bundle=bundle,
        dataloader=loader,
        optimizer=optimizer,
        grad_scaler=grad_scaler,
        device=torch.device("cpu"),
        runtime=runtime,
        epoch=0,
        grad_norm=1.0,
        grad_accum_steps=1,
        writer=None,
        log_interval=10,
        global_step_start=5,
    )

    assert global_step == 7
    assert metrics["num_samples"] == 4
    assert metrics["optimizer_steps"] == 2
    assert metrics["total_loss"] > 0
    assert metrics["mpjpe_mm"] > 0


def test_validate_pose_baseline_one_epoch_reports_mmdiff_metrics():
    bundle = _mmdiff_bundle()
    loader = DataLoader(_RadarPoseDataset(length=3), batch_size=1)
    runtime = configure_runtime({"precision": "fp32", "amp": False}, torch.device("cpu"), log_prefix="[test]")

    metrics = validate_pose_baseline_one_epoch(
        bundle=bundle,
        dataloader=loader,
        device=torch.device("cpu"),
        runtime=runtime,
        desc="unit",
    )

    assert metrics["num_samples"] == 3
    assert metrics["num_batches"] == 3
    assert metrics["total_loss"] > 0
    assert metrics["mpjpe_mm"] > 0


def test_train_pose_baseline_one_epoch_uses_pose_root_for_mmmesh():
    bundle = build_pose_baseline(
        {
            "baseline": {"name": "mmmesh"},
            "mmmesh": {"input_dim": 6, "anchor_nsample": 4},
            "mmmesh_loss": {
                "rotmat_weight": 0.0,
                "translation_weight": 0.0,
                "root_xy_weight": 1.0,
                "betas_weight": 0.0,
                "gender_weight": 0.0,
                "vertices_weight": 0.0,
                "joints_weight": 0.0,
            },
        },
        device="cpu",
    )
    loader = DataLoader(_RadarPoseDataset(length=2, time=3, points=16, channels=6, joints=24), batch_size=2)
    optimizer = torch.optim.AdamW(bundle.model.parameters(), lr=1.0e-3)
    runtime = configure_runtime({"precision": "fp32", "amp": False}, torch.device("cpu"), log_prefix="[test]")
    grad_scaler = build_grad_scaler(runtime)

    metrics, _ = train_pose_baseline_one_epoch(
        bundle=bundle,
        dataloader=loader,
        optimizer=optimizer,
        grad_scaler=grad_scaler,
        device=torch.device("cpu"),
        runtime=runtime,
    )

    assert metrics["num_samples"] == 2
    assert metrics["total_loss"] > 0


def test_validate_pose_baseline_reports_mmmesh_decoded_absolute_mpjpe():
    bundle = build_pose_baseline(
        {
            "baseline": {"name": "mmmesh"},
            "mmmesh": {
                "input_dim": 6,
                "anchor_nsample": 4,
                "body_decoder": _FakeMMMeshBodyDecoder(),
            },
            "mmmesh_loss": {
                "rotmat_weight": 0.0,
                "translation_weight": 0.0,
                "root_xy_weight": 0.0,
                "betas_weight": 0.0,
                "gender_weight": 0.0,
                "vertices_weight": 0.0,
                "joints_weight": 1.0,
            },
        },
        device="cpu",
    )
    batch = {
        "mmwave_points": torch.randn(2, 3, 16, 6),
        "target_poses": torch.zeros(2, 22, 3),
        "target_poses_absolute": torch.zeros(2, 22, 3),
    }
    loader = DataLoader([batch], batch_size=None)
    runtime = configure_runtime({"precision": "fp32", "amp": False}, torch.device("cpu"), log_prefix="[test]")

    metrics = validate_pose_baseline_one_epoch(
        bundle=bundle,
        dataloader=loader,
        device=torch.device("cpu"),
        runtime=runtime,
        desc="unit",
    )

    assert metrics["num_samples"] == 2
    assert metrics["joints_loss"] > 0
    assert metrics["mpjpe_mm"] > 0
