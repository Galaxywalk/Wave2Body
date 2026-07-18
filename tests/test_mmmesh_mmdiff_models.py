import torch
from types import SimpleNamespace

from wave2body.models.mmdiff import MMDiffDenoiser, MMDiffPoseLoss, MMDiffPoseModel, create_mmdiff_model
from wave2body.models.mmdiff.model import _swish
from wave2body.models.mmmesh import (
    MMMeshLoss,
    MMMeshModel,
    MMMeshSMPLDecoder,
    create_mmmesh_model,
    expand_mmmesh_rotmat_to_smpl_pose,
)


def test_mmmesh_model_forward_without_body_decoder():
    model = MMMeshModel(input_dim=6)
    model.eval()
    points = torch.randn(2, 3, 16, 6)

    with torch.no_grad():
        outputs = model(points)

    assert outputs["pred_rotmat"].shape == (2, 3, 9, 3, 3)
    assert outputs["pred_translation"].shape == (2, 3, 3)
    assert outputs["pred_betas"].shape == (2, 3, 10)
    assert outputs["pred_gender_logit"].shape == (2, 3, 1)
    assert outputs["pred_root_xy"].shape == (2, 3, 2)


def test_mmmesh_model_accepts_single_frame_point_batches():
    model = MMMeshModel(input_dim=5)
    model.eval()
    points = torch.randn(2, 16, 5)

    with torch.no_grad():
        outputs = model(points, gender=torch.ones(2, 1))

    assert outputs["pred_translation"].shape == (2, 1, 3)


def test_mmmesh_loss_uses_available_parameter_targets():
    model = create_mmmesh_model({"input_dim": 6})
    points = torch.randn(2, 3, 16, 6)
    outputs = model(points)
    criterion = MMMeshLoss(rotmat_weight=0.2, vertices_weight=0.0, joints_weight=0.0)

    losses = criterion(
        outputs,
        target_rotmat=torch.eye(3).view(1, 1, 1, 3, 3).expand(2, 3, 9, 3, 3),
        target_translation=torch.zeros(2, 3, 3),
        target_betas=torch.zeros(2, 3, 10),
        target_gender=torch.ones(2, 3, 1),
    )

    assert isinstance(model, MMMeshModel)
    assert sorted(losses) == [
        "betas_loss",
        "gender_loss",
        "root_xy_loss",
        "rotmat_loss",
        "total_loss",
        "translation_loss",
    ]
    assert torch.isfinite(losses["total_loss"])
    losses["total_loss"].backward()


def test_mmmesh_rotmat_expansion_matches_upstream_joint_slots():
    eye = torch.eye(3)
    rotmat = eye.view(1, 1, 1, 3, 3).repeat(1, 1, 9, 1, 1)
    for idx in range(9):
        rotmat[:, :, idx] = eye * float(idx + 1)

    world_rot, full_pose = expand_mmmesh_rotmat_to_smpl_pose(rotmat)

    assert world_rot.shape == (1, 3, 3)
    assert full_pose.shape == (1, 24, 3, 3)
    torch.testing.assert_close(world_rot[0], eye)
    torch.testing.assert_close(full_pose[0, 1], eye * 2.0)
    torch.testing.assert_close(full_pose[0, 2], eye * 3.0)
    torch.testing.assert_close(full_pose[0, 4], eye * 4.0)
    torch.testing.assert_close(full_pose[0, 5], eye * 5.0)
    torch.testing.assert_close(full_pose[0, 16], eye * 6.0)
    torch.testing.assert_close(full_pose[0, 19], eye * 9.0)
    torch.testing.assert_close(full_pose[0, 20], eye)


class _FakeSMPL(torch.nn.Module):
    def __init__(self, offset: float):
        super().__init__()
        self.offset = float(offset)

    def forward(self, betas, global_orient, body_pose, transl=None, pose2rot=False, return_verts=True):
        batch_size = betas.shape[0]
        joints = betas.new_zeros(batch_size, 4, 3) + self.offset
        vertices = betas.new_zeros(batch_size, 6, 3) + self.offset if return_verts else None
        return SimpleNamespace(joints=joints, vertices=vertices)


def test_mmmesh_smpl_decoder_uses_gender_masks_and_world_transform():
    decoder = MMMeshSMPLDecoder(
        male_model=_FakeSMPL(10.0),
        female_model=_FakeSMPL(20.0),
        joint_count=3,
    )
    rotmat = torch.eye(3).view(1, 1, 1, 3, 3).repeat(1, 2, 9, 1, 1)
    translation = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    betas = torch.zeros(1, 2, 10)
    gender = torch.tensor([[[1.0], [0.0]]])

    out = decoder(rotmat, translation, betas, gender)

    assert out["joints"].shape == (1, 2, 3, 3)
    assert out["vertices"].shape == (1, 2, 6, 3)
    torch.testing.assert_close(out["joints"][0, 0, 0], torch.tensor([11.0, 12.0, 13.0]))
    torch.testing.assert_close(out["joints"][0, 1, 0], torch.tensor([24.0, 25.0, 26.0]))
    assert out["full_pose"].shape == (1, 2, 24, 3, 3)
    assert out["world_rot"].shape == (1, 2, 3, 3)


def test_create_mmmesh_model_builds_optional_body_decoder_from_config():
    model = create_mmmesh_model(
        {
            "input_dim": 6,
            "body_decoder_config": {
                "enabled": True,
                "male_model": _FakeSMPL(1.0),
                "female_model": _FakeSMPL(2.0),
            },
        }
    )

    assert isinstance(model.body_decoder, MMMeshSMPLDecoder)


def test_mmdiff_denoiser_cpu_shapes():
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    model = MMDiffDenoiser(
        num_joints=5,
        hidden_dim=16,
        condition_dim=8,
        history_frames=3,
        radar_input_dim=4,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        local_points=4,
        edges=edges,
    )
    pose_history = torch.randn(2, 5, 9)
    condition = torch.randn(2, 5, 8)
    radar = torch.randn(2, 2, 10, 4)

    outputs = model(pose_history, timesteps=torch.tensor([0, 1]), condition=condition, radar=radar)

    assert outputs["pred_pose"].shape == (2, 5, 3)
    assert outputs["denoised_history"].shape == (2, 5, 9)
    assert outputs["limb_lengths"].shape == (2, 4)


def _reference_mmdiff_local_embedding(model: MMDiffDenoiser, pose_history: torch.Tensor, radar: torch.Tensor):
    if radar.ndim == 4:
        batch_size, time_size, point_size, channels = radar.shape
        point_cloud = radar.reshape(batch_size, time_size * point_size, channels)
    elif radar.ndim == 3:
        point_cloud = radar
    else:
        raise ValueError("radar must have shape [B, T, N, C] or [B, N, C]")

    anchors = pose_history[:, :, -3:]
    local_features = []
    k = min(model.local_points, point_cloud.shape[1])
    for joint_id in range(model.num_joints):
        anchor = anchors[:, joint_id : joint_id + 1, :]
        distances = torch.sum((point_cloud[:, :, :3] - anchor).square(), dim=-1)
        indices = torch.topk(distances, k=k, dim=-1, largest=False, sorted=True).indices
        selected = torch.gather(point_cloud, 1, indices.unsqueeze(-1).expand(-1, -1, point_cloud.shape[-1]))
        dist_selected = torch.sum((selected[:, :, :3] - anchor).square(), dim=-1)
        local_score = dist_selected.lt(model.local_radius_sq).float().mean(dim=-1, keepdim=True)

        emb = model.local_mlp[0](selected)
        emb = _swish(emb)
        emb = model.local_mlp[1](emb)
        for attn in model.local_attention:
            emb = emb + attn(emb, emb, emb)
        local_features.append(emb.mean(dim=1) * local_score)
    return torch.stack(local_features, dim=1)


def _reference_mmdiff_temporal_embedding(model: MMDiffDenoiser, pose_history: torch.Tensor):
    frame_embeds = []
    for frame_idx in range(model.history_frames):
        frame_pose = pose_history[:, :, frame_idx * 3 : (frame_idx + 1) * 3]
        frame_embeds.append(model.temporal_gconv(frame_pose, model.adj))
    stacked = torch.stack(frame_embeds, dim=1)
    batch_size = pose_history.shape[0]
    temporal = stacked.reshape(batch_size, model.history_frames, model.num_joints * model.hidden_dim).transpose(1, 2)
    temporal = model.temporal_conv(temporal).mean(dim=-1)
    return temporal.view(batch_size, model.num_joints, model.hidden_dim)


def test_mmdiff_batched_embeddings_match_reference_loops():
    torch.manual_seed(7)
    model = MMDiffDenoiser(
        num_joints=5,
        hidden_dim=16,
        condition_dim=8,
        history_frames=3,
        radar_input_dim=4,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        local_points=4,
        edges=[(0, 1), (1, 2), (2, 3), (3, 4)],
    )
    model.eval()
    pose_history = torch.randn(2, 5, 9)
    radar = torch.randn(2, 10, 4)

    with torch.no_grad():
        local_ref = _reference_mmdiff_local_embedding(model, pose_history, radar)
        temporal_ref = _reference_mmdiff_temporal_embedding(model, pose_history)
        local_out = model._local_embedding(pose_history, radar)
        temporal_out = model._temporal_embedding(pose_history)

    torch.testing.assert_close(local_out, local_ref, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(temporal_out, temporal_ref, atol=1.0e-6, rtol=1.0e-6)


class _FakeEncoder(torch.nn.Module):
    def __init__(self, num_joints: int = 5, feat_dim: int = 8) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.feat_dim = feat_dim

    def forward(self, radar):
        batch_size = radar.shape[0]
        pose = radar.new_zeros(batch_size, self.num_joints, 3)
        feat = radar.new_ones(batch_size, self.num_joints, self.feat_dim)
        return pose, feat


def test_mmdiff_pose_model_accepts_wave2body_encoder_contract():
    model = create_mmdiff_model(
        {
            "num_joints": 5,
            "radar_input_dim": 4,
            "condition_dim": 8,
            "hidden_dim": 16,
            "history_frames": 3,
            "num_layers": 1,
            "num_heads": 2,
            "dropout": 0.0,
            "local_points": 4,
            "edges": [(0, 1), (1, 2), (2, 3), (3, 4)],
        },
        feature_encoder=_FakeEncoder(),
    )
    radar = torch.randn(2, 2, 10, 4)

    outputs = model(radar, timesteps=0)

    assert isinstance(model, MMDiffPoseModel)
    assert outputs["pred_pose"].shape == (2, 5, 3)
    assert outputs["coarse_pose"].shape == (2, 5, 3)
    assert outputs["pose_history"].shape == (2, 5, 9)
    assert outputs["joint_feat"].shape == (2, 5, 8)


def test_mmdiff_pose_loss_supports_pose_history_and_limb_terms():
    outputs = {
        "pred_pose": torch.ones(2, 5, 3, requires_grad=True),
        "coarse_pose": torch.zeros(2, 5, 3, requires_grad=True),
        "denoised_history": torch.ones(2, 5, 9, requires_grad=True),
        "limb_lengths": torch.ones(2, 4, requires_grad=True),
    }
    criterion = MMDiffPoseLoss(
        pose_weight=1.0,
        coarse_weight=0.5,
        history_weight=0.25,
        limb_weight=0.1,
        reduction="mse",
    )

    losses = criterion(
        outputs,
        target_pose=torch.zeros(2, 5, 3),
        target_history=torch.zeros(2, 5, 9),
        target_limb_lengths=torch.zeros(2, 4),
    )

    assert sorted(losses) == [
        "coarse_pose_loss",
        "history_loss",
        "limb_loss",
        "mean_joint_error_m",
        "pose_loss",
        "total_loss",
    ]
    assert torch.isclose(losses["pose_loss"], torch.tensor(1.0))
    assert torch.isclose(losses["total_loss"], torch.tensor(1.35))
    losses["total_loss"].backward()
