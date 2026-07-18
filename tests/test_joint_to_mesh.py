from __future__ import annotations

from types import SimpleNamespace

import torch

from wave2body.evaluate import JointToSMPLMeshFitter, fit_joints_to_smpl_mesh, save_meshes_as_obj


class _ToyBodyModel(torch.nn.Module):
    NUM_BODY_JOINTS = 1

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "base_joints",
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "base_vertices",
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
        )
        self.faces = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.long)

    def forward(
        self,
        *,
        betas,
        global_orient,
        body_pose,
        transl=None,
        return_verts=True,
        **_: object,
    ):
        batch_size = int(betas.shape[0])
        joints = self.base_joints.to(device=betas.device, dtype=betas.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        vertices = self.base_vertices.to(device=betas.device, dtype=betas.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        if transl is not None:
            joints = joints + transl.unsqueeze(1)
            vertices = vertices + transl.unsqueeze(1)
        return SimpleNamespace(joints=joints, vertices=vertices if return_verts else None)


def test_joint_to_mesh_fitter_uses_standard_shape_and_fits_translation():
    model = _ToyBodyModel()
    offset = torch.tensor([1.0, 2.0, 3.0])
    target = model.base_joints + offset

    result = fit_joints_to_smpl_mesh(
        target,
        body_model=model,
        model_type="smpl",
        body_pose_dim=3,
        fit_global_orient=False,
        fit_body_pose=False,
        iterations=5,
        lr=0.1,
    )

    torch.testing.assert_close(result.betas, torch.zeros(10))
    torch.testing.assert_close(result.transl, offset)
    torch.testing.assert_close(result.fitted_joints, target)
    torch.testing.assert_close(result.vertices, model.base_vertices + offset)
    torch.testing.assert_close(result.faces, model.faces)


def test_joint_to_mesh_fitter_handles_prefixed_batches_and_joint_indices():
    model = _ToyBodyModel()
    offsets = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    target = model.base_joints[1:].view(1, 1, 2, 3) + offsets.view(1, 2, 1, 3)
    fitter = JointToSMPLMeshFitter(
        body_model=model,
        model_type="smpl",
        body_pose_dim=3,
        joint_indices=[1, 2],
        fit_global_orient=False,
        fit_body_pose=False,
        iterations=5,
    )

    result = fitter.fit(target)

    assert result.fitted_joints.shape == (1, 2, 2, 3)
    torch.testing.assert_close(result.fitted_joints, target)
    torch.testing.assert_close(result.transl, offsets)


def test_save_meshes_as_obj_writes_vertices_and_faces(tmp_path):
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    faces = torch.tensor([[0, 1, 2]])

    paths = save_meshes_as_obj(vertices, faces, tmp_path, stem="body")

    assert len(paths) == 1
    assert paths[0].name == "body_000000.obj"
    assert paths[0].read_text(encoding="utf-8").splitlines() == [
        "v 0.00000000 0.00000000 0.00000000",
        "v 1.00000000 0.00000000 0.00000000",
        "v 0.00000000 1.00000000 0.00000000",
        "f 1 2 3",
    ]
