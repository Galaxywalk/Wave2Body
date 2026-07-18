# -*- coding: utf-8 -*-
"""Fit SMPL/SMPL-X body meshes to predicted joints for evaluation exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

from wave2body.utils.paths import resolve_dataset_uri


@dataclass(frozen=True)
class JointToMeshFitResult:
    """Result of fitting a fixed-shape body model to target joints."""

    vertices: torch.Tensor | None
    fitted_joints: torch.Tensor
    full_joints: torch.Tensor
    global_orient: torch.Tensor
    body_pose: torch.Tensor
    betas: torch.Tensor
    transl: torch.Tensor
    joint_loss: torch.Tensor
    total_loss: torch.Tensor
    iterations: int
    faces: torch.Tensor | None = None


def _infer_module_device(module: nn.Module) -> torch.device:
    for tensor in list(module.parameters()) + list(module.buffers()):
        return tensor.device
    return torch.device("cpu")


def _body_pose_dim(model_type: str, body_model: nn.Module, override: int | None) -> int:
    if override is not None:
        return int(override)
    num_body_joints = getattr(body_model, "NUM_BODY_JOINTS", None)
    if num_body_joints is not None:
        return int(num_body_joints) * 3
    if str(model_type).lower() == "smpl":
        return 69
    return 63


def _as_batched_last_dim(
    value: torch.Tensor | Sequence[float] | None,
    *,
    batch_size: int,
    dim: int,
    prefix_shape: torch.Size,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if value is None:
        return torch.zeros((batch_size, dim), device=device, dtype=dtype)
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.shape == (dim,):
        return tensor.view(1, dim).expand(batch_size, dim).clone()
    if tensor.shape == (batch_size, dim):
        return tensor.clone()
    if tensor.shape == (*prefix_shape, dim):
        return tensor.reshape(batch_size, dim).clone()
    prefixed_shape = tuple((*prefix_shape, dim))
    raise ValueError(
        f"Expected tensor with shape [{dim}], [{batch_size},{dim}], or {prefixed_shape}; got {tuple(tensor.shape)}"
    )


def _freeze_or_train(tensor: torch.Tensor, train: bool) -> torch.Tensor:
    return tensor.detach().clone().requires_grad_(bool(train))


def _extract_tensor(output: object, name: str) -> torch.Tensor | None:
    value = getattr(output, name, None)
    if value is None and isinstance(output, dict):
        value = output.get(name)
    return value if torch.is_tensor(value) else None


class JointToSMPLMeshFitter:
    """Optimize SMPL/SMPL-X pose and translation so decoded joints match targets.

    The default setup keeps ``betas`` fixed at zero, which corresponds to the
    standard body shape.  This is intended for evaluation-time mesh export from
    joint-only models; it is not a unique inverse of SMPL.
    """

    def __init__(
        self,
        *,
        body_model: nn.Module | None = None,
        model_path: str | Path | None = None,
        model_type: str = "smplx",
        gender: str = "neutral",
        num_betas: int = 10,
        body_pose_dim: int | None = None,
        joint_indices: Sequence[int] | None = None,
        use_j_regressor: bool = True,
        root_index: int = 0,
        iterations: int = 100,
        lr: float = 0.05,
        pose_prior_weight: float = 1.0e-4,
        betas_prior_weight: float = 1.0e-3,
        fit_global_orient: bool = True,
        fit_body_pose: bool = True,
        fit_translation: bool = True,
        optimize_betas: bool = False,
        init_translation_from_root: bool = True,
        return_vertices: bool = True,
        tolerance: float = 0.0,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if body_model is None:
            if model_path is None:
                raise ValueError("JointToSMPLMeshFitter requires body_model or model_path")
            body_model = self._load_body_model(
                model_path=model_path,
                model_type=model_type,
                gender=gender,
                num_betas=num_betas,
            )

        self.body_model = body_model
        self.model_type = str(model_type).strip().lower()
        self.gender = str(gender)
        self.num_betas = int(num_betas)
        self.body_pose_dim = _body_pose_dim(self.model_type, body_model, body_pose_dim)
        self.joint_indices = tuple(int(i) for i in joint_indices) if joint_indices is not None else None
        self.use_j_regressor = bool(use_j_regressor)
        self.root_index = int(root_index)
        self.iterations = int(iterations)
        self.lr = float(lr)
        self.pose_prior_weight = float(pose_prior_weight)
        self.betas_prior_weight = float(betas_prior_weight)
        self.fit_global_orient = bool(fit_global_orient)
        self.fit_body_pose = bool(fit_body_pose)
        self.fit_translation = bool(fit_translation)
        self.optimize_betas = bool(optimize_betas)
        self.init_translation_from_root = bool(init_translation_from_root)
        self.return_vertices = bool(return_vertices)
        self.tolerance = float(tolerance)
        self.device = torch.device(device) if device is not None else _infer_module_device(body_model)
        self.dtype = dtype

        self.body_model.to(self.device)
        self.body_model.eval()

    @staticmethod
    def _load_body_model(
        *,
        model_path: str | Path,
        model_type: str,
        gender: str,
        num_betas: int,
    ) -> nn.Module:
        try:
            import smplx
        except ImportError as exc:  # pragma: no cover - dependency is declared in pyproject
            raise ImportError("JointToSMPLMeshFitter requires smplx when model_path is used") from exc

        path = Path(resolve_dataset_uri(str(model_path))).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Missing SMPL/SMPL-X model path: {path}")
        return smplx.create(
            model_path=str(path),
            model_type=str(model_type),
            gender=str(gender),
            num_betas=int(num_betas),
        )

    def _zero_extra_kwargs(self, batch_size: int, reference: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.model_type not in {"smplx", "smplh"}:
            return {}

        kwargs: dict[str, torch.Tensor] = {}
        if self.model_type == "smplx":
            kwargs.update(
                {
                    "jaw_pose": reference.new_zeros(batch_size, 3),
                    "leye_pose": reference.new_zeros(batch_size, 3),
                    "reye_pose": reference.new_zeros(batch_size, 3),
                    "expression": reference.new_zeros(
                        batch_size,
                        int(getattr(self.body_model, "num_expression_coeffs", 10)),
                    ),
                }
            )

        if bool(getattr(self.body_model, "use_pca", False)):
            hand_dim = int(getattr(self.body_model, "num_pca_comps", 6))
            kwargs["left_hand_pose"] = reference.new_zeros(batch_size, hand_dim)
            kwargs["right_hand_pose"] = reference.new_zeros(batch_size, hand_dim)
        else:
            kwargs["left_hand_pose"] = reference.new_zeros(batch_size, 15, 3)
            kwargs["right_hand_pose"] = reference.new_zeros(batch_size, 15, 3)
        return kwargs

    def _forward_body(
        self,
        *,
        global_orient: torch.Tensor,
        body_pose: torch.Tensor,
        betas: torch.Tensor,
        transl: torch.Tensor,
        need_vertices: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        batch_size = int(betas.shape[0])
        output = self.body_model(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            return_verts=bool(need_vertices or self.use_j_regressor),
            **self._zero_extra_kwargs(batch_size, betas),
        )
        vertices = _extract_tensor(output, "vertices")
        joints = _extract_tensor(output, "joints")

        regressor = getattr(self.body_model, "J_regressor", None)
        if self.use_j_regressor and vertices is not None and regressor is not None:
            regressor_t = torch.as_tensor(regressor, device=vertices.device, dtype=vertices.dtype)
            joints = torch.einsum("jv,bvc->bjc", regressor_t, vertices)

        if joints is None:
            raise KeyError("Body model output must expose joints, or vertices plus J_regressor")
        return vertices, joints

    def _select_fit_joints(self, joints: torch.Tensor, target_joint_count: int) -> torch.Tensor:
        if self.joint_indices is not None:
            indices = torch.as_tensor(self.joint_indices, dtype=torch.long, device=joints.device)
            selected = joints.index_select(dim=1, index=indices)
        else:
            selected = joints[:, :target_joint_count]
        if int(selected.shape[1]) != int(target_joint_count):
            raise ValueError(
                f"Selected body joints count {selected.shape[1]} does not match target joint count {target_joint_count}"
            )
        return selected

    def _initial_translation(
        self,
        *,
        target_joints: torch.Tensor,
        global_orient: torch.Tensor,
        body_pose: torch.Tensor,
        betas: torch.Tensor,
    ) -> torch.Tensor:
        transl = target_joints.new_zeros(target_joints.shape[0], 3)
        if not self.init_translation_from_root:
            return transl

        with torch.no_grad():
            _, model_joints = self._forward_body(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas,
                transl=transl,
                need_vertices=False,
            )
            fit_joints = self._select_fit_joints(model_joints, int(target_joints.shape[1]))
            root = min(max(self.root_index, 0), int(target_joints.shape[1]) - 1)
            return target_joints[:, root] - fit_joints[:, root]

    def fit(
        self,
        joints: torch.Tensor,
        *,
        joint_weights: torch.Tensor | None = None,
        betas: torch.Tensor | Sequence[float] | None = None,
    ) -> JointToMeshFitResult:
        """Fit the body model to target joints.

        Args:
            joints: Target joints with shape ``[J,3]``, ``[B,J,3]``, or any
                prefix dimensions followed by ``[J,3]``.
            joint_weights: Optional weights with shape matching ``joints[..., 0]``.
            betas: Optional fixed or initial shape coefficients. When omitted,
                zero betas are used.
        """

        target = torch.as_tensor(joints, device=self.device, dtype=self.dtype)
        if target.ndim < 2 or target.shape[-1] != 3:
            raise ValueError(f"joints must end with [J,3], got {tuple(target.shape)}")
        if target.ndim == 2:
            target = target.unsqueeze(0)
            squeeze_single = True
        else:
            squeeze_single = False

        prefix_shape = target.shape[:-2]
        target_joint_count = int(target.shape[-2])
        flat_target = target.reshape(-1, target_joint_count, 3)
        batch_size = int(flat_target.shape[0])

        if joint_weights is None:
            flat_weights = flat_target.new_ones(batch_size, target_joint_count, 1)
        else:
            weights = torch.as_tensor(joint_weights, device=self.device, dtype=self.dtype)
            if squeeze_single and weights.ndim == 1:
                weights = weights.unsqueeze(0)
            if weights.shape != (*prefix_shape, target_joint_count):
                raise ValueError(
                    f"joint_weights must have shape {tuple((*prefix_shape, target_joint_count))}, got {tuple(weights.shape)}"
                )
            flat_weights = weights.reshape(batch_size, target_joint_count, 1)

        global_orient = _freeze_or_train(
            flat_target.new_zeros(batch_size, 3),
            self.fit_global_orient,
        )
        body_pose = _freeze_or_train(
            flat_target.new_zeros(batch_size, self.body_pose_dim),
            self.fit_body_pose,
        )
        betas_t = _freeze_or_train(
            _as_batched_last_dim(
                betas,
                batch_size=batch_size,
                dim=self.num_betas,
                prefix_shape=prefix_shape,
                device=self.device,
                dtype=self.dtype,
            ),
            self.optimize_betas,
        )
        transl = _freeze_or_train(
            self._initial_translation(
                target_joints=flat_target,
                global_orient=global_orient.detach(),
                body_pose=body_pose.detach(),
                betas=betas_t.detach(),
            ),
            self.fit_translation,
        )

        params = [p for p in (global_orient, body_pose, betas_t, transl) if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self.lr) if params else None

        prev_total: float | None = None
        joint_loss = flat_target.new_tensor(0.0)
        total_loss = flat_target.new_tensor(0.0)
        ran_iterations = 0
        for iteration in range(max(self.iterations, 1)):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            _, model_joints = self._forward_body(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas_t,
                transl=transl,
                need_vertices=False,
            )
            fitted = self._select_fit_joints(model_joints, target_joint_count)
            diff = fitted - flat_target
            denom = flat_weights.sum().clamp_min(1.0)
            joint_loss = (diff.square() * flat_weights).sum() / denom
            total_loss = joint_loss
            if self.pose_prior_weight > 0.0:
                total_loss = total_loss + self.pose_prior_weight * body_pose.square().mean()
            if self.optimize_betas and self.betas_prior_weight > 0.0:
                total_loss = total_loss + self.betas_prior_weight * betas_t.square().mean()

            ran_iterations = iteration + 1
            if optimizer is not None:
                total_loss.backward()
                optimizer.step()

            total_value = float(total_loss.detach().cpu())
            if self.tolerance > 0.0 and prev_total is not None and abs(prev_total - total_value) < self.tolerance:
                break
            prev_total = total_value

        with torch.no_grad():
            vertices, model_joints = self._forward_body(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas_t,
                transl=transl,
                need_vertices=self.return_vertices,
            )
            fitted = self._select_fit_joints(model_joints, target_joint_count)

        faces = model_faces(self.body_model, device=self.device)

        def restore_prefix(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            if squeeze_single:
                return tensor.reshape(*tensor.shape[1:])
            return tensor.reshape(*prefix_shape, *tensor.shape[1:])

        return JointToMeshFitResult(
            vertices=restore_prefix(vertices.detach()) if vertices is not None else None,
            fitted_joints=restore_prefix(fitted.detach()),
            full_joints=restore_prefix(model_joints.detach()),
            global_orient=restore_prefix(global_orient.detach()),
            body_pose=restore_prefix(body_pose.detach()),
            betas=restore_prefix(betas_t.detach()),
            transl=restore_prefix(transl.detach()),
            joint_loss=joint_loss.detach(),
            total_loss=total_loss.detach(),
            iterations=ran_iterations,
            faces=faces.detach() if faces is not None else None,
        )


def model_faces(body_model: nn.Module, *, device: torch.device | str | None = None) -> torch.Tensor | None:
    """Return 0-based triangular faces from a SMPL-like model when available."""

    faces = getattr(body_model, "faces_tensor", None)
    if faces is None:
        faces = getattr(body_model, "faces", None)
    if faces is None:
        return None
    faces_t = torch.as_tensor(faces, dtype=torch.long)
    if device is not None:
        faces_t = faces_t.to(device)
    if faces_t.ndim != 2 or faces_t.shape[-1] != 3:
        raise ValueError(f"faces must have shape [F,3], got {tuple(faces_t.shape)}")
    return faces_t


def fit_joints_to_smpl_mesh(
    joints: torch.Tensor,
    *,
    body_model: nn.Module | None = None,
    model_path: str | Path | None = None,
    **kwargs: Any,
) -> JointToMeshFitResult:
    """Convenience wrapper around :class:`JointToSMPLMeshFitter`."""

    fitter = JointToSMPLMeshFitter(body_model=body_model, model_path=model_path, **kwargs)
    return fitter.fit(joints)


def save_meshes_as_obj(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    output_dir: str | Path,
    *,
    stem: str = "mesh",
) -> list[Path]:
    """Write one or more fitted meshes as OBJ files.

    ``vertices`` may be ``[V,3]`` or have any leading dimensions before
    ``[V,3]``.  Faces are expected to be 0-based ``[F,3]`` indices.
    """

    vertices_t = torch.as_tensor(vertices).detach().cpu()
    faces_t = torch.as_tensor(faces, dtype=torch.long).detach().cpu()
    if vertices_t.ndim < 2 or vertices_t.shape[-1] != 3:
        raise ValueError(f"vertices must end with [V,3], got {tuple(vertices_t.shape)}")
    if faces_t.ndim != 2 or faces_t.shape[-1] != 3:
        raise ValueError(f"faces must have shape [F,3], got {tuple(faces_t.shape)}")

    flat_vertices = vertices_t.reshape(-1, vertices_t.shape[-2], 3)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for idx, verts in enumerate(flat_vertices):
        path = out_dir / f"{stem}_{idx:06d}.obj"
        lines: list[str] = []
        for vertex in verts:
            x, y, z = (float(v) for v in vertex)
            lines.append(f"v {x:.8f} {y:.8f} {z:.8f}")
        for face in faces_t:
            i, j, k = (int(v) + 1 for v in face)
            lines.append(f"f {i} {j} {k}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


__all__ = [
    "JointToMeshFitResult",
    "JointToSMPLMeshFitter",
    "fit_joints_to_smpl_mesh",
    "model_faces",
    "save_meshes_as_obj",
]
