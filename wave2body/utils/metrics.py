# -*- coding: utf-8 -*-
"""
Evaluation metrics for pose estimation.
"""

import torch
import numpy as np


def compute_mpjpe(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean Per-Joint Position Error (Protocol #1).

    Args:
        predicted: Predicted joint positions [B, J, 3]
        target: Ground truth joint positions [B, J, 3]

    Returns:
        Mean error across all joints and samples
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=-1))


def compute_p_mpjpe(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Procrustes-aligned Mean Per-Joint Position Error (Protocol #2).

    Args:
        predicted: Predicted joint positions [B, J, 3]
        target: Ground truth joint positions [B, J, 3]

    Returns:
        Mean error after Procrustes alignment
    """
    assert predicted.shape == target.shape

    # Convert to numpy
    if torch.is_tensor(predicted):
        predicted = predicted.float().cpu().detach().numpy()
    if torch.is_tensor(target):
        target = target.float().cpu().detach().numpy()

    # Compute mean positions
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    # Center the data
    X0 = target - muX
    Y0 = predicted - muY

    # Normalize
    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    # Compute optimal rotation using SVD
    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections)
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Compute scale and translation
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Apply rigid transformation to predicted poses
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE after alignment
    return torch.tensor(np.mean(np.linalg.norm(predicted_aligned - target, axis=-1)))


def compute_per_sample_p_mpjpe(predicted: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """
    Per-sample Procrustes-aligned MPJPE.

    Args:
        predicted: Predicted joint positions [B, J, 3]
        target: Ground truth joint positions [B, J, 3]

    Returns:
        NumPy array [B] with per-sample P-MPJPE in the input coordinate unit.
    """
    assert predicted.shape == target.shape

    if torch.is_tensor(predicted):
        predicted = predicted.float().cpu().detach().numpy()
    if torch.is_tensor(target):
        target = target.float().cpu().detach().numpy()

    mu_x = np.mean(target, axis=1, keepdims=True)
    mu_y = np.mean(predicted, axis=1, keepdims=True)
    x0 = target - mu_x
    y0 = predicted - mu_y

    eps = 1e-8
    norm_x = np.sqrt(np.sum(x0 ** 2, axis=(1, 2), keepdims=True))
    norm_y = np.sqrt(np.sum(y0 ** 2, axis=(1, 2), keepdims=True))
    x0 = x0 / np.maximum(norm_x, eps)
    y0 = y0 / np.maximum(norm_y, eps)

    h = np.matmul(x0.transpose(0, 2, 1), y0)
    u, s, vt = np.linalg.svd(h)
    v = vt.transpose(0, 2, 1)
    r = np.matmul(v, u.transpose(0, 2, 1))

    sign_det_r = np.sign(np.expand_dims(np.linalg.det(r), axis=1))
    v[:, :, -1] *= sign_det_r
    s[:, -1] *= sign_det_r.flatten()
    r = np.matmul(v, u.transpose(0, 2, 1))

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    scale = tr * norm_x / np.maximum(norm_y, eps)
    translation = mu_x - scale * np.matmul(mu_y, r)
    predicted_aligned = scale * np.matmul(predicted, r) + translation
    return np.linalg.norm(predicted_aligned - target, axis=-1).mean(axis=1)


def compute_pck(predicted: torch.Tensor, target: torch.Tensor, threshold: float = 0.15) -> torch.Tensor:
    """
    Percentage of Correct Keypoints.

    Args:
        predicted: Predicted joint positions [B, J, 3]
        target: Ground truth joint positions [B, J, 3]
        threshold: Distance threshold in meters

    Returns:
        Percentage of joints within threshold
    """
    assert predicted.shape == target.shape
    distances = torch.norm(predicted - target, dim=-1)  # [B, J]
    correct = (distances < threshold).float()
    return correct.mean()


def compute_acceleration_error(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute acceleration error (for temporal sequences).

    Args:
        predicted: Predicted joint positions [B, T, J, 3]
        target: Ground truth joint positions [B, T, J, 3]

    Returns:
        Mean acceleration error
    """
    # Compute accelerations
    pred_acc = predicted[:, 2:] - 2 * predicted[:, 1:-1] + predicted[:, :-2]
    target_acc = target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]

    # Compute error
    return torch.mean(torch.norm(pred_acc - target_acc, dim=-1))


# Convenient aliases
mpjpe = compute_mpjpe
pa_mpjpe = compute_p_mpjpe
pck = compute_pck
