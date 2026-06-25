"""SE(3) / rotation-6D pose utilities, faithful to UMI's official ``pose_util.py``.

UMI (Universal Manipulation Interface) represents end-effector poses as a 4x4
homogeneous transform internally, stores poses on disk as ``[x, y, z] + 6D
rotation`` (Zhou et al. 2019, "On the Continuity of Rotation Representations"),
and defines a *relative trajectory* by expressing every pose in a chunk in the
frame of the current (base) observation pose:

    T_rel = inv(T_base) @ T_pose

This module implements exactly that, with numpy only (no torch / scipy needed at
import time, though scipy is used for robust quaternion<->matrix conversion).

Conventions
-----------
* Position is ``[x, y, z]``.
* Quaternions are ``[w, x, y, z]`` (the order used in the raw earphone dataset).
* Rotation 6D is the first two columns of the 3x3 rotation matrix, flattened
  column-major into 6 numbers (UMI's ``mat_to_rot6d`` convention).
* Poses without rotation-6d are ``pose9 = [x, y, z, r00, r10, r20, r01, r11, r21]``.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


# --------------------------------------------------------------------------- #
# rotation 6D <-> rotation matrix  (Zhou et al. 2019, as used by UMI)
# --------------------------------------------------------------------------- #
def mat_to_rot6d(mat: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> 6D rotation (first two columns, flattened).

    Args:
        mat: (..., 3, 3) rotation matrices.
    Returns:
        (..., 6) rotation-6d, ordered [col0 (3), col1 (3)].
    """
    mat = np.asarray(mat)
    # Take the first two columns of the rotation matrix.
    a1 = mat[..., :, 0]
    a2 = mat[..., :, 1]
    return np.concatenate([a1, a2], axis=-1)


def rot6d_to_mat(rot6d: np.ndarray) -> np.ndarray:
    """6D rotation -> 3x3 rotation matrix via Gram-Schmidt (Zhou et al. 2019).

    Args:
        rot6d: (..., 6) rotation-6d.
    Returns:
        (..., 3, 3) orthonormal rotation matrices.
    """
    rot6d = np.asarray(rot6d)
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    # Gram-Schmidt.
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2_proj = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2_proj / np.linalg.norm(a2_proj, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    # Stack as columns -> (..., 3, 3).
    return np.stack([b1, b2, b3], axis=-1)


# --------------------------------------------------------------------------- #
# quaternion (wxyz) <-> rotation matrix
# --------------------------------------------------------------------------- #
def quat_wxyz_to_mat(quat_wxyz: np.ndarray) -> np.ndarray:
    """Quaternion [w, x, y, z] -> 3x3 rotation matrix."""
    quat_wxyz = np.asarray(quat_wxyz)
    # scipy expects [x, y, z, w].
    quat_xyzw = np.concatenate([quat_wxyz[..., 1:4], quat_wxyz[..., 0:1]], axis=-1)
    # Normalize defensively (raw data is unit-norm, but guard against drift).
    quat_xyzw = quat_xyzw / np.linalg.norm(quat_xyzw, axis=-1, keepdims=True)
    return Rotation.from_quat(quat_xyzw).as_matrix()


def mat_to_quat_wxyz(mat: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> quaternion [w, x, y, z]."""
    quat_xyzw = Rotation.from_matrix(np.asarray(mat)).as_quat()
    return np.concatenate([quat_xyzw[..., 3:4], quat_xyzw[..., 0:3]], axis=-1)


# --------------------------------------------------------------------------- #
# pose (pos + quat / pos + rot6d) <-> 4x4 homogeneous matrix
# --------------------------------------------------------------------------- #
def posequat_to_mat(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """[x,y,z] + quat[w,x,y,z] -> 4x4 homogeneous transform."""
    pos = np.asarray(pos)
    rot = quat_wxyz_to_mat(quat_wxyz)
    out = np.zeros(rot.shape[:-2] + (4, 4), dtype=np.float64)
    out[..., :3, :3] = rot
    out[..., :3, 3] = pos
    out[..., 3, 3] = 1.0
    return out


def pose9_to_mat(pose9: np.ndarray) -> np.ndarray:
    """[x,y,z, rot6d(6)] -> 4x4 homogeneous transform."""
    pose9 = np.asarray(pose9)
    pos = pose9[..., 0:3]
    rot = rot6d_to_mat(pose9[..., 3:9])
    out = np.zeros(pose9.shape[:-1] + (4, 4), dtype=np.float64)
    out[..., :3, :3] = rot
    out[..., :3, 3] = pos
    out[..., 3, 3] = 1.0
    return out


def mat_to_pose9(mat: np.ndarray) -> np.ndarray:
    """4x4 homogeneous transform -> [x,y,z, rot6d(6)]."""
    mat = np.asarray(mat)
    pos = mat[..., :3, 3]
    rot6d = mat_to_rot6d(mat[..., :3, :3])
    return np.concatenate([pos, rot6d], axis=-1)


def mat_inv(mat: np.ndarray) -> np.ndarray:
    """Inverse of a 4x4 rigid transform (uses R^T, -R^T t -- no general inverse)."""
    mat = np.asarray(mat)
    r = mat[..., :3, :3]
    t = mat[..., :3, 3]
    r_inv = np.swapaxes(r, -1, -2)
    t_inv = -np.einsum("...ij,...j->...i", r_inv, t)
    out = np.zeros_like(mat)
    out[..., :3, :3] = r_inv
    out[..., :3, 3] = t_inv
    out[..., 3, 3] = 1.0
    return out


# --------------------------------------------------------------------------- #
# UMI relative trajectory
# --------------------------------------------------------------------------- #
def convert_pose_mat_rep_relative(
    pose_mat: np.ndarray, base_pose_mat: np.ndarray
) -> np.ndarray:
    """Express a sequence of poses relative to a single base pose (UMI).

    T_rel[i] = inv(base) @ pose_mat[i]

    Args:
        pose_mat: (T, 4, 4) absolute poses.
        base_pose_mat: (4, 4) base/observation pose.
    Returns:
        (T, 4, 4) relative poses (base-frame).
    """
    return mat_inv(base_pose_mat)[None] @ pose_mat


def convert_pose_mat_rep_absolute(
    rel_pose_mat: np.ndarray, base_pose_mat: np.ndarray
) -> np.ndarray:
    """Inverse of :func:`convert_pose_mat_rep_relative`.

    T_abs[i] = base @ rel_pose_mat[i]
    """
    return base_pose_mat[None] @ rel_pose_mat
