"""Data transforms for a dual-arm dataset with UMI-style 6D-rotation EE poses.

This is the 6D-rotation sibling of ``umi_dual_arm_policy.py`` (which keeps the
rotation as a quaternion). The 6D representation (Zhou et al. 2019, the first two
columns of the rotation matrix, reconstructed via Gram-Schmidt) is **continuous
everywhere on SO(3)** -- it has no double cover and no w=0 / 180deg sign-flip
discontinuity -- so it is the principled choice when the orientation data is
spread out or approaches half-turns. It is also free at the model: pi0 zero-pads
the action vector to ``action_dim=32`` regardless, so 20 dims cost no more than 16.

The on-disk LeRobot dataset is consumed *as-is* (no offline conversion); per frame:
    observation.state : (23,) = left[pos3, quat_wxyz4, grip1] + right[pos3, quat_wxyz4, grip1] + ego7  (ABSOLUTE)
    action            : (23,) = same layout, absolute target EE pose
    observation.images.wrist_image_1, observation.images.wrist_image_2 : uint8 frames
    observation.images.image : head cam (loaded by LeRobot but dropped here)

``UmiDualArmInputs`` slices the 23-dim vector to a 20-dim two-arm pose -- per arm
[pos3, rot6d6, grip1] -- by dropping the last 7 ego dims and converting each arm's
quaternion (wxyz) to rot6d. It then maps everything into pi0's expected dict and
-- crucially -- converts the absolute action *chunk* into a UMI relative trajectory:
every pose in the chunk is expressed in the frame of the current observation pose,

    T_rel = inv(T_obs) @ T_action

per arm, composed in SE(3) on the rotation matrix. The gripper width is left
absolute. This is the load-time equivalent of pi0's linear ``DeltaActions`` but
composed in SE(3) -- so pi0's built-in delta transform MUST stay OFF for this dataset.

``UmiDualArmOutputs`` inverts the relativization at inference: given the current
(20-dim, post-input-transform) state and the model's relative action chunk, it
returns absolute 20-dim poses. The model's predicted 6D vector need not be
orthonormal -- Gram-Schmidt re-orthonormalizes it when lifting back to a matrix.
The deployed runtime is responsible for any final 20-dim -> robot-native conversion.
"""

import dataclasses

import einops
import numpy as np
from scipy.spatial.transform import Rotation

from openpi import transforms
from openpi.models import model as _model

# Per-arm layout within the 10-dim block.
_POS = slice(0, 3)
_ROT6D = slice(3, 9)
_GRIP = 9
ARM_DIM = 10
N_ARMS = 2
STATE_DIM = ARM_DIM * N_ARMS  # 20

# Raw 23-dim layout in the LeRobot dataset (per meta/info.json of the
# earphone-style dataset). Indices below index into the raw vector.
_RAW_LEFT_POS = slice(0, 3)
_RAW_LEFT_QUAT_WXYZ = slice(3, 7)
_RAW_LEFT_GRIP = 7
_RAW_RIGHT_POS = slice(8, 11)
_RAW_RIGHT_QUAT_WXYZ = slice(11, 15)
_RAW_RIGHT_GRIP = 15
# dims 16:23 are ego-pose, intentionally dropped.


# --------------------------------------------------------------------------- #
# rotation-6D / SE(3) helpers (numpy + scipy).
# --------------------------------------------------------------------------- #
def _rot6d_to_mat(rot6d: np.ndarray) -> np.ndarray:
    """[..., 6] (two stacked columns) -> 3x3 rotation matrix via Gram-Schmidt.

    Re-orthonormalizes defensively, so a non-orthonormal predicted 6D vector still
    decodes to a valid rotation.
    """
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def _mat_to_rot6d(mat: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> [..., 6] (its first two columns)."""
    return np.concatenate([mat[..., :, 0], mat[..., :, 1]], axis=-1)


def _quat_wxyz_to_mat(quat_wxyz: np.ndarray) -> np.ndarray:
    """[..., w, x, y, z] -> 3x3 rotation matrix (defensively re-normalized)."""
    quat_xyzw = np.concatenate([quat_wxyz[..., 1:4], quat_wxyz[..., 0:1]], axis=-1)
    quat_xyzw = quat_xyzw / np.linalg.norm(quat_xyzw, axis=-1, keepdims=True)
    return Rotation.from_quat(quat_xyzw).as_matrix()


def _pose10_to_mat(pose10: np.ndarray) -> np.ndarray:
    """(...,10)=[pos3,rot6d6,grip1] -> (4x4 mat). Grip column is ignored."""
    out = np.zeros(pose10.shape[:-1] + (4, 4), dtype=np.float64)
    out[..., :3, :3] = _rot6d_to_mat(pose10[..., _ROT6D])
    out[..., :3, 3] = pose10[..., _POS]
    out[..., 3, 3] = 1.0
    return out


def _mat_to_pose9(mat: np.ndarray) -> np.ndarray:
    """(...,4,4) -> (...,9) = [pos3, rot6d6]."""
    return np.concatenate([mat[..., :3, 3], _mat_to_rot6d(mat[..., :3, :3])], axis=-1)


def _mat_inv(mat: np.ndarray) -> np.ndarray:
    r = mat[..., :3, :3]
    t = mat[..., :3, 3]
    r_inv = np.swapaxes(r, -1, -2)
    out = np.zeros_like(mat)
    out[..., :3, :3] = r_inv
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", r_inv, t)
    out[..., 3, 3] = 1.0
    return out


def _relativize_actions(state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Express each action pose relative to the current state pose, per arm (UMI).

    state: (20,) current absolute EE pose. actions: (T,20) absolute targets.
    Returns (T,20): per arm [rel_pos3, rel_rot6d6, grip1] (gripper stays absolute).
    """
    out = np.empty_like(actions)
    for a in range(N_ARMS):
        sl = slice(a * ARM_DIM, (a + 1) * ARM_DIM)
        block = actions[:, sl]
        base = _pose10_to_mat(state[sl])  # (4,4)
        act = _pose10_to_mat(block)  # (T,4,4)
        rel = _mat_inv(base)[None] @ act  # (T,4,4)
        out[:, sl] = np.concatenate([_mat_to_pose9(rel), block[:, _GRIP, None]], axis=-1)
    return out


def _absolutize_actions(state: np.ndarray, rel_actions: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_relativize_actions` (inference path)."""
    out = np.empty_like(rel_actions)
    for a in range(N_ARMS):
        sl = slice(a * ARM_DIM, (a + 1) * ARM_DIM)
        block = rel_actions[:, sl]
        base = _pose10_to_mat(state[sl])  # (4,4)
        rel = _pose10_to_mat(block)  # (T,4,4); grip column ignored by _pose10_to_mat
        absm = base[None] @ rel
        out[:, sl] = np.concatenate([_mat_to_pose9(absm), block[:, _GRIP, None]], axis=-1)
    return out


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:  # (C,H,W) -> (H,W,C)
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _arm_raw_to_pose10(pos: np.ndarray, quat_wxyz: np.ndarray, grip: np.ndarray) -> np.ndarray:
    """(...,3),(...,4 wxyz),(...,) -> (...,10) = [pos3, rot6d6, grip1]."""
    rot = _quat_wxyz_to_mat(quat_wxyz)  # (...,3,3)
    rot6d = np.concatenate([rot[..., :, 0], rot[..., :, 1]], axis=-1)
    return np.concatenate([pos, rot6d, grip[..., None]], axis=-1)


def _raw23_to_pose20(vec23: np.ndarray) -> np.ndarray:
    """Raw (...,23) two-arm + ego state/action -> (...,20) two-arm 6D pose.

    Drops the trailing 7 ego dims; converts each arm's wxyz quaternion to rot6d.
    """
    left = _arm_raw_to_pose10(vec23[..., _RAW_LEFT_POS], vec23[..., _RAW_LEFT_QUAT_WXYZ], vec23[..., _RAW_LEFT_GRIP])
    right = _arm_raw_to_pose10(
        vec23[..., _RAW_RIGHT_POS], vec23[..., _RAW_RIGHT_QUAT_WXYZ], vec23[..., _RAW_RIGHT_GRIP]
    )
    return np.concatenate([left, right], axis=-1)


@dataclasses.dataclass(frozen=True)
class UmiDualArmInputs(transforms.DataTransformFn):
    """Maps the dual-arm dataset into pi0 inputs and applies UMI relativization (6D rot).

    There is no third-person/head camera: the two wrist views fill the left/right
    wrist slots and the base slot is zero-padded and masked out.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # 23-dim raw state -> 20-dim two-arm 6D pose (drops 7 ego dims).
        state = _raw23_to_pose20(np.asarray(data["state"], dtype=np.float64))  # (20,)

        left_wrist = _parse_image(data["left_wrist_image"])
        right_wrist = _parse_image(data["right_wrist_image"])

        mask_base = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
        inputs = {
            "state": state.astype(np.float32),
            "image": {
                "base_0_rgb": np.zeros_like(left_wrist),  # no head cam
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": mask_base,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            # (T,23) raw absolute -> (T,20) -> UMI relative trajectory in SE(3).
            actions = _raw23_to_pose20(np.asarray(data["actions"], dtype=np.float64))
            inputs["actions"] = _relativize_actions(state, actions).astype(np.float32)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UmiDualArmOutputs(transforms.DataTransformFn):
    """Converts the model's relative 6D action chunk back to absolute EE poses."""

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float64)[:STATE_DIM]
        rel = np.asarray(data["actions"], dtype=np.float64)[:, :STATE_DIM]
        return {"actions": _absolutize_actions(state, rel).astype(np.float32)}
