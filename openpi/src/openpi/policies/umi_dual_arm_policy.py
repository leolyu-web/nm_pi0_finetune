"""Data transforms for a dual-arm dataset with UMI-style EE poses (quaternion).

This is task-agnostic (the earphone dataset is just one example). The on-disk
LeRobot dataset is consumed *as-is* (no offline conversion); per frame it stores:
    observation.state : (23,) = left[pos3, quat_wxyz4, grip1] + right[pos3, quat_wxyz4, grip1] + ego7  (ABSOLUTE)
    action            : (23,) = same layout, absolute target EE pose
    observation.images.wrist_image_1, observation.images.wrist_image_2 : uint8 frames
    observation.images.image : head cam (loaded by LeRobot but dropped here)

``UmiDualArmInputs`` slices the 23-dim vector down to a 16-dim two-arm pose --
per arm [pos3, quat_wxyz4, grip1] -- by dropping the last 7 ego dims. The
rotation is kept as a (w,x,y,z) quaternion (NOT converted to 6D). It then maps
everything into pi0's expected dict and -- crucially -- converts the absolute
action *chunk* into a UMI relative trajectory: every pose in the chunk is
expressed in the frame of the current observation pose,

    T_rel = inv(T_obs) @ T_action

per arm, composed in SE(3) (the quaternion is lifted to a rotation matrix for
the SE(3) math, then read back out as a quaternion). The relative quaternion is
canonicalized to w >= 0 to remove the double-cover ambiguity so pi0 regresses a
single consistent target. The gripper width is left absolute. This is the
load-time equivalent of pi0's linear ``DeltaActions`` but composed in SE(3) on
the orientation -- so pi0's built-in delta transform MUST stay OFF for this
dataset.

``UmiDualArmOutputs`` inverts the relativization at inference: given the current
(16-dim, post-input-transform) state and the model's relative action chunk, it
returns absolute 16-dim poses. The model's predicted quaternion need not be unit
norm -- it is re-normalized when lifted back to a rotation matrix. The deployed
runtime is responsible for any final 16-dim -> robot-native action conversion.
"""

import dataclasses

import einops
import numpy as np
from scipy.spatial.transform import Rotation

from openpi import transforms
from openpi.models import model as _model

# Per-arm layout within the 8-dim block.
_POS = slice(0, 3)
_QUAT_WXYZ = slice(3, 7)
_GRIP = 7
ARM_DIM = 8
N_ARMS = 2
STATE_DIM = ARM_DIM * N_ARMS  # 16

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
# quaternion / SE(3) helpers (numpy + scipy). Quaternions are (w, x, y, z).
# --------------------------------------------------------------------------- #
def _quat_wxyz_to_mat(quat_wxyz: np.ndarray) -> np.ndarray:
    """[..., w, x, y, z] -> 3x3 rotation matrix (defensively re-normalized)."""
    quat_xyzw = np.concatenate([quat_wxyz[..., 1:4], quat_wxyz[..., 0:1]], axis=-1)
    quat_xyzw = quat_xyzw / np.linalg.norm(quat_xyzw, axis=-1, keepdims=True)
    return Rotation.from_quat(quat_xyzw).as_matrix()


def _mat_to_quat_wxyz(mat: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> [..., w, x, y, z], canonicalized to w >= 0.

    Forcing w >= 0 picks a single representative from the quaternion double cover
    (q and -q are the same rotation), so the regression target is unambiguous.
    """
    quat_xyzw = Rotation.from_matrix(mat).as_quat()
    quat_wxyz = np.concatenate([quat_xyzw[..., 3:4], quat_xyzw[..., 0:3]], axis=-1)
    return np.where(quat_wxyz[..., 0:1] < 0, -quat_wxyz, quat_wxyz)


def _pose8_to_mat(pose8: np.ndarray) -> np.ndarray:
    """(...,8)=[pos3,quat_wxyz4,grip1] -> (4x4 mat). Grip column is ignored."""
    out = np.zeros(pose8.shape[:-1] + (4, 4), dtype=np.float64)
    out[..., :3, :3] = _quat_wxyz_to_mat(pose8[..., _QUAT_WXYZ])
    out[..., :3, 3] = pose8[..., _POS]
    out[..., 3, 3] = 1.0
    return out


def _mat_to_pose7(mat: np.ndarray) -> np.ndarray:
    """(...,4,4) -> (...,7) = [pos3, quat_wxyz4]."""
    return np.concatenate([mat[..., :3, 3], _mat_to_quat_wxyz(mat[..., :3, :3])], axis=-1)


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

    state: (16,) current absolute EE pose. actions: (T,16) absolute targets.
    Returns (T,16): per arm [rel_pos3, rel_quat_wxyz4, grip1] (gripper stays absolute).
    """
    out = np.empty_like(actions)
    for a in range(N_ARMS):
        sl = slice(a * ARM_DIM, (a + 1) * ARM_DIM)
        block = actions[:, sl]
        base = _pose8_to_mat(state[sl])  # (4,4)
        act = _pose8_to_mat(block)  # (T,4,4)
        rel = _mat_inv(base)[None] @ act  # (T,4,4)
        out[:, sl] = np.concatenate(
            [_mat_to_pose7(rel), block[:, _GRIP, None]], axis=-1
        )
    return out


def _absolutize_actions(state: np.ndarray, rel_actions: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_relativize_actions` (inference path)."""
    out = np.empty_like(rel_actions)
    for a in range(N_ARMS):
        sl = slice(a * ARM_DIM, (a + 1) * ARM_DIM)
        block = rel_actions[:, sl]
        base = _pose8_to_mat(state[sl])  # (4,4)
        rel = _pose8_to_mat(block)  # (T,4,4); grip column ignored by _pose8_to_mat
        absm = base[None] @ rel
        out[:, sl] = np.concatenate(
            [_mat_to_pose7(absm), block[:, _GRIP, None]], axis=-1
        )
    return out


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:  # (C,H,W) -> (H,W,C)
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _raw23_to_pose16(vec23: np.ndarray) -> np.ndarray:
    """Raw (...,23) two-arm + ego state/action -> (...,16) two-arm pose.

    Drops the trailing 7 ego dims; keeps each arm's (w,x,y,z) quaternion as-is.
    """
    left = np.concatenate(
        [vec23[..., _RAW_LEFT_POS], vec23[..., _RAW_LEFT_QUAT_WXYZ], vec23[..., _RAW_LEFT_GRIP, None]],
        axis=-1,
    )
    right = np.concatenate(
        [vec23[..., _RAW_RIGHT_POS], vec23[..., _RAW_RIGHT_QUAT_WXYZ], vec23[..., _RAW_RIGHT_GRIP, None]],
        axis=-1,
    )
    return np.concatenate([left, right], axis=-1)


# PLACEHOLDER_CLASSES
@dataclasses.dataclass(frozen=True)
class UmiDualArmInputs(transforms.DataTransformFn):
    """Maps the dual-arm dataset into pi0 inputs and applies UMI relativization.

    There is no third-person/head camera: the two wrist views fill the left/right
    wrist slots and the base slot is zero-padded and masked out.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # 23-dim raw state -> 16-dim two-arm pose (drops 7 ego dims, keeps quat).
        state = _raw23_to_pose16(np.asarray(data["state"], dtype=np.float64))  # (16,)

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
            # (T,23) raw absolute -> (T,16) -> UMI relative trajectory in SE(3).
            actions = _raw23_to_pose16(np.asarray(data["actions"], dtype=np.float64))
            inputs["actions"] = _relativize_actions(state, actions).astype(np.float32)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UmiDualArmOutputs(transforms.DataTransformFn):
    """Converts the model's relative action chunk back to absolute EE poses."""

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float64)[:STATE_DIM]
        rel = np.asarray(data["actions"], dtype=np.float64)[:, :STATE_DIM]
        return {"actions": _absolutize_actions(state, rel).astype(np.float32)}
