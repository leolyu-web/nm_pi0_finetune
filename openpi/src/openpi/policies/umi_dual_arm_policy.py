"""Data transforms for a dual-arm dataset with UMI-style 6D EE poses.

This is task-agnostic (the earphone dataset is just one example). The on-disk
dataset (produced by ``examples/umi_dual_arm/convert_data_to_lerobot.py``) stores,
per frame:
    state   : (20,)  = left[pos3, rot6d6, grip1] + right[pos3, rot6d6, grip1]  (ABSOLUTE)
    actions : (20,)  = same layout, absolute target EE pose
    left_wrist_image, right_wrist_image : (320, 240, 3) uint8

``UmiDualArmInputs`` maps these into pi0's expected dict, and -- crucially --
converts the absolute action *chunk* into a UMI relative trajectory: every pose
in the chunk is expressed in the frame of the current observation pose,

    T_rel = inv(T_obs) @ T_action

per arm, in SE(3). The gripper width is left absolute. This is the load-time
equivalent of pi0's linear ``DeltaActions`` but composed in SE(3) on the 6D
rotation -- so pi0's built-in delta transform MUST stay OFF for this dataset.

``UmiDualArmOutputs`` inverts the relativization at inference: given the current
state and the model's relative action chunk, it returns absolute poses.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# Per-arm layout within the 10-dim block.
_POS = slice(0, 3)
_ROT6D = slice(3, 9)
_GRIP = 9
ARM_DIM = 10
N_ARMS = 2
STATE_DIM = ARM_DIM * N_ARMS  # 20


# --------------------------------------------------------------------------- #
# rotation-6D / SE(3) helpers (numpy + scipy), matching examples/umi_dual_arm/pose_util.py
# --------------------------------------------------------------------------- #
def _rot6d_to_mat(rot6d: np.ndarray) -> np.ndarray:
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def _mat_to_rot6d(mat: np.ndarray) -> np.ndarray:
    return np.concatenate([mat[..., :, 0], mat[..., :, 1]], axis=-1)


def _pose10_to_mat(pose10: np.ndarray) -> np.ndarray:
    """(...,10)=[pos3,rot6d6,grip1] -> (4x4 mat, grip)."""
    out = np.zeros(pose10.shape[:-1] + (4, 4), dtype=np.float64)
    out[..., :3, :3] = _rot6d_to_mat(pose10[..., _ROT6D])
    out[..., :3, 3] = pose10[..., _POS]
    out[..., 3, 3] = 1.0
    return out


def _mat_to_pose9(mat: np.ndarray) -> np.ndarray:
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
        out[:, sl] = np.concatenate(
            [_mat_to_pose9(rel), block[:, _GRIP, None]], axis=-1
        )
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
        out[:, sl] = np.concatenate(
            [_mat_to_pose9(absm), block[:, _GRIP, None]], axis=-1
        )
    return out


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:  # (C,H,W) -> (H,W,C)
        image = einops.rearrange(image, "c h w -> h w c")
    return image
# PLACEHOLDER_CLASSES
@dataclasses.dataclass(frozen=True)
class UmiDualArmInputs(transforms.DataTransformFn):
    """Maps the dual-arm dataset into pi0 inputs and applies UMI relativization.

    There is no third-person/head camera: the two wrist views fill the left/right
    wrist slots and the base slot is zero-padded and masked out.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float64)  # (20,)

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
            actions = np.asarray(data["actions"], dtype=np.float64)  # (T,20)
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
