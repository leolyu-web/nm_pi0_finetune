"""Data transforms for a dual-arm dataset with UMI-style 6D-rotation EE poses.

This is the 6D-rotation sibling of ``umi_dual_arm_policy.py`` (which keeps the
rotation as a quaternion). The 6D representation (Zhou et al. 2019, the first two
**rows** of the rotation matrix, reconstructed via Gram-Schmidt) is **continuous
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
[pos3, rot6d6, grip1] -- by dropping the last 7 ego dims. Following UMI exactly,
the rotation is processed in SE(3) and only flattened to 6D **last**:

  * STATE  : per arm quat -> 3x3 matrix -> rot6d (rows). Kept ABSOLUTE.
  * ACTIONS: per arm the matrix is built directly from the quaternion, then the
             chunk is relativized in SE(3) against the current observation pose,

                 T_rel = inv(T_obs) @ T_action

             and the *relative matrix* is converted to rot6d (rows) only at the
             end. No intermediate quat->rot6d->matrix round-trip on the actions.

The gripper width is left absolute. This load-time relativization is the
equivalent of pi0's linear ``DeltaActions`` but composed in SE(3) -- so pi0's
built-in delta transform MUST stay OFF for this dataset.

``UmiDualArmOutputs`` inverts the relativization at inference: given the current
(20-dim, post-input-transform) rot6d state and the model's relative rot6d action
chunk, it returns absolute 20-dim poses. The model's predicted 6D vector need not
be orthonormal -- Gram-Schmidt re-orthonormalizes it when lifting back to a
matrix. The deployed runtime is responsible for any final 20-dim -> robot-native
conversion.
"""

import dataclasses

import einops
import numpy as np
from scipy.spatial.transform import Rotation

from openpi import transforms
from openpi.models import model as _model

# Per-arm OUTPUT layout (rot6d): [pos3, rot6d6, grip1].
_POS = slice(0, 3)
_ROT6D = slice(3, 9)
_GRIP = 9
ARM_DIM = 10
N_ARMS = 2
STATE_DIM = ARM_DIM * N_ARMS  # 20

# Per-arm INTERMEDIATE quaternion layout: [pos3, quat_wxyz4, grip1].
_Q_POS = slice(0, 3)
_Q_QUAT_WXYZ = slice(3, 7)
_Q_GRIP = 7
QUAT_ARM_DIM = 8
QUAT_STATE_DIM = QUAT_ARM_DIM * N_ARMS  # 16

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
#
# Convention: the 6D vector is the first two **ROWS** of the rotation matrix
# (matching UMI's ``mat_to_rot6d`` / ``rot6d_to_mat``), NOT the first two columns.
# --------------------------------------------------------------------------- #
def _rot6d_to_mat(rot6d: np.ndarray) -> np.ndarray:
    """[..., 6] (two stacked ROWS) -> 3x3 rotation matrix via Gram-Schmidt.

    Re-orthonormalizes defensively, so a non-orthonormal predicted 6D vector still
    decodes to a valid rotation. ``b1, b2, b3`` become the rows of the matrix.
    """
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def _mat_to_rot6d(mat: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> [..., 6] (its first two ROWS)."""
    return np.concatenate([mat[..., 0, :], mat[..., 1, :]], axis=-1)


def _quat_wxyz_to_mat(quat_wxyz: np.ndarray) -> np.ndarray:
    """[..., w, x, y, z] -> 3x3 rotation matrix (defensively re-normalized)."""
    quat_xyzw = np.concatenate([quat_wxyz[..., 1:4], quat_wxyz[..., 0:1]], axis=-1)
    quat_xyzw = quat_xyzw / np.linalg.norm(quat_xyzw, axis=-1, keepdims=True)
    return Rotation.from_quat(quat_xyzw).as_matrix()


def _quatpose_to_mat(pose8: np.ndarray) -> np.ndarray:
    """(...,8)=[pos3,quat_wxyz4,grip1] -> (4x4 mat). Grip column is ignored.

    Builds the SE(3) matrix directly from the quaternion (no rot6d round-trip).
    """
    out = np.zeros(pose8.shape[:-1] + (4, 4), dtype=np.float64)
    out[..., :3, :3] = _quat_wxyz_to_mat(pose8[..., _Q_QUAT_WXYZ])
    out[..., :3, 3] = pose8[..., _Q_POS]
    out[..., 3, 3] = 1.0
    return out


def _rot6dpose_to_mat(pose10: np.ndarray) -> np.ndarray:
    """(...,10)=[pos3,rot6d6,grip1] -> (4x4 mat). Grip column is ignored."""
    out = np.zeros(pose10.shape[:-1] + (4, 4), dtype=np.float64)
    out[..., :3, :3] = _rot6d_to_mat(pose10[..., _ROT6D])
    out[..., :3, 3] = pose10[..., _POS]
    out[..., 3, 3] = 1.0
    return out


def _mat_to_pose9(mat: np.ndarray) -> np.ndarray:
    """(...,4,4) -> (...,9) = [pos3, rot6d6] (rows convention)."""
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


def _relativize_actions(state_quat: np.ndarray, actions_quat: np.ndarray) -> np.ndarray:
    """Express each action pose relative to the current state pose, per arm (UMI).

    Inputs are in the intermediate QUATERNION layout; the SE(3) matrix is built
    directly from the quaternion, the chunk is relativized, and only the relative
    matrix is converted to rot6d (rows) -- i.e. rot6d is produced last.

    state_quat: (16,) current absolute EE pose [pos3,quat4,grip1] per arm.
    actions_quat: (T,16) absolute targets, same layout.
    Returns (T,20): per arm [rel_pos3, rel_rot6d6, grip1] (gripper stays absolute).
    """
    out = np.empty((actions_quat.shape[0], STATE_DIM), dtype=np.float64)
    for a in range(N_ARMS):
        sl_q = slice(a * QUAT_ARM_DIM, (a + 1) * QUAT_ARM_DIM)
        sl_o = slice(a * ARM_DIM, (a + 1) * ARM_DIM)
        block = actions_quat[:, sl_q]
        base = _quatpose_to_mat(state_quat[sl_q])  # (4,4), built from quat
        act = _quatpose_to_mat(block)  # (T,4,4), built from quat
        rel = _mat_inv(base)[None] @ act  # (T,4,4)
        out[:, sl_o] = np.concatenate([_mat_to_pose9(rel), block[:, _Q_GRIP, None]], axis=-1)
    return out


def _absolutize_actions(state_rot6d: np.ndarray, rel_actions: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_relativize_actions` (inference path).

    Operates in the rot6d layout: the (post-input-transform) state and the model's
    relative chunk are both rot6d (rows). Lifts to SE(3), composes ``base @ rel``,
    and reads back rot6d (rows).
    """
    out = np.empty_like(rel_actions)
    for a in range(N_ARMS):
        sl = slice(a * ARM_DIM, (a + 1) * ARM_DIM)
        block = rel_actions[:, sl]
        base = _rot6dpose_to_mat(state_rot6d[sl])  # (4,4)
        rel = _rot6dpose_to_mat(block)  # (T,4,4); grip column ignored
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


def _raw23_to_quatpose16(vec23: np.ndarray) -> np.ndarray:
    """Raw (...,23) two-arm + ego state/action -> (...,16) two-arm quaternion pose.

    Drops the trailing 7 ego dims; keeps each arm's (w,x,y,z) quaternion as-is.
    The quat->rot6d conversion is deliberately deferred (done last, after SE(3)).
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


def _quatpose16_to_rot6d20(pose16: np.ndarray) -> np.ndarray:
    """(...,16)=[pos3,quat4,grip1] per arm -> (...,20)=[pos3,rot6d6,grip1] per arm.

    Builds the matrix from the quaternion, then flattens to rot6d (rows) last.
    Used for the ABSOLUTE state feature (no relativization).
    """
    arms = []
    for a in range(N_ARMS):
        sl = slice(a * QUAT_ARM_DIM, (a + 1) * QUAT_ARM_DIM)
        block = pose16[..., sl]
        rot6d = _mat_to_rot6d(_quat_wxyz_to_mat(block[..., _Q_QUAT_WXYZ]))
        arms.append(np.concatenate([block[..., _Q_POS], rot6d, block[..., _Q_GRIP, None]], axis=-1))
    return np.concatenate(arms, axis=-1)


@dataclasses.dataclass(frozen=True)
class UmiDualArmInputs(transforms.DataTransformFn):
    """Maps the dual-arm dataset into pi0 inputs and applies UMI relativization (6D rot).

    There is no third-person/head camera: the two wrist views fill the left/right
    wrist slots and the base slot is zero-padded and masked out.
    """

    model_type: _model.ModelType
    # When True, drop the absolute EE pose (per-arm position + orientation) from the
    # state feature the model sees -- only the gripper widths survive. The action
    # chunk is still relativized against the TRUE pose, and the true pose is
    # forwarded at inference via the ``absolute_state`` side channel so
    # UmiDualArmOutputs can still absolutize (no deployed-runtime change). A config
    # that sets this MUST recompute norm_stats (the state distribution changes).
    mask_absolute_state_pose: bool = False
    # Only meaningful with ``mask_absolute_state_pose``: keep the per-arm absolute
    # z (height) position in the model's state feature (x, y, and orientation stay
    # masked). Lets the policy condition on height while staying blind to absolute
    # planar position + orientation.
    keep_z_position_in_state: bool = False

    def __call__(self, data: dict) -> dict:
        # 23-dim raw -> 16-dim two-arm QUATERNION pose (drops 7 ego dims).
        state_quat = _raw23_to_quatpose16(np.asarray(data["state"], dtype=np.float64))  # (16,)
        # ABSOLUTE state: quat -> matrix -> rot6d (rows), last. Kept unmasked below to
        # relativize the actions and to fill the ``absolute_state`` side channel.
        state = _quatpose16_to_rot6d20(state_quat)  # (20,)

        left_wrist = _parse_image(data["left_wrist_image"])
        right_wrist = _parse_image(data["right_wrist_image"])

        # The state feature actually fed to the model. With mask_absolute_state_pose
        # on, only the per-arm gripper survives; position + rot6d are zeroed so the
        # policy cannot condition on the absolute end-effector pose.
        model_state = state
        if self.mask_absolute_state_pose:
            model_state = np.zeros_like(state)
            for a in range(N_ARMS):
                base = a * ARM_DIM
                model_state[base + _GRIP] = state[base + _GRIP]
                if self.keep_z_position_in_state:
                    model_state[base + _POS.start + 2] = state[base + _POS.start + 2]  # absolute z (height)

        mask_base = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
        inputs = {
            "state": model_state.astype(np.float32),
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
            # (T,23) raw absolute -> (T,16) quat -> matrix-from-quat -> SE(3)
            # relativize -> rot6d (rows) last -> (T,20) UMI relative trajectory.
            actions_quat = _raw23_to_quatpose16(np.asarray(data["actions"], dtype=np.float64))
            inputs["actions"] = _relativize_actions(state_quat, actions_quat).astype(np.float32)
        elif self.mask_absolute_state_pose:
            # Inference path (no actions): the model no longer sees the absolute pose,
            # so carry the true (rot6d-20) pose through a side channel for the
            # UmiDualArmOutputs absolutize base. Skipped when actions are present
            # (training / norm-stats) since outputs never run there.
            inputs["absolute_state"] = state.astype(np.float32)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UmiDualArmOutputs(transforms.DataTransformFn):
    """Converts the model's relative 6D action chunk back to absolute EE poses."""

    def __call__(self, data: dict) -> dict:
        # Absolutize base: prefer the ``absolute_state`` side channel when present
        # (set by UmiDualArmInputs.mask_absolute_state_pose, where the model's own
        # ``state`` has the pose masked out); otherwise fall back to ``state``.
        base = data["absolute_state"] if "absolute_state" in data else data["state"]
        state = np.asarray(base, dtype=np.float64)[:STATE_DIM]
        rel = np.asarray(data["actions"], dtype=np.float64)[:, :STATE_DIM]
        return {"actions": _absolutize_actions(state, rel).astype(np.float32)}
