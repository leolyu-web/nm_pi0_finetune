"""Apply the UMI dual-arm transform to the FIRST episode only -- QUATERNION variant.

This is a *verification* variant of the training transform. It performs the SAME
SE(3) UMI relativization, but in the FINAL stage it keeps the rotation as a
quaternion (wxyz) instead of converting to 6D rotation, and it uses a chunk size
of 48 (vs. the training horizon of 50). NO normalization is applied.

Per-arm output layout (8 dims):  [pos3, quat_wxyz4, grip1]
Two arms -> 16-dim state / action vectors.

Pipeline (no Normalize, no model_transforms):
    raw 23-dim frame  ->  slice per-arm [pos3, quat4, grip1] (drop 7 ego dims)
                      ->  SE(3) relativize the action chunk: T_rel = inv(T_obs) @ T_act
                      ->  rotation kept as quaternion (canonicalized to w >= 0)

Outputs in ./outputs (images are NOT loaded or saved):
    states.npy        (N, 16)        absolute two-arm [pos, quat, grip]
    actions.npy       (N, 48, 16)    UMI-relative action chunk (quaternion)
    raw_states.npy    (N, 23)        raw on-disk state (for reference)
    summary.json      shapes, dtypes, relativization sanity checks
"""

import json
import pathlib

import numpy as np
from scipy.spatial.transform import Rotation

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

CONFIG_NAME = "pi0_umi_dual_arm"
CHUNK = 48  # verification chunk size (training uses 50)

N_ARMS = 2
ARM_DIM = 8  # [pos3, quat4, grip1]
STATE_DIM = ARM_DIM * N_ARMS  # 16

# Raw 23-dim layout (per arm: pos3 + quat_wxyz4 + grip1, then 7 ego dims dropped).
_RAW_ARM = (
    {"pos": slice(0, 3), "quat": slice(3, 7), "grip": 7},
    {"pos": slice(8, 11), "quat": slice(11, 15), "grip": 15},
)


# --------------------------------------------------------------------------- #
# quaternion / SE(3) helpers (wxyz convention, scipy uses xyzw internally)
# --------------------------------------------------------------------------- #
def _quat_wxyz_to_mat(quat_wxyz: np.ndarray) -> np.ndarray:
    xyzw = np.concatenate([quat_wxyz[..., 1:4], quat_wxyz[..., 0:1]], axis=-1)
    xyzw = xyzw / np.linalg.norm(xyzw, axis=-1, keepdims=True)
    return Rotation.from_quat(xyzw).as_matrix()


def _mat_to_quat_wxyz(mat: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(mat).as_quat()  # (...,4) xyzw
    wxyz = np.concatenate([xyzw[..., 3:4], xyzw[..., 0:3]], axis=-1)
    # Canonicalize the double cover: force w >= 0 (q and -q are the same rotation).
    sign = np.where(wxyz[..., 0:1] < 0, -1.0, 1.0)
    return wxyz * sign


def _pose_to_mat(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """(...,3),(...,4 wxyz) -> (...,4,4) homogeneous transform."""
    out = np.zeros(pos.shape[:-1] + (4, 4), dtype=np.float64)
    out[..., :3, :3] = _quat_wxyz_to_mat(quat_wxyz)
    out[..., :3, 3] = pos
    out[..., 3, 3] = 1.0
    return out


def _mat_inv(mat: np.ndarray) -> np.ndarray:
    r = mat[..., :3, :3]
    t = mat[..., :3, 3]
    r_inv = np.swapaxes(r, -1, -2)
    out = np.zeros_like(mat)
    out[..., :3, :3] = r_inv
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", r_inv, t)
    out[..., 3, 3] = 1.0
    return out


def _raw23_to_state16(vec23: np.ndarray) -> np.ndarray:
    """Raw (...,23) -> (...,16): per arm [pos3, quat_wxyz4, grip1] (absolute)."""
    arms = []
    for arm in _RAW_ARM:
        arms.append(
            np.concatenate(
                [vec23[..., arm["pos"]], vec23[..., arm["quat"]], vec23[..., arm["grip"], None]],
                axis=-1,
            )
        )
    return np.concatenate(arms, axis=-1)


def _relativize_quat(state16: np.ndarray, actions16: np.ndarray) -> np.ndarray:
    """Express each action pose relative to the current state pose, per arm (UMI).

    state16: (16,) current absolute pose. actions16: (T,16) absolute targets.
    Returns (T,16): per arm [rel_pos3, rel_quat_wxyz4, grip1] (gripper stays absolute).
    """
    out = np.empty_like(actions16)
    for a in range(N_ARMS):
        sl = slice(a * ARM_DIM, (a + 1) * ARM_DIM)
        block = actions16[:, sl]
        base = _pose_to_mat(state16[sl][0:3], state16[sl][3:7])  # (4,4)
        act = _pose_to_mat(block[:, 0:3], block[:, 3:7])  # (T,4,4)
        rel = _mat_inv(base)[None] @ act  # (T,4,4)
        rel_pos = rel[:, :3, 3]
        rel_quat = _mat_to_quat_wxyz(rel[:, :3, :3])
        out[:, sl] = np.concatenate([rel_pos, rel_quat, block[:, 7, None]], axis=-1)
    return out


def main() -> None:
    config = _config.get_config(CONFIG_NAME)
    data_config = config.data.create(config.assets_dirs, config.model)
    print(f"repo_id            : {data_config.repo_id}")
    print(f"chunk size (verify): {CHUNK}   (training horizon: {config.model.action_horizon})")
    print(f"action_sequence_key: {data_config.action_sequence_keys}")
    print("rotation output    : quaternion (wxyz), NOT rot6d")

    # Raw LeRobot dataset; load with CHUNK-length action chunks.
    dataset = _data_loader.create_torch_dataset(data_config, CHUNK, config.model)

    raw = dataset
    while not hasattr(raw, "episode_data_index") and hasattr(raw, "_dataset"):
        raw = raw._dataset
    ep_from = int(raw.episode_data_index["from"][0])
    ep_to = int(raw.episode_data_index["to"][0])
    print(f"episode 0 frames   : [{ep_from}, {ep_to})  ->  {ep_to - ep_from} frames")

    states, actions, raw_states = [], [], []
    prompt_seen = None

    for idx in range(ep_from, ep_to):
        frame = dataset[idx]
        raw_vec = np.asarray(frame["observation.state"], dtype=np.float64)  # (23,)
        raw_act = np.asarray(frame["action"], dtype=np.float64)  # (CHUNK,23)
        raw_states.append(raw_vec)

        state16 = _raw23_to_state16(raw_vec)  # (16,)
        act16 = _raw23_to_state16(raw_act)  # (CHUNK,16)
        rel = _relativize_quat(state16, act16)  # (CHUNK,16)

        states.append(state16.astype(np.float32))
        actions.append(rel.astype(np.float32))
        if prompt_seen is None and "prompt" in frame:
            prompt_seen = str(frame["prompt"])

    states = np.stack(states)        # (N, 16)
    actions = np.stack(actions)      # (N, 48, 16)
    raw_states = np.stack(raw_states)  # (N, 23)

    np.save(OUT / "states.npy", states)
    np.save(OUT / "actions.npy", actions)
    np.save(OUT / "raw_states.npy", raw_states)

    summary = _build_summary(states, actions, raw_states, prompt_seen)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote outputs to: {OUT}")


def _build_summary(states, actions, raw_states, prompt):
    # action[0] should be ~the current pose -> relative pose near identity
    # (rel_pos ~ 0, rel_quat ~ [1,0,0,0] wxyz).
    a0 = actions[:, 0, :]  # (N, 16)
    identity_quat = np.array([1, 0, 0, 0], dtype=np.float64)
    rel_checks = {}
    quat_norm_stats = {}
    for a, name in enumerate(("left", "right")):
        sl = slice(a * ARM_DIM, (a + 1) * ARM_DIM)
        pos = a0[:, sl][:, 0:3]
        quat = a0[:, sl][:, 3:7]
        rel_checks[name] = {
            "action0_rel_pos_mean_abs": float(np.mean(np.abs(pos))),
            "action0_rel_pos_max_abs": float(np.max(np.abs(pos))),
            "action0_rel_quat_dev_from_identity_mean_abs": float(
                np.mean(np.abs(quat - identity_quat))
            ),
        }
        # quaternion unit-norm over the whole relative chunk for this arm.
        all_quat = actions[:, :, sl][:, :, 3:7]
        norms = np.linalg.norm(all_quat, axis=-1)
        quat_norm_stats[name] = {
            "rel_quat_norm_mean": float(norms.mean()),
            "rel_quat_norm_min": float(norms.min()),
            "rel_quat_norm_max": float(norms.max()),
        }

    return {
        "config": CONFIG_NAME,
        "rotation_representation": "quaternion_wxyz",
        "chunk_size": CHUNK,
        "num_frames": int(states.shape[0]),
        "prompt": prompt,
        "shapes": {
            "raw_states": list(raw_states.shape),
            "states": list(states.shape),
            "actions": list(actions.shape),
        },
        "dtypes": {"states": str(states.dtype), "actions": str(actions.dtype)},
        "state_range": {
            "min": float(states.min()),
            "max": float(states.max()),
            "mean_abs": float(np.mean(np.abs(states))),
        },
        "action_range": {
            "min": float(actions.min()),
            "max": float(actions.max()),
            "mean_abs": float(np.mean(np.abs(actions))),
        },
        "relativization_check": rel_checks,
        "rel_quat_unit_norm_check": quat_norm_stats,
        "gripper_state_left_range": [float(states[:, 7].min()), float(states[:, 7].max())],
        "gripper_state_right_range": [float(states[:, 15].min()), float(states[:, 15].max())],
        "normalization_applied": False,
    }


if __name__ == "__main__":
    main()
