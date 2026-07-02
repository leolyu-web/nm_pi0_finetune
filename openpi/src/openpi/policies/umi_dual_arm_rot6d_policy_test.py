"""Tests for the 6D-rotation dual-arm UMI transform.

The 6D sibling of ``umi_dual_arm_policy_test``. Same SE(3) relativization
contract, but the rotation is a continuous 6D vector (first two ROWS of the
matrix, Gram-Schmidt decode) instead of a quaternion. These lock down the row
convention, the SE(3) round-trip, and that a non-orthonormal predicted 6D still
decodes to a valid rotation.

All tests are deterministic (seeded) and CPU-only; no checkpoints or hardware.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from openpi.models import model as _model
from openpi.policies import umi_dual_arm_rot6d_policy as umi


def _rand_quat_wxyz(rng: np.random.Generator, n: int) -> np.ndarray:
    quat_xyzw = Rotation.random(n, rng=rng).as_quat()
    return np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, 0:3]], axis=-1)


def _make_quatpose16(rng: np.random.Generator, n: int) -> np.ndarray:
    """(n,16) two-arm absolute pose in the intermediate quaternion layout."""
    out = np.empty((n, umi.QUAT_STATE_DIM), dtype=np.float64)
    for a in range(umi.N_ARMS):
        base = a * umi.QUAT_ARM_DIM
        out[:, base + 0 : base + 3] = rng.uniform(-0.5, 0.5, (n, 3))  # pos
        out[:, base + 3 : base + 7] = _rand_quat_wxyz(rng, n)  # quat
        out[:, base + 7] = rng.uniform(0.0, 1.0, n)  # grip
    return out


def _quatpose16_to_raw23(pose16: np.ndarray) -> np.ndarray:
    left = pose16[..., 0:8]
    right = pose16[..., 8:16]
    ego = np.full(pose16.shape[:-1] + (7,), 123.0)  # dropped by the transform
    return np.concatenate([left, right, ego], axis=-1)


def test_rot6d_mat_round_trip_rows_convention():
    """mat -> rot6d -> mat is exact, and rot6d is the first two ROWS of the matrix."""
    rng = np.random.default_rng(0)
    mats = Rotation.random(20, rng=rng).as_matrix()

    rot6d = umi._mat_to_rot6d(mats)
    back = umi._rot6d_to_mat(rot6d)

    np.testing.assert_allclose(back, mats, atol=1e-12)
    # Row convention: first 3 of the 6D vector are row 0, next 3 are row 1.
    np.testing.assert_allclose(rot6d[:, 0:3], mats[:, 0, :], atol=1e-12)
    np.testing.assert_allclose(rot6d[:, 3:6], mats[:, 1, :], atol=1e-12)


def test_rot6d_decodes_orthonormal():
    """Decoded matrices are proper rotations (orthonormal, det +1)."""
    rng = np.random.default_rng(1)
    rot6d = umi._mat_to_rot6d(Rotation.random(15, rng=rng).as_matrix())

    mats = umi._rot6d_to_mat(rot6d)

    eye = np.einsum("...ij,...kj->...ik", mats, mats)  # R @ R^T
    np.testing.assert_allclose(eye, np.broadcast_to(np.eye(3), eye.shape), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(mats), 1.0, atol=1e-10)


def test_relativize_absolutize_round_trip():
    """rel then abs recovers the absolute chunk; the actions path is quat-in, rot6d-out."""
    rng = np.random.default_rng(2)
    state_quat = _make_quatpose16(rng, 1)[0]
    actions_quat = _make_quatpose16(rng, 12)

    rel = umi._relativize_actions(state_quat, actions_quat)  # (T,20) rot6d layout
    # Absolutize needs the state in rot6d layout (post-input-transform).
    state_rot6d = umi._quatpose16_to_rot6d20(state_quat)
    recovered = umi._absolutize_actions(state_rot6d, rel)  # (T,20)

    expected = umi._quatpose16_to_rot6d20(actions_quat)
    for a in range(umi.N_ARMS):
        base = a * umi.ARM_DIM
        np.testing.assert_allclose(recovered[:, base : base + 3], expected[:, base : base + 3], atol=1e-9)  # pos
        got = umi._rot6d_to_mat(recovered[:, base + 3 : base + 9])
        want = umi._rot6d_to_mat(expected[:, base + 3 : base + 9])
        np.testing.assert_allclose(got, want, atol=1e-9)  # rotation
        np.testing.assert_allclose(recovered[:, base + 9], expected[:, base + 9], atol=1e-12)  # grip


def test_action_equal_to_state_gives_identity():
    """Action pose == state -> zero rel translation and identity-row rot6d."""
    rng = np.random.default_rng(3)
    state_quat = _make_quatpose16(rng, 1)[0]
    actions_quat = state_quat[None].copy()

    rel = umi._relativize_actions(state_quat, actions_quat)

    for a in range(umi.N_ARMS):
        base = a * umi.ARM_DIM
        np.testing.assert_allclose(rel[0, base : base + 3], 0.0, atol=1e-12)
        # Identity rotation rows: [1,0,0, 0,1,0].
        np.testing.assert_allclose(rel[0, base + 3 : base + 9], [1, 0, 0, 0, 1, 0], atol=1e-9)


def test_per_arm_independence():
    rng = np.random.default_rng(4)
    state_quat = _make_quatpose16(rng, 1)[0]
    actions_quat = _make_quatpose16(rng, 8)

    rel_a = umi._relativize_actions(state_quat, actions_quat)

    state_perturbed = state_quat.copy()
    state_perturbed[8:16] = _make_quatpose16(rng, 1)[0, 8:16]  # right arm only
    rel_b = umi._relativize_actions(state_perturbed, actions_quat)

    np.testing.assert_array_equal(rel_a[:, 0 : umi.ARM_DIM], rel_b[:, 0 : umi.ARM_DIM])
    assert not np.allclose(rel_a[:, umi.ARM_DIM :], rel_b[:, umi.ARM_DIM :])


def test_gripper_passed_through_absolute():
    rng = np.random.default_rng(5)
    state_quat = _make_quatpose16(rng, 1)[0]
    actions_quat = _make_quatpose16(rng, 10)

    rel = umi._relativize_actions(state_quat, actions_quat)

    for a in range(umi.N_ARMS):
        grip_rel = rel[:, a * umi.ARM_DIM + umi._GRIP]
        grip_src = actions_quat[:, a * umi.QUAT_ARM_DIM + umi._Q_GRIP]
        np.testing.assert_array_equal(grip_rel, grip_src)


def test_absolutize_tolerates_non_orthonormal_rot6d():
    """A non-orthonormal predicted 6D still decodes to a valid rotation (Gram-Schmidt)."""
    rng = np.random.default_rng(6)
    state_quat = _make_quatpose16(rng, 1)[0]
    state_rot6d = umi._quatpose16_to_rot6d20(state_quat)
    rel = np.zeros((5, umi.STATE_DIM))
    for a in range(umi.N_ARMS):
        base = a * umi.ARM_DIM
        rel[:, base + 3 : base + 9] = umi._mat_to_rot6d(Rotation.random(5, rng=rng).as_matrix())

    rel_scaled = rel.copy()
    for a in range(umi.N_ARMS):  # scale row-1 block; Gram-Schmidt should absorb it
        base = a * umi.ARM_DIM
        rel_scaled[:, base + 3 : base + 6] *= 2.5

    out = umi._absolutize_actions(state_rot6d, rel)
    out_scaled = umi._absolutize_actions(state_rot6d, rel_scaled)

    # Only the magnitude of the first row changed; the decoded rotation is identical.
    for a in range(umi.N_ARMS):
        base = a * umi.ARM_DIM
        got = umi._rot6d_to_mat(out_scaled[:, base + 3 : base + 9])
        want = umi._rot6d_to_mat(out[:, base + 3 : base + 9])
        np.testing.assert_allclose(got, want, atol=1e-9)


def test_raw23_drops_ego_and_state_is_20_dim():
    rng = np.random.default_rng(7)
    quat16 = _make_quatpose16(rng, 5)
    raw23 = _quatpose16_to_raw23(quat16)

    quat_out = umi._raw23_to_quatpose16(raw23)
    np.testing.assert_allclose(quat_out, quat16, atol=0)

    state20 = umi._quatpose16_to_rot6d20(quat16)
    assert state20.shape == (5, umi.STATE_DIM)  # 20-dim


def _make_data(rng, horizon=8):
    state16 = _make_quatpose16(rng, 1)[0]
    return {
        "state": _quatpose16_to_raw23(state16),
        "actions": _quatpose16_to_raw23(_make_quatpose16(rng, horizon)),
        "left_wrist_image": rng.random((3, 8, 8)).astype(np.float32),
        "right_wrist_image": rng.random((3, 8, 8)).astype(np.float32),
        "prompt": "do the thing",
    }


def test_inputs_produces_20_dim_shapes():
    rng = np.random.default_rng(8)
    data = _make_data(rng, horizon=8)

    out = umi.UmiDualArmInputs(model_type=_model.ModelType.PI0)(data)

    assert out["state"].shape == (umi.STATE_DIM,)  # 20
    assert out["actions"].shape == (8, umi.STATE_DIM)
    assert out["image"]["left_wrist_0_rgb"].shape == (8, 8, 3)
    assert out["image"]["left_wrist_0_rgb"].dtype == np.uint8
    assert np.all(out["image"]["base_0_rgb"] == 0)


def test_inputs_outputs_full_round_trip():
    """Inputs (relativize, rot6d) then Outputs (absolutize) recovers absolute targets."""
    rng = np.random.default_rng(9)
    data = _make_data(rng, horizon=10)

    inputs = umi.UmiDualArmInputs(model_type=_model.ModelType.PI0)(data)
    out = umi.UmiDualArmOutputs()({"state": inputs["state"], "actions": inputs["actions"]})

    expected = umi._quatpose16_to_rot6d20(umi._raw23_to_quatpose16(data["actions"].astype(np.float64)))
    for a in range(umi.N_ARMS):
        base = a * umi.ARM_DIM
        np.testing.assert_allclose(out["actions"][:, base : base + 3], expected[:, base : base + 3], atol=1e-5)
        got = umi._rot6d_to_mat(out["actions"][:, base + 3 : base + 9].astype(np.float64))
        want = umi._rot6d_to_mat(expected[:, base + 3 : base + 9])
        np.testing.assert_allclose(got, want, atol=1e-5)
        np.testing.assert_allclose(out["actions"][:, base + 9], expected[:, base + 9], atol=1e-5)


def test_mask_absolute_state_pose_keeps_only_gripper():
    """With masking on, the model state zeros pos+rot6d and keeps only the grippers."""
    rng = np.random.default_rng(10)
    data = _make_data(rng, horizon=8)

    inputs = umi.UmiDualArmInputs(model_type=_model.ModelType.PI0, mask_absolute_state_pose=True)(data)
    unmasked = umi._quatpose16_to_rot6d20(umi._raw23_to_quatpose16(data["state"].astype(np.float64)))

    state = inputs["state"]
    assert state.shape == (umi.STATE_DIM,)
    for a in range(umi.N_ARMS):
        base = a * umi.ARM_DIM
        np.testing.assert_array_equal(state[base : base + 9], 0.0)  # pos3 + rot6d6 zeroed
        assert state[base + umi._GRIP] == np.float32(unmasked[base + umi._GRIP])  # gripper survives

    # Actions are still relativized against the TRUE pose -> masking is state-only.
    ref = umi.UmiDualArmInputs(model_type=_model.ModelType.PI0)(data)
    np.testing.assert_array_equal(inputs["actions"], ref["actions"])


def test_masking_keeps_z_position():
    """keep_z retains the per-arm absolute z (height); x/y/rot6d stay masked."""
    rng = np.random.default_rng(11)
    data = _make_data(rng, horizon=6)

    inputs = umi.UmiDualArmInputs(
        model_type=_model.ModelType.PI0, mask_absolute_state_pose=True, keep_z_position_in_state=True
    )(data)
    unmasked = umi._quatpose16_to_rot6d20(umi._raw23_to_quatpose16(data["state"].astype(np.float64)))

    state = inputs["state"]
    for a in range(umi.N_ARMS):
        base = a * umi.ARM_DIM
        np.testing.assert_array_equal(state[base : base + 2], 0.0)  # x, y masked
        assert state[base + 2] == np.float32(unmasked[base + 2])  # z kept
        np.testing.assert_array_equal(state[base + 3 : base + 9], 0.0)  # rot6d masked
        assert state[base + umi._GRIP] == np.float32(unmasked[base + umi._GRIP])  # gripper kept


def test_masked_inference_side_channel_round_trip():
    """Masked inference: the absolute_state side channel drives the absolutize base."""
    rng = np.random.default_rng(12)
    data = _make_data(rng, horizon=10)

    # Training-style call (actions present) gives the relative targets to absolutize.
    train = umi.UmiDualArmInputs(model_type=_model.ModelType.PI0, mask_absolute_state_pose=True)(data)

    # Inference-style call: no actions -> the true pose is carried via absolute_state,
    # while the model's own state has pose masked out.
    infer_data = {k: v for k, v in data.items() if k != "actions"}
    infer = umi.UmiDualArmInputs(model_type=_model.ModelType.PI0, mask_absolute_state_pose=True)(infer_data)
    assert "absolute_state" in infer
    np.testing.assert_array_equal(infer["state"], train["state"])  # both masked identically

    out = umi.UmiDualArmOutputs()({
        "state": infer["state"],
        "absolute_state": infer["absolute_state"],
        "actions": train["actions"],
    })

    expected = umi._quatpose16_to_rot6d20(umi._raw23_to_quatpose16(data["actions"].astype(np.float64)))
    for a in range(umi.N_ARMS):
        base = a * umi.ARM_DIM
        np.testing.assert_allclose(out["actions"][:, base : base + 3], expected[:, base : base + 3], atol=1e-5)
        got = umi._rot6d_to_mat(out["actions"][:, base + 3 : base + 9].astype(np.float64))
        want = umi._rot6d_to_mat(expected[:, base + 3 : base + 9])
        np.testing.assert_allclose(got, want, atol=1e-5)
        np.testing.assert_allclose(out["actions"][:, base + 9], expected[:, base + 9], atol=1e-5)
