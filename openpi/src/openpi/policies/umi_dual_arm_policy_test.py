"""Tests for the quaternion dual-arm UMI transform.

These lock down the SE(3) relativization contract that pi0 depends on: the
action chunk is relativized at load time (``T_rel = inv(T_obs) @ T_act`` per arm)
and absolutized at inference, the gripper stays absolute, and the relative
quaternion is canonicalized to ``w >= 0``. If a refactor breaks the math, these
fail -- unlike the model-level tests, which never exercise this code.

All tests are deterministic (seeded) and CPU-only; no checkpoints or hardware.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from openpi.models import model as _model
from openpi.policies import umi_dual_arm_policy as umi

# Sample per-arm reframe W (inv(mean orientation)) for exercising the world_reframe path.
# The reframe field is still on UmiDualArmDataConfig; no shipped config sets it currently.
_V3_REFRAME = (
    0.45594531,
    -0.03043296,
    -0.23385571,
    0.85819533,  # left
    0.45044910,
    -0.75766090,
    0.41203593,
    -0.23080718,  # right
)


def _rand_quat_wxyz(rng: np.random.Generator, n: int) -> np.ndarray:
    """(n,4) random unit quaternions in (w,x,y,z), w>=0 for a stable reference."""
    quat_xyzw = Rotation.random(n, rng=rng).as_quat()
    quat_wxyz = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, 0:3]], axis=-1)
    return np.where(quat_wxyz[:, 0:1] < 0, -quat_wxyz, quat_wxyz)


def _make_pose16(rng: np.random.Generator, n: int) -> np.ndarray:
    """(n,16) two-arm absolute pose [pos3, quat_wxyz4, grip1] per arm."""
    out = np.empty((n, umi.STATE_DIM), dtype=np.float64)
    for a in range(umi.N_ARMS):
        sl = slice(a * umi.ARM_DIM, (a + 1) * umi.ARM_DIM)
        out[:, sl.start + 0 : sl.start + 3] = rng.uniform(-0.5, 0.5, (n, 3))  # pos
        out[:, sl.start + 3 : sl.start + 7] = _rand_quat_wxyz(rng, n)  # quat
        out[:, sl.start + 7] = rng.uniform(0.0, 1.0, n)  # grip
    return out


def _pose16_to_raw23(pose16: np.ndarray) -> np.ndarray:
    """(...,16) two-arm pose -> (...,23) raw vector (7 ego dims filled with junk)."""
    left = pose16[..., 0:8]
    right = pose16[..., 8:16]
    ego = np.full(pose16.shape[:-1] + (7,), 123.0)  # must be dropped by the transform
    return np.concatenate([left, right, ego], axis=-1)


def test_relativize_absolutize_round_trip():
    """rel then abs recovers the original absolute chunk (pos, rot, grip)."""
    rng = np.random.default_rng(0)
    state = _make_pose16(rng, 1)[0]
    actions = _make_pose16(rng, 12)

    rel = umi._relativize_actions(state, actions)
    recovered = umi._absolutize_actions(state, rel)

    # Positions and gripper compare directly.
    for a in range(umi.N_ARMS):
        base = a * umi.ARM_DIM
        np.testing.assert_allclose(recovered[:, base : base + 3], actions[:, base : base + 3], atol=1e-9)
        np.testing.assert_allclose(recovered[:, base + 7], actions[:, base + 7], atol=1e-12)
        # Quaternions compare up to sign (double cover): compare rotation matrices.
        got = umi._quat_wxyz_to_mat(recovered[:, base + 3 : base + 7])
        want = umi._quat_wxyz_to_mat(actions[:, base + 3 : base + 7])
        np.testing.assert_allclose(got, want, atol=1e-9)


def test_action_equal_to_state_gives_identity():
    """When an action pose equals the current state, its relative pose is identity."""
    rng = np.random.default_rng(1)
    state = _make_pose16(rng, 1)[0]
    actions = state[None].copy()  # (1,16) action == state

    rel = umi._relativize_actions(state, actions)

    for a in range(umi.N_ARMS):
        base = a * umi.ARM_DIM
        np.testing.assert_allclose(rel[0, base : base + 3], 0.0, atol=1e-12)  # zero rel translation
        # Identity rotation -> quat (1,0,0,0), canonicalized w>=0.
        np.testing.assert_allclose(rel[0, base + 3 : base + 7], [1.0, 0.0, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(rel[0, base + 7], state[base + 7], atol=1e-12)  # grip absolute


def test_relative_quaternion_canonicalized_w_nonneg():
    """The relativized quaternion always has w >= 0 (single double-cover branch)."""
    rng = np.random.default_rng(2)
    state = _make_pose16(rng, 1)[0]
    actions = _make_pose16(rng, 200)

    rel = umi._relativize_actions(state, actions)

    for a in range(umi.N_ARMS):
        w = rel[:, a * umi.ARM_DIM + 3]  # quat w component
        assert np.all(w >= 0.0)


def test_per_arm_independence():
    """Perturbing the right-arm state must not change the left-arm relative action."""
    rng = np.random.default_rng(3)
    state = _make_pose16(rng, 1)[0]
    actions = _make_pose16(rng, 8)

    rel_a = umi._relativize_actions(state, actions)

    state_perturbed = state.copy()
    state_perturbed[8:16] = _make_pose16(rng, 1)[0, 8:16]  # change only the right arm
    rel_b = umi._relativize_actions(state_perturbed, actions)

    np.testing.assert_array_equal(rel_a[:, 0:8], rel_b[:, 0:8])  # left arm identical
    assert not np.allclose(rel_a[:, 8:16], rel_b[:, 8:16])  # right arm changed


def test_gripper_passed_through_absolute():
    """The gripper width is never relativized -- it survives verbatim through rel."""
    rng = np.random.default_rng(4)
    state = _make_pose16(rng, 1)[0]
    actions = _make_pose16(rng, 10)

    rel = umi._relativize_actions(state, actions)

    for a in range(umi.N_ARMS):
        grip_idx = a * umi.ARM_DIM + 7
        np.testing.assert_array_equal(rel[:, grip_idx], actions[:, grip_idx])


def test_absolutize_tolerates_non_unit_quaternion():
    """A non-unit predicted quaternion decodes to a valid rotation (re-normalized)."""
    rng = np.random.default_rng(5)
    state = _make_pose16(rng, 1)[0]
    rel = _make_pose16(rng, 6)
    rel_scaled = rel.copy()
    for a in range(umi.N_ARMS):
        base = a * umi.ARM_DIM
        rel_scaled[:, base + 3 : base + 7] *= 3.7  # break unit norm

    out = umi._absolutize_actions(state, rel)
    out_scaled = umi._absolutize_actions(state, rel_scaled)

    # Scaling the quaternion is a no-op after re-normalization.
    np.testing.assert_allclose(out, out_scaled, atol=1e-9)


def test_raw23_drops_ego_and_keeps_quat():
    """The 23->16 slice drops the 7 ego dims and preserves each arm's quaternion."""
    rng = np.random.default_rng(6)
    pose16 = _make_pose16(rng, 5)
    raw23 = _pose16_to_raw23(pose16)

    got = umi._raw23_to_pose16(raw23)

    np.testing.assert_allclose(got, pose16, atol=0)  # exact slice, no arithmetic


def _make_data(rng, horizon=8, *, with_actions=True):
    state16 = _make_pose16(rng, 1)[0]
    data = {
        "state": _pose16_to_raw23(state16),
        "left_wrist_image": (rng.random((3, 8, 8))).astype(np.float32),  # CHW float in [0,1]
        "right_wrist_image": (rng.random((3, 8, 8))).astype(np.float32),
        "prompt": "do the thing",
    }
    if with_actions:
        data["actions"] = _pose16_to_raw23(_make_pose16(rng, horizon))
    return data


def test_inputs_produces_expected_shapes_and_images():
    """Inputs yields a 16-dim state, relativized actions, and HWC uint8 wrist images."""
    rng = np.random.default_rng(7)
    data = _make_data(rng, horizon=8)

    out = umi.UmiDualArmInputs(model_type=_model.ModelType.PI0)(data)

    assert out["state"].shape == (umi.STATE_DIM,)
    assert out["actions"].shape == (8, umi.STATE_DIM)
    # Wrist images: parsed to HWC uint8; base slot zeroed and masked off.
    assert out["image"]["left_wrist_0_rgb"].shape == (8, 8, 3)
    assert out["image"]["left_wrist_0_rgb"].dtype == np.uint8
    assert np.all(out["image"]["base_0_rgb"] == 0)
    assert bool(out["image_mask"]["left_wrist_0_rgb"]) is True
    assert bool(out["image_mask"]["base_0_rgb"]) is False


def test_inputs_outputs_full_round_trip():
    """Inputs (relativize) then Outputs (absolutize) recovers the absolute targets."""
    rng = np.random.default_rng(8)
    data = _make_data(rng, horizon=10)

    inputs = umi.UmiDualArmInputs(model_type=_model.ModelType.PI0)(data)
    # Outputs consumes the post-input-transform state + the (relativized) action chunk.
    out = umi.UmiDualArmOutputs()({"state": inputs["state"], "actions": inputs["actions"]})

    absolute_expected = umi._raw23_to_pose16(data["actions"].astype(np.float64))
    for a in range(umi.N_ARMS):
        base = a * umi.ARM_DIM
        np.testing.assert_allclose(out["actions"][:, base : base + 3], absolute_expected[:, base : base + 3], atol=1e-5)
        np.testing.assert_allclose(out["actions"][:, base + 7], absolute_expected[:, base + 7], atol=1e-5)
        got = umi._quat_wxyz_to_mat(out["actions"][:, base + 3 : base + 7].astype(np.float64))
        want = umi._quat_wxyz_to_mat(absolute_expected[:, base + 3 : base + 7])
        np.testing.assert_allclose(got, want, atol=1e-5)


def test_world_reframe_leaves_relative_target_invariant():
    """Option-2 reframe (v3): the relative action target is unchanged (W cancels)."""
    rng = np.random.default_rng(9)
    data = _make_data(rng, horizon=12)

    plain = umi.UmiDualArmInputs(model_type=_model.ModelType.PI0)(data)
    reframed = umi.UmiDualArmInputs(model_type=_model.ModelType.PI0, world_reframe_quat_wxyz=_V3_REFRAME)(data)

    # The regressed target (relative actions) must be identical; only the absolute
    # state feature moves.
    for a in range(umi.N_ARMS):
        base = a * umi.ARM_DIM
        np.testing.assert_allclose(
            reframed["actions"][:, base : base + 3], plain["actions"][:, base : base + 3], atol=1e-5
        )
        got = umi._quat_wxyz_to_mat(reframed["actions"][:, base + 3 : base + 7].astype(np.float64))
        want = umi._quat_wxyz_to_mat(plain["actions"][:, base + 3 : base + 7].astype(np.float64))
        np.testing.assert_allclose(got, want, atol=1e-5)
    assert not np.allclose(reframed["state"], plain["state"])  # state feature did move


def test_world_reframe_outputs_return_original_frame():
    """With reframe on, Outputs undoes W so the runtime sees the original-frame poses."""
    rng = np.random.default_rng(10)
    data = _make_data(rng, horizon=10)

    inputs = umi.UmiDualArmInputs(model_type=_model.ModelType.PI0, world_reframe_quat_wxyz=_V3_REFRAME)(data)
    out = umi.UmiDualArmOutputs(world_reframe_quat_wxyz=_V3_REFRAME)(
        {"state": inputs["state"], "actions": inputs["actions"]}
    )

    absolute_expected = umi._raw23_to_pose16(data["actions"].astype(np.float64))
    for a in range(umi.N_ARMS):
        base = a * umi.ARM_DIM
        np.testing.assert_allclose(out["actions"][:, base : base + 3], absolute_expected[:, base : base + 3], atol=1e-5)
        got = umi._quat_wxyz_to_mat(out["actions"][:, base + 3 : base + 7].astype(np.float64))
        want = umi._quat_wxyz_to_mat(absolute_expected[:, base + 3 : base + 7])
        np.testing.assert_allclose(got, want, atol=1e-5)


# --------------------------------------------------------------------------- #
# mask_absolute_state_pose: the model must not see the absolute EE pose, yet the
# action relativization and the inference-time absolutize base must be unaffected.
# --------------------------------------------------------------------------- #
def test_mask_hides_pose_but_keeps_gripper_and_actions():
    """With masking on, the model ``state`` has pose zeroed / gripper kept, and the
    relativized actions are byte-for-byte identical to the unmasked config."""
    rng = np.random.default_rng(11)
    data = _make_data(rng, horizon=9)

    plain = umi.UmiDualArmInputs(model_type=_model.ModelType.PI0)(data)
    masked = umi.UmiDualArmInputs(model_type=_model.ModelType.PI0, mask_absolute_state_pose=True)(data)

    for a in range(umi.N_ARMS):
        base = a * umi.ARM_DIM
        # Position + orientation zeroed in the model-facing state.
        np.testing.assert_array_equal(masked["state"][base : base + 7], np.zeros(7, dtype=np.float32))
        # Gripper preserved exactly.
        assert masked["state"][base + umi._GRIP] == plain["state"][base + umi._GRIP]
    # Masking the state must NOT change the regressed (relative) action target.
    np.testing.assert_array_equal(masked["actions"], plain["actions"])
    # No side channel while actions are present (training / norm-stats path).
    assert "absolute_state" not in masked


def test_mask_inference_side_channel_absolutizes_correctly():
    """Inference: model sees a masked state, but ``absolute_state`` lets Outputs
    recover the true absolute targets exactly (no runtime change needed)."""
    rng = np.random.default_rng(12)
    data = _make_data(rng, horizon=10)  # has actions -> gives us the relativized chunk

    train = umi.UmiDualArmInputs(model_type=_model.ModelType.PI0, mask_absolute_state_pose=True)(data)
    # Same observation, inference form (no actions) -> masked state + side channel.
    infer = umi.UmiDualArmInputs(model_type=_model.ModelType.PI0, mask_absolute_state_pose=True)(
        _drop_actions(data)
    )
    assert "absolute_state" in infer
    np.testing.assert_array_equal(infer["state"], train["state"])  # both masked identically

    out = umi.UmiDualArmOutputs()(
        {"state": infer["state"], "absolute_state": infer["absolute_state"], "actions": train["actions"]}
    )

    absolute_expected = umi._raw23_to_pose16(data["actions"].astype(np.float64))
    for a in range(umi.N_ARMS):
        base = a * umi.ARM_DIM
        np.testing.assert_allclose(out["actions"][:, base : base + 3], absolute_expected[:, base : base + 3], atol=1e-5)
        np.testing.assert_allclose(out["actions"][:, base + 7], absolute_expected[:, base + 7], atol=1e-5)
        got = umi._quat_wxyz_to_mat(out["actions"][:, base + 3 : base + 7].astype(np.float64))
        want = umi._quat_wxyz_to_mat(absolute_expected[:, base + 3 : base + 7])
        np.testing.assert_allclose(got, want, atol=1e-5)


def test_mask_keep_z_keeps_height_only():
    """z-axis variant: with masking + keep_z, the model state keeps per-arm absolute
    z (height) and gripper; x, y, and orientation stay zeroed. Actions and the
    absolutize side channel are unaffected."""
    rng = np.random.default_rng(14)
    data = _make_data(rng, horizon=7)

    plain = umi.UmiDualArmInputs(model_type=_model.ModelType.PI0)(data)
    masked = umi.UmiDualArmInputs(
        model_type=_model.ModelType.PI0, mask_absolute_state_pose=True, keep_z_position_in_state=True
    )(data)

    for a in range(umi.N_ARMS):
        base = a * umi.ARM_DIM
        # x, y zeroed; z (index 2) preserved; quaternion zeroed; gripper preserved.
        np.testing.assert_array_equal(masked["state"][base : base + 2], np.zeros(2, dtype=np.float32))
        assert masked["state"][base + 2] == plain["state"][base + 2]  # absolute z kept
        np.testing.assert_array_equal(masked["state"][base + 3 : base + 7], np.zeros(4, dtype=np.float32))
        assert masked["state"][base + umi._GRIP] == plain["state"][base + umi._GRIP]
    # Relativized action target unchanged by any state masking.
    np.testing.assert_array_equal(masked["actions"], plain["actions"])


def test_gripper_action_equals_state_pins_gripper():
    """Teleop gripper-bug: every action's gripper equals the observation gripper
    (per arm) for all timesteps, while the pose dims match the plain relativization."""
    rng = np.random.default_rng(15)
    data = _make_data(rng, horizon=9)

    plain = umi.UmiDualArmInputs(model_type=_model.ModelType.PI0)(data)
    bug = umi.UmiDualArmInputs(model_type=_model.ModelType.PI0, gripper_action_equals_state=True)(data)

    state16 = umi._raw23_to_pose16(data["state"].astype(np.float64))
    for a in range(umi.N_ARMS):
        base = a * umi.ARM_DIM
        g = base + umi._GRIP
        # Gripper pinned to the (constant) observation gripper across the whole chunk.
        np.testing.assert_allclose(bug["actions"][:, g], np.float32(state16[g]), atol=1e-6)
        # Pose dims (pos + quat) are untouched vs the plain relativization.
        np.testing.assert_array_equal(bug["actions"][:, base : base + 7], plain["actions"][:, base : base + 7])
    # State feature is unchanged (this variant only rewrites action grippers).
    np.testing.assert_array_equal(bug["state"], plain["state"])


def test_mask_without_side_channel_fails_loudly():
    """Sanity: absolutizing from the masked state (no side channel) cannot recover
    the true poses -- the zeroed base orientation is an invalid quaternion, so it
    raises rather than silently emitting wrong poses. This proves ``absolute_state``
    is the load-bearing base and a missing side channel fails loud, not silent."""
    rng = np.random.default_rng(13)
    data = _make_data(rng, horizon=6)

    train = umi.UmiDualArmInputs(model_type=_model.ModelType.PI0, mask_absolute_state_pose=True)(data)
    # Fall back to the (masked, pose-zeroed) state as the absolutize base.
    with np.testing.assert_raises(ValueError):
        umi.UmiDualArmOutputs()({"state": train["state"], "actions": train["actions"]})



def _drop_actions(data: dict) -> dict:
    return {k: v for k, v in data.items() if k != "actions"}

