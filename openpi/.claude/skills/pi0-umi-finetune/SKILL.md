---
name: pi0-umi-finetune
description: How the dual-arm UMI pi0/pi0.5 fine-tune is wired in this openpi fork — the custom transforms, the config menu, the SE(3) relativization contract, and the environment gotchas. Use when adding a dataset/config, changing the action representation, debugging norm stats or ffmpeg, or deploying the runtime.
---

# Dual-Arm UMI pi0 Fine-Tune

## Goal

Full fine-tune pi0 (and now pi0.5) base on our own **raw LeRobot v2.1 dual-arm UMI datasets**
(giftbox is the currently-shipped example) — consumed as-is, no offline conversion. All
customization lives in two
places: the UMI **transforms** in `src/openpi/policies/` and the **configs** in
`src/openpi/training/config.py`. The pi0 model code is upstream and untouched.

## The Data Pipeline

Assembled in `data_loader.py` (~lines 229-232), applied in this order:

```
repack → data_transforms (UMI) → Normalize → model_transforms (tokenize + PadStatesAndActions(32))
```

- **repack** (`config.py`, in each `UmiDualArm*DataConfig.create`) renames raw LeRobot keys:
  `observation.images.wrist_image_1/2` → left/right wrist, `observation.state`, `action`,
  `prompt`. The head cam (`observation.images.image`) is deliberately **dropped** here.
- **UMI transform** (`umi_dual_arm_policy.py` / `umi_dual_arm_rot6d_policy.py`) does the real
  work — see contract below.
- **Normalize** uses stats computed on the **relativized** state/actions (`compute_norm_stats.py`
  runs the same repack + data_transforms stack). z-score for PI0, **quantile for PI0.5**
  (`use_quantile_norm` auto-set for any non-PI0 model, `config.py:189`).
- **model_transforms** tokenizes the prompt and pads state/actions to `action_dim=32`. This is
  why 6D (20-dim) is **free** vs quaternion (16-dim) — pi0 zero-pads either way.

## The Core Contract: SE(3) UMI Relativization

The raw on-disk vector is **23-dim**: per arm `[pos3, quat_wxyz4, grip1]` + 7 ego dims (dropped).
Ground-truth layout from the example dataset's `meta/info.json` (`dual_arm`, 30 fps, AV1 video):

```
[0:3] left pos | [3:7] left quat_wxyz | [7] left grip
[8:11] right pos | [11:15] right quat_wxyz | [15] right grip
[16:23] ego pose (7 dims, DROPPED)
cameras: observation.images.wrist_image_1 (left), wrist_image_2 (right),
         image (head cam — loaded by LeRobot but dropped in repack)
```

- **State** is computed **ABSOLUTE** (per arm → 16-dim quat, or 20-dim after quat→matrix→rot6d),
  but the config option `mask_absolute_state_pose` can hide it from the model — see the Config Menu.
  The absolute pose always still drives action relativization + inference-time absolutization.
- **Actions** are **relativized in SE(3) at load time**, per arm:
  `T_rel = inv(T_obs) @ T_act`. Gripper width stays absolute.
- This load-time relativization is our equivalent of pi0's linear `DeltaActions`, so pi0's
  built-in `DeltaActions` MUST stay **OFF** (we never push it in the config).
- `action_sequence_keys=("action",)` — the raw dataset stores the chunk under singular `action`.
- **Outputs** (`UmiDualArm*Outputs`) invert the relativization at inference: `base @ rel` per
  arm → absolute poses. The deployed runtime consumes these absolute poses and is responsible for
  any final robot-native conversion.

Round-trip closes to ~1e-7 through the full Normalize → pad32 → Unnormalize → absolutize path
(float32 norm-stats precision). Verify this numerically after any transform change.

## Two Rotation Representations

| Module | Rep | Per-arm | STATE_DIM | Notes |
|---|---|---|---|---|
| `umi_dual_arm_policy.py` | quaternion (wxyz) | `[pos3, quat4, grip1]` = 8 | 16 | rel quat canonicalized to `w≥0` (kills double-cover); decode re-normalizes |
| `umi_dual_arm_rot6d_policy.py` | 6D rotation | `[pos3, rot6d6, grip1]` = 10 | 20 | 6D = first two **ROWS** of the matrix; Gram-Schmidt decode re-orthonormalizes |

**6D is the UMI-faithful representation** (Zhou et al. 2019, "On the Continuity of Rotation
Representations in Neural Networks", arXiv:1812.07035). It is continuous everywhere on SO(3) —
no double cover,
no `w=0`/180° sign flip. Quaternion works for our current data (the sign flip needs canonicalizing
1.5% of the time on the right arm, but relative-within-chunk rotations stay < 113°, far from the
180° discontinuity), and is free to keep. Prefer 6D for harsher tasks or longer horizons.

> Convention warning: in the rot6d module, 6D is the first two **ROWS** (not columns). The
> deployment runtime must decode the 20-dim output with the same rows convention.

## The Config Menu

All full-fine-tune the raw dual-arm dataset (two wrist cams, head cam dropped), horizon 48, 30k
steps. They differ by base model, rotation rep, and how much absolute state the model sees.

| Config name | Base | Rep | State seen | Dataset |
|---|---|---|---|---|
| `pi0_umi_dual_arm_quat` | pi0 | quat 16 | full pose | giftbox_0621_1758 |
| `pi0_umi_dual_arm_6Drot` | pi0 | 6D rot 20 | full pose | giftbox_0621_1758 |
| `pi05_umi_dual_arm_quat` | **pi0.5** | quat 16 | full pose | giftbox_0628_1912_qc |
| `pi05_umi_dual_arm_quat_gripper` | **pi0.5** | quat 16 | gripper only | giftbox_0628_1912_qc (gripper-only-state sibling of `pi05_umi_dual_arm_quat`) |
| `pi05_umi_dual_arm_6Drot` | **pi0.5** | 6D rot 20 | full pose | giftbox_0628_1912_qc |
| `pi05_umi_dual_arm_quat_multi` | **pi0.5** | quat 16 | full pose | giftbox_0621_1758 **+** _0628_1912_qc (concat) |

- **State masking** is controlled by `mask_absolute_state_pose` (gripper-only) + `keep_z_position_in_state`
  (adds absolute z), supported on **both** `UmiDualArmDataConfig` and `UmiDualArmRot6dDataConfig`.
  Most shipped configs run with `mask_absolute_state_pose=False` (the **full 16-dim absolute pose**
  goes into the model's state); `pi05_umi_dual_arm_quat_gripper` is the one that masks it on
  (gripper-only state). `gripper_action_equals_state` (quat only) is also available to pin each
  action's gripper to the observation gripper. **Any masking/target change needs a norm-stats
  recompute** — the state (or action) distribution moves.
- **assets_dirs** is namespaced by config `name` (`config.py:658`); `checkpoint_dir` =
  `checkpoint_base_dir/name/exp_name`. So sibling configs that share an `asset_id` (all `_0628_qc`
  pi0.5 configs; the `_quat`/`_6Drot` pi0 pair) write/load norm stats to distinct dirs — no collision.
- **Option-2 world reframe** (available but no shipped config sets it): optional
  `world_reframe_quat_wxyz` on `UmiDualArmDataConfig` (flat 8 floats, per-arm wxyz) left-applies
  a constant per-arm `W = inv(mean orientation)` so the absolute state cluster sits near `w=1`,
  away from the discontinuity. Relative targets are provably invariant; Outputs apply `inv(W)`
  so the runtime sees the original frame. If enabled, uses its own `asset_id` and requires
  recomputing norm stats. Kept because 6D makes it unnecessary but the code path is tested.
  - tyro gotcha: this field MUST be typed `tyro.conf.Suppress[tuple[float, ...] | None]` (FLAT,
    reshaped to `(2,4)` in the transform). Nested-tuple defaults crash `tyro.cli` and block ALL
    configs.

## Tests (the correctness net)

The custom transform math is covered by `umi_dual_arm_policy_test.py` and
`umi_dual_arm_rot6d_policy_test.py` (co-located in `src/openpi/policies/`). Deterministic,
CPU-only, no checkpoints. They lock down: the SE(3) relativize↔absolutize round-trip,
`action[0]=state → identity`, per-arm independence, gripper-stays-absolute, quaternion `w≥0`
canonicalization / rot6d rows convention, tolerance of non-unit / non-orthonormal predicted
rotations at decode, the 23→16/20 ego-drop slice, the full `Inputs→Outputs` round-trip, and the
v3 world-reframe invariants. Run: `uv run pytest src/openpi/policies/umi_dual_arm*_test.py -q`.

**Run these after any change to the transforms** — the model-level tests never exercise this code,
so CI would otherwise stay green through a broken relativization. Tests access module-private
helpers by design; ruff `SLF001` is ignored for `*_test.py` via `per-file-ignores` in
`pyproject.toml`.

## Common Workflows

### Add a new dataset/config
1. Copy a `TrainConfig` block; set unique `name`, absolute `repo_id`, unique `asset_id`.
2. `prompt_from_task=True` uses the dataset's task string as the prompt. Some datasets ship
   Chinese task strings (the giftbox family does); pi0 base was pretrained on English, so an
   English prompt may fine-tune better — operator's call.
3. `uv run scripts/compute_norm_stats.py --config-name <name>` (on the box with the dataset).
4. `uv run scripts/train.py <name> --exp-name=<run> --fsdp-devices=4 --overwrite` (remote GPU).

See the [README](../../../README.md) Fine-Tuning Guide for the operator-facing quick start.

### Train on multiple datasets jointly
Pass `repo_ids=(...)` (a tuple) to any `UmiDualArm*DataConfig` instead of a single `repo_id`. The
loader (`data_loader.py` `create_torch_dataset`) builds one LeRobotDataset per repo — each with its
OWN fps and its own `PromptFromLeRobotTask` — and wraps them in a torch `ConcatDataset`. Shuffling
(`shuffle=True` in training) then interleaves samples across all datasets over one flat index space,
sampled **proportional to each dataset's size** (bigger dataset appears more often; for equal
exposure you'd need a `WeightedRandomSampler`, not yet wired). All repos MUST share the raw 23-dim
UMI format (same state/action dims + camera keys) — one transform stack and one norm-stats vector
cover the union, so `compute_norm_stats` builds the same concat and the stats are combined
automatically. `pi05_umi_dual_arm_quat_multi` is the shipped example. Gotchas:
- Set an explicit `assets.asset_id` (a list has no single id to fall back on).
- Also set a plain `repo_id` (= `repo_ids[0]`) in the config: tyro marks the factory `repo_id`
  (default `MISSING`) required, so without it the CLI demands `--data.repo-id`. `create_base_config`
  keeps the given `repo_id` as the primary/asset_id-fallback and trains on the full `repo_ids` list.
- Recompute norm stats over the union: `uv run scripts/compute_norm_stats.py --config-name <name>`.

### Change the action representation
Recompute norm stats (old-dim stats are incompatible) AND update the deployed runtime to consume
the new action dim (16 quat vs 20 rot6d) and convention.

### Onboard a *different* dataset (not the 23-dim UMI layout)
The transform is the only dataset-specific code; everything downstream (Normalize, tokenize,
pad-to-32, the pi0 model) is generic. Steps: copy `umi_dual_arm_policy.py` → a new
`<robot>_policy.py`; edit `Inputs` to map your raw fields into pi0's dict (`state` any dim ≤ 32,
`image`/`image_mask` per camera, `actions` in your chosen action space, `prompt`); edit `Outputs`
to invert it for the runtime; add a `DataConfig` factory (copy `UmiDualArmDataConfig`) with your
`repack` key renames and `action_sequence_keys`; write transform tests. If the raw data needs
offline fixing (video re-encode, unit/field fixes) before it's a valid LeRobot dataset, put those
scripts in a top-level `data_processing/` folder — keep offline prep separate from the load-time
transform. Full walkthrough: [README](../../../README.md) Fine-Tuning Guide, Step 0.5.

## Common Issues

- **ffmpeg / AV1 decode**: the UMI videos are AV1; torchcodec needs ffmpeg 7 to decode them. It is
  installed system-wide on our machines, so this normally just works. If a box lacks it, decode
  fails — install ffmpeg 7 there. `compute_norm_stats` defaults `skip_videos=True` so the stats
  pass avoids decode; training still needs it.
- **Norm stats "not found" / trained on wrong scale**: stats load from `assets/<name>/<asset_id>/`.
  `compute_norm_stats.py:133` now writes to `assets_dirs/asset_id` (matches the load side,
  `config.py:196`) — this was previously mis-joined with the absolute `repo_id`; no manual move
  needed anymore. Confirm the config `name` matches between the two commands.
- **`tyro.cli` crashes on config load**: usually a nested-tuple default (see the reframe gotcha).
- **OOM on full fine-tune**: the local 8 GB GPU can't do it; run remote with `--fsdp-devices`.

## Known Deviations from Pure UMI (future work, not yet done)

All configs deviate from canonical UMI in two ways, both offered but not implemented:
1. **Proprioception is ABSOLUTE** here; UMI relativizes past poses to the current frame. This
   absolute-state choice is the root cause of the quaternion-flip concern — v3's reframe patches a
   non-UMI choice. Relativizing proprioception is the genuinely UMI-faithful fix (makes the reframe
   unnecessary and lets quaternion stay safe).
2. **No inter-arm relative pose**; UMI bimanual feeds the relative pose between grippers for
   coordination. Arms are currently treated independently.

Also: UMI uses Diffusion Policy; here it's pi0 (flow-matching VLA).
