# pi0 Dual-Arm UMI Fine-Tune

Full fine-tune of **π₀ / π₀.₅ base** on our own **raw LeRobot v2.1 dual-arm UMI datasets**
(giftbox is the shipped example), consumed as-is with no offline conversion.

Built on top of [openpi](https://github.com/Physical-Intelligence/openpi) by the
[Physical Intelligence team](https://www.physicalintelligence.company/). The pi0 model code is
upstream and untouched — see the official repo for model details, PyTorch support, LoRA, DROID/ALOHA
examples, and general documentation. **This README covers only what is custom to this fork.**

All customization lives in two places:
- **Data transforms** — `src/openpi/policies/umi_dual_arm_policy.py` (quaternion) and
  `umi_dual_arm_rot6d_policy.py` (6D rotation).
- **Train configs** — `src/openpi/training/config.py` (the `pi0_umi_dual_arm_*` / `pi05_umi_dual_arm_*`
  blocks).

> For the full internal contract (SE(3) relativization, rotation representations, norm-stat
> namespacing, the correctness tests), see the `pi0-umi-finetune` skill under `.claude/skills/`.

---

## Requirements

Full fine-tune needs a GPU with **> 70 GB** (A100 80 GB / H100), multi-GPU via `--fsdp-devices`.
Inference runs on **> 8 GB**. Tested on Ubuntu 22.04.

## Installation

This fork is part of the `nm_pi0_finetune` repo; the `openpi/` directory is the working tree.
We use [uv](https://docs.astral.sh/uv/) for Python deps:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

`GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot as a dependency.

---

## The Config Menu

All configs full-fine-tune on the raw dual-arm dataset with two wrist cameras only (head cam
dropped), action horizon 48, 30k steps. They differ along three axes: **base model**, **rotation
representation**, and **how much absolute state the model sees**.

| Config name | Base | Rotation | State seen | Notes |
|---|---|---|---|---|
| `pi0_umi_dual_arm_quat` | π₀ | quat (16-dim) | full pose | quaternion baseline |
| `pi0_umi_dual_arm_6Drot` | π₀ | 6D rot (20-dim) | full pose | continuous, UMI-faithful |
| `pi05_umi_dual_arm_quat` | π₀.₅ | quat (16-dim) | gripper only | `mask_absolute_state_pose=True` |
| `pi05_umi_dual_arm_quat_zaxis` | π₀.₅ | quat (16-dim) | z + gripper | `keep_z_position_in_state=True` |
| `pi05_umi_dual_arm_quat_allaxis` | π₀.₅ | quat (16-dim) | full pose | π₀.₅ counterpart of the quat baseline |
| `pi05_umi_dual_arm_quat_allaxis_teleo_gripbug` | π₀.₅ | quat (16-dim) | full pose | reproduces a teleop gripper-label bug |
| `pi05_umi_dual_arm_6Drot` | π₀.₅ | 6D rot (20-dim) | full pose | π₀.₅ 6D sibling |

- **Rotation**: 6D has no double cover / no `w=0` sign-flip discontinuity and is the UMI-faithful
  choice. It's free at the model — pi0 zero-pads actions to `action_dim=32` either way. The
  deployed runtime must decode the matching action dim (16 quat vs **20 rot6d, ROWS convention**).
- **Norm stats are namespaced by config `name`**, so quat / 6D siblings and the state-masking
  variants share a dataset but never collide.

---

## Fine-Tuning Guide

The codebase is **dataset-agnostic** — the pi0 model is generic. What makes a dataset work is a
matched pair of (1) a **data transform** mapping your raw fields into pi0's expected dict, and (2) a
**train config** wiring that transform to your dataset path. If your data has the same 23-dim UMI
structure you only edit a config (Steps 1–3); if it differs you also write a transform (Step 0.5).

### 0. The example dataset structure (dual-arm UMI)

LeRobot v2.1, `robot_type=dual_arm`, 30 fps, AV1-encoded video. Per frame:

| Field | dtype / shape | Meaning |
|---|---|---|
| `observation.state` | float32 `[23]` | absolute EE pose (layout below) |
| `action` | float32 `[23]` | absolute target EE pose (same layout) |
| `observation.images.wrist_image_1` | video `[320,240,3]` | left wrist cam |
| `observation.images.wrist_image_2` | video `[320,240,3]` | right wrist cam |
| `observation.images.image` | video `[240,320,3]` | head cam (**loaded but dropped** by the transform) |

The **23-dim** state/action vector:

```
[ 0:3 ]  left_x, left_y, left_z                          left arm position
[ 3:7 ]  left_quat_w, _x, _y, _z                         left arm orientation (quaternion, w-first)
[ 7   ]  left_gripper                                    left gripper width
[ 8:11]  right_x, right_y, right_z                       right arm position
[11:15]  right_quat_w, _x, _y, _z                        right arm orientation (quaternion, w-first)
[15   ]  right_gripper                                   right gripper width
[16:23]  ego_x, ego_y, ego_z, ego_quat_w, _x, _y, _z     ego pose (7 dims, DROPPED by the transform)
```

The transform slices 23→16 (two arms, ego dropped), keeps state **absolute**, and relativizes the
action chunk in SE(3) at load time (`T_rel = inv(T_obs) @ T_act` per arm). The gripper stays
absolute. This load-time relativization replaces pi0's built-in linear `DeltaActions`, which is
therefore intentionally **OFF**.

> **Dataset matches this exactly** (same 23-dim layout, two wrist cams `wrist_image_1`/`_2`)? →
> Steps 1–3, editing only the config.
>
> **Dataset is different** (dims, cameras, single arm, joint angles instead of EE poses)? → do
> **Step 0.5** first to add a transform, then Steps 1–3.

### 0.5. Onboarding a *different* dataset (write a transform)

Skip if your data already matches the 23-dim UMI layout. The transform is the only dataset-specific
code — everything downstream (normalization, tokenizing, padding to `action_dim=32`, the pi0 model)
is generic.

1. **Create the transform module.** Copy `src/openpi/policies/umi_dual_arm_policy.py` to
   `src/openpi/policies/<my_robot>_policy.py`. If your data needs **offline** preprocessing
   (re-encoding video, fixing fields/units) before it's a valid LeRobot dataset, put those scripts
   in a top-level `data_processing/` folder — keep offline prep separate from the load-time transform.

2. **Edit `Inputs.__call__`** to map your raw fields into pi0's dict:
   - `state`: your proprioception vector (any dim ≤ 32; pi0 zero-pads the rest).
   - `image`: fill `base_0_rgb` / `left_wrist_0_rgb` / `right_wrist_0_rgb`, and set `image_mask` —
     mask **off** any slot you don't have (the example zero-pads `base_0_rgb`, no head cam).
   - `actions`: your action chunk. Choose the action space here — absolute, linear delta (use pi0's
     built-in `DeltaActions` in the config), or SE(3)-relative (in the transform, as UMI does).
   - `prompt`: the task string (or rely on `prompt_from_task=True`).

3. **Edit `Outputs.__call__`** to invert whatever you did to the actions, returning the format your
   deployed runtime expects. Absolute Inputs → near no-op Outputs; relativizing Inputs → Outputs
   must absolutize (see the UMI example).

4. **Add a `DataConfig` factory** in `config.py` (copy `UmiDualArmDataConfig`). `repack_transforms`
   rename your raw LeRobot keys → the keys your `Inputs` reads; `data_transforms` point at your new
   `Inputs`/`Outputs`. Set `action_sequence_keys` to your action field name (the example uses
   `("action",)`, singular).

5. **Write tests** for the transform math (copy `umi_dual_arm_policy_test.py`). At minimum: the
   Inputs→Outputs round-trip closes, shapes/dtypes are right, any relativization inverts. CI runs
   `pytest` but never exercises your transform otherwise — untested transform code is the #1 way a
   silent data bug reaches training. Run:
   `uv run pytest src/openpi/policies/umi_dual_arm*_test.py -q`.

### 1. Register your dataset

In [src/openpi/training/config.py](src/openpi/training/config.py), copy the `pi0_umi_dual_arm_quat`
block (or `pi0_umi_dual_arm_6Drot` for 6D), and update the highlighted fields:

```python
TrainConfig(
    name="pi0_dataset2_quat",                             # ← unique name for your run
    model=pi0_config.Pi0Config(action_horizon=48),
    data=UmiDualArmDataConfig(
        repo_id="/absolute/path/to/dataset2",             # ← absolute path to dataset folder
        assets=AssetsConfig(asset_id="dataset2"),         # ← short unique label for norm stats
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi0_base/params"  # or pi05_base for pi0.5
    ),
    num_train_steps=30_000,
    num_workers=8,
),
```

| Field | What to set |
|---|---|
| `name` | The config name passed on the CLI. Keep it unique. |
| `repo_id` | Absolute path to the dataset root (folder containing `meta/`, `data/`, `videos/`). |
| `asset_id` | Short label namespacing norm stats. **Must be unique per dataset** or stats collide. |
| `num_train_steps` | Default 30k. Scale up for larger datasets. |
| `num_workers` | Data loader workers. Raise on machines with many CPU cores. |

Add multiple `TrainConfig(...)` blocks (one per dataset) — just give each a distinct `name`.
`prompt_from_task=True` uses the dataset's task string as the prompt; the giftbox family ships
Chinese task strings, and since pi0 base was pretrained on English an English prompt may fine-tune
better — operator's call.

### 2. Compute normalization statistics

Walks the dataset once, applies the UMI transforms, and writes per-dim stats. Pi0 expects
normalized inputs; **skipping this trains on the wrong scale.**

```bash
uv run scripts/compute_norm_stats.py --config-name <your-config-name>
```

Output → `assets/<your-config-name>/<asset_id>/norm_stats.json`. π₀ uses z-score; **π₀.₅ uses
quantile normalization automatically.** Any config that changes the state (masking) or action
targets (gripper bug) **must recompute norm stats** — old-dim stats are incompatible.

### 3. Full fine-tune

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7         # which GPUs to use
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9   # let JAX claim 90% of GPU memory

uv run scripts/train.py <your-config-name> \
    --exp-name=run1 \
    --fsdp-devices=4 \
    --overwrite
```

| Flag | Purpose |
|---|---|
| `--exp-name` | Subfolder under `checkpoints/<config_name>/`. |
| `--fsdp-devices=4` | Shard model + optimizer across GPUs (required for full fine-tune on multi-GPU). |
| `--overwrite` | Wipe the existing checkpoint dir for this `exp-name`. Use `--resume` to continue instead. |
| `--batch-size=64` | Optional — raise from default 32 if you have headroom. Must divide `fsdp-devices`. |
| `--no-wandb-enabled` | Optional — disable W&B logging. |

Outputs: checkpoints → `checkpoints/<config-name>/run1/` (every 1k steps); W&B run →
`https://wandb.ai/<your-entity>/openpi`.

> **First step is slow.** JAX JIT-compiles the train step (~2–5 min) before throughput stabilizes.
> That's normal, not a hang.

### 4. Serve for inference

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=<your-config-name> \
    --policy.dir=checkpoints/<config-name>/run1/30000
```

The server's `Outputs` transform inverts the SE(3) relativization and returns **absolute** per-arm
poses; the deployed runtime consumes those and does any final robot-native conversion. Match the
action dim to the config (16 quat vs 20 rot6d, ROWS convention).

---

## Troubleshooting

| Issue | Resolution |
|---|---|
| Norm stats "not found" / wrong scale | Stats load from `assets/<name>/<asset_id>/`. Confirm the config `name` matches between `compute_norm_stats.py` and `train.py`. |
| `tyro.cli` crashes on config load | Usually a nested-tuple default — the `world_reframe_quat_wxyz` field must stay a FLAT `tuple[float, ...]`. |
| OOM on full fine-tune | The local 8 GB GPU can't do it; run remote with `--fsdp-devices`. Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`. |
| Diverging training loss | Check `q01`/`q99`/`std` in `norm_stats.json`; rarely-used dims can get tiny values → huge normalized states/actions. |

For anything general (installation conflicts, PyTorch, DROID/ALOHA/LIBERO, remote inference), see
the [official openpi repo](https://github.com/Physical-Intelligence/openpi).
