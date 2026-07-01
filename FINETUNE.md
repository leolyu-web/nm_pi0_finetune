# Fine-tuning pi0 on a Custom Dataset

End-to-end guide to full fine-tune pi0 / pi0.5 base on your own LeRobot v2.1 dataset.

This codebase is **dataset-agnostic** — the pi0 model is generic. What makes a dataset work is a
matched pair of (1) a **data transform** that maps your raw fields into pi0's expected dict, and
(2) a **train config** that wires that transform to your dataset path. The example shipped here is
a **dual-arm UMI** dataset; if your data has the same structure you only edit a config, and if it
differs you also write a transform. Both paths are below.

---

## 0. The example dataset structure (dual-arm UMI)

The provided dataset is **LeRobot v2.1**, `robot_type=dual_arm`, 30 fps, AV1-encoded video. Per
frame:

| Field | dtype / shape | Meaning |
|---|---|---|
| `observation.state` | float32 `[23]` | absolute EE pose (layout below) |
| `action` | float32 `[23]` | absolute target EE pose (same layout) |
| `observation.images.wrist_image_1` | video `[320,240,3]` | left wrist cam |
| `observation.images.wrist_image_2` | video `[320,240,3]` | right wrist cam |
| `observation.images.image` | video `[240,320,3]` | head cam (**loaded but dropped** by the transform) |

The **23-dim** state/action vector is:

```
[ 0:3 ]  left_x, left_y, left_z                     left arm position
[ 3:7 ]  left_quat_w, _x, _y, _z                    left arm orientation (quaternion, w-first)
[ 7   ]  left_gripper                               left gripper width
[ 8:11]  right_x, right_y, right_z                  right arm position
[11:15]  right_quat_w, _x, _y, _z                   right arm orientation (quaternion, w-first)
[15   ]  right_gripper                              right gripper width
[16:23]  ego_x, ego_y, ego_z, ego_quat_w, _x, _y, _z    ego pose (7 dims, DROPPED by the transform)
```

The transform (`src/openpi/policies/umi_dual_arm_policy.py`) slices this 23→16 (two arms, ego
dropped), keeps state **absolute**, and relativizes the action chunk in SE(3) at load time
(`T_rel = inv(T_obs) @ T_act` per arm). The gripper stays absolute. See the `pi0-umi-finetune`
skill for the full contract.

> **Does your dataset match this exactly** (same 23-dim layout, two wrist cams named
> `wrist_image_1`/`_2`)? → follow Steps 1-3 below, editing only the config.
>
> **Is your dataset different** (different dims, different cameras, single arm, joint angles
> instead of EE poses, etc.)? → do **Step 0.5** first to add a transform, then Steps 1-3.

---

## 0.5. Onboarding a *different* dataset (write a transform)

Skip this section if your data already matches the 23-dim UMI layout above.

The transform is the only dataset-specific code. Everything downstream (normalization, tokenizing,
padding to `action_dim=32`, the pi0 model) is generic. To add a new dataset shape:

1. **Create the processing / transform module.** Copy `src/openpi/policies/umi_dual_arm_policy.py`
   to a new file named for your embodiment, e.g. `src/openpi/policies/<my_robot>_policy.py`. If
   your dataset also needs **offline** preprocessing (re-encoding video, recomputing fields, fixing
   units) before it's even a valid LeRobot dataset, put those scripts in a top-level
   `data_processing/` folder — keep offline data prep separate from the load-time transform.

2. **Edit `Inputs.__call__`** to map *your* raw fields into pi0's expected dict:
   - `state`: your proprioception vector (any dim ≤ 32; pi0 zero-pads the rest).
   - `image`: fill `base_0_rgb` / `left_wrist_0_rgb` / `right_wrist_0_rgb` from your cameras, and
     set `image_mask` — mask **off** any slot you don't have (the example zero-pads `base_0_rgb`
     since it has no head cam).
   - `actions`: your action chunk. Choose your action space here — absolute, linear delta (use
     pi0's built-in `DeltaActions` in the config), or SE(3)-relative (done in the transform, as
     UMI does). Whatever you encode, the model regresses it directly.
   - `prompt`: the task string (or rely on `prompt_from_task=True`).

3. **Edit `Outputs.__call__`** to invert whatever you did to the actions, returning the action
   format your deployed runtime expects. If Inputs is absolute, Outputs is a near no-op; if Inputs
   relativizes, Outputs must absolutize (see the UMI example).

4. **Add a `DataConfig` factory** in `src/openpi/training/config.py` (copy `UmiDualArmDataConfig`).
   The `repack_transforms` rename your raw LeRobot keys → the keys your `Inputs` reads; the
   `data_transforms` point at your new `Inputs`/`Outputs`. Set `action_sequence_keys` to whatever
   your dataset calls the action field (the example uses `("action",)`, singular).

5. **Write tests** for the transform math (copy `umi_dual_arm_policy_test.py`). At minimum: the
   Inputs→Outputs round-trip closes, shapes/dtypes are right, and any relativization inverts. CI
   runs `pytest` but never exercises your transform otherwise — untested transform code is the #1
   way a silent data bug reaches training.

Then continue with Step 1 (register a `TrainConfig` that uses your new `DataConfig`).

---

## 1. Register your dataset

Open [src/openpi/training/config.py](src/openpi/training/config.py), find the `pi0_umi_dual_arm_quat` block (the quaternion baseline; use `pi0_umi_dual_arm_6Drot` if you prefer 6D rotation), copy it, and update the highlighted fields:

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

**Rules of thumb**

| Field | What to set |
|---|---|
| `name` | The config name passed on the CLI. Keep it unique. |
| `repo_id` | Absolute path to the dataset root (the folder containing `meta/`, `data/`, `videos/`). |
| `asset_id` | Short label used to namespace norm stats. **Must be unique per dataset** or stats will collide. |
| `num_train_steps` | Default 30k. Scale up for larger datasets. |
| `num_workers` | Data loader workers. Raise on machines with many CPU cores. |

You can add multiple `TrainConfig(...)` blocks (one per dataset) — just give each a distinct `name`.

---

## 2. Compute normalization statistics

This walks the entire dataset once, applies the UMI transforms, and writes per-dim mean/std to disk. Pi0 expects normalized inputs; **skipping this step will train on the wrong scale.**

```bash
uv run scripts/compute_norm_stats.py --config-name <your-config-name>
```

Output → `assets/<your-config-name>/<asset_id>/norm_stats.json`

Norm stats are namespaced by config `name`, so quaternion and 6D-rotation siblings on the same dataset write to distinct directories and never collide.


---

## 3. Full fine-tune

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7         # which GPUs to use
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9   # let JAX claim 90% of GPU memory

uv run scripts/train.py <your-config-name> \
    --exp-name=run1 \
    --fsdp-devices=4 \
    --overwrite
```

**Flag reference**

| Flag | Purpose |
|---|---|
| `--exp-name` | Subfolder name under `checkpoints/<config_name>/`. |
| `--fsdp-devices=4` | Shard model + optimizer across all 4 GPUs (required for full fine-tune on multi-GPU). |
| `--overwrite` | Wipe any existing checkpoint dir for this `exp-name`. Drop and use `--resume` to continue an interrupted run. |
| `--batch-size=64` | Optional — raise from default 32 if you have memory headroom. Must be divisible by `fsdp-devices`. |
| `--no-wandb-enabled` | Optional — disable W&B logging. |

Outputs:
- **Checkpoints** → `checkpoints/<your-config-name>/run1/` (saved every 1k steps)
- **W&B run** → `https://wandb.ai/<your-entity>/openpi`

> **First step is slow.** JAX JIT-compiles the train step (~2–5 min) before throughput stabilizes. That's normal, not a hang.
