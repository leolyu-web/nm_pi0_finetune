# Dual-arm → pi0 (UMI-style relative 6D actions)

Pipeline to clean a raw dual-arm LeRobot v2.1 dataset and full-fine-tune the
**pi0 base** model on it, using UMI's relative-trajectory representation with
rotation-6D.

Task-agnostic: the `earphone_0620_316episodes` dataset is just one example of the
expected input structure. Any dataset with the same schema works (more episodes,
more chunks, and a different task/prompt are all fine — the prompt is read
per-episode from the dataset, not hardcoded).

## What the pipeline does

- **Drops** the head/ego motor (`ego_*`, 7 dims) and the head camera
  `observation.images.image`. Keeps **only the two wrist cameras**.
- **Per arm**, converts `[x,y,z] + quat(wxyz)` → 4×4 SE(3) transform →
  `[pos(3), rot6d(6), gripper(1)]` = 10 dims. Two arms → **20-dim** state & action.
- Stores **absolute** poses on disk. The **UMI relative trajectory**
  (`T_rel = inv(T_obs) @ T_action`, per arm, in SE(3); gripper stays absolute) is
  applied at **load time** by `UmiDualArmInputs`, because relative-to-current-obs
  cannot be baked per-frame under openpi's overlapping action chunks.
  → pi0's built-in linear `DeltaActions` stays **OFF**.
- Downsamples 30 fps → 10 fps, re-normalizes quaternions, and drops frames flagged
  invalid by the SLAM / width validity masks.
- No bimanual / inter-gripper relative pose is computed (per your instruction).

## Files

| File | Role |
|------|------|
| `examples/umi_dual_arm/pose_util.py` | UMI-faithful SE(3) ↔ 6D pose math (numpy/scipy). |
| `examples/umi_dual_arm/convert_data_to_lerobot.py` | Builds the cleaned LeRobot v2.1 dataset. |
| `src/openpi/policies/umi_dual_arm_policy.py` | `UmiDualArmInputs`/`UmiDualArmOutputs` + load-time UMI relativization. |
| `src/openpi/training/config.py` | `UmiDualArmDataConfig` factory + `pi0_umi_dual_arm` TrainConfig. |

## Run

```bash
# 0. Sync the env once (installs lerobot, opencv, etc.)
uv sync

# 1. Convert + clean (writes to $HF_LEROBOT_HOME/umi_dual_arm_6d)
uv run examples/umi_dual_arm/convert_data_to_lerobot.py \
    --raw_root /home/it002338/Junlin_lv/pi0/earphone_0620_316episodes \
    --repo_id umi_dual_arm_6d
# quick smoke test first:  add  --max_episodes 2

# 2. Compute normalization stats
uv run scripts/compute_norm_stats.py --config-name pi0_umi_dual_arm

# 3. Full fine-tune pi0 base
uv run scripts/train.py pi0_umi_dual_arm --exp-name umi_run_1 --overwrite
```

## Layout of the cleaned 20-dim vectors

```
[ 0: 3]  left  pos (x,y,z)
[ 3: 9]  left  rot6d
[ 9]     left  gripper width (absolute)
[10:13]  right pos
[13:19]  right rot6d
[19]     right gripper width (absolute)
```

State = absolute current EE pose. Action = absolute target on disk, converted to
UMI-relative (per arm) on the fly during training/inference.

## Notes

- `action(t) == state(t+1)` in the raw data (verified): actions are next-frame
  absolute EE poses.
- The two wrist views fill pi0's `left_wrist_0_rgb` / `right_wrist_0_rgb` slots;
  the `base_0_rgb` slot is zero-padded and masked (no third-person camera).
- If you later want a true single-obs UMI state (gripper-only proprio), change
  `UmiDualArmInputs` to zero the pose dims of `state`; the action relativization is
  unaffected.
