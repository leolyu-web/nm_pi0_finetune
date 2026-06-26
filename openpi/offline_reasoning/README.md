# Offline Reasoning — predicted vs. ground-truth trajectory

Visualize how the **fine-tuned pi0** reasons over an episode of the UMI dual-arm
dataset, open-loop, one action chunk at a time.

For each non-overlapping chunk along an episode the script:

1. feeds the observation (state + two wrist images + prompt) at the chunk start to the policy,
2. lets `UmiDualArmOutputs` reconstruct the model's **relative** prediction into absolute EE poses,
3. plots, per arm, the absolute 3D end-effector path:
   - **solid line** = ground truth (recorded `action` chunk),
   - **dotted line** = model prediction,
   - **dot marker** = each chunk's start point.

Because every chunk is anchored to its own observation pose, each segment begins
at a different absolute point — the visual signature of the UMI relative
representation.

## Run

```bash

export CUDA_VISIBLE_DEVICES=4,5,6,7
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

uv run offline_reasoning/offline_reasoning.py \
    --checkpoint-dir checkpoints/pi0_umi_dual_arm/run1/29999 \
    --episode 0
```

Output → `offline_reasoning/episode_0.png`.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint-dir` | *(required)* | Fine-tuned checkpoint step folder. Must contain `assets/<asset_id>/norm_stats.json`. |
| `--config-name` | `pi0_umi_dual_arm` | Train config the checkpoint was produced with. |
| `--episode` | `0` | Episode index to plot. |
| `--repo-id` | config's `repo_id` | Override the dataset path. |
| `--stride` | action horizon (50) | Frames between chunk starts. Lower → overlapping chunks. |
| `--out` | `offline_reasoning/episode_<ep>.png` | Output image path. |

## Notes

- The model is loaded with the **same norm stats** saved inside the checkpoint, so
  inputs are normalized exactly as during training.
- Ground truth and prediction are compared in the **same absolute frame**: GT is the
  raw recorded action (23→20 dim), prediction is `T_obs ⊗ relative_pred`.
- Inference needs the full pi0 model in memory; run on the box that holds the
  checkpoint (the 8 GB laptop GPU may OOM — use the remote A100/H100 or CPU).
