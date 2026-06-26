# Offline Reasoning — predicted vs. ground-truth trajectory

Visualize how the **fine-tuned pi0** reasons over an episode of the UMI dual-arm
dataset, open-loop, one action chunk at a time.

For each chunk along an episode the script:

1. feeds the observation (state + two wrist images + prompt) at the chunk start to the policy,
2. relativizes both the recorded `action` chunk and the model prediction into the
   chunk-start frame (the same SE(3) UMI transform used at training time),
3. plots, **per arm**, all 8 pose dimensions over the episode frame index:
   - **solid line** = ground truth (recorded `action` chunk, relativized),
   - **dashed line** = model prediction (re-expressed in the same relative frame).

This is the exact 8-dim per-arm target the model regresses:
`[rel_pos x/y/z, rel_quat w/x/y/z, gripper]`. Because every chunk re-anchors to its
own observation pose, each chunk's relative pose starts near identity
(`rel_pos ~ 0`, `rel_quat ~ [1,0,0,0]`) — the signature of the UMI representation.

## Output

**Two PNGs, one per arm**, each with 8 subplots:

- `<stem>_left.png`
- `<stem>_right.png`

Default stem → `offline_reasoning/episode_<ep>`.

## Run

```bash

export CUDA_VISIBLE_DEVICES=4,5,6,7
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

uv run offline_reasoning/offline_reasoning.py \
    --checkpoint-dir checkpoints/pi0_umi_dual_arm/run1/29999 \
    --episode 0
```

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint-dir` | *(required)* | Fine-tuned checkpoint step folder. Must contain `assets/<asset_id>/norm_stats.json`. |
| `--config-name` | `pi0_umi_dual_arm` | Train config the checkpoint was produced with. |
| `--episode` | `0` | Episode index to plot. |
| `--repo-id` | config's `repo_id` | Override the dataset path. |
| `--stride` | action horizon (50) | Frames between chunk starts. Lower → overlapping chunks. |
| `--out` | `offline_reasoning/episode_<ep>` | Output path stem; `_left.png` / `_right.png` are appended. |

## Notes

- The model is loaded with the **same norm stats** saved inside the checkpoint, so
  inputs are normalized exactly as during training.
- Ground truth and prediction are compared in the **same relative frame**: both are
  relativized against the chunk-start observation pose and canonicalized (`quat w >= 0`),
  so they are directly comparable per dimension.
- Inference needs the full pi0 model in memory; run on the box that holds the
  checkpoint (the 8 GB laptop GPU may OOM — use the remote A100/H100 or CPU).
