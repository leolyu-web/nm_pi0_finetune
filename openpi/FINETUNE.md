# Fine-tuning pi0 on a Dual-Arm UMI Dataset

End-to-end guide to full fine-tune pi0 base on your own LeRobot v2.1 dual-arm dataset.

> **Assumed dataset format:** LeRobot v2.1, 23-dim state/action (per arm: pos3 + quat_wxyz4 + grip1, plus 7 ego dims), with videos at `observation.images.wrist_image_1` and `observation.images.wrist_image_2`. If your layout differs, the transforms in `src/openpi/policies/umi_dual_arm_policy.py` need to be edited too.

---

## 1. Register your dataset

Open [src/openpi/training/config.py](src/openpi/training/config.py), find the `pi0_umi_dual_arm_v2` template block, and update the two highlighted fields:

```python
TrainConfig(
    name="pi0_umi_dual_arm_v2",                       # rename if you want (e.g. "pi0_dataset2")
    model=pi0_config.Pi0Config(),
    data=UmiDualArmDataConfig(
        repo_id="/absolute/path/to/dataset2",         # ← absolute path to dataset folder
        assets=AssetsConfig(asset_id="dataset2"),     # ← short unique label for norm stats
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi0_base/params"
    ),
    num_train_steps=30_000,
    num_workers=6,
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
uv run scripts/compute_norm_stats.py --config-name pi0_umi_dual_arm_v2
```

Output → `assets/pi0_umi_dual_arm_v2/<asset_id>/norm_stats.json`


---

## 3. Full fine-tune

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7         # which GPUs to use
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9   # let JAX claim 90% of GPU memory

uv run scripts/train.py pi0_umi_dual_arm_v2 \
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
- **Checkpoints** → `checkpoints/pi0_umi_dual_arm_v2/run1/` (saved every 1k steps)
- **W&B run** → `https://wandb.ai/<your-entity>/openpi`

> **First step is slow.** JAX JIT-compiles the train step (~2–5 min) before throughput stabilizes. That's normal, not a hang.
