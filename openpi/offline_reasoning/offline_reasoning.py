"""Offline reasoning / open-loop comparison for the fine-tuned pi0 (per-dim relative).

Given one episode of the LeRobot dataset and a fine-tuned pi0 checkpoint, this
script runs the policy open-loop in chunks along the episode and plots, **per
arm**, the model's prediction against ground truth in the *relative* (UMI)
action representation -- i.e. exactly the 8-dim per-arm vector the model is
trained to regress: ``[rel_pos3, rel_quat_wxyz4, grip1]``.

Output: TWO PNGs, one per arm (``..._left.png`` / ``..._right.png``). Each PNG
has 8 subplots (one per pose dimension). In every subplot the x-axis is the
episode frame index and two lines are drawn per chunk:

    solid  = ground truth   (recorded ``action`` chunk, relativized)
    dashed = model predicted (policy output, re-expressed in the same frame)

Both series are anchored to the chunk-start observation pose and canonicalized
(quaternion ``w >= 0``), so they live in an identical relative frame. Because
each chunk re-anchors, every chunk's relative pose starts near identity
(rel_pos ~ 0, rel_quat ~ [1,0,0,0]) -- the signature of the UMI representation.

Example
-------
    uv run offline_reasoning/offline_reasoning.py \
        --checkpoint-dir checkpoints/pi0_umi_dual_arm/run1/29999 \
        --episode 0
"""

import dataclasses
import logging
import pathlib

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import matplotlib

matplotlib.use("Agg")  # headless: write PNG, no display needed
import matplotlib.pyplot as plt
import numpy as np
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.policies import umi_dual_arm_policy as _umi
from openpi.training import config as _config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Per-arm 8-dim layout -- matches the fine-tuning target produced by
# ``_relativize_actions``: [rel_pos3, rel_quat_wxyz4, grip1] (gripper is absolute).
DIM_LABELS = [
    "rel x [m]",
    "rel y [m]",
    "rel z [m]",
    "rel quat w",
    "rel quat x",
    "rel quat y",
    "rel quat z",
    "gripper (abs)",
]
ARM_NAMES = ["left", "right"]


@dataclasses.dataclass
class Args:
    # Path to the fine-tuned checkpoint (the step folder, e.g. ``.../run1/29999``).
    checkpoint_dir: str
    # Train config name the checkpoint was produced with.
    config_name: str = "pi0_umi_dual_arm"
    # Episode index to visualize.
    episode: int = 0
    # Override the dataset path. Defaults to the ``repo_id`` baked into the config.
    repo_id: str | None = None
    # Chunk stride (frames). Defaults to the model action horizon (non-overlapping).
    stride: int | None = None
    # Output path stem. Per-arm files are written as ``<stem>_<arm>.png``.
    # Defaults to ``offline_reasoning/episode_<ep>``.
    out: str | None = None


@dataclasses.dataclass
class Chunk:
    """One open-loop chunk: relative GT vs. prediction, plus its frame x-axis."""

    start: int  # episode-relative frame index of the chunk start
    frames: np.ndarray  # (n,) absolute frame indices for the x-axis
    gt_rel: np.ndarray  # (n, 16) ground-truth relative pose (both arms)
    pred_rel: np.ndarray  # (n, 16) predicted relative pose (both arms)


def _arm_slice(arm: int) -> slice:
    base = arm * _umi.ARM_DIM
    return slice(base, base + _umi.ARM_DIM)


def _collect_chunks(args: Args, train_config) -> tuple[list[Chunk], int, int]:
    """Run the policy open-loop and return per-chunk relative GT/pred arrays."""
    repo_id = args.repo_id or train_config.data.repo_id
    action_horizon = train_config.model.action_horizon
    stride = args.stride or action_horizon

    logger.info("Loading fine-tuned policy from %s", args.checkpoint_dir)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint_dir)

    logger.info("Loading dataset %s", repo_id)
    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        # Stack ``action_horizon`` future action frames into the ``action`` key so
        # we have the ground-truth chunk for every frame (mirrors training).
        delta_timestamps={"action": [t / meta.fps for t in range(action_horizon)]},
    )

    n_eps = int(dataset.episode_data_index["from"].shape[0])
    if not 0 <= args.episode < n_eps:
        raise ValueError(f"episode {args.episode} out of range [0, {n_eps})")
    ep_from = int(dataset.episode_data_index["from"][args.episode])
    ep_to = int(dataset.episode_data_index["to"][args.episode])  # exclusive
    logger.info("Episode %d: frames [%d, %d) (%d frames)", args.episode, ep_from, ep_to, ep_to - ep_from)

    chunks: list[Chunk] = []
    for start in range(ep_from, ep_to, stride):
        frame = dataset[start]

        # Raw current observation pose (23,) -> 16-dim absolute, used as the
        # relativization base for BOTH ground truth and prediction.
        state16 = _umi._raw23_to_pose16(np.asarray(frame["observation.state"], dtype=np.float64))

        obs = {
            "state": np.asarray(frame["observation.state"], dtype=np.float32),  # raw (23,)
            "left_wrist_image": np.asarray(frame["observation.images.wrist_image_1"]),
            "right_wrist_image": np.asarray(frame["observation.images.wrist_image_2"]),
            "prompt": str(frame["task"]),
        }

        # Validity mask: False = padded step past the episode end.
        valid = ~np.asarray(frame["action_is_pad"], dtype=bool)  # (H,)
        if int(valid.sum()) < 2:
            continue

        # Ground truth: raw recorded action chunk -> 16-dim absolute -> relative
        # in the chunk-start frame (same SE(3) transform used at training time).
        gt_abs16 = _umi._raw23_to_pose16(np.asarray(frame["action"], dtype=np.float64))  # (H,16)
        gt_rel = _umi._relativize_actions(state16, gt_abs16)  # (H,16)

        # Prediction: policy returns ABSOLUTE 16-dim poses (UmiDualArmOutputs already
        # de-relativized them). Re-relativize against the SAME base so GT and pred
        # are compared in an identical, canonicalized relative frame.
        pred_abs16 = np.asarray(policy.infer(obs)["actions"], dtype=np.float64)  # (H,16)
        pred_rel = _umi._relativize_actions(state16, pred_abs16)  # (H,16)

        gt_rel, pred_rel = gt_rel[valid], pred_rel[valid]
        frames = np.nonzero(valid)[0] + start  # absolute frame index per valid step

        chunks.append(Chunk(start=start, frames=frames, gt_rel=gt_rel, pred_rel=pred_rel))

        # Per-arm mean position error (in the relative frame == absolute displacement error).
        errs = [
            np.linalg.norm(pred_rel[:, _arm_slice(a)][:, :3] - gt_rel[:, _arm_slice(a)][:, :3], axis=-1).mean()
            for a in range(_umi.N_ARMS)
        ]
        logger.info(
            "  chunk @frame %d: %d steps, mean rel-pos err L=%.4f m R=%.4f m",
            start, len(frames), errs[0], errs[1],
        )

    if not chunks:
        raise RuntimeError("No valid chunks found for this episode.")
    return chunks, ep_from, ep_to


def _plot_arm(arm: int, chunks: list[Chunk], args: Args, stride: int, out_path: pathlib.Path) -> None:
    """Render the 8-dim GT-vs-pred figure for a single arm and save it."""
    sl = _arm_slice(arm)
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    axes = axes.ravel()

    for d in range(_umi.ARM_DIM):
        ax = axes[d]
        for ci, ch in enumerate(chunks):
            gt = ch.gt_rel[:, sl][:, d]
            pred = ch.pred_rel[:, sl][:, d]
            # Label only the first chunk so the legend stays clean.
            ax.plot(ch.frames, gt, color="C0", linestyle="-", linewidth=1.5,
                    label="ground truth" if ci == 0 else None)
            ax.plot(ch.frames, pred, color="C1", linestyle="--", linewidth=1.5,
                    label="predicted" if ci == 0 else None)
        ax.set_title(DIM_LABELS[d])
        ax.set_ylabel(DIM_LABELS[d])
        ax.grid(True, alpha=0.3)
        if d >= _umi.ARM_DIM - 2:
            ax.set_xlabel("episode frame index")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(
        f"{args.config_name}  ·  episode {args.episode}  ·  {ARM_NAMES[arm]} arm  ·  "
        f"{len(chunks)} chunks (stride {stride})  ·  relative (UMI) frame",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s-arm plot -> %s", ARM_NAMES[arm], out_path)


def main(args: Args) -> None:
    train_config = _config.get_config(args.config_name)
    stride = args.stride or train_config.model.action_horizon

    chunks, _, _ = _collect_chunks(args, train_config)

    stem = pathlib.Path(args.out) if args.out else pathlib.Path(__file__).parent / f"episode_{args.episode}"
    stem.parent.mkdir(parents=True, exist_ok=True)

    for arm in range(_umi.N_ARMS):
        out_path = stem.with_name(f"{stem.name}_{ARM_NAMES[arm]}.png")
        _plot_arm(arm, chunks, args, stride, out_path)


if __name__ == "__main__":
    main(tyro.cli(Args))
