"""Offline reasoning / open-loop trajectory visualization for the fine-tuned pi0.

Given one episode of the LeRobot dataset and a fine-tuned pi0 checkpoint, this
script runs the policy open-loop in **non-overlapping chunks** along the episode
and plots, per arm, the end-effector trajectory in absolute 3D space:

    solid line  = ground-truth trajectory (recorded ``action`` chunk)
    dotted line = model-predicted trajectory

Both come from the *relative* (UMI) action representation: the model predicts a
relative chunk that ``UmiDualArmOutputs`` reconstructs into absolute poses using
the observation pose at the chunk start. Because every chunk is anchored to its
own observation pose, each chunk segment begins at a different absolute point --
which is exactly what these plots show.

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
    # Output image path. Defaults to ``offline_reasoning/episode_<ep>.png``.
    out: str | None = None


def _chunk_to_arm_xyz(chunk20: np.ndarray) -> list[np.ndarray]:
    """(T,20) two-arm pose chunk -> [left (T,3), right (T,3)] EE positions."""
    out = []
    for a in range(_umi.N_ARMS):
        base = a * _umi.ARM_DIM
        out.append(chunk20[:, base : base + 3])
    return out


def main(args: Args) -> None:
    train_config = _config.get_config(args.config_name)
    repo_id = args.repo_id or train_config.data.repo_id
    action_horizon = train_config.model.action_horizon
    stride = args.stride or action_horizon

    # --- policy -----------------------------------------------------------------
    logger.info("Loading fine-tuned policy from %s", args.checkpoint_dir)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint_dir)

    # --- dataset ----------------------------------------------------------------
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

    # --- roll the policy forward in non-overlapping chunks ----------------------
    # Per arm, a list of (gt_xyz, pred_xyz) segments -- one entry per chunk.
    segments: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in range(_umi.N_ARMS)]

    for start in range(ep_from, ep_to, stride):
        frame = dataset[start]

        obs = {
            "state": np.asarray(frame["observation.state"], dtype=np.float32),  # raw (23,)
            "left_wrist_image": np.asarray(frame["observation.images.wrist_image_1"]),
            "right_wrist_image": np.asarray(frame["observation.images.wrist_image_2"]),
            "prompt": str(frame["task"]),
        }

        # Validity mask for the chunk: False = padded step past the episode end.
        valid = ~np.asarray(frame["action_is_pad"], dtype=bool)  # (H,)
        n_valid = int(valid.sum())
        if n_valid < 2:
            continue

        # Ground truth: raw recorded action chunk -> absolute 20-dim poses.
        gt20 = _umi._raw23_to_pose20(np.asarray(frame["action"], dtype=np.float64))  # (H,20)
        # Predicted chunk: policy returns absolute 20-dim poses (already de-relativized
        # by UmiDualArmOutputs using this frame's observation pose).
        pred20 = np.asarray(policy.infer(obs)["actions"], dtype=np.float64)  # (H,20)
        # Index both by the SAME mask so they stay aligned regardless of pad layout.
        gt20, pred20 = gt20[valid], pred20[valid]  # (n,20) each

        gt_arms = _chunk_to_arm_xyz(gt20)
        pred_arms = _chunk_to_arm_xyz(pred20)
        for a in range(_umi.N_ARMS):
            segments[a].append((gt_arms[a], pred_arms[a]))

        errs = [np.linalg.norm(p - g, axis=-1).mean() for g, p in zip(gt_arms, pred_arms, strict=True)]
        logger.info(
            "  chunk @frame %d: %d steps, mean pos err L=%.4f m R=%.4f m", start, n_valid, errs[0], errs[1]
        )

    n_chunks = len(segments[0])
    if n_chunks == 0:
        raise RuntimeError("No valid chunks found for this episode.")

    # --- plot: one 3D axes per arm ---------------------------------------------
    arm_names = ["left", "right"]
    cmap = plt.colormaps["viridis"]
    fig = plt.figure(figsize=(7 * _umi.N_ARMS, 6))

    for a in range(_umi.N_ARMS):
        ax = fig.add_subplot(1, _umi.N_ARMS, a + 1, projection="3d")
        for ci, (gt_xyz, pred_xyz) in enumerate(segments[a]):
            color = cmap(ci / max(n_chunks - 1, 1))
            ax.plot(*gt_xyz.T, color=color, linestyle="-", linewidth=2.0)
            ax.plot(*pred_xyz.T, color=color, linestyle=":", linewidth=2.0)
            # Mark each chunk's starting point -- different per chunk by design.
            ax.scatter(*gt_xyz[0], color=color, marker="o", s=30)
        ax.set_title(f"{arm_names[a]} arm  ({n_chunks} chunks)")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")

    # Shared legend explaining the solid/dotted convention.
    proxy = [
        plt.Line2D([0], [0], color="black", linestyle="-", label="ground truth"),
        plt.Line2D([0], [0], color="black", linestyle=":", label="predicted"),
        plt.Line2D([0], [0], color="black", marker="o", linestyle="", label="chunk start"),
    ]
    fig.legend(handles=proxy, loc="upper center", ncol=3)
    fig.suptitle(
        f"{args.config_name}  ·  episode {args.episode}  ·  stride {stride}  ·  color = chunk order",
        y=0.02,
    )

    out = pathlib.Path(args.out) if args.out else pathlib.Path(__file__).parent / f"episode_{args.episode}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out, dpi=150)
    logger.info("Saved plot -> %s", out)


if __name__ == "__main__":
    main(tyro.cli(Args))
