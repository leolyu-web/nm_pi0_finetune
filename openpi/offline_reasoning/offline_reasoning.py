"""Offline reasoning / open-loop comparison for the fine-tuned pi0 (per-dim relative).

Given one episode of the LeRobot dataset and a fine-tuned pi0 checkpoint, this
script runs the policy open-loop in chunks along the episode and plots, **per
arm**, the model's prediction against ground truth in the *relative* (UMI)
action representation -- i.e. exactly the per-arm vector the model is trained to
regress.

The representation is auto-detected from the train config:

  * quaternion configs (``UmiDualArmDataConfig`` -- v2/v3): 8-dim per arm
    ``[rel_pos3, rel_quat_wxyz4, grip1]``.
  * 6D-rotation config (``UmiDualArmRot6dDataConfig`` -- v4): 10-dim per arm
    ``[rel_pos3, rel_rot6d6, grip1]``.

For v3 the policy's ``UmiDualArmOutputs`` already undoes the world reframe ``W``,
so ``policy.infer`` returns absolute poses in the ORIGINAL frame -- the same frame
the raw dataset action lives in -- and the relativization below is ``W``-invariant.
No reframe handling is needed here.

Output: TWO PNGs, one per arm (``..._left.png`` / ``..._right.png``). Each PNG has
one subplot per pose dimension. In every subplot the x-axis is the episode frame
index and two lines are drawn per chunk:

    solid  = ground truth   (recorded ``action`` chunk, relativized)
    dashed = model predicted (policy output, re-expressed in the same frame)

Both series are anchored to the chunk-start observation pose (and, for quaternions,
canonicalized ``w >= 0``), so they live in an identical relative frame. Because
each chunk re-anchors, every chunk's relative pose starts near identity
(rel_pos ~ 0) -- the signature of the UMI representation.

Example
-------
    uv run offline_reasoning/offline_reasoning.py \
        --checkpoint-dir checkpoints/pi0_umi_dual_arm_quat/run1/29999 \
        --episode 0

    uv run offline_reasoning/offline_reasoning.py \
        --config-name pi0_umi_dual_arm_6Drot \
        --checkpoint-dir checkpoints/pi0_umi_dual_arm_6Drot/run1/29999 \
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
from openpi.policies import umi_dual_arm_policy as _umi_quat
from openpi.policies import umi_dual_arm_rot6d_policy as _umi_rot6d
from openpi.training import config as _config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

N_ARMS = _umi_quat.N_ARMS  # 2 (identical in both modules)
_QUAT_ARM_DIM = _umi_quat.ARM_DIM  # 8; raw dataset is always quaternion
ARM_NAMES = ["left", "right"]

# Per-arm dimension labels for each native representation.
_QUAT_LABELS = [
    "rel x [m]", "rel y [m]", "rel z [m]",
    "rel quat w", "rel quat x", "rel quat y", "rel quat z",
    "gripper (abs)",
]
_ROT6D_LABELS = [
    "rel x [m]", "rel y [m]", "rel z [m]",
    "rel rot6d 1", "rel rot6d 2", "rel rot6d 3", "rel rot6d 4", "rel rot6d 5", "rel rot6d 6",
    "gripper (abs)",
]


@dataclasses.dataclass
class _Repr:
    """Native action representation the model regresses (auto-detected from config)."""

    name: str  # "quaternion" or "rot6d"
    arm_dim: int  # native per-arm dim (8 quat, 10 rot6d)
    labels: list[str]
    is_rot6d: bool

    def abs_block_to_mat(self, block: np.ndarray) -> np.ndarray:
        """Native ABSOLUTE per-arm pose block (...,arm_dim) -> (...,4,4)."""
        if self.is_rot6d:
            return _umi_rot6d._rot6dpose_to_mat(block)
        return _umi_quat._pose8_to_mat(block)

    def mat_to_native_rot(self, mat: np.ndarray) -> np.ndarray:
        """(...,4,4) -> (..., arm_dim-1) = [pos3, rot] in the native rep (no gripper)."""
        if self.is_rot6d:
            return _umi_rot6d._mat_to_pose9(mat)
        return _umi_quat._mat_to_pose7(mat)


def _make_repr(train_config) -> _Repr:
    if isinstance(train_config.data, _config.UmiDualArmRot6dDataConfig):
        return _Repr(name="rot6d", arm_dim=_umi_rot6d.ARM_DIM, labels=_ROT6D_LABELS, is_rot6d=True)
    return _Repr(name="quaternion", arm_dim=_umi_quat.ARM_DIM, labels=_QUAT_LABELS, is_rot6d=False)


@dataclasses.dataclass
class Args:
    # Path to the fine-tuned checkpoint (the step folder, e.g. ``.../run1/29999``).
    checkpoint_dir: str
    # Train config name the checkpoint was produced with.
    config_name: str = "pi0_umi_dual_arm_quat"
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
    gt_rel: np.ndarray  # (n, arm_dim*N_ARMS) ground-truth relative pose (both arms)
    pred_rel: np.ndarray  # (n, arm_dim*N_ARMS) predicted relative pose (both arms)


def _arm_slice(arm: int, arm_dim: int) -> slice:
    return slice(arm * arm_dim, (arm + 1) * arm_dim)


def _relativize_against_base(
    repr_: _Repr, base_mats: list[np.ndarray], abs_blocks: np.ndarray, grip_idx: int
) -> np.ndarray:
    """Express an absolute action chunk relative to a per-arm base, in the native rep.

    base_mats: list of N_ARMS (4,4) base (current-obs) matrices.
    abs_blocks: (H, arm_dim*N_ARMS) absolute per-arm native pose chunk.
    grip_idx:  index of the gripper within each native per-arm block.
    Returns (H, arm_dim*N_ARMS): per arm [rel_pos3, rel_rot, grip1] (gripper absolute).
    """
    out = np.empty_like(abs_blocks)
    for a in range(N_ARMS):
        sl = _arm_slice(a, repr_.arm_dim)
        block = abs_blocks[:, sl]
        act_mat = repr_.abs_block_to_mat(block)  # (H,4,4)
        rel_mat = _umi_quat._mat_inv(base_mats[a])[None] @ act_mat  # (H,4,4)
        out[:, sl] = np.concatenate([repr_.mat_to_native_rot(rel_mat), block[:, grip_idx, None]], axis=-1)
    return out


def _collect_chunks(args: Args, train_config, repr_: _Repr) -> list[Chunk]:
    """Run the policy open-loop and return per-chunk relative GT/pred arrays."""
    repo_id = args.repo_id or train_config.data.repo_id
    action_horizon = train_config.model.action_horizon
    stride = args.stride or action_horizon

    logger.info("Representation: %s (%d dim/arm)", repr_.name, repr_.arm_dim)
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

        # Raw current observation pose (23,) -> per-arm absolute base matrices.
        # The raw dataset is always quaternion, so the base is built with the quat
        # module; this matches the base both policies use internally for relativization.
        state_q16 = _umi_quat._raw23_to_pose16(np.asarray(frame["observation.state"], dtype=np.float64))
        base_mats = [
            _umi_quat._pose8_to_mat(state_q16[_arm_slice(a, _QUAT_ARM_DIM)]) for a in range(N_ARMS)
        ]

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

        # Ground truth: raw recorded action chunk (quaternion) relativized in the
        # native rep against the chunk-start base. For rot6d this reproduces the v4
        # training target exactly (matrix from quat -> relativize -> rot6d rows).
        gt_q16 = _umi_quat._raw23_to_pose16(np.asarray(frame["action"], dtype=np.float64))  # (H,16)
        gt_rel = _relativize_against_base(repr_, base_mats, gt_q16, grip_idx=_umi_quat._GRIP)

        # Prediction: policy returns ABSOLUTE native poses (UmiDualArmOutputs already
        # de-relativized them, and de-reframed for v3). Re-relativize against the SAME
        # base so GT and pred are compared in an identical relative frame.
        pred_abs = np.asarray(policy.infer(obs)["actions"], dtype=np.float64)  # (H, arm_dim*N_ARMS)
        grip_idx = _umi_rot6d._GRIP if repr_.is_rot6d else _umi_quat._GRIP
        pred_rel = _relativize_against_base(repr_, base_mats, pred_abs, grip_idx=grip_idx)

        gt_rel, pred_rel = gt_rel[valid], pred_rel[valid]
        frames = np.nonzero(valid)[0] + start  # absolute frame index per valid step

        chunks.append(Chunk(start=start, frames=frames, gt_rel=gt_rel, pred_rel=pred_rel))

        # Per-arm mean position error (in the relative frame == absolute displacement error).
        errs = [
            np.linalg.norm(
                pred_rel[:, _arm_slice(a, repr_.arm_dim)][:, :3]
                - gt_rel[:, _arm_slice(a, repr_.arm_dim)][:, :3],
                axis=-1,
            ).mean()
            for a in range(N_ARMS)
        ]
        logger.info(
            "  chunk @frame %d: %d steps, mean rel-pos err L=%.4f m R=%.4f m",
            start, len(frames), errs[0], errs[1],
        )

    if not chunks:
        raise RuntimeError("No valid chunks found for this episode.")
    return chunks


def _plot_arm(arm: int, chunks: list[Chunk], args: Args, stride: int, repr_: _Repr, out_path: pathlib.Path) -> None:
    """Render the per-dim GT-vs-pred figure for a single arm and save it."""
    sl = _arm_slice(arm, repr_.arm_dim)
    nrows = (repr_.arm_dim + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 3 * nrows), sharex=True)
    axes = axes.ravel()

    for d in range(repr_.arm_dim):
        ax = axes[d]
        for ci, ch in enumerate(chunks):
            gt = ch.gt_rel[:, sl][:, d]
            pred = ch.pred_rel[:, sl][:, d]
            # Label only the first chunk so the legend stays clean.
            ax.plot(ch.frames, gt, color="C0", linestyle="-", linewidth=1.5,
                    label="ground truth" if ci == 0 else None)
            ax.plot(ch.frames, pred, color="C1", linestyle="--", linewidth=1.5,
                    label="predicted" if ci == 0 else None)
        ax.set_title(repr_.labels[d])
        ax.set_ylabel(repr_.labels[d])
        ax.grid(True, alpha=0.3)
        if d >= repr_.arm_dim - 2:
            ax.set_xlabel("episode frame index")

    # Hide any unused subplot (odd dim count).
    for d in range(repr_.arm_dim, len(axes)):
        axes[d].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(
        f"{args.config_name}  ·  episode {args.episode}  ·  {ARM_NAMES[arm]} arm  ·  "
        f"{len(chunks)} chunks (stride {stride})  ·  relative (UMI, {repr_.name}) frame",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s-arm plot -> %s", ARM_NAMES[arm], out_path)


def main(args: Args) -> None:
    train_config = _config.get_config(args.config_name)
    repr_ = _make_repr(train_config)
    stride = args.stride or train_config.model.action_horizon

    chunks = _collect_chunks(args, train_config, repr_)

    stem = pathlib.Path(args.out) if args.out else pathlib.Path(__file__).parent / f"episode_{args.episode}"
    stem.parent.mkdir(parents=True, exist_ok=True)

    for arm in range(N_ARMS):
        out_path = stem.with_name(f"{stem.name}_{ARM_NAMES[arm]}.png")
        _plot_arm(arm, chunks, args, stride, repr_, out_path)


if __name__ == "__main__":
    main(tyro.cli(Args))
