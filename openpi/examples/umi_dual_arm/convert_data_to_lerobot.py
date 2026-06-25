"""Preprocess + clean a raw dual-arm LeRobot v2.1 dataset into a pi0-trainable
LeRobot v2.1 dataset using UMI-style end-effector pose representation.

Task-agnostic: the earphone dataset is just one example of the expected input
structure. Any dataset with the same schema (see step 1) works.

What it does
------------
1. Reads the raw dual-arm dataset (``observation.state`` / ``action`` are 23-dim:
   left[pos3+quat4+grip1], right[pos3+quat4+grip1], ego[pos3+quat4]).
2. Drops the ego (head) motor dims and the head camera ``observation.images.image``.
   Keeps ONLY the two wrist cameras.
3. Converts each arm's [x,y,z]+quat(wxyz) into a 4x4 SE(3) transform, then to a
   UMI-style 9D pose [x,y,z, rot6d(6)] and appends the gripper width:
   per arm -> 10 dims; two arms -> 20-dim state & action.
4. Stores ABSOLUTE poses on disk (state(t) = current EE pose, action(t) = the
   absolute target pose). The UMI *relative trajectory* (each future pose
   relative to the current observation pose, in SE(3)) is applied at LOAD time by
   ``UmiDualArmInputs`` in ``openpi``, NOT baked here -- relative-to-current-obs
   cannot be represented per-frame under openpi's overlapping action chunks.
   pi0's linear delta transform stays OFF.
5. Downsamples 30 fps -> ``TARGET_FPS`` (default 10) and re-encodes the two wrist
   videos at the new rate.
6. Light cleaning: re-normalizes quaternions and skips frames flagged invalid by
   the SLAM / width validity masks.

Usage (must run inside the synced openpi uv env):

    uv run examples/umi_dual_arm/convert_data_to_lerobot.py \
        --raw_root /home/it002338/Junlin_lv/pi0/earphone_0620_316episodes \
        --repo_id umi_dual_arm_6d

Output goes to ``$HF_LEROBOT_HOME/<repo_id>`` (or --output_root if given).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import shutil

import numpy as np
import pandas as pd
import tyro

import pose_util as pu

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME  # noqa: E402
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


# --------------------------------------------------------------------------- #
# Raw-dataset layout constants (per meta/info.json of the earphone dataset).
# 23-dim state/action: indices below index into that vector.
# --------------------------------------------------------------------------- #
LEFT_POS = slice(0, 3)
LEFT_QUAT = slice(3, 7)  # wxyz
LEFT_GRIP = 7
RIGHT_POS = slice(8, 11)
RIGHT_QUAT = slice(11, 15)  # wxyz
RIGHT_GRIP = 15
# ego dims 16:23 are intentionally dropped.

# Raw wrist cameras we keep (head cam ``observation.images.image`` is dropped).
WRIST_1 = "observation.images.wrist_image_1"  # -> left_wrist_0_rgb
WRIST_2 = "observation.images.wrist_image_2"  # -> right_wrist_0_rgb
WRIST_HW = (320, 240)  # raw (height, width) of wrist videos

# Output proprio/action layout: per arm [pos(3), rot6d(6), gripper(1)] = 10, x2 = 20.
ARM_DIM = 10
STATE_DIM = 20
ACTION_DIM = 20


def _arm_to_pose10(pos: np.ndarray, quat_wxyz: np.ndarray, grip: np.ndarray) -> np.ndarray:
    """(T,3),(T,4 wxyz),(T,) -> (T,10) = [pos3, rot6d6, grip1] absolute pose."""
    mat = pu.posequat_to_mat(pos, quat_wxyz)  # (T,4,4)
    pose9 = pu.mat_to_pose9(mat)  # (T,9) = [pos3, rot6d6]
    return np.concatenate([pose9, grip[:, None]], axis=-1).astype(np.float32)


def _build_state_action(vec23: np.ndarray) -> np.ndarray:
    """(T,23) raw state/action -> (T,20) two-arm [pos3+rot6d6+grip1] absolute."""
    left = _arm_to_pose10(vec23[:, LEFT_POS], vec23[:, LEFT_QUAT], vec23[:, LEFT_GRIP])
    right = _arm_to_pose10(vec23[:, RIGHT_POS], vec23[:, RIGHT_QUAT], vec23[:, RIGHT_GRIP])
    return np.concatenate([left, right], axis=-1)  # (T,20)


def _decode_video(path: pathlib.Path) -> np.ndarray:
    """Decode an mp4 into (T, H, W, 3) uint8 RGB using OpenCV."""
    import cv2  # imported lazily so the module imports without the synced env

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.stack(frames)
@dataclasses.dataclass
class Args:
    raw_root: str
    """Path to the raw dual-arm LeRobot dataset root (the dir containing meta/, data/, videos/)."""
    repo_id: str = "umi_dual_arm_6d"
    """Output dataset repo id (folder name under HF_LEROBOT_HOME, or under --output_root)."""
    output_root: str | None = None
    """If set, write the dataset here instead of $HF_LEROBOT_HOME/<repo_id>."""
    target_fps: int = 10
    """Output fps. Raw is 30 fps; we keep every round(30/target_fps)-th frame."""
    drop_invalid_frames: bool = True
    """If True, skip frames where either arm's SLAM or width validity mask is 0."""
    max_episodes: int | None = None
    """If set, only convert the first N episodes (for quick smoke tests)."""


def _load_episode_index(raw_root: pathlib.Path) -> list[dict]:
    with open(raw_root / "meta" / "episodes.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]
def main(args: Args) -> None:
    raw_root = pathlib.Path(args.raw_root)
    episodes = _load_episode_index(raw_root)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    raw_info = json.loads((raw_root / "meta" / "info.json").read_text())
    raw_fps = int(raw_info["fps"])
    stride = max(1, round(raw_fps / args.target_fps))
    print(f"raw_fps={raw_fps} target_fps={args.target_fps} -> frame stride={stride}")

    if args.output_root is not None:
        output_path = pathlib.Path(args.output_root) / args.repo_id
    else:
        output_path = HF_LEROBOT_HOME / args.repo_id
    if output_path.exists():
        print(f"Removing existing output at {output_path}")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="dual_arm",
        fps=args.target_fps,
        root=output_path,
        features={
            # Two wrist cameras only (head cam dropped) -> pi0 left/right wrist slots.
            "left_wrist_image": {
                "dtype": "video",
                "shape": (WRIST_HW[0], WRIST_HW[1], 3),
                "names": ["height", "width", "channel"],
            },
            "right_wrist_image": {
                "dtype": "video",
                "shape": (WRIST_HW[0], WRIST_HW[1], 3),
                "names": ["height", "width", "channel"],
            },
            "state": {"dtype": "float32", "shape": (STATE_DIM,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    # PLACEHOLDER_LOOP
    for ep in episodes:
        ep_idx = ep["episode_index"]
        chunk = ep_idx // raw_info["chunks_size"]
        parquet = raw_root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
        df = pd.read_parquet(parquet)
        n = len(df)

        state23 = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
        action23 = np.stack(df["action"].to_numpy()).astype(np.float64)

        # Per-frame validity: both arms' slam + width masks must be valid.
        valid = np.ones(n, dtype=bool)
        if args.drop_invalid_frames:
            for arm in ("left", "right"):
                valid &= df[f"slam_diagnostics_valid_mask.{arm}"].to_numpy().astype(bool)
                valid &= df[f"width_valid_mask.{arm}"].to_numpy().astype(bool)

        # Build absolute 20-dim state/action (quats re-normalized in posequat_to_mat).
        state20 = _build_state_action(state23)
        action20 = _build_state_action(action23)

        # Decode the two wrist videos (raw 30 fps, row-aligned with the parquet).
        v1 = _decode_video(
            raw_root / "videos" / f"chunk-{chunk:03d}" / WRIST_1 / f"episode_{ep_idx:06d}.mp4"
        )
        v2 = _decode_video(
            raw_root / "videos" / f"chunk-{chunk:03d}" / WRIST_2 / f"episode_{ep_idx:06d}.mp4"
        )
        n_use = min(n, len(v1), len(v2))
        prompt = ep.get("task_annotation") or (ep.get("tasks") or ["manipulation"])[0]

        kept = 0
        for t in range(0, n_use, stride):
            if not valid[t]:
                continue
            dataset.add_frame(
                {
                    "left_wrist_image": v1[t],
                    "right_wrist_image": v2[t],
                    "state": state20[t],
                    "actions": action20[t],
                    "task": prompt,
                }
            )
            kept += 1
        dataset.save_episode()
        print(f"episode {ep_idx:06d}: {n} raw -> {kept} kept frames")

    print(f"Done. Dataset written to {output_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))

