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


@dataclasses.dataclass
class Args:
    raw_root: str
    """Path to the raw dual-arm LeRobot dataset root (the dir containing meta/, data/, videos/)."""
    repo_id: str = "umi_dual_arm_6d"
    """Output dataset repo id (folder name under HF_LEROBOT_HOME, or under --output_root)."""
    output_root: str | None = None
    """If set, write the dataset here instead of $HF_LEROBOT_HOME/<repo_id>."""
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
    # We copy the wrist videos verbatim, so the output fps MUST equal the raw fps
    # and every parquet row maps 1:1 to a video frame (no downsampling/dropping).
    out_fps = raw_fps
    print(f"raw_fps={raw_fps} -> copying videos verbatim, output fps={out_fps}")

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
        fps=out_fps,
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

    # The image keys are stored as videos; add_frame still needs a correctly
    # shaped array per frame to satisfy validation, but we never decode pixels:
    # one reused dummy frame is enough. The throwaway PNGs the writer produces
    # are deleted by save_episode once it finds the (copied) mp4 already present.
    dummy_frame = np.zeros((WRIST_HW[0], WRIST_HW[1], 3), dtype=np.uint8)

    video_key_to_raw = {"left_wrist_image": WRIST_1, "right_wrist_image": WRIST_2}

    for ep in episodes:
        ep_idx = ep["episode_index"]
        chunk = ep_idx // raw_info["chunks_size"]
        parquet = raw_root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
        df = pd.read_parquet(parquet)
        n = len(df)

        state23 = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
        action23 = np.stack(df["action"].to_numpy()).astype(np.float64)

        # Build absolute 20-dim state/action (quats re-normalized in posequat_to_mat).
        state20 = _build_state_action(state23)
        action20 = _build_state_action(action23)

        # Source wrist videos (raw fps, row-aligned with the parquet).
        # The raw data is pre-validated to have parquet rows = video frames,
        # so we use the parquet length directly without counting video frames.
        src_videos = {
            key: raw_root / "videos" / f"chunk-{chunk:03d}" / raw_name / f"episode_{ep_idx:06d}.mp4"
            for key, raw_name in video_key_to_raw.items()
        }
        prompt = ep.get("task_annotation") or (ep.get("tasks") or ["manipulation"])[0]

        for t in range(n):
            dataset.add_frame(
                {
                    "left_wrist_image": dummy_frame,
                    "right_wrist_image": dummy_frame,
                    "state": state20[t],
                    "actions": action20[t],
                    "task": prompt,
                }
            )

        # Place each source mp4 at the exact path the LeRobot writer expects so
        # its ffmpeg encode step is a no-op (it skips when the file exists).
        for key, src in src_videos.items():
            dst = output_path / dataset.meta.get_video_file_path(ep_idx, key)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)

        dataset.save_episode()
        print(f"episode {ep_idx:06d}: {n} frames (videos copied)")

    print(f"Done. Dataset written to {output_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))

