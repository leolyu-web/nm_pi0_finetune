"""Build a schema-consistent copy of a LeRobot v2.1 dataset.

Some datasets (e.g. the qc_accept pipeline output) add diagnostic columns
(slam_anomaly.*, width_anomaly.*, ...) to SOME episode parquet files but not
others. HuggingFace `load_dataset("parquet", ...)` infers one schema and then
rejects files whose columns differ, so `LeRobotDataset(...)` crashes with a
CastError ("because column names don't match").

This projects every episode parquet down to the columns common to ALL files
(the intersection), which drops the inconsistent extras while keeping every
feature LeRobot/openpi needs (state, action, timestamps, observation.extra.*).
`meta/` is copied and `videos/` is symlinked, so the copy is cheap.

Usage:
    uv run scripts/normalize_lerobot_parquet_schema.py <src_dataset_dir> <dst_dataset_dir>
Then point the config's repo_id / repo_ids entry at <dst_dataset_dir>.
"""

from pathlib import Path
import shutil
import sys

import pyarrow.parquet as pq


def main(src: Path, dst: Path) -> None:
    pfiles = sorted(src.glob("data/**/*.parquet"))
    if not pfiles:
        raise SystemExit(f"No parquet files under {src / 'data'}")

    # Canonical column set = intersection across all files, ordered by the first file.
    all_cols = [set(pq.read_schema(f).names) for f in pfiles]
    common = set.intersection(*all_cols)
    order = [c for c in pq.read_schema(pfiles[0]).names if c in common]
    dropped = sorted(set().union(*all_cols) - set(order))
    print(f"{len(pfiles)} parquet files")
    print(f"KEEP ({len(order)}): {order}")
    print(f"DROP ({len(dropped)}): {dropped}")
    if not dropped:
        print("Schemas already consistent -- nothing to drop. (Copy still created.)")

    dst.mkdir(parents=True, exist_ok=True)
    # meta: copy (small, and LeRobot reads feature schema / stats from here).
    shutil.copytree(src / "meta", dst / "meta", dirs_exist_ok=True)
    # videos: symlink the whole tree (large, unchanged).
    if (src / "videos").exists() and not (dst / "videos").exists():
        (dst / "videos").symlink_to((src / "videos").resolve())

    for f in pfiles:
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pq.read_table(f).select(order), out)
    print(f"Done -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
