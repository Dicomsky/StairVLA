#!/usr/bin/env python
"""Write a copy of any LeRobot v3.0 dataset with selected episodes removed (same fps).

Works on raw joint-space recordings as well as EE datasets. Episodes are selected with the same
exclusion manifest / flags as resample.py; kept episodes are renumbered contiguously from zero.
Video files whose frames are all kept are hard-linked (or copied); files that lose episodes are
re-encoded (H.264) with exactly the kept frames.

You do not need this tool for the released datasets: resample.py applies the same filters while
resampling. Use it to clean a recording before converting it, or to filter at the original rate.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from examples.PiperX.dataset_tools.dataset_utils import (
    add_filter_args,
    filter_from_args,
    selected_episode_map,
    stat,
    update_flat_episode_stats,
    video_keys,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a LeRobot v3.0 dataset with selected episodes removed.")
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    add_filter_args(parser)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--video-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How to reuse video files whose frames are all kept (others are always re-encoded).",
    )
    return parser.parse_args()


def load_tables(root: Path, rel_dir: str) -> list[tuple[Path, pd.DataFrame]]:
    return [(path, pd.read_parquet(path)) for path in sorted((root / rel_dir).glob("chunk-*/*.parquet"))]


def compute_stats(df: pd.DataFrame, image_stats_source: dict | None = None) -> dict:
    stats: dict[str, dict[str, list]] = {}
    if "action" in df:
        stats["action"] = stat(np.stack(df["action"].to_numpy()))
    if "observation.state" in df:
        stats["observation.state"] = stat(np.stack(df["observation.state"].to_numpy()))
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        if key in df:
            stats[key] = stat(df[key].to_numpy())
    if image_stats_source:
        for key, value in image_stats_source.items():
            if key.startswith("observation.images."):
                stats[key] = value
    return stats


def ffprobe_frames(video: Path) -> int:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        text=True,
    ).strip()
    return int(out)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def write_exact_video(src_file: Path, dst_file: Path, frame_indices: list[int], fps: float, mode: str) -> None:
    before = ffprobe_frames(src_file)
    if frame_indices == list(range(before)):
        link_or_copy(src_file, dst_file, mode)
        print(f"[VIDEO] {dst_file}: linked/copied full {before} frames")
        return
    if not frame_indices:
        return
    select_expr = "+".join(f"eq(n\\,{int(index)})" for index in frame_indices)
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    filter_script = None
    if len(select_expr) > 100_000:
        # Long expressions exceed the command-line limit; pass them through a filter script.
        filter_script = dst_file.with_suffix(".ffmpeg-filter")
        filter_script.write_text(f"select={select_expr},setpts=N/({float(fps)}*TB)\n")
        filter_args = ["-filter_script:v", str(filter_script)]
    else:
        filter_args = ["-vf", f"select={select_expr},setpts=N/({float(fps)}*TB)"]
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(src_file),
                *filter_args,
                "-an",
                "-r",
                str(float(fps)),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                str(dst_file),
            ],
            check=True,
        )
    finally:
        if filter_script is not None:
            filter_script.unlink(missing_ok=True)
    after = ffprobe_frames(dst_file)
    if after != len(frame_indices):
        raise RuntimeError(f"{dst_file}: wrote {after}, expected {len(frame_indices)}")
    print(f"[VIDEO] {dst_file}: {before} -> {after} frames (re-encoded)")


def main() -> None:
    args = parse_args()
    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    episode_filter = filter_from_args(args)
    if not episode_filter.active:
        raise ValueError("No filters given: pass --exclusions, --exclude-range, --exclude-episode or --include-task.")
    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dst} already exists. Use --overwrite to replace it.")
        shutil.rmtree(dst)

    info = json.loads((src / "meta/info.json").read_text())
    fps = float(info["fps"])
    vkeys = video_keys(info)
    data_tables = load_tables(src, "data")
    episode_tables = load_tables(src, "meta/episodes")
    data_all = pd.concat([df for _, df in data_tables], ignore_index=True)
    episodes_all = pd.concat([df for _, df in episode_tables], ignore_index=True).sort_values("episode_index")

    episode_map = selected_episode_map(episodes_all, episode_filter)
    old_episode_ids = [int(ep) for ep in episodes_all["episode_index"]]
    excluded = sorted(set(old_episode_ids) - set(episode_map))
    if not excluded:
        raise ValueError("The filters matched no episodes of this dataset; nothing to remove.")
    keep_old_episode_ids = sorted(episode_map)

    dst.mkdir(parents=True)
    (dst / "data").mkdir()
    (dst / "meta").mkdir()
    shutil.copy2(src / "meta/tasks.parquet", dst / "meta/tasks.parquet")
    if (src / "meta/modality.json").exists():
        shutil.copy2(src / "meta/modality.json", dst / "meta/modality.json")

    kept = data_all[data_all["episode_index"].isin(keep_old_episode_ids)].copy()
    kept["episode_index"] = kept["episode_index"].map(episode_map).astype(np.int64)
    kept["index"] = np.arange(len(kept), dtype=np.int64)

    for src_file, old_df in data_tables:
        old_eps = [int(ep) for ep in old_df["episode_index"].unique()]
        new_eps = [episode_map[ep] for ep in old_eps if ep in episode_map]
        out_df = kept[kept["episode_index"].isin(new_eps)].copy()
        if len(out_df) == 0:
            continue
        out_path = dst / "data" / src_file.relative_to(src / "data")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(out_path, index=False)
        print(f"[DATA] wrote {out_path} rows={len(out_df)}")

    old_stats = json.loads((src / "meta/stats.json").read_text()) if (src / "meta/stats.json").exists() else {}
    stats = compute_stats(kept, image_stats_source=old_stats)
    (dst / "meta/stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    per_episode_stats = {
        int(ep): compute_stats(ep_df, image_stats_source=old_stats)
        for ep, ep_df in kept.groupby("episode_index", sort=False)
    }

    selections: dict[tuple[str, int, int], list[int]] = {}
    offsets: dict[tuple[str, int, int], float] = {}
    kept_episode_rows = []
    for _, row in episodes_all.iterrows():
        old_ep = int(row["episode_index"])
        if old_ep not in episode_map:
            continue
        new_ep = episode_map[old_ep]
        ep_df = kept[kept["episode_index"] == new_ep]
        new_row = row.copy()
        new_row["episode_index"] = new_ep
        new_row["length"] = int(len(ep_df))
        new_row["dataset_from_index"] = int(ep_df["index"].min())
        new_row["dataset_to_index"] = int(ep_df["index"].max() + 1)
        for vkey in vkeys:
            chunk = int(row[f"videos/{vkey}/chunk_index"])
            file_index = int(row[f"videos/{vkey}/file_index"])
            key = (vkey, chunk, file_index)
            source_start = int(round(float(row[f"videos/{vkey}/from_timestamp"]) * fps))
            frame_indices = (source_start + np.arange(int(row["length"]), dtype=np.int64)).tolist()
            selections.setdefault(key, []).extend(frame_indices)
            offset = offsets.get(key, 0.0)
            duration = len(ep_df) / fps
            new_row[f"videos/{vkey}/from_timestamp"] = offset
            new_row[f"videos/{vkey}/to_timestamp"] = offset + duration
            offsets[key] = offset + duration
        kept_episode_rows.append(update_flat_episode_stats(new_row, per_episode_stats[new_ep]))
    episodes_new = pd.DataFrame(kept_episode_rows)

    for src_file, old_eps_df in episode_tables:
        old_eps = [int(ep) for ep in old_eps_df["episode_index"]]
        new_eps = [episode_map[ep] for ep in old_eps if ep in episode_map]
        out = episodes_new[episodes_new["episode_index"].isin(new_eps)].copy()
        if len(out) == 0:
            continue
        out_path = dst / "meta/episodes" / src_file.relative_to(src / "meta/episodes")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path, index=False)
        print(f"[META] wrote {out_path} episodes={len(out)}")

    for vkey, chunk, file_index in sorted(selections):
        src_video = src / "videos" / vkey / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
        dst_video = dst / "videos" / vkey / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
        write_exact_video(src_video, dst_video, selections[(vkey, chunk, file_index)], fps, args.video_mode)

    info["total_episodes"] = int(episodes_new["episode_index"].nunique())
    info["total_frames"] = int(len(kept))
    info["splits"] = {"train": f"0:{info['total_episodes']}"}
    (dst / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"[DONE] wrote filtered dataset: {dst}")
    print(f"[CHECK] excluded source episodes ({len(excluded)}): {' '.join(str(v) for v in excluded)}")
    print(f"[CHECK] episodes={info['total_episodes']} frames={info['total_frames']} fps={fps:g}")


if __name__ == "__main__":
    main()
