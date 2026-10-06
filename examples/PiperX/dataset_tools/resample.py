#!/usr/bin/env python
"""Resample a PiperX EE LeRobot v3.0 dataset (output of convert_to_ee.py) to a lower FPS.

The EE *state* trajectory is interpolated at the output timestamps (linear for position and
gripper, slerp for rotation) and the temporal-next-state actions are recomputed between
consecutive output states; source actions are never interpolated. Episode filtering
(--exclusions manifest, --exclude-range, --exclude-episode, --include-task) is applied here,
kept episodes are renumbered contiguously from zero in ascending source order, and the applied
settings are written to meta/conversion_provenance.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from examples.PiperX.dataset_tools.dataset_utils import (
    EpisodeFilter,
    add_filter_args,
    filter_from_args,
    read_dataset_data,
    read_episode_metadata,
    selected_episode_map,
    stats_for_df,
    update_flat_episode_stats,
    video_keys,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resample a PiperX EE-delta LeRobot dataset to a lower FPS.")
    parser.add_argument("--src", type=Path, required=True, help="EE dataset written by convert_to_ee.py.")
    parser.add_argument("--dst", type=Path, required=True, help="Output dataset directory.")
    parser.add_argument("--fps", type=float, required=True, help="Output control rate, e.g. 8 (Fruit25) or 20 (PushBlock).")
    parser.add_argument("--overwrite", action="store_true")
    add_filter_args(parser)
    parser.add_argument(
        "--video-mode",
        choices=("reencode", "skip"),
        default="reencode",
        help="reencode: cut the frames matching the resampled rows (H.264). "
        "skip: write data/metadata only (for checks; the result has no videos).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print which source episodes the filters keep, then exit without writing.",
    )
    return parser.parse_args()


def interp_vectors(times: np.ndarray, values: np.ndarray, out_times: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(out_times, times, values[:, i]) for i in range(values.shape[1])])


def interp_rotations(times: np.ndarray, rotations: R, out_times: np.ndarray) -> R:
    if len(times) == 1:
        return R.from_quat(np.repeat(rotations.as_quat(), len(out_times), axis=0))
    unique_times, unique_indices = np.unique(times, return_index=True)
    unique_rots = R.from_quat(rotations.as_quat()[unique_indices])
    return Slerp(unique_times, unique_rots)(out_times)


def resample_episode(ep_df: pd.DataFrame, fps: float) -> pd.DataFrame:
    ep_df = ep_df.sort_values("frame_index").reset_index(drop=True)
    times = ep_df["timestamp"].to_numpy(dtype=np.float64)
    source_episode_indices = ep_df["frame_index"].to_numpy(dtype=np.int64)
    if len(ep_df) <= 1:
        out_times = np.array([0.0], dtype=np.float64)
    else:
        max_t = float(times[-1])
        out_times = np.arange(0.0, max_t + 1e-9, 1.0 / fps, dtype=np.float64)

    states = np.stack(ep_df["observation.state"].to_numpy()).astype(np.float64)
    actions = np.stack(ep_df["action"].to_numpy()).astype(np.float64)

    out_state_pos = interp_vectors(times, states[:, :3], out_times)
    out_state_rot = interp_rotations(times, R.from_quat(states[:, 3:7]), out_times)
    out_state_gripper = np.interp(out_times, times, states[:, 7])

    # Nearest source frame for every output timestamp: used to pick video frames.
    nearest = np.searchsorted(times, out_times, side="left")
    nearest = np.clip(nearest, 0, len(times) - 1)
    prev = np.clip(nearest - 1, 0, len(times) - 1)
    use_prev = np.abs(times[prev] - out_times) < np.abs(times[nearest] - out_times)
    nearest[use_prev] = prev[use_prev]
    out_action_gripper = actions[nearest, 6]

    out_states = np.concatenate(
        [out_state_pos, out_state_rot.as_quat(), out_state_gripper[:, None]], axis=1
    ).astype(np.float32)
    # Temporal-next-state between consecutive *output* states; last frame: zero delta, current gripper.
    out_delta_pos = np.zeros_like(out_state_pos)
    out_delta_rot = np.zeros((len(out_times), 3), dtype=np.float64)
    if len(out_times) > 1:
        out_delta_pos[:-1] = out_state_pos[1:] - out_state_pos[:-1]
        out_delta_rot[:-1] = (out_state_rot[:-1].inv() * out_state_rot[1:]).as_rotvec()
        out_action_gripper[:-1] = out_state_gripper[1:]
    out_action_gripper[-1] = out_state_gripper[-1]
    out_actions = np.concatenate(
        [out_delta_pos, out_delta_rot, out_action_gripper[:, None]], axis=1
    ).astype(np.float32)

    out = ep_df.iloc[: len(out_times)].copy()
    if len(out) != len(out_times):
        out = pd.concat([ep_df.iloc[[0]].copy() for _ in out_times], ignore_index=True)
    out["observation.state"] = list(out_states)
    out["action"] = list(out_actions)
    out["timestamp"] = out_times.astype(np.float32)
    out["frame_index"] = np.arange(len(out_times), dtype=np.int64)
    out["__source_episode_frame_index"] = source_episode_indices[nearest]
    return out


def probe_frame_count(dst_file: Path) -> str:
    return subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(dst_file),
        ],
        text=True,
    ).strip()


def write_exact_video(src_file: Path, dst_file: Path, frame_indices: np.ndarray, fps: float) -> None:
    """Re-encode exactly the source frames ``frame_indices`` (0-based, in order) at ``fps``."""
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    if len(frame_indices) == 0:
        raise RuntimeError(f"No frames requested for {dst_file}")
    if len(np.unique(frame_indices)) != len(frame_indices):
        raise RuntimeError(f"Duplicate source frame indices are not supported for exact video sampling: {dst_file}")

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
    cmd = [
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
    ]
    try:
        subprocess.run(cmd, check=True)
    finally:
        if filter_script is not None:
            filter_script.unlink(missing_ok=True)

    probe = probe_frame_count(dst_file)
    if probe and int(probe) != len(frame_indices):
        raise RuntimeError(f"Video frame mismatch for {dst_file}: wrote {probe}, expected {len(frame_indices)}")


def reencode_videos(src: Path, dst: Path, fps: float, video_frame_indices_by_video_rel: dict[Path, np.ndarray]) -> None:
    for rel in sorted(video_frame_indices_by_video_rel):
        src_file = src / "videos" / rel
        if not src_file.exists():
            raise FileNotFoundError(src_file)
        dst_file = dst / "videos" / rel
        write_exact_video(src_file, dst_file, video_frame_indices_by_video_rel[rel], fps)
        print(f"[VIDEO] wrote {dst_file}")


def update_info(info: dict, fps: float, reencoded: bool, total_frames: int | None = None) -> dict:
    out = dict(info)
    out["fps"] = fps
    if total_frames is not None:
        out["total_frames"] = int(total_frames)
    for feature in out["features"].values():
        if isinstance(feature, dict):
            if "fps" in feature:
                feature["fps"] = fps
            if "info" in feature and isinstance(feature["info"], dict) and "video.fps" in feature["info"]:
                feature["info"]["video.fps"] = fps
                if reencoded:
                    feature["info"]["video.codec"] = "h264"
                    feature["info"]["video.pix_fmt"] = "yuv420p"
    return out


def write_info_and_stats(dst: Path, src_info: dict, src_stats: dict, fps: float, reencoded: bool) -> pd.DataFrame:
    all_df = read_dataset_data(dst)
    if len(all_df) == 0:
        raise RuntimeError(f"No rows found in {dst}")
    info = update_info(src_info, fps, reencoded, len(all_df))
    info["total_episodes"] = int(all_df["episode_index"].nunique())
    used_task_ids = sorted(int(value) for value in all_df["task_index"].unique())
    info["total_tasks"] = len(used_task_ids)
    info["splits"] = {"train": f"0:{info['total_episodes']}"}
    (dst / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    tasks_path = dst / "meta/tasks.parquet"
    tasks = pd.read_parquet(tasks_path)
    tasks[tasks["task_index"].isin(used_task_ids)].to_parquet(tasks_path)
    src_stats.update(stats_for_df(all_df))
    (dst / "meta/stats.json").write_text(json.dumps(src_stats, indent=2) + "\n")
    return all_df


def episode_video_sources(metadata: pd.DataFrame, info: dict) -> dict[int, dict[str, tuple[Path, int]]]:
    """Source video file and first frame of every (episode, camera)."""
    vkeys = video_keys(info)
    src_fps = float(info.get("fps", 0.0))
    if not vkeys or src_fps <= 0:
        return {}

    out: dict[int, dict[str, tuple[Path, int]]] = {}
    for _, row in metadata.iterrows():
        ep = int(row["episode_index"])
        out[ep] = {}
        for vkey in vkeys:
            chunk = int(row[f"videos/{vkey}/chunk_index"])
            file_index = int(row[f"videos/{vkey}/file_index"])
            start = int(round(float(row[f"videos/{vkey}/from_timestamp"]) * src_fps))
            rel = Path(vkey) / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
            out[ep][vkey] = (rel, start)
    return out


def write_provenance(dst: Path, src_arg: Path, src_info: dict, fps: float, episode_filter: EpisodeFilter) -> None:
    provenance = {
        "source_dataset": str(src_arg),
        "source_fps": float(src_info["fps"]),
        "output_fps": float(fps),
        "action_definition": "temporal-next-state",
        "translation_action": "position[t+1] - position[t]",
        "rotation_action": "log(inverse(rotation[t]) * rotation[t+1])",
        "gripper_action": "absolute gripper state[t+1]",
        "last_frame": "zero translation/rotation delta and current absolute gripper",
        "excluded_episode_ranges": [list(item) for item in episode_filter.exclude_ranges],
        "excluded_episode_ids": sorted(episode_filter.exclude_episodes),
        "source_episode_order": "ascending; output episode ids are contiguous from zero",
    }
    if episode_filter.include_tasks:
        provenance["included_tasks"] = list(episode_filter.include_tasks)
    (dst / "meta/conversion_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    episode_filter = filter_from_args(args)
    src_info = json.loads((src / "meta/info.json").read_text())
    if list(src_info["features"]["observation.state"]["shape"]) != [8]:
        raise ValueError(f"{src} is not an EE dataset (expected 8D observation.state); run convert_to_ee.py first.")
    src_metadata = read_episode_metadata(src).sort_values("episode_index")
    episode_map = selected_episode_map(src_metadata, episode_filter)
    excluded_source = sorted(set(int(v) for v in src_metadata["episode_index"]) - set(episode_map))
    print(f"[FILTER] source_episodes={len(src_metadata)} kept={len(episode_map)} excluded={len(excluded_source)}")
    if args.dry_run:
        print(f"[FILTER] excluded source ids: {' '.join(str(v) for v in excluded_source) or 'none'}")
        return

    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dst} already exists. Use --overwrite to replace it.")
        shutil.rmtree(dst)

    (dst / "data").mkdir(parents=True, exist_ok=True)
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "meta/tasks.parquet", dst / "meta/tasks.parquet")
    if (src / "meta/modality.json").exists():
        shutil.copy2(src / "meta/modality.json", dst / "meta/modality.json")

    vkeys = video_keys(src_info)
    video_sources_by_episode = episode_video_sources(src_metadata, src_info)

    all_dfs: list[pd.DataFrame] = []
    video_frame_indices_by_video_rel: dict[Path, list[int]] = {}
    next_index = 0
    for src_file in sorted((src / "data").glob("chunk-*/*.parquet")):
        rel = src_file.relative_to(src / "data")
        out_path = dst / "data" / rel
        df = pd.read_parquet(src_file)
        df = df[df["episode_index"].isin(episode_map)].copy()
        if len(df) == 0:
            continue
        file_out: list[pd.DataFrame] = []
        for _, ep_df in df.groupby("episode_index", sort=False):
            source_episode = int(ep_df["episode_index"].iloc[0])
            out_ep = resample_episode(ep_df, args.fps)
            for vkey in vkeys:
                if source_episode not in video_sources_by_episode:
                    continue
                video_rel, start = video_sources_by_episode[source_episode][vkey]
                selected = start + out_ep["__source_episode_frame_index"].to_numpy(dtype=np.int64)
                video_frame_indices_by_video_rel.setdefault(video_rel, []).extend(selected.tolist())
            out_ep["episode_index"] = episode_map[source_episode]
            out_ep["index"] = np.arange(next_index, next_index + len(out_ep), dtype=np.int64)
            next_index += len(out_ep)
            file_out.append(out_ep)
        out_df = pd.concat(file_out, ignore_index=True).drop(columns=["__source_episode_frame_index"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(out_path, index=False)
        all_dfs.append(out_df)
        print(f"[DATA] wrote {out_path} rows={len(out_df)}")

    stats = json.loads((src / "meta/stats.json").read_text())
    all_df_for_new_stats = pd.concat(all_dfs, ignore_index=True)
    per_episode_stats = {
        int(ep): stats_for_df(ep_df) for ep, ep_df in all_df_for_new_stats.groupby("episode_index", sort=False)
    }
    episode_rows = {int(ep): ep_df for ep, ep_df in all_df_for_new_stats.groupby("episode_index", sort=False)}

    # Output videos keep the source file layout; each output file holds the selected frames of the
    # kept episodes in ascending order, so timestamps accumulate per (camera, chunk, file).
    video_offsets: dict[tuple[str, int, int], float] = {}
    for src_file in sorted((src / "meta/episodes").glob("chunk-*/*.parquet")):
        rel = src_file.relative_to(src / "meta/episodes")
        ep_meta = pd.read_parquet(src_file)
        rows = []
        for _, row in ep_meta.iterrows():
            source_ep = int(row["episode_index"])
            if source_ep not in episode_map:
                continue
            ep = episode_map[source_ep]
            ep_df = episode_rows[ep]
            new_row = row.copy()
            new_row["episode_index"] = ep
            new_row["length"] = int(len(ep_df))
            new_row["dataset_from_index"] = int(ep_df["index"].min())
            new_row["dataset_to_index"] = int(ep_df["index"].max() + 1)
            duration_s = len(ep_df) / args.fps
            for vkey in vkeys:
                key = (vkey, int(row[f"videos/{vkey}/chunk_index"]), int(row[f"videos/{vkey}/file_index"]))
                offset = video_offsets.get(key, 0.0)
                new_row[f"videos/{vkey}/from_timestamp"] = offset
                new_row[f"videos/{vkey}/to_timestamp"] = offset + duration_s
                video_offsets[key] = offset + duration_s
            rows.append(update_flat_episode_stats(new_row, per_episode_stats[ep]))
        if not rows:
            continue
        out_path = dst / "meta/episodes" / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        print(f"[META] wrote {out_path} episodes={len(rows)}")

    reencoded = args.video_mode == "reencode"
    if reencoded:
        video_frame_indices = {
            rel: np.asarray(indices, dtype=np.int64) for rel, indices in video_frame_indices_by_video_rel.items()
        }
        reencode_videos(src, dst, args.fps, video_frame_indices)
    all_df = write_info_and_stats(dst, src_info, stats, args.fps, reencoded)
    write_provenance(dst, args.src, src_info, args.fps, episode_filter)

    action = np.stack(all_df["action"].to_numpy())
    print(f"\n[DONE] wrote {args.fps:g}Hz dataset: {dst}")
    print(f"[CHECK] source_episodes_selected={len(episode_map)}")
    print(f"[CHECK] frames={len(all_df)} episodes={all_df['episode_index'].nunique()}")
    print(f"[CHECK] action dxyz m min={action[:, :3].min(axis=0)} max={action[:, :3].max(axis=0)}")
    print(f"[CHECK] action drot rad min={action[:, 3:6].min(axis=0)} max={action[:, 3:6].max(axis=0)}")
    print(f"[CHECK] action gripper min={action[:, 6].min():.1f} max={action[:, 6].max():.1f}")
    if not reencoded:
        print("[CHECK] --video-mode skip: no videos were written; the dataset is not trainable as is.")


if __name__ == "__main__":
    main()
