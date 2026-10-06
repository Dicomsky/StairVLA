#!/usr/bin/env python3
"""Interactively inspect PiperX EE-delta LeRobot episodes and export matching videos.

This is an offline viewer: it never connects to or commands the robot. For each episode it
writes the EE state/action sequence as CSV, cuts the wrist/top clips from the dataset video
shards, renders an action-replay video (XY/XZ trajectory, per-step deltas, gripper target) and a
side-by-side wrist | top | action-replay video. Use it to review episodes flagged by
check_quality.py. Pass --episode to inspect a list and exit, or type ids at the prompt.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import cv2
from scipy.spatial.transform import Rotation as R

STATE_NAMES = ["eef.x_m", "eef.y_m", "eef.z_m", "eef.qx", "eef.qy", "eef.qz", "eef.qw", "gripper.pos_mm"]
ACTION_NAMES = [
    "eef.dx_m",
    "eef.dy_m",
    "eef.dz_m",
    "eef.drx_rad",
    "eef.dry_rad",
    "eef.drz_rad",
    "gripper.target_mm",
]


def load_info(dataset: Path) -> dict[str, Any]:
    info_path = dataset / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing dataset info: {info_path}")
    return json.loads(info_path.read_text())


def load_episode_meta(dataset: Path) -> pd.DataFrame:
    paths = sorted((dataset / "meta" / "episodes").glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata found under {dataset / 'meta' / 'episodes'}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True).sort_values("episode_index")


def load_episode_frames(dataset: Path, meta_row: pd.Series) -> pd.DataFrame:
    chunk = int(meta_row["data/chunk_index"])
    file_index = int(meta_row["data/file_index"])
    start = int(meta_row["dataset_from_index"])
    end = int(meta_row["dataset_to_index"])
    data_path = dataset / "data" / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data shard: {data_path}")
    shard = pd.read_parquet(data_path)
    if "index" in shard.columns:
        ep = shard[(shard["index"] >= start) & (shard["index"] < end)].copy()
    else:
        episode = int(meta_row["episode_index"])
        ep = shard[shard["episode_index"] == episode].copy()
    if ep.empty:
        raise ValueError(f"Episode {int(meta_row['episode_index'])} has no frames in {data_path}")
    return ep.sort_values("frame_index").reset_index(drop=True)


def row_task(row: pd.Series) -> str:
    tasks = row.get("tasks", "")
    if isinstance(tasks, np.ndarray):
        tasks = tasks.tolist()
    if isinstance(tasks, list):
        return str(tasks[0]) if tasks else ""
    return str(tasks)


def quat_to_euler_deg(quat_xyzw: np.ndarray) -> np.ndarray:
    return R.from_quat(quat_xyzw).as_euler("xyz", degrees=True)


def action_stats(actions: np.ndarray) -> dict[str, Any]:
    xyz_norm = np.linalg.norm(actions[:, :3], axis=1)
    rot_norm = np.linalg.norm(actions[:, 3:6], axis=1)
    return {
        "frames": int(actions.shape[0]),
        "max_xyz_delta_m": float(xyz_norm.max(initial=0.0)),
        "mean_xyz_delta_m": float(xyz_norm.mean() if len(xyz_norm) else 0.0),
        "max_rot_delta_deg": float(np.rad2deg(rot_norm.max(initial=0.0))),
        "mean_rot_delta_deg": float(np.rad2deg(rot_norm.mean() if len(rot_norm) else 0.0)),
        "gripper_min_mm": float(np.min(actions[:, 6])) if actions.size else 0.0,
        "gripper_max_mm": float(np.max(actions[:, 6])) if actions.size else 0.0,
    }


def write_action_csv(ep: pd.DataFrame, out_csv: Path) -> None:
    states = np.stack(ep["observation.state"].to_numpy()).astype(np.float64)
    actions = np.stack(ep["action"].to_numpy()).astype(np.float64)
    euler = np.stack([quat_to_euler_deg(state[3:7]) for state in states])

    rows: dict[str, Any] = {
        "frame_index": ep["frame_index"].to_numpy(),
        "timestamp": ep["timestamp"].to_numpy(),
    }
    for i, name in enumerate(STATE_NAMES):
        rows[f"state.{name}"] = states[:, i]
    rows["state.euler_rx_deg"] = euler[:, 0]
    rows["state.euler_ry_deg"] = euler[:, 1]
    rows["state.euler_rz_deg"] = euler[:, 2]
    for i, name in enumerate(ACTION_NAMES):
        rows[f"action.{name}"] = actions[:, i]
    rows["action.xyz_norm_m"] = np.linalg.norm(actions[:, :3], axis=1)
    rows["action.rot_norm_deg"] = np.rad2deg(np.linalg.norm(actions[:, 3:6], axis=1))
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def video_path(dataset: Path, camera_key: str, chunk: int, file_index: int) -> Path:
    return dataset / "videos" / camera_key / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"


def ffmpeg_cut(src: Path, dst: Path, start_s: float, end_s: float, reencode: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing video shard: {src}")
    duration = max(0.001, end_s - start_s)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start_s:.6f}", "-i", str(src), "-t", f"{duration:.6f}"]
    if reencode:
        cmd += ["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-an", "-c", "copy"]
    cmd.append(str(dst))
    subprocess.run(cmd, check=True)


def has_video_stream(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                str(path),
            ],
            text=True,
        ).strip()
    except Exception:
        return False
    return out == "video"


def make_side_by_side_video(video_paths: list[Path], output_dir: Path, episode: int) -> Path | None:
    video_paths = [path for path in video_paths if has_video_stream(path)]
    if len(video_paths) < 2:
        return None
    by_name = {path.stem.rsplit("_", 1)[-1]: path for path in video_paths}
    left = by_name.get("wrist", video_paths[0])
    right = by_name.get("top", video_paths[1])
    dst = output_dir / f"episode_{episode:04d}_wrist_top.mp4"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(left),
        "-i",
        str(right),
        "-filter_complex",
        "[0:v]scale=640:480,setsar=1[left];[1:v]scale=640:480,setsar=1[right];[left][right]hstack=inputs=2[v]",
        "-map",
        "[v]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    return dst


def _project(values: np.ndarray, vmin: float, vmax: float, low: int, high: int) -> np.ndarray:
    span = max(vmax - vmin, 1e-9)
    return low + (values - vmin) / span * (high - low)


def _draw_polyline(img: np.ndarray, points: np.ndarray, color: tuple[int, int, int], thickness: int = 2) -> None:
    if len(points) < 2:
        return
    pts = np.round(points).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def _put(img: np.ndarray, text: str, xy: tuple[int, int], scale: float = 0.48, color=(235, 235, 235)) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def make_action_replay_video(ep: pd.DataFrame, output_dir: Path, episode: int, fps: float) -> Path:
    states = np.stack(ep["observation.state"].to_numpy()).astype(np.float64)
    actions = np.stack(ep["action"].to_numpy()).astype(np.float64)
    xyz = states[:, :3]
    eulers = np.stack([quat_to_euler_deg(state[3:7]) for state in states])
    dst = output_dir / f"episode_{episode:04d}_action_replay.mp4"

    width, height = 640, 480
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create action replay video: {dst}")

    pad = np.array([0.03, 0.03, 0.03])
    mins = xyz.min(axis=0) - pad
    maxs = xyz.max(axis=0) + pad
    grip = actions[:, 6]
    grip_min = min(0.0, float(grip.min(initial=0.0)))
    grip_max = max(100.0, float(grip.max(initial=100.0)))

    xy_pts = np.column_stack([
        _project(xyz[:, 0], mins[0], maxs[0], 56, 282),
        _project(xyz[:, 1], mins[1], maxs[1], 206, 58),
    ])
    xz_pts = np.column_stack([
        _project(xyz[:, 0], mins[0], maxs[0], 358, 584),
        _project(xyz[:, 2], mins[2], maxs[2], 206, 58),
    ])
    grip_x = _project(np.arange(len(grip)), 0, max(len(grip) - 1, 1), 56, 584)
    grip_y = _project(grip, grip_min, grip_max, 418, 278)
    grip_pts = np.column_stack([grip_x, grip_y])

    for i in range(len(ep)):
        img = np.full((height, width, 3), 24, dtype=np.uint8)
        cv2.rectangle(img, (34, 34), (304, 226), (70, 70, 70), 1)
        cv2.rectangle(img, (336, 34), (606, 226), (70, 70, 70), 1)
        cv2.rectangle(img, (34, 250), (606, 438), (70, 70, 70), 1)
        _put(img, f"Episode {episode:04d} action replay", (24, 24), 0.55, (255, 255, 255))
        _put(img, "XY top view", (44, 54), 0.42, (180, 210, 255))
        _put(img, "XZ side view", (346, 54), 0.42, (180, 210, 255))
        _put(img, "gripper target", (44, 270), 0.42, (180, 210, 255))

        _draw_polyline(img, xy_pts[: i + 1], (70, 190, 255), 2)
        _draw_polyline(img, xz_pts[: i + 1], (90, 220, 120), 2)
        _draw_polyline(img, grip_pts[: i + 1], (230, 190, 80), 2)

        for pts, action_xy in [
            (xy_pts, actions[i, [0, 1]]),
            (xz_pts, actions[i, [0, 2]]),
        ]:
            p = tuple(np.round(pts[i]).astype(int))
            cv2.circle(img, p, 5, (255, 255, 255), -1, cv2.LINE_AA)
            scale = 900.0
            q = (int(p[0] + action_xy[0] * scale), int(p[1] - action_xy[1] * scale))
            cv2.arrowedLine(img, p, q, (255, 120, 80), 2, cv2.LINE_AA, tipLength=0.25)

        gp = tuple(np.round(grip_pts[i]).astype(int))
        cv2.circle(img, gp, 4, (255, 255, 255), -1, cv2.LINE_AA)

        state = states[i]
        action = actions[i]
        _put(img, f"frame {i:04d}/{len(ep)-1:04d}  t={float(ep['timestamp'].iloc[i]):.3f}s", (42, 462), 0.46)
        _put(img, f"xyz m: {state[0]: .3f} {state[1]: .3f} {state[2]: .3f}", (322, 272), 0.43)
        _put(img, f"rpy deg: {eulers[i,0]: .1f} {eulers[i,1]: .1f} {eulers[i,2]: .1f}", (322, 294), 0.43)
        _put(img, f"dxyz mm: {action[0]*1000: .1f} {action[1]*1000: .1f} {action[2]*1000: .1f}", (322, 322), 0.43)
        _put(img, f"drot deg: {np.rad2deg(action[3]): .1f} {np.rad2deg(action[4]): .1f} {np.rad2deg(action[5]): .1f}", (322, 344), 0.43)
        _put(img, f"gripper target: {action[6]:.1f} mm", (322, 372), 0.46, (250, 220, 120))
        writer.write(img)

    writer.release()
    return dst


def make_full_replay_video(video_paths: list[Path], action_video: Path, output_dir: Path, episode: int) -> Path | None:
    video_paths = [path for path in video_paths if has_video_stream(path)]
    if len(video_paths) < 2 or not has_video_stream(action_video):
        return None
    by_name = {path.stem.rsplit("_", 1)[-1]: path for path in video_paths}
    wrist = by_name.get("wrist", video_paths[0])
    top = by_name.get("top", video_paths[1])
    dst = output_dir / f"episode_{episode:04d}_full_replay.mp4"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(wrist),
        "-i",
        str(top),
        "-i",
        str(action_video),
        "-filter_complex",
        "[0:v]scale=640:480,setsar=1[w];"
        "[1:v]scale=640:480,setsar=1[t];"
        "[2:v]scale=640:480,setsar=1[a];"
        "[w][t][a]hstack=inputs=3[v]",
        "-map",
        "[v]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    return dst


def play_video(path: Path) -> None:
    ffplay = shutil.which("ffplay")
    if ffplay:
        subprocess.run([ffplay, "-hide_banner", "-loglevel", "error", "-autoexit", str(path)], check=False)
        return
    opener = shutil.which("xdg-open")
    if opener:
        subprocess.Popen([opener, str(path)])
        return
    print(f"[REPLAY_VIEW] No ffplay/xdg-open found. Open manually: {path}")


def replay_actions_in_terminal(ep: pd.DataFrame, fps: float) -> None:
    states = np.stack(ep["observation.state"].to_numpy()).astype(np.float64)
    actions = np.stack(ep["action"].to_numpy()).astype(np.float64)
    dt = 1.0 / max(fps, 1e-6)
    print("[REPLAY_VIEW] terminal action replay starts. Ctrl+C skips this terminal replay.")
    try:
        for i, (state, action) in enumerate(zip(states, actions)):
            euler = quat_to_euler_deg(state[3:7])
            print(
                "\r"
                f"frame={i:04d}/{len(ep)-1:04d} "
                f"xyz=({state[0]:.3f},{state[1]:.3f},{state[2]:.3f})m "
                f"rpy=({euler[0]:6.1f},{euler[1]:6.1f},{euler[2]:6.1f})deg "
                f"dxyz=({action[0]*1000:6.1f},{action[1]*1000:6.1f},{action[2]*1000:6.1f})mm "
                f"drot={np.rad2deg(np.linalg.norm(action[3:6])):5.1f}deg "
                f"grip={action[6]:5.1f}mm",
                end="",
                flush=True,
            )
            time.sleep(dt)
        print()
    except KeyboardInterrupt:
        print("\n[REPLAY_VIEW] terminal action replay skipped.")


def export_videos(dataset: Path, row: pd.Series, output_dir: Path, episode: int, reencode: bool) -> list[Path]:
    outputs = []
    for camera_key in ["observation.images.wrist", "observation.images.top"]:
        chunk_key = f"videos/{camera_key}/chunk_index"
        file_key = f"videos/{camera_key}/file_index"
        from_key = f"videos/{camera_key}/from_timestamp"
        to_key = f"videos/{camera_key}/to_timestamp"
        if chunk_key not in row or file_key not in row:
            continue
        chunk = int(row[chunk_key])
        file_index = int(row[file_key])
        start_s = float(row[from_key])
        end_s = float(row[to_key])
        src = video_path(dataset, camera_key, chunk, file_index)
        dst = output_dir / f"episode_{episode:04d}_{camera_key.split('.')[-1]}.mp4"
        ffmpeg_cut(src, dst, start_s, end_s, reencode=reencode)
        outputs.append(dst)
    return outputs


def inspect_episode(
    dataset: Path,
    meta: pd.DataFrame,
    episode: int,
    output_dir: Path,
    reencode: bool,
    play: bool,
    terminal_replay: bool,
    fps: float,
) -> None:
    matches = meta[meta["episode_index"] == episode]
    if matches.empty:
        first = int(meta["episode_index"].min())
        last = int(meta["episode_index"].max())
        raise ValueError(f"Episode {episode} not found. Available range is roughly {first}..{last}.")
    row = matches.iloc[0]
    ep = load_episode_frames(dataset, row)
    states = np.stack(ep["observation.state"].to_numpy()).astype(np.float64)
    actions = np.stack(ep["action"].to_numpy()).astype(np.float64)
    stats = action_stats(actions)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_csv = output_dir / f"episode_{episode:04d}_actions.csv"
    write_action_csv(ep, out_csv)
    videos = export_videos(dataset, row, output_dir, episode, reencode=reencode)
    side_by_side = make_side_by_side_video(videos, output_dir, episode)
    action_video = make_action_replay_video(ep, output_dir, episode, fps=fps)
    full_replay = make_full_replay_video(videos, action_video, output_dir, episode)

    first_euler = quat_to_euler_deg(states[0, 3:7])
    last_euler = quat_to_euler_deg(states[-1, 3:7])
    summary = [
        f"dataset: {dataset}",
        f"episode: {episode}",
        f"task: {row_task(row)}",
        f"frames: {len(ep)}",
        f"time: {float(ep['timestamp'].iloc[0]):.3f}s -> {float(ep['timestamp'].iloc[-1]):.3f}s",
        "first state: "
        f"xyz=({states[0,0]:.4f}, {states[0,1]:.4f}, {states[0,2]:.4f}) m "
        f"euler=({first_euler[0]:.1f}, {first_euler[1]:.1f}, {first_euler[2]:.1f}) deg "
        f"gripper={states[0,7]:.1f} mm",
        "last state : "
        f"xyz=({states[-1,0]:.4f}, {states[-1,1]:.4f}, {states[-1,2]:.4f}) m "
        f"euler=({last_euler[0]:.1f}, {last_euler[1]:.1f}, {last_euler[2]:.1f}) deg "
        f"gripper={states[-1,7]:.1f} mm",
        "action stats: "
        f"max_xyz={stats['max_xyz_delta_m'] * 1000:.1f} mm, "
        f"mean_xyz={stats['mean_xyz_delta_m'] * 1000:.1f} mm, "
        f"max_rot={stats['max_rot_delta_deg']:.1f} deg, "
        f"mean_rot={stats['mean_rot_delta_deg']:.1f} deg, "
        f"gripper={stats['gripper_min_mm']:.1f}..{stats['gripper_max_mm']:.1f} mm",
        f"actions_csv: {out_csv}",
    ]
    summary += [f"video: {path}" for path in videos]
    summary.append(f"action_replay_video: {action_video}")
    if side_by_side:
        summary.append(f"replay_video: {side_by_side}")
    if full_replay:
        summary.append(f"full_replay_video: {full_replay}")
    out_summary = output_dir / f"episode_{episode:04d}_summary.txt"
    out_summary.write_text("\n".join(summary) + "\n")

    print("\n".join("[REPLAY_VIEW] " + line for line in summary))
    print(f"[REPLAY_VIEW] summary_txt: {out_summary}")
    if terminal_replay:
        replay_actions_in_terminal(ep, fps=fps)
    replay_path = full_replay or side_by_side or action_video
    if play and replay_path:
        print(f"[REPLAY_VIEW] playing: {replay_path}")
        play_video(replay_path)


def parse_episode_list(text: str) -> list[int]:
    episodes: list[int] = []
    for part in text.replace(",", " ").split():
        if "-" in part:
            lo, hi = part.split("-", 1)
            episodes.extend(range(int(lo), int(hi) + 1))
        else:
            episodes.append(int(part))
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline viewer for PiperX EE LeRobot episodes.")
    parser.add_argument("--dataset", type=Path, required=True, help="EE dataset directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where CSVs and videos are written.")
    parser.add_argument("--episode", type=int, nargs="*", default=None, help="Inspect these episodes then exit.")
    parser.add_argument("--copy-video", action="store_true", help="Fast cut with stream copy instead of re-encoding.")
    parser.add_argument("--no-play", action="store_true", help="Do not automatically play the side-by-side replay video.")
    parser.add_argument("--terminal-replay", action="store_true", help="Replay action/state values in the terminal too.")
    parser.add_argument("--terminal-fps", type=float, default=None, help="Terminal action replay fps. Defaults to dataset fps.")
    args = parser.parse_args()

    info = load_info(args.dataset)
    meta = load_episode_meta(args.dataset)
    print(f"[REPLAY_VIEW] dataset={args.dataset}")
    print(f"[REPLAY_VIEW] episodes={len(meta)} total_frames={info.get('total_frames')} fps={info.get('fps')}")
    print(f"[REPLAY_VIEW] output_dir={args.output_dir}")
    play = not args.no_play
    terminal_fps = float(args.terminal_fps or info.get("fps") or 8.0)

    if args.episode:
        for episode in args.episode:
            inspect_episode(
                args.dataset,
                meta,
                episode,
                args.output_dir,
                reencode=not args.copy_video,
                play=play,
                terminal_replay=args.terminal_replay,
                fps=terminal_fps,
            )
        return

    print("[REPLAY_VIEW] Enter episode ids, e.g. 0, 34, 3 7 9 or 100-105 (inclusive). Enter q to quit.")
    while True:
        try:
            text = input("episode> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[REPLAY_VIEW] bye")
            return
        if text.lower() in {"q", "quit", "exit"}:
            print("[REPLAY_VIEW] bye")
            return
        if not text:
            continue
        try:
            for episode in parse_episode_list(text):
                inspect_episode(
                    args.dataset,
                    meta,
                    episode,
                    args.output_dir,
                    reencode=not args.copy_video,
                    play=play,
                    terminal_replay=args.terminal_replay,
                    fps=terminal_fps,
                )
        except Exception as exc:
            print(f"[REPLAY_VIEW] ERROR: {exc}")


if __name__ == "__main__":
    main()
