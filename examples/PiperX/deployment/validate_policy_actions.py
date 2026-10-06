#!/usr/bin/env python3
"""Compare a running StairVLA policy server against LeRobot EE dataset actions.

This is an offline validation tool: it does not connect to PiperX. It sends
dataset observations to the same websocket policy path used by
eval_policy.py, then compares the returned delta-EE action chunk with
the dataset ground-truth action chunk.
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.PiperX.deployment.eval_policy import (
    NORMALIZATION_MODES,
    extract_action_chunk,
    load_policy_stats,
    normalize_state,
    reset_policy_cache,
)

ACTION_NAMES = ("dx_m", "dy_m", "dz_m", "drx_rad", "dry_rad", "drz_rad", "grip_mm")


def _load_tasks(dataset_root: Path) -> dict[int, str]:
    tasks = pd.read_parquet(dataset_root / "meta" / "tasks.parquet")
    if "task" not in tasks.columns and "task_index" in tasks.columns:
        return {int(row["task_index"]): str(index) for index, row in tasks.iterrows()}
    return {int(row["task_index"]): str(row["task"]) for _, row in tasks.iterrows()}


def _load_episodes(dataset_root: Path) -> dict[int, dict[str, Any]]:
    episodes = {}
    for path in sorted((dataset_root / "meta" / "episodes").glob("*/*.parquet")):
        frame = pd.read_parquet(path)
        for _, row in frame.iterrows():
            item = row.to_dict()
            episode_index = int(item["episode_index"])
            episodes[episode_index] = item
    return episodes


def _load_data(dataset_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted((dataset_root / "data").glob("*/*.parquet")):
        frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(f"No parquet data found under {dataset_root / 'data'}")
    return pd.concat(frames, ignore_index=True)


def _read_video_frame(video_path: Path, timestamp_s: float, backend: str) -> np.ndarray:
    from starVLA.dataloader.gr00t_lerobot.video import get_frames_by_timestamps

    frames = get_frames_by_timestamps(
        str(video_path),
        np.asarray([max(0.0, float(timestamp_s))], dtype=np.float32),
        video_backend=backend,
    )
    if len(frames) == 0:
        raise RuntimeError(f"Failed to read frame at {timestamp_s:.3f}s from {video_path}")
    return np.asarray(frames[0])


def _images_for_row(
    dataset_root: Path,
    row: pd.Series,
    episodes: dict[int, dict[str, Any]],
    image_order: list[str],
    video_backend: str,
) -> list[np.ndarray]:
    ep = episodes[int(row["episode_index"])]
    images = []
    for name in image_order:
        video_key = f"observation.images.{name}"
        chunk_index = int(ep[f"videos/{video_key}/chunk_index"])
        file_index = int(ep[f"videos/{video_key}/file_index"])
        from_timestamp = float(ep.get(f"videos/{video_key}/from_timestamp", 0.0))
        video_path = dataset_root / "videos" / video_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
        images.append(_read_video_frame(video_path, float(row["timestamp"]) + from_timestamp, video_backend))
    return images


def _action_horizon(data: pd.DataFrame, row: pd.Series, horizon: int) -> np.ndarray:
    """Ground-truth action chunk starting at ``row``, padded with the episode's last action."""
    ep_data = data[data["episode_index"] == int(row["episode_index"])].sort_values("frame_index")
    by_frame = {int(r["frame_index"]): np.asarray(r["action"], dtype=np.float32) for _, r in ep_data.iterrows()}
    start = int(row["frame_index"])
    last = by_frame[max(by_frame)]
    out = []
    for offset in range(horizon):
        out.append(by_frame.get(start + offset, last))
    return np.stack(out, axis=0)


def dataset_fps(dataset_root: Path) -> float:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return 8.0
    with info_path.open("r") as f:
        info = json.load(f)
    return float(info.get("fps", 8.0))


def parse_episode_range(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def select_rows(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    data = data.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    if args.episodes:
        episode_ids = parse_episode_range(args.episodes)
        rows = []
        for ep in episode_ids:
            ep_data = data[data["episode_index"] == ep]
            if ep_data.empty:
                print(f"[VALIDATE] warning: episode {ep} not found")
                continue
            ep_data = ep_data[ep_data["frame_index"] >= args.min_frame]
            if ep_data.empty:
                continue
            count = min(args.frames_per_episode, len(ep_data))
            if args.evenly_spaced:
                positions = np.linspace(0, len(ep_data) - 1, count).round().astype(int)
            else:
                positions = rng.choice(len(ep_data), size=count, replace=False)
            rows.append(ep_data.iloc[positions])
        if not rows:
            raise RuntimeError("No rows selected. Check --episodes/--min-frame.")
        return pd.concat(rows, ignore_index=True)

    candidates = data[data["frame_index"] >= args.min_frame]
    if candidates.empty:
        raise RuntimeError("No rows selected. Check --min-frame.")
    count = min(args.samples, len(candidates))
    return candidates.iloc[rng.choice(len(candidates), size=count, replace=False)].reset_index(drop=True)


def query_policy(
    client: WebsocketClientPolicy,
    images: list[np.ndarray],
    instruction: str,
    state: np.ndarray,
    state_stats: dict[str, np.ndarray] | None,
    action_stats: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    response = client.predict_action(
        {
            "type": "infer",
            "request_id": str(uuid.uuid4()),
            "examples": [
                {
                    "image": images,
                    "lang": instruction,
                    "state": normalize_state(state, state_stats, args.state_norm),
                }
            ],
            "do_sample": False,
            "use_ddim": args.use_ddim,
            "num_ddim_steps": args.num_ddim_steps,
        }
    )
    raw_chunk = raw_action_chunk(response)
    action_chunk = extract_action_chunk(
        response, args.action_input, action_stats, debug=args.debug_actions, norm_mode=args.action_norm
    )
    return action_chunk, raw_chunk


def raw_action_chunk(response: dict[str, Any]) -> np.ndarray:
    data = response.get("data", response)
    for key in ("normalized_actions", "actions", "action"):
        if key in data:
            chunk = np.asarray(data[key], dtype=np.float32)
            break
    else:
        raise KeyError(f"Could not find action in server response keys={list(data.keys())}")
    while chunk.ndim > 2:
        chunk = chunk[0]
    if chunk.ndim == 1:
        chunk = chunk[None, :]
    return chunk[:, :7]


def normalize_action_chunk(
    chunk: np.ndarray,
    stats: dict[str, np.ndarray],
    norm_mode: str = "q99",
) -> np.ndarray:
    """Forward action normalization, matching the training data config.

    This must use the same mode as the loader that produced the checkpoint
    (starVLA/dataloader/gr00t_lerobot/transform/state_action.py::Normalizer).
    Comparing against targets normalized with the wrong mode makes the
    normalized-space metrics meaningless.
    """
    if norm_mode == "q99":
        lows, highs, clip = stats["q01"], stats["q99"], True
    elif norm_mode == "min_max":
        lows, highs, clip = stats["min"], stats["max"], False
    else:
        raise ValueError(f"Unknown normalization mode {norm_mode!r}")
    chunk = np.asarray(chunk, dtype=np.float32)
    denom = np.where(highs == lows, 1.0, highs - lows)
    out = 2.0 * (chunk - lows) / denom - 1.0
    return np.clip(out, -1.0, 1.0) if clip else out


def state_for_policy(row: pd.Series, policy_state_space: str) -> np.ndarray:
    state = np.asarray(row["observation.state"], dtype=np.float32)
    if policy_state_space == "ee7":
        return state[:7]
    if policy_state_space != "ee":
        raise ValueError("This validator expects an EE dataset, so use --policy-state-space ee or ee7.")
    return state


def chunk_metrics(pred: np.ndarray, gt: np.ndarray, fps: float) -> dict[str, float]:
    n = min(len(pred), len(gt))
    pred = np.asarray(pred[:n], dtype=np.float64)
    gt = np.asarray(gt[:n], dtype=np.float64)
    diff = pred - gt

    xyz_rmse_m = float(np.sqrt(np.mean(diff[:, :3] ** 2)))
    rot_rmse_rad = float(np.sqrt(np.mean(diff[:, 3:6] ** 2)))
    grip_rmse_mm = float(np.sqrt(np.mean(diff[:, 6] ** 2)))
    first = diff[0]

    gt_norm = np.linalg.norm(gt[:, :3], axis=1)
    pred_norm = np.linalg.norm(pred[:, :3], axis=1)
    valid = (gt_norm > 1e-5) & (pred_norm > 1e-5)
    if np.any(valid):
        cos = np.sum(gt[valid, :3] * pred[valid, :3], axis=1) / (gt_norm[valid] * pred_norm[valid])
        trans_dir_cos = float(np.mean(np.clip(cos, -1.0, 1.0)))
    else:
        trans_dir_cos = float("nan")

    return {
        "steps": float(n),
        "action_mse_mixed": float(np.mean(diff**2)),
        "xyz_rmse_mm": xyz_rmse_m * 1000.0,
        "rot_rmse_deg": np.degrees(rot_rmse_rad),
        "grip_rmse_mm": grip_rmse_mm,
        "xyz_vel_rmse_mm_s": xyz_rmse_m * fps * 1000.0,
        "rot_vel_rmse_deg_s": np.degrees(rot_rmse_rad) * fps,
        "first_xyz_err_mm": float(np.linalg.norm(first[:3]) * 1000.0),
        "first_rot_err_deg": float(np.degrees(np.linalg.norm(first[3:6]))),
        "first_grip_err_mm": float(abs(first[6])),
        "trans_dir_cos": trans_dir_cos,
    }


def normalized_metrics(
    pred_norm: np.ndarray,
    gt: np.ndarray,
    action_stats: dict[str, np.ndarray],
    fps: float,
    norm_mode: str = "q99",
) -> dict[str, float]:
    gt_norm = normalize_action_chunk(gt, action_stats, norm_mode)
    n = min(len(pred_norm), len(gt_norm))
    diff = np.asarray(pred_norm[:n], dtype=np.float64) - np.asarray(gt_norm[:n], dtype=np.float64)
    return {
        "mse_score_like": float(np.mean(diff**2)),
        "norm_mse_all": float(np.mean(diff**2)),
        "norm_mse_xyz": float(np.mean(diff[:, :3] ** 2)),
        "norm_mse_rot": float(np.mean(diff[:, 3:6] ** 2)),
        "norm_mse_grip": float(np.mean(diff[:, 6] ** 2)),
        "norm_vel_rmse_all_per_s": float(np.sqrt(np.mean(diff**2)) * fps),
        "norm_vel_rmse_xyz_per_s": float(np.sqrt(np.mean(diff[:, :3] ** 2)) * fps),
        "norm_vel_rmse_rot_per_s": float(np.sqrt(np.mean(diff[:, 3:6] ** 2)) * fps),
        "norm_vel_rmse_grip_per_s": float(np.sqrt(np.mean(diff[:, 6] ** 2)) * fps),
        "norm_first_l2": float(np.linalg.norm(diff[0])),
        "norm_abs_max": float(np.max(np.abs(diff))),
    }


def fmt_action(vec: np.ndarray) -> str:
    return " ".join(f"{name}={value: .4f}" for name, value in zip(ACTION_NAMES, vec, strict=True))


def print_metric_summary(title: str, rows: list[dict[str, Any]], metric_names: list[str]) -> None:
    print(f"\n[{title}] n={len(rows)}")
    for name in metric_names:
        values = np.asarray([r[name] for r in rows], dtype=np.float64)
        print(
            f"  {name}: "
            f"mean={np.nanmean(values):.4f} "
            f"median={np.nanmedian(values):.4f} "
            f"p90={np.nanpercentile(values, 90):.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a running StairVLA policy server's action chunks with a PiperX EE dataset (no robot)."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--dataset-root", type=Path, required=True, help="LeRobot v3.0 EE dataset root.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint .pt file or run directory; dataset_statistics.json is found by walking up its parents.",
    )
    parser.add_argument("--stats-json", type=Path, default=None, help="Explicit dataset_statistics.json (overrides --checkpoint).")
    parser.add_argument("--state-stats-json", type=Path, default=None)
    parser.add_argument("--episodes", default="", help="Episode ids/ranges, e.g. '0,15,100-105'.")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--frames-per-episode", type=int, default=8)
    parser.add_argument("--min-frame", type=int, default=3)
    parser.add_argument("--evenly-spaced", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-state-space", choices=["ee", "ee7"], default="ee")
    parser.add_argument("--image-order", nargs="+", default=["top", "wrist"])
    parser.add_argument(
        "--video-backend",
        choices=["torchvision_av", "pyav", "decord", "torchcodec", "opencv"],
        default="torchvision_av",
    )
    parser.add_argument("--action-input", choices=["auto", "normalized", "absolute"], default="normalized")
    # Must match the training data config (q99 for the PiperX EE configs).
    parser.add_argument("--action-norm", choices=list(NORMALIZATION_MODES), default="q99")
    parser.add_argument("--state-norm", choices=list(NORMALIZATION_MODES), default="q99")
    parser.add_argument("--use-ddim", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-ddim-steps", type=int, default=10)
    parser.add_argument("--reset-cache-per-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-samples", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--debug-actions", action="store_true")
    parser.add_argument(
        "--out-csv", type=Path, default=Path("outputs/piperx_validate/policy_dataset_action_validation.csv")
    )
    args = parser.parse_args()
    # load_policy_stats validates the state space against the stats; this tool only replays EE datasets.
    args.action_space = "delta-ee"
    action_stats, state_stats = load_policy_stats(args, "[VALIDATE]")

    tasks = _load_tasks(args.dataset_root)
    episodes = _load_episodes(args.dataset_root)
    data = _load_data(args.dataset_root)
    fps = dataset_fps(args.dataset_root)
    picked = select_rows(data, args)

    print(f"[VALIDATE] dataset={args.dataset_root}")
    print(f"[VALIDATE] dataset_fps={fps:.3f} samples={len(picked)} horizon={args.horizon}")
    print(f"[VALIDATE] image_order={args.image_order} state_space={args.policy_state_space} action_input={args.action_input}")
    print(f"[VALIDATE] normalization: action={args.action_norm} state={args.state_norm}")
    print(f"[VALIDATE] state_dim={len(np.asarray(picked.iloc[0]['observation.state']))} action_dim={len(np.asarray(picked.iloc[0]['action']))}")

    metric_names = [
        "mse_score_like",
        "norm_mse_all",
        "norm_mse_xyz",
        "norm_mse_rot",
        "norm_mse_grip",
        "norm_vel_rmse_all_per_s",
        "norm_vel_rmse_xyz_per_s",
        "norm_vel_rmse_rot_per_s",
        "norm_vel_rmse_grip_per_s",
        "xyz_rmse_mm",
        "rot_rmse_deg",
        "grip_rmse_mm",
        "xyz_vel_rmse_mm_s",
        "rot_vel_rmse_deg_s",
        "first_xyz_err_mm",
        "first_rot_err_deg",
        "first_grip_err_mm",
        "trans_dir_cos",
        "action_mse_mixed",
        "norm_first_l2",
        "norm_abs_max",
    ]
    rows: list[dict[str, Any]] = []
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    try:
        for i, row in picked.iterrows():
            ep = int(row["episode_index"])
            frame = int(row["frame_index"])
            task = tasks[int(row["task_index"])]
            if args.reset_cache_per_sample:
                with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                    reset_policy_cache(client, "dataset_validation_sample", int(i), task)
            state = state_for_policy(row, args.policy_state_space)
            images = _images_for_row(args.dataset_root, row, episodes, args.image_order, args.video_backend)
            gt = _action_horizon(data, row, args.horizon)
            pred, pred_norm = query_policy(client, images, task, state, state_stats, action_stats, args)
            metrics = chunk_metrics(pred, gt, fps)
            metrics.update(normalized_metrics(pred_norm, gt, action_stats, fps, args.action_norm))
            record = {"sample": int(i), "episode": ep, "frame": frame, "task": task, **metrics}
            rows.append(record)

            if i < args.print_samples:
                print(f"\n[SAMPLE {i}] ep={ep} frame={frame} task={task}")
                print(f"  gt[0]   {fmt_action(gt[0])}")
                print(f"  pred[0] {fmt_action(pred[0])}")
                print(
                    "  errH    "
                    f"xyz={metrics['xyz_rmse_mm']:.1f}mm "
                    f"rot={metrics['rot_rmse_deg']:.2f}deg "
                    f"grip={metrics['grip_rmse_mm']:.1f}mm "
                    f"vel={metrics['xyz_vel_rmse_mm_s']:.1f}mm/s "
                    f"dir_cos={metrics['trans_dir_cos']:.3f} "
                    f"norm_mse={metrics['norm_mse_all']:.4f}"
                )
            elif args.progress_every > 0 and (i + 1) % args.progress_every == 0:
                print_metric_summary(
                    f"PROGRESS {i + 1}/{len(picked)}",
                    rows,
                    [
                        "norm_mse_all",
                        "norm_mse_xyz",
                        "norm_mse_rot",
                        "norm_mse_grip",
                        "norm_vel_rmse_all_per_s",
                        "norm_vel_rmse_xyz_per_s",
                        "norm_vel_rmse_rot_per_s",
                        "norm_vel_rmse_grip_per_s",
                        "xyz_rmse_mm",
                        "rot_rmse_deg",
                        "grip_rmse_mm",
                        "xyz_vel_rmse_mm_s",
                        "rot_vel_rmse_deg_s",
                        "trans_dir_cos",
                    ],
                )
    finally:
        client.close()

    print_metric_summary("SUMMARY", rows, metric_names)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[VALIDATE] wrote {args.out_csv}")


if __name__ == "__main__":
    main()
