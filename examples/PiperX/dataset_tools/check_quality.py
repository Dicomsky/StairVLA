#!/usr/bin/env python3
"""Quality-check a PiperX EE-delta LeRobot dataset episode by episode.

The script is read-only for the dataset. It computes per-episode metrics, writes CSV reports to
--out-dir, and proposes suspicious episode ids using robust, data-driven thresholds plus absolute
"strong" thresholds. With --write-manifest it also writes the candidate ids as an exclusion
manifest (the format read by resample.py --exclusions) for human review.

The default thresholds were tuned on the 8 Hz Fruit25 data (per-step deltas grow with lower fps).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from examples.PiperX.dataset_tools.dataset_utils import write_manifest


def load_info(dataset: Path) -> dict[str, Any]:
    path = dataset / "meta" / "info.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_episode_tasks(dataset: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    paths = sorted((dataset / "meta" / "episodes").glob("*/*.parquet"))
    for path in paths:
        meta = pd.read_parquet(path, columns=["episode_index", "tasks"])
        for _, row in meta.iterrows():
            tasks = row["tasks"]
            if isinstance(tasks, np.ndarray):
                tasks = tasks.tolist()
            if isinstance(tasks, list):
                task = str(tasks[0]) if tasks else ""
            else:
                task = str(tasks)
            out[int(row["episode_index"])] = task
    return out


def load_episode_meta(dataset: Path) -> pd.DataFrame:
    paths = sorted((dataset / "meta" / "episodes").glob("*/*.parquet"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True).sort_values("episode_index")


def load_data(dataset: Path) -> pd.DataFrame:
    frames = []
    for path in sorted((dataset / "data").glob("*/*.parquet")):
        frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(f"No parquet shards found under {dataset / 'data'}")
    df = pd.concat(frames, ignore_index=True)
    missing = {"episode_index", "frame_index", "timestamp", "observation.state", "action"} - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    return df


def robust_threshold(values: np.ndarray, q: float, mad_k: float, floor: float | None = None) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float(floor or 0.0)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    # 1.4826 makes MAD comparable to std for a normal distribution.
    robust = median + mad_k * 1.4826 * mad
    quantile = float(np.quantile(values, q))
    threshold = max(robust, quantile)
    if floor is not None:
        threshold = max(threshold, floor)
    return threshold


def safe_quat_to_rot(quats: np.ndarray) -> R:
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return R.from_quat(quats / norms)


def rotation_step_deg(quats: np.ndarray) -> np.ndarray:
    if len(quats) <= 1:
        return np.zeros(0, dtype=np.float64)
    rots = safe_quat_to_rot(quats)
    return np.rad2deg(np.linalg.norm((rots[:-1].inv() * rots[1:]).as_rotvec(), axis=1))


def action_to_future_errors(states: np.ndarray, actions: np.ndarray, horizons: list[int]) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    for horizon in horizons:
        if len(states) <= horizon:
            out[horizon] = (np.nan, np.nan)
            continue
        pred_pos = states[:-horizon, :3] + actions[:-horizon, :3]
        pos_err_mm = np.linalg.norm(pred_pos - states[horizon:, :3], axis=1) * 1000.0

        state_rot = safe_quat_to_rot(states[:-horizon, 3:7])
        pred_rot = state_rot * R.from_rotvec(actions[:-horizon, 3:6])
        future_rot = safe_quat_to_rot(states[horizon:, 3:7])
        rot_err_deg = np.rad2deg(np.linalg.norm((pred_rot.inv() * future_rot).as_rotvec(), axis=1))
        out[horizon] = (float(np.median(pos_err_mm)), float(np.median(rot_err_deg)))
    return out


def episode_metrics(ep: pd.DataFrame, task: str, horizons: list[int]) -> dict[str, Any]:
    ep = ep.sort_values("frame_index").reset_index(drop=True)
    states = np.stack(ep["observation.state"].to_numpy()).astype(np.float64)
    actions = np.stack(ep["action"].to_numpy()).astype(np.float64)
    xyz_action_mm = np.linalg.norm(actions[:, :3], axis=1) * 1000.0
    rot_action_deg = np.rad2deg(np.linalg.norm(actions[:, 3:6], axis=1))
    xyz_step_mm = np.linalg.norm(np.diff(states[:, :3], axis=0), axis=1) * 1000.0 if len(states) > 1 else np.zeros(0)
    rot_step = rotation_step_deg(states[:, 3:7])
    quat_norm = np.linalg.norm(states[:, 3:7], axis=1)
    gripper_action = actions[:, 6]
    gripper_state = states[:, 7]
    future_errors = action_to_future_errors(states, actions, horizons)

    best_horizon = min(
        future_errors,
        key=lambda h: future_errors[h][0] if np.isfinite(future_errors[h][0]) else float("inf"),
    )
    err_h1 = future_errors.get(1, (np.nan, np.nan))[0]
    err_best = future_errors[best_horizon][0]
    looks_future = bool(np.isfinite(err_h1) and np.isfinite(err_best) and best_horizon > 1 and err_best + 10.0 < err_h1)

    row: dict[str, Any] = {
        "episode": int(ep["episode_index"].iloc[0]),
        "task": task,
        "frames": int(len(ep)),
        "duration_s": float(ep["timestamp"].iloc[-1] - ep["timestamp"].iloc[0]) if len(ep) else 0.0,
        "nan_count": int(np.isnan(states).sum() + np.isnan(actions).sum()),
        "quat_norm_min": float(np.min(quat_norm)),
        "quat_norm_max": float(np.max(quat_norm)),
        "state_xyz_step_max_mm": float(np.max(xyz_step_mm)) if len(xyz_step_mm) else 0.0,
        "state_xyz_step_mean_mm": float(np.mean(xyz_step_mm)) if len(xyz_step_mm) else 0.0,
        "state_rot_step_max_deg": float(np.max(rot_step)) if len(rot_step) else 0.0,
        "state_rot_step_mean_deg": float(np.mean(rot_step)) if len(rot_step) else 0.0,
        "action_xyz_max_mm": float(np.max(xyz_action_mm)) if len(xyz_action_mm) else 0.0,
        "action_xyz_mean_mm": float(np.mean(xyz_action_mm)) if len(xyz_action_mm) else 0.0,
        "action_rot_max_deg": float(np.max(rot_action_deg)) if len(rot_action_deg) else 0.0,
        "action_rot_mean_deg": float(np.mean(rot_action_deg)) if len(rot_action_deg) else 0.0,
        "action_gripper_min_mm": float(np.min(gripper_action)) if len(gripper_action) else 0.0,
        "action_gripper_max_mm": float(np.max(gripper_action)) if len(gripper_action) else 0.0,
        "state_gripper_min_mm": float(np.min(gripper_state)) if len(gripper_state) else 0.0,
        "state_gripper_max_mm": float(np.max(gripper_state)) if len(gripper_state) else 0.0,
        "best_delta_horizon": int(best_horizon),
        "looks_like_future_delta": looks_future,
    }
    for horizon, (pos_err, rot_err) in future_errors.items():
        row[f"h{horizon}_pos_err_median_mm"] = pos_err
        row[f"h{horizon}_rot_err_median_deg"] = rot_err
    return row


def build_thresholds(metrics: pd.DataFrame, args: argparse.Namespace) -> dict[str, float]:
    return {
        "action_xyz_max_mm": args.action_xyz_max_mm
        or robust_threshold(metrics["action_xyz_max_mm"].to_numpy(), args.quantile, args.mad_k, floor=120.0),
        "action_xyz_mean_mm": args.action_xyz_mean_mm
        or robust_threshold(metrics["action_xyz_mean_mm"].to_numpy(), args.quantile, args.mad_k, floor=60.0),
        "action_rot_max_deg": args.action_rot_max_deg
        or robust_threshold(metrics["action_rot_max_deg"].to_numpy(), args.quantile, args.mad_k, floor=30.0),
        "action_rot_mean_deg": args.action_rot_mean_deg
        or robust_threshold(metrics["action_rot_mean_deg"].to_numpy(), args.quantile, args.mad_k, floor=10.0),
        "state_xyz_step_max_mm": args.state_xyz_step_max_mm
        or robust_threshold(metrics["state_xyz_step_max_mm"].to_numpy(), args.quantile, args.mad_k, floor=80.0),
        "h1_pos_err_median_mm": args.h1_pos_err_median_mm
        or robust_threshold(metrics["h1_pos_err_median_mm"].to_numpy(), args.quantile, args.mad_k, floor=80.0),
        "quat_norm_error": args.quat_norm_error,
        "min_frames": args.min_frames,
    }


def flag_metrics(metrics: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    flagged = metrics.copy()
    reasons: list[str] = []
    for _, row in flagged.iterrows():
        one: list[str] = []
        if row["nan_count"] > 0:
            one.append("nan")
        if row["frames"] < thresholds["min_frames"]:
            one.append(f"short<{thresholds['min_frames']:.0f}")
        if max(abs(row["quat_norm_min"] - 1.0), abs(row["quat_norm_max"] - 1.0)) > thresholds["quat_norm_error"]:
            one.append("bad_quat_norm")
        for key in [
            "action_xyz_max_mm",
            "action_xyz_mean_mm",
            "action_rot_max_deg",
            "action_rot_mean_deg",
            "state_xyz_step_max_mm",
            "h1_pos_err_median_mm",
        ]:
            if float(row[key]) > thresholds[key]:
                one.append(f"{key}>{thresholds[key]:.1f}")
        if bool(row["looks_like_future_delta"]):
            one.append(f"future_delta_h{int(row['best_delta_horizon'])}")
        reasons.append(";".join(one))
    flagged["suspicious"] = [bool(x) for x in reasons]
    flagged["reasons"] = reasons
    return flagged


def flag_strong(metrics: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    flagged = metrics.copy()
    strong_reasons: list[str] = []
    for _, row in flagged.iterrows():
        one: list[str] = []
        if row["nan_count"] > 0:
            one.append("nan")
        if row["action_xyz_max_mm"] > args.strong_action_xyz_max_mm:
            one.append(f"action_xyz_max>{args.strong_action_xyz_max_mm:.1f}")
        if row["action_xyz_mean_mm"] > args.strong_action_xyz_mean_mm:
            one.append(f"action_xyz_mean>{args.strong_action_xyz_mean_mm:.1f}")
        if row["action_rot_max_deg"] > args.strong_action_rot_max_deg:
            one.append(f"action_rot_max>{args.strong_action_rot_max_deg:.1f}")
        if row["action_rot_mean_deg"] > args.strong_action_rot_mean_deg:
            one.append(f"action_rot_mean>{args.strong_action_rot_mean_deg:.1f}")
        if row["h1_pos_err_median_mm"] > args.strong_h1_pos_err_median_mm:
            one.append(f"h1_pos_err>{args.strong_h1_pos_err_median_mm:.1f}")
        if row["state_xyz_step_max_mm"] > args.strong_state_xyz_step_max_mm:
            one.append(f"state_step>{args.strong_state_xyz_step_max_mm:.1f}")
        strong_reasons.append(";".join(one))
    flagged["strong_suspicious"] = [bool(x) for x in strong_reasons]
    flagged["strong_reasons"] = strong_reasons
    return flagged


def video_duration(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
            text=True,
        ).strip()
        return float(out)
    except Exception:
        return None


def check_video_metadata(dataset: Path, meta: pd.DataFrame, slack_s: float) -> pd.DataFrame:
    if meta.empty:
        return pd.DataFrame(columns=["episode", "camera", "file_index", "from_timestamp", "to_timestamp", "duration_s", "reason"])
    rows: list[dict[str, Any]] = []
    cache: dict[tuple[str, int, int], float | None] = {}
    for _, row in meta.iterrows():
        episode = int(row["episode_index"])
        for camera in ("observation.images.wrist", "observation.images.top"):
            file_col = f"videos/{camera}/file_index"
            chunk_col = f"videos/{camera}/chunk_index"
            from_col = f"videos/{camera}/from_timestamp"
            to_col = f"videos/{camera}/to_timestamp"
            if file_col not in row.index:
                continue
            chunk = int(row[chunk_col])
            file_index = int(row[file_col])
            start = float(row[from_col])
            end = float(row[to_col])
            key = (camera, chunk, file_index)
            if key not in cache:
                path = dataset / "videos" / camera / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
                cache[key] = video_duration(path)
            duration = cache[key]
            reason = ""
            if duration is None:
                reason = "missing_or_unreadable_video_file"
            elif start < -slack_s or end <= start:
                reason = "bad_time_range"
            elif end > duration + slack_s:
                reason = "timestamp_exceeds_video_duration"
            if reason:
                rows.append(
                    {
                        "episode": episode,
                        "camera": camera,
                        "file_index": file_index,
                        "from_timestamp": start,
                        "to_timestamp": end,
                        "duration_s": duration,
                        "reason": reason,
                    }
                )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only per-episode quality check of a PiperX EE dataset.")
    parser.add_argument("--dataset", type=Path, required=True, help="EE dataset (convert_to_ee.py / resample.py output).")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for the CSV/JSON/TXT reports.")
    parser.add_argument(
        "--write-manifest",
        type=Path,
        default=None,
        help="Also write the candidate episode ids as an exclusion manifest (JSON) for review.",
    )
    parser.add_argument(
        "--manifest-from",
        choices=("strong", "review", "strong+video"),
        default="strong",
        help="Which flagged set goes into --write-manifest: strong (absolute thresholds), review "
        "(robust per-dataset thresholds, broader), strong+video (strong plus bad video metadata).",
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--mad-k", type=float, default=8.0)
    parser.add_argument("--min-frames", type=float, default=35.0)
    parser.add_argument("--quat-norm-error", type=float, default=0.02)
    parser.add_argument("--action-xyz-max-mm", type=float, default=None)
    parser.add_argument("--action-xyz-mean-mm", type=float, default=None)
    parser.add_argument("--action-rot-max-deg", type=float, default=None)
    parser.add_argument("--action-rot-mean-deg", type=float, default=None)
    parser.add_argument("--state-xyz-step-max-mm", type=float, default=None)
    parser.add_argument("--h1-pos-err-median-mm", type=float, default=None)
    parser.add_argument("--strong-action-xyz-max-mm", type=float, default=220.0)
    parser.add_argument("--strong-action-xyz-mean-mm", type=float, default=80.0)
    parser.add_argument("--strong-action-rot-max-deg", type=float, default=45.0)
    parser.add_argument("--strong-action-rot-mean-deg", type=float, default=13.0)
    parser.add_argument("--strong-h1-pos-err-median-mm", type=float, default=80.0)
    parser.add_argument("--strong-state-xyz-step-max-mm", type=float, default=90.0)
    parser.add_argument("--video-metadata-slack-s", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.dataset / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Not a LeRobot dataset: {args.dataset}")
    info = load_info(args.dataset)
    state_shape = info.get("features", {}).get("observation.state", {}).get("shape")
    if state_shape is not None and list(state_shape) != [8]:
        raise ValueError(f"Expected an EE dataset (8D observation.state), got shape={state_shape}; run convert_to_ee.py first.")
    provenance_path = args.dataset / "meta" / "conversion_provenance.json"
    renumbered = False
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        renumbered = bool(
            provenance.get("excluded_episode_ranges") or provenance.get("excluded_episode_ids") or provenance.get("included_tasks")
        )
    if renumbered:
        print(
            "[EE_QC] WARNING: this dataset was filtered and renumbered (see meta/conversion_provenance.json); "
            "reported episode ids are output ids, not source ids. To build an exclusion manifest for the "
            "source, check an unfiltered dataset instead."
        )
    tasks = load_episode_tasks(args.dataset)
    meta = load_episode_meta(args.dataset)
    data = load_data(args.dataset)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for episode, ep in data.groupby("episode_index", sort=True):
        rows.append(episode_metrics(ep, tasks.get(int(episode), ""), args.horizons))
    metrics = pd.DataFrame(rows).sort_values("episode").reset_index(drop=True)
    thresholds = build_thresholds(metrics, args)
    flagged = flag_metrics(metrics, thresholds)
    flagged = flag_strong(flagged, args)

    metrics_path = args.out_dir / "episode_quality_metrics.csv"
    review_path = args.out_dir / "review_suspicious_episodes.csv"
    strong_path = args.out_dir / "strong_suspicious_episodes.csv"
    video_bad_path = args.out_dir / "bad_video_metadata.csv"
    thresholds_path = args.out_dir / "thresholds.json"
    summary_path = args.out_dir / "summary.txt"
    flagged.to_csv(metrics_path, index=False)
    flagged[flagged["suspicious"]].to_csv(review_path, index=False)
    flagged[flagged["strong_suspicious"]].to_csv(strong_path, index=False)
    if (args.dataset / "videos").exists():
        bad_video = check_video_metadata(args.dataset, meta, args.video_metadata_slack_s)
    else:
        print("[EE_QC] no videos/ directory (e.g. resample.py --video-mode skip): video metadata check skipped")
        bad_video = check_video_metadata(args.dataset, pd.DataFrame(), args.video_metadata_slack_s)
    bad_video.to_csv(video_bad_path, index=False)
    thresholds_path.write_text(json.dumps(thresholds, indent=2, sort_keys=True) + "\n")

    suspicious = flagged[flagged["suspicious"]]
    strong = flagged[flagged["strong_suspicious"]]
    lines = [
        f"dataset: {args.dataset}",
        f"dataset_fps: {info.get('fps')}",
        f"episodes: {len(flagged)}",
        f"frames: {len(data)}",
        f"strong_suspicious: {len(strong)}",
        f"review_suspicious: {len(suspicious)}",
        f"bad_video_metadata_rows: {len(bad_video)}",
        f"bad_video_metadata_episodes: {bad_video['episode'].nunique() if not bad_video.empty else 0}",
        "",
        "thresholds:",
    ]
    lines += [f"  {key}: {value}" for key, value in thresholds.items()]
    lines += ["", "strong suspicious episodes:"]
    if strong.empty:
        lines.append("  none")
    else:
        for _, row in strong.iterrows():
            lines.append(
                f"  {int(row['episode']):04d} | {row['strong_reasons']} | "
                f"action_xyz_max={row['action_xyz_max_mm']:.1f}mm "
                f"action_xyz_mean={row['action_xyz_mean_mm']:.1f}mm "
                f"action_rot_max={row['action_rot_max_deg']:.1f}deg "
                f"h1_err={row['h1_pos_err_median_mm']:.1f}mm | {row['task']}"
            )
    summary_path.write_text("\n".join(lines) + "\n")

    print("\n".join("[EE_QC] " + line for line in lines[:12]))
    print(f"[EE_QC] metrics_csv: {metrics_path}")
    print(f"[EE_QC] strong_csv: {strong_path}")
    print(f"[EE_QC] review_csv: {review_path}")
    print(f"[EE_QC] bad_video_metadata_csv: {video_bad_path}")
    print(f"[EE_QC] thresholds_json: {thresholds_path}")
    print(f"[EE_QC] summary_txt: {summary_path}")
    if not strong.empty:
        print("[EE_QC] strong suspicious ids:")
        print(" ".join(str(int(v)) for v in strong["episode"].to_numpy()))
    if not suspicious.empty:
        print("[EE_QC] review suspicious ids:")
        print(" ".join(str(int(v)) for v in suspicious["episode"].to_numpy()))
    if not bad_video.empty:
        print("[EE_QC] bad video metadata episode ids:")
        print(" ".join(str(int(v)) for v in sorted(bad_video["episode"].unique())))

    if args.write_manifest is not None:
        reasons: dict[int, str] = {}
        source = strong if args.manifest_from.startswith("strong") else suspicious
        reason_col = "strong_reasons" if args.manifest_from.startswith("strong") else "reasons"
        for _, row in source.iterrows():
            reasons[int(row["episode"])] = str(row[reason_col])
        if args.manifest_from == "strong+video" and not bad_video.empty:
            for episode, group in bad_video.groupby("episode"):
                note = "video:" + ",".join(sorted(set(group["reason"])))
                reasons[int(episode)] = ";".join(filter(None, [reasons.get(int(episode), ""), note]))
        description = (
            f"Candidate exclusions from check_quality.py ({args.manifest_from}) on {args.dataset}. "
            "Review every id (e.g. with inspect_episode.py) before using this with resample.py --exclusions."
        )
        if renumbered:
            description += " WARNING: ids are output ids of an already filtered dataset, not source ids."
        write_manifest(args.write_manifest, sorted(reasons), description=description, reasons=reasons)
        print(f"[EE_QC] candidate manifest ({len(reasons)} episodes): {args.write_manifest}")


if __name__ == "__main__":
    main()
