#!/usr/bin/env python3
"""Compute feedback-trajectory smoothness metrics for PiperX benchmark runs.

The input is the accepted success/failure trials produced by
eval_benchmark.py. Metrics are computed once per episode and then
aggregated, so longer episodes do not receive more statistical weight.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial.transform import Rotation, Slerp


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must use LABEL=/absolute/or/relative/path")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--run must use LABEL=PATH")
    return label.strip(), Path(path).expanduser().resolve()


def feedback_rows(attempt_dir: Path) -> tuple[list[dict[str, Any]], str]:
    high_rate_rows = read_jsonl(attempt_dir / "state_feedback_high_rate.jsonl")
    if len(high_rate_rows) >= 8 and all("ee_feedback_xyz_quat_gripper" in row for row in high_rate_rows):
        updated_rows = [row for row in high_rate_rows if row.get("joint_feedback_updated", True)]
        if len(updated_rows) >= 8:
            return updated_rows, "high_rate_unique"
        return high_rate_rows, "high_rate_polled"
    return read_jsonl(attempt_dir / "frames.jsonl"), "control_loop"


def inferred_sample_hz(rows: list[dict[str, Any]], fallback: float = 8.0) -> float:
    if len(rows) < 2:
        return fallback
    times = np.asarray([row["trial_elapsed_s"] for row in rows], dtype=np.float64)
    intervals = np.diff(times)
    intervals = intervals[np.isfinite(intervals) & (intervals > 1e-6)]
    if not len(intervals):
        return fallback
    return float(1.0 / np.median(intervals))


def odd_window(length: int, preferred: int = 7) -> int:
    window = min(preferred, length if length % 2 else length - 1)
    return window if window >= 5 else 0


def smooth(values: np.ndarray) -> np.ndarray:
    window = odd_window(len(values))
    if not window:
        return values.copy()
    return savgol_filter(values, window_length=window, polyorder=min(3, window - 2), axis=0, mode="interp")


def resample_episode(rows: list[dict[str, Any]], sample_hz: float) -> dict[str, np.ndarray] | None:
    if len(rows) < 8:
        return None
    time_s = np.asarray([row["trial_elapsed_s"] for row in rows], dtype=np.float64)
    keep = np.r_[True, np.diff(time_s) > 1e-6]
    time_s = time_s[keep]
    if len(time_s) < 8 or time_s[-1] - time_s[0] < 1.0:
        return None

    positions = np.asarray([row["ee_feedback_xyz_quat_gripper"][:3] for row in rows], dtype=np.float64)[keep]
    quaternions = np.asarray([row["ee_feedback_xyz_quat_gripper"][3:7] for row in rows], dtype=np.float64)[keep]
    joints = np.asarray([row["joint_feedback_deg"] for row in rows], dtype=np.float64)[keep]

    period = 1.0 / sample_hz
    sample_count = int(math.floor((time_s[-1] - time_s[0]) / period)) + 1
    uniform_t = time_s[0] + np.arange(sample_count, dtype=np.float64) * period
    if len(uniform_t) < 8:
        return None
    position_u = PchipInterpolator(time_s, positions, axis=0)(uniform_t)
    joint_u = PchipInterpolator(time_s, joints, axis=0)(uniform_t)
    rotation_u = Slerp(time_s, Rotation.from_quat(quaternions))(uniform_t)
    return {"time": uniform_t, "position": position_u, "rotation": rotation_u, "joints": joint_u}


def spectral_arc_length(speed: np.ndarray, sample_hz: float, amplitude_threshold: float = 0.05) -> float:
    """Dimensionless SPARC score. Higher (less negative) means smoother."""
    speed = np.asarray(speed, dtype=np.float64)
    if len(speed) < 8 or not np.isfinite(speed).all() or float(np.max(np.abs(speed))) < 1e-9:
        return math.nan
    nfft = max(64, 2 ** int(math.ceil(math.log2(len(speed) * 4))))
    magnitude = np.abs(np.fft.rfft(speed, n=nfft))
    peak = float(magnitude.max())
    if peak <= 0:
        return math.nan
    magnitude /= peak
    frequency = np.fft.rfftfreq(nfft, d=1.0 / sample_hz)
    candidates = np.flatnonzero(magnitude >= amplitude_threshold)
    if len(candidates) < 2:
        return math.nan
    last = max(2, int(candidates[-1]))
    frequency = frequency[: last + 1]
    magnitude = magnitude[: last + 1]
    normalized_frequency = frequency / max(float(frequency[-1]), np.finfo(float).eps)
    return -float(np.sum(np.sqrt(np.diff(normalized_frequency) ** 2 + np.diff(magnitude) ** 2)))


def high_frequency_power_ratio(signal: np.ndarray, sample_hz: float, cutoff_hz: float) -> float:
    """Power above cutoff, limited by the feedback stream's Nyquist frequency."""
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) < 8:
        return math.nan
    centered = signal - np.mean(signal)
    power = np.abs(np.fft.rfft(centered)) ** 2
    frequency = np.fft.rfftfreq(len(centered), d=1.0 / sample_hz)
    valid = frequency > 0
    denominator = float(power[valid].sum())
    if denominator <= 1e-12:
        return 0.0
    return float(power[frequency >= cutoff_hz].sum() / denominator)


def log_dimensionless_jerk_cost(position: np.ndarray, sample_hz: float) -> float:
    """log10 dimensionless jerk cost. Lower means smoother."""
    if len(position) < 8:
        return math.nan
    dt = 1.0 / sample_hz
    filtered = smooth(np.asarray(position, dtype=np.float64))
    velocity = np.gradient(filtered, dt, axis=0)
    acceleration = np.gradient(velocity, dt, axis=0)
    jerk = np.gradient(acceleration, dt, axis=0)
    duration = (len(filtered) - 1) * dt
    path_length = float(np.linalg.norm(np.diff(filtered, axis=0), axis=1).sum())
    if duration <= 0 or path_length <= 1e-9:
        return math.nan
    integral = float(np.trapz(np.sum(jerk**2, axis=1), dx=dt))
    cost = duration**5 * integral / path_length**2
    return float(np.log10(max(cost, np.finfo(float).eps)))


def angular_velocity(rotation: Rotation, sample_hz: float) -> np.ndarray:
    relative = rotation[:-1].inv() * rotation[1:]
    return relative.as_rotvec() * sample_hz


def active_slice(linear_speed: np.ndarray, angular_speed: np.ndarray, sample_hz: float) -> slice:
    common = min(len(linear_speed), len(angular_speed))
    if common < 8:
        return slice(0, common)
    linear = linear_speed[:common]
    angular = angular_speed[:common]
    linear_threshold = max(0.010, 0.05 * float(np.max(linear)))
    angular_threshold = max(np.deg2rad(5.0), 0.05 * float(np.max(angular)))
    active = np.flatnonzero((linear >= linear_threshold) | (angular >= angular_threshold))
    if len(active) < 8:
        return slice(0, common)
    padding = max(1, int(round(0.25 * sample_hz)))
    return slice(max(0, int(active[0]) - padding), min(common, int(active[-1]) + padding + 1))


def episode_metrics(rows: list[dict[str, Any]], sample_hz: float, hf_cutoff_hz: float) -> dict[str, float] | None:
    sampled = resample_episode(rows, sample_hz)
    if sampled is None:
        return None
    dt = 1.0 / sample_hz
    position = smooth(sampled["position"])
    linear_velocity = np.gradient(position, dt, axis=0)
    linear_speed = np.linalg.norm(linear_velocity, axis=1)
    omega = angular_velocity(sampled["rotation"], sample_hz)
    angular_speed = np.linalg.norm(omega, axis=1)
    selection = active_slice(linear_speed[:-1], angular_speed, sample_hz)

    position_active = position[:-1][selection]
    linear_speed_active = linear_speed[:-1][selection]
    omega_active = omega[selection]
    angular_speed_active = angular_speed[selection]
    joints_active = smooth(sampled["joints"][:-1][selection])
    if len(linear_speed_active) < 8:
        return None

    joint_velocity = np.gradient(joints_active, dt, axis=0)
    joint_sparc_values = [spectral_arc_length(np.abs(joint_velocity[:, idx]), sample_hz) for idx in range(6)]
    peak_prominence = max(0.010, 0.10 * float(np.max(linear_speed_active) - np.min(linear_speed_active)))
    peaks, _ = find_peaks(linear_speed_active, prominence=peak_prominence, distance=max(1, int(0.25 * sample_hz)))
    stop_ratio = float(np.mean(linear_speed_active < 0.010))

    # Angular jerk is computed from the integrated rotation-vector path so the
    # dimensionless normalization matches the translational implementation.
    angular_path = np.vstack([np.zeros(3), np.cumsum(omega_active * dt, axis=0)])
    return {
        "active_duration_s": len(linear_speed_active) * dt,
        "translation_sparc": spectral_arc_length(linear_speed_active, sample_hz),
        "rotation_sparc": spectral_arc_length(angular_speed_active, sample_hz),
        "translation_log_dimensionless_jerk_cost": log_dimensionless_jerk_cost(position_active, sample_hz),
        "rotation_log_dimensionless_jerk_cost": log_dimensionless_jerk_cost(angular_path, sample_hz),
        "translation_hf_power_ratio": high_frequency_power_ratio(linear_speed_active, sample_hz, hf_cutoff_hz),
        "rotation_hf_power_ratio": high_frequency_power_ratio(angular_speed_active, sample_hz, hf_cutoff_hz),
        "joint_sparc_median": float(np.nanmedian(joint_sparc_values)),
        "movement_units": float(len(peaks)),
        "stop_ratio": stop_ratio,
    }


def percentile_summary(values: list[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"median": None, "q25": None, "q75": None, "mean": None}
    return {
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "mean": float(np.mean(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True, metavar="LABEL=PATH")
    parser.add_argument(
        "--sample-hz",
        type=float,
        default=0.0,
        help="Uniform analysis rate. Default 0 uses each episode's measured feedback rate.",
    )
    parser.add_argument("--hf-cutoff-hz", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/piperx_benchmark/smoothness_analysis"))
    args = parser.parse_args()
    if args.sample_hz < 0:
        raise ValueError("--sample-hz must be >= 0")
    if args.hf_cutoff_hz <= 0:
        raise ValueError("--hf-cutoff-hz must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "sample_hz": args.sample_hz,
        "hf_cutoff_hz": args.hf_cutoff_hz,
        "warning": (
            "High-rate state feedback is used when present; legacy episodes fall back to control-loop feedback. "
            "Metrics from different sampling rates should be compared with their analysis_sample_hz reported."
        ),
        "runs": {},
    }
    for label, run_dir in args.run:
        records = [
            record
            for record in read_jsonl(run_dir / "manifest.jsonl")
            if record.get("label") in {"success", "failure"}
        ]
        run_metrics = []
        source_counts: dict[str, int] = {}
        for record in records:
            rows, feedback_source = feedback_rows(Path(record["attempt_dir"]))
            source_counts[feedback_source] = source_counts.get(feedback_source, 0) + 1
            analysis_hz = args.sample_hz or inferred_sample_hz(rows)
            if args.hf_cutoff_hz >= analysis_hz / 2:
                print(
                    f"[SMOOTHNESS] skipping trial {record['valid_trial_number']}: "
                    f"cutoff {args.hf_cutoff_hz:g} Hz >= Nyquist at {analysis_hz:.2f} Hz"
                )
                continue
            metrics = episode_metrics(rows, analysis_hz, args.hf_cutoff_hz)
            if metrics is None:
                continue
            row = {
                "run": label,
                "valid_trial_number": record["valid_trial_number"],
                "task_index": record["task_index"],
                "label": record["label"],
                "feedback_source": feedback_source,
                "analysis_sample_hz": analysis_hz,
                **metrics,
            }
            run_metrics.append(row)
            all_rows.append(row)
        excluded = {"run", "valid_trial_number", "task_index", "label", "feedback_source"}
        metric_names = [key for key in run_metrics[0] if key not in excluded] if run_metrics else []
        summary["runs"][label] = {
            "accepted_episodes": len(records),
            "analyzed_episodes": len(run_metrics),
            "feedback_sources": source_counts,
            "metrics": {name: percentile_summary([row[name] for row in run_metrics]) for name in metric_names},
        }

    csv_path = args.output_dir / "episode_smoothness_metrics.csv"
    if all_rows:
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
    summary_path = args.output_dir / "smoothness_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"[SMOOTHNESS] episode_csv={csv_path}")
    print(f"[SMOOTHNESS] summary_json={summary_path}")


if __name__ == "__main__":
    main()
