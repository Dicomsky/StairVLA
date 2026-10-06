#!/usr/bin/env python
"""Run a StairVLA (or StarVLA baseline) policy on a real AgileX PiperX arm.

Interactive session: type an instruction, the arm homes slowly, press ENTER to run the
trial, then label it success/failure/miss. Results are appended to ``--results-jsonl``.

The policy is queried through the websocket server in
``deployment/model_server/server_policy.py``. With the default (paper) settings the
policy sees the 8D EE state ``[x, y, z, qx, qy, qz, qw, gripper_mm]`` and returns 7D
delta-EE actions ``[dx, dy, dz (world), drx, dry, drz (axis-angle, EE frame), gripper_mm]``,
which are integrated over the chunk and converted to joint targets with local IK.

Without ``--execute`` the policy is queried but nothing is sent to the arm.
"""

from __future__ import annotations

import argparse
import json
import math
import select
import sys
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.PiperX.common.kinematics import DEFAULT_URDF, PiperXKinematics, seed_candidates
from examples.PiperX.common.robot import (
    DEFAULT_HOME_JOINTS_DEG,
    PIPER_ACTION_KEYS,
    PIPERX_HARD_LIMITS_DEG,
    PiperXRobot,
    PiperXRobotConfig,
    add_robot_args,
    limit_joint_step,
)

DEFAULT_TASK = "Pick up the apple and place it in the red basket."
DEFAULT_RESULTS_JSONL = Path("outputs/piperx_eval/results.jsonl")
STATS_FILENAME = "dataset_statistics.json"

# Policy state dimension for each --policy-state-space.
STATE_DIMS = {"ee": 8, "ee7": 7, "joint": 7}


class DeltaEEToJointController:
    def __init__(self, args: argparse.Namespace):
        self.kin = PiperXKinematics(args.urdf)
        self.seed_candidates = seed_candidates
        self.args = args
        self.ee_workspace = None
        if args.ee_workspace is not None:
            self.ee_workspace = np.asarray(args.ee_workspace, dtype=np.float64).reshape(3, 2)

    def fk(self, joints_deg: np.ndarray) -> np.ndarray:
        return self.kin.fk(np.deg2rad(np.asarray(joints_deg[:6], dtype=np.float64)))

    def ee_state_from_joint_state(self, joint_state: np.ndarray) -> np.ndarray:
        return self.kin.ee_state_from_joint_state(joint_state)

    def delta_to_joint_target(
        self,
        delta_action: np.ndarray,
        current_feedback: np.ndarray,
        current_cmd: np.ndarray,
        reference_t: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        delta = np.asarray(delta_action, dtype=np.float64).copy()
        raw_delta = delta.copy()
        max_pos = max(0.0, float(self.args.max_ee_delta_m))
        max_rot = max(0.0, float(self.args.max_ee_rot_delta_rad))
        if max_pos > 0:
            delta[:3] = np.clip(delta[:3], -max_pos, max_pos)
        if max_rot > 0:
            rot_norm = float(np.linalg.norm(delta[3:6]))
            if rot_norm > max_rot:
                delta[3:6] *= max_rot / max(rot_norm, 1e-12)

        base_t = self.fk(current_feedback) if reference_t is None else np.asarray(reference_t, dtype=np.float64)
        target_t = base_t.copy()
        target_t[:3, 3] = base_t[:3, 3] + delta[:3]
        ee_workspace_clipped = False
        if self.ee_workspace is not None:
            clipped_pos = np.clip(target_t[:3, 3], self.ee_workspace[:, 0], self.ee_workspace[:, 1])
            ee_workspace_clipped = bool(np.any(np.abs(clipped_pos - target_t[:3, 3]) > 1e-9))
            target_t[:3, 3] = clipped_pos
        target_t[:3, :3] = (R.from_matrix(base_t[:3, :3]) * R.from_rotvec(delta[3:6])).as_matrix()

        preferred = np.deg2rad(np.asarray(current_cmd[:6], dtype=np.float64))
        current_q = np.deg2rad(np.asarray(current_feedback[:6], dtype=np.float64))
        best = None
        for seed in self.seed_candidates(self.kin, preferred) + self.seed_candidates(self.kin, current_q):
            result = self.kin.solve_ik(
                target_t,
                seed,
                pos_weight=1.0,
                rot_weight=self.args.ik_rot_weight,
                max_nfev=self.args.ik_max_nfev,
                regularization_weight=self.args.ik_joint_regularization,
                regularization_target=preferred,
            )
            solved_t = self.kin.fk(result.x)
            pos_err = float(np.linalg.norm(solved_t[:3, 3] - target_t[:3, 3]))
            rot_err = float(np.linalg.norm(R.from_matrix(target_t[:3, :3].T @ solved_t[:3, :3]).as_rotvec()))
            score = pos_err + self.args.ik_rot_weight * rot_err + 0.002 * float(np.linalg.norm(result.x - preferred))
            if best is None or score < best[0]:
                best = (score, result, pos_err, rot_err)

        if best is None:
            return current_cmd.copy(), {
                "ik_success": False,
                "ik_reason": "no_result",
                "target_ee_transform": target_t,
            }
        _, result, pos_err, rot_err = best
        ik_success = bool(result.success) and pos_err <= self.args.ik_pos_tolerance_m and rot_err <= self.args.ik_rot_tolerance_rad
        if not ik_success:
            return current_cmd.copy(), {
                "ik_success": False,
                "ik_reason": "tolerance",
                "ik_pos_err_m": pos_err,
                "ik_rot_err_rad": rot_err,
                "target_ee_transform": target_t,
            }

        target = current_cmd.copy()
        target[:6] = np.rad2deg(result.x).astype(np.float32)
        target[6] = float(delta_action[6])
        return target, {
            "ik_success": True,
            "ik_pos_err_m": pos_err,
            "ik_rot_err_rad": rot_err,
            "ee_delta_clipped": bool(np.any(np.abs(raw_delta[:6] - delta[:6]) > 1e-9)),
            "ee_workspace_clipped": ee_workspace_clipped,
            "target_ee_pos_m": target_t[:3, 3].tolist(),
            "target_ee_transform": target_t,
        }


def build_robot(args: argparse.Namespace, connect_cameras: bool = True) -> PiperXRobot:
    return PiperXRobot(PiperXRobotConfig.from_args(args), connect_cameras=connect_cameras)


def resolve_stats_path(args: argparse.Namespace) -> Path:
    """Return ``--stats-json``, or the ``dataset_statistics.json`` next to/above ``--checkpoint``."""
    if args.stats_json is not None:
        path = Path(args.stats_json).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"--stats-json does not exist: {path}")
        return path
    if args.checkpoint is None:
        raise SystemExit(
            "Normalization statistics are required: pass --checkpoint <run dir or .pt file> "
            f"(its {STATS_FILENAME} is located automatically) or --stats-json <path>."
        )
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"--checkpoint does not exist: {checkpoint}")
    start = checkpoint if checkpoint.is_dir() else checkpoint.parent
    for directory in (start, *start.parents):
        candidate = directory / STATS_FILENAME
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No {STATS_FILENAME} found in {start} or any parent directory; pass --stats-json explicitly."
    )


def validate_spaces(args: argparse.Namespace, state_stats: dict[str, np.ndarray] | None) -> None:
    """Check that the action space, policy state space and state statistics agree."""
    if (args.action_space == "joint") != (args.policy_state_space == "joint"):
        raise SystemExit(
            "--action-space joint (legacy) must be used together with --policy-state-space joint, and "
            "--action-space delta-ee with --policy-state-space ee (or ee7). "
            f"Got --action-space {args.action_space} --policy-state-space {args.policy_state_space}."
        )
    if state_stats is None:
        return
    expected = STATE_DIMS[args.policy_state_space]
    for key in ("q01", "min", "mean"):
        if key in state_stats:
            dim = int(state_stats[key].shape[0])
            if dim != expected:
                raise SystemExit(
                    f"State statistics have dim={dim}, but --policy-state-space {args.policy_state_space} "
                    f"expects dim={expected} (EE state is 8D [x,y,z,qx,qy,qz,qw,gripper_mm], joint state "
                    "is 7D [6 joints deg, gripper_mm]). Use statistics from a checkpoint trained on the "
                    "matching state space, or pass --state-stats-json."
                )
            return


def load_policy_stats(
    args: argparse.Namespace, log_prefix: str
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    """Load action/state statistics from --checkpoint/--stats-json and validate them against the CLI."""
    stats_path = resolve_stats_path(args)
    dataset_stats = load_dataset_stats(stats_path)
    action_stats = dataset_stats["action"]
    state_source = args.state_stats_json or stats_path
    state_dataset_stats = load_dataset_stats(args.state_stats_json) if args.state_stats_json else dataset_stats
    state_stats = state_dataset_stats.get("state")
    print(f"{log_prefix} stats: action={stats_path} state={state_source}")
    if state_stats is None:
        print(f"{log_prefix} WARNING: no state statistics found; the policy state is sent unnormalized.")
    validate_spaces(args, state_stats)
    return action_stats, state_stats


def load_dataset_stats(path: Path | None) -> dict[str, dict[str, np.ndarray]] | None:
    if path is None:
        return None
    with path.open("r") as f:
        stats = json.load(f)
    if not ({"action", "state", "observation.state"} & set(stats.keys())) and len(stats) == 1:
        stats = next(iter(stats.values()))

    parsed_by_modality = {}
    for source_key, target_key in (("action", "action"), ("state", "state"), ("observation.state", "state")):
        modality_stats = stats.get(source_key)
        if modality_stats is None:
            continue
        parsed = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in modality_stats.items()
            if key in {"min", "max", "q01", "q99", "mean", "std", "mask"}
        }
        if "mask" in modality_stats:
            parsed["mask"] = np.asarray(modality_stats["mask"], dtype=bool)
        parsed_by_modality[target_key] = parsed
    if "action" not in parsed_by_modality:
        raise KeyError(f"No 'action' section in {path}")
    return parsed_by_modality


def load_action_stats(path: Path | None) -> dict[str, np.ndarray] | None:
    dataset_stats = load_dataset_stats(path)
    return None if dataset_stats is None else dataset_stats["action"]


# Normalization modes supported by the gr00t lerobot loader
# (starVLA/dataloader/gr00t_lerobot/transform/state_action.py::Normalizer).
#   "min_max": 2 * (x - min) / (max - min) - 1, no clipping
#   "q99":     2 * (x - q01) / (q99 - q01) - 1, clipped to [-1, 1]
# This MUST match the normalization_modes of the data config the checkpoint was
# trained with, otherwise the policy is asked to operate in the wrong units. The
# PiperX data configs (e.g. PiperXFruitV3EEDataConfig) use q99 for state and action.
NORMALIZATION_MODES = ("min_max", "q99")
DEFAULT_NORMALIZATION_MODE = "q99"


def _normalization_bounds(
    stats: dict[str, np.ndarray],
    norm_mode: str,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return (lows, highs, clip) for the requested normalization mode."""
    if norm_mode not in NORMALIZATION_MODES:
        raise ValueError(f"Unknown normalization mode {norm_mode!r}, expected one of {NORMALIZATION_MODES}")
    if norm_mode == "q99":
        if "q01" in stats and "q99" in stats:
            return stats["q01"], stats["q99"], True
        raise KeyError(f"norm_mode='q99' needs q01/q99 stats, got keys={list(stats.keys())}")
    if "min" in stats and "max" in stats:
        return stats["min"], stats["max"], False
    raise KeyError(f"norm_mode='min_max' needs min/max stats, got keys={list(stats.keys())}")


def normalize_state(
    state: np.ndarray,
    stats: dict[str, np.ndarray] | None,
    norm_mode: str = DEFAULT_NORMALIZATION_MODE,
) -> np.ndarray:
    if stats is None:
        return state.reshape(1, -1).astype(np.float32)
    for key in ("min", "q01", "mean"):
        if key in stats and stats[key].shape[0] != state.shape[0]:
            raise ValueError(
                f"State dim mismatch: policy state has dim={state.shape[0]}, "
                f"but stats[{key}] has dim={stats[key].shape[0]}. "
                "Use --state-stats-json to provide matching state stats."
            )
    try:
        lows, highs, clip = _normalization_bounds(stats, norm_mode)
    except KeyError:
        if "mean" in stats and "std" in stats:
            std = np.where(stats["std"] == 0, 1.0, stats["std"])
            return ((state - stats["mean"]) / std).reshape(1, -1).astype(np.float32)
        return state.reshape(1, -1).astype(np.float32)

    denom = np.where(highs == lows, 1.0, highs - lows)
    normalized = 2.0 * (state - lows) / denom - 1.0
    if clip:
        normalized = np.clip(normalized, -1.0, 1.0)
    mask = stats.get("mask")
    if mask is not None:
        normalized = np.where(mask, normalized, state)
    return normalized.reshape(1, -1).astype(np.float32)


def unnormalize_action_chunk(
    chunk: np.ndarray,
    stats: dict[str, np.ndarray],
    norm_mode: str = DEFAULT_NORMALIZATION_MODE,
) -> np.ndarray:
    # Keep this inverse matched with the training data config (see --action-norm).
    lows, highs, _ = _normalization_bounds(stats, norm_mode)
    if chunk.shape[-1] != lows.shape[0]:
        raise ValueError(f"Action dim {chunk.shape[-1]} does not match stats dim {lows.shape[0]}")
    clipped = np.clip(chunk, -1.0, 1.0)
    restored = 0.5 * (clipped + 1.0) * (highs - lows) + lows
    return restored


def gripper_target_mm(value: float, args: argparse.Namespace, previous_mm: float | None = None) -> float:
    if args.gripper_action_mode == "absolute":
        value = float(np.clip(value, args.gripper_close_mm, args.gripper_open_mm))
        if not args.gripper_hysteresis:
            return value
        if value <= args.gripper_close_threshold_mm:
            return args.gripper_close_mm
        if value >= args.gripper_open_threshold_mm:
            return args.gripper_open_mm
        if previous_mm is None:
            return value
        return float(np.clip(previous_mm, args.gripper_close_mm, args.gripper_open_mm))
    return args.gripper_open_mm if value >= args.gripper_binary_threshold else args.gripper_close_mm


def extract_action_chunk(
    response: dict[str, Any],
    action_input: str,
    stats: dict[str, np.ndarray] | None,
    debug: bool = False,
    norm_mode: str = DEFAULT_NORMALIZATION_MODE,
) -> np.ndarray:
    if isinstance(response, dict) and response.get("ok") is False:
        error = response.get("error", {})
        if isinstance(error, dict):
            message = error.get("message", str(error))
        else:
            message = str(error)
        raise RuntimeError(f"Policy server returned error: {message}")
    data = response.get("data", response)
    action_key = None
    for key in ("normalized_actions", "actions", "action"):
        if key in data:
            chunk = np.asarray(data[key], dtype=np.float32)
            action_key = key
            break
    else:
        raise KeyError(f"Could not find action in server response keys={list(data.keys())}")

    while chunk.ndim > 2:
        chunk = chunk[0]
    if chunk.ndim == 1:
        chunk = chunk[None, :]
    if chunk.shape[-1] < 7:
        raise ValueError(f"Expected at least 7 action dims, got shape={chunk.shape}")
    chunk = chunk[:, :7]
    raw_chunk = chunk.copy()

    should_unnormalize = action_input == "normalized" or action_key == "normalized_actions" or (
        action_input == "auto" and np.nanmax(np.abs(chunk)) <= 1.5
    )
    if should_unnormalize:
        if stats is None:
            raise ValueError("--action-input needs --stats-json when actions are normalized")
        chunk = unnormalize_action_chunk(chunk, stats, norm_mode)
    if debug:
        print(
            "[PIPERX_EVAL_ACTION] "
            f"key={action_key} input={action_input} unnorm={should_unnormalize} "
            f"raw0={np.array2string(raw_chunk[0], precision=4, suppress_small=False)} "
            f"cmd0={np.array2string(chunk[0], precision=4, suppress_small=False)} "
            f"range_abs_max={float(np.nanmax(np.abs(chunk[:, :6]))):.4f} "
            f"gripper_minmax=({float(np.nanmin(chunk[:, 6])):.1f},{float(np.nanmax(chunk[:, 6])):.1f})"
        )
    return chunk


def format_chunk_summary(chunk: np.ndarray, max_rows: int = 8) -> str:
    rows = np.asarray(chunk, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] < 7:
        return f"shape={rows.shape}"
    xyz_mm = rows[:, :3] * 1000.0
    rot_deg = np.rad2deg(rows[:, 3:6])
    step_mm = np.linalg.norm(xyz_mm, axis=1)
    net_xyz_mm = np.sum(xyz_mm, axis=0)
    grip = rows[:, 6]
    lines = [
        "[PIPERX_EVAL_CHUNK] "
        f"steps={len(rows)} "
        f"net_dxyz_mm=[{net_xyz_mm[0]:+.1f},{net_xyz_mm[1]:+.1f},{net_xyz_mm[2]:+.1f}] "
        f"step_mm_mean/max={float(np.mean(step_mm)):.1f}/{float(np.max(step_mm)):.1f} "
        f"grip_mm={float(np.min(grip)):.1f}..{float(np.max(grip)):.1f}"
    ]
    for idx, row in enumerate(rows[:max_rows]):
        lines.append(
            "[PIPERX_EVAL_CHUNK] "
            f"{idx:02d}: "
            f"dxyz_mm=[{xyz_mm[idx, 0]:+6.1f},{xyz_mm[idx, 1]:+6.1f},{xyz_mm[idx, 2]:+6.1f}] "
            f"drot_deg=[{rot_deg[idx, 0]:+5.1f},{rot_deg[idx, 1]:+5.1f},{rot_deg[idx, 2]:+5.1f}] "
            f"grip={row[6]:6.1f}mm"
        )
    if len(rows) > max_rows:
        lines.append(f"[PIPERX_EVAL_CHUNK] ... {len(rows) - max_rows} more steps")
    return "\n".join(lines)


def observation_joint_state(obs: dict[str, Any]) -> np.ndarray:
    return np.asarray([float(obs[key]) for key in PIPER_ACTION_KEYS], dtype=np.float32)


def observation_state(obs: dict[str, Any]) -> np.ndarray:
    return observation_joint_state(obs)


def observation_images(obs: dict[str, Any], image_order: list[str]) -> list[np.ndarray]:
    images = []
    for name in image_order:
        if name not in obs:
            raise KeyError(f"Camera '{name}' missing from robot observation keys={list(obs.keys())}")
        image = np.asarray(obs[name])
        if image.ndim != 3:
            raise ValueError(f"Camera '{name}' returned shape={image.shape}, expected HWC")
        images.append(image)
    return images


def warm_up_observations(robot, duration_s: float, camera_fps: float) -> tuple[int, float]:
    """Drain camera queues before an episode without recording or inference."""
    duration_s = max(0.0, float(duration_s))
    if duration_s == 0.0:
        return 0, 0.0

    fps = max(float(camera_fps), 1.0)
    target_reads = max(1, int(math.ceil(duration_s * fps)))
    start = time.perf_counter()
    for reads in range(1, target_reads + 1):
        robot.get_observation()
        next_read = start + reads / fps
        sleep_s = next_read - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
    return target_reads, time.perf_counter() - start


def safe_action_bounds(
    stats: dict[str, np.ndarray] | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    if args.action_space == "delta-ee":
        hard_low = np.asarray([-args.max_ee_delta_m, -args.max_ee_delta_m, -args.max_ee_delta_m, -args.max_ee_rot_delta_rad, -args.max_ee_rot_delta_rad, -args.max_ee_rot_delta_rad, args.gripper_close_mm], dtype=np.float32)
        hard_high = np.asarray([args.max_ee_delta_m, args.max_ee_delta_m, args.max_ee_delta_m, args.max_ee_rot_delta_rad, args.max_ee_rot_delta_rad, args.max_ee_rot_delta_rad, args.gripper_open_mm], dtype=np.float32)
    else:
        hard_low = PIPERX_HARD_LIMITS_DEG[:, 0].copy()
        hard_high = PIPERX_HARD_LIMITS_DEG[:, 1].copy()
    if not args.safe:
        return hard_low, hard_high

    if stats is None:
        return hard_low, hard_high

    if args.safe_range == "q01_q99" and "q01" in stats and "q99" in stats:
        data_low = stats["q01"].astype(np.float32).copy()
        data_high = stats["q99"].astype(np.float32).copy()
    elif "min" in stats and "max" in stats:
        data_low = stats["min"].astype(np.float32).copy()
        data_high = stats["max"].astype(np.float32).copy()
    elif "q01" in stats and "q99" in stats:
        data_low = stats["q01"].astype(np.float32).copy()
        data_high = stats["q99"].astype(np.float32).copy()
    else:
        return hard_low, hard_high

    if args.action_space == "delta-ee":
        margin = np.asarray([args.safe_margin_ee_m] * 3 + [args.safe_margin_ee_rot_rad] * 3 + [args.safe_margin_gripper_mm], dtype=np.float32)
    else:
        margin = np.asarray([args.safe_margin_deg] * 6 + [args.safe_margin_gripper_mm], dtype=np.float32)
    data_low -= margin
    data_high += margin
    low = np.maximum(hard_low, data_low)
    high = np.minimum(hard_high, data_high)
    if args.gripper_action_mode == "binary":
        # A binary decision must reach the configured endpoints. Dataset
        # statistics describe measured/recorded apertures (for Fruit25 the
        # minimum is about 40 mm), not the safe command range of the gripper.
        low[6] = args.gripper_close_mm
        high[6] = args.gripper_open_mm
    return low, high


def clip_action_to_safe_range(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, bool]:
    clipped = np.clip(action, low, high)
    return clipped, bool(np.any(np.abs(clipped - action) > 1e-5))


def configure_ee_workspace(args: argparse.Namespace, state_stats: dict[str, np.ndarray] | None) -> None:
    if args.action_space != "delta-ee":
        return
    if args.ee_workspace is not None:
        return
    if not args.ee_workspace_from_stats:
        return
    if state_stats is None or "min" not in state_stats or "max" not in state_stats:
        return
    if state_stats["min"].shape[0] < 3 or state_stats["max"].shape[0] < 3:
        return

    margin = max(0.0, float(args.ee_workspace_margin_m))
    low = state_stats["min"][:3].astype(np.float64) - margin
    high = state_stats["max"][:3].astype(np.float64) + margin
    args.ee_workspace = (
        float(low[0]),
        float(high[0]),
        float(low[1]),
        float(high[1]),
        float(low[2]),
        float(high[2]),
    )


def move_home(robot, home: np.ndarray, args: argparse.Namespace, current: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    print(
        "[PIPERX_EVAL] Slow home: "
        f"target_joints_deg={tuple(round(float(v), 3) for v in home[:6])}, "
        f"speed_ratio={args.home_speed_ratio}, "
        f"max_joint_speed={args.home_max_joint_speed_deg_s}deg/s, "
        f"timeout={args.home_timeout_s}s"
    )
    cmd = current.copy()
    cmd[6] = args.gripper_open_mm
    period = 1.0 / args.control_hz
    start_t = time.perf_counter()
    last_feedback = current.copy()
    while True:
        key = poll_keyboard()
        if key in {"q", "quit"}:
            print("\n[PIPERX_EVAL] Home canceled by user.")
            return cmd, {
                "home_success": False,
                "home_stop_reason": "user",
                "home_remaining_deg": None,
                "home_elapsed_s": time.perf_counter() - start_t,
            }

        obs = robot.get_observation()
        feedback = observation_joint_state(obs)
        last_feedback = feedback
        remaining = float(np.max(np.abs(home[:6] - feedback[:6])))
        if remaining <= args.home_tolerance_deg:
            print("\n[PIPERX_EVAL] Home reached.")
            print(
                "[PIPERX_EVAL] Home feedback joints: "
                + " ".join(f"j{i + 1}={feedback[i]:.2f}" for i in range(6))
                + f" gripper={feedback[6]:.1f}mm"
            )
            return home.copy(), {
                "home_success": True,
                "home_stop_reason": "reached",
                "home_remaining_deg": remaining,
                "home_elapsed_s": time.perf_counter() - start_t,
            }

        elapsed = time.perf_counter() - start_t
        if elapsed >= args.home_timeout_s:
            print(
                "\n[PIPERX_EVAL] Home timeout. Feedback joints: "
                + " ".join(f"j{i + 1}={last_feedback[i]:.2f}" for i in range(6))
                + f" gripper={last_feedback[6]:.1f}mm remaining={remaining:.1f}deg"
            )
            return cmd, {
                "home_success": False,
                "home_stop_reason": "timeout",
                "home_remaining_deg": remaining,
                "home_elapsed_s": elapsed,
            }

        cmd = limit_joint_step(home, cmd, args.home_max_joint_speed_deg_s, args.control_hz)
        if args.execute:
            robot.send_joint_command(cmd, speed_ratio=args.home_speed_ratio)
        print(
            f"[PIPERX_EVAL] homing remaining={remaining:.1f}deg elapsed={elapsed:.1f}s "
            "(type q + ENTER to cancel)",
            end="\r",
            flush=True,
        )
        time.sleep(period)


def poll_keyboard() -> str | None:
    if not sys.stdin.isatty():
        return None
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return None
    return sys.stdin.readline().strip().lower()


def append_result(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_label() -> str:
    while True:
        label = input("Label this trial: [s]uccess / [f]ailure / [m]iss / [q]uit session: ").strip().lower()
        if label in {"s", "success"}:
            return "success"
        if label in {"f", "failure", "fail"}:
            return "failure"
        if label in {"m", "miss", "skip"}:
            return "miss"
        if label in {"q", "quit"}:
            return "quit"
        print("Please type s, f, m, or q.")


def reset_policy_cache(client: WebsocketClientPolicy, reason: str, trial_index: int, instruction: str) -> None:
    response = client.reset(
        {
            "reason": reason,
            "trial_index": trial_index,
            "instruction": instruction,
        }
    )
    if not response.get("ok", False):
        raise RuntimeError(f"Policy cache reset failed: {response}")
    print(f"[PIPERX_EVAL] Policy internal cache reset ({reason}).")


def policy_query(
    client: WebsocketClientPolicy,
    obs: dict[str, Any],
    instruction: str,
    args: argparse.Namespace,
    state_stats: dict[str, np.ndarray] | None,
    ee_controller: DeltaEEToJointController | None = None,
) -> dict[str, Any]:
    del client
    joint_state = observation_joint_state(obs)
    if args.policy_state_space in {"ee", "ee7"}:
        if ee_controller is None:
            raise RuntimeError("EE policy state needs an EE controller")
        state = ee_controller.ee_state_from_joint_state(joint_state)
        if args.policy_state_space == "ee7":
            state = state[:7]
    else:
        state = joint_state
    return {
        "type": "infer",
        "request_id": str(uuid.uuid4()),
        "examples": [
            {
                "image": observation_images(obs, args.image_order),
                "lang": instruction,
                "state": normalize_state(state, state_stats, args.state_norm),
            }
        ],
        "do_sample": False,
        "use_ddim": args.use_ddim,
        "num_ddim_steps": args.num_ddim_steps,
    }


def run_trial(
    client: WebsocketClientPolicy,
    robot,
    instruction: str,
    trial_index: int,
    args: argparse.Namespace,
    stats: dict[str, np.ndarray] | None,
    state_stats: dict[str, np.ndarray] | None,
    safe_low: np.ndarray,
    safe_high: np.ndarray,
    ee_controller: DeltaEEToJointController | None = None,
) -> dict[str, Any]:
    obs = robot.get_observation()
    current = observation_joint_state(obs)
    current_cmd = current.copy()
    home = np.asarray([*args.home_joints_deg, args.gripper_open_mm], dtype=np.float32)
    home_info = {
        "home_success": True,
        "home_stop_reason": "not_requested",
        "home_remaining_deg": None,
        "home_elapsed_s": 0.0,
    }
    if args.move_home_each_trial and args.execute:
        current_cmd, home_info = move_home(robot, home, args, current)
        if not home_info["home_success"]:
            print("[PIPERX_EVAL] Trial will not start because home was not reached.")
            return {
                "time": datetime.now().isoformat(timespec="seconds"),
                "trial_index": trial_index,
                "instruction": instruction,
                "label": read_label(),
                "stop_reason": "home_failed",
                "execute": args.execute,
                "control_hz": args.control_hz,
                "steps": 0,
                "clip_count": 0,
                **home_info,
            }
    elif args.move_home_each_trial:
        print("[PIPERX_EVAL] DRY RUN: skipping slow home because no robot command will be sent.")
        home_info["home_stop_reason"] = "dry_run_skipped"

    input("[PIPERX_EVAL] Home is ready. Press ENTER to start this trial...")
    warmup_reads, warmup_elapsed_s = warm_up_observations(
        robot, args.camera_warmup_s, args.camera_fps
    )
    if warmup_reads:
        print(
            "[PIPERX_EVAL] Camera cache drained before trial: "
            f"reads={warmup_reads}, elapsed={warmup_elapsed_s:.2f}s"
        )
    print("[PIPERX_EVAL] Trial running. Press ENTER to stop early.")

    chunk: deque[np.ndarray] = deque()
    ee_target_t: np.ndarray | None = None
    period = 1.0 / args.control_hz
    deadline = time.perf_counter() + args.trial_duration_s
    next_t = time.perf_counter()
    last_print_t = 0.0
    steps = 0
    clip_count = 0
    stop_reason = "timeout"

    while time.perf_counter() < deadline:
        key = poll_keyboard()
        if key == "":
            stop_reason = "user"
            break
        if key == "q":
            stop_reason = "quit"
            break

        obs = robot.get_observation()
        if not chunk:
            try:
                response = client.predict_action(policy_query(client, obs, instruction, args, state_stats, ee_controller))
                action_chunk = extract_action_chunk(
                    response, args.action_input, stats, args.debug_actions, args.action_norm
                )
            except Exception as exc:
                print(f"\n[PIPERX_EVAL] Policy inference failed: {exc}")
                stop_reason = "server_error"
                break
            if args.action_chunk_stride > 1:
                action_chunk = action_chunk[:: args.action_chunk_stride]
            if args.debug_chunk_summary:
                print(format_chunk_summary(action_chunk, args.debug_chunk_rows))
            chunk.extend(action_chunk)
            if args.action_space == "delta-ee":
                # Temporal delta actions describe consecutive desired EE poses.
                # Anchor once to feedback, then integrate the complete chunk.
                ee_target_t = ee_controller.fk(observation_joint_state(obs))

        raw_target = chunk.popleft()
        raw_gripper_mm = float(raw_target[6])
        raw_target[6] = gripper_target_mm(raw_gripper_mm, args, previous_mm=float(current_cmd[6]))
        if args.safe:
            target, was_clipped = clip_action_to_safe_range(raw_target, safe_low, safe_high)
            clip_count += int(was_clipped)
        else:
            target = raw_target

        ik_info: dict[str, Any] = {}
        ee_action_for_debug = target.copy()
        if args.action_space == "delta-ee":
            if ee_controller is None:
                raise RuntimeError("delta-ee action space needs an EE controller")
            feedback = observation_joint_state(obs)
            target, ik_info = ee_controller.delta_to_joint_target(
                target,
                feedback,
                current_cmd,
                reference_t=ee_target_t,
            )
            ee_target_t = ik_info.pop("target_ee_transform", ee_target_t)
            clip_count += int(ik_info.get("ee_delta_clipped", False))
            clip_count += int(ik_info.get("ee_workspace_clipped", False))
            if not ik_info.get("ik_success", False):
                clip_count += 1
                target = current_cmd.copy()
        elif args.safe:
            target, joint_was_clipped = clip_action_to_safe_range(target, PIPERX_HARD_LIMITS_DEG[:, 0], PIPERX_HARD_LIMITS_DEG[:, 1])
            clip_count += int(joint_was_clipped)

        current_cmd = limit_joint_step(target, current_cmd, args.max_joint_speed_deg_s, args.control_hz)

        if args.execute:
            robot.send_joint_command(current_cmd, speed_ratio=args.speed_ratio)

        steps += 1
        now = time.perf_counter()
        if now - last_print_t >= 1.0 / max(args.print_hz, 0.1):
            remaining = max(0.0, deadline - now)
            debug_ee = ""
            if args.debug_actions and args.action_space == "delta-ee":
                target_ee = ik_info.get("target_ee_pos_m")
                if target_ee is not None:
                    target_ee_mm = [float(v) * 1000.0 for v in target_ee]
                    debug_ee = (
                        f" dxyz_mm=[{ee_action_for_debug[0] * 1000:.1f},{ee_action_for_debug[1] * 1000:.1f},{ee_action_for_debug[2] * 1000:.1f}]"
                        f" target_ee_mm=[{target_ee_mm[0]:.1f},{target_ee_mm[1]:.1f},{target_ee_mm[2]:.1f}]"
                    )
            print(
                "[PIPERX_EVAL_TARGET] "
                + " ".join(f"j{i + 1}={current_cmd[i]:7.2f}" for i in range(6))
                + f" gripper={current_cmd[6]:6.1f}mm queued={len(chunk)} "
                + f"clips={clip_count} left={remaining:4.1f}s"
                + (f" ik_pos={ik_info.get('ik_pos_err_m', 0.0) * 1000:.1f}mm" if args.action_space == "delta-ee" else "")
                + (" ee_ws_clip=1" if ik_info.get("ee_workspace_clipped", False) else "")
                + (f" raw_grip={raw_gripper_mm:5.1f}mm" if args.debug_actions else "")
                + debug_ee
            )
            last_print_t = now

        next_t += period
        sleep_s = next_t - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            next_t = time.perf_counter()

    if stop_reason == "quit":
        label = "quit"
    else:
        label = read_label()
    return {
        "time": datetime.now().isoformat(timespec="seconds"),
        "trial_index": trial_index,
        "instruction": instruction,
        "label": label,
        "stop_reason": stop_reason,
        "execute": args.execute,
        "control_hz": args.control_hz,
        "steps": steps,
        "clip_count": clip_count,
        "camera_warmup_reads": warmup_reads,
        "camera_warmup_elapsed_s": warmup_elapsed_s,
        **home_info,
    }


def main() -> None:
    args = build_argparser().parse_args()
    if args.control_hz <= 0:
        raise ValueError("--control-hz must be > 0")
    if args.action_chunk_stride <= 0:
        raise ValueError("--action-chunk-stride must be > 0")
    if args.trial_duration_s <= 0:
        raise ValueError("--trial-duration-s must be > 0")
    if not args.gripper_close_mm < args.gripper_binary_threshold < args.gripper_open_mm:
        raise ValueError(
            "--gripper-binary-threshold must be strictly between "
            "--gripper-close-mm and --gripper-open-mm"
        )

    action_stats, state_stats = load_policy_stats(args, "[PIPERX_EVAL]")
    print(f"[PIPERX_EVAL] normalization: action={args.action_norm} state={args.state_norm}")
    if args.action_norm == "min_max" or args.state_norm == "min_max":
        print(
            "[PIPERX_EVAL] WARNING: using min_max normalization. The released PiperX "
            "checkpoints were trained with q99 state/action normalization."
        )
    configure_ee_workspace(args, state_stats)
    safe_low, safe_high = safe_action_bounds(action_stats, args)
    ee_controller = DeltaEEToJointController(args) if args.action_space == "delta-ee" else None
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    robot = build_robot(args)

    print("[PIPERX_EVAL] Connecting robot and cameras...")
    robot.connect()
    print("[PIPERX_EVAL] Connected.")
    print("[PIPERX_EVAL] Session mode: one instruction per trial; Ctrl+C also stops.")
    print(
        "[PIPERX_EVAL] Safety: "
        f"safe={args.safe}, control_hz={args.control_hz}, "
        f"action_space={args.action_space}, policy_state_space={args.policy_state_space}, "
        f"trial_limit={args.trial_duration_s}s, max_joint_speed={args.max_joint_speed_deg_s}deg/s"
    )
    if args.gripper_action_mode == "binary":
        print(
            "[PIPERX_EVAL] Gripper binary rule: "
            f"prediction < {args.gripper_binary_threshold:.1f}mm -> "
            f"CLOSE {args.gripper_close_mm:.1f}mm; otherwise -> "
            f"OPEN {args.gripper_open_mm:.1f}mm"
        )
    if args.safe:
        print(
            "[PIPERX_EVAL] Safe action low/high:\n  low = "
            + " ".join(f"{v:.4f}" for v in safe_low)
            + "\n  high= "
            + " ".join(f"{v:.4f}" for v in safe_high)
        )
    if args.action_space == "delta-ee" and args.ee_workspace is not None:
        x0, x1, y0, y1, z0, z1 = args.ee_workspace
        print(
            "[PIPERX_EVAL] EE workspace xyz m: "
            f"x=[{x0:.3f},{x1:.3f}] y=[{y0:.3f},{y1:.3f}] z=[{z0:.3f},{z1:.3f}]"
        )
    if not args.execute:
        print("[PIPERX_EVAL] DRY RUN: policy is queried, but actions are not sent to the robot.")

    try:
        trial_index = 0
        last_instruction = args.instruction
        while args.max_trials <= 0 or trial_index < args.max_trials:
            entered_instruction = input(
                f"\nInstruction for trial {trial_index} "
                f"(ENTER=previous: {last_instruction!r}, q=quit): "
            ).strip()
            if entered_instruction.lower() in {"q", "quit", "exit"}:
                break
            instruction = entered_instruction or last_instruction
            last_instruction = instruction
            reset_policy_cache(client, "trial_start", trial_index, instruction)
            record = run_trial(
                client,
                robot,
                instruction,
                trial_index,
                args,
                action_stats,
                state_stats,
                safe_low,
                safe_high,
                ee_controller,
            )
            append_result(args.results_jsonl, record)
            print(f"[PIPERX_EVAL] Saved result: {record}")
            trial_index += 1
            if record["label"] == "quit" or record["stop_reason"] == "quit":
                break

    except KeyboardInterrupt:
        print("\n[PIPERX_EVAL] Stopping.")
    finally:
        try:
            client.close()
        finally:
            if robot.is_connected:
                robot.disconnect()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a StairVLA policy on a real AgileX PiperX arm (dry run unless --execute).",
        epilog=(
            "Defaults are the paper's Fruit25 settings (8 Hz). For PushBlock pass --control-hz 20 "
            "--gripper-action-mode absolute (paper settings; see examples/PiperX/deployment/README.md)."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1", help="StairVLA policy server host.")
    parser.add_argument("--port", type=int, default=10093, help="StairVLA policy server port.")
    parser.add_argument("--instruction", default=DEFAULT_TASK)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=f"Checkpoint .pt file or run directory; {STATS_FILENAME} is found by walking up its parents.",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=None,
        help=f"Explicit {STATS_FILENAME} used to (un)normalize policy actions and state (overrides --checkpoint).",
    )
    parser.add_argument("--state-stats-json", type=Path, default=None, help="Optional separate stats used to normalize policy state.")
    parser.add_argument("--results-jsonl", type=Path, default=DEFAULT_RESULTS_JSONL)
    parser.add_argument("--action-input", choices=["auto", "normalized", "absolute"], default="auto")
    # Must match the training data config's normalization_modes (q99 for the PiperX configs).
    parser.add_argument("--action-norm", choices=list(NORMALIZATION_MODES), default=DEFAULT_NORMALIZATION_MODE)
    parser.add_argument("--state-norm", choices=list(NORMALIZATION_MODES), default=DEFAULT_NORMALIZATION_MODE)
    parser.add_argument(
        "--action-space",
        choices=["delta-ee", "joint"],
        default="delta-ee",
        help="delta-ee: 7D EE delta actions (paper). joint: LEGACY absolute joint actions; needs --policy-state-space joint.",
    )
    parser.add_argument(
        "--policy-state-space",
        choices=["ee", "ee7", "joint"],
        default="ee",
        help="ee: 8D EE state (paper). ee7: EE state without gripper. joint: LEGACY 7D joint state.",
    )
    parser.add_argument("--use-ddim", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-ddim-steps", type=int, default=10)

    add_robot_args(parser)
    parser.add_argument("--execute", action="store_true", help="Actually send actions to the robot.")
    parser.add_argument(
        "--control-hz",
        "--fps",
        dest="control_hz",
        type=float,
        default=8.0,
        help="Control rate; must match the dataset rate (Fruit25: 8, PushBlock: 20).",
    )
    parser.add_argument("--max-joint-speed-deg-s", type=float, default=25.0)
    parser.add_argument("--trial-duration-s", type=float, default=30.0)
    parser.add_argument("--max-trials", type=int, default=0, help="0 means keep asking for new trials.")
    parser.add_argument("--action-chunk-stride", type=int, default=1)
    parser.add_argument("--safe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--safe-range", choices=["min_max", "q01_q99"], default="min_max")
    parser.add_argument("--safe-margin-deg", type=float, default=2.0)
    parser.add_argument("--safe-margin-gripper-mm", type=float, default=0.0)
    parser.add_argument("--safe-margin-ee-m", type=float, default=0.0)
    parser.add_argument("--safe-margin-ee-rot-rad", type=float, default=0.0)
    parser.add_argument("--max-ee-delta-m", type=float, default=0.05, help="Per-step translation clip (m, per axis).")
    parser.add_argument("--max-ee-rot-delta-rad", type=float, default=0.20, help="Per-step rotation clip (rad, norm).")
    parser.add_argument(
        "--ee-workspace",
        nargs=6,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="Absolute EE xyz workspace in meters. Delta-EE targets are clamped inside it.",
    )
    parser.add_argument(
        "--ee-workspace-from-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --ee-workspace is omitted, derive xyz bounds from state stats.",
    )
    parser.add_argument("--ee-workspace-margin-m", type=float, default=0.02)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="PiperX URDF (default: bundled copy).")
    parser.add_argument("--ik-max-nfev", type=int, default=60)
    parser.add_argument("--ik-rot-weight", type=float, default=0.25)
    parser.add_argument("--ik-joint-regularization", type=float, default=0.02)
    parser.add_argument("--ik-pos-tolerance-m", type=float, default=0.008)
    parser.add_argument("--ik-rot-tolerance-rad", type=float, default=0.12)

    parser.add_argument("--move-home-each-trial", "--move-home-on-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--home-joints-deg", nargs=6, type=float, default=DEFAULT_HOME_JOINTS_DEG)
    parser.add_argument("--home-speed-ratio", type=int, default=2)
    parser.add_argument("--home-max-joint-speed-deg-s", type=float, default=8.0)
    parser.add_argument("--home-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--home-timeout-s", type=float, default=30.0)

    parser.add_argument("--gripper-open-mm", type=float, default=100.0)
    parser.add_argument("--gripper-close-mm", type=float, default=0.0)
    parser.add_argument("--gripper-action-mode", choices=["binary", "absolute"], default="binary")
    parser.add_argument(
        "--gripper-binary-threshold",
        type=float,
        default=80.6,
        help=(
            "For --gripper-action-mode binary, predictions below this aperture "
            "close fully and predictions at or above it open fully. The 80.6 mm "
            "default is the Otsu split of the Fruit25 gripper actions."
        ),
    )
    parser.add_argument("--gripper-hysteresis", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gripper-close-threshold-mm", type=float, default=35.0)
    parser.add_argument("--gripper-open-threshold-mm", type=float, default=65.0)

    parser.add_argument(
        "--camera-warmup-s",
        type=float,
        default=0.5,
        help=(
            "After the operator starts each trial, discard observations for this many seconds "
            "before recording or policy inference. This drains camera/driver frame queues."
        ),
    )
    parser.add_argument("--image-order", nargs="+", default=["top", "wrist"])
    parser.add_argument("--print-hz", type=float, default=5.0)
    parser.add_argument("--debug-actions", action="store_true", help="Print raw and unnormalized policy action chunks.")
    parser.add_argument("--debug-chunk-summary", action="store_true", help="Print every newly predicted action chunk after unnormalization.")
    parser.add_argument("--debug-chunk-rows", type=int, default=8, help="Maximum action rows to print for each chunk summary.")
    return parser


if __name__ == "__main__":
    main()
