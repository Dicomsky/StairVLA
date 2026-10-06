#!/usr/bin/env python3
"""Replay a recorded EE-delta episode (LeRobot v3.0 dataset) through the PiperX delta-EE controller.

Uses the same delta-EE -> IK -> rate-limit path as ``eval_policy.py``, so it checks the
dataset, the kinematics and the arm without a policy in the loop.

Dry-run is the default: no CAN connection is opened and the IK is integrated from a
simulated command. Add --execute to connect to the robot and send JointCtrl.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.PiperX.common.kinematics import DEFAULT_URDF
from examples.PiperX.common.robot import (
    DEFAULT_HOME_JOINTS_DEG,
    PIPERX_HARD_LIMITS_DEG,
    add_robot_args,
    limit_joint_step,
)
from examples.PiperX.deployment.eval_policy import (
    DeltaEEToJointController,
    build_robot,
    clip_action_to_safe_range,
    observation_joint_state,
    poll_keyboard,
)

DEFAULT_VIDEO_OUTPUT_DIR = Path("outputs/piperx_replay")
ACTION_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]


def load_tasks(dataset: Path) -> dict[int, str]:
    tasks_path = dataset / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return {}
    tasks = pd.read_parquet(tasks_path)
    if "task" not in tasks.columns and "task_index" in tasks.columns:
        return {int(row["task_index"]): str(index) for index, row in tasks.iterrows()}
    if "task" in tasks.columns and "task_index" in tasks.columns:
        return {int(row["task_index"]): str(row["task"]) for _, row in tasks.iterrows()}
    return {}


def load_info(dataset: Path) -> dict[str, Any]:
    info_path = dataset / "meta" / "info.json"
    if not info_path.exists():
        return {}
    with info_path.open("r") as f:
        return json.load(f)


def load_episode_table(dataset: Path) -> pd.DataFrame:
    frames = []
    for path in sorted((dataset / "data").glob("*/*.parquet")):
        frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(f"No episode parquet files found under {dataset / 'data'}")
    data = pd.concat(frames, ignore_index=True)
    required = {"episode_index", "frame_index", "timestamp", "observation.state", "action"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise KeyError(f"Dataset is missing required columns: {missing}")
    return data


def select_episode(data: pd.DataFrame, episode: int | None, random_seed: int) -> int:
    episodes = sorted(int(v) for v in data["episode_index"].unique())
    if not episodes:
        raise ValueError("Dataset contains no episodes.")
    if episode is not None:
        if episode not in episodes:
            raise ValueError(f"Episode {episode} not found. Available range: {episodes[0]}..{episodes[-1]}")
        return episode
    rng = random.Random(random_seed)
    return rng.choice(episodes)


def episode_frames(data: pd.DataFrame, episode: int) -> pd.DataFrame:
    ep = data[data["episode_index"] == episode].copy()
    if ep.empty:
        raise ValueError(f"Episode {episode} is empty.")
    return ep.sort_values("frame_index").reset_index(drop=True)


def ee_state_to_transform(state: np.ndarray) -> np.ndarray:
    if state.shape[0] < 7:
        raise ValueError(f"Expected EE state dim >= 7, got {state.shape}")
    t = np.eye(4, dtype=np.float64)
    t[:3, 3] = state[:3]
    t[:3, :3] = R.from_quat(state[3:7]).as_matrix()
    return t


def apply_delta_to_ee_state(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Use eval's EE convention: world xyz delta, local/body rotation delta."""
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    target = state.copy()
    target[:3] = state[:3] + action[:3]
    target[3:7] = (R.from_quat(state[3:7]) * R.from_rotvec(action[3:6])).as_quat()
    target[7] = action[6]
    return target.astype(np.float32)


def solve_ee_pose_to_joints(
    controller: DeltaEEToJointController,
    target_state: np.ndarray,
    current_joints: np.ndarray,
    current_cmd: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    target_t = ee_state_to_transform(target_state)
    preferred = np.deg2rad(np.asarray(current_cmd[:6], dtype=np.float64))
    current_q = np.deg2rad(np.asarray(current_joints[:6], dtype=np.float64))
    best = None
    for seed in controller.seed_candidates(controller.kin, preferred) + controller.seed_candidates(controller.kin, current_q):
        result = controller.kin.solve_ik(
            target_t,
            seed,
            pos_weight=1.0,
            rot_weight=controller.args.ik_rot_weight,
            max_nfev=controller.args.ik_max_nfev,
            regularization_weight=controller.args.ik_joint_regularization,
            regularization_target=preferred,
        )
        solved_t = controller.kin.fk(result.x)
        pos_err = float(np.linalg.norm(solved_t[:3, 3] - target_t[:3, 3]))
        rot_err = float(np.linalg.norm(R.from_matrix(target_t[:3, :3].T @ solved_t[:3, :3]).as_rotvec()))
        score = pos_err + controller.args.ik_rot_weight * rot_err + 0.002 * float(np.linalg.norm(result.x - preferred))
        if best is None or score < best[0]:
            best = (score, result, pos_err, rot_err)

    if best is None:
        return current_cmd.copy(), {"ik_success": False, "ik_reason": "no_result"}
    _, result, pos_err, rot_err = best
    success = bool(result.success) and pos_err <= controller.args.ik_pos_tolerance_m and rot_err <= controller.args.ik_rot_tolerance_rad
    if not success:
        return current_cmd.copy(), {
            "ik_success": False,
            "ik_reason": "tolerance",
            "ik_pos_err_m": pos_err,
            "ik_rot_err_rad": rot_err,
        }

    target = current_cmd.copy()
    target[:6] = np.rad2deg(result.x).astype(np.float32)
    target[6] = float(target_state[7]) if target_state.shape[0] > 7 else current_cmd[6]
    return target, {
        "ik_success": True,
        "ik_pos_err_m": pos_err,
        "ik_rot_err_rad": rot_err,
    }


def format_joints(joints: np.ndarray) -> str:
    return " ".join(f"j{i + 1}={joints[i]:7.2f}" for i in range(6)) + f" gripper={joints[6]:6.1f}mm"


def format_ee_state(state: np.ndarray) -> str:
    xyz = state[:3]
    quat = state[3:7]
    grip = state[7] if state.shape[0] > 7 else np.nan
    return (
        f"x={xyz[0]:.4f}m y={xyz[1]:.4f}m z={xyz[2]:.4f}m | "
        f"qx={quat[0]:.4f} qy={quat[1]:.4f} qz={quat[2]:.4f} qw={quat[3]:.4f} | "
        f"gripper={grip:.1f}mm"
    )


def action_safety_bounds(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    low = np.asarray(
        [
            -args.max_ee_delta_m,
            -args.max_ee_delta_m,
            -args.max_ee_delta_m,
            -args.max_ee_rot_delta_rad,
            -args.max_ee_rot_delta_rad,
            -args.max_ee_rot_delta_rad,
            args.gripper_close_mm,
        ],
        dtype=np.float32,
    )
    high = np.asarray(
        [
            args.max_ee_delta_m,
            args.max_ee_delta_m,
            args.max_ee_delta_m,
            args.max_ee_rot_delta_rad,
            args.max_ee_rot_delta_rad,
            args.max_ee_rot_delta_rad,
            args.gripper_open_mm,
        ],
        dtype=np.float32,
    )
    return low, high


def move_to_episode_start(
    robot,
    controller: DeltaEEToJointController,
    target_ee_state: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, bool]:
    if args.execute:
        current = observation_joint_state(robot.get_observation())
    else:
        current = np.asarray([*args.dry_run_seed_joints_deg, target_ee_state[7]], dtype=np.float32)
    current_cmd = current.copy()
    target_joints, ik_info = solve_ee_pose_to_joints(controller, target_ee_state, current, current_cmd)
    print(f"[REPLAY] episode-start IK: success={ik_info.get('ik_success')} "
          f"pos_err={ik_info.get('ik_pos_err_m', 0.0) * 1000:.2f}mm "
          f"rot_err={np.rad2deg(ik_info.get('ik_rot_err_rad', 0.0)):.2f}deg")
    if not ik_info.get("ik_success", False):
        return current_cmd, False

    if not args.move_to_episode_start:
        print("[REPLAY] --no-move-to-episode-start set; using current command as replay seed.")
        return current_cmd, True

    print(f"[REPLAY] Moving slowly to episode start: {format_joints(target_joints)}")
    period = 1.0 / max(args.replay_hz, 1.0)
    start_t = time.perf_counter()
    last_print = 0.0
    motion_mode_set = False
    while True:
        key = poll_keyboard()
        if key == "" or key == "q":
            print("\n[REPLAY] Home canceled by user.")
            return current_cmd, False

        feedback = observation_joint_state(robot.get_observation()) if args.execute else current_cmd
        remaining = float(np.max(np.abs(target_joints[:6] - feedback[:6])))
        if remaining <= args.home_tolerance_deg:
            print(f"\n[REPLAY] Episode start reached. {format_joints(feedback)}")
            return target_joints.copy(), True
        elapsed = time.perf_counter() - start_t
        if elapsed >= args.home_timeout_s:
            print(f"\n[REPLAY] Episode-start timeout. remaining={remaining:.1f}deg")
            return current_cmd, False

        current_cmd = limit_joint_step(target_joints, current_cmd, args.home_max_joint_speed_deg_s, args.replay_hz)
        if args.execute:
            robot.send_joint_command(current_cmd, speed_ratio=args.home_speed_ratio if not motion_mode_set else None)
            motion_mode_set = True

        now = time.perf_counter()
        if now - last_print >= 1.0 / max(args.print_hz, 0.1):
            print(f"[REPLAY] home remaining={remaining:5.1f}deg elapsed={elapsed:4.1f}s", end="\r", flush=True)
            last_print = now
        time.sleep(period)


def replay_episode(
    robot,
    controller: DeltaEEToJointController,
    ep: pd.DataFrame,
    current_cmd: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    low, high = action_safety_bounds(args)
    stride = max(1, int(args.frame_stride))
    sampled = ep.iloc[::stride].reset_index(drop=True)
    max_steps = min(len(sampled), int(max(1, round(args.max_duration_s * args.replay_hz))))
    sampled = sampled.iloc[:max_steps]
    print(
        f"[REPLAY] source_frames={len(ep)} replay_steps={len(sampled)} "
        f"frame_stride={stride} replay_hz={args.replay_hz} base={args.replay_base}"
    )
    if stride > 1:
        print("[REPLAY] Warning: frame_stride > 1 skips delta-EE actions and changes the integrated path.")
    print("[REPLAY] Running. Press ENTER or type q + ENTER to stop.")

    period = 1.0 / args.replay_hz
    next_t = time.perf_counter()
    last_print = 0.0
    ik_skip = 0
    clip_count = 0
    steps = 0
    stop_reason = "done"
    ee_target_t: np.ndarray | None = None
    motion_mode_set = False

    for replay_index, (_, row) in enumerate(sampled.iterrows()):
        key = poll_keyboard()
        if key == "" or key == "q":
            stop_reason = "user"
            break

        raw_action = np.asarray(row["action"], dtype=np.float32).copy()
        if args.safe:
            safe_action, was_clipped = clip_action_to_safe_range(raw_action, low, high)
            clip_count += int(was_clipped)
        else:
            safe_action = raw_action
            was_clipped = False

        feedback = observation_joint_state(robot.get_observation()) if args.execute else current_cmd
        tracking_lag_deg = float(np.max(np.abs(current_cmd[:6] - feedback[:6])))
        if args.replay_base in {"integrated", "eval-chunk"}:
            should_anchor = replay_index == 0 or (
                args.replay_base == "eval-chunk" and replay_index % args.action_chunk_horizon == 0
            )
            if should_anchor:
                if ee_target_t is not None:
                    feedback_t = controller.fk(feedback)
                    reanchor_pos_mm = float(np.linalg.norm(ee_target_t[:3, 3] - feedback_t[:3, 3]) * 1000.0)
                    reanchor_rot_deg = float(
                        np.degrees(
                            np.linalg.norm(
                                R.from_matrix(feedback_t[:3, :3].T @ ee_target_t[:3, :3]).as_rotvec()
                            )
                        )
                    )
                    print(
                        f"[REPLAY_REANCHOR] step={replay_index + 1} "
                        f"dropped_residual={reanchor_pos_mm:.1f}mm/{reanchor_rot_deg:.2f}deg "
                        f"joint_track_lag={tracking_lag_deg:.1f}deg"
                    )
                ee_target_t = controller.fk(feedback)
            target, ik_info = controller.delta_to_joint_target(
                safe_action,
                feedback,
                current_cmd,
                reference_t=ee_target_t,
            )
            ee_target_t = ik_info.pop("target_ee_transform", ee_target_t)
        elif args.replay_base == "feedback":
            target, ik_info = controller.delta_to_joint_target(safe_action, feedback, current_cmd)
        else:
            dataset_state = np.asarray(row["observation.state"], dtype=np.float32)
            target_ee_state = apply_delta_to_ee_state(dataset_state, safe_action)
            target, ik_info = solve_ee_pose_to_joints(controller, target_ee_state, feedback, current_cmd)
        if not ik_info.get("ik_success", False):
            ik_skip += 1
            target = current_cmd.copy()
        else:
            target, joint_was_clipped = clip_action_to_safe_range(target, PIPERX_HARD_LIMITS_DEG[:, 0], PIPERX_HARD_LIMITS_DEG[:, 1])
            clip_count += int(joint_was_clipped)

        limited_cmd = limit_joint_step(target, current_cmd, args.max_joint_speed_deg_s, args.replay_hz)
        rate_limit_lag_deg = float(np.max(np.abs(target[:6] - limited_cmd[:6])))
        current_cmd = limited_cmd
        if args.execute:
            robot.send_joint_command(current_cmd, speed_ratio=args.speed_ratio if not motion_mode_set else None)
            motion_mode_set = True

        steps += 1
        now = time.perf_counter()
        if now - last_print >= 1.0 / max(args.print_hz, 0.1):
            print(
                "[REPLAY_TARGET] "
                + format_joints(current_cmd)
                + f" step={steps}/{len(sampled)} ik_skip={ik_skip} clips={clip_count}"
                + f" ik_pos={ik_info.get('ik_pos_err_m', 0.0) * 1000:.1f}mm"
                + f" track_lag={tracking_lag_deg:.1f}deg"
                + f" rate_lag={rate_limit_lag_deg:.1f}deg"
                + (" delta_clip=1" if was_clipped else "")
            )
            last_print = now

        next_t += period
        sleep_s = next_t - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            next_t = time.perf_counter()

    return {
        "steps": steps,
        "ik_skip": ik_skip,
        "clip_count": clip_count,
        "stop_reason": stop_reason,
    }


def start_episode_video_preview(args: argparse.Namespace, episode_id: int) -> Any | None:
    if args.no_video:
        return None
    try:
        from examples.PiperX.dataset_tools.inspect_episode import (
            export_videos,
            load_episode_meta,
            make_action_replay_video,
            make_full_replay_video,
            make_side_by_side_video,
        )
    except Exception as exc:
        print(f"[REPLAY] Video preview disabled: failed to import viewer helpers: {exc}")
        return None

    try:
        meta = load_episode_meta(args.dataset)
        match = meta[meta["episode_index"] == episode_id]
        if match.empty:
            print(f"[REPLAY] Video preview skipped: episode {episode_id} metadata not found.")
            return None
        row = match.iloc[0]
        ep = episode_frames(load_episode_table(args.dataset), episode_id)
        args.video_output_dir.mkdir(parents=True, exist_ok=True)
        videos = export_videos(args.dataset, row, args.video_output_dir, episode_id, reencode=not args.copy_video)
        side_by_side = make_side_by_side_video(videos, args.video_output_dir, episode_id)
        action_video = make_action_replay_video(ep, args.video_output_dir, episode_id, fps=float(args.replay_hz))
        full_replay = make_full_replay_video(videos, action_video, args.video_output_dir, episode_id)
        replay_path = full_replay or side_by_side or action_video
        if replay_path is None:
            print("[REPLAY] Video preview skipped: exported video clip has no readable stream.")
            return None
        print(f"[REPLAY] video preview: {replay_path}")
        if args.no_play_video or replay_path is None:
            return None

        import shutil
        import subprocess

        ffplay = shutil.which("ffplay")
        if not ffplay:
            print("[REPLAY] ffplay not found; video exported but not auto-played.")
            return None
        return subprocess.Popen([ffplay, "-hide_banner", "-loglevel", "error", "-autoexit", str(replay_path)])
    except Exception as exc:
        print(f"[REPLAY] Video preview failed, continuing robot replay: {exc}")
        return None


def stop_video_preview(proc: Any | None) -> None:
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
    except Exception:
        pass


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay one PiperX EE-delta LeRobot episode through delta-EE IK (dry run unless --execute)."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="LeRobot v3.0 EE dataset root (contains meta/, data/, videos/).")
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--interactive", action="store_true", help="Keep robot connected and replay episode ids entered at the prompt.")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    add_robot_args(parser, speed_ratio=10)
    parser.add_argument("--home-speed-ratio", type=int, default=2)
    parser.add_argument("--replay-hz", type=float, default=None, help="Replay frequency. Defaults to the dataset fps.")
    parser.add_argument(
        "--replay-base",
        choices=["integrated", "eval-chunk", "dataset", "feedback"],
        default="dataset",
        help=(
            "integrated anchors once and integrates the complete temporal trajectory. eval-chunk "
            "matches policy eval by re-anchoring once per action chunk. dataset solves each recorded "
            "absolute EE target. feedback applies every delta directly to current robot feedback."
        ),
    )
    parser.add_argument(
        "--action-chunk-horizon",
        type=int,
        default=8,
        help="Number of temporal delta actions integrated before eval-chunk re-anchors to feedback.",
    )
    parser.add_argument("--frame-stride", type=int, default=1, help="Default 1 replays every delta-EE action. Values >1 are only for debugging.")
    parser.add_argument("--max-duration-s", type=float, default=30.0)
    parser.add_argument("--max-joint-speed-deg-s", type=float, default=8.0)
    parser.add_argument("--home-max-joint-speed-deg-s", type=float, default=5.0)
    parser.add_argument("--home-timeout-s", type=float, default=45.0)
    parser.add_argument("--home-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--move-to-episode-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="PiperX URDF (default: bundled copy).")
    parser.add_argument("--ik-max-nfev", type=int, default=60)
    parser.add_argument("--ik-rot-weight", type=float, default=0.25)
    parser.add_argument("--ik-joint-regularization", type=float, default=0.02)
    parser.add_argument("--ik-pos-tolerance-m", type=float, default=0.008)
    parser.add_argument("--ik-rot-tolerance-rad", type=float, default=0.12)
    parser.add_argument("--ee-workspace", nargs=6, type=float, default=None, help="Optional EE xyz workspace in meters: xmin xmax ymin ymax zmin zmax.")
    parser.add_argument("--safe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-ee-delta-m", type=float, default=0.05)
    parser.add_argument("--max-ee-rot-delta-rad", type=float, default=0.20)
    parser.add_argument("--gripper-open-mm", type=float, default=100.0)
    parser.add_argument("--gripper-close-mm", type=float, default=0.0)
    parser.add_argument("--print-hz", type=float, default=2.0)
    parser.add_argument("--video-output-dir", type=Path, default=DEFAULT_VIDEO_OUTPUT_DIR)
    parser.add_argument("--no-video", action="store_true", help="Do not export matching dataset videos.")
    parser.add_argument("--no-play-video", action="store_true", help="Export videos but do not launch ffplay.")
    parser.add_argument("--copy-video", action="store_true", help="Fast cut videos with stream copy instead of re-encoding.")
    parser.add_argument(
        "--dry-run-seed-joints-deg",
        nargs=6,
        type=float,
        default=DEFAULT_HOME_JOINTS_DEG,
        help="Seed joints used only when --execute is not set.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.max_duration_s <= 0:
        raise ValueError("--max-duration-s must be > 0")
    if args.action_chunk_horizon <= 0:
        raise ValueError("--action-chunk-horizon must be > 0")
    info = load_info(args.dataset)
    source_fps = float(info.get("fps", 30.0))
    if args.replay_hz is None:
        args.replay_hz = source_fps
    if args.replay_hz <= 0:
        raise ValueError("--replay-hz must be > 0")

    data = load_episode_table(args.dataset)
    tasks = load_tasks(args.dataset)

    print(f"[REPLAY] dataset={args.dataset}")
    print(f"[REPLAY] dataset_fps={source_fps}")
    print(f"[REPLAY] episodes={data['episode_index'].nunique()} frames={len(data)}")
    print(f"[REPLAY] execute={args.execute}. {'Robot will move.' if args.execute else 'DRY RUN: no CAN commands will be sent.'}")

    controller = DeltaEEToJointController(args)
    robot = None
    try:
        if args.execute:
            # Replay does not use the cameras.
            robot = build_robot(args, connect_cameras=False)
            robot.connect()
            print(f"[REPLAY] robot connected on {args.can}")

        if args.interactive:
            print("[REPLAY] Interactive mode. Enter episode ids to replay, e.g. 15 or 100-105; q to quit.")
            while True:
                text = input("episode> ").strip()
                if text.lower() in {"q", "quit", "exit"}:
                    break
                if not text:
                    continue
                try:
                    episode_ids: list[int] = []
                    for part in text.replace(",", " ").split():
                        if "-" in part:
                            lo, hi = part.split("-", 1)
                            episode_ids.extend(range(int(lo), int(hi) + 1))
                        else:
                            episode_ids.append(int(part))
                    for episode_id in episode_ids:
                        run_one_episode(data, tasks, robot, controller, episode_id, args)
                except Exception as exc:
                    print(f"[REPLAY] ERROR: {exc}")
            return

        episode_id = select_episode(data, args.episode, args.random_seed)
        run_one_episode(data, tasks, robot, controller, episode_id, args)
    except KeyboardInterrupt:
        print("\n[REPLAY] Stopped by Ctrl+C.")
    finally:
        if robot is not None and getattr(robot, "is_connected", False):
            robot.disconnect()
            print("[REPLAY] robot disconnected.")


def run_one_episode(
    data: pd.DataFrame,
    tasks: dict[int, str],
    robot: Any,
    controller: DeltaEEToJointController,
    episode_id: int,
    args: argparse.Namespace,
) -> None:
    ep = episode_frames(data, episode_id)
    task_index = int(ep.iloc[0].get("task_index", -1))
    task = tasks.get(task_index, "unknown")
    first_state = np.asarray(ep.iloc[0]["observation.state"], dtype=np.float32)

    print(f"\n[REPLAY] episode={episode_id} task_index={task_index} task={task}")
    print(f"[REPLAY] first EE state: {format_ee_state(first_state)}")
    current_cmd, ok = move_to_episode_start(robot, controller, first_state, args)
    if not ok:
        print("[REPLAY] Episode start was not reached; replay will not start.")
        return

    video_proc = start_episode_video_preview(args, episode_id)
    try:
        result = replay_episode(robot, controller, ep, current_cmd, args)
    finally:
        stop_video_preview(video_proc)
    print(
        f"[REPLAY] done episode={episode_id} stop_reason={result['stop_reason']} steps={result['steps']} "
        f"ik_skip={result['ik_skip']} clips={result['clip_count']}"
    )


if __name__ == "__main__":
    main()
