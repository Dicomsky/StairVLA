#!/usr/bin/env python3
"""Run a resumable, video-recorded PiperX policy benchmark (Fruit25 by default).

The benchmark schedule is deterministic: every task receives N accepted trials.
Success/failure accepts and saves the current slot. Retry discards the temporary
recording and repeats the same slot; quit discards it and stops. See BENCHMARK.md.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.PiperX.common.robot import milli_to_unit
from examples.PiperX.deployment import eval_policy as base
from examples.PiperX.deployment.phase_video import (
    ContinuousPhaseVideoRecorder,
    PhaseTimeline,
)

FRUIT25_TASKS = [
    "Pick up the apple and place it in the red basket.",
    "Pick up the apple and place it on the metal tray.",
    "Pick up the orange and place it on the metal tray.",
    "Pick up the orange and place it in the orange basket.",
    "Pick up the lemon and place it on the metal tray.",
    "Pick up the lemon and place it in the yellow basket.",
    "Pick up the lime and place it in the green basket.",
    "Pick up the apple and place it in the orange basket.",
    "Pick up the apple and place it in the blue basket.",
    "Pick up the apple and place it in the green basket.",
    "Pick up the apple and place it in the yellow basket.",
    "Pick up the orange and place it in the red basket.",
    "Pick up the orange and place it in the blue basket.",
    "Pick up the orange and place it in the green basket.",
    "Pick up the orange and place it in the yellow basket.",
    "Pick up the lemon and place it in the orange basket.",
    "Pick up the lemon and place it in the red basket.",
    "Pick up the lemon and place it in the blue basket.",
    "Pick up the lemon and place it in the green basket.",
    "Pick up the lime and place it in the orange basket.",
    "Pick up the lime and place it in the red basket.",
    "Pick up the lime and place it in the blue basket.",
    "Pick up the lime and place it in the yellow basket.",
    "Pick up the lime and place it on the metal tray.",
    "Sort the fruits into their corresponding baskets.",
]

# These 16 compositional tasks have roughly 20 training episodes each. The
# remaining Fruit25 tasks have substantially more data and keep the base count.
FRUIT25_LOW_DATA_TASK_INDICES = frozenset(range(7, 23))

PROTOCOL_VERSION = 1


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(jsonable(value), stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", buffering=1) as stream:
        stream.write(json.dumps(jsonable(value), ensure_ascii=False) + "\n")


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_tasks(path: Path | None) -> list[str]:
    if path is None:
        return FRUIT25_TASKS.copy()
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError("--tasks-json must contain a non-empty JSON list of instruction strings")
    return [item.strip() for item in value]


def read_attempt_label(demo: bool = False) -> str:
    if demo:
        # Demonstration runs are kept with their videos but never scored.
        while True:
            value = input("Demo run done: [ENTER] next task / [r] run this task again / [q] quit: ").strip().lower()
            if value in {"", "n", "next"}:
                return "demo"
            if value in {"r", "retry", "redo", "again"}:
                return "retry"
            if value in {"q", "quit", "exit"}:
                return "quit"
            print("Press ENTER, or type r or q.")
    while True:
        value = input(
            "Result: [s]uccess / [f]ailure / [r]etry same trial / [q]uit session: "
        ).strip().lower()
        if value in {"s", "success"}:
            return "success"
        if value in {"f", "failure", "fail"}:
            return "failure"
        if value in {"r", "retry", "redo", "again"}:
            return "retry"
        if value in {"q", "quit", "exit"}:
            return "quit"
        print("Please type s, f, r, or q.")


class AttemptRecorder:
    def __init__(self, directory: Path, cameras: list[str], fps: float, record_video: bool):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=False)
        self.cameras = cameras
        self.fps = fps
        self.record_video = record_video
        self.frame_stream = (directory / "frames.jsonl").open("w", buffering=1)
        self.chunk_stream = (directory / "chunks.jsonl").open("w", buffering=1)
        self.rows: list[dict[str, Any]] = []
        self.writers: dict[str, Any] = {}
        self.video_paths: dict[str, str] = {}
        self.closed = False

    def _writer(self, camera: str, image: np.ndarray):
        import cv2

        if camera in self.writers:
            return self.writers[camera]
        height, width = image.shape[:2]
        path = self.directory / f"{camera}.mp4"
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {path}")
        self.writers[camera] = writer
        self.video_paths[camera] = path.name
        return writer

    def write_frame(self, obs: dict[str, Any], row: dict[str, Any]) -> None:
        if self.record_video:
            import cv2

            for camera in self.cameras:
                if camera not in obs:
                    raise KeyError(f"Camera {camera!r} is absent from observation keys={list(obs)}")
                image = np.asarray(obs[camera])
                if image.ndim != 3 or image.shape[2] != 3:
                    raise ValueError(f"Camera {camera!r} returned invalid RGB shape {image.shape}")
                self._writer(camera, image).write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        clean = jsonable(row)
        self.rows.append(clean)
        self.frame_stream.write(json.dumps(clean, ensure_ascii=False) + "\n")

    def write_chunk(self, row: dict[str, Any]) -> None:
        self.chunk_stream.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.frame_stream.close()
        self.chunk_stream.close()
        for writer in self.writers.values():
            writer.release()
        if self.rows:
            try:
                import pandas as pd

                pd.DataFrame(self.rows).to_parquet(self.directory / "frames.parquet", index=False)
            except Exception as exc:
                print(f"[BENCHMARK] Warning: could not write frames.parquet: {exc}")


class HighRateStateRecorder:
    """Sample cached Piper feedback independently from policy and camera I/O."""

    def __init__(
        self,
        directory: Path,
        robot,
        ee_controller: base.DeltaEEToJointController,
        sample_hz: float,
        trial_start: float,
    ):
        self.directory = directory
        self.robot = robot
        self.ee_controller = ee_controller
        self.sample_hz = float(sample_hz)
        self.trial_start = trial_start
        self.rows: list[dict[str, Any]] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.sample_hz <= 0:
            return
        arm = getattr(self.robot, "arm", None)
        if arm is None or not hasattr(arm, "GetArmJointMsgs") or not hasattr(arm, "GetArmGripperMsgs"):
            raise RuntimeError("High-rate state logging requires a Piper SDK arm interface")
        self._thread = threading.Thread(target=self._sample_loop, name="piper-state-recorder", daemon=True)
        self._thread.start()

    def _sample_loop(self) -> None:
        period = 1.0 / self.sample_hz
        next_t = time.perf_counter()
        previous_joint_timestamp: float | None = None
        try:
            while not self._stop.is_set():
                sample_t = time.perf_counter()
                joint_msg = self.robot.arm.GetArmJointMsgs()
                joint_state = getattr(joint_msg, "joint_state", None)
                gripper_msg = self.robot.arm.GetArmGripperMsgs()
                gripper_state = getattr(gripper_msg, "gripper_state", None)
                joints = [
                    milli_to_unit(getattr(joint_state, f"joint_{index}", 0))
                    for index in range(1, 7)
                ]
                gripper_mm = abs(milli_to_unit(getattr(gripper_state, "grippers_angle", 0)))
                joint_timestamp = float(getattr(joint_msg, "time_stamp", 0.0))
                row = {
                    "sample_index": len(self.rows),
                    "wall_time_ns": time.time_ns(),
                    "monotonic_time_ns": time.monotonic_ns(),
                    "trial_elapsed_s": sample_t - self.trial_start,
                    "joint_feedback_deg": joints,
                    "gripper_feedback_mm": gripper_mm,
                    "joint_sdk_timestamp": joint_timestamp,
                    "gripper_sdk_timestamp": float(getattr(gripper_msg, "time_stamp", 0.0)),
                    "joint_sdk_hz": float(getattr(joint_msg, "Hz", 0.0)),
                    "gripper_sdk_hz": float(getattr(gripper_msg, "Hz", 0.0)),
                    "joint_feedback_updated": (
                        previous_joint_timestamp is None or joint_timestamp != previous_joint_timestamp
                    ),
                }
                if hasattr(self.robot.arm, "GetArmHighSpdInfoMsgs"):
                    high_msg = self.robot.arm.GetArmHighSpdInfoMsgs()
                    motors = [getattr(high_msg, f"motor_{index}", None) for index in range(1, 7)]
                    row.update(
                        {
                            "motor_speed_rad_s": [float(getattr(motor, "motor_speed", 0)) * 1e-3 for motor in motors],
                            "motor_current_a": [float(getattr(motor, "current", 0)) * 1e-3 for motor in motors],
                            "motor_effort_nm": [float(getattr(motor, "effort", 0)) * 1e-3 for motor in motors],
                            "motor_position_rad": [float(getattr(motor, "pos", 0)) * 1e-3 for motor in motors],
                            "motor_sdk_timestamp": float(getattr(high_msg, "time_stamp", 0.0)),
                            "motor_sdk_hz": float(getattr(high_msg, "Hz", 0.0)),
                        }
                    )
                self.rows.append(row)
                previous_joint_timestamp = joint_timestamp
                next_t += period
                wait_s = next_t - time.perf_counter()
                if wait_s > 0:
                    self._stop.wait(wait_s)
                else:
                    next_t = time.perf_counter()
        except Exception as exc:
            self.error = repr(exc)

    def stop_and_write(self) -> dict[str, Any]:
        if self._thread is None:
            return {
                "state_log_requested_hz": self.sample_hz,
                "state_log_samples": 0,
                "state_log_effective_hz": None,
                "state_log_unique_feedback_hz": None,
                "state_log_duplicate_ratio": None,
                "state_log_error": None,
            }
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            self.error = self.error or "state recorder thread did not stop within 2 seconds"

        for row in self.rows:
            joint_state = np.asarray([*row["joint_feedback_deg"], row["gripper_feedback_mm"]], dtype=np.float32)
            row["ee_feedback_xyz_quat_gripper"] = self.ee_controller.ee_state_from_joint_state(joint_state)

        jsonl_path = self.directory / "state_feedback_high_rate.jsonl"
        with jsonl_path.open("w") as stream:
            for row in self.rows:
                stream.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")
        if self.rows:
            try:
                import pandas as pd

                pd.DataFrame([jsonable(row) for row in self.rows]).to_parquet(
                    self.directory / "state_feedback_high_rate.parquet", index=False
                )
            except Exception as exc:
                print(f"[BENCHMARK] Warning: could not write high-rate state parquet: {exc}")

        elapsed = self.rows[-1]["trial_elapsed_s"] - self.rows[0]["trial_elapsed_s"] if len(self.rows) > 1 else 0.0
        effective_hz = (len(self.rows) - 1) / elapsed if elapsed > 0 else None
        updates = sum(bool(row["joint_feedback_updated"]) for row in self.rows)
        unique_hz = (updates - 1) / elapsed if elapsed > 0 and updates > 1 else None
        duplicate_ratio = 1.0 - updates / len(self.rows) if self.rows else None
        return {
            "state_log_requested_hz": self.sample_hz,
            "state_log_samples": len(self.rows),
            "state_log_effective_hz": effective_hz,
            "state_log_unique_feedback_hz": unique_hz,
            "state_log_duplicate_ratio": duplicate_ratio,
            "state_log_error": self.error,
        }


def pose_from_transform(transform: np.ndarray) -> list[float]:
    from scipy.spatial.transform import Rotation as Rotation

    quat = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return [*transform[:3, 3].tolist(), *quat.tolist()]


def normalized_chunk_from_response(response: dict[str, Any]) -> np.ndarray | None:
    data = response.get("data", response)
    for key in ("normalized_actions", "actions", "action"):
        if key not in data:
            continue
        chunk = np.asarray(data[key], dtype=np.float32)
        while chunk.ndim > 2:
            chunk = chunk[0]
        if chunk.ndim == 1:
            chunk = chunk[None, :]
        return chunk[:, :7]
    return None


def next_attempt_number(run_dir: Path, valid_trial_number: int) -> int:
    attempts_dir = run_dir / "attempts"
    prefix = f"trial_{valid_trial_number:04d}_attempt_"
    numbers = []
    for path in attempts_dir.glob(f"{prefix}*"):
        try:
            numbers.append(int(path.name[len(prefix) :].split("_", 1)[0]))
        except ValueError:
            continue
    return max(numbers, default=0) + 1


def read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open() as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_episode_counts(
    tasks: list[str], episodes_per_task: int, low_data_episodes_per_task: int | None
) -> list[int]:
    counts = [episodes_per_task] * len(tasks)
    if low_data_episodes_per_task is None:
        return counts
    if tasks != FRUIT25_TASKS:
        raise ValueError("--low-data-episodes-per-task is only supported with the built-in Fruit25 tasks")
    for task_index in FRUIT25_LOW_DATA_TASK_INDICES:
        counts[task_index] = low_data_episodes_per_task
    return counts


def build_schedule(tasks: list[str], episode_counts: list[int]) -> list[tuple[int, int, str]]:
    return [
        (task_index, repetition_index, instruction)
        for task_index, (instruction, count) in enumerate(zip(tasks, episode_counts))
        for repetition_index in range(count)
    ]


def build_summary(tasks: list[str], episode_counts: list[int], records: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [record for record in records if record.get("label") in {"success", "failure"}]
    successes = sum(record["label"] == "success" for record in accepted)
    per_task = []
    for task_index, instruction in enumerate(tasks):
        selected = [record for record in accepted if int(record["task_index"]) == task_index]
        task_successes = sum(record["label"] == "success" for record in selected)
        per_task.append(
            {
                "task_index": task_index,
                "instruction": instruction,
                "accepted": len(selected),
                "target": episode_counts[task_index],
                "successes": task_successes,
                "failures": len(selected) - task_successes,
                "success_rate": task_successes / len(selected) if selected else None,
            }
        )
    return {
        "updated_at": now_iso(),
        "protocol_version": PROTOCOL_VERSION,
        "accepted_trials": len(accepted),
        "target_trials": sum(episode_counts),
        "successes": successes,
        "failures": len(accepted) - successes,
        "success_rate": successes / len(accepted) if accepted else None,
        "per_task": per_task,
    }


def run_attempt(
    client: WebsocketClientPolicy,
    robot,
    instruction: str,
    task_index: int,
    repetition_index: int,
    valid_trial_number: int,
    attempt_number: int,
    attempt_dir: Path,
    args: argparse.Namespace,
    action_stats: dict[str, np.ndarray] | None,
    state_stats: dict[str, np.ndarray] | None,
    safe_low: np.ndarray,
    safe_high: np.ndarray,
    ee_controller: base.DeltaEEToJointController,
) -> dict[str, Any]:
    attempt_started_at = now_iso()
    obs = robot.get_observation()
    current = base.observation_joint_state(obs)
    current_cmd = current.copy()
    home = np.asarray([*args.home_joints_deg, args.gripper_open_mm], dtype=np.float32)
    home_info: dict[str, Any] = {
        "home_success": True,
        "home_stop_reason": "not_requested",
        "home_remaining_deg": None,
        "home_elapsed_s": 0.0,
    }
    if args.move_home_each_trial and args.execute:
        current_cmd, home_info = base.move_home(robot, home, args, current)
    elif args.move_home_each_trial:
        home_info["home_stop_reason"] = "dry_run_skipped"

    recorder = AttemptRecorder(attempt_dir, args.record_cameras, args.control_hz, args.record_video)
    if not home_info["home_success"]:
        recorder.close()
        print("[BENCHMARK] Home was not reached. This attempt cannot count as a policy result.")
        label = "quit" if home_info["home_stop_reason"] == "user" else read_attempt_label(args.demo)
        return {
            "protocol_version": PROTOCOL_VERSION,
            "attempt_started_at": attempt_started_at,
            "attempt_finished_at": now_iso(),
            "valid_trial_number": valid_trial_number,
            "task_index": task_index,
            "repetition_index": repetition_index,
            "attempt_number": attempt_number,
            "instruction": instruction,
            "label": label,
            "stop_reason": "home_failed",
            "attempt_dir": str(attempt_dir),
            "frame_count": 0,
            **home_info,
        }

    input("[BENCHMARK] Home ready. Arrange the scene, then press ENTER to start recording...")
    warmup_reads, warmup_elapsed_s = base.warm_up_observations(
        robot, args.camera_warmup_s, args.camera_fps
    )
    if warmup_reads:
        print(
            "[BENCHMARK] Camera cache drained before recording/inference: "
            f"reads={warmup_reads}, elapsed={warmup_elapsed_s:.2f}s"
        )
    print(
        f"[BENCHMARK] Trial running. Time limit={args.trial_duration_s:.1f}s. "
        "Press ENTER to stop early; type q + ENTER to quit."
    )

    chunk: deque[np.ndarray] = deque()
    ee_target_t: np.ndarray | None = None
    period = 1.0 / args.control_hz
    trial_start = time.perf_counter()
    deadline = trial_start + args.trial_duration_s
    phase_timeline = PhaseTimeline(trial_start)
    phase_video_recorder: ContinuousPhaseVideoRecorder | None = None
    if args.continuous_top_video:
        if args.phase_video_camera not in robot.cameras:
            recorder.close()
            raise KeyError(
                f"Continuous video camera {args.phase_video_camera!r} is unavailable; "
                f"configured cameras={list(robot.cameras)}"
            )
        phase_video_recorder = ContinuousPhaseVideoRecorder(
            attempt_dir,
            robot.cameras[args.phase_video_camera],
            args.phase_video_camera,
            phase_timeline,
            trial_start,
            fps=args.phase_video_fps,
            border_px=args.phase_video_border_px,
        )
    state_recorder = HighRateStateRecorder(
        attempt_dir, robot, ee_controller, args.state_log_hz, trial_start
    )
    try:
        state_recorder.start()
        if phase_video_recorder is not None:
            phase_video_recorder.start()
    except Exception:
        state_recorder.stop_and_write()
        recorder.close()
        raise
    next_t = trial_start
    frame_index = 0
    clip_count = 0
    ik_skip_count = 0
    inference_calls = 0
    inference_times_ms: list[float] = []
    loop_times_ms: list[float] = []
    overrun_count = 0
    chunk_id = -1
    chunk_step = 0
    stop_reason = "timeout"
    server_error: str | None = None
    last_print_t = 0.0

    try:
        while time.perf_counter() < deadline:
            key = base.poll_keyboard()
            if key == "":
                stop_reason = "user"
                break
            if key == "q":
                stop_reason = "quit"
                break

            loop_start = time.perf_counter()
            obs_start = loop_start
            obs = robot.get_observation()
            observation_ms = (time.perf_counter() - obs_start) * 1000.0
            feedback = base.observation_joint_state(obs)
            ee_feedback = ee_controller.ee_state_from_joint_state(feedback)

            inference_requested = not chunk
            inference_ms = 0.0
            request_id = None
            if inference_requested:
                query = base.policy_query(client, obs, instruction, args, state_stats, ee_controller)
                request_id = query["request_id"]
                phase_timeline.transition(
                    "inference", chunk_id=chunk_id + 1, request_id=request_id
                )
                infer_start = time.perf_counter()
                try:
                    response = client.predict_action(query)
                    inference_ms = (time.perf_counter() - infer_start) * 1000.0
                    normalized_chunk = normalized_chunk_from_response(response)
                    action_chunk = base.extract_action_chunk(
                        response,
                        args.action_input,
                        action_stats,
                        args.debug_actions,
                        args.action_norm,
                    )
                except Exception as exc:
                    server_error = repr(exc)
                    stop_reason = "server_error"
                    print(f"[BENCHMARK] Policy inference failed: {exc}")
                    break
                finally:
                    phase_timeline.transition(
                        "execute", chunk_id=chunk_id + 1, request_id=request_id
                    )
                if args.action_chunk_stride > 1:
                    action_chunk = action_chunk[:: args.action_chunk_stride]
                chunk.extend(action_chunk)
                chunk_id += 1
                chunk_step = 0
                inference_calls += 1
                inference_times_ms.append(inference_ms)
                ee_target_t = ee_controller.fk(feedback)
                recorder.write_chunk(
                    {
                        "chunk_id": chunk_id,
                        "frame_index": frame_index,
                        "request_id": request_id,
                        "wall_time": now_iso(),
                        "trial_elapsed_s": time.perf_counter() - trial_start,
                        "inference_ms": inference_ms,
                        "policy_state_normalized": query["examples"][0]["state"],
                        "normalized_action_chunk": normalized_chunk,
                        "action_chunk": action_chunk,
                    }
                )

            policy_action = chunk.popleft().copy()
            policy_action_before_gripper = policy_action.copy()
            raw_gripper_mm = float(policy_action[6])
            policy_action[6] = base.gripper_target_mm(raw_gripper_mm, args, previous_mm=float(current_cmd[6]))
            if args.safe:
                safe_action, was_clipped = base.clip_action_to_safe_range(policy_action, safe_low, safe_high)
                clip_count += int(was_clipped)
            else:
                safe_action = policy_action.copy()
                was_clipped = False

            ik_start = time.perf_counter()
            joint_target, ik_info = ee_controller.delta_to_joint_target(
                safe_action,
                feedback,
                current_cmd,
                reference_t=ee_target_t,
            )
            ee_target_t = ik_info.pop("target_ee_transform", ee_target_t)
            ik_ms = (time.perf_counter() - ik_start) * 1000.0
            clip_count += int(ik_info.get("ee_delta_clipped", False))
            clip_count += int(ik_info.get("ee_workspace_clipped", False))
            if not ik_info.get("ik_success", False):
                ik_skip_count += 1
                clip_count += 1
                joint_target = current_cmd.copy()

            current_cmd = base.limit_joint_step(
                joint_target, current_cmd, args.max_joint_speed_deg_s, args.control_hz
            )
            send_start = time.perf_counter()
            if args.execute:
                robot.send_joint_command(current_cmd, speed_ratio=args.speed_ratio)
            send_ms = (time.perf_counter() - send_start) * 1000.0
            loop_ms_before_record = (time.perf_counter() - loop_start) * 1000.0

            row = {
                "frame_index": frame_index,
                "wall_time": now_iso(),
                "trial_elapsed_s": time.perf_counter() - trial_start,
                "scheduled_elapsed_s": frame_index * period,
                "phase": "inference_and_execute" if inference_requested else "execute",
                "inference_requested": inference_requested,
                "inference_request_id": request_id,
                "inference_ms": inference_ms,
                "chunk_id": chunk_id,
                "chunk_step": chunk_step,
                "queue_remaining": len(chunk),
                "observation_ms": observation_ms,
                "ik_ms": ik_ms,
                "send_ms": send_ms,
                "loop_ms_before_record": loop_ms_before_record,
                "joint_feedback_deg": feedback[:6],
                "gripper_feedback_mm": feedback[6],
                "ee_feedback_xyz_quat_gripper": ee_feedback,
                "policy_action_raw": policy_action_before_gripper,
                "policy_gripper_raw_mm": raw_gripper_mm,
                "safe_ee_action": safe_action,
                "safe_range_clipped": was_clipped,
                "ik_success": bool(ik_info.get("ik_success", False)),
                "ik_reason": ik_info.get("ik_reason"),
                "ik_pos_err_m": ik_info.get("ik_pos_err_m"),
                "ik_rot_err_rad": ik_info.get("ik_rot_err_rad"),
                "ee_delta_clipped": bool(ik_info.get("ee_delta_clipped", False)),
                "ee_workspace_clipped": bool(ik_info.get("ee_workspace_clipped", False)),
                "target_ee_xyz_quat": pose_from_transform(ee_target_t),
                "joint_target_deg": joint_target[:6],
                "gripper_target_mm": joint_target[6],
                "sent_joint_command_deg": current_cmd[:6],
                "sent_gripper_command_mm": current_cmd[6],
                "command_sent": bool(args.execute),
            }
            recorder.write_frame(obs, row)
            loop_ms = (time.perf_counter() - loop_start) * 1000.0
            loop_times_ms.append(loop_ms)

            now = time.perf_counter()
            if now - last_print_t >= 1.0 / max(args.print_hz, 0.1):
                infer_text = f" infer={inference_ms:.0f}ms" if inference_requested else ""
                print(
                    f"[BENCHMARK] trial={valid_trial_number} frame={frame_index:04d} "
                    f"phase={'infer+exec' if inference_requested else 'execute':10s} "
                    f"queue={len(chunk)} ik={'ok' if ik_info.get('ik_success') else 'skip'} "
                    f"grip_raw/cmd={raw_gripper_mm:.1f}/{current_cmd[6]:.1f}mm"
                    f"{infer_text} elapsed={now - trial_start:.1f}s "
                    f"remaining={max(0.0, deadline - now):.1f}s"
                )
                last_print_t = now

            frame_index += 1
            chunk_step += 1
            next_t += period
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                overrun_count += 1
                next_t = time.perf_counter()
    finally:
        if phase_video_recorder is not None:
            try:
                phase_video_info = phase_video_recorder.stop_and_write()
            except Exception as exc:
                phase_video_info = {
                    "continuous_video_frames": len(phase_video_recorder.rows),
                    "continuous_video_error": repr(exc),
                }
        else:
            phase_video_info = {
                "continuous_video_frames": 0,
                "continuous_video_error": None,
            }
        try:
            state_log_info = state_recorder.stop_and_write()
        except Exception as exc:
            state_log_info = {
                "state_log_requested_hz": args.state_log_hz,
                "state_log_samples": len(state_recorder.rows),
                "state_log_effective_hz": None,
                "state_log_unique_feedback_hz": None,
                "state_log_duplicate_ratio": None,
                "state_log_error": repr(exc),
            }
        finally:
            recorder.close()

    if state_log_info["state_log_error"]:
        print(f"[BENCHMARK] High-rate state logger error: {state_log_info['state_log_error']}")
    elif state_log_info["state_log_samples"]:
        print(
            "[BENCHMARK] High-rate state log: "
            f"samples={state_log_info['state_log_samples']} "
            f"poll={state_log_info['state_log_effective_hz']:.1f}Hz "
            f"unique={state_log_info['state_log_unique_feedback_hz'] or 0.0:.1f}Hz "
            f"duplicates={100.0 * (state_log_info['state_log_duplicate_ratio'] or 0.0):.1f}%"
        )

    if stop_reason == "quit":
        label = "quit"
    else:
        label = read_attempt_label(args.demo)

    inference_array = np.asarray(inference_times_ms, dtype=np.float64)
    loop_array = np.asarray(loop_times_ms, dtype=np.float64)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "attempt_started_at": attempt_started_at,
        "attempt_finished_at": now_iso(),
        "valid_trial_number": valid_trial_number,
        "task_index": task_index,
        "repetition_index": repetition_index,
        "attempt_number": attempt_number,
        "instruction": instruction,
        "label": label,
        "stop_reason": stop_reason,
        "attempt_dir": str(attempt_dir),
        "videos": recorder.video_paths,
        "state_feedback": "state_feedback_high_rate.parquet" if state_log_info["state_log_samples"] else None,
        "frame_count": frame_index,
        "control_hz": args.control_hz,
        "trial_wall_duration_s": time.perf_counter() - trial_start,
        "inference_calls": inference_calls,
        "inference_ms_mean": float(inference_array.mean()) if inference_array.size else None,
        "inference_ms_p95": float(np.quantile(inference_array, 0.95)) if inference_array.size else None,
        "loop_ms_mean": float(loop_array.mean()) if loop_array.size else None,
        "loop_ms_p95": float(np.quantile(loop_array, 0.95)) if loop_array.size else None,
        "control_overrun_count": overrun_count,
        "clip_count": clip_count,
        "ik_skip_count": ik_skip_count,
        "server_error": server_error,
        "camera_warmup_reads": warmup_reads,
        "camera_warmup_elapsed_s": warmup_elapsed_s,
        **state_log_info,
        **phase_video_info,
        **home_info,
    }


def prepare_run(
    args: argparse.Namespace, tasks: list[str], episode_counts: list[int]
) -> tuple[Path, list[dict[str, Any]]]:
    run_dir = args.output_root.expanduser().resolve() / args.run_name
    metadata_path = run_dir / "metadata.json"
    manifest_path = run_dir / "manifest.jsonl"
    if run_dir.exists() and not args.resume_benchmark:
        raise FileExistsError(
            f"Benchmark run already exists: {run_dir}. Use a new --run-name or add --resume-benchmark."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    records = read_manifest(manifest_path)
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous.get("tasks") != tasks:
            raise ValueError("Existing benchmark uses a different task list")
        previous_counts = previous.get("episodes_per_task_by_task")
        if previous_counts is None:
            previous_counts = [int(previous.get("episodes_per_task", -1))] * len(tasks)
        if previous_counts != episode_counts:
            raise ValueError(
                "Existing benchmark uses a different per-task episode schedule: "
                f"existing={previous_counts}, requested={episode_counts}"
            )
    else:
        metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "created_at": now_iso(),
            "run_name": args.run_name,
            "checkpoint_id": args.checkpoint_id,
            "git_revision": git_revision(),
            "episodes_per_task": args.episodes_per_task,
            "episodes_per_task_by_task": episode_counts,
            "tasks": tasks,
            "arguments": vars(args),
            "frame_semantics": {
                "phase": (
                    "inference_and_execute marks the first action of a newly inferred chunk; "
                    "execute consumes a cached chunk action"
                ),
                "videos": (
                    "One RGB observation frame per control step at nominal control_hz; "
                    "frames.jsonl stores actual wall timing"
                ),
                "continuous_phase_video": (
                    "When enabled, the top camera is recorded independently at its native nominal 30 Hz. "
                    "The untouched original video and red=inference/green=execute overlay share frame order; "
                    "the exact camera timestamp and phase for every frame are in top_video_frames.jsonl."
                ),
                "state": "Joint feedback is degrees, EE xyz is meters, quaternion order is xyzw, gripper is millimeters",
                "high_rate_state": (
                    "state_feedback_high_rate.* samples cached Piper SDK feedback independently from policy, "
                    "camera, and command timing; SDK timestamps identify repeated cache reads"
                ),
            },
        }
        atomic_write_json(metadata_path, metadata)
        atomic_write_json(run_dir / "tasks.json", tasks)
    append_jsonl(
        run_dir / "sessions.jsonl",
        {
            "started_at": now_iso(),
            "start_trial": args.start_trial,
            "resume": args.resume_benchmark,
            "arguments": vars(args),
        },
    )
    return run_dir, records


def print_plan(
    tasks: list[str], episode_counts: list[int], start_trial: int, trial_duration_s: float
) -> None:
    total = sum(episode_counts)
    print(f"[BENCHMARK] tasks={len(tasks)} episode_counts={episode_counts} total={total}")
    print(f"[BENCHMARK] start_trial={start_trial} (1-based)")
    print(f"[BENCHMARK] trial_time_limit={trial_duration_s:.1f}s per episode")
    first = 1
    for task_index, instruction in enumerate(tasks):
        last = first + episode_counts[task_index] - 1
        print(f"  task {task_index + 1:02d}: trials {first:03d}-{last:03d} | {instruction}")
        first = last + 1


def main() -> None:
    args = build_argparser().parse_args()
    if args.episodes_per_task <= 0:
        raise ValueError("--episodes-per-task must be > 0")
    if args.low_data_episodes_per_task is not None and args.low_data_episodes_per_task <= 0:
        raise ValueError("--low-data-episodes-per-task must be > 0")
    if args.control_hz <= 0 or args.trial_duration_s <= 0:
        raise ValueError("--control-hz and --trial-duration-s must be > 0")
    if args.action_chunk_stride <= 0:
        raise ValueError("--action-chunk-stride must be > 0")
    if args.state_log_hz < 0:
        raise ValueError("--state-log-hz must be >= 0")
    if not args.gripper_close_mm < args.gripper_binary_threshold < args.gripper_open_mm:
        raise ValueError(
            "--gripper-binary-threshold must be strictly between the closed and open commands"
        )
    tasks = load_tasks(args.tasks_json)
    episode_counts = build_episode_counts(
        tasks, args.episodes_per_task, args.low_data_episodes_per_task
    )
    schedule = build_schedule(tasks, episode_counts)
    total_trials = len(schedule)
    if not 1 <= args.start_trial <= total_trials:
        raise ValueError(f"--start-trial must be in [1, {total_trials}]")
    print_plan(tasks, episode_counts, args.start_trial, args.trial_duration_s)
    if args.plan_only:
        return
    if args.action_space != "delta-ee" or args.policy_state_space not in {"ee", "ee7"}:
        raise ValueError("The formal benchmark requires --action-space delta-ee --policy-state-space ee")

    run_dir, records = prepare_run(args, tasks, episode_counts)
    accepted_slots = {
        int(record["valid_trial_number"])
        for record in records
        if record.get("label") in {"success", "failure"}
    }
    conflicts = sorted(slot for slot in accepted_slots if slot >= args.start_trial)
    if conflicts:
        raise ValueError(
            f"Trials at or after --start-trial are already accepted: {conflicts[:10]}. "
            "Resume from the first unfinished trial to avoid duplicate results."
        )

    action_stats, state_stats = base.load_policy_stats(args, "[BENCHMARK]")
    base.configure_ee_workspace(args, state_stats)
    safe_low, safe_high = base.safe_action_bounds(action_stats, args)
    ee_controller = base.DeltaEEToJointController(args)
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    robot = base.build_robot(args)

    print(f"[BENCHMARK] output={run_dir}")
    print(f"[BENCHMARK] normalization action={args.action_norm} state={args.state_norm}")
    print(
        "[BENCHMARK] gripper: "
        f"prediction < {args.gripper_binary_threshold:.1f}mm -> close; otherwise -> open"
    )
    print("[BENCHMARK] Connecting robot and cameras...")
    robot.connect()
    print("[BENCHMARK] Connected.")

    slot = args.start_trial - 1
    try:
        while slot < total_trials:
            task_index, repetition_index, instruction = schedule[slot]
            valid_trial_number = slot + 1
            attempt_number = next_attempt_number(run_dir, valid_trial_number)
            attempt_dir = (
                run_dir
                / "attempts"
                / f"trial_{valid_trial_number:04d}_attempt_{attempt_number:03d}_{datetime.now():%Y%m%d_%H%M%S}"
            )
            print("\n" + "=" * 88)
            print(
                f"[BENCHMARK] VALID TRIAL {valid_trial_number}/{total_trials} | "
                f"task {task_index + 1}/{len(tasks)} | "
                f"repeat {repetition_index + 1}/{episode_counts[task_index]} | "
                f"attempt {attempt_number}"
            )
            print(f"[BENCHMARK] {instruction}")
            print("=" * 88)
            base.reset_policy_cache(client, "benchmark_attempt_start", valid_trial_number, instruction)
            result = run_attempt(
                client,
                robot,
                instruction,
                task_index,
                repetition_index,
                valid_trial_number,
                attempt_number,
                attempt_dir,
                args,
                action_stats,
                state_stats,
                safe_low,
                safe_high,
                ee_controller,
            )
            if result["label"] in {"success", "failure", "demo"}:
                result.pop("attempt_number", None)
                append_jsonl(run_dir / "manifest.jsonl", result)
                records.append(result)
                atomic_write_json(run_dir / "summary.json", build_summary(tasks, episode_counts, records))
                atomic_write_json(attempt_dir / "result.json", result)
                print(f"[BENCHMARK] Saved accepted trial: {attempt_dir}")
                slot += 1
            elif result["label"] == "retry":
                if args.keep_all_recordings:
                    atomic_write_json(attempt_dir / "result.json", result)
                    append_jsonl(run_dir / "nonaccepted_manifest.jsonl", result)
                    print(f"[BENCHMARK] Kept retry recording: {attempt_dir}")
                else:
                    shutil.rmtree(attempt_dir, ignore_errors=True)
                    print(f"[BENCHMARK] Discarded retry recording: {attempt_dir}")
                print(f"[BENCHMARK] Retrying valid trial {valid_trial_number}; schedule does not advance.")
            else:
                if args.keep_all_recordings:
                    atomic_write_json(attempt_dir / "result.json", result)
                    append_jsonl(run_dir / "nonaccepted_manifest.jsonl", result)
                    print(f"[BENCHMARK] Kept incomplete recording: {attempt_dir}")
                else:
                    shutil.rmtree(attempt_dir, ignore_errors=True)
                    print(f"[BENCHMARK] Discarded incomplete trial: {attempt_dir}")
                break
    except KeyboardInterrupt:
        print("\n[BENCHMARK] Interrupted. Resume later with the same --run-name and --resume-benchmark.")
    finally:
        append_jsonl(run_dir / "sessions.jsonl", {"finished_at": now_iso(), "next_trial": slot + 1})
        try:
            client.close()
        finally:
            if robot.is_connected:
                robot.disconnect()
        print(f"[BENCHMARK] Next valid trial: {slot + 1}")
        print(f"[BENCHMARK] Summary: {run_dir / 'summary.json'}")


def build_argparser() -> argparse.ArgumentParser:
    parser = base.build_argparser()
    parser.description = "Run the resumable PiperX real-robot benchmark (built-in: the 25 Fruit25 tasks)."
    parser.epilog = (
        "Defaults force the paper's Fruit25 protocol (8 Hz). For PushBlock pass "
        "--tasks-json <one-task list> --control-hz 20 --gripper-action-mode absolute (paper settings)."
    )
    parser.set_defaults(
        action_space="delta-ee",
        policy_state_space="ee",
        action_input="normalized",
        action_norm="q99",
        state_norm="q99",
        control_hz=8.0,
        max_ee_delta_m=0.05,
        max_ee_rot_delta_rad=0.20,
        max_joint_speed_deg_s=25.0,
        speed_ratio=20,
        gripper_action_mode="binary",
        gripper_binary_threshold=80.6,
        trial_duration_s=50.0,
        print_hz=1.0,
    )
    group = parser.add_argument_group("formal benchmark")
    group.add_argument("--run-name", required=True, help="Stable output name identifying this model/configuration.")
    group.add_argument("--checkpoint-id", default=None, help="Checkpoint/repository label saved in metadata.")
    group.add_argument("--output-root", type=Path, default=Path("outputs/piperx_benchmark"))
    group.add_argument(
        "--tasks-json", type=Path, default=None, help="Optional JSON list replacing the built-in 25 tasks."
    )
    group.add_argument("--episodes-per-task", type=int, default=10)
    group.add_argument(
        "--low-data-episodes-per-task",
        type=int,
        default=None,
        help=(
            "Override accepted trials for the 16 low-data Fruit25 compositional tasks "
            "(zero-based task indices 7-22). Other tasks keep --episodes-per-task."
        ),
    )
    group.add_argument("--start-trial", type=int, default=1, help="1-based valid trial number to start/resume from.")
    group.add_argument(
        "--demo",
        action="store_true",
        help="Demonstration runs: no success/failure question; runs are recorded but not scored.",
    )
    group.add_argument("--resume-benchmark", action="store_true", help="Append to an existing --run-name.")
    group.add_argument("--record-video", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--record-cameras", nargs="+", default=["top", "wrist"])
    group.add_argument(
        "--keep-all-recordings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep retry and interrupted episode directories instead of deleting their recordings.",
    )
    group.add_argument(
        "--continuous-top-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record the top camera independently of the control loop and render a phase-border video.",
    )
    group.add_argument("--phase-video-camera", default="top")
    group.add_argument("--phase-video-fps", type=float, default=30.0)
    group.add_argument("--phase-video-border-px", type=int, default=14)
    group.add_argument(
        "--state-log-hz",
        type=float,
        default=50.0,
        help="Independent Piper feedback sampling rate; 0 disables high-rate state logging.",
    )
    group.add_argument(
        "--plan-only", action="store_true", help="Print the complete schedule without connecting anything."
    )
    return parser


if __name__ == "__main__":
    main()
