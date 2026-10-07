#!/usr/bin/env python
"""Collect PiperX demonstrations with a Meta Quest (WebXR) and save them as a LeRobot v3.0 dataset.

    python examples/PiperX/teleoperation/record.py \
        --root data/piperx/FruitRaw --task "Pick up the apple and place it in the red basket." \
        --num-episodes 50

Open the printed https:// URL in the headset browser, accept the certificate warning once,
then press "Enter VR". Each episode: A (move home) -> A (start recording) -> A (save) or B (discard).
Recorded per frame (default 30 Hz): measured joints + gripper as ``observation.state``, the IK
joint target sent to the arm as ``action``, and the ``top`` / ``wrist`` camera images.
Use ``../dataset_tools`` to turn the recording into the EE-delta training dataset.
"""

from __future__ import annotations

import argparse
import os
import queue
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.PiperX.common.kinematics import PiperXKinematics  # noqa: E402
from examples.PiperX.common.robot import (  # noqa: E402
    DEFAULT_HOME_JOINTS_DEG,
    PIPER_ACTION_KEYS,
    PiperXRobot,
    PiperXRobotConfig,
    add_robot_args,
    list_cameras,
)
from examples.PiperX.teleoperation.console import Dashboard  # noqa: E402
from examples.PiperX.teleoperation.dataset_writer import VIDEO_CODECS, LeRobotV3Writer  # noqa: E402
from examples.PiperX.teleoperation.vr_server import VRServer  # noqa: E402
from examples.PiperX.teleoperation.vr_teleop import PiperXVRTeleop, TeleopStep, VRTeleopConfig  # noqa: E402

ROBOT_TYPE = "piperx_follower"  # matches the released raw recordings

HINTS = {
    "idle": "Reset the scene, then press A to move home",
    "homing": "Moving home... (B cancels)",
    "ready": "Press A to start recording  (left X: align red X axis to robot +X)",
    "recording": "A save  |  B discard",
    "saving": "Saving episode...",
    "stopped": "All episodes recorded. Press q on the PC to quit.",
    "teleop": "Hold grip to move. Y: home. X: align (red X axis -> robot +X)",
}
TITLES = {"idle": "Idle", "homing": "Homing", "ready": "Ready", "recording": "Rec", "saving": "Saving", "stopped": "Done", "teleop": "Teleop"}


# ---------------------------------------------------------------------------- mock robot
class MockPiperXRobot:
    """Simulated arm and cameras: the joints follow the command, cameras show a top-down sketch."""

    def __init__(self, config: PiperXRobotConfig):
        self.config = config
        self.cameras = {"wrist": None, "top": None}
        self.is_connected = False
        self._state = np.asarray([*DEFAULT_HOME_JOINTS_DEG, 100.0], dtype=np.float32)
        self._state[1] -= 20.0  # start away from home so homing is visible
        self._target = self._state.copy()
        self._last_t = time.perf_counter()
        self._lock = threading.Lock()
        self._kin = PiperXKinematics()

    def connect(self) -> None:
        self.is_connected = True

    def _advance(self) -> None:
        now = time.perf_counter()
        dt, self._last_t = now - self._last_t, now
        step = 120.0 * dt
        self._state[:6] += np.clip(self._target[:6] - self._state[:6], -step, step)
        self._state[6] += np.clip(self._target[6] - self._state[6], -300.0 * dt, 300.0 * dt)

    def read_joint_state(self) -> np.ndarray:
        with self._lock:
            self._advance()
            return self._state.copy()

    def read_images(self) -> dict[str, np.ndarray]:
        import cv2

        h, w = self.config.camera_height, self.config.camera_width
        pos, _ = self._kin.fk_pose_deg(self.read_joint_state())
        top = np.full((h, w, 3), 60, np.uint8)
        cx, cy = int(w / 2 - pos[1] * 800), int(h - pos[0] * 800)
        cv2.circle(top, (cx, cy), 12 + int(pos[2] * 40), (230, 120, 40), -1)
        wrist = np.full((h, w, 3), 30, np.uint8)
        cv2.putText(wrist, f"z={pos[2]*1000:.0f}mm", (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 2)
        return {"wrist": wrist, "top": top}

    def send_joint_command(self, target: np.ndarray, speed_ratio: int | None = None) -> np.ndarray:
        with self._lock:
            self._target = np.asarray(target, dtype=np.float32).copy()
        return self._target.copy()

    def disconnect(self) -> None:
        self.is_connected = False


# ---------------------------------------------------------------------------- keyboard
class Keyboard:
    """Single-key controls from the terminal (works over SSH; no X server needed)."""

    KEYMAP = {" ": "A", "\n": "A", "\r": "A", "RIGHT": "A", "\x7f": "B", "\b": "B", "LEFT": "B", "h": "Y", "q": "QUIT", "ESC": "QUIT"}

    def __init__(self, events: queue.Queue):
        self.events = events
        self.enabled = sys.stdin.isatty()
        self._stop = threading.Event()
        self._old = None

    def start(self) -> None:
        if not self.enabled:
            return
        fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        threading.Thread(target=self._run, name="keyboard", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        if self._old is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old)
            self._old = None

    def _run(self) -> None:
        fd = sys.stdin.fileno()
        while not self._stop.is_set():
            if not select.select([fd], [], [], 0.1)[0]:
                continue
            ch = os.read(fd, 1).decode(errors="ignore")
            if ch == "\x1b":
                seq = ch
                while select.select([fd], [], [], 0.01)[0] and len(seq) < 4:
                    seq += os.read(fd, 1).decode(errors="ignore")
                ch = {"\x1b[C": "RIGHT", "\x1bOC": "RIGHT", "\x1b[D": "LEFT", "\x1bOD": "LEFT", "\x1b": "ESC"}.get(seq, "")
            key = self.KEYMAP.get(ch if len(ch) > 1 else ch.lower())
            if key:
                self.events.put(("key", key))


# ---------------------------------------------------------------------------- session
class Session:
    def __init__(
        self,
        args: argparse.Namespace,
        robot,
        teleop: PiperXVRTeleop,
        vr: VRServer,
        writer: LeRobotV3Writer | None,
        dashboard: Dashboard,
    ):
        self.args = args
        self.robot = robot
        self.teleop = teleop
        self.vr = vr
        self.writer = writer
        self.events: queue.Queue = queue.Queue()
        self.keyboard = Keyboard(self.events)
        self.phase = "idle" if writer is not None else "teleop"
        self.after_home = "ready" if writer is not None else "teleop"
        self.saved_this_session = 0
        self.message = ""
        self.record_start_t = 0.0
        self.running = True
        self._teleop_lock = threading.Lock()
        self._step: TeleopStep | None = None
        self._stop = threading.Event()
        self._control_error_t = 0.0
        self._last_deadman = False
        self._last_ik_ok = True
        self.dashboard = dashboard
        self._feedback: np.ndarray | None = None
        self._control_count = 0
        self._rate_t = time.perf_counter()
        self._rate_counts = (0, 0)

    # ------------------------------------------------------------------ control thread
    def control_loop(self) -> None:
        period = 1.0 / self.teleop.cfg.freq
        next_t = time.perf_counter()
        while not self._stop.is_set():
            try:
                feedback = self.robot.read_joint_state()
                left, right, age = self.vr.latest()
                if age > self.args.vr_timeout_s:
                    left = right = None  # stale headset data: hold position
                with self._teleop_lock:
                    step = self.teleop.step(feedback, left, right)
                self.robot.send_joint_command(step.command, step.speed_ratio)
                self._step = step
                self._feedback = feedback
                self._control_count += 1
                for name in step.events:
                    self.events.put(("vr", name))
                if step.home_reached:
                    self.events.put(("robot", "home_reached"))
                self._haptics(step)
            except Exception as exc:  # noqa: BLE001 - keep holding the arm; report at most once per second
                now = time.perf_counter()
                if now - self._control_error_t > 1.0:
                    self.log(f"Control loop error: {exc}")
                    self._control_error_t = now
            next_t += period
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.perf_counter()

    def _haptics(self, step: TeleopStep) -> None:
        if step.deadman and not self._last_deadman:
            self.vr.haptic("right", 0.25, 30)
        if not step.ik_ok and self._last_ik_ok:
            self.vr.haptic("right", 0.8, 120)
        self._last_deadman, self._last_ik_ok = step.deadman, step.ik_ok

    # ------------------------------------------------------------------ state machine
    def handle(self, source: str, name: str) -> None:
        button = {"right_a": "A", "right_b": "B", "left_y": "Y"}.get(name, name) if source == "vr" else name
        if button == "QUIT":
            if self.phase == "recording":
                self.save_episode()  # same as lerobot-record: stopping keeps the current episode
            self.running = False
        elif button == "home_reached":
            if self.phase == "homing":
                self.set_phase(self.after_home, "Home reached.")
                self.vr.haptic("right", 0.4, 60)
        elif self.phase == "teleop":
            if button == "Y":
                self.start_home()
        elif self.phase in ("idle", "ready"):
            if button == "Y" or (button == "A" and self.phase == "idle"):
                self.start_home()
            elif button == "A":
                self.start_recording()
        elif self.phase == "homing":
            if button == "B":
                with self._teleop_lock:
                    self.teleop.cancel_home()
                self.set_phase("idle", "Homing cancelled.")
            elif button == "A":
                self.message = "Wait until the arm reaches home."
        elif self.phase == "recording":
            if button == "A":
                self.save_episode()
            elif button == "B":
                self.discard_episode()
            elif button == "Y":
                self.message = "Home is disabled while recording (B discards, A saves)."

    def start_home(self) -> None:
        with self._teleop_lock:
            self.teleop.start_home()
        self.set_phase("homing", "")

    def start_recording(self) -> None:
        self.writer.start_episode(self.args.task)
        self.record_start_t = time.perf_counter()
        self.set_phase("recording", f"Recording episode {self.writer.num_episodes}.")
        self.vr.haptic("right", 0.7, 150)

    def save_episode(self) -> None:
        self.set_phase("saving", "")
        self.publish_status()
        frames = self.writer.episode_length
        try:
            index = self.writer.save_episode()
        except ValueError as exc:
            self.set_phase("idle", f"Not saved: {exc}")
            return
        self.saved_this_session += 1
        self.vr.haptic("right", 0.7, 80)
        done = self.saved_this_session >= self.args.num_episodes
        self.set_phase("stopped" if done else "idle", f"Saved episode {index} ({frames} frames, {frames / self.args.fps:.1f}s).")

    def discard_episode(self) -> None:
        self.writer.discard_episode()
        self.set_phase("idle", "Episode discarded; record it again.")
        self.vr.haptic("right", 1.0, 300)

    def set_phase(self, phase: str, message: str) -> None:
        self.phase = phase
        if message:
            self.message = message
            self.log(message)

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        step = self._step
        elapsed = time.perf_counter() - self.record_start_t if self.phase == "recording" else None
        hint = HINTS[self.phase]
        feedback = self.robot.read_joint_state()
        if self.phase == "homing" and step is not None:
            hint = f"Moving home... {self.teleop.home_remaining_deg(feedback):.0f} deg left (B cancels)"
        base_q, gripper_q = self.teleop.frames_in_vr(feedback)
        return {
            "type": "status",
            "phase": self.phase,
            "title": TITLES[self.phase],
            "hint": hint,
            "record": self.writer is not None,
            "episode": self.writer.num_episodes if self.writer else None,
            "target_episodes": (self.writer.num_episodes - self.saved_this_session + self.args.num_episodes) if self.writer else None,
            "saved": self.saved_this_session,
            "task": self.args.task or "",
            "elapsed_s": elapsed,
            "tracking": bool(step and step.tracking),
            "deadman": bool(step and step.deadman),
            "gripper_closed": bool(step and step.gripper_closed),
            "ik_ok": bool(step is None or step.ik_ok),
            "encoder_backlog": self.writer.encoder_backlog if self.writer else 0,
            "message": self.message,
            "robot_frame_q": base_q,
            "gripper_frame_q": gripper_q,
            "vr_yaw_deg": self.teleop.mapper.vr_yaw_deg,
        }

    def publish_status(self) -> None:
        status = self.status()
        self.vr.publish(status)
        now = time.perf_counter()
        dt = max(now - self._rate_t, 1e-3)
        control_hz = (self._control_count - self._rate_counts[0]) / dt
        headset_hz = (self.vr.packet_count - self._rate_counts[1]) / dt
        self._rate_t, self._rate_counts = now, (self._control_count, self.vr.packet_count)
        _, _, age = self.vr.latest()
        feedback = self._feedback
        ee_mm = None
        if feedback is not None:
            pos, _ = self.teleop.ik.kin.fk_pose_deg(feedback)
            ee_mm = tuple(float(v) * 1000.0 for v in pos)
        self.dashboard.update(
            phase=self.phase,
            elapsed_s=status["elapsed_s"],
            frames=self.writer.episode_length if self.writer else 0,
            episode=status["episode"],
            saved=status["saved"],
            target=self.args.num_episodes,
            headset_connected=self.vr.num_clients > 0,
            headset_hz=headset_hz,
            headset_age_ms=age * 1000.0 if np.isfinite(age) else None,
            control_hz=control_hz,
            tracking=status["tracking"],
            deadman=status["deadman"],
            gripper_closed=status["gripper_closed"],
            ik_ok=status["ik_ok"],
            encoder_backlog=status["encoder_backlog"],
            ee_mm=ee_mm,
            joints_deg=[] if feedback is None else [float(v) for v in feedback[:6]],
            gripper_mm=None if feedback is None else float(feedback[6]),
            message=self.message,
        )

    def log(self, text: str) -> None:
        self.dashboard.log(text)

    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        control = threading.Thread(target=self.control_loop, name="control", daemon=True)
        control.start()
        self.keyboard.start()
        self.dashboard.start()
        period = 1.0 / self.args.fps
        next_t = time.perf_counter()
        last_status_t = 0.0
        try:
            while self.running:
                while not self.events.empty():
                    self.handle(*self.events.get_nowait())
                if self.phase == "recording" and self._step is not None:
                    state = self.robot.read_joint_state()
                    images = self.robot.read_images()
                    self.writer.add_frame(state, self._step.command, images)
                    if self.args.max_episode_s > 0 and time.perf_counter() - self.record_start_t >= self.args.max_episode_s:
                        self.log(f"Reached --max-episode-s={self.args.max_episode_s}; saving.")
                        self.save_episode()
                now = time.perf_counter()
                if now - last_status_t >= 0.1:
                    self.publish_status()
                    last_status_t = now
                next_t += period
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.perf_counter()
        except KeyboardInterrupt:
            if self.phase == "recording":
                self.writer.discard_episode()
                self.log("Interrupted: the episode in progress was discarded.")
        finally:
            self.keyboard.stop()
            self._stop.set()
            control.join(timeout=2.0)
            self.dashboard.stop()


# ---------------------------------------------------------------------------- main
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = parser.add_argument_group("dataset")
    data.add_argument("--root", type=Path, help="Dataset directory (LeRobot v3.0). Required unless --no-record.")
    data.add_argument("--task", default=None, help="Language instruction stored with every episode of this session.")
    data.add_argument("--num-episodes", type=int, default=50, help="Stop after saving this many episodes.")
    data.add_argument("--fps", type=int, default=30, help="Recording rate. The released datasets were recorded at 30 Hz.")
    data.add_argument("--resume", action="store_true", help="Append to an existing dataset at --root.")
    data.add_argument("--vcodec", choices=list(VIDEO_CODECS), default="av1")
    data.add_argument("--max-episode-s", type=float, default=0.0, help="Auto-save an episode after this many seconds (0 = off).")
    data.add_argument("--no-record", action="store_true", help="Teleoperate only; nothing is saved.")

    vr = parser.add_argument_group("vr")
    vr.add_argument("--vr-host", default="0.0.0.0")
    vr.add_argument("--vr-port", type=int, default=8443)
    vr.add_argument("--vr-cert", type=Path, default=None, help="TLS certificate (default: self-signed, generated once).")
    vr.add_argument("--vr-key", type=Path, default=None)
    vr.add_argument("--vr-timeout-s", type=float, default=0.5, help="Hold the arm if no headset data arrives for this long.")
    vr.add_argument("--scale", type=float, default=600.0, help="Robot millimetres per metre of controller motion.")
    vr.add_argument("--max-linear-speed", type=float, default=350.0, help="mm/s")
    vr.add_argument("--max-angular-speed", type=float, default=160.0, help="deg/s")
    vr.add_argument("--control-hz", type=float, default=30.0, help="Teleoperation control rate.")
    vr.add_argument("--home-joints-deg", nargs=6, type=float, default=DEFAULT_HOME_JOINTS_DEG)

    parser.add_argument("--mock-robot", action="store_true", help="No hardware: simulated arm and cameras (to try the VR setup).")
    parser.add_argument("--list-cameras", action="store_true", help="Print RealSense serials and UVC camera indices, then exit.")
    add_robot_args(parser, speed_ratio=100, mode_refresh_interval_s=1.0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.list_cameras:
        list_cameras()
        return
    record = not args.no_record
    if record and (args.root is None or not args.task):
        raise SystemExit("--root and --task are required when recording (or pass --no-record).")

    robot_cfg = PiperXRobotConfig.from_args(args)
    if robot_cfg.wrist_realsense_serial == "":
        robot_cfg.wrist_realsense_serial = None
    robot = MockPiperXRobot(robot_cfg) if args.mock_robot else PiperXRobot(robot_cfg, connect_cameras=record)
    writer = None
    if record:
        writer = LeRobotV3Writer(
            args.root,
            fps=args.fps,
            robot_type=ROBOT_TYPE,
            state_names=list(PIPER_ACTION_KEYS),
            action_names=list(PIPER_ACTION_KEYS),
            camera_names=list(robot.cameras),
            image_height=args.camera_height,
            image_width=args.camera_width,
            vcodec=args.vcodec,
            resume=args.resume,
        )
        if not robot.cameras:
            raise SystemExit("No cameras configured; recording needs --wrist-realsense-serial and/or --top-opencv-index.")

    teleop = PiperXVRTeleop(
        VRTeleopConfig(
            freq=args.control_hz,
            scale=args.scale,
            max_linear_speed=args.max_linear_speed,
            max_angular_speed=args.max_angular_speed,
            home_joints_deg=tuple(args.home_joints_deg),
        ),
        normal_speed_ratio=args.speed_ratio,
    )
    vr = VRServer(args.vr_host, args.vr_port, args.vr_cert, args.vr_key)

    print("[PiperX] Connecting robot and cameras..." if not args.mock_robot else "[PiperX] Using the simulated robot.")
    robot.connect()
    try:
        teleop.sync_to_robot(robot.read_joint_state())
        vr.start()
        dashboard = Dashboard(
            "PiperX VR teleop" + (" (simulated robot)" if args.mock_robot else ""),
            vr.url,
            f"{args.root} ({writer.num_episodes} episodes, {args.fps} Hz)" if writer is not None else None,
            args.task,
        )
        dashboard.log(f"Headset: open {vr.url} in the Quest browser, then 'Start Controller Tracking'.")
        Session(args, robot, teleop, vr, writer, dashboard).run()
    finally:
        vr.stop()
        if writer is not None:
            writer.close()
            print(f"[Dataset] {args.root}: {writer.num_episodes} episodes, {writer.num_frames} frames.")
        robot.disconnect()


if __name__ == "__main__":
    main()
