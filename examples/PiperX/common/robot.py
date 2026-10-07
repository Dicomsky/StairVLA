"""Direct AgileX PiperX driver (``piper_sdk`` over CAN) plus the two cameras used in the paper.

Used by the robot client, the benchmark, episode replay and VR data collection, so that
deployment and recording talk to the hardware through exactly the same code path.

Units follow the recorded datasets: joints in degrees, gripper opening in millimetres.
"""

from __future__ import annotations

import argparse
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

PIPER_JOINT_NAMES = tuple(f"joint_{idx}" for idx in range(1, 7))
PIPER_JOINT_KEYS = tuple(f"{name}.pos" for name in PIPER_JOINT_NAMES)
PIPER_ACTION_KEYS = (*PIPER_JOINT_KEYS, "gripper.pos")

# Joint pose used as the initial state of every recorded episode and evaluation trial
# (end effector pointing down, -Z).
DEFAULT_HOME_JOINTS_DEG = (0.0, 62.512, -66.452, 83.0, 0.0, 0.0)

# [lower, upper] per joint (deg) and gripper (mm).
PIPERX_HARD_LIMITS_DEG = np.asarray(
    [
        [-150.0, 150.0],
        [0.0, 180.0],
        [-170.0, 0.0],
        [-89.0, 89.0],
        [-89.0, 89.0],
        [-180.0, 180.0],
        [0.0, 100.0],
    ],
    dtype=np.float32,
)

PIPER_CTRL_MODE_TEACH = 0x02
PIPER_CTRL_MODE_LINKAGE_TEACH_INPUT = 0x06


def unit_to_milli(value: float | int) -> int:
    return int(round(float(value) * 1e3))


def milli_to_unit(value: float | int) -> float:
    return float(value) * 1e-3


class OpenCVCamera:
    """UVC camera read by a background thread; ``read()`` returns the latest RGB frame."""

    def __init__(self, index: int | str, width: int, height: int, fps: int):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_connected = False
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_t: float | None = None  # time.perf_counter() when _frame was captured
        self._error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def connect(self) -> None:
        import cv2

        self.cap = cv2.VideoCapture(self.index)
        # Do not force CAP_PROP_BUFFERSIZE=1: on the Innomaker UVC driver it halves a
        # 30 Hz stream to ~15 Hz. The capture thread drains the queue instead. The
        # camera also enables exposure_dynamic_framerate in low light, which silently
        # drops it to ~15 Hz, so turn that off.
        if isinstance(self.index, int):
            subprocess.run(
                ["v4l2-ctl", "-d", f"/dev/video{self.index}", "-c", "exposure_dynamic_framerate=0"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"OpenCV camera {self.index} failed to read a frame.")
        with self._lock:
            self._frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._frame_t = time.perf_counter()
        self.is_connected = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=f"opencv-camera-{self.index}", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        import cv2

        while not self._stop.is_set():
            ok, frame = self.cap.read()
            capture_t = time.perf_counter()
            if not ok:
                self._error = f"OpenCV camera {self.index} failed to read a frame."
                self._stop.wait(0.01)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._frame = rgb
                self._frame_t = capture_t
            self._error = None

    def read(self) -> np.ndarray:
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
        if frame is None:
            raise RuntimeError(self._error or f"OpenCV camera {self.index} has no frame yet.")
        return frame

    def read_latest(self, previous_t: float | None = None) -> tuple[float | None, np.ndarray | None]:
        """Return ``(capture_time, frame)`` of the cached frame; ``frame`` is None if it is not newer than ``previous_t``.

        ``capture_time`` is a ``time.perf_counter()`` value. Used by recorders that sample the
        camera independently of the control loop.
        """
        with self._lock:
            if self._frame is None or self._frame_t is None or self._frame_t == previous_t:
                return self._frame_t, None
            return self._frame_t, self._frame.copy()

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
        with self._lock:
            self._frame = None
            self._frame_t = None
        self._thread = None
        self.is_connected = False


class RealSenseCamera:
    """Intel RealSense color stream read by a background thread; ``read()`` returns the latest RGB frame."""

    def __init__(self, serial: str, width: int, height: int, fps: int):
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.is_connected = False
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_t: float | None = None  # time.perf_counter() when _frame was captured
        self._error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def connect(self) -> None:
        import pyrealsense2 as rs

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        self.pipeline.start(config)
        for _ in range(5):
            self.pipeline.wait_for_frames()
        frame = self._grab()
        with self._lock:
            self._frame = frame
            self._frame_t = time.perf_counter()
        self.is_connected = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=f"realsense-{self.serial}", daemon=True)
        self._thread.start()

    def _grab(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError(f"RealSense camera {self.serial} did not return a color frame.")
        return np.asanyarray(color.get_data()).copy()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._grab()
            except Exception as exc:  # noqa: BLE001 - keep the capture thread alive
                self._error = str(exc)
                self._stop.wait(0.01)
                continue
            capture_t = time.perf_counter()
            with self._lock:
                self._frame = frame
                self._frame_t = capture_t
            self._error = None

    def read(self) -> np.ndarray:
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
        if frame is None:
            raise RuntimeError(self._error or f"RealSense camera {self.serial} has no frame yet.")
        return frame

    def read_latest(self, previous_t: float | None = None) -> tuple[float | None, np.ndarray | None]:
        """Return ``(capture_time, frame)`` of the cached frame; ``frame`` is None if it is not newer than ``previous_t``.

        ``capture_time`` is a ``time.perf_counter()`` value. Used by recorders that sample the
        camera independently of the control loop.
        """
        with self._lock:
            if self._frame is None or self._frame_t is None or self._frame_t == previous_t:
                return self._frame_t, None
            return self._frame_t, self._frame.copy()

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.pipeline is not None:
            self.pipeline.stop()
        self._thread = None
        self.is_connected = False


@dataclass
class PiperXRobotConfig:
    can: str = "can0"
    speed_ratio: int = 30
    high_follow: bool = True
    # Re-send MotionCtrl_2 periodically so the arm stays in CAN joint-control mode.
    # 0 disables the refresh (the mode is still sent on connect and on speed changes).
    mode_refresh_interval_s: float = 0.0
    piper_log_level: str = "WARNING"
    judge_flag: bool = False
    can_auto_init: bool = True
    startup_sleep_s: float = 0.1
    enable_timeout_s: float = 3.0
    disable_on_disconnect: bool = False
    gripper_effort: int = 1000
    gripper_status_code: int = 0x01
    wrist_realsense_serial: str | None = "250122075719"
    top_opencv_index: int | None = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> PiperXRobotConfig:
        return cls(**{name: getattr(args, name) for name in cls.__dataclass_fields__ if hasattr(args, name)})


def add_robot_args(parser: argparse.ArgumentParser, *, speed_ratio: int = 30, mode_refresh_interval_s: float = 0.0) -> None:
    """Register the hardware flags shared by every PiperX script."""
    group = parser.add_argument_group("robot")
    group.add_argument("--can", default="can0", help="CAN interface of the PiperX arm.")
    group.add_argument("--speed-ratio", type=int, default=speed_ratio, help="MotionCtrl_2 speed ratio (0-100).")
    group.add_argument("--high-follow", action=argparse.BooleanOptionalAction, default=True, help="Use the 0xAD high-follow mode.")
    group.add_argument("--mode-refresh-interval-s", type=float, default=mode_refresh_interval_s)
    group.add_argument("--piper-log-level", default="WARNING")
    group.add_argument("--judge-flag", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--can-auto-init", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--startup-sleep-s", type=float, default=0.1)
    group.add_argument("--enable-timeout-s", type=float, default=3.0)
    group.add_argument("--disable-on-disconnect", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--gripper-effort", type=int, default=1000)
    group.add_argument("--gripper-status-code", type=lambda value: int(value, 0), default=0x01)

    cams = parser.add_argument_group("cameras")
    cams.add_argument("--wrist-realsense-serial", default="250122075719", help="RealSense serial of the wrist camera ('' to disable).")
    cams.add_argument("--top-opencv-index", type=int, default=0, help="OpenCV index of the top camera (-1 to disable).")
    cams.add_argument("--camera-width", type=int, default=640)
    cams.add_argument("--camera-height", type=int, default=480)
    cams.add_argument("--camera-fps", type=int, default=30)


class PiperXRobot:
    """PiperX arm + cameras. Thread-safe: a teleop control thread and a recording thread may share it."""

    def __init__(self, config: PiperXRobotConfig, *, connect_cameras: bool = True):
        self.config = config
        self.is_connected = False
        self.arm = None
        self._arm_lock = threading.RLock()
        self._last_mode_refresh_t = 0.0
        self._speed_ratio = int(config.speed_ratio)
        self.cameras: dict[str, Any] = {}
        if connect_cameras and config.wrist_realsense_serial:
            self.cameras["wrist"] = RealSenseCamera(
                config.wrist_realsense_serial, config.camera_width, config.camera_height, config.camera_fps
            )
        if connect_cameras and config.top_opencv_index is not None and config.top_opencv_index >= 0:
            self.cameras["top"] = OpenCVCamera(
                config.top_opencv_index, config.camera_width, config.camera_height, config.camera_fps
            )

    def connect(self) -> None:
        try:
            from piper_sdk import C_PiperInterface_V2, LogLevel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("piper_sdk is not installed. Run `pip install piper_sdk`.") from exc

        cfg = self.config
        self.arm = C_PiperInterface_V2(
            can_name=cfg.can,
            judge_flag=cfg.judge_flag,
            can_auto_init=cfg.can_auto_init,
            logger_level=getattr(LogLevel, cfg.piper_log_level.upper(), LogLevel.WARNING),
        )
        self.arm.ConnectPort()
        if cfg.startup_sleep_s > 0:
            time.sleep(cfg.startup_sleep_s)
        self._guard_ctrl_mode()
        self._enable()
        self._send_motion_mode(self._speed_ratio)
        connected = []
        try:
            for camera in self.cameras.values():
                camera.connect()
                connected.append(camera)
        except Exception:
            for camera in connected:
                camera.disconnect()
            self.arm.DisconnectPort()
            raise
        self.is_connected = True

    def _guard_ctrl_mode(self, timeout_s: float = 0.5) -> None:
        """Refuse to drive an arm that is configured as a teaching/master arm."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            status = self.arm.GetArmStatus()
            if getattr(status, "time_stamp", 0.0) > 0.0:
                mode = getattr(getattr(status, "arm_status", None), "ctrl_mode", None)
                mode = int(getattr(mode, "value", mode)) if mode is not None else None
                if mode in {PIPER_CTRL_MODE_TEACH, PIPER_CTRL_MODE_LINKAGE_TEACH_INPUT}:
                    self.arm.MasterSlaveConfig(0xFC, 0x00, 0x00, 0x00)
                    raise RuntimeError(
                        f"[{self.config.can}] arm is in master/teaching role (ctrl_mode=0x{mode:02X}). "
                        "The follower role command was sent; power-cycle the arm and retry."
                    )
                return
            time.sleep(0.02)
        print(f"[PiperX] Warning: could not read ctrl_mode on {self.config.can}; check CAN wiring and power.")

    def _enable(self) -> None:
        deadline = time.monotonic() + max(0.0, self.config.enable_timeout_s)
        while time.monotonic() < deadline:
            if bool(self.arm.EnablePiper()):
                return
            time.sleep(0.2)
        print("[PiperX] Warning: EnablePiper did not report success before timeout.")

    def _send_motion_mode(self, speed_ratio: int) -> None:
        speed = min(100, max(0, int(speed_ratio)))
        mit_mode = 0xAD if self.config.high_follow else 0x00
        with self._arm_lock:
            self.arm.MotionCtrl_2(0x01, 0x01, speed, mit_mode)
        self._speed_ratio = speed
        self._last_mode_refresh_t = time.monotonic()

    def read_joint_state(self) -> np.ndarray:
        """7D state: six joints (deg) + gripper opening (mm)."""
        with self._arm_lock:
            joint_state = getattr(self.arm.GetArmJointMsgs(), "joint_state", None)
            gripper_state = getattr(self.arm.GetArmGripperMsgs(), "gripper_state", None)
        values = [milli_to_unit(getattr(joint_state, name, 0)) for name in PIPER_JOINT_NAMES]
        values.append(abs(milli_to_unit(getattr(gripper_state, "grippers_angle", 0))))
        return np.asarray(values, dtype=np.float32)

    def read_images(self) -> dict[str, np.ndarray]:
        return {name: camera.read() for name, camera in self.cameras.items()}

    def get_observation(self) -> dict[str, Any]:
        """``{"joint_1.pos": ..., "gripper.pos": ..., "<camera>": HxWx3 uint8}``."""
        state = self.read_joint_state()
        obs: dict[str, Any] = {key: float(value) for key, value in zip(PIPER_ACTION_KEYS, state, strict=True)}
        obs.update(self.read_images())
        return obs

    def send_joint_command(self, target: np.ndarray, speed_ratio: int | None = None) -> np.ndarray:
        """Send a 7D target (six joints in deg + gripper mm). Returns the quantised command actually sent.

        Passing ``speed_ratio`` re-sends MotionCtrl_2 with that speed before the joint command.
        """
        if speed_ratio is not None:
            self._send_motion_mode(int(speed_ratio))
        elif self.config.mode_refresh_interval_s > 0 and (
            time.monotonic() - self._last_mode_refresh_t >= self.config.mode_refresh_interval_s
        ):
            self._send_motion_mode(self._speed_ratio)

        target = np.asarray(target, dtype=np.float64)
        joint_milli = [unit_to_milli(value) for value in target[:6]]
        gripper_milli = unit_to_milli(target[6])
        with self._arm_lock:
            self.arm.JointCtrl(*joint_milli)
            self.arm.GripperCtrl(gripper_milli, self.config.gripper_effort, self.config.gripper_status_code, 0x00)
        return np.asarray([milli_to_unit(v) for v in (*joint_milli, gripper_milli)], dtype=np.float32)

    def send_action(self, action: dict[str, float]) -> np.ndarray:
        """Dict form of :meth:`send_joint_command` (keys from ``PIPER_ACTION_KEYS``)."""
        return self.send_joint_command(np.asarray([action[key] for key in PIPER_ACTION_KEYS]))

    def disconnect(self) -> None:
        try:
            if self.config.disable_on_disconnect and self.arm is not None:
                with self._arm_lock:
                    self.arm.DisableArm(7)
        finally:
            if self.arm is not None:
                self.arm.DisconnectPort()
            for camera in self.cameras.values():
                if camera.is_connected:
                    camera.disconnect()
            self.is_connected = False


def list_cameras() -> None:
    """Print RealSense serial numbers and UVC capture devices (for --wrist-realsense-serial / --top-opencv-index)."""
    from pathlib import Path

    print("RealSense cameras (--wrist-realsense-serial):")
    try:
        import pyrealsense2 as rs

        devices = list(rs.context().query_devices())
        for dev in devices:
            print(f"  {dev.get_info(rs.camera_info.serial_number)}   {dev.get_info(rs.camera_info.name)}")
        if not devices:
            print("  none found")
    except ImportError:
        print("  pyrealsense2 is not installed")

    print("UVC cameras (--top-opencv-index):")
    found = False
    for node in sorted(Path("/sys/class/video4linux").glob("video*"), key=lambda p: int(p.name[5:])):
        name = (node / "name").read_text().strip()
        index_file = node / "index"
        # Each camera exposes several nodes; the capture node has index 0. RealSense nodes are listed above.
        if "RealSense" in name or (index_file.exists() and index_file.read_text().strip() != "0"):
            continue
        print(f"  {node.name[5:]:>3}   /dev/{node.name}   {name}")
        found = True
    if not found:
        print("  none found")


def limit_joint_step(target: np.ndarray, current_cmd: np.ndarray, max_joint_speed_deg_s: float, hz: float) -> np.ndarray:
    """Rate-limit the six joints of ``target`` relative to ``current_cmd``; the gripper is passed through."""
    if max_joint_speed_deg_s <= 0:
        return np.asarray(target, dtype=np.float32).copy()
    max_step = max_joint_speed_deg_s / max(hz, 1.0)
    next_cmd = np.asarray(target, dtype=np.float32).copy()
    next_cmd[:6] = current_cmd[:6] + np.clip(target[:6] - current_cmd[:6], -max_step, max_step)
    return next_cmd
