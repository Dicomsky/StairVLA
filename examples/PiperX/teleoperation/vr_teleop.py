"""VR controller -> PiperX joint targets (pose mapping + local IK + slow homing).

Controls (right hand drives the arm):
  right grip (hold)  deadman: the arm follows the controller only while held
  right trigger      close the gripper (release to open)
  left X             align robot +X with the direction the right controller points
  left Y             slow move to the home pose
Recording buttons (right A / B) are reported as events and handled by ``record.py``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R

from examples.PiperX.common.kinematics import DEFAULT_URDF, PiperXKinematics, seed_candidates
from examples.PiperX.common.robot import DEFAULT_HOME_JOINTS_DEG
from examples.PiperX.teleoperation.vr_mapping import (
    PiperPose,
    PiperXVRMapper,
    Workspace,
    clamp,
    should_send_pose,
    transform_to_pose,
)
from examples.PiperX.teleoperation.vr_server import ControllerState


@dataclass
class VRTeleopConfig:
    urdf: str = str(DEFAULT_URDF)
    freq: float = 30.0
    max_linear_speed: float = 350.0  # mm/s
    max_angular_speed: float = 160.0  # deg/s
    scale: float = 600.0  # robot mm per controller metre
    target_filter_alpha: float = 0.35
    target_deadband_mm: float = 2.0
    target_deadband_deg: float = 1.0
    min_command_delta_mm: float = 1.0
    min_command_delta_deg: float = 0.5
    y_sign: float = -1.0
    z_sign: float = 1.0
    rot_x_sign: float = 1.0
    rot_y_sign: float = 1.0
    rot_z_sign: float = 1.0
    rot_scale: float = 1.0
    vr_yaw_deg: float = 0.0
    frame_forward_axis: str = "+x"
    home_joints_deg: tuple[float, ...] = DEFAULT_HOME_JOINTS_DEG
    clamp_position: bool = False
    clamp_orientation: bool = False
    workspace: Workspace = field(default_factory=Workspace)
    home_speed_ratio: int = 2
    home_max_joint_speed_deg_s: float = 8.0
    home_reached_tolerance_deg: float = 2.0
    gripper_open_mm: float = 100.0
    gripper_close_mm: float = 0.0
    gripper_trigger_threshold: float = 0.5
    ik_pos_tolerance_mm: float = 5.0
    ik_rot_tolerance_deg: float = 5.0
    ik_max_nfev: int = 60
    ik_joint_regularization: float = 0.02
    ik_fast_path: bool = True
    ik_fast_path_max_nfev: int = 25


class LocalPiperXIK:
    def __init__(self, config: VRTeleopConfig, current_q: np.ndarray):
        self.kin = PiperXKinematics(config.urdf)
        self.cfg = config
        self.current_q = np.array(current_q, dtype=float)
        self.fail_count = 0

    def pose_from_joints(self, q_rad: np.ndarray) -> PiperPose:
        return transform_to_pose(self.kin.fk(q_rad))

    def solve(self, pose: PiperPose) -> np.ndarray | None:
        cfg = self.cfg
        target_t = pose.as_transform()

        def solve_from(seed: np.ndarray, max_nfev: int):
            return self.kin.solve_ik(
                target_t,
                seed,
                pos_weight=1.0,
                rot_weight=0.25,
                max_nfev=max_nfev,
                regularization_weight=cfg.ik_joint_regularization,
                regularization_target=self.current_q,
            )

        def acceptable(result) -> bool:
            fk_t = self.kin.fk(result.x)
            pos_err_mm = np.linalg.norm(fk_t[:3, 3] - target_t[:3, 3]) * 1000.0
            rot_err_deg = np.degrees(np.linalg.norm(R.from_matrix(target_t[:3, :3].T @ fk_t[:3, :3]).as_rotvec()))
            return bool(result.success and pos_err_mm <= cfg.ik_pos_tolerance_mm and rot_err_deg <= cfg.ik_rot_tolerance_deg)

        if cfg.ik_fast_path:
            fast = solve_from(self.current_q, max(1, cfg.ik_fast_path_max_nfev))
            if acceptable(fast):
                return fast.x

        seeds = [self.current_q]
        for seed in seed_candidates(self.kin, self.current_q):
            if not any(np.allclose(seed, existing) for existing in seeds):
                seeds.append(seed)
        best = None
        for seed in seeds:
            result = solve_from(seed, cfg.ik_max_nfev)
            score = np.linalg.norm(result.fun)
            if best is None or score < best[0]:
                best = (score, result)
        if not acceptable(best[1]):
            self.fail_count += 1
            return None
        return best[1].x


@dataclass
class TeleopStep:
    command: np.ndarray  # 6 joints (deg) + gripper (mm)
    speed_ratio: int | None  # MotionCtrl_2 speed to apply with this command (None = keep)
    events: set[str]  # rising edges: "right_a", "right_b", "left_x", "left_y"
    tracking: bool  # right controller pose received
    deadman: bool
    gripper_closed: bool
    ik_ok: bool
    homing: bool
    home_reached: bool  # True only on the step the home pose is reached


class PiperXVRTeleop:
    def __init__(self, config: VRTeleopConfig, normal_speed_ratio: int):
        self.cfg = config
        self.normal_speed_ratio = int(normal_speed_ratio)
        self.home_q = np.deg2rad(np.asarray(config.home_joints_deg, dtype=float))
        self.ik = LocalPiperXIK(config, self.home_q)
        self.home_pose = self.ik.pose_from_joints(self.home_q)
        self.mapper = PiperXVRMapper(
            home=self.home_pose,
            workspace=config.workspace,
            scale_mm_per_m=config.scale,
            y_sign=config.y_sign,
            z_sign=config.z_sign,
            rot_x_sign=config.rot_x_sign,
            rot_y_sign=config.rot_y_sign,
            rot_z_sign=config.rot_z_sign,
            rot_scale=config.rot_scale,
            vr_yaw_deg=config.vr_yaw_deg,
            frame_forward_axis=config.frame_forward_axis,
            max_step_mm=config.max_linear_speed / config.freq,
            max_step_deg=config.max_angular_speed / config.freq,
            clamp_position=config.clamp_position,
            clamp_orientation=config.clamp_orientation,
            target_filter_alpha=clamp(config.target_filter_alpha, 0.0, 1.0),
            target_deadband_mm=config.target_deadband_mm,
            target_deadband_deg=config.target_deadband_deg,
        )
        self.command: np.ndarray | None = None
        self.last_sent_pose: PiperPose | None = None
        self._buttons_prev: dict[str, bool] = {}
        self._homing = False
        self._home_cmd_q: np.ndarray | None = None
        self._home_last_t: float | None = None
        self._restore_speed = False
        self._ik_ok = True

    # ------------------------------------------------------------------ state
    def sync_to_robot(self, feedback: np.ndarray) -> None:
        """Hold the current measured pose (call once after connecting)."""
        q = np.deg2rad(np.asarray(feedback[:6], dtype=float))
        self.command = np.asarray(feedback, dtype=np.float32).copy()
        self.command[6] = self.cfg.gripper_open_mm
        self.ik.current_q = q
        pose = self.ik.pose_from_joints(q)
        self.mapper.current_target = pose
        self.mapper.pose_origin = pose
        self.mapper.vr_home_pos = None
        self.last_sent_pose = pose

    @property
    def homing(self) -> bool:
        return self._homing

    def start_home(self) -> None:
        self._homing = True
        self._home_cmd_q = None
        self._home_last_t = None

    def cancel_home(self) -> None:
        if self._homing:
            self._homing = False
            self._restore_speed = True

    # ------------------------------------------------------------------ control step
    def step(self, feedback: np.ndarray, left: ControllerState | None, right: ControllerState | None) -> TeleopStep:
        cfg = self.cfg
        if self.command is None:
            self.sync_to_robot(feedback)
        events = self._button_events(left, right)
        gripper_closed = right is not None and right.trigger >= cfg.gripper_trigger_threshold
        gripper_mm = cfg.gripper_close_mm if gripper_closed else cfg.gripper_open_mm
        deadman = right is not None and right.squeeze

        if "left_x" in events and right is not None:
            self.mapper.calibrate_frame_from_controller(right.quaternion)

        if self._homing:
            home_reached = self._home_step(feedback, gripper_mm)
            return TeleopStep(
                self.command.copy(),
                self.normal_speed_ratio if home_reached else cfg.home_speed_ratio,
                events, right is not None, deadman, gripper_closed, True, not home_reached, home_reached,
            )

        speed = None
        if self._restore_speed:
            speed, self._restore_speed = self.normal_speed_ratio, False
        self.command[6] = gripper_mm
        if right is None:
            return TeleopStep(self.command.copy(), speed, events, False, False, gripper_closed, self._ik_ok, False, False)

        if not deadman:
            # Clutch released: re-anchor so the next grip continues from the current target.
            self.mapper.anchor(right.position, right.quaternion, pose_origin=self.mapper.current_target)
            self._ik_ok = True
            return TeleopStep(self.command.copy(), speed, events, True, False, gripper_closed, self._ik_ok, False, False)

        target = self.mapper.map(right.position, right.quaternion)
        if should_send_pose(target, self.last_sent_pose, cfg.min_command_delta_mm, cfg.min_command_delta_deg):
            q = self.ik.solve(target)
            if q is None:
                # Unreachable: hold the last reachable target and continue mapping from it.
                self.mapper.current_target = self.last_sent_pose
                self.mapper.pose_origin = self.last_sent_pose
                self._ik_ok = False
            else:
                self.ik.current_q = q
                self.last_sent_pose = target
                self.command[:6] = np.rad2deg(q).astype(np.float32)
                self._ik_ok = True
        return TeleopStep(self.command.copy(), speed, events, True, True, gripper_closed, self._ik_ok, False, False)

    def _home_step(self, feedback: np.ndarray, gripper_mm: float) -> bool:
        cfg = self.cfg
        fb_q = np.deg2rad(np.asarray(feedback[:6], dtype=float))
        now = time.perf_counter()
        if self._home_cmd_q is None:
            self._home_cmd_q = fb_q.copy()
            self._home_last_t = now
        if np.rad2deg(np.max(np.abs(self.home_q - fb_q))) <= cfg.home_reached_tolerance_deg:
            self._homing = False
            self._home_cmd_q = None
            self.ik.current_q = self.home_q.copy()
            self.mapper.current_target = self.home_pose
            self.mapper.pose_origin = self.home_pose
            self.mapper.vr_home_pos = None
            self.last_sent_pose = self.home_pose
            self.command = np.asarray([*np.rad2deg(self.home_q), gripper_mm], dtype=np.float32)
            return True
        dt = min(max(1.0 / max(cfg.freq, 1.0), now - self._home_last_t), 0.2)
        self._home_last_t = now
        max_step = math.radians(cfg.home_max_joint_speed_deg_s) * dt
        self._home_cmd_q = self._home_cmd_q + np.clip(self.home_q - self._home_cmd_q, -max_step, max_step)
        self.ik.current_q = self._home_cmd_q.copy()
        self.mapper.current_target = self.home_pose
        self.mapper.pose_origin = self.home_pose
        self.last_sent_pose = self.home_pose
        self.command = np.asarray([*np.rad2deg(self._home_cmd_q), gripper_mm], dtype=np.float32)
        return False

    def home_remaining_deg(self, feedback: np.ndarray) -> float:
        return float(np.max(np.abs(np.asarray(self.cfg.home_joints_deg) - np.asarray(feedback[:6]))))

    def _button_events(self, left: ControllerState | None, right: ControllerState | None) -> set[str]:
        current = {
            # Some WebXR runtimes report the left face buttons as a/b instead of x/y.
            "left_x": left is not None and (left.pressed("x") or left.pressed("b")),
            "left_y": left is not None and (left.pressed("y") or left.pressed("a")),
            "right_a": right is not None and right.pressed("a"),
            "right_b": right is not None and right.pressed("b"),
        }
        events = {name for name, down in current.items() if down and not self._buttons_prev.get(name, False)}
        self._buttons_prev = current
        return events
