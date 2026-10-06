"""Map a VR controller pose to a PiperX end-effector pose target.

The mapping is relative: while the deadman (grip/squeeze) is held, the controller's
displacement from where the grip started is scaled and added to the end-effector pose
at that moment; the controller's relative rotation is applied in the robot frame.
Releasing the grip re-anchors, so the operator can "clutch" to reposition their hand.

Robot frame: +X forward (away from the base), +Y left, +Z up.
WebXR frame: +X right, +Y up, -Z forward (relative to where the headset faced at start).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class PiperPose:
    x_mm: float
    y_mm: float
    z_mm: float
    rx_deg: float
    ry_deg: float
    rz_deg: float

    def as_transform(self) -> np.ndarray:
        xyz = np.array([self.x_mm, self.y_mm, self.z_mm], dtype=float) / 1000.0
        rot = R.from_euler("xyz", [self.rx_deg, self.ry_deg, self.rz_deg], degrees=True).as_matrix()
        t = np.eye(4)
        t[:3, :3] = rot
        t[:3, 3] = xyz
        return t

    def __str__(self) -> str:
        return (
            f"x={self.x_mm:.1f} y={self.y_mm:.1f} z={self.z_mm:.1f} mm "
            f"rx={self.rx_deg:.1f} ry={self.ry_deg:.1f} rz={self.rz_deg:.1f} deg"
        )


@dataclass
class Workspace:
    x_min: float = -250.0
    x_max: float = 450.0
    y_min: float = -350.0
    y_max: float = 350.0
    z_min: float = 80.0
    z_max: float = 650.0
    rx_min: float = -90.0
    rx_max: float = 90.0
    ry_min: float = -90.0
    ry_max: float = 90.0
    rz_min: float = -180.0
    rz_max: float = 180.0


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def transform_to_pose(t: np.ndarray) -> PiperPose:
    xyz_mm = t[:3, 3] * 1000.0
    rpy_deg = R.from_matrix(t[:3, :3]).as_euler("xyz", degrees=True)
    return PiperPose(float(xyz_mm[0]), float(xyz_mm[1]), float(xyz_mm[2]), *[float(v) for v in rpy_deg])


def clamp_pose(pose: PiperPose, ws: Workspace, clamp_position: bool, clamp_orientation: bool) -> PiperPose:
    return PiperPose(
        x_mm=clamp(pose.x_mm, ws.x_min, ws.x_max) if clamp_position else pose.x_mm,
        y_mm=clamp(pose.y_mm, ws.y_min, ws.y_max) if clamp_position else pose.y_mm,
        z_mm=clamp(pose.z_mm, ws.z_min, ws.z_max) if clamp_position else pose.z_mm,
        rx_deg=clamp(pose.rx_deg, ws.rx_min, ws.rx_max) if clamp_orientation else pose.rx_deg,
        ry_deg=clamp(pose.ry_deg, ws.ry_min, ws.ry_max) if clamp_orientation else pose.ry_deg,
        rz_deg=clamp(pose.rz_deg, ws.rz_min, ws.rz_max) if clamp_orientation else pose.rz_deg,
    )


def limit_step(current: PiperPose, target: PiperPose, max_step_mm: float, max_step_deg: float) -> PiperPose:
    def step(cur: float, tgt: float, limit: float) -> float:
        return cur + clamp(tgt - cur, -limit, limit)

    return PiperPose(
        x_mm=step(current.x_mm, target.x_mm, max_step_mm),
        y_mm=step(current.y_mm, target.y_mm, max_step_mm),
        z_mm=step(current.z_mm, target.z_mm, max_step_mm),
        rx_deg=step(current.rx_deg, target.rx_deg, max_step_deg),
        ry_deg=step(current.ry_deg, target.ry_deg, max_step_deg),
        rz_deg=step(current.rz_deg, target.rz_deg, max_step_deg),
    )


def blend_angle_deg(current: float, target: float, alpha: float) -> float:
    delta = (target - current + 180.0) % 360.0 - 180.0
    return current + delta * alpha


def blend_pose(current: PiperPose, target: PiperPose, alpha: float) -> PiperPose:
    return PiperPose(
        x_mm=current.x_mm + (target.x_mm - current.x_mm) * alpha,
        y_mm=current.y_mm + (target.y_mm - current.y_mm) * alpha,
        z_mm=current.z_mm + (target.z_mm - current.z_mm) * alpha,
        rx_deg=blend_angle_deg(current.rx_deg, target.rx_deg, alpha),
        ry_deg=blend_angle_deg(current.ry_deg, target.ry_deg, alpha),
        rz_deg=blend_angle_deg(current.rz_deg, target.rz_deg, alpha),
    )


def pose_distance_mm_deg(a: PiperPose, b: PiperPose) -> tuple[float, float]:
    linear = math.sqrt((a.x_mm - b.x_mm) ** 2 + (a.y_mm - b.y_mm) ** 2 + (a.z_mm - b.z_mm) ** 2)
    angular = math.sqrt((a.rx_deg - b.rx_deg) ** 2 + (a.ry_deg - b.ry_deg) ** 2 + (a.rz_deg - b.rz_deg) ** 2)
    return linear, angular


def should_send_pose(target: PiperPose, last_sent_pose: PiperPose | None, min_delta_mm: float, min_delta_deg: float) -> bool:
    if last_sent_pose is None:
        return True
    linear, angular = pose_distance_mm_deg(target, last_sent_pose)
    return linear >= min_delta_mm or angular >= min_delta_deg


class PiperXVRMapper:
    def __init__(
        self,
        home: PiperPose,
        workspace: Workspace,
        scale_mm_per_m: float,
        y_sign: float,
        z_sign: float,
        rot_x_sign: float,
        rot_y_sign: float,
        rot_z_sign: float,
        rot_scale: float,
        vr_yaw_deg: float,
        frame_forward_axis: str,
        max_step_mm: float,
        max_step_deg: float,
        clamp_position: bool,
        clamp_orientation: bool,
        target_filter_alpha: float,
        target_deadband_mm: float,
        target_deadband_deg: float,
    ):
        self.home = home
        self.workspace = workspace
        self.scale_mm_per_m = scale_mm_per_m
        self.y_sign = y_sign
        self.z_sign = z_sign
        self.rot_x_sign = rot_x_sign
        self.rot_y_sign = rot_y_sign
        self.rot_z_sign = rot_z_sign
        self.rot_scale = rot_scale
        self.vr_yaw_deg = vr_yaw_deg
        self.frame_forward_axis = frame_forward_axis
        self.max_step_mm = max_step_mm
        self.max_step_deg = max_step_deg
        self.clamp_position = clamp_position
        self.clamp_orientation = clamp_orientation
        self.target_filter_alpha = target_filter_alpha
        self.target_deadband_mm = target_deadband_mm
        self.target_deadband_deg = target_deadband_deg
        self.vr_home_pos: np.ndarray | None = None
        self.vr_home_quaternion: np.ndarray | None = None
        self.pose_origin = home
        self.current_target = home

    def _vr_yaw_rotation(self) -> R:
        return R.from_euler("y", self.vr_yaw_deg, degrees=True)

    def calibrate_frame_from_controller(self, quaternion: np.ndarray | None) -> bool:
        """Align robot +X with the direction the controller currently points (horizontal part)."""
        if quaternion is None:
            print("[VR] No controller orientation; cannot align the frame.")
            return False
        axis_map = {
            "+x": np.array([1.0, 0.0, 0.0]),
            "-x": np.array([-1.0, 0.0, 0.0]),
            "+z": np.array([0.0, 0.0, 1.0]),
            "-z": np.array([0.0, 0.0, -1.0]),
        }
        forward = R.from_quat(quaternion).apply(axis_map[self.frame_forward_axis])
        horizontal = np.array([forward[0], 0.0, forward[2]], dtype=float)
        norm = np.linalg.norm(horizontal)
        if norm < 1e-6:
            print("[VR] Controller is too vertical; point it horizontally along robot +X.")
            return False
        horizontal /= norm
        self.vr_yaw_deg = math.degrees(math.atan2(horizontal[2], horizontal[0])) + 90.0
        print(f"[VR] Frame aligned: controller {self.frame_forward_axis} -> robot +X (vr_yaw_deg={self.vr_yaw_deg:.1f})")
        return True

    def anchor(self, position: np.ndarray, quaternion: np.ndarray | None, pose_origin: PiperPose | None = None) -> None:
        """Start a new relative-motion segment at the current controller pose."""
        self.vr_home_pos = np.asarray(position, dtype=float).copy()
        self.vr_home_quaternion = None if quaternion is None else np.asarray(quaternion, dtype=float).copy()
        self.pose_origin = pose_origin or self.home
        self.current_target = self.pose_origin

    @property
    def is_anchored(self) -> bool:
        return self.vr_home_pos is not None

    def map(self, position: np.ndarray, quaternion: np.ndarray | None) -> PiperPose:
        if self.vr_home_pos is None:
            self.anchor(position, quaternion)
        d = self._vr_yaw_rotation().apply(np.asarray(position, dtype=float) - self.vr_home_pos)
        rx_deg, ry_deg, rz_deg = self._map_orientation(quaternion)
        raw_target = PiperPose(
            x_mm=self.pose_origin.x_mm + (-d[2]) * self.scale_mm_per_m,
            y_mm=self.pose_origin.y_mm + self.y_sign * d[0] * self.scale_mm_per_m,
            z_mm=self.pose_origin.z_mm + self.z_sign * d[1] * self.scale_mm_per_m,
            rx_deg=rx_deg,
            ry_deg=ry_deg,
            rz_deg=rz_deg,
        )
        clamped = clamp_pose(raw_target, self.workspace, self.clamp_position, self.clamp_orientation)
        limited = limit_step(self.current_target, clamped, self.max_step_mm, self.max_step_deg)
        linear_delta, angular_delta = pose_distance_mm_deg(self.current_target, limited)
        if linear_delta < self.target_deadband_mm and angular_delta < self.target_deadband_deg:
            return self.current_target
        if self.target_filter_alpha < 1.0:
            limited = blend_pose(self.current_target, limited, self.target_filter_alpha)
        self.current_target = limited
        return self.current_target

    @staticmethod
    def _angle_delta(current: float, origin: float) -> float:
        return (current - origin + 180.0) % 360.0 - 180.0

    def _map_orientation(self, quaternion: np.ndarray | None) -> tuple[float, float, float]:
        origin = self.pose_origin
        if self.vr_home_quaternion is None or quaternion is None:
            return origin.rx_deg, origin.ry_deg, origin.rz_deg
        vr_relative = R.from_quat(quaternion) * R.from_quat(self.vr_home_quaternion).inv()
        vr_yaw = self._vr_yaw_rotation()
        vr_relative = vr_yaw * vr_relative * vr_yaw.inv()
        basis_vr_to_piper = np.array([[0.0, 0.0, -1.0], [self.y_sign, 0.0, 0.0], [0.0, self.z_sign, 0.0]])
        piper_relative = R.from_matrix(basis_vr_to_piper @ vr_relative.as_matrix() @ basis_vr_to_piper.T)
        if self.rot_scale != 1.0:
            piper_relative = R.from_rotvec(piper_relative.as_rotvec() * self.rot_scale)
        origin_piper = R.from_euler("xyz", [origin.rx_deg, origin.ry_deg, origin.rz_deg], degrees=True)
        rx_deg, ry_deg, rz_deg = (piper_relative * origin_piper).as_euler("xyz", degrees=True)
        return (
            origin.rx_deg + self._angle_delta(float(rx_deg), origin.rx_deg) * self.rot_x_sign,
            origin.ry_deg + self._angle_delta(float(ry_deg), origin.ry_deg) * self.rot_y_sign,
            origin.rz_deg + self._angle_delta(float(rz_deg), origin.rz_deg) * self.rot_z_sign,
        )
