"""PiperX forward/inverse kinematics from the bundled URDF.

Shared by the robot client (delta-EE -> joint targets), the VR teleoperation
(controller pose -> joint targets) and the dataset tools (joint state -> EE pose).
Keeping a single implementation guarantees that recorded joints, converted EE
states and deployed EE actions all use the same kinematic model.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

DEFAULT_URDF = Path(__file__).resolve().parent / "assets" / "piper_x_description" / "urdf" / "piper_x_description.urdf"


@dataclass
class Joint:
    name: str
    parent: str
    child: str
    joint_type: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


def parse_vec(text: str | None, default: Iterable[float]) -> np.ndarray:
    if text is None:
        return np.array(list(default), dtype=float)
    return np.array([float(v) for v in text.split()], dtype=float)


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def transform(xyz: np.ndarray, rot: np.ndarray) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = xyz
    return t


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    return R.from_rotvec(axis / norm * angle).as_matrix()


class PiperXKinematics:
    def __init__(self, urdf: Path = DEFAULT_URDF, base_link: str = "base_link", target_link: str = "link6"):
        self.urdf = Path(urdf)
        self.base_link = base_link
        self.target_link = target_link
        self.joints_by_child = self._load_joints(self.urdf)
        self.chain = self._build_chain(base_link, target_link)
        self.actuated = [j for j in self.chain if j.joint_type != "fixed"]
        self.names = [j.name for j in self.actuated]
        self.lower = np.array([j.lower for j in self.actuated], dtype=float)
        self.upper = np.array([j.upper for j in self.actuated], dtype=float)

    @staticmethod
    def _load_joints(urdf: Path) -> dict[str, Joint]:
        root = ET.parse(urdf).getroot()
        joints: dict[str, Joint] = {}
        for elem in root.findall("joint"):
            name = elem.attrib["name"]
            joint_type = elem.attrib.get("type", "fixed")
            parent = elem.find("parent").attrib["link"]
            child = elem.find("child").attrib["link"]
            origin = elem.find("origin")
            xyz = parse_vec(origin.attrib.get("xyz") if origin is not None else None, [0, 0, 0])
            rpy = parse_vec(origin.attrib.get("rpy") if origin is not None else None, [0, 0, 0])
            axis_elem = elem.find("axis")
            axis = parse_vec(axis_elem.attrib.get("xyz") if axis_elem is not None else None, [0, 0, 1])
            limit = elem.find("limit")
            if joint_type == "fixed":
                lower = upper = 0.0
            elif limit is not None:
                lower = float(limit.attrib.get("lower", -math.pi))
                upper = float(limit.attrib.get("upper", math.pi))
            else:
                lower, upper = -math.pi, math.pi
            joints[child] = Joint(name, parent, child, joint_type, xyz, rpy, axis, lower, upper)
        return joints

    def _build_chain(self, base_link: str, target_link: str) -> list[Joint]:
        chain: list[Joint] = []
        link = target_link
        while link != base_link:
            if link not in self.joints_by_child:
                raise ValueError(f"Cannot find joint from {base_link} to {target_link}; stuck at {link}")
            joint = self.joints_by_child[link]
            chain.append(joint)
            link = joint.parent
        chain.reverse()
        return chain

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Return the 4x4 base->link6 transform for joint angles ``q`` in radians."""
        if len(q) != len(self.actuated):
            raise ValueError(f"Expected {len(self.actuated)} joints, got {len(q)}")

        t = np.eye(4)
        q_index = 0
        for joint in self.chain:
            t = t @ transform(joint.xyz, rpy_matrix(joint.rpy))
            if joint.joint_type != "fixed":
                t = t @ transform(np.zeros(3), axis_angle_matrix(joint.axis, q[q_index]))
                q_index += 1
        return t

    def solve_ik(
        self,
        target_t: np.ndarray,
        seed: np.ndarray,
        pos_weight: float = 1.0,
        rot_weight: float = 0.25,
        max_nfev: int = 80,
        regularization_weight: float = 0.0,
        regularization_target: np.ndarray | None = None,
    ):
        seed = np.clip(np.asarray(seed, dtype=float), self.lower, self.upper)
        if regularization_target is None:
            regularization_target = seed
        regularization_target = np.clip(np.asarray(regularization_target, dtype=float), self.lower, self.upper)
        target_pos = target_t[:3, 3]
        target_rot = target_t[:3, :3]

        def residual(q: np.ndarray) -> np.ndarray:
            cur_t = self.fk(q)
            pos_err = (cur_t[:3, 3] - target_pos) * pos_weight
            rot_err = R.from_matrix(target_rot.T @ cur_t[:3, :3]).as_rotvec() * rot_weight
            if regularization_weight > 0:
                return np.concatenate([pos_err, rot_err, (q - regularization_target) * regularization_weight])
            return np.concatenate([pos_err, rot_err])

        return least_squares(
            residual,
            seed,
            bounds=(self.lower, self.upper),
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
            max_nfev=max_nfev,
        )

    def fk_pose_deg(self, joints_deg: np.ndarray) -> tuple[np.ndarray, R]:
        """FK for joint angles in degrees; returns (position in m, rotation)."""
        t = self.fk(np.deg2rad(np.asarray(joints_deg[:6], dtype=np.float64)))
        return t[:3, 3].copy(), R.from_matrix(t[:3, :3])

    def ee_state_from_joint_state(self, joint_state: np.ndarray) -> np.ndarray:
        """Map a 7D joint state (6 joints in deg + gripper mm) to the 8D EE policy state.

        Layout: ``[x, y, z, qx, qy, qz, qw, gripper_mm]`` -- the same layout written by
        ``dataset_tools/convert_to_ee.py`` and used to train the PiperX checkpoints.
        """
        pos, rot = self.fk_pose_deg(joint_state)
        return np.concatenate([pos, rot.as_quat(), [float(joint_state[6])]]).astype(np.float32)


def seed_candidates(kin: PiperXKinematics, preferred: np.ndarray) -> list[np.ndarray]:
    mid = (kin.lower + kin.upper) * 0.5
    candidates = [
        preferred,
        mid,
        np.array([0.0, math.pi / 2.0, -math.pi / 2.0, 0.0, 0.0, 0.0]),
        np.array([0.0, math.pi / 3.0, -math.pi / 3.0, 0.0, 0.0, 0.0]),
    ]
    return [np.clip(q, kin.lower, kin.upper) for q in candidates]
