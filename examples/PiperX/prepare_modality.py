#!/usr/bin/env python3
"""Create modality metadata for the PiperX EE LeRobot datasets (Fruit25, PushBlock)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STATE_NAMES = [
    "eef.x_m",
    "eef.y_m",
    "eef.z_m",
    "eef.qx",
    "eef.qy",
    "eef.qz",
    "eef.qw",
    "gripper.pos_mm",
]

ACTION_NAMES = [
    "eef.dx_m",
    "eef.dy_m",
    "eef.dz_m",
    "eef.drx_rad",
    "eef.dry_rad",
    "eef.drz_rad",
    "gripper.pos_mm",
]


def _feature_dim(info: dict, key: str) -> int:
    shape = info["features"][key]["shape"]
    if len(shape) != 1:
        raise ValueError(f"Expected 1D feature for {key}, got shape={shape}")
    return int(shape[0])


def build_modality() -> dict:
    state = {}
    action = {}
    for idx, name in enumerate(STATE_NAMES):
        state[name] = {
            "start": idx,
            "end": idx + 1,
            "dtype": "float32",
            "absolute": True,
            "rotation_type": None,
            "original_key": "observation.state",
        }
    for idx, name in enumerate(ACTION_NAMES):
        action[name] = {
            "start": idx,
            "end": idx + 1,
            "dtype": "float32",
            "absolute": False,
            "rotation_type": None,
            "original_key": "action",
        }

    return {
        "state": state,
        "action": action,
        "video": {
            "top": {"original_key": "observation.images.top"},
            "wrist": {"original_key": "observation.images.wrist"},
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Local LeRobot dataset directory.")
    parser.add_argument("--expected-fps", type=int, default=None, help="Optional FPS check.")
    args = parser.parse_args()

    info_path = args.dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing {info_path}")

    with info_path.open("r") as f:
        info = json.load(f)

    expected = {
        "action",
        "observation.state",
        "observation.images.top",
        "observation.images.wrist",
        "task_index",
    }
    missing = sorted(expected - set(info.get("features", {}).keys()))
    if missing:
        raise KeyError(f"Dataset is missing expected feature keys: {missing}")

    fps = int(info.get("fps"))
    state_dim = _feature_dim(info, "observation.state")
    action_dim = _feature_dim(info, "action")
    if args.expected_fps is not None and fps != args.expected_fps:
        raise ValueError(f"Expected fps={args.expected_fps}, got fps={fps}")
    if state_dim != len(STATE_NAMES):
        raise ValueError(f"Expected state_dim={len(STATE_NAMES)}, got {state_dim}")
    if action_dim != len(ACTION_NAMES):
        raise ValueError(f"Expected action_dim={len(ACTION_NAMES)}, got {action_dim}")

    modality_path = args.dataset_dir / "meta" / "modality.json"
    modality_path.write_text(json.dumps(build_modality(), indent=2) + "\n")
    print(f"Wrote {modality_path}")
    print(f"Verified fps={fps}, state_dim={state_dim}, action_dim={action_dim}")


if __name__ == "__main__":
    main()
