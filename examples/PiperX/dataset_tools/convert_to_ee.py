#!/usr/bin/env python
"""Convert a PiperX joint-space LeRobot v3.0 recording to the EE-delta layout.

Input  (raw teleoperation recording):
    observation.state = measured joints [j1..j6 deg, gripper mm]
    action            = IK joint target (ignored; actions are rebuilt from the measured states)
Output (same fps, same episode ids, videos reused):
    observation.state = [x, y, z, qx, qy, qz, qw, gripper_mm]   (FK of the measured joints)
    action            = temporal-next-state delta:
        translation = position[t+1] - position[t]                  (base frame, m)
        rotation    = log(inverse(rotation[t]) * rotation[t+1])     (rotvec in the EE frame, rad)
        gripper     = absolute gripper state[t+1]                   (mm)
        last frame  = zero translation/rotation delta and the current gripper state
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from examples.PiperX.common.kinematics import DEFAULT_URDF, PiperXKinematics
from examples.PiperX.dataset_tools.dataset_utils import read_dataset_data, stats_for_df, update_flat_episode_stats
from examples.PiperX.prepare_modality import build_modality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PiperX joint LeRobot data to EE-delta data.")
    parser.add_argument("--src", type=Path, required=True, help="Raw joint-space LeRobot v3.0 dataset.")
    parser.add_argument("--dst", type=Path, required=True, help="Output EE dataset directory.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="PiperX URDF (default: bundled).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--video-mode",
        choices=("hardlink", "copy", "symlink", "skip"),
        default="hardlink",
        help="How to reuse videos in the output dataset (they are not modified).",
    )
    return parser.parse_args()


def copy_or_link_tree(src: Path, dst: Path, mode: str) -> None:
    if mode == "skip" or not src.exists():
        return
    for file in src.rglob("*"):
        if not file.is_file():
            continue
        rel = file.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        if mode == "hardlink":
            os.link(file, target)
        elif mode == "symlink":
            target.symlink_to(file)
        elif mode == "copy":
            shutil.copy2(file, target)


def convert_dataframe(df: pd.DataFrame, kin: PiperXKinematics) -> pd.DataFrame:
    converted = df.copy()
    state_arrays = [np.asarray(value, dtype=np.float64) for value in df["observation.state"].to_numpy()]
    poses = [kin.fk_pose_deg(value) for value in state_arrays]
    new_states = [
        np.concatenate([pos, rot.as_quat(), [state[6]]]).astype(np.float32)
        for state, (pos, rot) in zip(state_arrays, poses)
    ]
    new_actions: list[np.ndarray] = []

    episode_ids = df["episode_index"].to_numpy(dtype=np.int64)
    for index, (state_arr, (state_pos, state_rot)) in enumerate(zip(state_arrays, poses)):
        has_next = index + 1 < len(poses) and episode_ids[index + 1] == episode_ids[index]
        if has_next:
            next_state = state_arrays[index + 1]
            next_pos, next_rot = poses[index + 1]
            delta_pos = next_pos - state_pos
            delta_rotvec = (state_rot.inv() * next_rot).as_rotvec()
            gripper = next_state[6]
        else:
            delta_pos = np.zeros(3, dtype=np.float64)
            delta_rotvec = np.zeros(3, dtype=np.float64)
            gripper = state_arr[6]
        new_actions.append(np.concatenate([delta_pos, delta_rotvec, [gripper]]).astype(np.float32))

    converted["observation.state"] = new_states
    converted["action"] = new_actions
    return converted


def update_info(src_info: dict) -> dict:
    info = dict(src_info)
    features = dict(info["features"])
    features["observation.state"] = {
        "dtype": "float32",
        "names": [
            "eef.x_m",
            "eef.y_m",
            "eef.z_m",
            "eef.qx",
            "eef.qy",
            "eef.qz",
            "eef.qw",
            "gripper.pos_mm",
        ],
        "shape": [8],
    }
    features["action"] = {
        "dtype": "float32",
        "names": [
            "eef.dx_m",
            "eef.dy_m",
            "eef.dz_m",
            "eef.drx_rad",
            "eef.dry_rad",
            "eef.drz_rad",
            "gripper.pos_mm",
        ],
        "shape": [7],
    }
    info["features"] = features
    info["robot_type"] = "piperx_follower_ee_delta"
    return info


def write_info_and_stats(dst: Path, src_stats: dict) -> pd.DataFrame:
    all_df = read_dataset_data(dst)
    if len(all_df) == 0:
        raise RuntimeError(f"No converted rows found in {dst}")
    src_stats.update(stats_for_df(all_df))
    (dst / "meta/stats.json").write_text(json.dumps(src_stats, indent=2) + "\n")

    info_path = dst / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["total_frames"] = int(len(all_df))
    info["total_episodes"] = int(all_df["episode_index"].nunique())
    info["splits"] = {"train": f"0:{info['total_episodes']}"}
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    return all_df


def main() -> None:
    args = parse_args()
    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    if not args.urdf.exists():
        raise FileNotFoundError(args.urdf)
    src_info = json.loads((src / "meta/info.json").read_text())
    state_shape = src_info["features"]["observation.state"]["shape"]
    if list(state_shape) != [7]:
        raise ValueError(
            f"Expected a joint-space recording with 7D observation.state (6 joints deg + gripper mm), "
            f"got shape={state_shape}. Is {src} already an EE dataset?"
        )
    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dst} already exists. Use --overwrite to replace it.")
        shutil.rmtree(dst)

    dst_data = dst / "data"
    dst_meta = dst / "meta"
    dst_videos = dst / "videos"
    dst_data.mkdir(parents=True, exist_ok=True)
    dst_meta.mkdir(parents=True, exist_ok=True)

    kin = PiperXKinematics(args.urdf)

    shutil.copy2(src / "meta/tasks.parquet", dst_meta / "tasks.parquet")
    copy_or_link_tree(src / "videos", dst_videos, args.video_mode)
    (dst_meta / "info.json").write_text(json.dumps(update_info(src_info), indent=2) + "\n")
    # Same layout that examples/PiperX/prepare_modality.py writes before training.
    (dst_meta / "modality.json").write_text(json.dumps(build_modality(), indent=2) + "\n")

    per_episode_stats: dict[int, dict[str, dict[str, list]]] = {}
    for src_file in sorted((src / "data").glob("chunk-*/*.parquet")):
        rel = src_file.relative_to(src / "data")
        out_file = dst_data / rel
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(src_file)
        if len(df) == 0:
            continue
        out_df = convert_dataframe(df, kin)
        out_df.to_parquet(out_file, index=False)
        for episode_index, ep_df in out_df.groupby("episode_index", sort=False):
            per_episode_stats[int(episode_index)] = stats_for_df(ep_df)
        print(f"[DATA] wrote {out_file} rows={len(out_df)}")

    src_stats = json.loads((src / "meta/stats.json").read_text())

    for src_file in sorted((src / "meta/episodes").glob("chunk-*/*.parquet")):
        rel = src_file.relative_to(src / "meta/episodes")
        out_file = dst_meta / "episodes" / rel
        out_file.parent.mkdir(parents=True, exist_ok=True)
        ep_meta = pd.read_parquet(src_file)
        rows = [
            update_flat_episode_stats(row.copy(), per_episode_stats[int(row["episode_index"])])
            for _, row in ep_meta.iterrows()
        ]
        if not rows:
            continue
        pd.DataFrame(rows).to_parquet(out_file, index=False)
        print(f"[META] wrote {out_file} episodes={len(rows)}")

    all_df = write_info_and_stats(dst, src_stats)

    action = np.stack(all_df["action"].to_numpy())
    state = np.stack(all_df["observation.state"].to_numpy())
    print(f"\n[DONE] wrote EE dataset: {dst}")
    print(f"[CHECK] frames={len(all_df)} episodes={all_df['episode_index'].nunique()}")
    print(f"[CHECK] state xyz m min={state[:, :3].min(axis=0)} max={state[:, :3].max(axis=0)}")
    print(f"[CHECK] action dxyz m min={action[:, :3].min(axis=0)} max={action[:, :3].max(axis=0)}")
    print(f"[CHECK] action drot rad min={action[:, 3:6].min(axis=0)} max={action[:, 3:6].max(axis=0)}")
    print(f"[CHECK] action gripper min={action[:, 6].min():.1f} max={action[:, 6].max():.1f}")


if __name__ == "__main__":
    main()
