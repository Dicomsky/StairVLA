#!/usr/bin/env python3
"""Run selected PiperX tasks with continuous, phase-labelled top-camera video.

Starts the policy server for the chosen checkpoint (unless --no-launch-server), then runs
``eval_benchmark.py`` with ``--continuous-top-video`` and the suite's control settings.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.PiperX.deployment.eval_benchmark import FRUIT25_TASKS
from examples.PiperX.deployment.eval_policy import STATS_FILENAME


MODEL_SERVER_ARGS = {
    "starvla_gr00t": [],
    "stairvla": [
            "--hier_eval_mode", "default",
            "--hier_eval_num_chunks", "3",
            "--denoise_step_scale", "0.97",
            "--context_denoise_step_scale", "0.9",
            "--num_inference_timesteps", "2",
            "--lower_refine_steps", "1",
            "--lower_assumed_step_scale", "0.97",
    ],
    "starvla_pi": [],
}

# Default checkpoint run directories are the ones written by the launchers in
# examples/PiperX/{fruit25,pushblock}/ (run_root_dir=./results/Checkpoints).
SUITE_PRESETS = {
    "fruit25": {
        "tasks": FRUIT25_TASKS,
        "control_hz": 8.0,
        "gripper_action_mode": "binary",
        "gripper_binary_threshold": 80.6,
        "gripper_close_mm": 0.0,
        "gripper_open_mm": 100.0,
        "checkpoints": {
            "starvla_gr00t": "results/Checkpoints/fruit25_starvla_gr00t",
            "stairvla": "results/Checkpoints/fruit25_stairvla_stage2",
            "starvla_pi": "results/Checkpoints/fruit25_starvla_pi",
        },
    },
    "pushblock": {
        # Binary gripper split derived from the dataset for the demo videos. The paper's PushBlock
        # benchmark used --gripper-action-mode absolute (open 100 mm / close 0 mm) instead.
        "tasks": ["Push the black block into the blue square target at the center."],
        "control_hz": 20.0,
        "gripper_action_mode": "binary",
        "gripper_binary_threshold": 47.35,
        "gripper_close_mm": 0.3,
        "gripper_open_mm": 100.0,
        "checkpoints": {
            "starvla_gr00t": "results/Checkpoints/pushblock_starvla_gr00t",
            "stairvla": "results/Checkpoints/pushblock_stairvla_stage2",
            "starvla_pi": "results/Checkpoints/pushblock_starvla_pi",
        },
    },
}


def parse_task_ids(values: list[str]) -> list[int]:
    selected: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_text, end_text = part.split("-", 1)
                start, end = int(start_text), int(end_text)
                if end < start:
                    raise ValueError(f"Invalid descending task range: {part}")
                selected.extend(range(start, end + 1))
            else:
                selected.append(int(part))
    unique = list(dict.fromkeys(selected))
    invalid = [index for index in unique if not 1 <= index <= len(FRUIT25_TASKS)]
    if invalid:
        raise ValueError(f"Task IDs must be in [1, {len(FRUIT25_TASKS)}], got {invalid}")
    return unique


def websocket_server_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        from websockets.sync.client import connect

        with connect(
            f"ws://{host}:{port}",
            compression=None,
            max_size=None,
            open_timeout=timeout,
            close_timeout=timeout,
        ) as connection:
            connection.recv(timeout=timeout)
        return True
    except Exception:
        return False


def wait_for_server(process: subprocess.Popen, host: str, port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Policy server exited during startup with code {process.returncode}")
        if websocket_server_open(host, port):
            print(f"[VISUAL_DEMO] Policy server ready on {host}:{port}")
            return
        time.sleep(1.0)
    raise TimeoutError(f"Policy server did not open {host}:{port} within {timeout_s:.0f}s")


def checkpoint_file(checkpoint_dir: Path) -> Path:
    if checkpoint_dir.is_file():
        return checkpoint_dir
    candidates = [
        checkpoint_dir / "final_model" / "pytorch_model.pt",
        checkpoint_dir / "checkpoints" / "steps_30000_pytorch_model.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    found = sorted(checkpoint_dir.glob("**/*pytorch_model.pt"))
    if not found:
        raise FileNotFoundError(f"No pytorch_model.pt found below {checkpoint_dir}")
    return found[-1]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run selected Fruit25/PushBlock tasks and save an untouched 30 Hz top video plus a copy with "
            "red=inference and green=execution borders."
        )
    )
    parser.add_argument("--suite", choices=sorted(SUITE_PRESETS), default="fruit25")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_SERVER_ARGS),
        required=True,
        help="Selects the policy server arguments and the default checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Checkpoint run directory or .pt file. Defaults to the suite/model run directory under "
            "results/Checkpoints/ (as written by the examples/PiperX launchers)."
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        required=False,
        metavar="ID",
        help="One-based task IDs or ranges, for example: --tasks 1 4 24-25",
    )
    parser.add_argument("--episodes-per-task", type=int, default=3)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--start-trial", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/piperx_visual_demos"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--launch-server", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--server-start-timeout-s", type=float, default=300.0)
    parser.add_argument("--execute", action="store_true", help="Actually command the robot.")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--trial-duration-s", type=float, default=50.0)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--border-px", type=int, default=14)
    parser.add_argument("--state-log-hz", type=float, default=50.0)
    parser.add_argument("--wrist-realsense-serial", default="250122075719")
    parser.add_argument("--top-opencv-index", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.episodes_per_task <= 0:
        raise ValueError("--episodes-per-task must be > 0")
    suite = SUITE_PRESETS[args.suite]
    suite_tasks = suite["tasks"]
    if args.suite == "fruit25":
        if not args.tasks:
            raise ValueError("Fruit25 visualization requires --tasks with one-based Fruit25 task IDs")
        task_ids = parse_task_ids(args.tasks)
        tasks = [suite_tasks[index - 1] for index in task_ids]
    else:
        if args.tasks and args.tasks != ["1"]:
            raise ValueError(f"Suite {args.suite!r} has one task; omit --tasks or use --tasks 1")
        task_ids = [1]
        tasks = list(suite_tasks)
    checkpoint = args.checkpoint if args.checkpoint is not None else REPO_ROOT / suite["checkpoints"][args.model]
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Pass --checkpoint <run dir or .pt file>.")
    model_path = checkpoint_file(checkpoint)
    # The run directory is the nearest parent holding dataset_statistics.json (and config.yaml).
    checkpoint_dir = next(
        (directory for directory in model_path.parents if (directory / STATS_FILENAME).is_file()), None
    )
    if checkpoint_dir is None or not (checkpoint_dir / "config.yaml").is_file():
        raise FileNotFoundError(
            f"Incomplete checkpoint for {model_path}; expected config.yaml and {STATS_FILENAME} in its run directory"
        )
    stats_path = checkpoint_dir / STATS_FILENAME

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"visual_{args.suite}_{args.model}_{timestamp}"
    config_dir = args.output_root.expanduser().resolve() / ".task_lists"
    config_dir.mkdir(parents=True, exist_ok=True)
    task_path = config_dir / f"{run_name}.json"
    task_path.write_text(json.dumps(tasks, indent=2, ensure_ascii=False) + "\n")
    selection_path = config_dir / f"{run_name}.selection.json"
    selection = {
        "model": args.model,
        "suite": args.suite,
        "checkpoint": str(checkpoint_dir),
        "source_task_ids_1_based": task_ids,
        "tasks": tasks,
        "episodes_per_task": args.episodes_per_task,
    }
    selection_path.write_text(json.dumps(selection, indent=2, ensure_ascii=False) + "\n")

    print(f"[VISUAL_DEMO] suite={args.suite} model={args.model}")
    print(f"[VISUAL_DEMO] checkpoint={model_path}")
    print(
        "[VISUAL_DEMO] gripper="
        f"{suite['gripper_action_mode']} close={suite['gripper_close_mm']:.1f}mm "
        f"open={suite['gripper_open_mm']:.1f}mm "
        f"threshold={suite['gripper_binary_threshold']:.2f}mm"
    )
    print(f"[VISUAL_DEMO] tasks={task_ids} episodes_per_task={args.episodes_per_task}")
    for index, instruction in zip(task_ids, tasks):
        print(f"  task {index:02d}: {instruction}")
    print(f"[VISUAL_DEMO] output={args.output_root.expanduser().resolve() / run_name}")

    benchmark_command = [
        sys.executable,
        "-m", "examples.PiperX.deployment.eval_benchmark",
        "--host", args.host,
        "--port", str(args.port),
        "--stats-json", str(stats_path),
        "--run-name", run_name,
        "--checkpoint-id", f"{args.model}:{checkpoint_dir.name}",
        "--output-root", str(args.output_root.expanduser().resolve()),
        "--tasks-json", str(task_path),
        "--episodes-per-task", str(args.episodes_per_task),
        "--start-trial", str(args.start_trial),
        "--trial-duration-s", str(args.trial_duration_s),
        "--can", args.can,
        "--control-hz", str(suite["control_hz"]),
        "--action-chunk-stride", "1",
        "--max-ee-delta-m", "0.05",
        "--max-ee-rot-delta-rad", "0.20",
        "--max-joint-speed-deg-s", "25",
        "--speed-ratio", "20",
        "--gripper-action-mode", suite["gripper_action_mode"],
        "--gripper-binary-threshold", str(suite["gripper_binary_threshold"]),
        "--gripper-open-mm", str(suite["gripper_open_mm"]),
        "--gripper-close-mm", str(suite["gripper_close_mm"]),
        "--gripper-effort", "1000",
        "--wrist-realsense-serial", args.wrist_realsense_serial,
        "--top-opencv-index", str(args.top_opencv_index),
        "--camera-width", str(args.camera_width),
        "--camera-height", str(args.camera_height),
        "--camera-fps", str(args.camera_fps),
        "--state-log-hz", str(args.state_log_hz),
        "--no-record-video",
        "--continuous-top-video",
        "--phase-video-camera", "top",
        "--phase-video-fps", str(args.video_fps),
        "--phase-video-border-px", str(args.border_px),
        "--keep-all-recordings",
        "--demo",
    ]
    if args.execute:
        benchmark_command.append("--execute")
    if args.resume:
        benchmark_command.append("--resume-benchmark")
    if args.plan_only:
        benchmark_command.append("--plan-only")

    server_process: subprocess.Popen | None = None
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["DEBUG"] = "0"
    try:
        if args.launch_server and not args.plan_only:
            if websocket_server_open(args.host, args.port):
                raise RuntimeError(
                    f"{args.host}:{args.port} is already in use. Stop the old server, or use "
                    "--no-launch-server to connect to it intentionally."
                )
            server_command = [
                sys.executable,
                "-m", "deployment.model_server.server_policy",
                "--ckpt_path", str(model_path),
                "--port", str(args.port),
                "--use_bf16",
                "--idle_timeout", "-1",
                *MODEL_SERVER_ARGS[args.model],
            ]
            print("[VISUAL_DEMO] Starting policy server:")
            print("  " + " ".join(server_command))
            server_process = subprocess.Popen(server_command, cwd=REPO_ROOT, env=env)
            wait_for_server(server_process, args.host, args.port, args.server_start_timeout_s)
        elif not args.launch_server and not args.plan_only and not websocket_server_open(args.host, args.port):
            raise RuntimeError(f"No policy server is listening on {args.host}:{args.port}")

        print("[VISUAL_DEMO] Starting benchmark recorder...")
        subprocess.run(benchmark_command, cwd=REPO_ROOT, env=env, check=True)
        run_dir = args.output_root.expanduser().resolve() / run_name
        if run_dir.exists():
            (run_dir / "visual_demo_selection.json").write_text(
                json.dumps(selection, indent=2, ensure_ascii=False) + "\n"
            )
    finally:
        if server_process is not None and server_process.poll() is None:
            print("[VISUAL_DEMO] Stopping policy server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                server_process.kill()
                server_process.wait(timeout=5.0)


if __name__ == "__main__":
    main()
