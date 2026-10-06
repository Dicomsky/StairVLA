# PiperX robot client

Real-robot client used for the paper's PiperX experiments: camera capture, the websocket client
that queries [`deployment/model_server/server_policy.py`](../../../deployment/model_server/server_policy.py),
delta-EE to joint-target conversion through local inverse kinematics, and the safety limits.
Hardware access goes through [`../common/robot.py`](../common/robot.py) (direct `piper_sdk` over
CAN) and the kinematics through [`../common/kinematics.py`](../common/kinematics.py) (bundled URDF).

All commands below are run from the repository root. The scripts also work from any other
directory.

## 1. Hardware

- AgileX PiperX 6-DoF arm on a CAN adapter (`--can can0`).
- Top camera: UVC camera, OpenCV index `--top-opencv-index 0`.
- Wrist camera: Intel RealSense, `--wrist-realsense-serial <serial>` (replace the default with yours).
- Both cameras stream 640×480 RGB at 30 Hz; the policy receives `[top, wrist]`.

Python packages on the robot PC: `piper_sdk`, `pyrealsense2`, `opencv-python`, `numpy`, `scipy`,
`pandas`, `websockets`, `msgpack` (no LeRobot needed). Bring the CAN interface up once per boot,
e.g. with the `can_activate.sh` script shipped with `piper_sdk`, or:

```bash
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

The arm must be configured as a follower (not a teaching arm); the client refuses to drive it
otherwise.

## 2. Start the policy server

On the GPU machine, serve a stage-2 StairVLA checkpoint with the paper's inference settings
(see [`../README.md`](../README.md#3-inference)):

```bash
python deployment/model_server/server_policy.py \
    --ckpt_path results/Checkpoints/fruit25_stairvla_stage2/final_model/pytorch_model.pt \
    --port 10093 --use_bf16 --idle_timeout -1 \
    --hier_eval_num_chunks 3 \
    --denoise_step_scale 0.97 \
    --num_inference_timesteps 2 \
    --lower_refine_steps 1 \
    --lower_assumed_step_scale 0.97
```

The StarVLA baselines are served with just `--ckpt_path ... --port 10093 --use_bf16`.

## 3. Run the robot client

The client needs the checkpoint's normalization statistics. `--checkpoint` accepts the `.pt` file
or the run directory and finds `dataset_statistics.json` in it or a parent directory; use
`--stats-json` to point at the file directly. On the robot PC only that JSON file is needed.

First run a **dry run**: the arm and cameras are connected and the policy is queried, but no
joint command is sent. Check the printed targets.

```bash
python examples/PiperX/deployment/eval_policy.py \
    --checkpoint results/Checkpoints/fruit25_stairvla_stage2 \
    --host <server ip> --port 10093
```

Add `--execute` only when the dry run looks reasonable and the workspace is clear. Each trial:
type an instruction (ENTER reuses the previous one), the arm moves slowly to the home pose
`(0, 62.512, -66.452, 83, 0, 0)` deg, press ENTER to start, press ENTER to stop early (or wait
for `--trial-duration-s`), then label it `s`/`f`/`m` (or `q` to quit). Results are appended to
`outputs/piperx_eval/results.jsonl` (`--results-jsonl`).

| Task | Flags |
|---|---|
| Fruit25 | defaults (8 Hz, binary gripper threshold 80.6 mm, close 0 mm / open 100 mm) |
| PushBlock | `--control-hz 20 --gripper-binary-threshold 47.35 --gripper-close-mm 0.3` |

`--control-hz` must match the rate of the training data (8 Hz for Fruit25, 20 Hz for PushBlock).

### Control loop

When the action buffer is empty the client sends the current observation (two images, instruction,
8D EE state from forward kinematics, q99-normalized) and blocks until the server returns a chunk.
The chunk is unnormalized, anchored once to the measured EE pose, and integrated step by step:
translation deltas in the world frame, rotation deltas (axis-angle) in the EE frame. Each target
pose is solved to joints with IK seeded by the previous command, rate-limited, and sent with
`JointCtrl`. The gripper prediction is binarized: below `--gripper-binary-threshold` closes to
`--gripper-close-mm`, otherwise opens to `--gripper-open-mm`.

### Safety limits

| Limit | Flag | Default |
|---|---|---|
| Per-step translation (per axis) | `--max-ee-delta-m` | 0.05 m |
| Per-step rotation (norm) | `--max-ee-rot-delta-rad` | 0.20 rad |
| Joint velocity | `--max-joint-speed-deg-s` | 25 deg/s |
| Action range | `--safe` / `--safe-range` | dataset min/max ∩ the limits above |
| EE workspace | `--ee-workspace` / `--ee-workspace-margin-m` | state min/max from the statistics ± 0.02 m |
| Joint limits | built in | PiperX hard limits |
| IK tolerance | `--ik-pos-tolerance-m` / `--ik-rot-tolerance-rad` | 8 mm / 0.12 rad (step skipped if exceeded) |
| Homing | `--home-max-joint-speed-deg-s` / `--home-speed-ratio` | 8 deg/s / 2 |

## 4. Benchmark

[`eval_benchmark.py`](eval_benchmark.py) runs the resumable, video-recorded benchmark used for the
paper (25 Fruit25 tasks × 10 accepted trials by default; PushBlock via `--tasks-json`). See
[BENCHMARK.md](BENCHMARK.md) for the protocol, commands and output layout.

## 5. Tools

| Script | Purpose | Needs |
|---|---|---|
| [`replay_episode.py`](replay_episode.py) | Replay a recorded EE-delta episode through the same delta-EE → IK path (`--replay-base dataset/integrated/eval-chunk/feedback`); exports the episode video via `dataset_tools/inspect_episode.py` | dataset; arm only with `--execute` |
| [`validate_policy_actions.py`](validate_policy_actions.py) | Send dataset observations to a running server and compare the predicted chunks with the ground-truth actions (physical and normalized errors, CSV output) | dataset + server, no robot |
| [`run_visual_demo.py`](run_visual_demo.py) | Start a server for a suite/model preset and run selected tasks with the continuous phase-labelled top video | robot + GPU |
| [`render_phase_chunk_videos.py`](render_phase_chunk_videos.py) | Re-render the chunk-counter phase videos of benchmark runs | benchmark output |
| [`analyze_smoothness.py`](analyze_smoothness.py) | Trajectory smoothness metrics from benchmark runs | benchmark output |
| [`phase_video.py`](phase_video.py) | Library: continuous camera recorder with inference/execution phase overlay | |

```bash
# Dry-run replay (no CAN connection) of episode 3
python examples/PiperX/deployment/replay_episode.py \
    --dataset playground/Datasets/FruitV3_EE_8Hz_temporal_clean_v2 --episode 3

# Offline action check against a running server
python examples/PiperX/deployment/validate_policy_actions.py \
    --dataset-root playground/Datasets/FruitV3_EE_8Hz_temporal_clean_v2 \
    --checkpoint results/Checkpoints/fruit25_stairvla_stage2 --episodes 0-9
```

## 6. Flag reference (`eval_policy.py`, shared by `eval_benchmark.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint` / `--stats-json` | required (one of) | normalization statistics |
| `--host`, `--port` | `127.0.0.1`, `10093` | policy server |
| `--execute` | off | send commands to the arm |
| `--control-hz` | 8 | control rate (PushBlock: 20) |
| `--action-space` | `delta-ee` | `joint` is a LEGACY absolute-joint mode, requires `--policy-state-space joint` and 7D state statistics |
| `--policy-state-space` | `ee` | 8D EE state |
| `--action-norm`, `--state-norm` | `q99` | must match the training data config |
| `--trial-duration-s` | 30 (benchmark: 50) | trial time limit |
| `--speed-ratio` | 30 (benchmark: 20) | `MotionCtrl_2` speed ratio |
| `--gripper-binary-threshold`, `--gripper-open-mm`, `--gripper-close-mm` | 80.6, 100, 0 | binary gripper rule |
| `--camera-warmup-s` | 0.5 | camera frames discarded before each trial |
| `--home-joints-deg` | `0 62.512 -66.452 83 0 0` | home pose |
| `--urdf` | bundled | PiperX URDF override |
| `--debug-actions`, `--debug-chunk-summary` | off | print raw/unnormalized chunks |

Robot and camera flags (`--can`, `--wrist-realsense-serial`, `--top-opencv-index`, ...) come from
`add_robot_args` in [`../common/robot.py`](../common/robot.py); run any script with `--help` for
the full list.
