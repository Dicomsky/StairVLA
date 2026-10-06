# PiperX real-robot experiments

Real-robot experiments on an AgileX PiperX 6-DoF arm with a fixed top camera and a wrist-mounted
RealSense camera (both 640×480 RGB).

- **State (8D):** end-effector position and quaternion `[x, y, z, qx, qy, qz, qw]` plus gripper opening.
- **Action (7D):** translation delta in the world frame, axis-angle rotation delta in the
  end-effector frame, and the absolute gripper target.

| Task | Description | Demos | Frames | Control rate |
|---|---|:---:|:---:|:---:|
| Fruit25 | 24 single-fruit pick-and-place tasks (4 fruits × 6 destinations) + 1 long-horizon sorting task | 1,186 | 109,684 | 8 Hz |
| PushBlock | Push a black block into a blue target region | 100 | 32,776 | 20 Hz |

## 1. Data

<!-- TODO: add Hugging Face dataset links once released -->
The datasets will be released on Hugging Face (TODO). Place them at:

```
playground/Datasets/FruitV3_EE_8Hz_temporal_clean_v2/              # Fruit25
playground/Datasets/PushBlockBlueSquare_EE_20Hz_temporal_clean_v2/  # PushBlock
```

Each launcher runs [`prepare_modality.py`](prepare_modality.py) first, which writes the
`meta/modality.json` layout the PiperX data config expects. The data mixtures are named `fruit25`
and `pushblock` in `starVLA/dataloader/gr00t_lerobot/mixtures.py`.

## 2. Training

All models use Qwen3-VL-2B-Instruct (`playground/Pretrained_models/Qwen3-VL-2B-Instruct`). The two
baselines are fine-tuned end to end with action horizon 8. StairVLA uses a high-level horizon of
H=32 and a refiner chunk size of h=8.

| Launcher | Fruit25 | PushBlock |
|---|---|---|
| StarVLA-GR00T baseline | [`fruit25/run_starvla_gr00t.sh`](fruit25/run_starvla_gr00t.sh) | [`pushblock/run_starvla_gr00t.sh`](pushblock/run_starvla_gr00t.sh) |
| StarVLA-π baseline | [`fruit25/run_starvla_pi.sh`](fruit25/run_starvla_pi.sh) | [`pushblock/run_starvla_pi.sh`](pushblock/run_starvla_pi.sh) |
| StairVLA stage 1 (high-level policy) | [`fruit25/run_stairvla_stage1.sh`](fruit25/run_stairvla_stage1.sh) | [`pushblock/run_stairvla_stage1.sh`](pushblock/run_stairvla_stage1.sh) |
| StairVLA stage 2 (refiner) | [`fruit25/run_stairvla_stage2.sh`](fruit25/run_stairvla_stage2.sh) | [`pushblock/run_stairvla_stage2.sh`](pushblock/run_stairvla_stage2.sh) |
| Steps | 30k | 10k |
| Global batch | 64 (StarVLA-π: 32), 4 GPUs | 8, 1 GPU |

The configs use `attn_implementation: sdpa`, so flash-attn is not required for these runs.

## 3. Inference

Serve a stage-2 checkpoint with the inference settings used in the paper:

- **High level:** denoising progress α=0.97 with 2 inference steps; action-context denoising scale 0.9
  (`--context_denoise_step_scale`).
- **Refiner:** 1 refinement step from an assumed progress of 0.97.
- **Reuse:** M=3 refinement cycles per high-level trajectory (24 executed actions).

```bash
python deployment/model_server/server_policy.py \
    --ckpt_path <stage-2 checkpoint> --port 5694 --use_bf16 \
    --hier_eval_num_chunks 3 \
    --denoise_step_scale 0.97 \
    --context_denoise_step_scale 0.9 \
    --num_inference_timesteps 2 \
    --lower_refine_steps 1 \
    --lower_assumed_step_scale 0.97
```

All methods run on the same RTX A4000. Inference is synchronous: when the action buffer is empty,
the control loop requests a new chunk and executes it at the task's control rate.

## 4. Robot client

[`deployment/`](deployment/README.md) holds the robot client that connects the policy server to the
PiperX. It captures both cameras, sends the 8D EE state, integrates each predicted delta-EE chunk and
converts it to joint targets with local IK under the paper's safety limits (≤0.05 m and ≤0.20 rad
per step, ≤25°/s per joint). The folder also has the 25-task Fruit25 benchmark
([BENCHMARK.md](deployment/BENCHMARK.md)), episode replay and offline action validation.

```bash
python examples/PiperX/deployment/eval_policy.py --checkpoint <stage-2 run dir> --port 5694           # dry run
python examples/PiperX/deployment/eval_policy.py --checkpoint <stage-2 run dir> --port 5694 --execute
```

Use `--control-hz 20` for PushBlock.

## 5. Teleoperation and data collection

The demonstrations were collected with a Meta Quest through WebXR:
[`teleoperation/`](teleoperation/README.md). Recordings are raw 30 Hz joint-space LeRobot datasets.
[`dataset_tools/`](dataset_tools/README.md) converts them into the EE-delta training datasets.
It computes temporal EE deltas, resamples to 8 Hz or 20 Hz, excludes episodes listed in a reviewed
manifest, and runs quality checks.

```
record.py (VR, 30 Hz joints) → convert_to_ee.py → resample.py (--fps 8 | 20, --exclusions) → prepare_modality.py → training
```

The robot-side extras (`piper_sdk`, `pyrealsense2`, `opencv-python`) are listed in
[`requirements.txt`](requirements.txt). Code shared by all three folders (IK, the bundled URDF, the
CAN/camera driver) is in [`common/`](common/).
