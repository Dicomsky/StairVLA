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

- **High level:** denoising progress α=0.97 with 2 inference steps.
- **Refiner:** 1 refinement step from an assumed progress of 0.97.
- **Reuse:** M=3 refinement cycles per high-level trajectory (24 executed actions).

```bash
python deployment/model_server/server_policy.py \
    --ckpt_path <stage-2 checkpoint> --port 5694 --use_bf16 \
    --hier_eval_num_chunks 3 \
    --denoise_step_scale 0.97 \
    --num_inference_timesteps 2 \
    --lower_refine_steps 1 \
    --lower_assumed_step_scale 0.97
```

All methods run on the same RTX A4000. Inference is synchronous: when the action buffer is empty,
the control loop requests a new chunk and executes it at the task's control rate.

## 4. Robot client

The PiperX control client (camera capture, inverse kinematics, safety limits, websocket client)
will be added under [`deployment/`](deployment/). TODO.

## 5. Teleoperation and data collection

The teleoperation setup used to collect the real-robot demonstrations (Fruit25 was
collected with VR teleoperation) will be added under
[`teleoperation/`](teleoperation/). TODO.
