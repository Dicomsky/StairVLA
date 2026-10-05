# LIBERO

Training and evaluation of StairVLA on the four LIBERO suites (Spatial, Object, Goal, Long). One model
is trained jointly on all four suites; every task is evaluated with 50 rollouts.

## 1. Data

Download the four LeRobot-format LIBERO datasets (no-noops versions) and place `modality.json` in
each dataset's `meta/` folder:

- [libero_spatial_no_noops_1.0.0_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot)
- [libero_object_no_noops_1.0.0_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot)
- [libero_goal_no_noops_1.0.0_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot)
- [libero_10_no_noops_1.0.0_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot)

The StarVLA script below does all of this and links the result to
`playground/Datasets/LEROBOT_LIBERO_DATA`:

```bash
export DEST=/path/to/your/data/directory
bash examples/LIBERO/data_preparation.sh
```

## 2. Training

| Launcher | What it trains | Global batch | Steps |
|---|---|:---:|:---:|
| [`run_stairvla_stage1.sh`](train_files/run_stairvla_stage1.sh) | High-level policy, GR00T-style (Qwen3-VL-4B, H=32) | 128 | 30k |
| [`run_stairvla_stage2.sh`](train_files/run_stairvla_stage2.sh) | Refiner (h=5, k=5), on top of the frozen stage-1 policy | 128 | 30k |
| [`run_stairvla_stage2_no_context.sh`](train_files/run_stairvla_stage2_no_context.sh) | Ablation: refiner without temporal action context | 128 | 30k |
| [`run_stairvla_pi_stage1.sh`](train_files/run_stairvla_pi_stage1.sh) | High-level policy, π-style (Qwen3-VL-4B, H=32) | see YAML | 30k |
| TODO | Refiner for the π-base model | | |

All launchers use 8 GPUs and are run from the repository root. Stage 2 loads
`./results/Checkpoints/libero_stairvla_stage1/checkpoints/steps_30000_pytorch_model.pt`; change
`trainer.pretrained_checkpoint` in the stage-2 YAML if your stage-1 run lives elsewhere.

## 3. Evaluation

Set up the LIBERO simulator by following the [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO).
Python 3.10 avoids many issues. Then, inside the LIBERO environment:

```bash
pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4
```

Run the following two commands from the repository root, each in its own terminal.

**Terminal 1, StairVLA environment: policy server.**

```bash
your_ckpt=./results/Checkpoints/libero_stairvla_stage2/checkpoints/steps_30000_pytorch_model.pt \
bash examples/LIBERO/eval_files/run_policy_server.sh
```

**Terminal 2, LIBERO environment: all four suites, 50 trials per task.**

```bash
export LIBERO_HOME=/path/to/LIBERO
your_ckpt=./results/Checkpoints/libero_stairvla_stage2/checkpoints/steps_30000_pytorch_model.pt \
bash examples/LIBERO/eval_files/eval_libero_all.sh
```

`eval_libero_all.sh` reads the checkpoint path only to load the action normalization statistics,
so pass the same `your_ckpt` to both scripts. It prints per-suite and overall success rates and
writes `overall_summary.json` under `logs/`.

### Server options

`run_policy_server.sh` reads these environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `hier_eval_num_chunks` | 4 | M: refinement cycles per high-level trajectory |
| `denoise_step_scale` | 0.97 | α: denoising progress at which the high-level trajectory is handed to the refiner |
| `hier_eval_mode` | `default` | `top32` runs the high-level policy alone (the "Top-only" ablation) |
| `context_denoise_step_scale`, `num_inference_timesteps`, `lower_refine_steps`, `lower_assumed_step_scale` | from checkpoint | further inference overrides; see `deployment/model_server/server_policy.py` |

To measure per-chunk latency without the simulator:

```bash
python scripts/benchmark_latency.py --ckpt <stage-2 checkpoint> --eval-num-chunks 4 --denoise-step-scale 0.97
```
