#!/usr/bin/env bash
# LIBERO: StairVLA stage 1 (pi-base): high-level policy, Qwen3-VL-4B + layer-wise flow-matching head, action horizon H=32.
# All hyperparameters live in the YAML; this script only sets locations and the GPU count.
# Global batch = 8 GPUs x 4 per GPU = 32.
set -euo pipefail

config_yaml=examples/LIBERO/train_files/stairvla_pi_stage1.yaml
run_root_dir=./results/Checkpoints
run_id=libero_stairvla_pi_stage1
num_gpus=8

export PYTHONPATH=$(pwd):${PYTHONPATH:-}

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${config_yaml}" "${output_dir}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_gpus} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id}
