#!/usr/bin/env bash
# LIBERO: Ablation: StairVLA stage 2 without temporal action context ("Ours w/o Action Context").
# Requires the stage-1 checkpoint (trainer.pretrained_checkpoint in the YAML); the VLM and
# high-level action head are frozen.
# All hyperparameters live in the YAML; this script only sets locations and the GPU count.
# Global batch = 8 GPUs x 16 per GPU = 128.
set -euo pipefail

config_yaml=examples/LIBERO/train_files/stairvla_stage2_no_context.yaml
run_root_dir=./results/Checkpoints
run_id=libero_stairvla_stage2_no_context
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
