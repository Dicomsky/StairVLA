#!/usr/bin/env bash
# PiperX Fruit25: StairVLA stage 1: high-level GR00T-style policy (Qwen3-VL-2B, action horizon H=32).
# All hyperparameters live in the YAML; this script only sets locations and the GPU count.
# Global batch = 4 GPU(s) x 16 per GPU = 64.
set -euo pipefail

config_yaml=examples/PiperX/fruit25/stairvla_stage1.yaml
run_root_dir=./results/Checkpoints
run_id=fruit25_stairvla_stage1
num_gpus=4
dataset_dir=playground/Datasets/FruitV3_EE_8Hz_temporal_clean_v2

export PYTHONPATH=$(pwd):${PYTHONPATH:-}

if [[ ! -f "${dataset_dir}/meta/info.json" ]]; then
  echo "[error] Dataset not found at ${dataset_dir}. See examples/PiperX/README.md." >&2
  exit 1
fi
# Rewrite meta/modality.json into the per-scalar layout the PiperX data config expects.
python examples/PiperX/prepare_modality.py --dataset-dir "${dataset_dir}" --expected-fps 8

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
