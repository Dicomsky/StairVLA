#!/bin/bash
# Start the StairVLA policy server for LIBERO evaluation (run from the repository root,
# in the starVLA environment). The simulator side is eval_files/eval_libero_all.sh.
set -euo pipefail
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

your_ckpt=${your_ckpt:-./results/Checkpoints/libero_stairvla_stage2/checkpoints/steps_30000_pytorch_model.pt}
gpu_id=${gpu_id:-0}
port=${port:-5679}

# Inference settings reported in the paper (Sec. 4.1): the high-level trajectory is handed
# over at denoising progress alpha=0.97 and reused for M refinement cycles of h actions.
hier_eval_mode=${hier_eval_mode:-default}            # default | top32 (high-level policy only, "Top-only" ablation)
hier_eval_num_chunks=${hier_eval_num_chunks:-4}      # M: refinement cycles per high-level update
denoise_step_scale=${denoise_step_scale:-0.97}       # alpha: high-level denoising progress at hand-over
context_denoise_step_scale=${context_denoise_step_scale:-default}
num_inference_timesteps=${num_inference_timesteps:-default}
lower_refine_steps=${lower_refine_steps:-default}
lower_assumed_step_scale=${lower_assumed_step_scale:-default}

extra_args=()
add_arg() { [ "$2" != "default" ] && extra_args+=("$1" "$2") || true; }
add_arg --hier_eval_num_chunks "${hier_eval_num_chunks}"
add_arg --denoise_step_scale "${denoise_step_scale}"
add_arg --context_denoise_step_scale "${context_denoise_step_scale}"
add_arg --num_inference_timesteps "${num_inference_timesteps}"
add_arg --lower_refine_steps "${lower_refine_steps}"
add_arg --lower_assumed_step_scale "${lower_assumed_step_scale}"

CUDA_VISIBLE_DEVICES=${gpu_id} python deployment/model_server/server_policy.py \
    --ckpt_path "${your_ckpt}" \
    --port "${port}" \
    --use_bf16 \
    --hier_eval_mode "${hier_eval_mode}" \
    "${extra_args[@]}"
