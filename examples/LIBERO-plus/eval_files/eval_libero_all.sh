#!/bin/bash
set -euo pipefail

# Run all four standard LIBERO-plus suites sequentially and print
# per-suite/global accuracy. Assumes the policy server is already running.

export LIBERO_HOME=${LIBERO_HOME:?"set LIBERO_HOME to your LIBERO-plus checkout"}
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}
export LIBERO_Python=${LIBERO_Python:-python}  # python of the LIBERO-plus environment
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}

export PYTHONPATH=${PYTHONPATH:-}:${LIBERO_HOME}
export PYTHONPATH=$(pwd):${PYTHONPATH}
export ROBOSUITE_LOGFILE=${ROBOSUITE_LOGFILE:-../robosuite_plus.log}

host=${host:-127.0.0.1}
base_port=${base_port:-9883}
eval_tag=${eval_tag:-default}
action_chunk_size_override=${action_chunk_size_override:-}
your_ckpt=${your_ckpt:-./results/Checkpoints/libero_plus_stairvla_stage2/checkpoints/steps_50000_pytorch_model.pt}
output_dir=${output_dir:-results/LIBERO-plus}
num_trials_per_task=${num_trials_per_task:-1}
num_steps_wait=${num_steps_wait:-50}

# Four common suites. Override with:
# suites="libero_spatial libero_object" bash examples/LIBERO-plus/eval_files/eval_libero_all.sh
suites=${suites:-"libero_10 libero_goal libero_spatial libero_object"}
# suites=${suites:-"libero_object"}

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
folder_name="${folder_name}_${eval_tag}"
if [ -n "${action_chunk_size_override}" ]; then
    folder_name="${folder_name}_chunk${action_chunk_size_override}"
fi

LOG_DIR="logs/eval_plus_all_$(date +"%Y%m%d_%H%M%S")"
mkdir -p "${LOG_DIR}" "${output_dir}"

GLOBAL_SUCCESS=0
GLOBAL_EPISODES=0

if [ -t 1 ]; then
    BOLD=$(printf '\033[1m')
    GREEN=$(printf '\033[32m')
    YELLOW=$(printf '\033[33m')
    CYAN=$(printf '\033[36m')
    RESET=$(printf '\033[0m')
else
    BOLD=""
    GREEN=""
    YELLOW=""
    CYAN=""
    RESET=""
fi

echo "============================================================"
echo "Checkpoint: ${your_ckpt}"
echo "Suites: ${suites}"
echo "Trials per task: ${num_trials_per_task}"
echo "Wait steps: ${num_steps_wait}"
echo "Policy server: ${host}:${base_port}"
echo "Logs: ${LOG_DIR}"
echo "Videos: ${output_dir}"
echo "============================================================"

for task_suite_name in ${suites}; do
    echo
    echo "================ Evaluating ${task_suite_name} ================"
    video_out_path="${output_dir}/${task_suite_name}/${folder_name}"
    summary_json="${LOG_DIR}/${task_suite_name}_summary.json"
    suite_log="${LOG_DIR}/${task_suite_name}.log"

    FORCE_COLOR=1 PYTHONUNBUFFERED=1 TERM="${TERM:-xterm-256color}" \
    "${LIBERO_Python}" ./examples/LIBERO-plus/eval_files/eval_libero.py \
        --args.pretrained-path "${your_ckpt}" \
        --args.host "${host}" \
        --args.port "${base_port}" \
        --args.task-suite-name "${task_suite_name}" \
        --args.num-trials-per-task "${num_trials_per_task}" \
        --args.num-steps-wait "${num_steps_wait}" \
        --args.video-out-path "${video_out_path}" \
        --args.log-path "${LOG_DIR}" \
        --args.summary-json-path "${summary_json}" \
        --args.global-success-offset "${GLOBAL_SUCCESS}" \
        --args.global-episode-offset "${GLOBAL_EPISODES}" \
        ${action_chunk_size_override:+--args.action-chunk-size-override "${action_chunk_size_override}"} \
        2>&1 | tee "${suite_log}"

    read SUITE_SUCCESS SUITE_EPISODES SUITE_RATE < <("${LIBERO_Python}" -c "import json; s=json.load(open('${summary_json}')); print(s['total_successes'], s['total_episodes'], s['success_rate'])")
    GLOBAL_SUCCESS=$((GLOBAL_SUCCESS + SUITE_SUCCESS))
    GLOBAL_EPISODES=$((GLOBAL_EPISODES + SUITE_EPISODES))
    GLOBAL_RATE=$("${LIBERO_Python}" -c "print(${GLOBAL_SUCCESS} / ${GLOBAL_EPISODES} if ${GLOBAL_EPISODES} else 0.0)")

    echo "${BOLD}${CYAN}---- Suite done: ${task_suite_name} ----${RESET}"
    printf '%sSuite accuracy:%s  %s%.4f%s (%d/%d)\n' "${BOLD}" "${RESET}" "${GREEN}" "${SUITE_RATE}" "${RESET}" "${SUITE_SUCCESS}" "${SUITE_EPISODES}"
    printf '%sGlobal accuracy:%s %s%.4f%s (%d/%d)\n' "${BOLD}" "${RESET}" "${YELLOW}" "${GLOBAL_RATE}" "${RESET}" "${GLOBAL_SUCCESS}" "${GLOBAL_EPISODES}"
done

FINAL_RATE=$("${LIBERO_Python}" -c "print(${GLOBAL_SUCCESS} / ${GLOBAL_EPISODES} if ${GLOBAL_EPISODES} else 0.0)")
OVERALL_SUMMARY="${LOG_DIR}/overall_summary.json" "${LIBERO_Python}" -c "import json, os; summary={'checkpoint': '${your_ckpt}', 'suites': '${suites}', 'num_trials_per_task': ${num_trials_per_task}, 'total_successes': ${GLOBAL_SUCCESS}, 'total_episodes': ${GLOBAL_EPISODES}, 'success_rate': ${FINAL_RATE}}; open(os.environ['OVERALL_SUMMARY'], 'w').write(json.dumps(summary, indent=2))"

echo
echo "${BOLD}${CYAN}================ Final LIBERO-plus-4 Summary ================${RESET}"
printf '%sFinal accuracy:%s %s%.4f%s (%d/%d)\n' "${BOLD}" "${RESET}" "${GREEN}" "${FINAL_RATE}" "${RESET}" "${GLOBAL_SUCCESS}" "${GLOBAL_EPISODES}"
echo "Summary saved to: ${LOG_DIR}/overall_summary.json"
