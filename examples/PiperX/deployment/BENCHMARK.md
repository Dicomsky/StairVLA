# PiperX formal benchmark

`eval_benchmark.py` runs a deterministic task schedule and records only
accepted success/failure trials. The built-in Fruit25 protocol contains 25
tasks and defaults to 10 accepted trials per task (250 total). Its defaults
force the paper settings: delta-EE actions, 8D EE state, q99 normalization,
8 Hz control, per-step clips of 0.05 m / 0.20 rad, 25 deg/s joint speed limit,
binary gripper (threshold 80.6 mm), 50 s per trial.

```bash
python examples/PiperX/deployment/eval_benchmark.py \
    --checkpoint results/Checkpoints/fruit25_stairvla_stage2 \
    --run-name fruit25_stairvla --checkpoint-id fruit25_stairvla_stage2 \
    --port 10093 --execute
```

For PushBlock, pass the single instruction through `--tasks-json` and the
PushBlock control settings:

```bash
echo '["Push the black block into the blue square target at the center."]' > pushblock_tasks.json
python examples/PiperX/deployment/eval_benchmark.py \
    --checkpoint results/Checkpoints/pushblock_stairvla_stage2 \
    --run-name pushblock_stairvla --tasks-json pushblock_tasks.json \
    --control-hz 20 --gripper-action-mode absolute \
    --port 10093 --execute
```

## Trial flow

1. The script selects the instruction from the fixed schedule and resets the
   policy cache.
2. PiperX moves slowly to home.
3. The operator arranges the scene and presses Enter to start.
4. Camera queues are drained for `--camera-warmup-s` (0.5 seconds by
   default). These observations are neither recorded nor sent to the policy.
5. The policy runs until the timeout or the operator presses Enter.
6. The operator labels the attempt:
   - `s`: success; accept and advance.
   - `f`: failure; accept and advance.
   - `r`: discard this recording, then repeat the same scheduled trial.
   - `q`: discard this recording and stop the session.

`--start-trial` is one-based. A resumed run must use the same `--run-name`, add
`--resume-benchmark`, and start at the first trial that has not been accepted.

## Output layout

The default root is `outputs/piperx_benchmark/<run-name>/` (relative to the
current directory; change it with `--output-root`):

- `metadata.json`: immutable protocol, task list, CLI settings, checkpoint ID,
  git revision, and frame conventions.
- `sessions.jsonl`: start/end information for each interrupted or resumed run.
- `manifest.jsonl`: one result row per accepted success/failure trial.
- `summary.json`: overall and per-task success rates, updated after every accepted trial.
- `tasks.json`: ordered task list.
- `attempts/trial_N_attempt_M_TIMESTAMP/`:
  - `top.mp4`, `wrist.mp4`: observations aligned by frame index
    (`--no-record-video` disables them).
  - `frames.jsonl`, `frames.parquet`: feedback state, EE state, action, IK,
    command, timing, clipping, and inference/execution phase for every frame.
  - `state_feedback_high_rate.jsonl`, `state_feedback_high_rate.parquet`:
    actual joint, gripper, motor speed/current/effort, and FK-derived EE
    feedback sampled independently at 50 Hz by default. SDK timestamps and
    update flags expose repeated cached reads. Use `--state-log-hz 0` to
    disable or another positive value to change the requested rate.
  - `chunks.jsonl`: normalized and physical action chunks, normalized policy
    state, request ID, and inference latency.
  - `result.json`: final label and aggregate timing/safety counters.

Videos contain one observation image per control step at nominal `control_hz`.
Use `trial_elapsed_s` and `wall_time` in `frames.*` for real timing because a
blocking inference can make the wall-clock interval longer than one video frame.

With `--continuous-top-video`, the top camera is additionally recorded at its
native 30 Hz independently of the control loop (`top_original_30hz.mp4`), plus
copies with a red (inference) / green (execution) border
(`top_phase_30hz.mp4`) and a chunk counter (`top_phase_chunks_30hz.mp4`);
per-frame timing and phase are in `top_video_frames.jsonl`.
`render_phase_chunk_videos.py <run dir>` re-renders the chunk-counter videos.

Use `--plan-only` to inspect all task/trial mappings without connecting the
model server, cameras, or robot. `--low-data-episodes-per-task N` changes the
count for the 16 low-data compositional tasks (zero-based indices 7-22) only.

`analyze_smoothness.py --run LABEL=<run dir> [--run ...]` computes per-episode
trajectory smoothness (SPARC, log dimensionless jerk, high-frequency power,
movement units) from the high-rate feedback of accepted trials and writes a CSV
and a summary JSON.
