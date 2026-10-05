# LIBERO-Plus

[LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) adds seven kinds of perturbation (camera,
robot, language, light, background, noise, layout) to the LIBERO tasks. We report two settings:

- **Zero-shot:** the LIBERO-trained StairVLA model (see [../LIBERO](../LIBERO/README.md)) is evaluated
  directly in the perturbed environments.
- **Fine-tuned:** StairVLA is trained on the LIBERO-Plus training set with the launchers below.

| Setting | Camera | Robot | Language | Light | Background | Noise | Layout | Avg. |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Zero-shot | 43.7 | 54.5 | 82.5 | 93.4 | 90.5 | 67.8 | 74.0 | 70.4 |
| Fine-tuned | 95.5 | 48.4 | 83.7 | 96.8 | 94.9 | 95.4 | 77.4 | 83.7 |

## 1. Data

Put the LeRobot-format LIBERO-Plus training data under `playground/Datasets/LEROBOT_LIBERO_PLUS_DATA`
with one folder per suite (`libero_plus_spatial`, `libero_plus_object`, `libero_plus_goal`,
`libero_plus_10`), and copy [`train_files/modality.json`](train_files/modality.json) into each
folder's `meta/`. The `libero_plus_all` mixture in
`starVLA/dataloader/gr00t_lerobot/mixtures.py` combines the four suites.

## 2. Training

| Launcher | What it trains | Global batch | Steps |
|---|---|:---:|:---:|
| [`run_stairvla_stage1.sh`](train_files/run_stairvla_stage1.sh) | High-level policy (Qwen3-VL-4B, H=32) | 128 | 50k |
| [`run_stairvla_stage2.sh`](train_files/run_stairvla_stage2.sh) | Refiner on top of the frozen stage-1 policy | 128 | 50k |

## 3. Evaluation

Install LIBERO-Plus by following its [repository](https://github.com/sylvestf/LIBERO-plus), then
inside that environment:

```bash
pip install -r examples/LIBERO-plus/eval_files/libero_plus_requirements.txt
```

Run the following two commands from the repository root, each in its own terminal.

**Terminal 1, StairVLA environment: policy server.**

```bash
your_ckpt=<stage-2 checkpoint> bash examples/LIBERO-plus/eval_files/run_policy_server.sh
```

**Terminal 2, LIBERO-Plus environment: all four suites.**

```bash
export LIBERO_HOME=/path/to/LIBERO-plus
your_ckpt=<stage-2 checkpoint> bash examples/LIBERO-plus/eval_files/eval_libero_all.sh
```

For zero-shot evaluation, pass the LIBERO stage-2 checkpoint instead. The server takes the same
options as for [LIBERO](../LIBERO/README.md#server-options).

LIBERO-Plus has more than 10,000 task variants, so a full evaluation takes a long time; running
several server/simulator pairs in parallel (different `port`/`base_port` and `suites`) helps.
`LOG_DIR=<log folder> python examples/LIBERO-plus/eval_files/aggregate_results.py` merges the
per-suite, per-category results into `overall_results.json`.
