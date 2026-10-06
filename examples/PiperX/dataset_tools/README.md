# PiperX dataset tools

Scripts that turn PiperX teleoperation recordings into the EE-delta LeRobot datasets used for
training (Fruit25, PushBlock). They read and write LeRobot v3.0 directories directly with
pandas/pyarrow and `ffmpeg`; the `lerobot` package is not required. Raw recordings come from the
VR teleoperation setup in [`../teleoperation/`](../teleoperation/).

```
raw recording (30 Hz, joint space)            observation.state = 6 joints [deg] + gripper [mm]
   │                                          action = IK joint target (not used)
   │  convert_to_ee.py                        FK of the measured joints -> EE state, temporal-next-state action
   ▼
EE dataset (30 Hz, same episode ids)    ──►  check_quality.py --write-manifest  ──►  candidate manifest
   │                                                                                     │ review (inspect_episode.py),
   │  resample.py --fps N --exclusions manifest.json   ◄─────────────────────────────────┘ edit
   ▼
training dataset (8 Hz / 20 Hz, episodes renumbered from 0, meta/conversion_provenance.json)
   │
   └─ check_quality.py (final check)
```

## Data layout and action definition

- **State (8D):** `[x, y, z, qx, qy, qz, qw, gripper_mm]`, the forward kinematics of the
  *measured* joints (bundled URDF, [`../common/kinematics.py`](../common/kinematics.py)).
  Positions are in metres in the robot base frame.
- **Action (7D), temporal-next-state:**
  - translation = `position[t+1] - position[t]` (base frame, m)
  - rotation = `log(inverse(R[t]) * R[t+1])` as a rotation vector (end-effector frame, rad)
  - gripper = absolute gripper state at `t+1` (mm)
  - last frame of an episode: zero translation/rotation and the current gripper state.

`resample.py` interpolates the EE **state** trajectory at the output timestamps (linear for
position and gripper, slerp for rotation) and recomputes the actions between consecutive output
states; actions are never interpolated. Each output frame takes the video frame nearest in time.

## Tools

| Script | Purpose |
|---|---|
| [`convert_to_ee.py`](convert_to_ee.py) | Raw joint recording -> EE dataset at the recording rate. Keeps episode ids, reuses videos (`--video-mode hardlink\|copy\|symlink\|skip`), writes `meta/modality.json` (same layout as [`../prepare_modality.py`](../prepare_modality.py)). |
| [`resample.py`](resample.py) | EE dataset -> lower rate (`--fps`), with episode filtering; re-encodes the matching video frames (H.264). `--dry-run` prints the selection only; `--video-mode skip` writes data and metadata only. Writes `meta/conversion_provenance.json`. |
| [`check_quality.py`](check_quality.py) | Read-only per-episode checks (NaNs, quaternion norm, step/action outliers, actions that do not match the next state, video timestamps vs. file duration). Writes CSV reports to `--out-dir`; `--write-manifest` writes the flagged ids as an exclusion manifest. |
| [`inspect_episode.py`](inspect_episode.py) | Offline viewer: CSV of an episode plus wrist, top and action-replay videos side by side. Used to review flagged episodes. |
| [`filter_episodes.py`](filter_episodes.py) | Drop episodes from any v3.0 dataset (raw or EE) at its own rate, using the same manifest. Not needed for the released datasets, since `resample.py` filters while resampling. |

All scripts can be run from any directory; they find the repository root from their own path.

## Exclusion manifest

```json
{
  "description": "optional free text",
  "exclude_ranges": [[600, 700]],
  "exclude_episodes": [345, 396],
  "include_tasks": [],
  "reasons": {"345": "optional notes, ignored when filtering"}
}
```

- Ranges are **half-open** `[start, end)`: `[600, 700]` drops episodes 600 to 699. On the
  command line, `--exclude-range 600:700` means the same.
- Ids are the `episode_index` of the dataset the filter is applied to. `convert_to_ee.py`
  keeps ids, so ids from the raw recording and from its 30 Hz EE conversion are the same.
- `include_tasks` keeps only episodes whose task string matches exactly (empty = all tasks).
- `--exclusions`, `--exclude-range`, `--exclude-episode` and `--include-task` can be combined;
  the result is the union of all exclusions.

Run `check_quality.py` on an **unfiltered** dataset when you build a manifest. `resample.py`
renumbers the episodes it keeps, so the ids reported for a filtered dataset are output ids.

## Reproducing the released datasets

`$DATA` is the directory that holds the LeRobot datasets, e.g. `~/.cache/huggingface/lerobot/Dicomsky`.

**Fruit25** (`FruitV3`: 1,320 recorded episodes -> 1,186 episodes at 8 Hz):

```bash
python examples/PiperX/dataset_tools/convert_to_ee.py \
    --src $DATA/FruitV3 --dst $DATA/FruitV3_EE_temporal_30Hz_work
python examples/PiperX/dataset_tools/resample.py \
    --src $DATA/FruitV3_EE_temporal_30Hz_work --dst $DATA/FruitV3_EE_8Hz_temporal_clean_v2 --fps 8 \
    --exclusions examples/PiperX/dataset_tools/manifests/fruit25_v2_exclusions.json
```

[`manifests/fruit25_v2_exclusions.json`](manifests/fruit25_v2_exclusions.json) excludes source
episodes 600 to 699 plus 35 individual ids (one of them, 692, is also inside the range), so
1,320 - 100 - 34 = 1,186 episodes are kept.

**PushBlock** (`PushBlockBlueSquare`: 100 episodes, nothing excluded -> 20 Hz):

```bash
python examples/PiperX/dataset_tools/convert_to_ee.py \
    --src $DATA/PushBlockBlueSquare --dst $DATA/PushBlockBlueSquare_EE_temporal_30Hz_work
python examples/PiperX/dataset_tools/resample.py \
    --src $DATA/PushBlockBlueSquare_EE_temporal_30Hz_work \
    --dst $DATA/PushBlockBlueSquare_EE_20Hz_temporal_clean_v2 --fps 20
```

`--video-mode hardlink` (the default of `convert_to_ee.py`) needs the source and destination on
the same filesystem; otherwise use `copy` or `symlink`.

## Processing your own recordings

```bash
T=examples/PiperX/dataset_tools
python $T/convert_to_ee.py --src $DATA/MyTask --dst $DATA/MyTask_EE_30Hz
# Optional: an unfiltered low-rate copy without videos, so the default thresholds apply (tuned at 8 Hz).
python $T/resample.py --src $DATA/MyTask_EE_30Hz --dst /tmp/MyTask_EE_8Hz_qc --fps 8 --video-mode skip
python $T/check_quality.py --dataset /tmp/MyTask_EE_8Hz_qc --out-dir /tmp/MyTask_qc \
    --write-manifest my_task_exclusions.json
# Review the flagged ids (videos are in the 30 Hz dataset), then edit the manifest.
python $T/inspect_episode.py --dataset $DATA/MyTask_EE_30Hz --output-dir /tmp/MyTask_inspect --episode 12 40
python $T/resample.py --src $DATA/MyTask_EE_30Hz --dst $DATA/MyTask_EE_8Hz --fps 8 \
    --exclusions my_task_exclusions.json
python $T/check_quality.py --dataset $DATA/MyTask_EE_8Hz --out-dir /tmp/MyTask_qc_final
```

`--manifest-from review` puts the broader per-dataset outlier set into the manifest instead of
the strong outliers, and `strong+video` adds episodes with bad video metadata. Adjust the
thresholds with the `--strong-*` and `--action-*` flags (see `--help`). Before training, the
launchers run [`../prepare_modality.py`](../prepare_modality.py), which writes the same
`meta/modality.json` as `convert_to_ee.py`. Add the dataset to
`starVLA/dataloader/gr00t_lerobot/mixtures.py` to use it.

## Provenance

`resample.py` writes `meta/conversion_provenance.json`:

```json
{
  "source_dataset": "<--src as given>",
  "source_fps": 30.0,
  "output_fps": 8.0,
  "action_definition": "temporal-next-state",
  "translation_action": "position[t+1] - position[t]",
  "rotation_action": "log(inverse(rotation[t]) * rotation[t+1])",
  "gripper_action": "absolute gripper state[t+1]",
  "last_frame": "zero translation/rotation delta and current absolute gripper",
  "excluded_episode_ranges": [[600, 700]],
  "excluded_episode_ids": [345, 396, "..."],
  "source_episode_order": "ascending; output episode ids are contiguous from zero"
}
```

Ranges are half-open, as in the manifest. If `--include-task` was used, an extra
`included_tasks` list is written. Output episode `k` is the `k`-th kept source episode in
ascending order, so the source id of any output episode can be recovered from this file and the
source episode list.
