"""Helpers shared by the PiperX dataset tools.

- LeRobot v3.0 reading helpers and the per-feature statistics written to ``meta/stats.json``
  and ``meta/episodes``.
- The episode exclusion manifest read by ``resample.py`` / ``filter_episodes.py`` and written by
  ``check_quality.py``.

Exclusion manifest (JSON)::

    {
      "description": "free text, optional",
      "exclude_ranges": [[600, 700]],      # half-open [start, end): excludes 600..699
      "exclude_episodes": [345, 396],      # single episode ids
      "include_tasks": [],                 # exact task strings; empty = all tasks
      "reasons": {"345": "..."}            # optional notes, ignored when filtering
    }

Episode ids always refer to ``episode_index`` of the dataset the manifest is applied to.
``convert_to_ee.py`` keeps episode ids unchanged, so ids taken from a raw recording, from its
30 Hz EE conversion, or from an *unfiltered* resample of it are interchangeable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


MANIFEST_KEYS = {"description", "exclude_ranges", "exclude_episodes", "include_tasks", "reasons"}


# ---------------------------------------------------------------------------
# Episode exclusion manifest
# ---------------------------------------------------------------------------


@dataclass
class EpisodeFilter:
    exclude_ranges: list[tuple[int, int]] = field(default_factory=list)  # half-open [start, end)
    exclude_episodes: list[int] = field(default_factory=list)
    include_tasks: list[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return bool(self.exclude_ranges or self.exclude_episodes or self.include_tasks)

    def excluded_ids(self) -> set[int]:
        out = set(self.exclude_episodes)
        for start, end in self.exclude_ranges:
            out.update(range(start, end))
        return out


def _check_range(start: int, end: int, origin: str) -> tuple[int, int]:
    if end <= start:
        raise ValueError(f"Invalid exclude range {origin}: end must be > start (ranges are half-open [start, end)).")
    return int(start), int(end)


def parse_range_text(value: str) -> tuple[int, int]:
    """Parse a CLI range ``START:END`` (half-open, ``600:700`` excludes 600..699)."""
    start_text, end_text = value.split(":", 1)
    return _check_range(int(start_text), int(end_text), repr(value))


def load_manifest(path: Path) -> EpisodeFilter:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: manifest must be a JSON object.")
    unknown = sorted(set(data) - MANIFEST_KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown manifest keys {unknown}; allowed: {sorted(MANIFEST_KEYS)}")
    ranges = []
    for item in data.get("exclude_ranges", []):
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            raise ValueError(f"{path}: exclude_ranges entries must be [start, end], got {item!r}")
        ranges.append(_check_range(int(item[0]), int(item[1]), f"{item!r} in {path}"))
    return EpisodeFilter(
        exclude_ranges=ranges,
        exclude_episodes=[int(v) for v in data.get("exclude_episodes", [])],
        include_tasks=[str(v) for v in data.get("include_tasks", [])],
    )


def merge_filters(*filters: EpisodeFilter) -> EpisodeFilter:
    out = EpisodeFilter()
    for one in filters:
        for item in one.exclude_ranges:
            if item not in out.exclude_ranges:
                out.exclude_ranges.append(item)
        for ep in one.exclude_episodes:
            if ep not in out.exclude_episodes:
                out.exclude_episodes.append(ep)
        for task in one.include_tasks:
            if task not in out.include_tasks:
                out.include_tasks.append(task)
    return out


def add_filter_args(parser) -> None:
    parser.add_argument(
        "--exclusions",
        type=Path,
        help="JSON exclusion manifest (see dataset_tools/README.md). Merged with the flags below.",
    )
    parser.add_argument(
        "--exclude-episode",
        action="append",
        type=int,
        default=[],
        help="Source episode id to exclude. May be repeated.",
    )
    parser.add_argument(
        "--exclude-range",
        action="append",
        default=[],
        help="Half-open source episode range to exclude, START:END; 600:700 excludes 600..699. May be repeated.",
    )
    parser.add_argument(
        "--include-task",
        action="append",
        default=[],
        help="Exact task string to include. May be repeated; empty means all tasks.",
    )


def filter_from_args(args) -> EpisodeFilter:
    cli = EpisodeFilter(
        exclude_ranges=[parse_range_text(value) for value in args.exclude_range],
        exclude_episodes=list(args.exclude_episode),
        include_tasks=list(args.include_task),
    )
    if args.exclusions is None:
        return cli
    return merge_filters(load_manifest(args.exclusions), cli)


def task_text(value: object) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        return str(value[0]) if len(value) else ""
    return str(value)


def selected_episode_map(metadata: pd.DataFrame, episode_filter: EpisodeFilter) -> dict[int, int]:
    """Map kept source episode ids (ascending) to contiguous output ids starting at zero."""
    metadata = metadata.sort_values("episode_index")
    excluded = episode_filter.excluded_ids()
    include_tasks = set(episode_filter.include_tasks)
    selected: list[int] = []
    for _, row in metadata.iterrows():
        episode = int(row["episode_index"])
        if episode in excluded:
            continue
        if include_tasks and task_text(row["tasks"]) not in include_tasks:
            continue
        selected.append(episode)
    if not selected:
        raise ValueError("Episode filters selected no episodes.")
    return {source_episode: output_episode for output_episode, source_episode in enumerate(selected)}


def write_manifest(
    path: Path,
    exclude_episodes: list[int],
    description: str = "",
    reasons: dict[int, str] | None = None,
    exclude_ranges: list[tuple[int, int]] | None = None,
    include_tasks: list[str] | None = None,
) -> None:
    data: dict = {}
    if description:
        data["description"] = description
    data["exclude_ranges"] = [list(item) for item in (exclude_ranges or [])]
    data["exclude_episodes"] = sorted(int(v) for v in exclude_episodes)
    data["include_tasks"] = list(include_tasks or [])
    if reasons:
        data["reasons"] = {str(int(k)): str(v) for k, v in sorted(reasons.items())}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


# ---------------------------------------------------------------------------
# LeRobot v3.0 helpers
# ---------------------------------------------------------------------------


def stat(values: np.ndarray) -> dict[str, list]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q10": np.quantile(arr, 0.10, axis=0).tolist(),
        "q50": np.quantile(arr, 0.50, axis=0).tolist(),
        "q90": np.quantile(arr, 0.90, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }


def stats_for_df(df: pd.DataFrame) -> dict[str, dict[str, list]]:
    out: dict[str, dict[str, list]] = {}
    out["action"] = stat(np.stack(df["action"].to_numpy()))
    out["observation.state"] = stat(np.stack(df["observation.state"].to_numpy()))
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        out[key] = stat(df[key].to_numpy())
    return out


def update_flat_episode_stats(row: pd.Series, episode_stats: dict[str, dict[str, list]]) -> pd.Series:
    for key, one in episode_stats.items():
        for stat_name, value in one.items():
            col = f"stats/{key}/{stat_name}"
            if col in row.index:
                row[col] = value
    return row


def read_dataset_data(root: Path) -> pd.DataFrame:
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def read_episode_metadata(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))
    if not files:
        raise RuntimeError(f"No episode metadata found in {root}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def video_keys(info: dict) -> list[str]:
    return [
        key
        for key, feature in info.get("features", {}).items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]
