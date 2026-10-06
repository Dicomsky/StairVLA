"""Minimal writer for LeRobot v3.0 datasets (the format of the released PiperX recordings).

Produces the same layout, schemas and metadata as ``lerobot-record`` so the output can be
read by ``lerobot.datasets.LeRobotDataset`` and by ``../dataset_tools`` -- without depending
on the ``lerobot`` package (whose pins conflict with the StairVLA environment).

Each saved episode gets its own data parquet file and one mp4 per camera; the episode
metadata records the chunk/file indices, which v3.0 readers use to locate them.
Videos are encoded while recording, in background threads (AV1 by default, GOP 2).
"""

from __future__ import annotations

import json
import queue
import shutil
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CODEBASE_VERSION = "v3.0"
CHUNKS_SIZE = 1000
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
STAT_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
SCALAR_FEATURES = ("timestamp", "frame_index", "episode_index", "index", "task_index")

VIDEO_CODECS = {
    # name -> (PyAV encoder, info.json codec name, encoder options)
    "av1": ("libaom-av1", "av1", {"g": "2", "crf": "30", "cpu-used": "8", "usage": "realtime", "row-mt": "1"}),
    "h264": ("libx264", "h264", {"g": "2", "crf": "23", "preset": "veryfast"}),
}


# ---------------------------------------------------------------------------- stats
def _vector_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    stats = {
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "count": np.asarray([len(values)], dtype=np.int64),
    }
    for q in QUANTILES:
        stats[f"q{int(round(q * 100)):02d}"] = np.quantile(values, q, axis=0)
    return stats


def _image_stats(samples: np.ndarray) -> dict[str, np.ndarray]:
    """Per-channel stats in [0, 1] with shape (3, 1, 1), like LeRobot; ``samples`` is N x H x W x 3 uint8."""
    pixels = samples.reshape(-1, 3).astype(np.float64) / 255.0
    stats = _vector_stats(pixels)
    out = {k: v.reshape(3, 1, 1) for k, v in stats.items() if k != "count"}
    out["count"] = np.asarray([len(samples)], dtype=np.int64)
    return out


def aggregate_stats(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Combine two stat dicts the way LeRobot does (exact min/max/mean/std, count-weighted quantiles)."""
    na, nb = float(a["count"][0]), float(b["count"][0])
    n = na + nb
    mean = (a["mean"] * na + b["mean"] * nb) / n
    var = ((a["std"] ** 2 + (a["mean"] - mean) ** 2) * na + (b["std"] ** 2 + (b["mean"] - mean) ** 2) * nb) / n
    out = {
        "min": np.minimum(a["min"], b["min"]),
        "max": np.maximum(a["max"], b["max"]),
        "mean": mean,
        "std": np.sqrt(var),
        "count": np.asarray([int(n)], dtype=np.int64),
    }
    for key in ("q01", "q10", "q50", "q90", "q99"):
        out[key] = (a[key] * na + b[key] * nb) / n
    return out


def _to_json(stats: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, Any]]:
    return {ft: {k: np.asarray(v).tolist() for k, v in s.items()} for ft, s in stats.items()}


def _from_json(stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    return {ft: {k: np.asarray(v) for k, v in s.items()} for ft, s in stats.items()}


# ---------------------------------------------------------------------------- video
class _VideoEncoder:
    """Encodes RGB frames pushed from the recording loop in a background thread."""

    def __init__(self, path: Path, fps: int, width: int, height: int, vcodec: str):
        import av

        encoder, _, options = VIDEO_CODECS[vcodec]
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._container = av.open(str(path), "w")
        self._stream = self._container.add_stream(encoder, rate=fps)
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"
        self._stream.options = dict(options)
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name=f"encode-{path.parent.name}", daemon=True)
        self._thread.start()

    def push(self, frame: np.ndarray) -> None:
        if self._error is not None:
            raise RuntimeError(f"Video encoder for {self.path} failed: {self._error}")
        self._queue.put(frame)

    @property
    def backlog(self) -> int:
        return self._queue.qsize()

    def _run(self) -> None:
        import av

        try:
            while True:
                frame = self._queue.get()
                if frame is None:
                    break
                for packet in self._stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                    self._container.mux(packet)
            for packet in self._stream.encode():
                self._container.mux(packet)
        except BaseException as exc:  # noqa: BLE001 - reported by push()/close()
            self._error = exc
        finally:
            self._container.close()

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            raise RuntimeError(f"Video encoder for {self.path} failed: {self._error}") from self._error


# ---------------------------------------------------------------------------- writer
class LeRobotV3Writer:
    def __init__(
        self,
        root: Path,
        *,
        fps: int,
        robot_type: str,
        state_names: list[str],
        action_names: list[str],
        camera_names: list[str],
        image_height: int,
        image_width: int,
        vcodec: str = "av1",
        resume: bool = False,
        image_stats_stride: int = 4,
    ):
        if vcodec not in VIDEO_CODECS:
            raise ValueError(f"vcodec must be one of {list(VIDEO_CODECS)}")
        self.root = Path(root)
        self.fps = int(fps)
        self.camera_names = list(camera_names)
        self.video_keys = [f"observation.images.{name}" for name in self.camera_names]
        self.vcodec = vcodec
        self.height, self.width = int(image_height), int(image_width)
        self.image_stats_stride = max(1, int(image_stats_stride))
        info_path = self.root / "meta" / "info.json"

        if info_path.exists():
            if not resume:
                raise FileExistsError(f"{self.root} already contains a dataset; pass --resume to append to it.")
            self.info = json.loads(info_path.read_text())
            self._check_compatible(state_names, action_names)
            self.tasks = self._load_tasks()
            self.episodes = self._load_episodes()
            stats_path = self.root / "meta" / "stats.json"
            self.stats = _from_json(json.loads(stats_path.read_text())) if stats_path.exists() else {}
        else:
            self.info = self._new_info(robot_type, state_names, action_names)
            self.tasks: dict[str, int] = {}
            self.episodes = pd.DataFrame()
            self.stats: dict[str, dict[str, np.ndarray]] = {}
        self._episode: dict[str, Any] | None = None

    # ------------------------------------------------------------------ metadata
    def _new_info(self, robot_type: str, state_names: list[str], action_names: list[str]) -> dict[str, Any]:
        _, codec_name, _ = VIDEO_CODECS[self.vcodec]
        features: dict[str, Any] = {
            "action": {"dtype": "float32", "names": list(action_names), "shape": [len(action_names)]},
            "observation.state": {"dtype": "float32", "names": list(state_names), "shape": [len(state_names)]},
        }
        for key in self.video_keys:
            features[key] = {
                "dtype": "video",
                "shape": [self.height, self.width, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": self.height,
                    "video.width": self.width,
                    "video.codec": codec_name,
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": self.fps,
                    "video.channels": 3,
                    "has_audio": False,
                },
            }
        features["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
        for key in ("frame_index", "episode_index", "index", "task_index"):
            features[key] = {"dtype": "int64", "shape": [1], "names": None}
        return {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": robot_type,
            "total_episodes": 0,
            "total_frames": 0,
            "total_tasks": 0,
            "chunks_size": CHUNKS_SIZE,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 200,
            "fps": self.fps,
            "splits": {},
            "data_path": DATA_PATH,
            "video_path": VIDEO_PATH,
            "features": features,
        }

    def _check_compatible(self, state_names: list[str], action_names: list[str]) -> None:
        feats = self.info["features"]
        problems = []
        if int(self.info["fps"]) != self.fps:
            problems.append(f"fps {self.info['fps']} != {self.fps}")
        if feats["observation.state"]["names"] != list(state_names) or feats["action"]["names"] != list(action_names):
            problems.append("state/action names differ")
        existing_videos = [k for k, v in feats.items() if v["dtype"] == "video"]
        if sorted(existing_videos) != sorted(self.video_keys):
            problems.append(f"cameras {existing_videos} != {self.video_keys}")
        else:
            self.video_keys = existing_videos  # keep the existing order
        for key in self.video_keys:
            shape = feats[key]["shape"]
            if shape[:2] != [self.height, self.width]:
                problems.append(f"{key} shape {shape} != {[self.height, self.width, 3]}")
        if problems:
            raise ValueError(f"Cannot resume {self.root}: " + "; ".join(problems))

    def _load_tasks(self) -> dict[str, int]:
        path = self.root / "meta" / "tasks.parquet"
        if not path.exists():
            return {}
        df = pd.read_parquet(path)
        return {str(task): int(idx) for task, idx in zip(df.index, df["task_index"], strict=True)}

    def _load_episodes(self) -> pd.DataFrame:
        files = sorted((self.root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        if not files:
            return pd.DataFrame()
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    @property
    def num_episodes(self) -> int:
        return int(self.info["total_episodes"])

    @property
    def num_frames(self) -> int:
        return int(self.info["total_frames"])

    def _next_file_index(self, prefix: str) -> tuple[int, int]:
        if self.episodes.empty or f"{prefix}/chunk_index" not in self.episodes:
            return 0, 0
        pairs = self.episodes[[f"{prefix}/chunk_index", f"{prefix}/file_index"]].to_numpy()
        chunk, file = max((int(c), int(f)) for c, f in pairs)
        file += 1
        if file >= CHUNKS_SIZE:
            chunk, file = chunk + 1, 0
        return chunk, file

    # ------------------------------------------------------------------ episodes
    @property
    def recording(self) -> bool:
        return self._episode is not None

    @property
    def episode_length(self) -> int:
        return 0 if self._episode is None else len(self._episode["state"])

    @property
    def encoder_backlog(self) -> int:
        if self._episode is None:
            return 0
        return max(enc.backlog for enc in self._episode["encoders"].values())

    def start_episode(self, task: str) -> None:
        if self._episode is not None:
            raise RuntimeError("An episode is already being recorded.")
        tmp_dir = self.root / ".recording"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        self._episode = {
            "task": task,
            "tmp_dir": tmp_dir,
            "state": [],
            "action": [],
            "image_samples": {key: [] for key in self.video_keys},
            "encoders": {
                key: _VideoEncoder(tmp_dir / f"{key}.mp4", self.fps, self.width, self.height, self.vcodec)
                for key in self.video_keys
            },
        }

    def add_frame(self, state: np.ndarray, action: np.ndarray, images: dict[str, np.ndarray]) -> None:
        ep = self._episode
        if ep is None:
            raise RuntimeError("Call start_episode() first.")
        frame_index = len(ep["state"])
        for name, key in zip(self.camera_names, self.video_keys, strict=True):
            image = images[name]
            if image.shape != (self.height, self.width, 3):
                raise ValueError(f"Camera {name} returned {image.shape}, expected {(self.height, self.width, 3)}")
            ep["encoders"][key].push(image)
            if frame_index % self.image_stats_stride == 0:
                ep["image_samples"][key].append(image[::4, ::4].copy())
        ep["state"].append(np.asarray(state, dtype=np.float32))
        ep["action"].append(np.asarray(action, dtype=np.float32))

    def discard_episode(self) -> None:
        ep, self._episode = self._episode, None
        if ep is None:
            return
        for encoder in ep["encoders"].values():
            try:
                encoder.close()
            except RuntimeError:
                pass
        shutil.rmtree(ep["tmp_dir"], ignore_errors=True)

    def save_episode(self) -> int:
        """Finish encoding and write the episode; returns its episode index."""
        ep, self._episode = self._episode, None
        if ep is None:
            raise RuntimeError("No episode is being recorded.")
        length = len(ep["state"])
        for encoder in ep["encoders"].values():
            encoder.close()
        if length == 0:
            shutil.rmtree(ep["tmp_dir"], ignore_errors=True)
            raise ValueError("Refusing to save an empty episode.")

        episode_index = self.num_episodes
        global_from = self.num_frames
        task_index = self.tasks.setdefault(ep["task"], len(self.tasks))

        frame_index = np.arange(length, dtype=np.int64)
        timestamps = (frame_index / self.fps).astype(np.float32)
        state = np.stack(ep["state"])
        action = np.stack(ep["action"])
        columns = {
            "action": action,
            "observation.state": state,
            "timestamp": timestamps,
            "frame_index": frame_index,
            "episode_index": np.full(length, episode_index, dtype=np.int64),
            "index": np.arange(global_from, global_from + length, dtype=np.int64),
            "task_index": np.full(length, task_index, dtype=np.int64),
        }

        # data parquet
        data_chunk, data_file = self._next_file_index("data")
        data_path = self.root / DATA_PATH.format(chunk_index=data_chunk, file_index=data_file)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(self._data_table(columns), data_path)

        # videos
        row: dict[str, Any] = {
            "episode_index": episode_index,
            "tasks": [ep["task"]],
            "length": length,
            "data/chunk_index": data_chunk,
            "data/file_index": data_file,
            "dataset_from_index": global_from,
            "dataset_to_index": global_from + length,
        }
        for key in self.video_keys:
            chunk, file = self._next_file_index(f"videos/{key}")
            video_path = self.root / VIDEO_PATH.format(video_key=key, chunk_index=chunk, file_index=file)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(ep["tmp_dir"] / f"{key}.mp4"), video_path)
            row[f"videos/{key}/chunk_index"] = chunk
            row[f"videos/{key}/file_index"] = file
            row[f"videos/{key}/from_timestamp"] = 0.0
            row[f"videos/{key}/to_timestamp"] = length / self.fps
        shutil.rmtree(ep["tmp_dir"], ignore_errors=True)

        # episode stats, in the feature order of info.json
        ep_stats: dict[str, dict[str, np.ndarray]] = {}
        for key, feature in self.info["features"].items():
            if feature["dtype"] == "video":
                ep_stats[key] = _image_stats(np.stack(ep["image_samples"][key]))
            else:
                values = columns[key]
                ep_stats[key] = _vector_stats(values if values.ndim == 2 else values[:, None])
        for ft, stats in ep_stats.items():
            for stat in STAT_KEYS:
                row[f"stats/{ft}/{stat}"] = stats[stat].tolist()
        row["meta/episodes/chunk_index"], row["meta/episodes/file_index"] = self._episodes_meta_file()
        new_row = pd.DataFrame([row])
        self.episodes = new_row if self.episodes.empty else pd.concat([self.episodes, new_row], ignore_index=True)

        self.stats = {ft: aggregate_stats(self.stats[ft], s) if ft in self.stats else s for ft, s in ep_stats.items()}
        self.info["total_episodes"] = episode_index + 1
        self.info["total_frames"] = global_from + length
        self.info["total_tasks"] = len(self.tasks)
        self.info["splits"] = {"train": f"0:{episode_index + 1}"}
        self._write_meta()
        return episode_index

    def _data_table(self, columns: dict[str, np.ndarray]) -> pa.Table:
        arrays, fields, hf_features = [], [], {}
        for key in ("action", "observation.state"):
            values = columns[key]
            dim = values.shape[1]
            arrays.append(pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), pa.float32()), dim))
            fields.append(pa.field(key, pa.list_(pa.float32(), dim)))
            hf_features[key] = {"feature": {"dtype": "float32", "_type": "Value"}, "length": dim, "_type": "List"}
        for key in SCALAR_FEATURES:
            dtype = pa.float32() if key == "timestamp" else pa.int64()
            arrays.append(pa.array(columns[key], dtype))
            fields.append(pa.field(key, dtype))
            hf_features[key] = {"dtype": "float32" if key == "timestamp" else "int64", "_type": "Value"}
        metadata = {b"huggingface": json.dumps({"info": {"features": hf_features}}).encode()}
        return pa.Table.from_arrays(arrays, schema=pa.schema(fields, metadata=metadata))

    def _write_meta(self) -> None:
        meta = self.root / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        tasks = pd.DataFrame({"task_index": list(self.tasks.values())}, index=list(self.tasks.keys()))
        tasks.to_parquet(meta / "tasks.parquet")
        self._write_episodes_meta(meta)
        (meta / "stats.json").write_text(json.dumps(_to_json(self.stats), indent=4))
        (meta / "info.json").write_text(json.dumps(self.info, indent=4))

    def _episodes_meta_file(self) -> tuple[int, int]:
        """New episodes are appended to the last episodes-metadata file (files from lerobot-record stay untouched)."""
        if self.episodes.empty:
            return 0, 0
        pairs = self.episodes[["meta/episodes/chunk_index", "meta/episodes/file_index"]].astype(int).to_numpy()
        chunk, file = max((int(c), int(f)) for c, f in pairs)
        return chunk, file

    def _write_episodes_meta(self, meta: Path) -> None:
        chunk, file = self._episodes_meta_file()
        df = self.episodes
        part = df[(df["meta/episodes/chunk_index"].astype(int) == chunk) & (df["meta/episodes/file_index"].astype(int) == file)]
        path = meta / "episodes" / f"chunk-{chunk:03d}" / f"file-{file:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        part.reset_index(drop=True).to_parquet(path, index=False)

    def close(self) -> None:
        if self._episode is not None:
            self.discard_episode()
