#!/usr/bin/env python3
"""Continuous camera recording with an inference/execution phase overlay.

Used by ``eval_benchmark.py --continuous-top-video``: the camera's cached frames are
sampled in a background thread, independently of the control loop, and written as an
untouched video plus copies with a red (inference) / green (execution) border.
"""

from __future__ import annotations

import bisect
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


def displayed_chunk_count(phase_event: dict[str, Any]) -> int:
    """Return a monotonic count of the chunk currently reached by execution."""
    chunk_id = phase_event.get("chunk_id")
    if chunk_id is None:
        return 0
    # chunk_id is zero-based. During inference for chunk k, chunks [0, k)
    # have already run; once execution starts, chunk k is the current chunk.
    return max(0, int(chunk_id) + (str(phase_event.get("phase")) == "execute"))


def draw_phase_overlay(
    bgr: np.ndarray,
    phase_event: dict[str, Any],
    border_px: int,
    *,
    show_chunk_count: bool = False,
) -> np.ndarray:
    """Draw the phase border and optional chunk counter on a BGR frame."""
    import cv2

    overlay = bgr.copy()
    phase = str(phase_event["phase"])
    # OpenCV uses BGR: red marks inference, green marks execution.
    color = (0, 0, 255) if phase == "inference" else (0, 255, 0)
    height, width = overlay.shape[:2]
    thickness = max(1, min(int(border_px), min(height, width) // 8))
    cv2.rectangle(overlay, (0, 0), (width - 1, height - 1), color, thickness)
    if not show_chunk_count:
        return overlay

    label = f"Chunk {displayed_chunk_count(phase_event)}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.55, min(0.85, min(height, width) / 700.0))
    text_thickness = max(1, int(round(font_scale * 2.0)))
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, text_thickness
    )
    padding = max(6, int(round(8 * font_scale)))
    right = width - thickness - padding
    bottom = height - thickness - padding
    left = right - text_width - 2 * padding
    top = bottom - text_height - baseline - 2 * padding
    cv2.rectangle(overlay, (left, top), (right, bottom), (20, 20, 20), -1)
    cv2.putText(
        overlay,
        label,
        (left + padding, bottom - padding - baseline),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )
    return overlay


class PhaseTimeline:
    """Thread-safe monotonic timeline used to label asynchronously captured frames."""

    def __init__(self, trial_start: float, initial_phase: str = "execute"):
        self.trial_start = float(trial_start)
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []
        self.transition(initial_phase, reason="trial_start", timestamp=self.trial_start)

    def transition(self, phase: str, *, timestamp: float | None = None, **details: Any) -> None:
        if phase not in {"inference", "execute"}:
            raise ValueError(f"Unsupported phase: {phase}")
        event_t = time.perf_counter() if timestamp is None else float(timestamp)
        event = {
            "event_index": 0,
            "phase": phase,
            "monotonic_s": event_t,
            "trial_elapsed_s": event_t - self.trial_start,
            "wall_time_ns": time.time_ns(),
            **details,
        }
        with self._lock:
            event["event_index"] = len(self._events)
            self._events.append(event)

    def phase_at(self, timestamp: float) -> dict[str, Any]:
        with self._lock:
            times = [float(event["monotonic_s"]) for event in self._events]
            index = max(0, bisect.bisect_right(times, float(timestamp)) - 1)
            return dict(self._events[index])

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(event) for event in self._events]


class ContinuousPhaseVideoRecorder:
    """Record every new cached camera frame independently of the control loop."""

    def __init__(
        self,
        directory: Path,
        camera: Any,
        camera_name: str,
        timeline: PhaseTimeline,
        trial_start: float,
        fps: float = 30.0,
        border_px: int = 14,
    ):
        self.directory = directory
        self.camera = camera
        self.camera_name = camera_name
        self.timeline = timeline
        self.trial_start = float(trial_start)
        self.fps = float(fps)
        self.border_px = int(border_px)
        self.rows: list[dict[str, Any]] = []
        self.capture_rows: list[dict[str, Any]] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._original_writer = None
        self._overlay_writer = None
        self._chunk_overlay_writer = None
        self._previous_rgb: np.ndarray | None = None
        self._previous_capture_t: float | None = None
        self._previous_capture_index: int | None = None
        self._next_video_t: float | None = None

    def start(self) -> None:
        if not hasattr(self.camera, "read_latest"):
            raise RuntimeError(
                f"Camera {self.camera_name!r} does not expose the asynchronous frame cache required "
                "for continuous recording"
            )
        self._thread = threading.Thread(
            target=self._record_loop,
            name=f"{self.camera_name}-phase-video-recorder",
            daemon=True,
        )
        self._thread.start()

    def _cached_rgb_frame(self, previous_timestamp: float | None) -> tuple[float | None, np.ndarray | None]:
        timestamp, frame = self.camera.read_latest(previous_timestamp)
        if timestamp is None or frame is None:
            return timestamp, None
        return float(timestamp), np.asarray(frame)

    def _open_writers(self, rgb: np.ndarray) -> None:
        import cv2

        height, width = rgb.shape[:2]
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        original_path = self.directory / f"{self.camera_name}_original_30hz.mp4"
        overlay_path = self.directory / f"{self.camera_name}_phase_30hz.mp4"
        chunk_overlay_path = self.directory / f"{self.camera_name}_phase_chunks_30hz.mp4"
        self._original_writer = cv2.VideoWriter(str(original_path), codec, self.fps, (width, height))
        self._overlay_writer = cv2.VideoWriter(str(overlay_path), codec, self.fps, (width, height))
        self._chunk_overlay_writer = cv2.VideoWriter(
            str(chunk_overlay_path), codec, self.fps, (width, height)
        )
        if (
            not self._original_writer.isOpened()
            or not self._overlay_writer.isOpened()
            or not self._chunk_overlay_writer.isOpened()
        ):
            raise RuntimeError(f"Could not open 30 Hz video writers in {self.directory}")

    def _record_loop(self) -> None:
        previous_timestamp: float | None = None
        poll_s = min(0.004, 1.0 / max(self.fps * 4.0, 1.0))
        try:
            while not self._stop.is_set():
                capture_t, rgb = self._cached_rgb_frame(previous_timestamp)
                if rgb is None:
                    self._stop.wait(poll_s)
                    continue
                previous_timestamp = capture_t
                if capture_t is None or capture_t < self.trial_start:
                    continue
                if rgb.ndim != 3 or rgb.shape[2] != 3:
                    raise ValueError(f"Camera {self.camera_name!r} returned invalid RGB shape {rgb.shape}")
                if self._original_writer is None:
                    self._open_writers(rgb)
                capture_index = len(self.capture_rows)
                self.capture_rows.append(
                    {
                        "capture_index": capture_index,
                        "camera_capture_monotonic_s": capture_t,
                        "trial_elapsed_s": capture_t - self.trial_start,
                        "wall_time_ns": time.time_ns(),
                    }
                )
                if self._previous_rgb is None:
                    self._next_video_t = capture_t
                else:
                    self._write_video_slots_until(capture_t)
                self._previous_rgb = rgb
                self._previous_capture_t = capture_t
                self._previous_capture_index = capture_index
        except Exception as exc:
            self.error = repr(exc)

    def _write_video_slots_until(self, end_t: float) -> None:
        import cv2

        if (
            self._previous_rgb is None
            or self._previous_capture_t is None
            or self._previous_capture_index is None
            or self._next_video_t is None
        ):
            return
        period = 1.0 / self.fps
        while self._next_video_t < end_t:
            phase_event = self.timeline.phase_at(self._next_video_t)
            phase = str(phase_event["phase"])
            bgr = cv2.cvtColor(self._previous_rgb, cv2.COLOR_RGB2BGR)
            overlay = draw_phase_overlay(bgr, phase_event, self.border_px)
            chunk_overlay = draw_phase_overlay(
                bgr, phase_event, self.border_px, show_chunk_count=True
            )
            self._original_writer.write(bgr)
            self._overlay_writer.write(overlay)
            self._chunk_overlay_writer.write(chunk_overlay)

            record_t = time.perf_counter()
            previous_row = self.rows[-1] if self.rows else None
            self.rows.append(
                {
                    "frame_index": len(self.rows),
                    "camera": self.camera_name,
                    "phase": phase,
                    "phase_event_index": phase_event["event_index"],
                    "chunk_id": phase_event.get("chunk_id"),
                    "request_id": phase_event.get("request_id"),
                    "video_monotonic_s": self._next_video_t,
                    "trial_elapsed_s": self._next_video_t - self.trial_start,
                    "source_capture_index": self._previous_capture_index,
                    "source_camera_capture_monotonic_s": self._previous_capture_t,
                    "source_frame_age_ms": (self._next_video_t - self._previous_capture_t) * 1000.0,
                    "source_frame_repeated": bool(
                        previous_row is not None
                        and previous_row["source_capture_index"] == self._previous_capture_index
                    ),
                    "record_monotonic_s": record_t,
                    "wall_time_ns": time.time_ns(),
                }
            )
            self._next_video_t += period

    def stop_and_write(self) -> dict[str, Any]:
        stop_t = time.perf_counter()
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                self.error = self.error or "continuous video recorder did not stop within 3 seconds"
        # Fill the fixed-rate output timeline through the end of the trial.
        # Repeating the latest source frame preserves real elapsed time when
        # the physical camera delivers fewer than the requested 30 fps.
        self._write_video_slots_until(stop_t)
        if self._original_writer is not None:
            self._original_writer.release()
        if self._overlay_writer is not None:
            self._overlay_writer.release()
        if self._chunk_overlay_writer is not None:
            self._chunk_overlay_writer.release()

        frame_path = self.directory / f"{self.camera_name}_video_frames.jsonl"
        with frame_path.open("w") as stream:
            for row in self.rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        transition_path = self.directory / "phase_transitions.jsonl"
        with transition_path.open("w") as stream:
            for event in self.timeline.events():
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        capture_path = self.directory / f"{self.camera_name}_camera_captures.jsonl"
        with capture_path.open("w") as stream:
            for row in self.capture_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

        capture_elapsed = (
            self.capture_rows[-1]["trial_elapsed_s"] - self.capture_rows[0]["trial_elapsed_s"]
            if len(self.capture_rows) > 1
            else 0.0
        )
        capture_hz = (
            (len(self.capture_rows) - 1) / capture_elapsed if capture_elapsed > 0 else None
        )
        inference_frames = sum(row["phase"] == "inference" for row in self.rows)
        repeated_frames = sum(bool(row["source_frame_repeated"]) for row in self.rows)
        return {
            "continuous_video_camera": self.camera_name,
            "continuous_video_nominal_hz": self.fps,
            "continuous_video_effective_hz": self.fps if self.rows else None,
            "continuous_video_frames": len(self.rows),
            "continuous_video_source_captures": len(self.capture_rows),
            "continuous_video_source_capture_hz": capture_hz,
            "continuous_video_repeated_frames": repeated_frames,
            "continuous_video_inference_frames": inference_frames,
            "continuous_video_execute_frames": len(self.rows) - inference_frames,
            "continuous_video_original": (
                f"{self.camera_name}_original_30hz.mp4" if self._original_writer is not None else None
            ),
            "continuous_video_overlay": (
                f"{self.camera_name}_phase_30hz.mp4" if self._overlay_writer is not None else None
            ),
            "continuous_video_chunk_overlay": (
                f"{self.camera_name}_phase_chunks_30hz.mp4"
                if self._chunk_overlay_writer is not None
                else None
            ),
            "continuous_video_frame_log": frame_path.name,
            "continuous_video_capture_log": capture_path.name,
            "continuous_video_phase_log": transition_path.name,
            "continuous_video_error": self.error,
        }
