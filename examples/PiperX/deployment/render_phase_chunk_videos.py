#!/usr/bin/env python3
"""Re-render phase-border videos with a synchronized chunk counter.

Reads ``top_original_30hz.mp4`` + ``top_video_frames.jsonl`` written by
``eval_benchmark.py --continuous-top-video`` and writes ``top_phase_chunks_30hz.mp4``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.PiperX.deployment.phase_video import draw_phase_overlay


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def attempt_directories(root: Path) -> list[Path]:
    if (root / "top_original_30hz.mp4").exists():
        return [root]
    return sorted(path.parent for path in root.rglob("top_original_30hz.mp4"))


def render_attempt(directory: Path, border_px: int, overwrite: bool) -> tuple[str, int]:
    source = directory / "top_original_30hz.mp4"
    frame_log = directory / "top_video_frames.jsonl"
    destination = directory / "top_phase_chunks_30hz.mp4"
    if destination.exists() and not overwrite:
        return "skipped", 0
    if not frame_log.exists():
        raise FileNotFoundError(frame_log)
    rows = read_jsonl(frame_log)
    capture = cv2.VideoCapture(str(source))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not capture.isOpened() or fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Could not read source video metadata: {source}")

    temporary = directory / ".top_phase_chunks_30hz.tmp.mp4"
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {temporary}")

    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index >= len(rows):
                raise RuntimeError(
                    f"Video contains more frames than its frame log: {source} ({frame_index + 1} > {len(rows)})"
                )
            writer.write(
                draw_phase_overlay(
                    frame,
                    rows[frame_index],
                    border_px,
                    show_chunk_count=True,
                )
            )
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    if frame_index != len(rows):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Frame count mismatch for {source}: video={frame_index}, log={len(rows)}"
        )
    temporary.replace(destination)
    return "rendered", frame_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Run, attempt, or common root directories.")
    parser.add_argument("--border-px", type=int, default=14)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.border_px < 1:
        raise ValueError("--border-px must be positive")

    attempts: list[Path] = []
    for root in args.paths:
        attempts.extend(attempt_directories(root.expanduser().resolve()))
    attempts = sorted(set(attempts))
    rendered = skipped = failed = 0
    for index, attempt in enumerate(attempts, start=1):
        try:
            status, frames = render_attempt(attempt, args.border_px, args.overwrite)
            rendered += status == "rendered"
            skipped += status == "skipped"
            print(f"[CHUNK_VIDEO {index}/{len(attempts)}] {status} frames={frames} {attempt}")
        except Exception as exc:
            failed += 1
            print(f"[CHUNK_VIDEO {index}/{len(attempts)}] FAILED {attempt}: {exc}")
    print(f"[CHUNK_VIDEO] rendered={rendered} skipped={skipped} failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
