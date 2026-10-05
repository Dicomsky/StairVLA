#!/usr/bin/env python3
"""Benchmark HierarchicalVLA eval latency over top/lower denoise schedules.

This script mirrors the knobs exposed by
examples/LIBERO/eval_files/run_policy_server.sh, but runs locally without a
websocket server. It is meant for quick latency sweeps such as:

  CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_latency.py \
    --top-steps 1,2,4 --eval-num-chunks 1,2,3,4,5 \
    --top-only-horizons 10,15,20,25,32 \
    --lower-refine-steps 1 \
    --denoise-step-scale 0.97 --context-denoise-step-scale 0.9 \
    --lower-assumed-step-scale 0.95

Primary reported latency is per returned action chunk. The optional
per-action number is only derived by dividing by the returned action horizon.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset
from starVLA.model.framework.base_framework import baseframework


def parse_int_list(text: str, name: str) -> list[int]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"{name} values must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError(f"empty --{name}")
    return values


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def strip_action(batch):
    if isinstance(batch, list):
        return [{k: v for k, v in item.items() if k != "action"} for item in batch]
    return {k: v for k, v in batch.items() if k != "action"}


def time_call(fn: Callable[[], object]) -> float:
    cuda_sync()
    start = time.perf_counter()
    _ = fn()
    cuda_sync()
    return (time.perf_counter() - start) * 1000.0


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "std_ms": float(arr.std(ddof=0)),
    }


def set_or_clear_attr(obj, name: str, value) -> None:
    if value is None:
        if hasattr(obj, name):
            delattr(obj, name)
    else:
        setattr(obj, name, value)


def configure_eval_knobs(
    model,
    *,
    top_steps: int,
    lower_steps: int,
    denoise_step_scale: float | None,
    context_denoise_step_scale: float | None,
    lower_assumed_step_scale: float | None,
) -> None:
    # Match deployment/model_server/server_policy.py overrides.
    set_or_clear_attr(model, "eval_num_inference_timesteps", int(top_steps))
    model.action_model.num_inference_timesteps = int(top_steps)
    model.action_model.config.num_inference_timesteps = int(top_steps)

    set_or_clear_attr(
        model,
        "eval_denoise_step_scale",
        None if denoise_step_scale is None else float(denoise_step_scale),
    )
    set_or_clear_attr(
        model,
        "eval_context_denoise_step_scale",
        None if context_denoise_step_scale is None else float(context_denoise_step_scale),
    )
    set_or_clear_attr(model, "eval_lower_refine_steps", int(lower_steps))
    set_or_clear_attr(
        model,
        "eval_lower_assumed_step_scale",
        None if lower_assumed_step_scale is None else float(lower_assumed_step_scale),
    )


def benchmark_cold(model, batches, warmup: int) -> list[float]:
    sample = strip_action(batches[0])
    for _ in range(warmup):
        model.reset()
        _ = model.predict_action(sample)
    cuda_sync()

    times = []
    for batch in batches:
        no_action = strip_action(batch)
        times.append(time_call(lambda no_action=no_action: (model.reset(), model.predict_action(no_action))))
    return times


def benchmark_cached_second(model, batches, warmup: int) -> list[float]:
    sample = strip_action(batches[0])
    for _ in range(warmup):
        model.reset()
        _ = model.predict_action(sample)
        _ = model.predict_action(sample)
    cuda_sync()

    times = []
    for batch in batches:
        no_action = strip_action(batch)
        model.reset()
        _ = model.predict_action(no_action)
        times.append(time_call(lambda no_action=no_action: model.predict_action(no_action)))
    return times


def _online_call_needs_top_refresh(model) -> bool:
    cache = getattr(model, "_inference_cache", None)
    if not isinstance(cache, dict):
        return False
    if cache.get("signature") is None:
        return True
    num_chunks = int(getattr(model.hierarchical_action_head, "eval_num_chunks", 1))
    return int(cache.get("chunk_idx", 0)) % max(1, num_chunks) == 0


def benchmark_online_cycle(model, batch, warmup: int, cycle_steps: int) -> dict[str, list[float]]:
    no_action = strip_action(batch)
    model.reset()
    for _ in range(warmup):
        _ = model.predict_action(no_action)
    cuda_sync()

    all_times = []
    cached_times = []
    top_refresh_times = []
    for _ in range(cycle_steps):
        needs_top_refresh = _online_call_needs_top_refresh(model)
        ms = time_call(lambda: model.predict_action(no_action))
        all_times.append(ms)
        if needs_top_refresh:
            top_refresh_times.append(ms)
        else:
            cached_times.append(ms)
    return {
        "online_cycle_chunk": all_times,
        "online_cached_lower_chunk": cached_times,
        "online_top_refresh_chunk": top_refresh_times,
    }


def add_measurement_rows(
    *,
    raw_rows: list[dict],
    summary_rows: list[dict],
    eval_mode: str,
    top_steps: int,
    lower_steps: int,
    name: str,
    values: list[float],
    returned_horizon: int,
    eval_num_chunks: int,
    top_horizon: int | None = None,
) -> None:
    if not values:
        return
    for idx, ms in enumerate(values):
        raw_rows.append(
            {
                "eval_mode": eval_mode,
                "top_num_inference_timesteps": top_steps,
                "lower_refine_steps": lower_steps,
                "top_horizon": top_horizon,
                "name": name,
                "call_idx": idx,
                "per_chunk_latency_ms": ms,
                "per_env_action_ms": ms / returned_horizon,
                "returned_horizon": returned_horizon,
                "eval_num_chunks": eval_num_chunks,
            }
        )
    row = {
        "eval_mode": eval_mode,
        "top_num_inference_timesteps": top_steps,
        "lower_refine_steps": lower_steps,
        "top_horizon": top_horizon,
        "name": name,
        **summarize(values),
        "per_chunk_mean_ms": float(np.mean(values)),
        "per_chunk_p90_ms": float(np.percentile(values, 90)),
        "per_env_action_mean_ms": float(np.mean(values) / returned_horizon),
        "returned_horizon": returned_horizon,
        "eval_num_chunks": eval_num_chunks,
    }
    summary_rows.append(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="./results/Checkpoints/libero_stairvla_stage2/checkpoints/steps_30000_pytorch_model.pt",
    )
    parser.add_argument("--top-steps", default="1,2,4", help="Comma-separated top num_inference_timesteps sweep.")
    parser.add_argument("--eval-num-chunks", default="1,2,3,4,5", help="Comma-separated eval_num_chunks sweep.")
    parser.add_argument(
        "--top-only-horizons",
        default="",
        help="Optional comma-separated horizons for top-only/top_custom latency, e.g. 10,15,20,25,32.",
    )
    parser.add_argument("--lower-refine-steps", type=int, default=1, help="Fixed lower_refine_steps value.")
    parser.add_argument("--denoise-step-scale", type=float, default=None)
    parser.add_argument("--context-denoise-step-scale", type=float, default=None)
    parser.add_argument("--lower-assumed-step-scale", type=float, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--cycle-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="results/latency")
    args = parser.parse_args()

    top_steps_values = parse_int_list(args.top_steps, "top-steps")
    eval_num_chunks_values = parse_int_list(args.eval_num_chunks, "eval-num-chunks")
    top_only_horizons = (
        parse_int_list(args.top_only_horizons, "top-only-horizons") if args.top_only_horizons else []
    )
    if args.lower_refine_steps <= 0:
        raise ValueError("--lower-refine-steps must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    device = args.device if torch.cuda.is_available() else "cpu"
    print("checkpoint:", args.ckpt)
    print("device:", device)
    print("top steps:", top_steps_values)
    print("eval_num_chunks:", eval_num_chunks_values)
    print("top_only_horizons:", top_only_horizons if top_only_horizons else "disabled")
    print("lower_refine_steps:", args.lower_refine_steps)
    print("denoise_step_scale:", args.denoise_step_scale)
    print("context_denoise_step_scale:", args.context_denoise_step_scale)
    print("lower_assumed_step_scale:", args.lower_assumed_step_scale)

    model = baseframework.from_pretrained(args.ckpt)
    if args.bf16:
        model = model.to(torch.bfloat16)
    model = model.to(device).eval()

    if not hasattr(model, "hierarchical_action_head"):
        raise ValueError("This benchmark requires a HierarchicalVLA checkpoint.")

    max_chunks = int(model.hierarchical_action_head.num_refine_chunks)
    chunk_horizon = int(model.hierarchical_action_head.chunk_action_horizon)
    top_action_horizon = int(getattr(model, "top_action_horizon", 32))
    print("chunk_action_horizon:", chunk_horizon)
    print("top_action_horizon:", top_action_horizon)
    print("max_eval_num_chunks:", max_chunks)
    for top_horizon in top_only_horizons:
        if top_horizon > top_action_horizon:
            raise ValueError(f"--top-only-horizons includes {top_horizon}, but top_action_horizon={top_action_horizon}")

    dataset = get_vla_dataset(data_cfg=model.config.datasets.vla_data)
    loader = DataLoader(
        Subset(dataset, list(range(min(args.num_samples, len(dataset))))),
        batch_size=1,
        num_workers=0,
        collate_fn=collate_fn,
    )
    batches = list(loader)
    if not batches:
        raise RuntimeError("No samples available for latency benchmark.")
    print("Dataset benchmark samples:", len(batches))

    raw_rows: list[dict] = []
    summary_rows: list[dict] = []
    for top_steps in top_steps_values:
        for top_horizon in top_only_horizons:
            model.eval_action_mode = "top_custom"
            model.eval_top_horizon = int(top_horizon)
            configure_eval_knobs(
                model,
                top_steps=top_steps,
                lower_steps=args.lower_refine_steps,
                denoise_step_scale=args.denoise_step_scale,
                context_denoise_step_scale=args.context_denoise_step_scale,
                lower_assumed_step_scale=args.lower_assumed_step_scale,
            )
            print(f"\n=== top-only: top_steps={top_steps}, top_horizon={top_horizon} ===")
            values = benchmark_cold(model, batches, warmup=args.warmup)
            add_measurement_rows(
                raw_rows=raw_rows,
                summary_rows=summary_rows,
                eval_mode="top_only",
                top_steps=top_steps,
                lower_steps=0,
                name="top_only_chunk",
                values=values,
                returned_horizon=top_horizon,
                eval_num_chunks=0,
                top_horizon=top_horizon,
            )
            top_only_chunk_count = float(top_horizon) / float(chunk_horizon)
            print(
                f"{'top_only_chunk':24s} full-horizon mean={np.mean(values):.1f} ms "
                f"p90={np.percentile(values, 90):.1f} ms "
                f"per-{chunk_horizon}action-chunk mean={np.mean(values) / top_only_chunk_count:.1f} ms"
            )

        model.eval_action_mode = "default"
        if hasattr(model, "eval_top_horizon"):
            delattr(model, "eval_top_horizon")
        for requested_eval_num_chunks in eval_num_chunks_values:
            eval_num_chunks = max(1, min(int(requested_eval_num_chunks), max_chunks))
            model.hierarchical_action_head.eval_num_chunks = eval_num_chunks
            configure_eval_knobs(
                model,
                top_steps=top_steps,
                lower_steps=args.lower_refine_steps,
                denoise_step_scale=args.denoise_step_scale,
                context_denoise_step_scale=args.context_denoise_step_scale,
                lower_assumed_step_scale=args.lower_assumed_step_scale,
            )
            print(
                f"\n=== top_steps={top_steps}, eval_num_chunks={eval_num_chunks}, "
                f"lower_refine_steps={args.lower_refine_steps} ==="
            )

            online_measurements = benchmark_online_cycle(
                model,
                batches[0],
                warmup=args.warmup,
                cycle_steps=args.cycle_steps,
            )
            measurements = {
                "cold_chunk": benchmark_cold(model, batches, warmup=args.warmup),
                "cached_lower_chunk": benchmark_cached_second(model, batches, warmup=args.warmup),
                **online_measurements,
            }
            for name, values in measurements.items():
                add_measurement_rows(
                    raw_rows=raw_rows,
                    summary_rows=summary_rows,
                    eval_mode="hierarchical",
                    top_steps=top_steps,
                    lower_steps=args.lower_refine_steps,
                    name=name,
                    values=values,
                    returned_horizon=chunk_horizon,
                    eval_num_chunks=eval_num_chunks,
                )
                if values:
                    print(
                        f"{name:24s} per-chunk mean={np.mean(values):.1f} ms "
                        f"p90={np.percentile(values, 90):.1f} ms"
                    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_df = pd.DataFrame(raw_rows)
    summary_df = pd.DataFrame(summary_rows)
    raw_path = out_dir / "top_lower_refine_latency_raw.csv"
    summary_path = out_dir / "top_lower_refine_latency_summary.csv"
    compact_path = out_dir / "top_lower_refine_latency_compact.csv"
    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    meta_path = out_dir / "top_lower_refine_latency_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "checkpoint": args.ckpt,
                "device": device,
                "bf16": args.bf16,
                "top_steps": top_steps_values,
                "eval_num_chunks": eval_num_chunks_values,
                "top_only_horizons": top_only_horizons,
                "lower_refine_steps": args.lower_refine_steps,
                "denoise_step_scale": args.denoise_step_scale,
                "context_denoise_step_scale": args.context_denoise_step_scale,
                "lower_assumed_step_scale": args.lower_assumed_step_scale,
                "num_samples": len(batches),
                "warmup": args.warmup,
                "cycle_steps": args.cycle_steps,
                "chunk_action_horizon": chunk_horizon,
            },
            indent=2,
        )
    )

    compact = (
        summary_df[(summary_df["eval_mode"] == "hierarchical") & (summary_df["name"] == "online_cycle_chunk")][
            [
                "top_num_inference_timesteps",
                "eval_num_chunks",
                "lower_refine_steps",
                "per_chunk_mean_ms",
                "per_chunk_p90_ms",
            ]
        ]
        .sort_values(["top_num_inference_timesteps", "eval_num_chunks"])
        .rename(
            columns={
                "top_num_inference_timesteps": "top_steps",
                "per_chunk_mean_ms": "online_cycle_per_chunk_mean_ms",
                "per_chunk_p90_ms": "online_cycle_per_chunk_p90_ms",
            }
        )
    )
    compact.to_csv(compact_path, index=False)

    print("\nsaved:", summary_path)
    print("saved:", raw_path)
    print("saved:", compact_path)
    if top_only_horizons:
        top_only_compact_path = out_dir / "top_only_latency_compact.csv"
        top_only_compact = (
            summary_df[(summary_df["eval_mode"] == "top_only") & (summary_df["name"] == "top_only_chunk")][
                [
                    "top_num_inference_timesteps",
                    "top_horizon",
                    "per_chunk_mean_ms",
                    "per_chunk_p90_ms",
                ]
            ]
            .sort_values(["top_num_inference_timesteps", "top_horizon"])
            .rename(
                columns={
                    "top_num_inference_timesteps": "top_steps",
                    "per_chunk_mean_ms": "top_only_full_horizon_mean_ms",
                    "per_chunk_p90_ms": "top_only_full_horizon_p90_ms",
                }
            )
        )
        top_only_compact["chunk_action_horizon"] = chunk_horizon
        top_only_compact["top_only_num_chunks"] = top_only_compact["top_horizon"] / float(chunk_horizon)
        top_only_compact["top_only_per_chunk_mean_ms"] = (
            top_only_compact["top_only_full_horizon_mean_ms"] / top_only_compact["top_only_num_chunks"]
        )
        top_only_compact["top_only_per_chunk_p90_ms"] = (
            top_only_compact["top_only_full_horizon_p90_ms"] / top_only_compact["top_only_num_chunks"]
        )
        top_only_compact = top_only_compact[
            [
                "top_steps",
                "top_horizon",
                "chunk_action_horizon",
                "top_only_num_chunks",
                "top_only_per_chunk_mean_ms",
                "top_only_per_chunk_p90_ms",
                "top_only_full_horizon_mean_ms",
                "top_only_full_horizon_p90_ms",
            ]
        ]
        top_only_compact.to_csv(top_only_compact_path, index=False)
        print("saved:", top_only_compact_path)
        print(f"\nTop-only latency normalized per {chunk_horizon}-action chunk:")
        print(top_only_compact.to_string(index=False))
        top_only_pivot = top_only_compact.pivot(
            index="top_steps", columns="top_horizon", values="top_only_per_chunk_mean_ms"
        )
        print(f"\nTop-only mean ms per {chunk_horizon}-action chunk: rows=top_steps, cols=top_horizon")
        print(top_only_pivot.to_string(float_format=lambda x: f"{x:.1f}"))

    print("\nCompact online per-chunk latency:")
    print(compact.to_string(index=False))

    pivot = compact.pivot(
        index="top_steps",
        columns="eval_num_chunks",
        values="online_cycle_per_chunk_mean_ms",
    )
    print(
        "\nMean ms per returned chunk: rows=top_steps, cols=eval_num_chunks "
        f"(lower_refine_steps={args.lower_refine_steps})"
    )
    print(pivot.to_string(float_format=lambda x: f"{x:.1f}"))


if __name__ == "__main__":
    main()
