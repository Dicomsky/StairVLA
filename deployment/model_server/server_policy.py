# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import logging
import socket
import argparse
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.base_framework import baseframework
import torch, os


def main(args) -> None:
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()

    vla = baseframework.from_pretrained( # TODO should auto detect framework from model path
        args.ckpt_path,
    )

    if args.use_bf16: # False
        vla = vla.to(torch.bfloat16)
    if args.hier_eval_mode != "default":
        setattr(vla, "eval_action_mode", args.hier_eval_mode)
        logging.info("Using hierarchical eval action mode: %s", args.hier_eval_mode)
    if args.hier_top_horizon is not None:
        if not hasattr(vla, "top_action_horizon"):
            raise ValueError("--hier_top_horizon is only valid for VLA checkpoints with a top action horizon")
        top_horizon = int(vla.top_action_horizon)
        hier_top_horizon = int(args.hier_top_horizon)
        if hier_top_horizon < 1 or hier_top_horizon > top_horizon:
            raise ValueError(f"--hier_top_horizon must be in [1, {top_horizon}], got {hier_top_horizon}")
        setattr(vla, "eval_top_horizon", hier_top_horizon)
        logging.info("Using hierarchical top-only horizon override: %d/%d", hier_top_horizon, top_horizon)
    if args.hier_eval_num_chunks is not None:
        if not hasattr(vla, "hierarchical_action_head"):
            raise ValueError("--hier_eval_num_chunks is only valid for HierarchicalVLA checkpoints")
        max_chunks = int(vla.hierarchical_action_head.num_refine_chunks)
        eval_num_chunks = max(1, min(int(args.hier_eval_num_chunks), max_chunks))
        vla.hierarchical_action_head.eval_num_chunks = eval_num_chunks
        logging.info(
            "Using hierarchical eval_num_chunks override: %s/%s",
            eval_num_chunks,
            max_chunks,
        )
    if args.denoise_step_scale is not None:
        setattr(vla, "eval_denoise_step_scale", float(args.denoise_step_scale))
        logging.info("Using eval denoise step scale override: %.4f", float(args.denoise_step_scale))
    if args.context_denoise_step_scale is not None:
        if not hasattr(vla, "hierarchical_action_head"):
            raise ValueError("--context_denoise_step_scale is only valid for HierarchicalVLA checkpoints")
        setattr(vla, "eval_context_denoise_step_scale", float(args.context_denoise_step_scale))
        logging.info("Using eval context denoise step scale override: %.4f", float(args.context_denoise_step_scale))
    if args.num_inference_timesteps is not None:
        steps = max(1, int(args.num_inference_timesteps))
        setattr(vla, "eval_num_inference_timesteps", steps)
        logging.info("Using eval num_inference_timesteps override: %d", steps)
    if args.lower_refine_steps is not None:
        if not hasattr(vla, "hierarchical_action_head"):
            raise ValueError("--lower_refine_steps is only valid for HierarchicalVLA checkpoints")
        lower_refine_steps = max(1, int(args.lower_refine_steps))
        setattr(vla, "eval_lower_refine_steps", lower_refine_steps)
        logging.info("Using eval lower_refine_steps override: %d", lower_refine_steps)
    if args.lower_assumed_step_scale is not None:
        if not hasattr(vla, "hierarchical_action_head"):
            raise ValueError("--lower_assumed_step_scale is only valid for HierarchicalVLA checkpoints")
        lower_assumed_step_scale = float(args.lower_assumed_step_scale)
        if not 0.0 <= lower_assumed_step_scale <= 1.0:
            raise ValueError("--lower_assumed_step_scale must be in [0, 1]")
        setattr(vla, "eval_lower_assumed_step_scale", lower_assumed_step_scale)
        logging.info("Using eval lower assumed step scale override: %.4f", lower_assumed_step_scale)
    if args.lower_velocity_min_step_compensation:
        if not hasattr(vla, "hierarchical_action_head"):
            raise ValueError("--lower_velocity_min_step_compensation is only valid for HierarchicalVLA checkpoints")
        setattr(vla.hierarchical_action_head, "eval_lower_velocity_min_step_compensation", True)
        logging.info("Using eval lower velocity min-step compensation")
    vla = vla.to("cuda").eval()

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # start websocket server
    server = WebsocketPolicyServer(
        policy=vla,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={"env": "simpler_env"},
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout" , type=int, default=1800, help="Idle timeout in seconds, -1 means never close")
    parser.add_argument(
        "--hier_eval_mode",
        type=str,
        default="default",
        choices=["default", "lower_first8", "lower_first16", "top_first8", "top32", "top_custom"],
        help=(
            "Eval-only ablation for HierarchicalVLA: default uses configured hierarchical online chunks; "
            "lower_first8 always refreshes top and returns only lower chunk 0; "
            "lower_first16 returns the first two lower chunks; "
            "top_first8 returns top coarse first 8 actions; top32 returns top coarse 32 actions; "
            "top_custom returns the first --hier_top_horizon top actions."
        ),
    )
    parser.add_argument(
        "--hier_top_horizon",
        type=int,
        default=None,
        help="Top-only eval horizon used with --hier_eval_mode top_custom.",
    )
    parser.add_argument(
        "--hier_eval_num_chunks",
        type=int,
        default=None,
        help=(
            "Optional eval-time override for HierarchicalVLA online top-refresh cycle. "
            "For chunk_action_horizon=5, value 6 means 30 env steps per top plan, "
            "value 3 means 15 env steps per top plan."
        ),
    )
    parser.add_argument(
        "--denoise_step_scale",
        type=float,
        default=None,
        help=(
            "Optional eval-time override for flow-matching action heads. "
            "For example, 0.9 runs the flow until cumulative progress reaches 0.9, "
            "truncating the final Euler update if needed."
        ),
    )
    parser.add_argument(
        "--num_inference_timesteps",
        type=int,
        default=None,
        help=(
            "Optional eval-time override for flow-matching action head inference steps. "
            "Use 1 together with --denoise_step_scale 0.9 to run noise + 0.9 * velocity once."
        ),
    )
    parser.add_argument(
        "--context_denoise_step_scale",
        type=float,
        default=None,
        help=(
            "Optional eval-time override for HierarchicalVLA context coarse-plan denoise scale. "
            "If unset, the checkpoint's context_top_plan_step_scale config is used."
        ),
    )
    parser.add_argument(
        "--lower_refine_steps",
        type=int,
        default=None,
        help=(
            "Optional eval-time override for HierarchicalVLA lower refiner passes. "
            "The top policy is not rerun; the remaining lower flow interval is split across these passes."
        ),
    )
    parser.add_argument(
        "--lower_assumed_step_scale",
        type=float,
        default=None,
        help=(
            "Optional eval-time override for the lower refiner's assumed flow timestep. "
            "This leaves the top coarse action generated by --denoise_step_scale unchanged, "
            "but computes the lower update interval as 1 - lower_assumed_step_scale."
        ),
    )
    parser.add_argument(
        "--lower_velocity_min_step_compensation",
        action="store_true",
        help=(
            "Eval-only HierarchicalVLA lower velocity compensation. "
            "When enabled, lower velocity updates use at least lower_flow_min_step_size as the Euler step, "
            "matching checkpoints trained with a clamped lower flow target denominator."
        ),
    )
    return parser


def start_debugpy_once():
    """start debugpy once"""
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10095))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10095 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    # Only an explicit DEBUG=1/true/yes/on starts debugpy (DEBUG=0 or an empty value must not block).
    if os.getenv("DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
        print("🔍 DEBUGPY is enabled")
        start_debugpy_once()
    main(args)
