"""Hierarchical StarVLA framework.

This framework keeps the original top policy stack intact enough to reuse
existing checkpoints, while attaching a generic hierarchical refinement head
below it.

Current top-policy support:
  - QwenGR00T
  - QwenOFT
  - QwenPI
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.action_model.GR00T_ActionHeader import (
    FlowmatchingActionHead,
    get_action_model as get_gr00t_action_model,
)
from starVLA.model.modules.action_model.Hierarchical_ActionHead import (
    HierarchicalRefinerActionHead,
    gather_temporal_chunk,
    gather_temporal_context,
    linear_resample,
)
from starVLA.model.modules.action_model.MLP_ActionHeader import (
    L1RegressionActionHead,
    get_action_model as get_oft_action_model,
)
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    LayerwiseFlowmatchingActionHead,
    get_action_model as get_pi_action_model,
)
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


def _cfg_get(obj, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, key):
        return getattr(obj, key)
    try:
        return obj.get(key, default)
    except Exception:
        return default


def _cfg_set(obj, key: str, value) -> None:
    if obj is None:
        return
    if isinstance(obj, dict):
        obj[key] = value
        return
    try:
        setattr(obj, key, value)
    except Exception:
        try:
            obj[key] = value
        except Exception:
            pass


def _require_cfg_value(obj, key: str, path_hint: str):
    value = _cfg_get(obj, key, None)
    if value is None:
        raise ValueError(
            f"Missing required config `{path_hint}` for HierarchicalVLA. "
            "Please set it explicitly in the benchmark yaml."
        )
    return value


def _slice_or_pad_actions(actions: torch.Tensor, target_len: int) -> torch.Tensor:
    if actions.shape[1] >= target_len:
        return actions[:, :target_len]
    pad_len = target_len - actions.shape[1]
    pad = actions[:, -1:, :].expand(-1, pad_len, -1)
    return torch.cat([actions, pad], dim=1)


def _slice_or_pad_mask(mask: torch.Tensor, target_len: int) -> torch.Tensor:
    if mask.shape[1] >= target_len:
        return mask[:, :target_len]
    pad_len = target_len - mask.shape[1]
    pad = torch.zeros(mask.shape[0], pad_len, device=mask.device, dtype=torch.bool)
    return torch.cat([mask, pad], dim=1)


@FRAMEWORK_REGISTRY.register("HierarchicalVLA")
class HierarchicalVLA(baseframework):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.top_policy_name = self._get_top_policy_name()

        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.hidden_size = self.qwen_vl_interface.model.config.hidden_size
        self.action_model = self._build_top_action_model()
        self.top_action_horizon = self._get_top_action_horizon()
        self.action_dim = int(
            _require_cfg_value(
                config.framework.action_model,
                "action_dim",
                "framework.action_model.action_dim",
            )
        )
        self.state_dim = int(
            _require_cfg_value(
                config.framework.action_model,
                "state_dim",
                "framework.action_model.state_dim",
            )
        )

        # Hierarchical refiner sits below the original top policy.
        self.hierarchical_action_head = HierarchicalRefinerActionHead(
            config=self.config,
            top_hidden_dim=self.hidden_size,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
            top_action_horizon=self.top_action_horizon,
        )

        head_cfg = _cfg_get(self.config.framework, "hierarchical_action_head", None)
        self.lower_init_from_top_action_head = bool(
            _cfg_get(head_cfg, "lower_init_from_top_action_head", False)
        )
        self._lower_init_from_top_done = False
        self.top_loss_weight = float(_cfg_get(head_cfg, "top_loss_weight", 1.0))
        self.refine_loss_weight = float(_cfg_get(head_cfg, "refine_loss_weight", 1.0))
        self.eval_action_mode = str(_cfg_get(head_cfg, "eval_action_mode", "default"))
        self.coarse_plan_tail_steps = int(_cfg_get(head_cfg, "coarse_plan_tail_steps", 0))
        self.coarse_plan_tail_mode = str(_cfg_get(head_cfg, "coarse_plan_tail_mode", "repeat_last")).lower()
        self.coarse_plan_tail_decay = float(_cfg_get(head_cfg, "coarse_plan_tail_decay", 0.5))
        self.top_plan_step_scale_enabled = bool(_cfg_get(head_cfg, "top_plan_step_scale_enabled", False))
        self.top_plan_step_scale_min = float(_cfg_get(head_cfg, "top_plan_step_scale_min", 1.0))
        self.top_plan_step_scale_max = float(_cfg_get(head_cfg, "top_plan_step_scale_max", 1.0))
        self.context_top_plan_step_scale_enabled = bool(
            _cfg_get(head_cfg, "context_top_plan_step_scale_enabled", False)
        )
        self.context_top_plan_step_scale_min = float(
            _cfg_get(head_cfg, "context_top_plan_step_scale_min", self.top_plan_step_scale_min)
        )
        self.context_top_plan_step_scale_max = float(
            _cfg_get(head_cfg, "context_top_plan_step_scale_max", self.top_plan_step_scale_max)
        )
        # When true the flow-velocity TARGET is anchored at the upstream (coarse) plan and
        # the full remaining span (1 - base_scale), instead of at the refiner's own input
        # x_t over (1 - t). On the straight path the two are algebraically identical, but
        # the x_t form is 0/0 as t -> 1 and needed lower_flow_min_step_size to stay finite.
        self.lower_velocity_target_from_upstream = bool(
            _cfg_get(head_cfg, "lower_velocity_target_from_upstream", False)
        )
        self.lower_flow_path_mix_enabled = bool(_cfg_get(head_cfg, "lower_flow_path_mix_enabled", False))
        self.lower_flow_path_mix_min_scale = float(
            _cfg_get(head_cfg, "lower_flow_path_mix_min_scale", self.top_plan_step_scale_min)
        )
        self.lower_flow_path_mix_max_scale = float(
            _cfg_get(head_cfg, "lower_flow_path_mix_max_scale", self.top_plan_step_scale_max)
        )
        self.debug_print = bool(_cfg_get(head_cfg, "debug_print", False))
        self.debug_max_prints = int(_cfg_get(head_cfg, "debug_max_prints", 3))
        self._debug_data_print_count = 0
        self._debug_top_print_count = 0

        self.action_token = "🔍"
        self.action_token_id = None
        if self.top_policy_name == "QwenOFT":
            self.action_token_id = self.qwen_vl_interface.processor.tokenizer(
                self.action_token, add_special_tokens=False
            )["input_ids"][0]
        self._reset_inference_cache()

    @staticmethod
    def _copy_matching_module_state(src: nn.Module, dst: nn.Module) -> tuple[int, int]:
        src_state = src.state_dict()
        dst_state = dst.state_dict()
        copied = 0
        skipped = 0
        new_state = {}
        for key, dst_value in dst_state.items():
            src_value = src_state.get(key)
            if src_value is not None and tuple(src_value.shape) == tuple(dst_value.shape):
                new_state[key] = src_value.detach().to(device=dst_value.device, dtype=dst_value.dtype)
                copied += 1
            else:
                new_state[key] = dst_value
                skipped += 1
        dst.load_state_dict(new_state, strict=True)
        return copied, skipped

    @staticmethod
    def _copy_embedding_prefix(src: nn.Embedding | None, dst: nn.Embedding | None) -> int:
        if src is None or dst is None:
            return 0
        rows = min(src.weight.shape[0], dst.weight.shape[0])
        cols = min(src.weight.shape[1], dst.weight.shape[1])
        if rows <= 0 or cols <= 0:
            return 0
        with torch.no_grad():
            dst.weight[:rows, :cols].copy_(src.weight[:rows, :cols].to(device=dst.weight.device, dtype=dst.weight.dtype))
        return rows

    def _maybe_initialize_lower_from_top_action_head(self) -> None:
        if self._lower_init_from_top_done or not self.lower_init_from_top_action_head:
            return
        if not self.training:
            return
        self._lower_init_from_top_done = True

        lower = self.hierarchical_action_head
        if not isinstance(self.action_model, FlowmatchingActionHead) or not getattr(lower, "use_dit_refiner", False):
            print("[HierarchicalVLA:init] skip lower_init_from_top_action_head: incompatible top/lower action heads")
            return

        copy_report: list[str] = []
        if getattr(lower, "dit_action_encoder", None) is not None and hasattr(self.action_model, "action_encoder"):
            copied, skipped = self._copy_matching_module_state(
                self.action_model.action_encoder,
                lower.dit_action_encoder,
            )
            copy_report.append(f"action_encoder={copied} copied/{skipped} kept")

        if getattr(lower, "dit_model", None) is not None and hasattr(self.action_model, "model"):
            copied, skipped = self._copy_matching_module_state(self.action_model.model, lower.dit_model)
            copy_report.append(f"dit_model={copied} copied/{skipped} kept")

        if hasattr(self.action_model, "action_decoder"):
            copied, skipped = self._copy_matching_module_state(self.action_model.action_decoder, lower.output_head)
            copy_report.append(f"action_decoder={copied} copied/{skipped} kept")

        rows = self._copy_embedding_prefix(
            getattr(self.action_model, "future_tokens", None),
            getattr(lower, "dit_future_tokens", None),
        )
        if rows:
            copy_report.append(f"future_tokens={rows} rows")

        rows = self._copy_embedding_prefix(
            getattr(self.action_model, "position_embedding", None),
            getattr(lower, "dit_position_embedding", None),
        )
        if rows:
            copy_report.append(f"position_embedding={rows} rows")

        self.lower_init_from_top_action_head = False
        _cfg_set(
            _cfg_get(self.config.framework, "hierarchical_action_head", None),
            "lower_init_from_top_action_head",
            False,
        )
        print("[HierarchicalVLA:init] initialized lower DiT from top action head:", "; ".join(copy_report))

    def _debug_log_data_flow(
        self,
        top_images,
        refine_images,
        instructions,
        actions_np,
        top_state,
        refine_state,
    ) -> None:
        if not self.debug_print or self._debug_data_print_count >= self.debug_max_prints:
            return
        top_views = len(top_images[0]) if top_images else 0
        refine_first = refine_images[0] if refine_images else []
        anchor_count = len(refine_first) if refine_first and isinstance(refine_first[0], (list, tuple)) else 1
        refine_views = len(refine_first[0]) if anchor_count > 1 else len(refine_first)
        action_shape = None if actions_np is None else np.array(actions_np).shape
        top_state_shape = None if top_state is None else np.array(top_state).shape
        refine_state_shape = None if refine_state is None else np.array(refine_state).shape
        print(
            "[HierarchicalVLA:data]",
            f"top_policy={self.top_policy_name}",
            f"batch={len(instructions)}",
            f"top_views={top_views}",
            f"refine_anchors={anchor_count}",
            f"refine_views={refine_views}",
            f"actions={action_shape}",
            f"top_state={top_state_shape}",
            f"refine_state={refine_state_shape}",
            f"top_horizon={self.top_action_horizon}",
            f"dense_horizon={self.hierarchical_action_head.dense_action_horizon}",
            f"chunk_horizon={self.hierarchical_action_head.chunk_action_horizon}",
        )
        self._debug_data_print_count += 1

    def _debug_log_top_policy(
        self,
        top_plan: torch.Tensor,
        top_features: torch.Tensor,
        top_target: torch.Tensor | None,
        dense_target: torch.Tensor | None,
        state_tensor: torch.Tensor | None,
    ) -> None:
        if not self.debug_print or self._debug_top_print_count >= self.debug_max_prints:
            return
        print(
            "[HierarchicalVLA:top]",
            f"top_plan={tuple(top_plan.shape)}",
            f"top_features={tuple(top_features.shape)}",
            f"top_target={None if top_target is None else tuple(top_target.shape)}",
            f"dense_target={None if dense_target is None else tuple(dense_target.shape)}",
            f"state={None if state_tensor is None else tuple(state_tensor.shape)}",
            f"top_plan_mean={top_plan.detach().float().mean().item():.4f}",
            f"top_plan_std={top_plan.detach().float().std(unbiased=False).item():.4f}",
        )
        self._debug_top_print_count += 1

    def _get_top_policy_name(self) -> str:
        top_policy_cfg = _cfg_get(self.config.framework, "top_policy", None)
        return _cfg_get(top_policy_cfg, "name", _cfg_get(self.config.framework, "top_policy_name", "QwenGR00T"))

    def _get_top_action_horizon(self) -> int:
        action_cfg = self.config.framework.action_model
        if self.top_policy_name == "QwenOFT":
            future = int(_cfg_get(action_cfg, "future_action_window_size", 0))
            past = int(_cfg_get(action_cfg, "past_action_window_size", 0))
            return future + past + 1
        return int(_cfg_get(action_cfg, "action_horizon", _cfg_get(action_cfg, "future_action_window_size", 0) + 1))

    def _build_top_action_model(self):
        if self.top_policy_name == "QwenGR00T":
            self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.hidden_size
            return get_gr00t_action_model(config=self.config)
        if self.top_policy_name == "QwenPI":
            self.config.framework.qwenvl.vl_hidden_dim = self.hidden_size
            self.config.framework.qwenvl.num_vl_layers = 36
            return get_pi_action_model(config=self.config)
        if self.top_policy_name == "QwenOFT":
            self.config.framework.action_model.action_hidden_dim = self.hidden_size
            return get_oft_action_model(config=self.config)
        raise NotImplementedError(f"Unsupported top policy `{self.top_policy_name}` for HierarchicalVLA")

    def _prepare_examples(self, examples: List[dict], resize_for_infer: bool = False):
        if not isinstance(examples, list):
            examples = [examples]
        top_images = [example["image"] for example in examples]
        refine_images = [example.get("obs_anchor_images", example["image"]) for example in examples]
        if resize_for_infer:
            top_images = [to_pil_preserve(imgs) for imgs in top_images]
        if resize_for_infer:
            train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
            if train_obs_image_size:
                top_images = resize_images(top_images, target_size=train_obs_image_size)
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples] if "action" in examples[0] else None
        action_valid_mask = (
            [example["action_valid_mask"] for example in examples]
            if "action_valid_mask" in examples[0]
            else None
        )
        top_state = None
        refine_state = None
        state_key = None
        anchor_state_key = None
        if "state" in examples[0]:
            state_key = "state"
            anchor_state_key = "obs_anchor_states"

        if state_key is not None:
            raw_state = [example[state_key] for example in examples]
            top_state = []
            for sample_state in raw_state:
                if hasattr(sample_state, "ndim") and sample_state.ndim == 2 and sample_state.shape[0] > 1:
                    top_state.append(sample_state[:1])
                else:
                    top_state.append(sample_state)
            refine_state = [example.get(anchor_state_key, example[state_key]) for example in examples]
        return top_images, refine_images, instructions, actions, action_valid_mask, top_state, refine_state

    def _encode_qwen(self, batch_images, instructions):
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        last_hidden = qwenvl_outputs.hidden_states[-1]
        return last_hidden, qwen_inputs, qwenvl_outputs.hidden_states

    def _build_oft_prompt(self, instructions):
        action_tokens = self.action_token * self.top_action_horizon
        suffix = f" Please predict the next {self.top_action_horizon} robot actions: <action>{action_tokens}<action>."
        return [instruction + suffix for instruction in instructions]

    def _gather_action_token_embeddings(self, last_hidden: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        if self.action_token_id is None:
            raise ValueError("action_token_id is not initialized for QwenOFT top policy.")

        device = input_ids.device
        batch_size, seq_len, hidden_dim = last_hidden.shape
        mask = input_ids == self.action_token_id
        counts = mask.sum(dim=1)
        if (counts < self.top_action_horizon).any():
            insufficient = (counts < self.top_action_horizon).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"Not enough OFT action tokens for samples {insufficient}; counts={counts.tolist()}"
            )

        idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))
        selected_pos = masked_pos.topk(k=self.top_action_horizon, dim=-1).values.sort(dim=-1).values
        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, hidden_dim)
        return last_hidden.gather(dim=1, index=expanded_index)

    def _sample_top_plan_step_scale(self, device) -> torch.Tensor:
        if not self.training:
            if hasattr(self, "eval_denoise_step_scale"):
                return torch.tensor(float(self.eval_denoise_step_scale), device=device)
            if self.lower_flow_path_mix_enabled:
                return torch.tensor(max(0.0, self.lower_flow_path_mix_min_scale), device=device)
            if self.top_plan_step_scale_enabled:
                scale_min = max(0.0, self.top_plan_step_scale_min)
                scale_max = max(scale_min, self.top_plan_step_scale_max)
                return torch.tensor((scale_min + scale_max) * 0.5, device=device)
            return torch.tensor(
                1.0,
                device=device,
            )
        if self.lower_flow_path_mix_enabled:
            return torch.tensor(max(0.0, self.lower_flow_path_mix_min_scale), device=device)
        if not self.top_plan_step_scale_enabled:
            return torch.tensor(1.0, device=device)
        scale_min = max(0.0, self.top_plan_step_scale_min)
        scale_max = max(scale_min, self.top_plan_step_scale_max)
        return torch.empty((), device=device).uniform_(scale_min, scale_max)

    def _sample_lower_flow_path_t(
        self,
        base_scale: torch.Tensor,
        batch_size: int,
        device,
        dtype,
    ) -> torch.Tensor:
        if not torch.is_tensor(base_scale):
            base_scale = torch.tensor(base_scale, device=device, dtype=dtype)
        else:
            base_scale = base_scale.to(device=device, dtype=dtype)
        if base_scale.dim() == 0:
            base_scale = base_scale.expand(batch_size)
        else:
            base_scale = base_scale.reshape(-1)
            if base_scale.numel() == 1:
                base_scale = base_scale.expand(batch_size)
            elif base_scale.numel() != batch_size:
                base_scale = base_scale[:batch_size]

        if not self.training or not self.lower_flow_path_mix_enabled:
            return base_scale

        max_scale = max(float(base_scale.detach().float().max().item()), self.lower_flow_path_mix_max_scale)
        max_scale = min(1.0, max_scale)
        if max_scale <= float(base_scale.detach().float().min().item()):
            return base_scale
        rand = torch.rand(batch_size, device=device, dtype=dtype)
        return base_scale + rand * (max_scale - base_scale)

    def _sample_context_top_plan_step_scale(self, device) -> torch.Tensor | None:
        if not self.training and hasattr(self, "eval_context_denoise_step_scale"):
            return torch.tensor(float(self.eval_context_denoise_step_scale), device=device)
        if not self.context_top_plan_step_scale_enabled:
            return None
        if not self.training:
            scale_min = max(0.0, self.context_top_plan_step_scale_min)
            scale_max = max(scale_min, self.context_top_plan_step_scale_max)
            return torch.tensor((scale_min + scale_max) * 0.5, device=device)
        scale_min = max(0.0, self.context_top_plan_step_scale_min)
        scale_max = max(scale_min, self.context_top_plan_step_scale_max)
        return torch.empty((), device=device).uniform_(scale_min, scale_max)

    def _predict_top_plan_from_encoded(
        self,
        last_hidden,
        qwen_inputs,
        all_hidden,
        state_tensor,
        step_scale,
        context_step_scale=None,
    ):
        if self.top_policy_name == "QwenGR00T":
            eval_num_steps = (
                getattr(self, "eval_num_inference_timesteps", None)
                if not self.training
                else None
            )
            with torch.autocast("cuda", dtype=torch.float32):
                top_output = self.action_model.predict_action(
                    last_hidden,
                    state_tensor,
                    denoise_step_scale=step_scale,
                    context_denoise_step_scale=context_step_scale,
                    num_inference_timesteps=eval_num_steps,
                )
                if context_step_scale is not None:
                    top_plan, context_top_plan = top_output
                else:
                    top_plan, context_top_plan = top_output, None
            return top_plan, last_hidden, context_top_plan
        if context_step_scale is not None:
            raise NotImplementedError(
                "context_top_plan_step_scale currently reuses a single top action-head pass only for QwenGR00T."
            )
        if self.top_policy_name == "QwenPI":
            expected_layers = len(self.action_model.model.transformer_blocks)
            vl_embs_list = list(all_hidden[-expected_layers:])
            with torch.autocast("cuda", dtype=torch.float32):
                top_plan = self.action_model.predict_action(
                    vl_embs_list,
                    state_tensor,
                    denoise_step_scale=step_scale,
                    num_inference_timesteps=getattr(self, "eval_num_inference_timesteps", None)
                    if not self.training
                    else None,
                )
            return top_plan, last_hidden, None
        if self.top_policy_name == "QwenOFT":
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(last_hidden, input_ids)
            with torch.autocast("cuda", dtype=torch.float32):
                top_plan = self.action_model.predict_action(action_queries)
            return top_plan, last_hidden, None
        raise NotImplementedError(f"Unsupported top policy `{self.top_policy_name}`")

    def _compute_top_policy_outputs(self, batch_images, instructions, state):
        state_tensor = None
        encoded_instructions = instructions
        if self.top_policy_name == "QwenOFT":
            encoded_instructions = self._build_oft_prompt(instructions)

        last_hidden, qwen_inputs, all_hidden = self._encode_qwen(batch_images, encoded_instructions)
        if state is not None:
            state_tensor = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)

        top_plan_step_scale = self._sample_top_plan_step_scale(last_hidden.device)
        context_step_scale = self._sample_context_top_plan_step_scale(last_hidden.device)
        top_plan, top_features, context_top_plan = self._predict_top_plan_from_encoded(
            last_hidden,
            qwen_inputs,
            all_hidden,
            state_tensor,
            top_plan_step_scale,
            context_step_scale=context_step_scale,
        )
        return (
            top_plan,
            top_features,
            state_tensor,
            qwen_inputs,
            last_hidden,
            all_hidden,
            context_top_plan,
            top_plan_step_scale,
        )

    def _compute_top_loss(self, top_target, top_features, state_tensor, qwen_inputs, last_hidden, all_hidden):
        if self.top_policy_name == "QwenGR00T":
            repeated_diffusion_steps = int(_cfg_get(self.config.trainer, "repeated_diffusion_steps", 4))
            top_target_repeated = top_target.repeat(repeated_diffusion_steps, 1, 1)
            hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            state_repeated = state_tensor.repeat(repeated_diffusion_steps, 1, 1) if state_tensor is not None else None
            with torch.autocast("cuda", dtype=torch.float32):
                return self.action_model(hidden_repeated, top_target_repeated, state_repeated)

        if self.top_policy_name == "QwenPI":
            repeated_diffusion_steps = 2
            expected_layers = len(self.action_model.model.transformer_blocks)
            vl_embs_list = list(all_hidden[-expected_layers:])
            vl_embs_list_repeated = [hidden.repeat(repeated_diffusion_steps, 1, 1) for hidden in vl_embs_list]
            top_target_repeated = top_target.repeat(repeated_diffusion_steps, 1, 1)
            state_repeated = state_tensor.repeat(repeated_diffusion_steps, 1, 1) if state_tensor is not None else None
            with torch.autocast("cuda", dtype=torch.float32):
                return self.action_model(vl_embs_list_repeated, top_target_repeated, state_repeated)

        if self.top_policy_name == "QwenOFT":
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(last_hidden, input_ids)
            with torch.autocast("cuda", dtype=torch.float32):
                pred_top = self.action_model.predict_action(action_queries)
            return F.l1_loss(pred_top, top_target)

        raise NotImplementedError(f"Unsupported top policy `{self.top_policy_name}`")

    def _prepare_targets(self, actions_np, action_valid_mask_np, device, dtype):
        dense_horizon = self.hierarchical_action_head.dense_action_horizon
        actions = torch.tensor(np.array(actions_np), device=device, dtype=dtype)
        dense_target = _slice_or_pad_actions(actions, dense_horizon)
        dense_target_valid = None
        if action_valid_mask_np is not None:
            valid = torch.tensor(np.array(action_valid_mask_np), device=device, dtype=torch.bool)
            dense_target_valid = _slice_or_pad_mask(valid, dense_horizon)
        top_target = linear_resample(dense_target, self.top_action_horizon)
        return dense_target, top_target, dense_target_valid

    def _build_dense_coarse_trajectory(self, top_plan: torch.Tensor, apply_corruption: bool) -> torch.Tensor:
        dense_coarse = linear_resample(top_plan, self.hierarchical_action_head.dense_action_horizon)
        if apply_corruption:
            # Keep the query/init coarse action unmasked. The refiner can still see
            # masked coarse-plan tokens through its temporal context.
            dense_coarse, _ = self.hierarchical_action_head.coarse_plan_corruptor(
                dense_coarse,
                apply_mask=False,
            )
        return dense_coarse

    def _make_lower_flow_step_size(
        self,
        top_plan_step_scale: torch.Tensor | float,
        batch_size: int,
        device,
        dtype,
    ) -> torch.Tensor:
        if not self.training and hasattr(self, "eval_lower_assumed_step_scale"):
            top_plan_step_scale = float(self.eval_lower_assumed_step_scale)
        if not torch.is_tensor(top_plan_step_scale):
            top_plan_step_scale = torch.tensor(top_plan_step_scale, device=device, dtype=dtype)
        else:
            top_plan_step_scale = top_plan_step_scale.to(device=device, dtype=dtype)
        flow_step = 1.0 - top_plan_step_scale
        if flow_step.dim() == 0:
            flow_step = flow_step.expand(batch_size)
        elif flow_step.shape[0] != batch_size:
            flow_step = flow_step.reshape(-1)
            if flow_step.numel() == 1:
                flow_step = flow_step.expand(batch_size)
            elif batch_size % flow_step.numel() == 0:
                flow_step = flow_step.repeat_interleave(batch_size // flow_step.numel())
            else:
                flow_step = flow_step[:batch_size]
        return flow_step

    def _build_lower_flow_path_input(
        self,
        dense_base_coarse: torch.Tensor,
        dense_target: torch.Tensor | None,
        base_scale: torch.Tensor | float,
        lower_t: torch.Tensor | float,
    ) -> torch.Tensor:
        if not self.training or not self.lower_flow_path_mix_enabled or dense_target is None:
            return dense_base_coarse

        device = dense_base_coarse.device
        dtype = dense_base_coarse.dtype
        batch_size = dense_base_coarse.shape[0]
        if not torch.is_tensor(base_scale):
            base_scale = torch.tensor(base_scale, device=device, dtype=dtype)
        else:
            base_scale = base_scale.to(device=device, dtype=dtype)
        if not torch.is_tensor(lower_t):
            lower_t = torch.tensor(lower_t, device=device, dtype=dtype)
        else:
            lower_t = lower_t.to(device=device, dtype=dtype)

        if base_scale.dim() == 0:
            base_scale = base_scale.expand(batch_size)
        else:
            base_scale = base_scale.reshape(-1)
            if base_scale.numel() == 1:
                base_scale = base_scale.expand(batch_size)
        if lower_t.dim() == 0:
            lower_t = lower_t.expand(batch_size)
        else:
            lower_t = lower_t.reshape(-1)
            if lower_t.numel() == 1:
                lower_t = lower_t.expand(batch_size)

        denom = (1.0 - base_scale).clamp_min(torch.finfo(dtype).eps)
        alpha = ((lower_t - base_scale) / denom).clamp(min=0.0, max=1.0).view(batch_size, 1, 1)
        return dense_base_coarse + alpha * (dense_target - dense_base_coarse)

    def _run_lower_refinement_for_eval(
        self,
        init_guess,
        top_hidden_features,
        batch_images,
        state,
        coarse_context,
        coarse_context_valid,
        chunk_start_indices,
        flow_step_size,
    ) -> dict:
        refine_steps = max(1, int(getattr(self, "eval_lower_refine_steps", 1)))
        current_init = init_guess
        total_step_size = flow_step_size
        step_size = total_step_size
        if refine_steps > 1:
            step_size = total_step_size / refine_steps

        refine_outputs = None
        start_timestep = 1.0 - total_step_size
        precomputed_context = None
        precomputed_context_info = None
        if refine_steps > 1 and getattr(self.hierarchical_action_head, "use_dit_refiner", False):
            precomputed_context, precomputed_context_info = (
                self.hierarchical_action_head.build_condition_tokens_for_eval(
                    top_hidden_features=top_hidden_features,
                    batch_images=batch_images,
                    state=None,
                    coarse_context=coarse_context,
                    coarse_context_valid=coarse_context_valid,
                )
            )
        for step_idx in range(refine_steps):
            current_timestep = start_timestep + step_size * step_idx
            refine_outputs = self.hierarchical_action_head(
                init_guess=current_init,
                top_hidden_features=top_hidden_features,
                batch_images=batch_images,
                state=state,
                dense_target=None,
                coarse_context=coarse_context,
                coarse_context_valid=coarse_context_valid,
                chunk_start_indices=chunk_start_indices,
                flow_step_size=step_size,
                flow_timestep=current_timestep,
                precomputed_context=precomputed_context,
                precomputed_context_info=precomputed_context_info,
            )
            current_init = refine_outputs["refined_actions"]
        return refine_outputs

    def _extend_dense_coarse_tail(self, dense_coarse: torch.Tensor) -> torch.Tensor:
        tail_steps = max(0, self.coarse_plan_tail_steps)
        if tail_steps == 0:
            return dense_coarse
        if dense_coarse.shape[1] == 0:
            return dense_coarse

        last = dense_coarse[:, -1:, :]
        if self.coarse_plan_tail_mode in {"repeat", "repeat_last", "hold"} or dense_coarse.shape[1] < 2:
            tail = last.expand(-1, tail_steps, -1)
        elif self.coarse_plan_tail_mode in {"linear", "linear_decay", "velocity"}:
            prev = dense_coarse[:, -2:-1, :]
            delta = last - prev
            decay = max(0.0, min(1.0, self.coarse_plan_tail_decay))
            values = []
            current = last
            for step_idx in range(tail_steps):
                step_delta = delta * (decay ** step_idx)
                current = current + step_delta
                values.append(current)
            tail = torch.cat(values, dim=1)
        else:
            raise ValueError(
                f"Unsupported coarse_plan_tail_mode={self.coarse_plan_tail_mode!r}. "
                "Use one of: repeat_last, linear_decay."
            )
        return torch.cat([dense_coarse, tail], dim=1)

    def _reset_inference_cache(self):
        self._inference_cache = {
            "signature": None,
            "top_plan": None,
            "dense_coarse": None,
            "dense_context_coarse": None,
            "top_features": None,
            "state_tensor": None,
            "top_plan_step_scale": None,
            "chunk_idx": 0,
        }

    def reset(self, **kwargs):
        self._reset_inference_cache()
        return None

    def _get_chunk_start_offsets(self, device) -> torch.Tensor:
        num_chunks = self.hierarchical_action_head.num_refine_chunks
        chunk_horizon = self.hierarchical_action_head.chunk_action_horizon
        return torch.arange(num_chunks, device=device, dtype=torch.long) * chunk_horizon

    def _sample_training_chunk_indices(self, device) -> torch.Tensor | None:
        sample_count = int(getattr(self.hierarchical_action_head, "train_num_sampled_chunks", 0) or 0)
        num_chunks = self.hierarchical_action_head.num_refine_chunks
        if (not self.training) or sample_count <= 0 or sample_count >= num_chunks:
            return None
        return torch.randperm(num_chunks, device=device)[:sample_count].sort().values

    def _context_execution_horizon(self) -> int:
        head = self.hierarchical_action_head
        return head.num_refine_chunks * head.chunk_action_horizon

    def _select_anchor_item(self, sample_items, chunk_idx: int):
        if sample_items is None:
            return None
        if sample_items and isinstance(sample_items[0], (list, tuple)):
            return sample_items[min(chunk_idx, len(sample_items) - 1)]
        return sample_items

    def _select_state_anchor(self, sample_state, chunk_idx: int):
        if sample_state is None:
            return None
        sample_arr = np.asarray(sample_state)
        if sample_arr.ndim == 2:
            anchor = sample_arr[min(chunk_idx, sample_arr.shape[0] - 1)]
            return np.expand_dims(anchor, axis=0)
        return sample_arr

    def _sample_temporal_shift(self, chunk_idx: int) -> int:
        head = self.hierarchical_action_head
        if not self.training or not head.temporal_shift_enabled:
            return 0
        max_shift = head.temporal_shift_max(chunk_idx)
        if max_shift <= 0:
            return 0
        return int(torch.randint(-max_shift, max_shift + 1, (1,)).item())

    def _build_chunk_batch(
        self,
        refine_images,
        refine_state,
        dense_target,
        dense_target_valid,
        dense_coarse,
        top_features,
        dense_context_coarse=None,
        apply_temporal_shift: bool = True,
        chunk_indices: torch.Tensor | None = None,
        dense_target_velocity=None,
    ):
        total_chunks = self.hierarchical_action_head.num_refine_chunks
        chunk_horizon = self.hierarchical_action_head.chunk_action_horizon
        context_extra = self.hierarchical_action_head.temporal_context_extra
        chunk_starts = self._get_chunk_start_offsets(dense_coarse.device)
        if chunk_indices is None:
            chunk_indices = torch.arange(total_chunks, device=dense_coarse.device, dtype=torch.long)
        else:
            chunk_indices = chunk_indices.to(device=dense_coarse.device, dtype=torch.long)
        num_chunks = int(chunk_indices.numel())
        if num_chunks <= 0:
            raise ValueError("chunk_indices must contain at least one chunk.")

        flat_images = []
        flat_states = []
        chunk_targets = []
        chunk_target_velocity = []
        chunk_target_valid = []
        chunk_inits = []
        chunk_contexts = []
        chunk_context_targets = []
        chunk_context_valid = []
        chunk_start_values = []

        if dense_context_coarse is None:
            dense_context_coarse = dense_coarse

        batch_size = dense_coarse.shape[0]
        init_dense_coarse = self._extend_dense_coarse_tail(dense_coarse)
        max_start = max(init_dense_coarse.shape[1] - chunk_horizon, 0)
        max_context_start = max(dense_context_coarse.shape[1] - 1, 0)
        for batch_idx in range(batch_size):
            for chunk_idx_tensor in chunk_indices:
                chunk_idx = int(chunk_idx_tensor.item())
                chunk_start = int(chunk_starts[chunk_idx].item())
                shift = self._sample_temporal_shift(chunk_idx) if apply_temporal_shift else 0
                source_start = max(0, min(chunk_start + shift, max_start))
                context_source_start = max(0, min(chunk_start + shift, max_context_start))
                flat_images.append(self._select_anchor_item(refine_images[batch_idx], chunk_idx))
                if refine_state is not None:
                    flat_states.append(self._select_state_anchor(refine_state[batch_idx], chunk_idx))
                if dense_target is not None:
                    chunk_targets.append(
                        dense_target[batch_idx, chunk_start : chunk_start + chunk_horizon]
                    )
                    if dense_target_velocity is not None:
                        chunk_target_velocity.append(
                            dense_target_velocity[batch_idx, chunk_start : chunk_start + chunk_horizon]
                        )
                    if dense_target_valid is not None:
                        chunk_target_valid.append(
                            dense_target_valid[batch_idx, chunk_start : chunk_start + chunk_horizon]
                        )
                source_start_tensor = torch.tensor(
                    [source_start],
                    device=dense_coarse.device,
                    dtype=torch.long,
                )
                init_guess = gather_temporal_chunk(
                    init_dense_coarse[batch_idx : batch_idx + 1],
                    source_start_tensor,
                    chunk_horizon,
                )
                context_source_start_tensor = torch.tensor(
                    [context_source_start],
                    device=dense_context_coarse.device,
                    dtype=torch.long,
                )
                coarse_context, coarse_valid, context_offsets = gather_temporal_context(
                    dense_context_coarse[batch_idx : batch_idx + 1],
                    context_source_start_tensor,
                    chunk_horizon,
                    context_extra,
                )
                if dense_target is not None:
                    target_context, _, _ = gather_temporal_context(
                        dense_target[batch_idx : batch_idx + 1],
                        source_start_tensor,
                        chunk_horizon,
                        context_extra,
                    )
                    chunk_context_targets.append(target_context.squeeze(0))
                context_raw_index = context_source_start_tensor.view(-1, 1) + context_offsets.view(1, -1)
                coarse_valid = coarse_valid & (context_raw_index < self._context_execution_horizon())
                chunk_inits.append(init_guess.squeeze(0))
                chunk_contexts.append(coarse_context.squeeze(0))
                chunk_context_valid.append(coarse_valid.squeeze(0))
                chunk_start_values.append(source_start)

        repeated_top_features = top_features.repeat_interleave(num_chunks, dim=0)
        init_tensor = torch.stack(chunk_inits, dim=0)
        context_tensor = torch.stack(chunk_contexts, dim=0)
        context_valid_tensor = torch.stack(chunk_context_valid, dim=0)
        chunk_start_tensor = torch.tensor(chunk_start_values, device=dense_coarse.device, dtype=torch.long)

        state_tensor = None
        if refine_state is not None:
            state_tensor = torch.tensor(np.array(flat_states), device=dense_coarse.device, dtype=dense_coarse.dtype)

        target_tensor = None
        target_valid_tensor = None
        context_target_tensor = None
        target_velocity_tensor = None
        if dense_target is not None:
            target_tensor = torch.stack(chunk_targets, dim=0)
            if dense_target_velocity is not None:
                target_velocity_tensor = torch.stack(chunk_target_velocity, dim=0)
            if dense_target_valid is not None:
                target_valid_tensor = torch.stack(chunk_target_valid, dim=0)
            context_target_tensor = torch.stack(chunk_context_targets, dim=0)

        return (
            init_tensor,
            repeated_top_features,
            flat_images,
            state_tensor,
            target_tensor,
            target_valid_tensor,
            context_tensor,
            context_target_tensor,
            context_valid_tensor,
            chunk_start_tensor,
            target_velocity_tensor,
        )

    def _merge_chunk_predictions(self, chunk_actions: torch.Tensor, batch_size: int) -> torch.Tensor:
        num_chunks = self.hierarchical_action_head.num_refine_chunks
        chunk_horizon = self.hierarchical_action_head.chunk_action_horizon
        return chunk_actions.view(batch_size, num_chunks, chunk_horizon, self.action_dim).reshape(
            batch_size, num_chunks * chunk_horizon, self.action_dim
        )

    def forward(self, examples: List[dict] = None, **kwargs):
        self._maybe_initialize_lower_from_top_action_head()
        (
            top_images,
            refine_images,
            instructions,
            actions_np,
            action_valid_mask_np,
            top_state,
            refine_state,
        ) = self._prepare_examples(
            examples, resize_for_infer=False
        )
        self._debug_log_data_flow(top_images, refine_images, instructions, actions_np, top_state, refine_state)
        (
            top_plan,
            top_features,
            state_tensor,
            qwen_inputs,
            last_hidden,
            all_hidden,
            context_top_plan,
            top_plan_step_scale,
        ) = self._compute_top_policy_outputs(top_images, instructions, top_state)
        dense_target, top_target, dense_target_valid = self._prepare_targets(
            actions_np,
            action_valid_mask_np,
            top_plan.device,
            top_plan.dtype,
        )
        self._debug_log_top_policy(top_plan, top_features, top_target, dense_target, state_tensor)

        top_loss = self._compute_top_loss(top_target, top_features, state_tensor, qwen_inputs, last_hidden, all_hidden)
        dense_base_coarse = self._build_dense_coarse_trajectory(top_plan, apply_corruption=True)
        lower_flow_t = self._sample_lower_flow_path_t(
            top_plan_step_scale,
            batch_size=dense_base_coarse.shape[0],
            device=dense_base_coarse.device,
            dtype=dense_base_coarse.dtype,
        )
        dense_coarse = self._build_lower_flow_path_input(
            dense_base_coarse=dense_base_coarse,
            dense_target=dense_target,
            base_scale=top_plan_step_scale,
            lower_t=lower_flow_t,
        )
        dense_context_coarse = (
            self._build_dense_coarse_trajectory(context_top_plan, apply_corruption=True)
            if context_top_plan is not None
            else dense_base_coarse
        )
        # Flow-velocity target anchored at the UPSTREAM plan over the full remaining span
        # (1 - base_scale), rather than at the refiner's input x_t over (1 - t). Identical
        # on the straight path, but finite and well-conditioned as t -> 1.
        dense_target_velocity = None
        if (
            self.lower_velocity_target_from_upstream
            and dense_target is not None
            and getattr(self.hierarchical_action_head, "refine_prediction_type", None)
            in {"flow", "flow_velocity", "velocity"}
        ):
            base_span = (1.0 - top_plan_step_scale.reshape(-1).to(dense_target.dtype))
            base_span = base_span.clamp_min(torch.finfo(dense_target.dtype).eps)
            if base_span.numel() == 1:
                base_span = base_span.expand(dense_target.shape[0])
            dense_target_velocity = (dense_target - dense_base_coarse) / base_span.view(-1, 1, 1)

        (
            chunk_init_tensor,
            chunk_top_features,
            chunk_images,
            chunk_state_tensor,
            chunk_target_tensor,
            chunk_target_valid_tensor,
            chunk_context_tensor,
            chunk_context_target_tensor,
            chunk_context_valid_tensor,
            chunk_start_tensor,
            chunk_target_velocity_tensor,
        ) = (
            self._build_chunk_batch(
                refine_images=refine_images,
                refine_state=refine_state,
                dense_target=dense_target,
                dense_target_valid=dense_target_valid,
                dense_coarse=dense_coarse,
                dense_context_coarse=dense_context_coarse,
                top_features=top_features,
                apply_temporal_shift=True,
                chunk_indices=self._sample_training_chunk_indices(dense_coarse.device),
                dense_target_velocity=dense_target_velocity,
            )
        )
        refine_outputs = self.hierarchical_action_head(
            init_guess=chunk_init_tensor,
            top_hidden_features=chunk_top_features,
            batch_images=chunk_images,
            state=chunk_state_tensor,
            dense_target=chunk_target_tensor,
            dense_target_valid=chunk_target_valid_tensor,
            coarse_context=chunk_context_tensor,
            coarse_context_target=chunk_context_target_tensor,
            coarse_context_valid=chunk_context_valid_tensor,
            chunk_start_indices=chunk_start_tensor,
            target_velocity=chunk_target_velocity_tensor,
            flow_step_size=self._make_lower_flow_step_size(
                lower_flow_t,
                batch_size=chunk_init_tensor.shape[0],
                device=chunk_init_tensor.device,
                dtype=chunk_init_tensor.dtype,
            ),
        )
        refine_loss = refine_outputs["refine_loss"]
        total_loss = self.top_loss_weight * top_loss + self.refine_loss_weight * refine_loss

        output_dict = {
            "action_loss": total_loss,
            "top_loss": top_loss.detach(),
            "refine_loss": refine_loss.detach(),
        }
        if "residual_bound_loss" in refine_outputs:
            output_dict["residual_bound_loss"] = refine_outputs["residual_bound_loss"].detach()
        if "future_context_residual_loss" in refine_outputs:
            output_dict["future_context_residual_loss"] = refine_outputs["future_context_residual_loss"].detach()
        if "action_valid_ratio" in refine_outputs:
            output_dict["action_valid_ratio"] = refine_outputs["action_valid_ratio"].detach()
        return output_dict

    def _predict_dense_sequence_for_eval(self, examples: List[dict]):
        top_images, refine_images, instructions, _, _, top_state, refine_state = self._prepare_examples(
            examples, resize_for_infer=True
        )
        (
            top_plan,
            top_features,
            _state_tensor,
            _qwen_inputs,
            _last_hidden,
            _all_hidden,
            context_top_plan,
            top_plan_step_scale,
        ) = self._compute_top_policy_outputs(top_images, instructions, top_state)
        dense_coarse = self._build_dense_coarse_trajectory(top_plan, apply_corruption=False)
        dense_context_coarse = (
            self._build_dense_coarse_trajectory(context_top_plan, apply_corruption=False)
            if context_top_plan is not None
            else dense_coarse
        )
        (
            chunk_init_tensor,
            chunk_top_features,
            chunk_images,
            chunk_state_tensor,
            _,
            _,
            chunk_context_tensor,
            _,
            chunk_context_valid_tensor,
            chunk_start_tensor,
            _,
        ) = (
            self._build_chunk_batch(
                refine_images=refine_images,
                refine_state=refine_state,
                dense_target=None,
                dense_target_valid=None,
                dense_coarse=dense_coarse,
                dense_context_coarse=dense_context_coarse,
                top_features=top_features,
                apply_temporal_shift=False,
            )
        )
        flow_step_size = self._make_lower_flow_step_size(
            top_plan_step_scale,
            batch_size=chunk_init_tensor.shape[0],
            device=chunk_init_tensor.device,
            dtype=chunk_init_tensor.dtype,
        )
        refine_outputs = self._run_lower_refinement_for_eval(
            init_guess=chunk_init_tensor,
            top_hidden_features=chunk_top_features,
            batch_images=chunk_images,
            state=chunk_state_tensor,
            coarse_context=chunk_context_tensor,
            coarse_context_valid=chunk_context_valid_tensor,
            chunk_start_indices=chunk_start_tensor,
            flow_step_size=flow_step_size,
        )
        refined = self._merge_chunk_predictions(refine_outputs["refined_actions"], batch_size=top_plan.shape[0])
        return {"normalized_actions": refined.detach().float().cpu().numpy()}

    def _predict_chunk_online(self, examples: List[dict]):
        top_images, refine_images, instructions, _, _, top_state, refine_state = self._prepare_examples(
            examples, resize_for_infer=True
        )
        signature = tuple(instructions)
        num_chunks = 1 if self.eval_action_mode == "lower_first8" else self.hierarchical_action_head.eval_num_chunks
        cache_miss = self._inference_cache["signature"] != signature
        need_top_refresh = cache_miss or self._inference_cache["chunk_idx"] % num_chunks == 0

        if need_top_refresh:
            (
                top_plan,
                top_features,
                state_tensor,
                _qwen_inputs,
                _last_hidden,
                _all_hidden,
                context_top_plan,
                top_plan_step_scale,
            ) = self._compute_top_policy_outputs(top_images, instructions, top_state)
            dense_coarse = self._build_dense_coarse_trajectory(top_plan.detach(), apply_corruption=False).detach()
            dense_context_coarse = (
                self._build_dense_coarse_trajectory(context_top_plan.detach(), apply_corruption=False).detach()
                if context_top_plan is not None
                else dense_coarse
            )
            self._inference_cache = {
                "signature": signature,
                "top_plan": top_plan.detach(),
                "dense_coarse": dense_coarse,
                "dense_context_coarse": dense_context_coarse,
                "top_features": top_features.detach(),
                "state_tensor": state_tensor.detach() if state_tensor is not None else None,
                "top_plan_step_scale": top_plan_step_scale.detach(),
                "chunk_idx": 0,
            }

        chunk_idx = int(self._inference_cache["chunk_idx"])
        chunk_start = chunk_idx * self.hierarchical_action_head.chunk_action_horizon
        chunk_images = []
        for sample_images in refine_images:
            chunk_images.append(self._select_anchor_item(sample_images, chunk_idx))

        chunk_state_tensor = None
        if refine_state is not None:
            flat_states = [self._select_state_anchor(sample_state, chunk_idx) for sample_state in refine_state]
            chunk_state_tensor = torch.tensor(
                np.array(flat_states),
                device=self._inference_cache["top_plan"].device,
                dtype=self._inference_cache["top_plan"].dtype,
            )

        chunk_start_tensor = torch.full(
            (self._inference_cache["dense_coarse"].shape[0],),
            chunk_start,
            device=self._inference_cache["dense_coarse"].device,
            dtype=torch.long,
        )
        chunk_init = gather_temporal_chunk(
            self._inference_cache["dense_coarse"],
            chunk_start_tensor,
            self.hierarchical_action_head.chunk_action_horizon,
        )
        coarse_context, coarse_context_valid, context_offsets = gather_temporal_context(
            self._inference_cache.get("dense_context_coarse", self._inference_cache["dense_coarse"]),
            chunk_start_tensor,
            self.hierarchical_action_head.chunk_action_horizon,
            self.hierarchical_action_head.temporal_context_extra,
        )
        context_raw_index = chunk_start_tensor.view(-1, 1) + context_offsets.view(1, -1)
        coarse_context_valid = coarse_context_valid & (context_raw_index < self._context_execution_horizon())
        flow_step_size = self._make_lower_flow_step_size(
            self._inference_cache["top_plan_step_scale"],
            batch_size=chunk_init.shape[0],
            device=chunk_init.device,
            dtype=chunk_init.dtype,
        )
        refine_outputs = self._run_lower_refinement_for_eval(
            init_guess=chunk_init,
            top_hidden_features=self._inference_cache["top_features"],
            batch_images=chunk_images,
            state=chunk_state_tensor,
            coarse_context=coarse_context,
            coarse_context_valid=coarse_context_valid,
            chunk_start_indices=chunk_start_tensor,
            flow_step_size=flow_step_size,
        )
        self._inference_cache["chunk_idx"] = (chunk_idx + 1) % num_chunks
        normalized_actions = refine_outputs["refined_actions"].detach().float().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    def _predict_top_actions_for_eval(self, examples: List[dict], horizon: int):
        top_images, _, instructions, _, _, top_state, _ = self._prepare_examples(
            examples, resize_for_infer=True
        )
        top_plan, _, _, _, _, _, _, _ = self._compute_top_policy_outputs(top_images, instructions, top_state)
        normalized_actions = top_plan[:, :horizon].detach().float().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    def _predict_lower_first_actions_for_eval(self, examples: List[dict], horizon: int):
        output = self._predict_dense_sequence_for_eval(examples)
        output["normalized_actions"] = output["normalized_actions"][:, :horizon]
        return output

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs):
        if not isinstance(examples, list):
            examples = [examples]
        if examples and "action" in examples[0]:
            return self._predict_dense_sequence_for_eval(examples)
        if self.eval_action_mode == "lower_first16":
            return self._predict_lower_first_actions_for_eval(
                examples,
                horizon=2 * self.hierarchical_action_head.chunk_action_horizon,
            )
        if self.eval_action_mode == "top_first8":
            return self._predict_top_actions_for_eval(
                examples,
                horizon=self.hierarchical_action_head.chunk_action_horizon,
            )
        if self.eval_action_mode == "top32":
            return self._predict_top_actions_for_eval(
                examples,
                horizon=self.hierarchical_action_head.dense_action_horizon,
            )
        if self.eval_action_mode == "top_custom":
            return self._predict_top_actions_for_eval(
                examples,
                horizon=int(getattr(self, "eval_top_horizon", self.hierarchical_action_head.chunk_action_horizon)),
            )
        return self._predict_chunk_online(examples)
