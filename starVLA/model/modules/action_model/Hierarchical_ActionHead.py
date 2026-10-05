"""Hierarchical action refinement head for StarVLA.

This module refines a coarse top-policy action plan into a denser action
sequence using:
  - compact top-policy semantic tokens
  - the coarse action plan itself
  - lightweight real-time observation tokens
  - optional state tokens

The design intentionally avoids consuming raw high-resolution backbone patch
tokens so it stays lightweight and adaptable across different top policies.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torchvision.models import resnet18
from torchvision.transforms.functional import pil_to_tensor
from transformers import AutoImageProcessor, SiglipVisionModel

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.modules.action_model.GR00T_ActionHeader import (
    ActionEncoder as FlowActionEncoder,
    MLP as FlowMLP,
)
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT


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


def linear_resample(seq: Tensor, target_len: int) -> Tensor:
    if seq.shape[1] == target_len:
        return seq
    x = seq.transpose(1, 2)
    x = F.interpolate(x, size=target_len, mode="linear", align_corners=True)
    return x.transpose(1, 2)


def gather_temporal_chunk(seq: Tensor, start_indices: Tensor, chunk_len: int) -> Tensor:
    if seq.dim() != 3:
        raise ValueError(f"Expected 3D tensor [B, T, D], got shape {tuple(seq.shape)}")
    if start_indices.dim() != 1 or start_indices.shape[0] != seq.shape[0]:
        raise ValueError(
            f"Expected start_indices shape [{seq.shape[0]}], got {tuple(start_indices.shape)}"
        )
    max_start = max(seq.shape[1] - chunk_len, 0)
    start_indices = start_indices.clamp(min=0, max=max_start)
    offsets = torch.arange(chunk_len, device=seq.device).view(1, chunk_len)
    gather_index = start_indices.view(-1, 1) + offsets
    gather_index = gather_index.unsqueeze(-1).expand(-1, -1, seq.shape[-1])
    return seq.gather(dim=1, index=gather_index)


def gather_temporal_context(
    seq: Tensor,
    start_indices: Tensor,
    chunk_len: int,
    extra_len: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if seq.dim() != 3:
        raise ValueError(f"Expected 3D tensor [B, T, D], got shape {tuple(seq.shape)}")
    if start_indices.dim() != 1 or start_indices.shape[0] != seq.shape[0]:
        raise ValueError(
            f"Expected start_indices shape [{seq.shape[0]}], got {tuple(start_indices.shape)}"
        )
    if extra_len < 0:
        raise ValueError(f"extra_len must be non-negative, got {extra_len}")

    offsets = torch.arange(-extra_len, chunk_len + extra_len, device=seq.device, dtype=torch.long)
    raw_index = start_indices.view(-1, 1) + offsets.view(1, -1)
    valid_mask = (raw_index >= 0) & (raw_index < seq.shape[1])
    safe_index = raw_index.clamp(min=0, max=seq.shape[1] - 1)
    gather_index = safe_index.unsqueeze(-1).expand(-1, -1, seq.shape[-1])
    return seq.gather(dim=1, index=gather_index), valid_mask, offsets


def get_1d_sincos_pos_embed_torch(
    embed_dim: int,
    positions: Tensor,
    dtype: torch.dtype,
) -> Tensor:
    even_dim = embed_dim if embed_dim % 2 == 0 else embed_dim + 1
    omega = torch.arange(even_dim // 2, device=positions.device, dtype=torch.float32)
    omega = omega / (even_dim / 2.0)
    omega = 1.0 / (10000**omega)
    out = positions.float().reshape(-1, 1) * omega.reshape(1, -1)
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    return emb[:, :embed_dim].to(dtype=dtype)


class CoarsePlanCorruptor(nn.Module):
    def __init__(self, action_dim: int, cfg) -> None:
        super().__init__()
        self.noise_std = float(_cfg_get(cfg, "coarse_plan_noise_std", 0.0))
        self.noise_type = str(_cfg_get(cfg, "coarse_plan_noise_type", "gaussian")).lower()
        self.sine_noise_amp_min = float(_cfg_get(cfg, "coarse_plan_sine_noise_amp_min", 0.0))
        self.sine_noise_amp_max = float(_cfg_get(cfg, "coarse_plan_sine_noise_amp_max", self.noise_std))
        self.sine_noise_freq_min = float(_cfg_get(cfg, "coarse_plan_sine_noise_freq_min", 0.5))
        self.sine_noise_freq_max = float(_cfg_get(cfg, "coarse_plan_sine_noise_freq_max", 2.0))
        self.sine_noise_per_dim = bool(_cfg_get(cfg, "coarse_plan_sine_noise_per_dim", True))
        self.sine_noise_zero_mean = bool(_cfg_get(cfg, "coarse_plan_sine_noise_zero_mean", True))
        self.mask_ratio = float(_cfg_get(cfg, "coarse_plan_mask_ratio", 0.0))
        self.mask_token = nn.Parameter(torch.zeros(action_dim))

    def _sample_temporal_mask(self, seq: Tensor) -> Tensor | None:
        if self.mask_ratio <= 0:
            return None
        return torch.rand(seq.size(0), seq.size(1), device=seq.device) < self.mask_ratio

    def apply_context_mask(self, coarse_context: Tensor) -> tuple[Tensor, Tensor | None]:
        if not self.training:
            return coarse_context, None
        temporal_mask = self._sample_temporal_mask(coarse_context)
        if temporal_mask is None:
            return coarse_context, None
        token_mask = temporal_mask.unsqueeze(-1)
        mask_token = self.mask_token.view(1, 1, -1).expand_as(coarse_context)
        return torch.where(token_mask, mask_token, coarse_context), temporal_mask

    def _add_gaussian_noise(self, seq: Tensor) -> Tensor:
        if self.noise_std <= 0:
            return seq
        return seq + self.noise_std * torch.randn_like(seq)

    def _add_sinusoidal_noise(self, seq: Tensor) -> Tensor:
        amp_max = max(self.sine_noise_amp_min, self.sine_noise_amp_max)
        if amp_max <= 0:
            return seq

        batch_size, horizon, action_dim = seq.shape
        amp_min = min(self.sine_noise_amp_min, amp_max)
        noise_dim = action_dim if self.sine_noise_per_dim else 1
        amp = torch.empty(batch_size, 1, noise_dim, device=seq.device, dtype=seq.dtype).uniform_(amp_min, amp_max)
        freq = torch.empty(batch_size, 1, noise_dim, device=seq.device, dtype=seq.dtype).uniform_(
            self.sine_noise_freq_min,
            self.sine_noise_freq_max,
        )
        phase = torch.empty(batch_size, 1, noise_dim, device=seq.device, dtype=seq.dtype).uniform_(
            0.0,
            6.283185307179586,
        )
        t = torch.linspace(0.0, 1.0, horizon, device=seq.device, dtype=seq.dtype).view(1, horizon, 1)
        noise = amp * torch.sin(6.283185307179586 * freq * t + phase)
        if self.sine_noise_zero_mean:
            noise = noise - noise.mean(dim=1, keepdim=True)
        if noise_dim == 1:
            noise = noise.expand(-1, -1, action_dim)
        return seq + noise

    def _add_noise(self, seq: Tensor) -> Tensor:
        if self.noise_type in {"none", "off", "false"}:
            return seq
        if self.noise_type in {"gaussian", "normal"}:
            return self._add_gaussian_noise(seq)
        if self.noise_type in {"sine", "sin", "sinusoidal"}:
            return self._add_sinusoidal_noise(seq)
        if self.noise_type in {"mixed", "gaussian+sine", "sine+gaussian"}:
            return self._add_sinusoidal_noise(self._add_gaussian_noise(seq))
        raise ValueError(
            f"Unsupported coarse_plan_noise_type={self.noise_type!r}. "
            "Use one of: gaussian, sinusoidal, mixed, none."
        )

    def forward(self, coarse_trajectory: Tensor, apply_mask: bool = True) -> tuple[Tensor, Tensor | None]:
        if not self.training:
            return coarse_trajectory, None

        out = self._add_noise(coarse_trajectory)

        if not apply_mask:
            return out, None
        return self.apply_context_mask(out)


class RealtimeObservationTokenizer(nn.Module):
    def __init__(self, hidden_dim: int, cfg) -> None:
        super().__init__()
        self.encoder_type = str(_cfg_get(cfg, "realtime_obs_encoder", "cnn")).lower()
        self.target_size = int(_cfg_get(cfg, "realtime_obs_size", 96))
        mid_channels = int(_cfg_get(cfg, "realtime_obs_channels", 64))
        self.token_grid_size = int(_cfg_get(cfg, "realtime_obs_token_grid", 6))
        self.freeze_vision = False
        self.conv = None
        self.resnet = None
        self.resnet_proj = None
        self.image_processor = None
        self.vision_model = None
        self.vision_proj = None

        if self.encoder_type in {"cnn", "conv", "tiny_cnn"}:
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
                nn.GELU(),
                nn.Conv2d(32, mid_channels, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(mid_channels, hidden_dim, kernel_size=1),
            )
        elif self.encoder_type in {"resnet18", "resnet18_spatial", "resnet"}:
            stage = int(_cfg_get(cfg, "realtime_obs_resnet_stage", 3))
            if stage not in {3, 4}:
                raise ValueError(f"realtime_obs_resnet_stage must be 3 or 4, got {stage}")
            backbone = resnet18(weights=None)
            layers = [
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
            ]
            out_channels = 256
            if stage == 4:
                layers.append(backbone.layer4)
                out_channels = 512
            self.resnet = nn.Sequential(*layers)
            self.resnet_proj = nn.Conv2d(out_channels, hidden_dim, kernel_size=1)
        elif self.encoder_type in {"siglip", "siglip_vision"}:
            model_id = str(_cfg_get(cfg, "realtime_obs_siglip_model_id", "google/siglip-large-patch16-384"))
            dtype_name = str(_cfg_get(cfg, "realtime_obs_siglip_dtype", "bf16")).lower()
            model_dtype = {
                "bf16": torch.bfloat16,
                "bfloat16": torch.bfloat16,
                "fp16": torch.float16,
                "float16": torch.float16,
                "fp32": torch.float32,
                "float32": torch.float32,
                "auto": "auto",
            }.get(dtype_name, torch.bfloat16)
            self.image_processor = AutoImageProcessor.from_pretrained(model_id)
            self.vision_model = SiglipVisionModel.from_pretrained(model_id, dtype=model_dtype)
            self.freeze_vision = bool(_cfg_get(cfg, "realtime_obs_siglip_freeze", True))
            self.siglip_interpolate_pos_encoding = bool(
                _cfg_get(cfg, "realtime_obs_siglip_interpolate_pos_encoding", True)
            )
            self.siglip_batch_size = int(_cfg_get(cfg, "realtime_obs_siglip_batch_size", 32) or 0)
            if self.freeze_vision:
                self.vision_model.eval()
                for param in self.vision_model.parameters():
                    param.requires_grad_(False)
            self.vision_proj = nn.Linear(int(self.vision_model.config.hidden_size), hidden_dim)
        else:
            raise ValueError(
                f"Unsupported realtime_obs_encoder={self.encoder_type!r}. "
                "Use one of: cnn, resnet18_spatial, siglip."
            )
        self.view_embed = nn.Embedding(int(_cfg_get(cfg, "max_camera_views", 8)), hidden_dim)
        self.anchor_embed = nn.Embedding(int(_cfg_get(cfg, "max_obs_anchors", 8)), hidden_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision and self.vision_model is not None:
            self.vision_model.eval()
        return self

    @staticmethod
    def _to_pil(image) -> Image.Image:
        pil = to_pil_preserve(image)
        if not isinstance(pil, Image.Image):
            pil = Image.fromarray(np.asarray(pil))
        return pil

    def _resize_to_tensor(self, image) -> Tensor:
        pil = self._to_pil(image)
        pil = pil.resize((self.target_size, self.target_size))
        return pil_to_tensor(pil).float() / 255.0

    def _encode_cnn_views(self, images: List) -> Tensor:
        device = next(self.conv.parameters()).device
        dtype = next(self.conv.parameters()).dtype
        sample_tensors = [self._resize_to_tensor(img) for img in images]
        sample_tensor = torch.stack(sample_tensors, dim=0).to(device=device, dtype=dtype)
        feats = self.conv(sample_tensor)
        feats = F.adaptive_avg_pool2d(feats, (self.token_grid_size, self.token_grid_size))
        return feats.flatten(2).transpose(1, 2)

    def _encode_resnet_views(self, images: List) -> Tensor:
        device = next(self.resnet_proj.parameters()).device
        dtype = next(self.resnet_proj.parameters()).dtype
        sample_tensors = [self._resize_to_tensor(img) for img in images]
        sample_tensor = torch.stack(sample_tensors, dim=0).to(device=device, dtype=dtype)
        feats = self.resnet(sample_tensor)
        feats = self.resnet_proj(feats)
        feats = F.adaptive_avg_pool2d(feats, (self.token_grid_size, self.token_grid_size))
        return feats.flatten(2).transpose(1, 2)

    def _pool_siglip_tokens(self, feats: Tensor) -> Tensor:
        token_count = feats.shape[1]
        grid = int(token_count**0.5)
        if grid * grid == token_count:
            feats_2d = feats.transpose(1, 2).reshape(feats.shape[0], feats.shape[2], grid, grid)
            feats = F.adaptive_avg_pool2d(feats_2d, (self.token_grid_size, self.token_grid_size))
            return feats.flatten(2).transpose(1, 2)
        target_tokens = self.token_grid_size * self.token_grid_size
        return F.adaptive_avg_pool1d(feats.transpose(1, 2), target_tokens).transpose(1, 2)

    def _encode_siglip_pil_images(self, pil_images: List[Image.Image]) -> Tensor:
        if not pil_images:
            raise ValueError("SigLIP encoder received no images")
        device = next(self.vision_proj.parameters()).device
        vision_dtype = next(self.vision_model.parameters()).dtype
        proj_dtype = next(self.vision_proj.parameters()).dtype
        micro_batch = self.siglip_batch_size if self.siglip_batch_size > 0 else len(pil_images)
        encoded_chunks = []
        for start in range(0, len(pil_images), micro_batch):
            image_chunk = pil_images[start : start + micro_batch]
            inputs = self.image_processor(images=image_chunk, return_tensors="pt", do_resize=False)
            pixel_values = inputs["pixel_values"].to(device=device, dtype=vision_dtype)
            grad_context = torch.no_grad() if self.freeze_vision else nullcontext()
            with grad_context:
                feats = self.vision_model(
                    pixel_values=pixel_values,
                    interpolate_pos_encoding=self.siglip_interpolate_pos_encoding,
                ).last_hidden_state
            encoded_chunks.append(self._pool_siglip_tokens(feats))
        feats = torch.cat(encoded_chunks, dim=0)
        return self.vision_proj(feats.to(dtype=proj_dtype))

    def _encode_siglip_views(self, images: List) -> Tensor:
        pil_images = [self._to_pil(img).resize((self.target_size, self.target_size)) for img in images]
        return self._encode_siglip_pil_images(pil_images)

    def _forward_siglip_batched(self, batch_images: List) -> Tensor:
        device = self.view_embed.weight.device
        flat_pil_images = []
        sample_specs = []
        for sample_images in batch_images:
            sample_spec = []
            if sample_images and isinstance(sample_images[0], (list, tuple)):
                for anchor_idx, anchor_images in enumerate(sample_images):
                    start = len(flat_pil_images)
                    flat_pil_images.extend(
                        self._to_pil(img).resize((self.target_size, self.target_size)) for img in anchor_images
                    )
                    sample_spec.append((start, len(anchor_images), anchor_idx, True))
            else:
                start = len(flat_pil_images)
                flat_pil_images.extend(
                    self._to_pil(img).resize((self.target_size, self.target_size)) for img in sample_images
                )
                sample_spec.append((start, len(sample_images), 0, False))
            sample_specs.append(sample_spec)

        encoded = self._encode_siglip_pil_images(flat_pil_images)
        batch_tokens = []
        for sample_spec in sample_specs:
            anchor_tokens = []
            for start, view_count, anchor_idx, use_anchor_embed in sample_spec:
                feats = encoded[start : start + view_count]
                view_ids = torch.arange(feats.size(0), device=device).clamp_max(self.view_embed.num_embeddings - 1)
                feats = feats + self.view_embed(view_ids)[:, None, :]
                if use_anchor_embed:
                    anchor_id = min(anchor_idx, self.anchor_embed.num_embeddings - 1)
                    feats = feats + self.anchor_embed.weight[anchor_id][None, None, :]
                anchor_tokens.append(feats.reshape(-1, feats.size(-1)))
            batch_tokens.append(torch.cat(anchor_tokens, dim=0))
        return torch.stack(batch_tokens, dim=0)

    def _encode_views(self, images: List) -> Tensor:
        if self.encoder_type in {"cnn", "conv", "tiny_cnn"}:
            return self._encode_cnn_views(images)
        if self.encoder_type in {"resnet18", "resnet18_spatial", "resnet"}:
            return self._encode_resnet_views(images)
        return self._encode_siglip_views(images)

    def forward(self, batch_images: List) -> Tensor:
        if not batch_images:
            raise ValueError("batch_images cannot be empty")
        if self.encoder_type in {"siglip", "siglip_vision"}:
            return self._forward_siglip_batched(batch_images)

        device = self.view_embed.weight.device
        batch_tokens = []
        for sample_images in batch_images:
            if sample_images and isinstance(sample_images[0], (list, tuple)):
                anchor_tokens = []
                for anchor_idx, anchor_images in enumerate(sample_images):
                    feats = self._encode_views(anchor_images)
                    view_ids = torch.arange(feats.size(0), device=device).clamp_max(self.view_embed.num_embeddings - 1)
                    anchor_id = min(anchor_idx, self.anchor_embed.num_embeddings - 1)
                    feats = (
                        feats
                        + self.view_embed(view_ids)[:, None, :]
                        + self.anchor_embed.weight[anchor_id][None, None, :]
                    )
                    anchor_tokens.append(feats.reshape(-1, feats.size(-1)))
                batch_tokens.append(torch.cat(anchor_tokens, dim=0))
            else:
                feats = self._encode_views(sample_images)
                view_ids = torch.arange(feats.size(0), device=device).clamp_max(self.view_embed.num_embeddings - 1)
                feats = feats + self.view_embed(view_ids)[:, None, :]
                batch_tokens.append(feats.reshape(-1, feats.size(-1)))
        return torch.stack(batch_tokens, dim=0)


class StateTokenizer(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = int(state_dim)
        self.proj = nn.Linear(state_dim, hidden_dim)
        self._mismatch_print_count = 0

    def forward(self, state: Tensor | None) -> Tensor | None:
        if state is None:
            return None
        if state.shape[-1] != self.input_dim:
            if self._mismatch_print_count < 3:
                print(
                    "[HierarchicalHead:state]",
                    f"runtime_state_dim={state.shape[-1]}",
                    f"configured_state_dim={self.input_dim}",
                    "adapting_by_crop_or_pad=true",
                )
                self._mismatch_print_count += 1
            if state.shape[-1] > self.input_dim:
                state = state[..., : self.input_dim]
            else:
                pad = torch.zeros(
                    *state.shape[:-1],
                    self.input_dim - state.shape[-1],
                    device=state.device,
                    dtype=state.dtype,
                )
                state = torch.cat([state, pad], dim=-1)
        return self.proj(state)


class HierarchicalRefinerActionHead(nn.Module):
    def __init__(
        self,
        config,
        top_hidden_dim: int,
        action_dim: int,
        state_dim: int,
        top_action_horizon: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.top_action_horizon = top_action_horizon
        head_cfg = _cfg_get(config.framework, "hierarchical_action_head", None)

        self.hidden_dim = int(_cfg_get(head_cfg, "hidden_dim", 512))
        self.num_top_latent_tokens = int(_cfg_get(head_cfg, "num_top_latent_tokens", 8))
        self.downsample_factor = int(_cfg_get(head_cfg, "downsample_factor", 4))
        self.dense_action_horizon = int(
            _cfg_get(head_cfg, "dense_action_horizon", self.top_action_horizon * self.downsample_factor)
        )
        self.chunk_action_horizon = int(
            _cfg_get(head_cfg, "chunk_action_horizon", self.top_action_horizon)
        )
        default_num_chunks = self.dense_action_horizon // self.chunk_action_horizon
        if default_num_chunks <= 0:
            raise ValueError(
                "Hierarchical_ActionHead requires chunk_action_horizon <= dense_action_horizon, "
                f"got {self.dense_action_horizon=} and {self.chunk_action_horizon=}."
            )
        self.num_refine_chunks = int(_cfg_get(head_cfg, "num_refine_chunks", default_num_chunks))
        if self.num_refine_chunks <= 0:
            raise ValueError(f"num_refine_chunks must be positive, got {self.num_refine_chunks}.")
        covered_horizon = self.num_refine_chunks * self.chunk_action_horizon
        if covered_horizon > self.dense_action_horizon:
            raise ValueError(
                "num_refine_chunks * chunk_action_horizon must not exceed dense_action_horizon, "
                f"got {self.num_refine_chunks} * {self.chunk_action_horizon} > {self.dense_action_horizon}."
            )
        self.temporal_context_extra = int(_cfg_get(head_cfg, "temporal_context_extra", 0))
        self.temporal_context_len = self.chunk_action_horizon + 2 * self.temporal_context_extra
        self.eval_num_chunks = int(_cfg_get(head_cfg, "eval_num_chunks", self.num_refine_chunks))
        self.eval_num_chunks = max(1, min(self.eval_num_chunks, self.num_refine_chunks))
        self.train_num_sampled_chunks = int(_cfg_get(head_cfg, "train_num_sampled_chunks", 0) or 0)
        if self.train_num_sampled_chunks < 0:
            raise ValueError(
                f"train_num_sampled_chunks must be non-negative, got {self.train_num_sampled_chunks}."
            )
        self.train_num_sampled_chunks = min(self.train_num_sampled_chunks, self.num_refine_chunks)
        self.temporal_shift_enabled = bool(_cfg_get(head_cfg, "temporal_shift_enabled", False))
        self.temporal_shift_schedule = list(_cfg_get(head_cfg, "temporal_shift_schedule", [0] * self.num_refine_chunks))
        self.use_sincos_temporal_pos = bool(_cfg_get(head_cfg, "use_sincos_temporal_pos", True))
        self.coarse_plan_replacement_enabled = bool(
            _cfg_get(head_cfg, "coarse_plan_mix_enabled", _cfg_get(head_cfg, "coarse_plan_replacement_enabled", False))
        )
        self.coarse_plan_replacement_prob_min = float(
            _cfg_get(head_cfg, "coarse_plan_mix_alpha_min", _cfg_get(head_cfg, "coarse_plan_replacement_prob_min", 0.02))
        )
        self.coarse_plan_replacement_prob_max = float(
            _cfg_get(head_cfg, "coarse_plan_mix_alpha_max", _cfg_get(head_cfg, "coarse_plan_replacement_prob_max", 0.10))
        )
        self.loss_weight = float(_cfg_get(head_cfg, "loss_weight", 1.0))
        self.refine_arch = str(_cfg_get(head_cfg, "refine_arch", "transformer")).lower()
        self.refine_prediction_type = str(_cfg_get(head_cfg, "refine_prediction_type", "residual")).lower()
        self.lower_flow_min_step_size = float(_cfg_get(head_cfg, "lower_flow_min_step_size", 0.05))
        action_model_cfg = _cfg_get(config.framework, "action_model", None)
        self.lower_num_timestep_buckets = int(
            _cfg_get(head_cfg, "lower_num_timestep_buckets", _cfg_get(action_model_cfg, "num_timestep_buckets", 1000))
        )
        self.residual_correction_enabled = bool(_cfg_get(head_cfg, "residual_correction_enabled", False))
        self.residual_constraint_type = str(_cfg_get(head_cfg, "residual_constraint_type", "hard")).lower()
        self.residual_bound_penalty_weight = float(_cfg_get(head_cfg, "residual_bound_penalty_weight", 0.0))
        self.residual_eval_clip = bool(_cfg_get(head_cfg, "residual_eval_clip", True))
        self.future_context_residual_loss_weight = float(
            _cfg_get(head_cfg, "future_context_residual_loss_weight", 0.0)
        )
        self.future_context_residual_loss_unmasked_only = bool(
            _cfg_get(head_cfg, "future_context_residual_loss_unmasked_only", True)
        )
        residual_max = _cfg_get(head_cfg, "residual_max", 0.05)
        if hasattr(residual_max, "__iter__") and not isinstance(residual_max, (str, bytes)):
            residual_max_tensor = torch.tensor([float(x) for x in residual_max], dtype=torch.float32)
            if residual_max_tensor.numel() != self.action_dim:
                raise ValueError(
                    f"hierarchical_action_head.residual_max must be scalar or length {self.action_dim}, "
                    f"got length {residual_max_tensor.numel()}"
                )
        else:
            residual_max_tensor = torch.tensor([float(residual_max)], dtype=torch.float32)
        self.register_buffer("residual_max", residual_max_tensor.view(1, 1, -1), persistent=False)
        self.debug_print = bool(_cfg_get(head_cfg, "debug_print", False))
        self.debug_max_prints = int(_cfg_get(head_cfg, "debug_max_prints", 3))
        self._debug_print_count = 0

        self.top_feature_proj = nn.Linear(top_hidden_dim, self.hidden_dim)
        self.init_guess_proj = nn.Linear(action_dim, self.hidden_dim)
        self.coarse_context_proj = nn.Linear(action_dim, self.hidden_dim)
        self.state_tokenizer = (
            StateTokenizer(
                state_dim=self.state_dim,
                hidden_dim=self.hidden_dim,
            )
            if self.state_dim > 0
            else None
        )
        self.realtime_obs_tokenizer = RealtimeObservationTokenizer(self.hidden_dim, head_cfg)
        self.coarse_plan_corruptor = CoarsePlanCorruptor(action_dim=action_dim, cfg=head_cfg)

        n_heads = int(_cfg_get(head_cfg, "num_heads", 8))
        attention_head_dim = int(_cfg_get(head_cfg, "lower_dit_attention_head_dim", self.hidden_dim // n_heads))
        dit_output_dim = int(_cfg_get(head_cfg, "lower_dit_output_dim", self.hidden_dim))
        ff_dim = int(_cfg_get(head_cfg, "feedforward_dim", self.hidden_dim * 4))
        dropout = float(_cfg_get(head_cfg, "dropout", 0.1))
        num_layers = int(_cfg_get(head_cfg, "num_layers", 4))

        self.query_embed = nn.Embedding(self.chunk_action_horizon, self.hidden_dim)
        self.temporal_pos = nn.Embedding(self.chunk_action_horizon, self.hidden_dim)
        self.query_type_embed = nn.Parameter(torch.zeros(self.hidden_dim))
        self.coarse_context_type_embed = nn.Parameter(torch.zeros(self.hidden_dim))
        self.temporal_pad_token = nn.Parameter(torch.zeros(self.hidden_dim))
        self.use_dit_refiner = self.refine_arch in {"dit", "dit_flow", "flow_dit", "gr00t_dit"}
        self.transformer = None
        self.dit_action_encoder = None
        self.dit_model = None
        self.dit_future_tokens = None
        self.dit_position_embedding = None
        self.dit_state_proj = None
        self.lower_dit_use_top_decoder_head = False
        self.refined_token_dim = self.hidden_dim
        if self.use_dit_refiner:
            dit_inner_dim = n_heads * attention_head_dim
            if dit_inner_dim != self.hidden_dim:
                raise ValueError(
                    "DiT lower refiner currently expects hidden_dim == num_heads * lower_dit_attention_head_dim, "
                    f"got {self.hidden_dim=} but {n_heads} * {attention_head_dim} = {dit_inner_dim}."
                )
            self.refined_token_dim = dit_output_dim
            self.dit_action_encoder = FlowActionEncoder(action_dim=action_dim, hidden_size=self.hidden_dim)
            self.dit_future_tokens = nn.Embedding(self.chunk_action_horizon, self.hidden_dim)
            self.dit_state_proj = nn.Linear(self.state_dim, self.hidden_dim) if self.state_dim > 0 else None
            self.dit_add_pos_embed = bool(_cfg_get(head_cfg, "lower_dit_add_pos_embed", True))
            self.dit_hidden_seq_len = self.chunk_action_horizon + self.chunk_action_horizon + (1 if self.state_dim > 0 else 0)
            if self.dit_add_pos_embed:
                max_seq_len = int(_cfg_get(head_cfg, "lower_dit_max_seq_len", max(64, self.dit_hidden_seq_len)))
                self.dit_position_embedding = nn.Embedding(max_seq_len, self.hidden_dim)
                nn.init.normal_(self.dit_position_embedding.weight, mean=0.0, std=0.02)
            self.dit_model = DiT(
                num_attention_heads=n_heads,
                attention_head_dim=attention_head_dim,
                output_dim=dit_output_dim,
                num_layers=num_layers,
                dropout=dropout,
                final_dropout=bool(_cfg_get(head_cfg, "lower_dit_final_dropout", True)),
                interleave_self_attention=bool(_cfg_get(head_cfg, "lower_dit_interleave_self_attention", True)),
                norm_type=str(_cfg_get(head_cfg, "lower_dit_norm_type", "ada_norm")),
                positional_embeddings=_cfg_get(head_cfg, "lower_dit_positional_embeddings", None),
                cross_attention_dim=self.hidden_dim,
            )
            self.lower_dit_use_top_decoder_head = bool(
                _cfg_get(head_cfg, "lower_dit_use_top_decoder_head", False)
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=n_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        if self.use_dit_refiner and self.lower_dit_use_top_decoder_head:
            decoder_hidden_dim = int(_cfg_get(head_cfg, "lower_dit_decoder_hidden_dim", dit_output_dim))
            self.output_head = FlowMLP(
                input_dim=self.refined_token_dim,
                hidden_dim=decoder_hidden_dim,
                output_dim=action_dim,
            )
        else:
            self.output_head = nn.Sequential(
                nn.LayerNorm(self.refined_token_dim),
                nn.Linear(self.refined_token_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, action_dim),
            )
        self.loss_fn = nn.L1Loss()

    def temporal_shift_max(self, chunk_idx: int) -> int:
        if not self.temporal_shift_schedule:
            return 0
        idx = min(chunk_idx, len(self.temporal_shift_schedule) - 1)
        return max(0, int(self.temporal_shift_schedule[idx]))

    def _temporal_position_tokens(self, positions: Tensor) -> Tensor:
        if self.use_sincos_temporal_pos:
            return get_1d_sincos_pos_embed_torch(self.hidden_dim, positions, dtype=next(self.parameters()).dtype)
        if positions.numel() == self.chunk_action_horizon and self.temporal_context_extra == 0:
            return self.temporal_pos.weight
        return torch.zeros(positions.numel(), self.hidden_dim, device=positions.device, dtype=next(self.parameters()).dtype)

    def _maybe_mix_coarse_inputs(
        self,
        init_guess: Tensor,
        coarse_context: Tensor | None,
        coarse_context_valid: Tensor | None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        if not self.training or not self.coarse_plan_replacement_enabled or init_guess.shape[0] <= 1:
            return init_guess, coarse_context, coarse_context_valid

        alpha_min = max(0.0, min(1.0, self.coarse_plan_replacement_prob_min))
        alpha_max = max(alpha_min, min(1.0, self.coarse_plan_replacement_prob_max))
        if alpha_max <= 0:
            return init_guess, coarse_context, coarse_context_valid

        permutation = torch.randperm(init_guess.shape[0], device=init_guess.device)
        if (permutation == torch.arange(init_guess.shape[0], device=init_guess.device)).all():
            permutation = torch.roll(permutation, shifts=1)

        mix_alpha = torch.empty(
            init_guess.shape[0],
            1,
            1,
            device=init_guess.device,
            dtype=init_guess.dtype,
        ).uniform_(alpha_min, alpha_max)
        init_guess = (1.0 - mix_alpha) * init_guess + mix_alpha * init_guess[permutation]

        if coarse_context is not None:
            other_context = coarse_context[permutation]
            if coarse_context_valid is not None:
                other_valid = coarse_context_valid[permutation].unsqueeze(-1)
                other_context = torch.where(other_valid, other_context, coarse_context)
            context_alpha = mix_alpha.to(device=coarse_context.device, dtype=coarse_context.dtype)
            coarse_context = (1.0 - context_alpha) * coarse_context + context_alpha * other_context
        return init_guess, coarse_context, coarse_context_valid

    def _debug_log(
        self,
        init_guess: Tensor,
        top_hidden_features: Tensor,
        context: Tensor,
        query: Tensor,
        refined_actions: Tensor,
        chunk_start_indices: Tensor,
        coarse_context: Tensor | None,
        state: Tensor | None,
        dense_target: Tensor | None,
        batch_images: List,
    ) -> None:
        if not self.debug_print or self._debug_print_count >= self.debug_max_prints:
            return
        first_sample = batch_images[0] if batch_images else []
        anchor_count = len(first_sample) if first_sample and isinstance(first_sample[0], (list, tuple)) else 1
        view_count = len(first_sample[0]) if anchor_count > 1 else len(first_sample)
        print(
            "[HierarchicalHead]",
            f"init_guess={tuple(init_guess.shape)}",
            f"top_hidden={tuple(top_hidden_features.shape)}",
            f"context={tuple(context.shape)}",
            f"query={tuple(query.shape)}",
            f"coarse_context={None if coarse_context is None else tuple(coarse_context.shape)}",
            f"refined={tuple(refined_actions.shape)}",
            f"chunk_starts={chunk_start_indices.detach().cpu().tolist()}",
            f"anchors={anchor_count}",
            f"views={view_count}",
            f"state={None if state is None else tuple(state.shape)}",
            f"dense_target={None if dense_target is None else tuple(dense_target.shape)}",
        )
        self._debug_print_count += 1

    def _compress_top_features(self, top_hidden_features: Tensor) -> Tensor:
        x = self.top_feature_proj(top_hidden_features)
        if x.size(1) == self.num_top_latent_tokens:
            return x
        return F.adaptive_avg_pool1d(x.transpose(1, 2), self.num_top_latent_tokens).transpose(1, 2)

    def _build_condition_tokens(
        self,
        top_hidden_features: Tensor,
        batch_images: List[List[Image.Image]],
        state: Tensor | None,
        coarse_context: Tensor | None,
        coarse_context_valid: Tensor | None,
    ) -> tuple[Tensor, dict]:
        compressed_top = self._compress_top_features(top_hidden_features)
        realtime_obs = self.realtime_obs_tokenizer(batch_images)

        context_tokens = [compressed_top, realtime_obs]
        context_info = {
            "coarse_token_start": None,
            "coarse_token_len": None,
            "coarse_positions": None,
            "coarse_keep_mask": None,
        }
        if coarse_context is not None:
            coarse_positions = torch.arange(
                -self.temporal_context_extra,
                self.chunk_action_horizon + self.temporal_context_extra,
                device=coarse_context.device,
            )
            keep_mask = (coarse_positions < 0) | (coarse_positions >= self.chunk_action_horizon)
            if keep_mask.any():
                filtered_context = coarse_context[:, keep_mask]
                filtered_valid = coarse_context_valid[:, keep_mask] if coarse_context_valid is not None else None
                filtered_positions = coarse_positions[keep_mask]
                context_info["coarse_token_start"] = sum(token.shape[1] for token in context_tokens)
                context_info["coarse_token_len"] = filtered_context.shape[1]
                context_info["coarse_positions"] = filtered_positions
                context_info["coarse_keep_mask"] = keep_mask
                context_tokens.append(
                    self._build_coarse_context_tokens(
                        filtered_context,
                        filtered_valid,
                        filtered_positions,
                    )
                )
        if self.state_tokenizer is not None and state is not None:
            state_tokens = self.state_tokenizer(state)
            context_tokens.append(state_tokens)
        return torch.cat(context_tokens, dim=1), context_info

    def build_condition_tokens_for_eval(
        self,
        top_hidden_features: Tensor,
        batch_images: List[List[Image.Image]],
        state: Tensor | None,
        coarse_context: Tensor | None,
        coarse_context_valid: Tensor | None,
    ) -> tuple[Tensor, dict]:
        """Precompute lower-refiner condition tokens for repeated eval steps.

        The realtime observation, top features, and context action tokens do not
        change across Euler/refine steps for a single returned chunk. Caching
        these tokens avoids re-running the visual encoder on every lower step.
        """
        param = next(self.parameters())
        top_hidden_features = top_hidden_features.to(device=param.device, dtype=param.dtype)
        if coarse_context is not None:
            coarse_context = coarse_context.to(device=param.device, dtype=param.dtype)
        if coarse_context_valid is not None:
            coarse_context_valid = coarse_context_valid.to(device=param.device, dtype=torch.bool)
        if state is not None:
            state = state.to(device=param.device, dtype=param.dtype)
        return self._build_condition_tokens(
            top_hidden_features=top_hidden_features,
            batch_images=batch_images,
            state=state,
            coarse_context=coarse_context,
            coarse_context_valid=coarse_context_valid,
        )

    def _build_context_tokens(
        self,
        init_guess: Tensor,
        top_hidden_features: Tensor,
        batch_images: List[List[Image.Image]],
        state: Tensor | None,
        coarse_context: Tensor | None,
        coarse_context_valid: Tensor | None,
    ) -> tuple[Tensor, Tensor, dict]:
        context, context_info = self._build_condition_tokens(
            top_hidden_features=top_hidden_features,
            batch_images=batch_images,
            state=state,
            coarse_context=coarse_context,
            coarse_context_valid=coarse_context_valid,
        )
        init_tokens = self.init_guess_proj(init_guess)
        query_positions = torch.arange(self.chunk_action_horizon, device=init_guess.device)
        query_pos = self._temporal_position_tokens(query_positions).unsqueeze(0)
        query = (
            init_tokens
            + self.query_embed.weight.unsqueeze(0)
            + self.query_type_embed.view(1, 1, -1)
            + query_pos
        )
        return context, query, context_info

    def _flow_timesteps(self, top_scale: Tensor) -> Tensor:
        top_scale = top_scale.clamp(min=0.0, max=1.0)
        timesteps = (top_scale * float(self.lower_num_timestep_buckets)).long()
        return timesteps.clamp(min=0, max=max(0, self.lower_num_timestep_buckets - 1))

    def _build_dit_hidden_states(
        self,
        init_guess: Tensor,
        state: Tensor | None,
        timesteps: Tensor,
    ) -> Tensor:
        action_tokens = self.dit_action_encoder(init_guess, timesteps)
        future_tokens = self.dit_future_tokens.weight.unsqueeze(0).expand(init_guess.shape[0], -1, -1)
        hidden_tokens = []
        if self.dit_state_proj is not None and state is not None:
            hidden_tokens.append(self.dit_state_proj(state).unsqueeze(1))
        hidden_tokens.extend([future_tokens, action_tokens])
        hidden_states = torch.cat(hidden_tokens, dim=1)
        if self.dit_position_embedding is not None:
            if hidden_states.shape[1] > self.dit_position_embedding.num_embeddings:
                raise ValueError(
                    "lower_dit_max_seq_len is too small for lower DiT hidden states: "
                    f"{hidden_states.shape[1]} > {self.dit_position_embedding.num_embeddings}"
                )
            pos_ids = torch.arange(hidden_states.shape[1], dtype=torch.long, device=hidden_states.device)
            hidden_states = hidden_states + self.dit_position_embedding(pos_ids).unsqueeze(0)
        return hidden_states

    def _compute_future_context_residual_loss(
        self,
        encoded: Tensor,
        context_info: dict,
        coarse_context_input: Tensor | None,
        coarse_context_target: Tensor | None,
        coarse_context_valid: Tensor | None,
        coarse_context_mask: Tensor | None,
    ) -> Tensor | None:
        if (
            not self.training
            or not self.residual_correction_enabled
            or self.refine_prediction_type in {"flow", "flow_velocity", "velocity"}
            or self.future_context_residual_loss_weight <= 0
            or coarse_context_input is None
            or coarse_context_target is None
        ):
            return None
        context_start = context_info.get("coarse_token_start")
        context_len = context_info.get("coarse_token_len")
        context_positions = context_info.get("coarse_positions")
        keep_mask = context_info.get("coarse_keep_mask")
        if context_start is None or context_len is None or context_positions is None:
            return None

        context_tokens = encoded[:, context_start : context_start + context_len, :]
        pred_residual = self.output_head(context_tokens)
        if keep_mask is not None:
            coarse_context_input = coarse_context_input[:, keep_mask]
            coarse_context_target = coarse_context_target[:, keep_mask]
            if coarse_context_valid is not None:
                coarse_context_valid = coarse_context_valid[:, keep_mask]
            if coarse_context_mask is not None:
                coarse_context_mask = coarse_context_mask[:, keep_mask]
        target_residual = coarse_context_target.to(device=pred_residual.device, dtype=pred_residual.dtype) - coarse_context_input.to(
            device=pred_residual.device,
            dtype=pred_residual.dtype,
        )

        valid = torch.ones(pred_residual.shape[:2], device=pred_residual.device, dtype=torch.bool)
        if coarse_context_valid is not None:
            valid = coarse_context_valid.to(device=pred_residual.device, dtype=torch.bool)
        future = (context_positions.to(device=pred_residual.device) >= self.chunk_action_horizon).view(1, -1)
        valid = valid & future
        if (
            self.future_context_residual_loss_unmasked_only
            and coarse_context_mask is not None
        ):
            valid = valid & (~coarse_context_mask.to(device=pred_residual.device, dtype=torch.bool))
        if not valid.any():
            return None

        per_token_loss = (pred_residual - target_residual).abs().mean(dim=-1)
        denom = valid.to(dtype=per_token_loss.dtype).sum().clamp_min(torch.finfo(per_token_loss.dtype).eps)
        return self.future_context_residual_loss_weight * (
            per_token_loss * valid.to(dtype=per_token_loss.dtype)
        ).sum() / denom

    def _build_coarse_context_tokens(
        self,
        coarse_context: Tensor,
        coarse_context_valid: Tensor | None,
        context_positions: Tensor,
    ) -> Tensor:
        tokens = self.coarse_context_proj(coarse_context)
        if coarse_context_valid is None:
            coarse_context_valid = torch.ones(
                tokens.shape[:2],
                device=tokens.device,
                dtype=torch.bool,
            )
        pad = self.temporal_pad_token.view(1, 1, -1).expand_as(tokens)
        tokens = torch.where(coarse_context_valid.unsqueeze(-1), tokens, pad)

        context_pos = self._temporal_position_tokens(context_positions.to(device=tokens.device)).unsqueeze(0)
        return tokens + self.coarse_context_type_embed.view(1, 1, -1) + context_pos

    def forward(
        self,
        init_guess: Tensor,
        top_hidden_features: Tensor,
        batch_images: List[List[Image.Image]],
        state: Tensor | None = None,
        dense_target: Tensor | None = None,
        dense_target_valid: Tensor | None = None,
        coarse_context: Tensor | None = None,
        coarse_context_target: Tensor | None = None,
        coarse_context_valid: Tensor | None = None,
        chunk_start_indices: Tensor | None = None,
        flow_step_size: Tensor | float | None = None,
        flow_timestep: Tensor | float | None = None,
        precomputed_context: Tensor | None = None,
        precomputed_context_info: dict | None = None,
        target_velocity: Tensor | None = None,
    ) -> dict:
        param = next(self.parameters())
        init_guess = init_guess.to(device=param.device, dtype=param.dtype)
        top_hidden_features = top_hidden_features.to(device=param.device, dtype=param.dtype)
        if coarse_context is not None:
            coarse_context = coarse_context.to(device=param.device, dtype=param.dtype)
        if coarse_context_target is not None:
            coarse_context_target = coarse_context_target.to(device=param.device, dtype=param.dtype)
        if coarse_context_valid is not None:
            coarse_context_valid = coarse_context_valid.to(device=param.device, dtype=torch.bool)
        if state is not None:
            state = state.to(device=param.device, dtype=param.dtype)
        if dense_target is not None:
            dense_target = dense_target.to(device=param.device, dtype=param.dtype)
        if dense_target_valid is not None:
            dense_target_valid = dense_target_valid.to(device=param.device, dtype=torch.bool)
        if flow_step_size is None:
            flow_step_size = torch.ones(
                init_guess.shape[0],
                device=param.device,
                dtype=param.dtype,
            )
        elif not torch.is_tensor(flow_step_size):
            flow_step_size = torch.tensor(flow_step_size, device=param.device, dtype=param.dtype)
        else:
            flow_step_size = flow_step_size.to(device=param.device, dtype=param.dtype)
        if flow_step_size.dim() == 0:
            flow_step_size = flow_step_size.expand(init_guess.shape[0])
        flow_step = flow_step_size.view(-1, 1, 1)
        target_flow_step = flow_step.clamp_min(self.lower_flow_min_step_size)
        if flow_timestep is None:
            flow_timestep = 1.0 - flow_step_size
        elif not torch.is_tensor(flow_timestep):
            flow_timestep = torch.tensor(flow_timestep, device=param.device, dtype=param.dtype)
        else:
            flow_timestep = flow_timestep.to(device=param.device, dtype=param.dtype)
        if flow_timestep.dim() == 0:
            flow_timestep = flow_timestep.expand(init_guess.shape[0])
        flow_timesteps = self._flow_timesteps(flow_timestep.reshape(-1))

        init_guess, coarse_context, coarse_context_valid = self._maybe_mix_coarse_inputs(
            init_guess,
            coarse_context,
            coarse_context_valid,
        )
        coarse_context_input = coarse_context
        coarse_context_mask = None
        if self.training and coarse_context is not None:
            coarse_context, coarse_context_mask = self.coarse_plan_corruptor.apply_context_mask(coarse_context)

        if self.use_dit_refiner:
            if precomputed_context is None:
                context, context_info = self._build_condition_tokens(
                    top_hidden_features=top_hidden_features,
                    batch_images=batch_images,
                    state=None,
                    coarse_context=coarse_context,
                    coarse_context_valid=coarse_context_valid,
                )
            else:
                context = precomputed_context.to(device=param.device, dtype=param.dtype)
                context_info = precomputed_context_info or {
                    "coarse_token_start": None,
                    "coarse_token_len": None,
                    "coarse_positions": None,
                    "coarse_keep_mask": None,
                }
            query = self.dit_action_encoder(init_guess, flow_timesteps)
            hidden_states = self._build_dit_hidden_states(init_guess, state, flow_timesteps)
            encoded = self.dit_model(
                hidden_states=hidden_states,
                encoder_hidden_states=context,
                timestep=flow_timesteps,
            )
            refined_tokens = encoded[:, -self.chunk_action_horizon :, :]
        else:
            context, query, context_info = self._build_context_tokens(
                init_guess,
                top_hidden_features,
                batch_images,
                state,
                coarse_context,
                coarse_context_valid,
            )
            tokens = torch.cat([context, query], dim=1)
            encoded = self.transformer(tokens)
            refined_tokens = encoded[:, -self.chunk_action_horizon :, :]
        action_delta_or_pred = self.output_head(refined_tokens)
        future_context_residual_loss = self._compute_future_context_residual_loss(
            encoded=encoded,
            context_info=context_info,
            coarse_context_input=coarse_context_input,
            coarse_context_target=coarse_context_target,
            coarse_context_valid=coarse_context_valid,
            coarse_context_mask=coarse_context_mask,
        )
        residual_for_action = None
        residual_bound_loss = None
        velocity_for_action = None
        if self.refine_prediction_type in {"flow", "flow_velocity", "velocity"}:
            velocity_for_action = action_delta_or_pred
            eval_flow_step = flow_step
            if (
                not self.training
                and getattr(self, "eval_lower_velocity_min_step_compensation", False)
            ):
                eval_flow_step = target_flow_step
            refined_actions = init_guess + eval_flow_step * velocity_for_action
        elif self.residual_correction_enabled:
            residual_max = self.residual_max.to(device=action_delta_or_pred.device, dtype=action_delta_or_pred.dtype)
            if self.residual_constraint_type == "hard":
                residual_for_action = torch.tanh(action_delta_or_pred) * residual_max
            elif self.residual_constraint_type == "soft":
                residual_for_action = action_delta_or_pred
                if self.residual_eval_clip and not self.training:
                    residual_for_action = residual_for_action.clamp(min=-residual_max, max=residual_max)
                if self.residual_bound_penalty_weight > 0:
                    denom = residual_max.clamp_min(torch.finfo(action_delta_or_pred.dtype).eps)
                    excess = F.relu(action_delta_or_pred.abs() / denom - 1.0)
                    residual_bound_loss = excess.square().mean()
            else:
                raise ValueError(
                    f"Unsupported residual_constraint_type={self.residual_constraint_type!r}. "
                    "Use one of: hard, soft."
                )
            refined_actions = init_guess + residual_for_action
        else:
            refined_actions = action_delta_or_pred
        self._debug_log(
            init_guess=init_guess,
            top_hidden_features=top_hidden_features,
            context=context,
            query=query,
            refined_actions=refined_actions,
            chunk_start_indices=(
                chunk_start_indices.to(device=init_guess.device)
                if chunk_start_indices is not None
                else torch.zeros(init_guess.shape[0], device=init_guess.device, dtype=torch.long)
            ),
            coarse_context=coarse_context,
            state=state,
            dense_target=dense_target,
            batch_images=batch_images,
        )

        outputs = {"refined_actions": refined_actions}
        if velocity_for_action is not None:
            outputs["velocity_actions"] = velocity_for_action
        if residual_for_action is not None:
            outputs["residual_actions"] = residual_for_action
            outputs["raw_residual_actions"] = action_delta_or_pred
        if residual_bound_loss is not None:
            outputs["residual_bound_loss"] = residual_bound_loss
        if future_context_residual_loss is not None:
            outputs["future_context_residual_loss"] = future_context_residual_loss
        if dense_target is not None:
            if dense_target_valid is None:
                valid = torch.ones(
                    dense_target.shape[:2],
                    device=dense_target.device,
                    dtype=torch.bool,
                )
            else:
                valid = dense_target_valid.to(device=dense_target.device, dtype=torch.bool)
            valid_weight = valid.to(dtype=dense_target.dtype).unsqueeze(-1)
            valid_denom = (valid_weight.sum() * dense_target.shape[-1]).clamp_min(
                torch.finfo(dense_target.dtype).eps
            )
            if velocity_for_action is not None:
                if target_velocity is None:
                    # Legacy path: anchored at the refiner's own input x_t. As t -> 1 both
                    # (dense_target - init_guess) and the true step (1 - t) vanish, so this
                    # is 0/0; lower_flow_min_step_size clamps the denominator, which keeps
                    # it finite but makes the target WRONG by a factor of (1-t)/0.05 and
                    # disagrees with the unclamped flow_step used to reconstruct above.
                    target_velocity = (dense_target - init_guess) / target_flow_step
                else:
                    target_velocity = target_velocity.to(
                        device=dense_target.device, dtype=dense_target.dtype
                    )
                per_element_loss = (velocity_for_action - target_velocity).square()
            else:
                per_element_loss = (refined_actions - dense_target).abs()
            refine_loss = self.loss_weight * (per_element_loss * valid_weight).sum() / valid_denom
            if residual_bound_loss is not None:
                refine_loss = refine_loss + self.residual_bound_penalty_weight * residual_bound_loss
            if future_context_residual_loss is not None:
                refine_loss = refine_loss + future_context_residual_loss
            outputs["refine_loss"] = refine_loss
            outputs["action_valid_ratio"] = valid.float().mean().detach()
        return outputs
