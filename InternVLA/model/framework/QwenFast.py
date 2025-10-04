"""
Qwenvl OFT framework:
Vision-Language-Action diffusion model integrating:
  - Qwen2.5 vision-language backbone
  - learnable raw action token + positional embeddings for action sequences len = action chunk
  - DiT diffusion head for future action sequence prediction
Primary goal: predict continuous future actions special token conditioned on multi-view images + instruction.

"""

from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from qwen_vl_utils import process_vision_info


from InternVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from InternVLA.model.framework.base_framework import baseframework
from InternVLA.model.modules.vlm.QWen2_5 import get_qwen2_5_interface
from InternVLA.model.modules.action_model.fast_ActionHeader import get_action_model
from InternVLA.training.trainer_utils.metrics import resize_images


class Qwenvl_Fast(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise QFormer for multi-layer feature aggregation
      - DINO encoder for dense multi-view spatial tokens
      - DiT diffusion head for future action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_qwen2_5_interface(config=self.config)
        self.action_model = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        self.hidden_dim = config.framework.action_model.action_hidden_dim
        

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        训练前向：直接next token prediction 预测未来动作（无扩散）。

        Flow:
          1. Build QwenVL inputs (images + instruction tokens)
          2. Extract hidden states from configured layer range
          7. Predict action and compute L1 loss

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
            **kwargs: Reserved.

        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]
        
        # step 0: map_raw_action_to_vlm_action
        vlm_action_tokens = self.action_model.encoder_action2vlmtoken(actions)  # List[str]


        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions, solutions=vlm_action_tokens)
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        
        vlm_action_loss = qwenvl_outputs.loss
        if vlm_action_loss is None or torch.isnan(vlm_action_loss): 
            vlm_action_loss = torch.tensor(0.0, device=self.qwen_vl_interface.model.device)

        return {"action_loss": vlm_action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],  # Batch of PIL Image list as [view1, view2]
        instructions: List[str],
        **kwargs: str,
    ) -> np.ndarray:
        """
        推理：单次前向直接回归未来动作（无扩散采样）。

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Args:
            batch_images: List of samples; each sample is List[PIL.Image] (multi-view).
            instructions: List[str] natural language task instructions.
            cfg_scale: >1 enables classifier-free guidance (scales conditional vs unconditional).
            use_ddim: Whether to use DDIM deterministic sampling.
            num_ddim_steps: Number of DDIM steps if enabled.
            **kwargs: Reserved.

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
        instructions = [instruction for instruction in instructions]


        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = self.qwen_vl_interface.model.generate(
                **qwen_inputs,
            )

        # decoder vlm_action to continue actions

        # --- Extract and decoder vlm_action to continue actions ---
        # Find the indices of tokens within the action token range
       
        # --- 抽取动作 token (已偏移区间) ---
        batch_vlm_action_token_ids = self._extract_action_token_ids(generated_ids)

        # --- 解码回 fast tokenizer 语义 ---
        decoded_actions = self._decode_action_tokens(batch_vlm_action_token_ids)

        return {"normalized_actions": decoded_actions}

    def _extract_action_token_ids(
        self,
        generated_ids: torch.LongTensor,
    ) -> List[List[int]]:
        """
        从生成的 token 序列里抽取动作 token（已加偏移），返回一个二维列表：
        ret[b] = [vlm_action_token_id_0, vlm_action_token_id_1, ...]
        规则：保留所有落在 [_ACTION_TOKEN_MIN, _ACTION_TOKEN_MAX] 内的 token，按出现顺序。
        你可按需要改成“只取首次出现后连续段”。
        """
        act_min = self.action_model._ACTION_TOKEN_MIN
        act_max = self.action_model._ACTION_TOKEN_MAX
        mask = (generated_ids >= act_min) & (generated_ids <= act_max)  # [B, L]
        results = []
        for b in range(generated_ids.size(0)):
            idx = mask[b].nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                results.append([])
                continue
            # 全部动作 token
            tokens = generated_ids[b, idx].tolist()
            results.append(tokens)
        return results

    def _decode_action_tokens(self, batch_vlm_tokens: List[List[int]]) -> List[Any]:
        """
        将带偏移的 VLM 动作 token 列表解码回 fast tokenizer 语义。
        fast_tokenizer.decode 预期输入：原始 fast token id 序列（未加偏移）。
        """
        act_min = self.action_model._ACTION_TOKEN_MIN
        decoded = []
        for seq in batch_vlm_tokens:
            if not seq:
                decoded.append(None)
                continue
            fast_ids = [t - act_min for t in seq]
            try:
                out = self.action_model.fast_tokenizer.decode(fast_ids)
            except Exception:
                # 若 fast_tokenizer.decode 接口不同，可在这里调整
                out = None
            decoded.append(out)
        return decoded


def build_model_framework(config: dict = {}) -> Qwenvl_Fast:
    """
    工厂方法：返回简化版 Qwenvl_OFT
    """
    model = Qwenvl_Fast(config=config)
    return model


if __name__ == "__main__":
    from omegaconf import OmegaConf

    # model parameters
    import debugpy

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    config_yaml = "InternVLA/config/training/internvla_cotrain_oxe.yaml"
    cfg = OmegaConf.load(config_yaml)
    cfg.framework.qwenvl.base_vlm = "playground/Pretrained_models/nora"

    cfg.framework.action_model.action_hidden_dim 
    # try get model
    model = build_model_framework(cfg)
    print(model)

    # try forward model
    # can be fake sample， but here get from dataloader for simpler
    from InternVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    vla_dataset_cfg = cfg.datasets.vla_data
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,  # For Debug
        collate_fn=collate_fn,
    )
    # zhe
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        batch
        break

    # try get model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model(batch)
    pass
    action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]])
