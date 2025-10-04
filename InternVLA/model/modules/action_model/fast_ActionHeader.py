"""Fast Action Tokenizer Adapter
"this file is adap from https://huggingface.co/physical-intelligence/fast"

概述:
    本模块封装了一个轻量级“动作 → 语言模型可读序列”转换器 (Fast_Action_Tokenizer)。
    其核心目标是：把连续 / 离散的原始机器人动作（raw_actions）转换为
    形如 <robot_action_12><robot_action_3><robot_action_87> ... 的伪自然语言 token 串，
    便于直接拼接进多模态大模型 (VLM / LLM) 的对话模板，复用其语言建模能力进行动作预测。
"""

import torch.nn as nn
from typing import List, Dict, Any, Callable, Optional
import os
import numpy as np
from transformers import AutoProcessor

class Fast_Action_Tokenizer(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, fast_tokenizer_name="physical-intelligence/fast"):
        super().__init__()
        
        self.fast_tokenizer = AutoProcessor.from_pretrained(
            fast_tokenizer_name, trust_remote_code=True
        )

    def encoder_action2vlmtoken(self, raw_actions):
        # x: (batch_size, chunck, dim)
        batch_actions = np.stack(raw_actions, axis=0)  # (B, T, D)
        batch_fast_tokens = self.fast_tokenizer(batch_actions)
        # batch_fast_tokens = [self.fast_tokenizer(raw_action)[0] for raw_action in raw_actions]
        batch_vlm_actions = [self.map_fast_token_to_vlm_action(fast_tokens) for fast_tokens in batch_fast_tokens]
        return batch_vlm_actions # List[str]
    
    def decoder_action(self, pred_actions):
        # x: (batch_size, chunck, dim)
        return pred_actions
    
    def map_fast_token_to_vlm_action(self, tokens) -> str:
        """Maps fast action tokens to the VLM action format.
        Action token 0 is mapped to the string <robot_action_0>  ... and so on 
        """
        return ''.join([f"<robot_action_{token}>" for token in tokens]) #@Jinhui, I'm not sure why we using this

    def fit_tokenizer_on_datasets(self, action_dataset, datasets_path="<your_local_path>", ):
        # 如果 datasets_path 存在， 直接读取
        if os.path.exists(datasets_path):

            self.fast_tokenizer = AutoProcessor.from_pretrained(
            datasets_path, trust_remote_code=True
        )
            return
        else:
            # 如果不存在，Fit the tokenizer on the new dataset
            new_tokenizer = self.fast_tokenizer.tokenizer.fit(action_dataset)
            self.fast_tokenizer = new_tokenizer

            # Save the new tokenizer, optionally push it to the Hugging Face model hub
            self.fast_tokenizer.save_pretrained(datasets_path)


def get_action_model(config=None):
    """
    Factory: build ActionModel from global framework config.

    Args:
        config: Global config (expects config.framework.action_model namespace).
    Returns:
        ActionModel: Initialized diffusion action head.
    """
    action_model = Fast_Action_Tokenizer()

    return action_model
