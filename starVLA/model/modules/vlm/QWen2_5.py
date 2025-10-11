# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import torch
import transformers
from typing import Optional, List
import copy
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Dict, Optional, List
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchFeature

from qwen_vl_utils import process_vision_info


from accelerate.logging import get_logger

logger = get_logger(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

_ACTION_TOKEN_MIN = 151665 # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
_ACTION_TOKEN_MAX = 153712 # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md


import torch.nn as nn


class _QWen_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen2.5-VL (Qwen2_5_VLForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    Notes:
        - Keeps original model behavior; does not modify internal architecture.
        - Mixed precision handled via torch.autocast in forward / generate.
        - Adaptation layer can be extended for future multi-modal routing if needed.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen2.5-VL wrapper.

        Parameters:
            config (dict | Any | None):
                Expected to expose a nested attribute/namespace `framework.get("qwenvl", {})`
                where:
                    framework.qwenvl.base_vlm (str): HuggingFace model id or local path.
                Optional expected structure (illustrative):
                    config.framework.get("qwenvl", {}) -> {
                        "base_vlm": "Qwen/Qwen2.5-VL-3B-Instruct"
                    }
                    config.datasets.vla_data.get("CoT_prompt", str) may be used later in build_qwenvl_inputs.
            **kwargs:
                Ignored currently; placeholder for future extension (e.g., override device_map, dtype).

        Side Effects:
            - Downloads / loads pretrained Qwen2.5-VL weights (unless cached).
            - Instantiates AutoProcessor and enforces left padding (required for some FlashAttention paths).

        Attributes Set:
            self.model (Qwen2_5_VLForConditionalGeneration)
            self.processor (AutoProcessor)
            self.config (original config reference)

        Notes:
            - device_map='cuda' is passed to from_pretrained (single or multi-GPU depending on HF accelerate mapping).
            - torch_dtype='auto' lets HF decide best available (prefers bfloat16 on supported hardware).
            - tokenizer padding_side forced to 'left' (important for generation + KV caching alignment).
        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen2.5-VL-3B-Instruct")

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation="flash_attention_2",
            torch_dtype="auto",
            device_map="cuda",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen2.5-VL backbone.

        Args:
            input_ids (LongTensor | None): [B, T] token ids (mutually exclusive with inputs_embeds).
            attention_mask (Tensor | None): [B, T], 1 = attend, 0 = masked.
            pixel_values (FloatTensor | None): Vision batch (model-specific preprocessed shape).
            labels (LongTensor | None): [B, T] LM targets; ignored positions = -100 (IGNORE_INDEX).
            image_grid_thw (FloatTensor | None): Optional tiling metadata (e.g., [B, 3] for temporal/height/width splits).
            inputs_embeds (FloatTensor | None): [B, T, D] alternative embedding input.
            past_key_values (List[FloatTensor] | None): Cached KV states for incremental decoding.
            use_cache (bool | None): If True, returns updated past_key_values.
            output_attentions (bool): Whether to include attention maps.
            output_hidden_states (bool): Must be True if downstream modules consume hidden states.
            return_dict (bool): Return HF dataclass if True; else tuple.
            **kwargs: Extra args forwarded to underlying model.

        Returns:
            CausalLMOutputWithPast | tuple: HF-standard structure (logits, past_key_values, hidden_states, etc.).

        Notes:
            - Autocast(bfloat16) used for efficiency.
            - padding_side already set to 'left' in tokenizer at init.
            - Hidden states required for auxiliary alignment or feature extraction modules.
        """

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        return outputs

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Construct and tokenize multimodal chat-style inputs for Qwen2.5-VL (batched).

        Overview:
            For each sample i:
                - Takes a list of PIL images: images[i] = [img_0, img_1, ...]
                - Takes a matching instruction string instructions[i]
                - Optionally formats instruction with a chain-of-thought template (CoT_prompt) if present in config.
                - Builds a single-turn chat message containing:
                      [{"role": "user", "content": [
                          {"type": "image", "image": <PIL.Image>}, ...,
                          {"type": "text", "text": <final_prompt>}
                      ]}]
                - Applies processor.apply_chat_template(..., add_generation_prompt=True)
                - Extracts vision inputs via process_vision_info
                - Calls processor(...) to produce a BatchFeature with token + vision tensors.

        Parameters:
            images (List[List[PIL.Image.Image]]):
                Length B. Each element is a (possibly empty) list of PIL images associated with that instruction.
                Supports multi-image inputs (ordered). For video-as-frames, upstream code should decide packaging.
            instructions (List[str]):
                Length B textual prompts or task instructions.
            **kwargs:
                Reserved for future extensions (e.g., system prompts, style controls, additional metadata).

        Config Dependencies:
            self.config.datasets.vla_data.get("CoT_prompt", str):
                If present, each instruction string is injected into the template by replacing "{instruction}".

        Returns:
            BatchFeature (HF):
                Typical keys (moved to self.model.device):
                    input_ids: LongTensor [B, T]
                    attention_mask: LongTensor/Bool [B, T]
                    pixel_values / image_grid / video specifics (model-dependent)
                    (Possibly) token_type_ids or other processor outputs
                The structure aligns with what Qwen2_5_VLForConditionalGeneration.forward expects.

        Shapes / Notes:
            - Sequence length T varies by number of images (special tokens) + prompt length.
            - pixel_values may have internal batching distinct from B if images are flattened; underlying model maps them.
            - The association between images and textual placeholders is preserved by processor ordering.

        Edge Cases:
            - Empty image list per sample is allowed (pure text prompt).
            - Mismatched lengths of images and instructions raise AssertionError.
            - CoT prompt replacement is naive string replace; ensure template contains "{instruction}" placeholder.

        Performance:
            - This path aims for faster inference vs. more granular per-turn assembly.
            - Minor tokenization differences (e.g., whitespace) can affect highly overfitted benchmarks.

        Does Not:
            - Perform augmentation.
            - Cache processed pixel tensors.
            - Handle streaming input.

        """

        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # Prepare text prompts using processor
        # default process is json --> message --> texts --> input_ids
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]

        # image_inputs = list of PIL
        image_inputs, video_inputs = process_vision_info(messages)
        batch_input = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")


        # if solutions, mask out the solution tokens in labels
        if solutions is not None:
            action_token_min = _ACTION_TOKEN_MIN # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
            action_token_max = _ACTION_TOKEN_MAX # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
            labels = batch_input['input_ids'].clone()
            # For each sequence in the batch, find the first occurrence of an action token.
            for i in range(labels.size(0)):
                seq = labels[i]
                # Create a mask for tokens within the action token range.
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    # Mask out all tokens before the first action token.
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    # If no action token is found, mask the entire sequence.
                    seq[:] = IGNORE_INDEX
                    RuntimeWarning (f"action token are on in yout tokenizer, plz see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md.")
            
            labels[labels == self.processor.tokenizer.pad_token_id] = -100 ## mask out pad tokens as well
            batch_input['labels'] = labels

        return batch_input.to(self.model.device)



def get_qwen2_5_interface(config=None, **kwargs):
    """
    Factory function returning the wrapped Qwen2.5-VL interface.

    Parameters:
        config (dict | Any | None):
            Passed to _QWen_VL_Interface. Expected (optional) structure:
                config.framework.get("qwenvl", {}) -> {
                    "base_vlm": "<model_id or local path>"
                }
                config.datasets.vla_data.get("CoT_prompt") optionally used in build_qwenvl_inputs.
        **kwargs:
            Currently unused; placeholder for future (e.g., override device_map, precision modes).

    Returns:
        _QWen_VL_Interface:
            Instance exposing:
                .forward(...)
                .generate(...)
                .build_qwenvl_inputs(...)
                .model (raw HF model)
                .processor (tokenizer + image/video processor)

    Notes:
        - Does not wrap with additional adapters; extension point for future multi-head / routing logic.
        - Device placement handled by underlying from_pretrained (device_map='cuda').

    """
    model = _QWen_VL_Interface(config=config)

    return model



def messages_to_sources(batch_messages):
    """
    将 batch 格式的 messages 转换为 sources 格式，支持多模态（image/text）。
    
    Args:
        batch_messages: List[List[Dict]]，每个样本是一组 message 对话

    Returns:
        List[List[Dict]]，每个样本的 source 对话
    """
    batch_sources = []

    for messages in batch_messages:
        source = []
        for msg in messages:
            role = msg["role"]
            segments = msg["content"]

            parts = []
            for seg in segments:
                if seg["type"] == "text":
                    parts.append(seg["text"])
                elif seg["type"] == "image":
                    parts.append(DEFAULT_IMAGE_TOKEN) # VIDEO 还不支持
                else:
                    raise ValueError(f"Unsupported content type: {seg['type']}")

            content = "\n".join(parts)
            source.append({"from": "human" if role == "user" else "gpt", "value": content})

        batch_sources.append(source)

    return batch_sources

def preprocess_qwen_2_visual(
    messages,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw: List = [],
    visual_type: str = "image",
) -> Dict:
    # message --> sources json
    pass
    

    sources = messages_to_sources(messages)  # 转换为 source 格式
    # torch.distributed.barrier()
    # 复用 QWenvl 代码
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."
    if visual_type not in ["image", "video"]:
        raise ValueError("visual_type must be either 'image' or 'video'")
    
    # tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index = 0
    input_ids, targets = [], []
    action_obs_mask = [] # 记录那些token 是obs 可以看到的 --> 传递给 Q-Former # TODO 看一下是否有更好的实现方式
    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                visual_tag = f"<{visual_type}>"
                if visual_tag in content:
                    parts = content.split(visual_tag)
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|{visual_type}_pad|>"
                            * grid_thw[visual_replicate_index]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target) # TODO 看一下是如何处理结束符号的 @JinhuiYE


    # TODO Batch padding 
    # TODO 不建议在这里执行padding
    
    # Padding input_ids 和 targets
    input_ids = pad_sequence(
        [torch.tensor(ids, dtype=torch.long) for ids in input_ids],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
        padding_side=tokenizer.padding_side
    )
    targets = pad_sequence(
        [torch.tensor(tgt, dtype=torch.long) for tgt in targets],
        batch_first=True,
        padding_value=IGNORE_INDEX,
        padding_side=tokenizer.padding_side
    )

    # 构建 attention_mask：非 pad 的位置为 1，pad 的为 0
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=attention_mask,
    )


if __name__ == "__main__":
    model_id = "./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct"
    qwen_vl = get_qwen2_5_interface(model_id)
    pass
