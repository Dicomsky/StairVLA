import os
from typing import List, Dict
import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from .QWen2_5 import get_qwen2_5_interface
from accelerate.logging import get_logger

logger = get_logger(__name__)

def add_spatial_tokens(
    interface,
    tokens: List[str],
    init_strategy: str = "avg",
    as_special: bool = True,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    给现有 Qwen 接口添加 tokens。
    init_strategy: avg / normal / zero
    """
    tok = interface.processor.tokenizer
    vocab = tok.get_vocab()
    new_tokens = [t for t in tokens if t not in vocab]
    if not new_tokens:
        return {t: tok.convert_tokens_to_ids(t) for t in tokens}

    added = tok.add_special_tokens({"additional_special_tokens": new_tokens}) if as_special else tok.add_tokens(new_tokens)
    old_embed = interface.model.get_input_embeddings()
    interface.model.resize_token_embeddings(len(tok))
    new_embed = interface.model.get_input_embeddings()
    start = len(tok) - added
    hidden = new_embed.weight.shape[1]

    with torch.no_grad():
        if init_strategy == "avg":
            ref = old_embed.weight.mean(dim=0, keepdim=True)
        elif init_strategy == "zero":
            ref = torch.zeros(1, hidden, device=new_embed.weight.device, dtype=new_embed.weight.dtype)
        else:
            ref = None
        for i in range(added):
            w = new_embed.weight[start + i]
            if init_strategy == "normal":
                nn.init.normal_(w, mean=0.0, std=0.02)
            else:
                w.copy_(ref[0])
    mapping = {t: tok.convert_tokens_to_ids(t) for t in tokens}
    if verbose:
        logger.info(f"[SpatialTokens] Added {added} tokens -> {mapping}")
    return mapping

def save_spatial_bundle(interface, save_dir: str, token_map: Dict[str, int] = None):
    os.makedirs(save_dir, exist_ok=True)
    interface.model.save_pretrained(save_dir)
    try:
        interface.processor.save_pretrained(save_dir)
    except Exception:
        interface.processor.tokenizer.save_pretrained(save_dir)
    if token_map:
        import json
        with open(os.path.join(save_dir, "added_token_id_map.json"), "w", encoding="utf-8") as f:
            json.dump(token_map, f, ensure_ascii=False, indent=2)
    logger.info(f"[SpatialTokens] Saved bundle -> {save_dir}")

def ensure_spatial_tokens_on_load(model_dir: str, expected_tokens: List[str]) -> List[str]:
    tok = AutoProcessor.from_pretrained(model_dir).tokenizer
    missing = [t for t in expected_tokens if t not in tok.get_vocab()]
    return missing

def load_interface_with_spatial(config, model_dir: str, spatial_tokens: List[str], init_strategy="avg"):
    iface = get_qwen2_5_interface(config=config)
    iface.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        attn_implementation="flash_attention_2",
        torch_dtype="auto",
        device_map="cuda",
    )
    iface.processor = AutoProcessor.from_pretrained(model_dir)
    iface.processor.tokenizer.padding_side = "left"
    missing = [t for t in spatial_tokens if t not in iface.processor.tokenizer.get_vocab()]
    if missing:
        add_spatial_tokens(iface, missing, init_strategy=init_strategy)
        logger.info(f"[SpatialTokens] 补齐缺失 tokens: {missing}")
    return iface