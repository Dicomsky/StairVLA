import argparse
import json
import os
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

def add_new_tokens(
    model,
    tokenizer,
    new_tokens: List[str],
    init_strategy: str = "avg",
    as_special: bool = True,
) -> Tuple[Dict[str, int], int]:
    """
    向模型与 tokenizer 中添加新的 tokens（若不存在）。
    init_strategy: avg / normal / zero
    返回: (所有目标 tokens 的 token_id 映射, 实际新增数量)
    """
    vocab = tokenizer.get_vocab()
    to_add = [t for t in new_tokens if t not in vocab]

    if not to_add:
        # 无新增，直接返回映射
        return {t: tokenizer.convert_tokens_to_ids(t) for t in new_tokens}, 0

    if as_special:
        added = tokenizer.add_special_tokens({"additional_special_tokens": to_add})
    else:
        added = tokenizer.add_tokens(to_add)

    if added == 0:
        return {t: tokenizer.convert_tokens_to_ids(t) for t in new_tokens}, 0

    old_embed = model.get_input_embeddings()
    old_vocab_size, hidden = old_embed.weight.shape

    model.resize_token_embeddings(len(tokenizer))
    new_embed = model.get_input_embeddings()
    new_vocab_size = new_embed.weight.shape[0]
    assert new_vocab_size - old_vocab_size == added, "新增 token 数不一致"

    # 需要初始化的区间
    start = old_vocab_size
    end = new_vocab_size

    with torch.no_grad():
        if init_strategy == "avg":
            ref_vec = old_embed.weight.mean(dim=0, keepdim=True)
            for idx in range(start, end):
                new_embed.weight[idx].copy_(ref_vec[0])
        elif init_strategy == "zero":
            for idx in range(start, end):
                new_embed.weight[idx].zero_()
        elif init_strategy == "normal":
            for idx in range(start, end):
                nn.init.normal_(new_embed.weight[idx], mean=0.0, std=0.02)
        else:
            raise ValueError(f"未知 init_strategy: {init_strategy}")

    mapping = {t: tokenizer.convert_tokens_to_ids(t) for t in new_tokens}
    return mapping, added

def save_bundle(model, tokenizer, mapping: Dict[str, int], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "added_token_id_map.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"[OK] 已保存到: {save_dir}")

def reload_and_check(save_dir: str, tokens: List[str]) -> bool:
    tok = AutoTokenizer.from_pretrained(save_dir, trust_remote_code=True)
    vocab = tok.get_vocab()
    missing = [t for t in tokens if t not in vocab]
    if missing:
        print(f"[WARN] 重新加载后仍缺失: {missing}")
        return False
    print("[OK] 重新加载检查通过，所有 token 均存在。")
    return True

def parse_tokens(args) -> List[str]:
    tokens: List[str] = []
    if args.tokens:
        tokens.extend([t.strip() for t in args.tokens.split(",") if t.strip()])
    if args.tokens_file:
        with open(args.tokens_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens.append(line)
    # 去重保持顺序
    seen = set()
    ordered = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered

def main():
    parser = argparse.ArgumentParser(
        description="为 Qwen2.5-VL 模型添加特殊 tokens 并保存到本地。"
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct", help="HF Hub 模型或本地路径")
    parser.add_argument("--save-dir", required=True, help="保存目录")
    parser.add_argument("--tokens", default="", help="逗号分隔 tokens，例如: <loc_x>,<loc_y>")
    parser.add_argument("--tokens-file", help="包含待添加 token 的文本文件（每行一个）")
    parser.add_argument("--init-strategy", default="avg", choices=["avg", "normal", "zero"], help="新增 embedding 初始化策略")
    parser.add_argument("--as-special", action="store_true", help="是否作为 special tokens 添加")
    parser.add_argument("--no-as-special", dest="as_special", action="store_false")
    parser.set_defaults(as_special=True)
    parser.add_argument("--padding-side", default="left", choices=["left", "right"])
    parser.add_argument("--device", default="cuda", help="cuda / cpu / mps / auto")
    args = parser.parse_args()

    tokens = parse_tokens(args)
    if not tokens:
        print("未提供任何 token，可使用 --tokens 或 --tokens-file")
        return

    print(f"[INFO] 待处理 tokens: {tokens}")

    print(f"[INFO] 加载模型: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.padding_side = args.padding_side
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto" if args.device == "auto" else None,
        trust_remote_code=True,
    )


    mapping, added = add_new_tokens(
        model=model,
        tokenizer=tokenizer,
        new_tokens=tokens,
        init_strategy=args.init_strategy,
        as_special=args.as_special,
    )
    print(f"[INFO] 新增数量: {added}")
    print(f"[INFO] Token 映射: {mapping}")

    save_bundle(model, tokenizer, mapping, args.save_dir)

    # 重新验证
    reload_and_check(args.save_dir, tokens)

if __name__ == "__main__":
    main()
