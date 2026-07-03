#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="原始 base model，例如 Qwen/Qwen3-8B 或本地 Qwen3-8B 目录",
    )
    parser.add_argument(
        "--fakequant-pth",
        type=str,
        required=True,
        help="fake_quant_weight.pth 路径",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="你好，简单介绍一下你自己。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="构造 base_model 时使用的 dtype。要尽量和原始训练/生成代码一致。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda / cpu",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
    )
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="是否使用 tokenizer.apply_chat_template",
    )
    parser.add_argument(
        "--map-location",
        type=str,
        default="none",
        choices=["none", "cpu"],
    )

    return parser.parse_args()


def get_dtype(dtype_str):
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    raise ValueError(dtype_str)


def print_tensor_stats(name, tensor):
    x = tensor.detach().float().cpu()
    print(f"{name}:")
    print(f"  shape   : {tuple(tensor.shape)}")
    print(f"  dtype   : {tensor.dtype}")
    print(f"  mean    : {x.mean().item()}")
    print(f"  std     : {x.std().item()}")
    print(f"  min     : {x.min().item()}")
    print(f"  max     : {x.max().item()}")
    print(f"  absmax  : {x.abs().max().item()}")
    print(f"  finite  : {torch.isfinite(x).all().item()}")


def print_model_weight_stats(model):
    print("\n========== selected weight stats after load_state_dict ==========")

    sd = model.state_dict()

    keys = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.35.self_attn.q_proj.weight",
        "lm_head.weight",
    ]

    for k in keys:
        if k in sd:
            print_tensor_stats(k, sd[k])
        else:
            print(f"{k}: MISSING")


@torch.no_grad()
def print_logits_stats(model, tokenizer, input_ids):
    print("\n========== forward logits check ==========")

    outputs = model(input_ids=input_ids)
    logits = outputs.logits

    print_tensor_stats("logits", logits)

    last_logits = logits[0, -1].detach().float().cpu()
    values, indices = torch.topk(last_logits, k=20)

    print("\nTop-20 next token logits:")
    for rank, (v, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        token = tokenizer.decode([idx])
        print(f"{rank:02d}. id={idx:8d}, logit={v:12.4f}, token={repr(token)}")


def main():
    args = parse_args()

    print("========== arguments ==========")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    if not os.path.exists(args.fakequant_pth):
        raise FileNotFoundError(args.fakequant_pth)

    dtype = get_dtype(args.dtype)

    print("\n========== load tokenizer ==========")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
    )

    print("\n========== load base model ==========")
    print(f"base_model = {args.base_model}")
    print(f"dtype      = {dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        device_map=None,
    )
    model.eval()

    print("\n========== direct load fake quant pth ==========")
    print(f"LOAD_BASE_FAKEQUANT={os.getenv('LOAD_BASE_FAKEQUANT', 'false')}")

    if os.getenv("LOAD_BASE_FAKEQUANT", "false") == "true":
        if args.map_location == "none":
            state_dict = torch.load(args.fakequant_pth)
        else:
            state_dict = torch.load(args.fakequant_pth, map_location=args.map_location)

        msg = model.load_state_dict(state_dict)
        print("load fake quant base done")
        print(f"load_state_dict msg: {msg}")
    else:
        print("skip fake quant pth loading because LOAD_BASE_FAKEQUANT != true")

    print_model_weight_stats(model)

    print("\n========== move model to device ==========")
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    print(f"model device: {next(model.parameters()).device}")

    print("\n========== build input ==========")

    if args.use_chat_template:
        messages = [
            {"role": "user", "content": args.prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = args.prompt

    print("input text:")
    print(text)

    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    print(f"input_ids shape: {tuple(input_ids.shape)}")
    print(f"input_ids: {input_ids[0].tolist()}")

    print_logits_stats(model, tokenizer, input_ids)

    print("\n========== generate ==========")

    generated = model.generate(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tokenizer.eos_token_id,
    )

    output_ids = generated[0].tolist()
    new_ids = generated[0, input_ids.shape[1]:].tolist()

    print("\nfull output ids:")
    print(output_ids)

    print("\nnew token ids:")
    print(new_ids)

    print("\n========== decoded full output ==========")
    print(tokenizer.decode(generated[0], skip_special_tokens=False))

    print("\n========== decoded new output only ==========")
    print(tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=False))


if __name__ == "__main__":
    main()