#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert fake_quant_weight.pth into a Hugging Face model directory.

Example:

python convert_fakequant_pth_to_hf.py \
  --base-model Qwen/Qwen3-8B \
  --fakequant-pth ./Qwen3-8B/Qwen3-8B/fake_quant_weight.pth \
  --out-dir ./Qwen3-8B-fakequant-hf \
  --torch-dtype bfloat16 \
  --trust-remote-code \
  --verify-forward
"""

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from typing import Dict, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Original HF model path or repo id, e.g. Qwen/Qwen3-8B or /path/to/Qwen3-8B",
    )
    parser.add_argument(
        "--fakequant-pth",
        type=str,
        required=True,
        help="Path to fake_quant_weight.pth",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output HF model directory",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Dtype used to instantiate base model. Use the same dtype as original pipeline if possible.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to transformers.",
    )
    parser.add_argument(
        "--max-shard-size",
        type=str,
        default="2GB",
        help="Shard size for save_pretrained, e.g. 2GB, 5GB.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite out-dir if it already exists.",
    )
    parser.add_argument(
        "--verify-forward",
        action="store_true",
        help="Run a small forward test before and after saving.",
    )
    parser.add_argument(
        "--num-verify-tensors",
        type=int,
        default=20,
        help="Number of tensors to compare after reload.",
    )

    return parser.parse_args()


def resolve_dtype(dtype_str: str):
    if dtype_str == "auto":
        return "auto"
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def safe_torch_load(path: str):
    """
    Prefer weights_only=True if supported by installed PyTorch.
    Fallback for older versions.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def unwrap_state_dict(ckpt):
    """
    Compatible with:
      1. direct state_dict
      2. {"state_dict": state_dict}
      3. {"model": state_dict}
      4. {"model_state_dict": state_dict}
    """
    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint is not a dict. Got: {type(ckpt)}")

    for key in ["state_dict", "model_state_dict", "model", "module"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            print(f"[INFO] Unwrapped checkpoint by key: {key}")
            return ckpt[key]

    return ckpt


def strip_prefix_from_state_dict(sd: Dict[str, torch.Tensor], prefix: str):
    if not prefix:
        return sd
    return {
        k[len(prefix):] if k.startswith(prefix) else k: v
        for k, v in sd.items()
    }


def pick_best_key_format(
    ckpt_sd: Dict[str, torch.Tensor],
    model_sd: Dict[str, torch.Tensor],
) -> Tuple[str, Dict[str, torch.Tensor]]:
    """
    Try common prefixes and pick the one with the most matched keys.
    """
    model_keys = set(model_sd.keys())

    candidates = {
        "raw": ckpt_sd,
        "strip_module.": strip_prefix_from_state_dict(ckpt_sd, "module."),
        "strip_base_model.": strip_prefix_from_state_dict(ckpt_sd, "base_model."),
        "strip_model.": strip_prefix_from_state_dict(ckpt_sd, "model."),
        "strip_base_model_model.": strip_prefix_from_state_dict(
            strip_prefix_from_state_dict(ckpt_sd, "base_model."),
            "model.",
        ),
    }

    scored = []
    for name, sd in candidates.items():
        matched = len(set(sd.keys()) & model_keys)
        scored.append((matched, name, sd))

    scored.sort(reverse=True, key=lambda x: x[0])
    best_matched, best_name, best_sd = scored[0]

    print("[INFO] Key format candidates:")
    for matched, name, _ in scored:
        print(f"  - {name:24s}: matched {matched} / {len(model_keys)}")

    print(f"[INFO] Best key format: {best_name}")
    return best_name, best_sd


def tensor_sha256(t: torch.Tensor) -> str:
    """
    Hash full tensor content on CPU.
    For large tensors this costs some time, but only used for sampled tensors.
    """
    x = t.detach().cpu().contiguous()
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


def print_state_dict_summary(name: str, sd: Dict[str, torch.Tensor], max_items: int = 10):
    print(f"\n========== {name} summary ==========")
    print(f"num keys: {len(sd)}")

    dtype_counter = Counter()
    shape_examples = []

    for k, v in sd.items():
        if torch.is_tensor(v):
            dtype_counter[str(v.dtype)] += 1
            if len(shape_examples) < max_items:
                shape_examples.append((k, tuple(v.shape), str(v.dtype)))
        else:
            dtype_counter[type(v).__name__] += 1

    print("dtype distribution:")
    for dtype, cnt in dtype_counter.most_common():
        print(f"  {dtype}: {cnt}")

    print("examples:")
    for k, shape, dtype in shape_examples:
        print(f"  {k}: shape={shape}, dtype={dtype}")


def check_keys_and_shapes(
    ckpt_sd: Dict[str, torch.Tensor],
    model_sd: Dict[str, torch.Tensor],
):
    ckpt_keys = set(ckpt_sd.keys())
    model_keys = set(model_sd.keys())

    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)

    shape_mismatch = []
    for k in sorted(model_keys & ckpt_keys):
        v1 = ckpt_sd[k]
        v2 = model_sd[k]
        if torch.is_tensor(v1) and torch.is_tensor(v2):
            if tuple(v1.shape) != tuple(v2.shape):
                shape_mismatch.append(
                    (k, tuple(v1.shape), tuple(v2.shape))
                )

    print("\n========== key / shape check ==========")
    print(f"model keys     : {len(model_keys)}")
    print(f"checkpoint keys: {len(ckpt_keys)}")
    print(f"matched keys   : {len(model_keys & ckpt_keys)}")
    print(f"missing keys   : {len(missing)}")
    print(f"unexpected keys: {len(unexpected)}")
    print(f"shape mismatch : {len(shape_mismatch)}")

    if missing:
        print("\n[ERROR] Missing key examples:")
        for k in missing[:30]:
            print(f"  {k}")

    if unexpected:
        print("\n[ERROR] Unexpected key examples:")
        for k in unexpected[:30]:
            print(f"  {k}")

    if shape_mismatch:
        print("\n[ERROR] Shape mismatch examples:")
        for k, s1, s2 in shape_mismatch[:30]:
            print(f"  {k}: ckpt={s1}, model={s2}")

    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            "Checkpoint does not exactly match base model. "
            "Do not save. Please check base model path, checkpoint source, or key prefixes."
        )


def select_verify_keys(model_sd: Dict[str, torch.Tensor], num_keys: int):
    """
    Pick a stable set of representative keys:
      - embeddings
      - first layer
      - middle layer
      - last layer
      - lm_head if exists
    """
    keys = list(model_sd.keys())
    selected = []

    preferred_substrings = [
        "embed_tokens.weight",
        "layers.0.",
        "layers.1.",
        "layers.10.",
        "layers.20.",
        "layers.30.",
        "norm.weight",
        "lm_head.weight",
    ]

    for sub in preferred_substrings:
        for k in keys:
            if sub in k and k not in selected and torch.is_tensor(model_sd[k]):
                selected.append(k)
                break

    for k in keys:
        if len(selected) >= num_keys:
            break
        if torch.is_tensor(model_sd[k]) and k not in selected:
            selected.append(k)

    return selected[:num_keys]


@torch.no_grad()
def run_forward_check(model, tokenizer, device: str = "cpu"):
    model.eval()

    text = "你好，简单介绍一下你自己。"
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    logits = outputs.logits.detach().cpu()

    info = {
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
        "logits_mean": float(logits.float().mean().item()),
        "logits_std": float(logits.float().std().item()),
        "logits_abs_max": float(logits.float().abs().max().item()),
    }

    return info


def main():
    args = parse_args()

    print("========== arguments ==========")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    if not os.path.exists(args.fakequant_pth):
        raise FileNotFoundError(f"fakequant pth not found: {args.fakequant_pth}")

    if os.path.exists(args.out_dir):
        if args.overwrite:
            print(f"[WARN] Removing existing out-dir: {args.out_dir}")
            shutil.rmtree(args.out_dir)
        else:
            raise FileExistsError(
                f"out-dir already exists: {args.out_dir}. "
                f"Use --overwrite if you want to replace it."
            )

    dtype = resolve_dtype(args.torch_dtype)

    print("\n========== load base config ==========")
    config = AutoConfig.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"model_type: {getattr(config, 'model_type', None)}")
    print(f"architectures: {getattr(config, 'architectures', None)}")
    print(f"vocab_size: {getattr(config, 'vocab_size', None)}")
    print(f"hidden_size: {getattr(config, 'hidden_size', None)}")
    print(f"num_hidden_layers: {getattr(config, 'num_hidden_layers', None)}")

    print("\n========== load base model ==========")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()

    model_sd = model.state_dict()
    print_state_dict_summary("base model state_dict before loading fake quant", model_sd)

    print("\n========== load fake quant pth ==========")
    ckpt = safe_torch_load(args.fakequant_pth)
    ckpt_sd = unwrap_state_dict(ckpt)

    if not isinstance(ckpt_sd, dict):
        raise TypeError(f"Unwrapped checkpoint is not dict. Got: {type(ckpt_sd)}")

    print_state_dict_summary("fake quant checkpoint raw", ckpt_sd)

    _, ckpt_sd = pick_best_key_format(ckpt_sd, model_sd)
    print_state_dict_summary("fake quant checkpoint normalized", ckpt_sd)

    check_keys_and_shapes(ckpt_sd, model_sd)

    print("\n========== dtype compatibility check ==========")
    ckpt_dtype_counter = Counter()
    model_dtype_counter = Counter()

    for k, v in ckpt_sd.items():
        if torch.is_tensor(v):
            ckpt_dtype_counter[str(v.dtype)] += 1

    for k, v in model_sd.items():
        if torch.is_tensor(v):
            model_dtype_counter[str(v.dtype)] += 1

    print("checkpoint dtype distribution:")
    for k, v in ckpt_dtype_counter.most_common():
        print(f"  {k}: {v}")

    print("model dtype distribution:")
    for k, v in model_dtype_counter.most_common():
        print(f"  {k}: {v}")

    print(
        "\n[INFO] If checkpoint dtype differs from model dtype, "
        "load_state_dict will cast checkpoint tensors to model parameter dtype. "
        "This matches normal PyTorch behavior."
    )

    print("\n========== load fake quant weights into base model ==========")
    ret = model.load_state_dict(ckpt_sd, strict=True)
    print(f"load_state_dict result: {ret}")

    loaded_sd = model.state_dict()

    verify_keys = select_verify_keys(loaded_sd, args.num_verify_tensors)
    print("\n========== pre-save tensor hash ==========")
    pre_hash = {}
    for k in verify_keys:
        pre_hash[k] = tensor_sha256(loaded_sd[k])
        print(f"{k}: {pre_hash[k]}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
    )

    forward_before = None
    if args.verify_forward:
        print("\n========== forward check before save ==========")
        forward_before = run_forward_check(model, tokenizer)
        print(json.dumps(forward_before, ensure_ascii=False, indent=2))

    print("\n========== save_pretrained ==========")
    model.save_pretrained(
        args.out_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(args.out_dir)

    print(f"[INFO] Saved to: {args.out_dir}")

    print("\n========== reload saved HF model ==========")
    reloaded_model = AutoModelForCausalLM.from_pretrained(
        args.out_dir,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    reloaded_model.eval()

    reloaded_sd = reloaded_model.state_dict()

    print("\n========== post-save reload verification ==========")
    errors = []

    for k in verify_keys:
        h1 = pre_hash[k]
        h2 = tensor_sha256(reloaded_sd[k])

        same_hash = h1 == h2

        a = loaded_sd[k].detach().cpu()
        b = reloaded_sd[k].detach().cpu()

        max_abs_diff = float((a.float() - b.float()).abs().max().item())

        print(
            f"{k}: same_hash={same_hash}, "
            f"max_abs_diff={max_abs_diff}, "
            f"shape={tuple(b.shape)}, dtype={b.dtype}"
        )

        if not same_hash or max_abs_diff != 0.0:
            errors.append((k, h1, h2, max_abs_diff))

    if errors:
        print("\n[ERROR] Tensor mismatch after reload:")
        for k, h1, h2, diff in errors:
            print(f"  {k}: pre={h1}, post={h2}, max_abs_diff={diff}")
        raise RuntimeError("Saved model reload verification failed.")

    forward_after = None
    if args.verify_forward:
        print("\n========== forward check after reload ==========")
        forward_after = run_forward_check(reloaded_model, tokenizer)
        print(json.dumps(forward_after, ensure_ascii=False, indent=2))

        print("\n========== forward check diff ==========")
        for k in forward_before:
            if isinstance(forward_before[k], (int, float)):
                diff = abs(forward_before[k] - forward_after[k])
                print(f"{k}: before={forward_before[k]}, after={forward_after[k]}, diff={diff}")
            else:
                print(f"{k}: before={forward_before[k]}, after={forward_after[k]}")

    print("\n========== final check ==========")
    required_files = [
        "config.json",
        "tokenizer_config.json",
    ]

    for f in required_files:
        path = os.path.join(args.out_dir, f)
        print(f"{f}: {'OK' if os.path.exists(path) else 'MISSING'}")

    has_safetensors = any(
        name.endswith(".safetensors")
        for name in os.listdir(args.out_dir)
    )
    print(f"safetensors weights: {'OK' if has_safetensors else 'MISSING'}")

    if not has_safetensors:
        raise RuntimeError("No .safetensors file found in output directory.")

    print("\n[SUCCESS] Conversion finished and verification passed.")
    print(f"[SUCCESS] HF fake quant model directory: {args.out_dir}")


if __name__ == "__main__":
    main()