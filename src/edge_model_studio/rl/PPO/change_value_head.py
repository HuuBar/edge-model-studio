#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为 HuggingFace 格式的 CausalLM 添加 value head，
并 **保持原始权重的 dtype 与存储格式**（safetensors / bin）。

需要 pip install trl[s3] transformers safetensors
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

# ------------------------------------------------------------------
# 🔧 修复 unwrap_model 报错（与原脚本一致）
import transformers
from transformers.modeling_utils import unwrap_model
import types


def _patched_unwrap_model(_, model, *a, **kw):
    if hasattr(model, "pretrained_model"):
        return model.pretrained_model
    return model


transformers.modeling_utils.unwrap_model = types.MethodType(
    _patched_unwrap_model, unwrap_model
)


# ------------------------------------------------------------------


def detect_original_dtype(model_dir: Path) -> torch.dtype:
    """
    1. 先看 config.json 里的 "torch_dtype"（推荐）。
    2. 若缺失，再扫首个权重文件，取出现频次最高的 dtype。
    """
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        if "torch_dtype" in cfg:
            try:
                return getattr(torch, cfg["torch_dtype"])
            except AttributeError:
                pass  # 非标准取值，继续后备方案

    # 后备：读第一个权重文件
    first_file: Optional[Path] = None
    for suf in (".safetensors", ".bin"):
        files = sorted(model_dir.glob(f"*{suf}"))
        if files:
            first_file = files[0]
            break

    if first_file is None:
        raise FileNotFoundError(f"⚠️ 未在 {model_dir} 找到权重文件！")

    if first_file.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(first_file, device="cpu", framework="pt")
    else:  # .bin
        sd = torch.load(first_file, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

    # 统计最多的 dtype
    dtype_counts = {}
    for t in sd.values():
        if isinstance(t, torch.Tensor):
            dtype_counts[t.dtype] = dtype_counts.get(t.dtype, 0) + 1
    return max(dtype_counts, key=dtype_counts.get)


def is_safe_serialization(model_dir: Path) -> bool:
    """若目录里有 *.safetensors 文件，则认为用 safetensors 存储"""
    return any(p.suffix == ".safetensors" for p in model_dir.iterdir())


def cast_module_dtype(module: torch.nn.Module, target_dtype: torch.dtype):
    """递归把 module 里的所有参数 / buffer 转成 target_dtype"""
    for param in module.parameters(recurse=True):
        param.data = param.data.to(target_dtype)
    for buf_name, buf in module._buffers.items():
        if isinstance(buf, torch.Tensor):
            module._buffers[buf_name] = buf.to(target_dtype)


def add_value_head_to_model(source: str, target: str):
    src_dir = Path(source).expanduser()
    tgt_dir = Path(target).expanduser()
    tgt_dir.mkdir(parents=True, exist_ok=True)

    # 1️⃣ 判定原始 dtype 与权重格式
    original_dtype = detect_original_dtype(src_dir)
    use_safe = is_safe_serialization(src_dir)
    print(f"🔍 检测到原始模型 dtype = {original_dtype}, "
          f"文件格式 = {'safetensors' if use_safe else 'bin'}")

    # 2️⃣ 加载模型 + tokenizer
    print(f"🔄 正在加载原始模型：{src_dir}")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        str(src_dir),
        torch_dtype=original_dtype,
        device_map="cpu",  # 全放 CPU，避免 GPU 不支持的 dtype 自动 cast
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(src_dir), trust_remote_code=True)

    # 3️⃣ value head 本身会默认以 fp32 新建 → 手动 cast 回原始 dtype
    if hasattr(model, "v_head"):
        cast_module_dtype(model.v_head, original_dtype)

    # 4️⃣ 保存
    print(f"💾 正在保存带 value head 的模型到：{tgt_dir}")
    model.save_pretrained(
        tgt_dir,
        safe_serialization=use_safe,  # 跟随原始格式
    )
    tokenizer.save_pretrained(tgt_dir)

    # 5️⃣ 同步修改 config.json 的 torch_dtype（有时 trl 不会改）
    cfg_path = tgt_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg["torch_dtype"] = str(original_dtype).replace("torch.", "")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

    print("✅ 保存完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="为预训练模型添加 value head 并保持原始精度/格式"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="原始模型路径",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="输出模型路径",
    )
    args = parser.parse_args()

    add_value_head_to_model(args.source, args.target)
