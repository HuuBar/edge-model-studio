#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检查 HuggingFace 格式 LLM 的参数规模与真实权重 dtype。

用法：
    python inspect_llm.py --model_path /path/to/model
"""

import argparse
import os
from pathlib import Path
from typing import Set

import torch
from transformers import AutoModelForCausalLM

# optional: safetensors
try:
    from safetensors.torch import load_file as safe_load
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


# -------- 文件级扫描：不实例化模型也能看 dtype --------
def scan_weight_files(model_dir: str) -> Set[torch.dtype]:
    """遍历 *.safetensors / *.bin，收集所有张量的 dtype"""
    dtypes: Set[torch.dtype] = set()

    for f in Path(model_dir).glob("*"):
        if f.suffix == ".safetensors" and HAS_SAFETENSORS:
            state_dict = safe_load(f, device="cpu")
        elif f.suffix == ".bin":
            state_dict = torch.load(f, map_location="cpu")
            # 兼容 trainer 保存的 {"state_dict": …}
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
        else:
            continue

        for t in state_dict.values():
            if isinstance(t, torch.Tensor):
                dtypes.add(t.dtype)
    return dtypes


# -------- 实例化模型，统计参数数量 --------
def inspect_model(model_path: str) -> None:
    """
    1. 按 config.json 指定 dtype（若有）加载到 CPU；
       若未指定，则保持磁盘原始 dtype（torch_dtype="auto"）。
    2. 统计参数规模、可训练参数规模与实际 dtype。
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",     # 尊重 config.json 里的 "torch_dtype"
        device_map="cpu",       # 强制只放 CPU，兼容 bf16
        trust_remote_code=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mem_dtypes = {p.dtype for p in model.parameters()}

    print(f"✅ 模型路径: {model_path}")
    print(f"📦 总参数量        : {total_params / 1e6:.2f} M")
    print(f"🛠️  可训练参数量   : {trainable_params / 1e6:.2f} M")
    print(f"💾 内存中张量 dtype : {mem_dtypes}")

    # ------------------ 可选：文件级二次校验 ------------------
    file_dtypes = scan_weight_files(model_path)
    if file_dtypes:
        print(f"🗄️  磁盘权重 dtype   : {file_dtypes}")
        if file_dtypes != mem_dtypes:
            print("⚠️  注意：加载过程中出现 dtype 转换！")



def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect model parameter count and dtype")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    args = parser.parse_args()

    if not torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES"):
        # 避免意外把模型拉到 GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    inspect_model(args.model_path)


if __name__ == "__main__":
    main()
