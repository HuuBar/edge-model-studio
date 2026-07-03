import argparse
import sys
import gc
import os
import subprocess
import json
import numpy as np
import torch

from modeling_mole import MoleForCausalLM
from modeling_mole_rep import MoleForCausalLM as MoleForCausalLM_rep


def auto_select_gpu(min_free_mb=1000):
    """
    根据显存剩余自动选择单卡，若失败或不满足则转为默认行为。
    老老实实解析 nvidia-smi，避免引入外部重度依赖。
    """
    try:
        cmd = ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader']
        res = subprocess.check_output(cmd, encoding='utf-8')
        
        gpu_stats = []
        for idx, line in enumerate(res.strip().split('\n')):
            if not line.strip():
                continue
            used, total = map(int, line.split(','))
            gpu_stats.append((idx, total - used))
            
        if not gpu_stats:
            return

        best_idx, max_free = max(gpu_stats, key=lambda x: x[1])
        if max_free > min_free_mb:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best_idx)
            print(f"[INFO] Selected GPU {best_idx} with {max_free} MiB free memory.")
        else:
            print(f"[WARN] No GPU has free memory > {min_free_mb} MiB. Using default environment.")
    except Exception as e:
        print(f"[WARN] Failed to auto-select GPU: {e}. Relying on system default.")


def parse_args():
    parser = argparse.ArgumentParser(description="MoE Model Reparameterization & LUT Extraction")
    parser.add_argument('--from_path', type=str, required=True, help='Path to original model')
    parser.add_argument('--to_path', type=str, required=True, help='Path to save reparameterized model')
    parser.add_argument('--lut_name', type=str, default='moe_table.dat', help='Output LUT binary filename')
    parser.add_argument('--chunk_size', type=int, default=4096, help='Chunk size for token embeddings if OOM occurs')
    return parser.parse_args()


def main():
    args = parse_args()
    auto_select_gpu()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    os.makedirs(args.to_path, exist_ok=True)
    lut_path = os.path.join(args.to_path, args.lut_name)

    # 1. 提取原始模型专家输出并构建 LUT
    print(f"[INFO] Loading original model from: {args.from_path}")
    # 明确指定不采用默认的 device_map 强绑定到 cuda:0，方便后续自由移动 expert
    model = MoleForCausalLM.from_pretrained(
        args.from_path, 
        torch_dtype=torch.float16, 
        local_files_only=True
    ).to(device)
    model.eval()
    
    config = model.config
    num_layers = config.num_hidden_layers
    num_experts = config.num_experts

    print("[INFO] Preparing token embeddings...")
    with torch.no_grad():
        # 这里提取基础 embedding 并过第一层 norm
        raw_embeds = model.model.embed_tokens.weight.to(device)
        token_embeds = model.model.layers[0].expert_layernorm(raw_embeds)
        
        vocab_size, hidden_size = token_embeds.shape
        print(f"[INFO] Matrix shape: Vocab={vocab_size}, Hidden={hidden_size}")
        print(f"[INFO] Initializing memmap LUT: {lut_path}")

        # 初始化 memmap
        lut_memmap = np.memmap(
            lut_path, 
            dtype=np.float32, 
            mode='w+', 
            shape=(vocab_size, num_layers, num_experts, hidden_size)
        )

        # 核心抽取循环
        for l_idx, layer in enumerate(model.model.layers):
            print(f"[PROCESSING] Layer {l_idx + 1}/{num_layers}")
            
            for e_idx, expert in enumerate(layer.experts):
                # 防御性显存清理
                torch.cuda.empty_cache()
                
                # 若 vocab 过大，支持分 chunk 推理防止 OOM
                outputs = []
                for i in range(0, vocab_size, args.chunk_size):
                    chunk = token_embeds[i:i + args.chunk_size]
                    chunk_out = expert(chunk).cpu().numpy().astype(np.float32)
                    outputs.append(chunk_out)
                
                lut_memmap[:, l_idx, e_idx] = np.concatenate(outputs, axis=0)
        
        lut_memmap.flush()
        del lut_memmap
        print("[INFO] LUT extraction completed successfully.")

    # 彻底释放原始模型显存
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 2. 加载重参数化模型结构并重构权重
    print(f"[INFO] Loading reparameterized model architecture...")
    model_rep = MoleForCausalLM_rep.from_pretrained(
        args.from_path,
        torch_dtype=torch.float32,
        local_files_only=True
    )

    # 注入 LUT 配置参数
    model_rep.config.lut_path = args.lut_name
    model_rep.config.use_memmap_lut = True

    # 清理不必要的 moe_table 字段以减小体积
    print("[INFO] Filtering state dict (removing explicit moe_table weights)...")
    if hasattr(model_rep.model, "moe_table"):
        delattr(model_rep.model, "moe_table")

    state_dict = model_rep.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("model.moe_table")}

    # 保存精简后的 bin 权重和 config
    torch.save(filtered_state_dict, os.path.join(args.to_path, "pytorch_model.bin"))
    model_rep.config.save_pretrained(args.to_path)
    
    # 额外导出一份显式的 json 元数据供推理端/C++ 绑核直接校验
    lut_info = {
        "lut_path": args.lut_name,
        "shape": [vocab_size, num_layers, num_experts, hidden_size],
        "dtype": "float32"
    }
    with open(os.path.join(args.to_path, "lut_config.json"), "w") as f:
        json.dump(lut_info, f, indent=2)

    print(f"[SUCCESS] Reparameterization done. Target: {args.to_path}")

    # 3. 闭环精度与结构验证
    print("\n" + "="*40 + "\n[VERIFICATION] Running sanity check...")
    try:
        meta = np.memmap(lut_path, dtype=np.float32, mode='r')
        expected_elements = vocab_size * num_layers * num_experts * hidden_size
        print(f" -> LUT File Elements: {meta.size} (Expected: {expected_elements})")
        print(f" -> Model Param Precision: {next(model_rep.parameters()).dtype}")
        if meta.size == expected_elements:
            print("[VERIFICATION] Pass: Size and shapes match.")
        else:
            print("[ERROR] Fail: Data size mismatch.")
    except Exception as e:
        print(f"[ERROR] Verification failed with exception: {e}")


if __name__ == "__main__":
    main()