import os
import sys
import argparse
import logging
import torch
import transformers
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

# 规范化日志格式，移除繁琐的 Emoji，保持纯文本服务器日志规范
logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# 动态拦截并修复 TRL 与 Transformers 的 unwrap_model 递归提取 bug
_orig_unwrap = transformers.modeling_utils.unwrap_model

def _patched_unwrap(model, *args, **kwargs):
    if hasattr(model, 'pretrained_model'):
        return model.pretrained_model
    return _orig_unwrap(model, *args, **kwargs)

transformers.modeling_utils.unwrap_model = _patched_unwrap


def init_and_calibrate_value_head(model):
    """
    针对新挂载的 Value Head 进行防御性初始化。
    防止 trl 默认的高斯随机初始化导致 PPO 训练初期 Critic 产生离群极值（Spike），
    引发梯度崩塌。
    """
    if hasattr(model, 'v_head'):
        logging.info("Calibrating newly attached Value Head weights...")
        # 针对常规线性层进行 Xavier 或小方差正态初始化，使初始 Value 预测尽量收拢在 0 附近
        with torch.no_grad():
            if hasattr(model.v_head, 'summary'):
                # trl 的结构通常是 v_head.summary (nn.Linear)
                torch.nn.init.orthogonal_(model.v_head.summary.weight, gain=0.1)
                if model.v_head.summary.bias is not None:
                    torch.nn.init.constant_(model.v_head.summary.bias, 0.0)
            elif isinstance(model.v_head, torch.nn.Linear):
                torch.nn.init.orthogonal_(model.v_head.weight, gain=0.1)
                if model.v_head.bias is not None:
                    torch.nn.init.constant_(model.v_head.bias, 0.0)
        logging.info("Value Head weights normalized successfully.")
    else:
        logging.warning("v_head attribute not detected in the current model wrapper.")


def main(source_path: str, target_path: str):
    if not os.path.exists(source_path):
        logging.error(f"Source model path does not exist: {source_path}")
        sys.exit(1)

    logging.info(f"Loading base tokenizer and model from: {source_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(source_path, use_fast=False)
        
        # 强行指定精度类型，避免其隐式回落到 float32 撑爆显存或导致显存碎片化
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            source_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
    except Exception as e:
        logging.error(f"Failed to compile model architecture: {e}")
        sys.exit(1)

    # 触发核心参数初始化对齐
    init_and_calibrate_value_head(model)

    os.makedirs(target_path, exist_ok=True)
    logging.info(f"Serializing value-head model down to disk: {target_path}")
    
    try:
        # 1. 优先通过 trl 原生接口持久化模型
        model.save_pretrained(target_path)
        tokenizer.save_pretrained(target_path)
        
        # 2. 防御性工程闭环：显式对 v_head 专属状态字典（State Dict）进行隔离备份。
        # 这样即使未来 trl 核心框架升级破坏了兼容性，PPO 脚本也能通过独立加载该 bin 稳健运行。
        if hasattr(model, 'v_head'):
            v_head_state = model.v_head.state_dict()
            v_head_bin_path = os.path.join(target_path, "value_head.bin")
            torch.save(v_head_state, v_head_bin_path)
            logging.info(f"Independent Value Head state dict saved safely to: {v_head_bin_path}")

        logging.info("[SUCCESS] Model conversion accomplished safely.")
    except Exception as e:
        logging.error(f"Failed to checkpoint serialization pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRL Value-Head Model Initialization Pipeline")
    parser.add_argument(
        "--source", "-s", 
        type=str, 
        default="/data2/jwllm/models_origin/qwen3_0.6b",
        help="Path to the original causal LM directory"
    )
    parser.add_argument(
        "--target", "-t", 
        type=str, 
        default="/data2/jwllm/model_process/qwen3_0.6b_valuehead",
        help="Target storage directory for RL initialization"
    )

    args = parser.parse_args()
    main(args.source, args.target)