from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import bitsandbytes as bnb

# 加载模型（从FP32开始）
# model_id = "Qwen/Qwen3-0.6B"
model_id ="/data2/jwllm/model_process/process_20250613/qwen3_0.6b_sft_round_4"

print("start tokenizer!")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("start model quantization!")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,           # 使用INT8量化
    device_map="auto",           # 自动管理设备映射
    torch_dtype=torch.float16,   # 加载时使用float16作为基础精度
    low_cpu_mem_usage=True
)

# 保存量化模型
print("start save model!")
model.save_pretrained("/data2/jwllm/model_process/process_20250613/qwen3_0.6b_sft_round_4_int8")

print("start save tokenizer!")
tokenizer.save_pretrained("/data2/jwllm/model_process/process_20250613/qwen3_0.6b_sft_round_4_int8")