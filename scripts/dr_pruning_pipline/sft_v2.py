import os
import time
import torch
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset

# 初始化分布式训练
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
dist.init_process_group(backend="nccl")
PROMPT = "你是一位专业新闻编辑，精通文章写作与摘要技巧，能够准确提炼关键信息，不得虚构、猜测或省略原文中的重要数据，确保摘要真实、客观、精炼。请基于下面内容，生成一段不超过70字的摘要。"

# === 命令行参数 ===
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

# 设备设置 
local_rank = int(os.environ["LOCAL_RANK"])  
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

# 加载模型和 Tokenizer
model_path = args.model_path
data_path = args.data_path
output_dir = args.output_dir
batch_size = 4 # 适当调整
gradient_accumulation_steps = 8
num_train_epochs = 3
learning_rate = 1e-4

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
model.gradient_checkpointing_enable()

# 使用 DDP 进行多 GPU 训练
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# 预处理函数
def preprocess_function(example):
    source_text = example["source"]
    target_text = example["ground_truth"]
    prompt = f"<|im_start|>system\n{PROMPT}<|im_end|>\n<|im_start|>user\n{source_text}<|im_end|>\n<|im_start|>assistant\n"
    prompt_ids = tokenizer(prompt, truncation=True, max_length=2048, padding=False, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text, truncation=True, max_length=512, padding=False, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids

    max_len = 2048
    pad_len = max_len - len(input_ids)
    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len
        labels += [-100] * pad_len
    else:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]

    return {
        "input_ids": input_ids,
        "labels": labels
    }

# 修改预处理函数，添加缓存
dataset = load_dataset("json", data_files=data_path, split="train")
dataset = dataset.select(range(19000))
dataset = dataset.map(
    preprocess_function, 
    remove_columns=dataset.column_names,
    load_from_cache_file=True,  # 启用缓存
    num_proc=8,  # 增加预处理进程数
    batch_size=1000  # 批量处理
)

# # 加载数据集
# dataset = load_dataset("json", data_files=data_path, split="train")
# dataset = dataset.select(range(19000))
# dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names)
split_dataset = dataset.train_test_split(test_size=0.1)

# 训练参数
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=400,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    max_grad_norm=5.0,
    num_train_epochs=num_train_epochs,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_torch_fused",
    fp16=True,
    logging_steps=10,
    report_to="none",  # 可修改日志工具
    run_name=f"qwen3-0.6B-{time.strftime('%Y%m%d_%H%M%S')}",  # 日志时间戳
    load_best_model_at_end=True,
    remove_unused_columns=False,
    logging_dir="./logs",
    dataloader_pin_memory=True,
    dataloader_num_workers=8,
    ddp_find_unused_parameters=False,  # 适用于 DDP
    dataloader_prefetch_factor=4,  # 增加预取因子
    gradient_checkpointing=True,  # 确保启用
    torch_compile=True  # 启用模型编译优化
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    tokenizer=tokenizer,
)

# 梯度检查
print("===== 梯度检查 =====")
sample = next(iter(trainer.get_train_dataloader()))
outputs = model(**{k: v.to(model.device) for k, v in sample.items()})
loss = outputs.loss
loss.backward()
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.data.norm(2).item()
        print(f"{name}: grad_norm={grad_norm:.2f}")
model.zero_grad()
print("===================")

# 开始训练
trainer.train()

# 保存模型
model = model.to(torch.float16)
model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)
print("Training completed and model saved to", output_dir)
