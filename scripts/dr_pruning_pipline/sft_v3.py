import os
import json
import time
import torch
import argparse
import swanlab
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from torch.distributed import init_process_group, barrier, broadcast_object_list
from torch.nn.parallel import DistributedDataParallel as DDP

# ================= 初始化分布式和环境变量 =================
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

def setup_distributed():
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    init_process_group(backend="nccl")
    return local_rank

local_rank = setup_distributed()
is_main_process = (local_rank == 0)

# ================= 参数配置 =================
BASE_MODEL = "/data2/jwllm/models_origin/Qwen3-0.6B"
DATA_PATH = "/data2/jwllm/datasets/total_dataset/total_train_v2.json"

OUTPUT_BASE_DIR = "/data2/jwllm/models"
PROMPT = "你是一位专业新闻编辑，精通文章写作与摘要技巧，能够准确提炼关键信息，不得虚构、猜测或省略原文中的重要数据，确保摘要真实、客观、精炼。请基于下面内容，生成一段不超过70字的摘要。"

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default=BASE_MODEL)
parser.add_argument("--data_path", type=str, default=DATA_PATH)
parser.add_argument("--output_dir", type=str, default=OUTPUT_BASE_DIR)
args = parser.parse_args()

model_path = args.model_path
data_path = args.data_path
output_dir = args.output_dir
batch_size = 2
gradient_accumulation_steps = 8
num_train_epochs = 1
learning_rate = 1e-4
run_name = f"qwen3-0.6B-{time.strftime('%Y%m%d_%H%M%S')}"

# ================= 初始化 SwanLab（仅主进程） =================
if is_main_process:
    os.environ["SWANLAB_PROJECT"] = "qwen3-sft-news-summary"
    swanlab.init(mode="local")
    swanlab.config.update({
        "model": f"Qwen/{os.path.basename(output_dir.rstrip('/'))}",
        "prompt": PROMPT,
        "data_max_length": 3072,
        "batch_size": batch_size,
        "epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "world_size": int(os.environ.get("WORLD_SIZE", 1))
    })

# ================= 模型加载和封装 DDP =================
device = torch.device("cuda", local_rank)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_bf16_supported() else torch.float32
).to(device)
model.gradient_checkpointing_enable()  # DDP 之前启用
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ================= 数据预处理 =================
def preprocess_function(example):
    PROMPT = example["prompt"]
    source = example["context"]
    target = example["output"]
    prompt = f"<|im_start|>system\n{PROMPT}<|im_end|>\n<|im_start|>user\n{source}<|im_end|>\n<|im_start|>assistant\n"
    prompt_ids = tokenizer(prompt, truncation=True, max_length=2048, padding=False)["input_ids"]
    target_ids = tokenizer(target, truncation=True, max_length=512, padding=False)["input_ids"]
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
    return {"input_ids": input_ids, "labels": labels}

if is_main_process:
    raw_dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = raw_dataset.map(preprocess_function, remove_columns=raw_dataset.column_names, num_proc=8)
    split_dataset = dataset.train_test_split(test_size=0.1)
    broadcast_data = [split_dataset]
else:
    broadcast_data = [None]

barrier()
broadcast_object_list(broadcast_data, src=0)
split_dataset = broadcast_data[0]

# ================= 训练参数设置 =================
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
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    report_to="swanlab" if is_main_process else "none",
    run_name=run_name,
    load_best_model_at_end=True,
    remove_unused_columns=False,
    logging_dir="./swanlog",
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
    ddp_find_unused_parameters=False,
    # gradient_checkpointing=True,
    # torch_compile=True
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ================= 梯度检查（仅主进程） =================
if is_main_process:
    print("===== 梯度检查 =====")
    sample = next(iter(trainer.get_train_dataloader()))
    outputs = model.module(**{k: v.to(device) for k, v in sample.items()})
    loss = outputs.loss
    loss.backward()
    for name, param in model.module.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad_norm={param.grad.data.norm(2).item():.2f}")
    model.zero_grad()
    print("===================")

# ================= 开始训练 =================
trainer.train()

# ================= 模型保存（仅主进程） =================
if is_main_process:
    model.module.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ 模型已保存至：{output_dir}")