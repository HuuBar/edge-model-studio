import argparse
import os
import time

import swanlab
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

# 禁用NCCL点对点通信
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# 启动SWANLAB，归档到qwen3-sft-running下
os.environ["SWANLAB_PROJECT"] = "qwen3-sft-running"

# 不使用auto，在指令处直接使用
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# 禁用 tokenizers 并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 执行指令
# torchrun --nproc-per-node 4 sft_multi.py --model_path "/data2/jwllm/models_origin/qwen3_0.6b" --data_path "/data2/jwllm/datasets/exercise_generate/exercise_dataset_train.jsonl" --output_dir "/data2/jwllm/model_process/qwen3-0.6b-sft-multi"

# 可以在localhost:xxx查看实验曲线
swanlab.init(mode="local")

# ==== 命令行参数 ====
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

# ==== 参数配置 ====
batch_size = 2  # 每张卡，每次执行2个step
gradient_accumulation_steps = 4  # 每4个参数更新一次，针对小模型，可以变成8，更加稳定；但耗时更长
num_train_epochs = 3  # 完整的训练次数
learning_rate = 1e-5  # 优化器步长
max_length = 2048     # 入参最大的token数
model_name = os.path.basename(args.model_path.rstrip("/"))
run_name = f"qwen3-running-{time.strftime('%Y%m%d_%H%M%S')}"

# ==== SwanLab 配置 ====
swanlab.config.update({
    "model": model_name,
    "task": "跑步问答",
    "data_max_length": max_length,
    "batch_size": batch_size,
    "epochs": num_train_epochs,
    "learning_rate": learning_rate
})

# ==== 模型与 tokenizer ====
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    trust_remote_code=True,
    # device_map="auto",  # auto有可能被拆成4卡1模型
    torch_dtype=torch.bfloat16,  # 模型直接按照bf16加载
    use_cache=False  # SFT掐尖关闭KV-cache,否则梯度checkpoint不生效
)
model.gradient_checkpointing_enable()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ==== 构建输入 ====
def preprocess_function(example):
    # 保持原有代码不变...
    date = example["date"]

    # metrics = example["exercise_metrics"]
    prompt = example["exercise_prompt"]
    summary = example["exercise_summary"]

    prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(summary, truncation=True, max_length=1024, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids

    # Padding 这种截断方式会优先将target截断，而不是prompt截断，这种肯定又会问题，更优的方式应该是优先截断prompt
    pad_len = max_length - len(input_ids)
    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len  # input_ids 后边进行补齐
        labels += [-100] * pad_len  # labels 用-100补齐多出来的部分
    else:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    return {"input_ids": input_ids, "labels": labels}


# ==== HF常用的数据加载方法，读取json/jsonl，按行完成数据处理，随机切分训练集和测试集 ====
dataset = load_dataset("json", data_files=args.data_path, split="train")
dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names)
split_dataset = dataset.train_test_split(test_size=0.1)

# ==== 训练参数 ====
training_args = TrainingArguments(
    output_dir=args.output_dir,  # 保存路径
    eval_strategy="steps",  # evaluation_strategy XX步跑一遍验证集
    eval_steps=100,
    save_strategy="steps",  # 按照step保存，每XX步一个checkpoint
    save_steps=200,
    per_device_train_batch_size=batch_size,  # train每张卡的批量大小
    per_device_eval_batch_size=batch_size * 2,  # evaluation每张卡批量大小
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs,
    learning_rate=learning_rate,
    warmup_ratio=0.1,   # 前10%步数线性从0→lr，防止一开始梯度爆炸。
    lr_scheduler_type="cosine",  # 余弦退火，使曲率缓慢下降
    weight_decay=0.01,  # L2正则，防止过拟合
    max_grad_norm=1.0,  # 梯度裁剪阈值，防止梯度爆炸
    logging_steps=10,   # 每10步保存一次loss
    bf16=True,          # 禁止混合精度训练
    report_to="tensorboard",  # 日志存储的位置
    run_name=run_name,
    load_best_model_at_end=True,   # 训练结束后自动加载验证集表现最好的checkpoint
    remove_unused_columns=False,   # 去除数据集dataset中除input_ids/labels之外的列
    dataloader_pin_memory=True,    # PyTorch 优化：数据 loader 把张量放到 page-locked 内存区，加速GPU拷贝
    dataloader_num_workers=8,      # DataLoader 后台线程数，I/O 预取更快
    # 添加分布式训练参数
    # deepspeed="ds_config.json",  # 可选：使用DeepSpeed配置
    local_rank=os.environ.get("LOCAL_RANK", -1)   # 用于 DistributedDataParallel (DDP) 初始化
)

# ==== Trainer ====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    tokenizer=tokenizer,
)

# ==== 开始训练 ====
trainer.train()

# ==== 保存模型 ====
model.save_pretrained(args.output_dir, safe_serialization=True)
tokenizer.save_pretrained(args.output_dir)
print("✅ 训练完成，模型保存至", args.output_dir)
