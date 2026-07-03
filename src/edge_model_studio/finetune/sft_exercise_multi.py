import argparse
import json
import os
import time
from datetime import datetime

import swanlab
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)


# ==== 环境变量与全局实验环境初始化 ====
def init_env():
    os.environ["NCCL_P2P_DISABLE"] = "1"  # 禁用NCCL点对点通信
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["SWANLAB_PROJECT"] = "qwen3-sft-running"  # 启动SWANLAB，归档到qwen3-sft-running下
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用 tokenizers 并行警告
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))  # 只在主进程初始化 swanlab
    if local_rank in (-1, 0):  # 只让主进程执行 swanlab.init
        swanlab.init(mode="local")


# ==== 命令行参数解析，必须要每次输入 ====
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="启动的模型名称")
    parser.add_argument("--data_path", type=str, required=True, help="训练集的具体位置和路径")
    parser.add_argument("--output_dir", type=str, required=True, help="训练后模型输出的位置")
    return parser.parse_args()


# ==== 获得tokenizer ====
def get_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return tokenizer


# ==== 获得模型 ====
def get_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        # device_map="auto",  # auto有可能被拆成4卡1模型
        torch_dtype=torch.bfloat16,  # 模型直接按照bf16加载
        use_cache=False  # SFT掐尖关闭KV-cache,否则梯度checkpoint不生效
    )
    model.gradient_checkpointing_enable()
    return model


# ==== SwanLab 配置 ====
def update_swanlab_config(model_name, max_length, batch_size, num_train_epochs, learning_rate):
    swanlab.config.update({
        "model": model_name,
        "task": "跑步问答",
        "data_max_length": max_length,
        "batch_size": batch_size,
        "epochs": num_train_epochs,
        "learning_rate": learning_rate
    })


# ==== 数据加载、处理、切分 ====
def load_and_process_data(data_path, tokenizer, max_length=2048, split_ratio=0.95, enable_thinking=False):
    encoded_prompts = []
    with open(data_path, "r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except Exception as e:
                print(f"[Line {idx}] JSON parse error: {e}")
                continue

            # 获取三段内容，兼容不同字段名
            prompt = sample.get("prompt") or sample.get("exercise_prompt", "")
            metrics = sample.get("metrics") or sample.get("exercise_metrics", "")
            summary = sample.get("summary") or sample.get("exercise_summary", "")

            # system role 用 prompt（每条自己的 system 设定）
            system_msg = {
                "role": "system",
                "content": prompt.strip().replace(" ", "")
            }
            # user role 用 metrics
            user_msg = {
                "role": "user",
                "content": metrics.strip().replace(" ", "")
            }
            # assistant role 用 summary
            assistant_msg = {
                "role": "assistant",
                "content": summary.strip().replace(" ", "")
            }
            messages = [system_msg, user_msg, assistant_msg]

            # 用qwen模板生成prompt字符串, 推理时需要增加续写模版，SFT不要增加续写模版
            prompt_str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=enable_thinking
            )
            # print(f"prompt_str is: {prompt_str}")

            # 数据集处理完成后进行encoding化
            out = tokenizer(
                prompt_str, truncation=True, max_length=max_length, padding="max_length", return_tensors=None)
            out["labels"] = out["input_ids"].copy()
            encoded_prompts.append(out)
    print(f"========== 已加载当前数据集，累计{len(encoded_prompts)}条 ==========")

    # 切分train/test
    train_size = int(len(encoded_prompts) * split_ratio)
    train_data = encoded_prompts[:train_size]
    test_data = encoded_prompts[train_size:]
    print(f"========== train: {len(train_data)} test: {len(test_data)} ==========")

    # 用 HuggingFace Dataset 构建 DatasetDict
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_data),
        "test": Dataset.from_list(test_data)
    })
    return dataset_dict


# ==== 训练参数配置 ====
def build_training_args(output_dir, run_name, batch_size, gradient_accumulation_steps, num_train_epochs, learning_rate,
                        max_length):
    training_args = TrainingArguments(
        output_dir=output_dir,  # 保存路径
        eval_strategy="steps",  # evaluation_strategy XX步跑一遍验证集
        eval_steps=100,
        save_strategy="steps",  # 按照step保存，每XX步一个checkpoint
        save_steps=400,
        per_device_train_batch_size=batch_size,  # train每张卡的批量大小
        per_device_eval_batch_size=batch_size * 2,  # evaluation每张卡批量大小
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        warmup_ratio=0.1,  # 前10%步数线性从0→lr，防止一开始梯度爆炸。
        lr_scheduler_type="cosine",  # 余弦退火，使曲率缓慢下降
        weight_decay=0.01,  # L2正则，防止过拟合
        max_grad_norm=1.0,  # 梯度裁剪阈值，防止梯度爆炸
        logging_steps=10,  # 每10步保存一次loss
        bf16=True,  # 禁止混合精度训练
        report_to="swanlab",  # 日志存储的位置
        run_name=run_name,
        load_best_model_at_end=True,  # 训练结束后自动加载验证集表现最好的checkpoint
        remove_unused_columns=False,  # 去除数据集dataset中除input_ids/labels之外的列
        dataloader_pin_memory=True,  # PyTorch 优化：数据 loader 把张量放到 page-locked 内存区，加速GPU拷贝
        dataloader_num_workers=8,  # DataLoader 后台线程数，I/O 预取更快
        local_rank=int(os.environ.get("LOCAL_RANK", -1))  # 用于 DistributedDataParallel (DDP) 初始化
    )
    return training_args


# ==== 模型训练 ====
def train_model(model, tokenizer, split_dataset, training_args, max_length=2048):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        tokenizer=tokenizer,
        data_collator=None,
    )
    return trainer


# ==== 模型与tokenizer保存 ====
def save_model(model, tokenizer, output_dir):
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print("✅ 训练完成，模型保存至", output_dir)


# ==== 主流程 ====
def main():
    # ==== 固定参数配置 ====
    batch_size = 2  # 每张卡，每次执行2个step
    gradient_accumulation_steps = 4  # 每4个参数更新一次，针对小模型，可以变成8，更加稳定；但耗时更长
    num_train_epochs = 3  # 完整的训练次数
    learning_rate = 1e-5  # 优化器步长
    max_length = 2048  # 入参最大的token数
    split_rate = 0.95  # 划分训练集的比例

    args = parse_args()
    init_env()

    model_name = os.path.basename(args.model_path.rstrip("/"))
    run_name = f"qwen3-running-{time.strftime('%Y%m%d_%H%M%S')}"
    update_swanlab_config(model_name, max_length, batch_size, num_train_epochs, learning_rate)

    tokenizer = get_tokenizer(args.model_path)
    model = get_model(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    split_dataset = load_and_process_data(args.data_path, tokenizer, max_length, split_rate)

    training_args = build_training_args(
        output_dir=args.output_dir,
        run_name=run_name,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        max_length=max_length
    )

    trainer = train_model(model, tokenizer, split_dataset, training_args)

    print(f"=================== start SFT training:{datetime.now()} ===================")
    trainer.train()
    print(f"=================== SFT training successfully:{datetime.now()} ===================")

    print(f"=================== start save models:{datetime.now()} ===================")
    save_model(model, tokenizer, args.output_dir)
    print(f"=================== models save successfully:{datetime.now()} ===================")


if __name__ == "__main__":
    main()
