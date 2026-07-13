#!/usr/bin/env python3
"""
SFT多卡最小可复现测试
基于 transformers Trainer + torchrun，不依赖accelerate/trl

用法:
    torchrun --nproc_per_node=8 test_sft_multi_npu.py
"""

import os
import torch
import torch.distributed as dist
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")

    if rank == 0:
        print(f"🚀 SFT多卡测试启动: world_size={world_size}")

    # 用超小模型测试，避免下载大模型
    model_name = "Qwen/Qwen2.5-0.5B"
    if rank == 0:
        print(f"📥 Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(f"npu:{local_rank}")

    # 构造假数据
    def make_dummy_data():
        texts = ["这是一个测试样本，用于验证NPU多卡SFT训练。" * 10] * 32
        return Dataset.from_dict({"text": texts})

    dataset = make_dummy_data()

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # 训练参数 - 只跑几步验证
    args = TrainingArguments(
        output_dir="/tmp/test_sft_npu",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        max_steps=5,  # 只跑5步
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        ddp_backend="hccl",
        local_rank=local_rank,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    if rank == 0:
        print("🏃 开始训练...")

    trainer.train()

    dist.barrier()
    if rank == 0:
        print("\n🎉 SFT多卡训练测试通过!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
