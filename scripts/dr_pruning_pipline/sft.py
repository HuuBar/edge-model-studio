import os
import time
import argparse
import torch
import swanlab
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

# ==== 环境变量 ====
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

def _check_gradients(model, trainer):
    """验证模型梯度是否正常回传"""
    print("===== Gradient Check =====")
    sample = next(iter(trainer.get_train_dataloader()))
    outputs = model(**{k: v.to(model.device) for k, v in sample.items()})
    outputs.loss.backward()
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            print(f"{name}: grad_norm={grad_norm:.2f}")
            
    model.zero_grad()
    print("==========================")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    # ==== 超参配置 ====
    BATCH_SIZE = 2
    GRAD_ACCUM_STEPS = 8
    EPOCHS = 3
    LR = 1e-4
    MAX_LENGTH = 2048
    
    model_name = os.path.basename(args.model_path.rstrip("/"))
    run_name = f"qwen3-running-{time.strftime('%Y%m%d_%H%M%S')}"

    # ==== 实验追踪配置 (SwanLab) ====
    os.environ["SWANLAB_PROJECT"] = "qwen3-sft-running"
    swanlab.init(
        mode="local",
        config={
            "model": model_name,
            "task": "跑步问答SFT",
            "data_max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LR
        }
    )

    # ==== 模型与 Tokenizer 初始化 ====
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        trust_remote_code=True,
        device_map="auto"  # 替代硬编码的 .cuda()
    )
    model.gradient_checkpointing_enable()

    # ==== 数据预处理 ====
    def preprocess_function(example):
        # 构建 ChatML 格式 Prompt
        prompt = (
            "<|im_start|>system\n你是一个跑步专家，你能理解跑步指标的解析，并会形成200字以内的跑步报告<|im_end|>\n"
            f"<|im_start|>user\n\n{example['exercise_metrics']}\n\n"
            f"# 任务\n请生成一段回复用户提问{example['date']}跑步跑得怎么样？的答复，所有数据和定性判断不能有任何修改，需要固括三个指标。\n\n"
            "# 限制\n不要幻想，不要偷懒，回复字数限制在200字内，语言一定要精简简洁。\n<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

        prompt_ids = tokenizer(prompt, truncation=True, max_length=MAX_LENGTH - 256, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(example["exercise_summary"], truncation=True, max_length=256, add_special_tokens=False)["input_ids"]
        
        # 补齐 EOS token 确保模型学会停止
        input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
        labels = [-100] * len(prompt_ids) + target_ids + [tokenizer.eos_token_id]

        # 注意：此处移除了手动 Padding，交由 DataCollator 动态处理
        return {"input_ids": input_ids, "labels": labels}

    dataset = load_dataset("json", data_files=args.data_path, split="train")
    dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names)
    split_dataset = dataset.train_test_split(test_size=0.1)

    # ==== 训练配置 ====
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=400,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=10,
        fp16=True, 
        report_to="swanlab",
        run_name=run_name,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
    )

    # 使用 DataCollator 实现动态 Padding
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    _check_gradients(model, trainer)

    # ==== 开始训练 ====
    print("🚀 Starting training...")
    trainer.train()

    # ==== 保存制品 ====
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Training complete. Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()