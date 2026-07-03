import argparse
import json
import os
import sys
import time
import subprocess
from datetime import datetime

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
)
from modeling_mole import MoleForCausalLM


def auto_select_gpu(min_free_mb=1000):
    """
    针对单卡任务自动分配合适的 GPU。
    如果检测到已经在多卡分布式环境（如 torchrun），则跳过自动选择，避免冲突。
    """
    if "RANK" in os.environ or "LOCAL_RANK" in os.environ:
        return

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
            print(f"[WARN] No GPU with free memory > {min_free_mb} MiB. Using default setting.")
    except Exception as e:
        print(f"[WARN] Failed to auto-select GPU: {e}. Relying on system default.")


def parse_args():
    parser = argparse.ArgumentParser(description="MOLE SFT Training Pipeline")
    parser.add_argument("--model_path", type=str, required=True, help="Pretrained model directory")
    parser.add_argument("--data_path", type=str, required=True, help="Path to jsonl training data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length")
    return parser.parse_args()


def get_tokenizer(model_path):
    print(f"[INFO] Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        # MOLE 默认使用 <|endoftext|> 作为 pad
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_model(model_path, tokenizer):
    print(f"[INFO] Loading model architecture from: {model_path}")
    # SFT 阶段根据需要调整精度，原逻辑保留 float32
    model = MoleForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        use_cache=False
    )
    
    model.resize_token_embeddings(len(tokenizer))
    
    if hasattr(model, "gradient_checkpointing_enable"):
        print("[INFO] Enabling gradient checkpointing to save VRAM.")
        model.gradient_checkpointing_enable()
        
    return model


def tokenize_and_mask_labels(tokenizer, prompt, metrics, summary, max_length):
    """
    精确计算 Prefix 长度，防止 Pad 数量反推带来的边界错误。
    只对真正的 target (summary) 部分计算 Loss。
    """
    prefix_text = f"系统：{prompt.strip()}\n指标：{metrics.strip()}\n报告："
    full_text = f"{prefix_text}{summary.strip()}"

    # 不在编码阶段做填充，先拿到真实文本序列长度
    full_enc = tokenizer(full_text, truncation=True, max_length=max_length)
    prefix_enc = tokenizer(prefix_text, truncation=False)

    input_ids = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]
    
    prefix_len = len(prefix_enc["input_ids"])
    
    # 极端边界防御：如果整个 prompt 超过了 max_length，直接丢弃该样本
    if prefix_len >= max_length or prefix_len >= len(input_ids):
        return None

    # 构建 Labels: Prompt 部分用 -100 屏蔽，后面保留实际 Token ID
    labels = [-100] * prefix_len + input_ids[prefix_len:]

    # 手动进行右侧 Padding，保持原逻辑的统一矩阵大小
    pad_len = max_length - len(input_ids)
    if pad_len > 0:
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        labels = labels + [-100] * pad_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def load_and_process_data(data_path, tokenizer, max_length, split_ratio=0.95):
    print(f"[INFO] Loading dataset from: {data_path}")
    encoded_data = []
    
    if not os.path.exists(data_path):
        print(f"[CRITICAL] Data path not found: {data_path}")
        sys.exit(1)

    with open(data_path, "r", encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                prompt = sample.get("prompt") or sample.get("exercise_prompt", "")
                metrics = sample.get("metrics") or sample.get("exercise_metrics", "")
                summary = sample.get("summary") or sample.get("exercise_summary", "")

                encoded = tokenize_and_mask_labels(tokenizer, prompt, metrics, summary, max_length)
                if encoded:
                    encoded_data.append(encoded)
            except Exception as e:
                print(f"[WARN] Skipped line {idx} due to error: {e}")

    total_samples = len(encoded_data)
    print(f"[INFO] Successfully parsed {total_samples} valid samples.")
    
    if total_samples == 0:
        print("[CRITICAL] No valid training data left after preprocessing.")
        sys.exit(1)

    train_size = int(total_samples * split_ratio)
    return DatasetDict({
        "train": Dataset.from_list(encoded_data[:train_size]),
        "test": Dataset.from_list(encoded_data[train_size:])
    })


def build_training_args(output_dir, batch_size, grad_accum, num_epochs, lr):
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,
        learning_rate=lr,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=400,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=10,
        fp16=False,  # 显式使用 float32 训练
        run_name=f"mole-sft-{time.strftime('%Y%m%d_%H%M%S')}",
        load_best_model_at_end=True,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="tensorboard"
    )


def main():
    args = parse_args()
    auto_select_gpu()

    tokenizer = get_tokenizer(args.model_path)
    model = get_model(args.model_path, tokenizer)

    # 统一使用 args.max_length 收拢超参数
    dataset = load_and_process_data(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        split_ratio=0.95,
    )

    training_args = build_training_args(
        output_dir=args.output_dir,
        batch_size=2,
        grad_accum=4,
        num_epochs=3,
        lr=5e-5
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"]
    )

    print(f"[INFO] Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    trainer.train()
    print(f"[INFO] Training finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 分布式训练环境下，通常只在主进程（Rank 0）保存模型，防止写冲突
    is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

    if is_main_process:
        print(f"[INFO] Saving model and tokenizer to: {args.output_dir}")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print("[SUCCESS] Pipeline completed.")


if __name__ == "__main__":
    main()