import argparse
import json
import os
import time
from datetime import datetime

# 🔥 环境变量设置 - 放在最前面
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["SWANLAB_PROJECT"] = "qwen3-sft-running"

# 🔥 禁用 flash-attn 相关设置（避免兼容性问题）
os.environ["DISABLE_FLASH_ATTN"] = "1"


# 🔥 修复 accelerate 兼容性问题
def patch_accelerate_memory():
    """修复 accelerate 兼容性问题"""
    # 1. 修复 clear_device_cache
    try:
        from accelerate.utils.memory import clear_device_cache
        print("✅ accelerate.utils.memory.clear_device_cache 已存在")
    except ImportError:
        print("⚠️ 正在修复 accelerate.utils.memory.clear_device_cache...")
        import accelerate.utils.memory
        import torch

        def clear_device_cache():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        accelerate.utils.memory.clear_device_cache = clear_device_cache
        print("✅ 已修复 accelerate.utils.memory.clear_device_cache")

    # 2. 修复 AcceleratedOptimizer.train() 方法
    try:
        import accelerate.optimizer
        OriginalAcceleratedOptimizer = accelerate.optimizer.AcceleratedOptimizer

        class PatchedAcceleratedOptimizer(OriginalAcceleratedOptimizer):
            def train(self, mode=True):
                return self

            def eval(self):
                return self

        accelerate.optimizer.AcceleratedOptimizer = PatchedAcceleratedOptimizer
        print("✅ 已修复 accelerate.optimizer.AcceleratedOptimizer")

    except Exception as e:
        print(f"⚠️ AcceleratedOptimizer 修复失败: {e}")

    # 3. 直接为 AdamW 添加 train 方法
    try:
        import torch.optim

        if not hasattr(torch.optim.AdamW, 'train'):
            def train_method(self, mode=True):
                return self

            def eval_method(self):
                return self

            torch.optim.AdamW.train = train_method
            torch.optim.AdamW.eval = eval_method
            print("✅ 已为 torch.optim.AdamW 添加 train/eval 方法")

    except Exception as e:
        print(f"⚠️ AdamW 修复失败: {e}")


# 在导入 transformers 之前先修复
patch_accelerate_memory()

import swanlab
import torch
from datasets import Dataset, DatasetDict

# 导入 transformers
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        DataCollatorForLanguageModeling,
        Trainer,
    )

    print("✅ transformers 导入成功")
except ImportError as e:
    print(f"❌ transformers 导入失败: {e}")
    raise


# ==== 环境变量与全局实验环境初始化 ====
def init_env():
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank in (-1, 0):
        try:
            swanlab.init(mode="local")
            print("✅ SwanLab 初始化成功")
        except Exception as e:
            print(f"⚠️ SwanLab 初始化失败: {e}")


# ==== 命令行参数解析 ====
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="启动的模型名称")
    parser.add_argument("--data_path", type=str, required=True, help="训练集的具体位置和路径")
    parser.add_argument("--output_dir", type=str, required=True, help="训练后模型输出的位置")
    return parser.parse_args()


# ==== 获得tokenizer ====
def get_tokenizer(model_path):
    print(f"🔥 Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("✅ Tokenizer 加载成功")
    return tokenizer


# ==== 获得模型 ====
def get_model(model_path):
    print(f"🔥 Loading model from: {model_path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            use_cache=False,
            attn_implementation="sdpa",
            device_map=None,
        )
        print("✅ Model 加载成功")
    except Exception as e:
        print(f"⚠️ 模型加载失败，尝试不使用 attn_implementation: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            use_cache=False,
            device_map=None,
        )
        print("✅ Model 加载成功（降级模式）")

    try:
        model.gradient_checkpointing_enable()
        print("✅ 梯度检查点启用成功")
    except Exception as e:
        print(f"⚠️ 梯度检查点启用失败: {e}")

    model.train()
    print(f"✅ Model loaded: {model.__class__.__name__}")
    print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    return model


# ==== SwanLab 配置 ====
def update_swanlab_config(model_name, max_length, batch_size, num_train_epochs, learning_rate):
    try:
        swanlab.config.update({
            "model": model_name,
            "task": "跑步问答",
            "data_max_length": max_length,
            "batch_size": batch_size,
            "epochs": num_train_epochs,
            "learning_rate": learning_rate
        })
        print("✅ SwanLab 配置更新成功")
    except Exception as e:
        print(f"⚠️ SwanLab 配置更新失败: {e}")


# ==== 数据加载、处理、切分 ====
def load_and_process_data(data_path, tokenizer, max_length=2048, split_ratio=0.95):
    encoded_prompts = []

    print(f"🔥 Loading data from: {data_path}")

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

            prompt = sample.get("prompt") or sample.get("exercise_prompt", "")
            metrics = sample.get("metrics") or sample.get("exercise_metrics", "")
            summary = sample.get("summary") or sample.get("exercise_summary", "")

            messages = [
                {"role": "system", "content": prompt.strip()},
                {"role": "user", "content": metrics.strip()},
                {"role": "assistant", "content": summary.strip()}
            ]

            try:
                prompt_str = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception as e:
                print(f"[Line {idx}] Template error: {e}")
                prompt_str = f"System: {prompt.strip()}\nUser: {metrics.strip()}\nAssistant: {summary.strip()}"

            encoded = tokenizer(
                prompt_str,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None
            )

            encoded["labels"] = encoded["input_ids"].copy()
            encoded_prompts.append(encoded)

            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1} samples...")

    print(f"========== 已加载数据集，累计 {len(encoded_prompts)} 条 ==========")

    train_size = int(len(encoded_prompts) * split_ratio)
    train_data = encoded_prompts[:train_size]
    test_data = encoded_prompts[train_size:]
    print(f"========== train: {len(train_data)}, test: {len(test_data)} ==========")

    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_data),
        "test": Dataset.from_list(test_data)
    })
    return dataset_dict


# ==== 训练参数配置 ====
def build_training_args(output_dir, run_name, batch_size, gradient_accumulation_steps,
                        num_train_epochs, learning_rate, max_length):
    # 训练参数都可以在这里修改
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=400,
        logging_steps=10,

        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,

        learning_rate=learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,

        bf16=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,

        save_total_limit=3,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        remove_unused_columns=False,
        report_to=["swanlab"] if swanlab else [],
        run_name=run_name,

        ddp_find_unused_parameters=False,
    )
    return training_args


# ==== 模型训练 ====
def train_model(model, tokenizer, split_dataset, training_args):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
        return_tensors="pt",  # 🔥 添加：明确返回类型
    )

    # 🔥 使用 processing_class 替代 tokenizer（避免 FutureWarning）
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
        processing_class=tokenizer,  # 🔥 修改：使用 processing_class
    )
    return trainer


# ==== 模型与tokenizer保存 ====
def save_model(model, tokenizer, output_dir):
    print(f"🔥 Saving model to: {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print("✅ 训练完成，模型保存至", output_dir)


# ==== 主流程 ====
def main():
    print("🔥 开始初始化...")

    batch_size = 8
    gradient_accumulation_steps = 4
    num_train_epochs = 50  # 训练若干轮
    learning_rate = 2e-5
    max_length = 2048
    split_rate = 0.95

    args = parse_args()
    print(f"✅ 参数解析完成: {args}")

    init_env()

    model_name = os.path.basename(args.model_path.rstrip("/"))
    run_name = f"qwen3-sft-{time.strftime('%Y%m%d_%H%M%S')}"

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    print(f"🔥 Local rank: {local_rank}")

    if local_rank in (-1, 0):
        update_swanlab_config(model_name, max_length, batch_size, num_train_epochs, learning_rate)

    tokenizer = get_tokenizer(args.model_path)
    model = get_model(args.model_path)

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

    print(f"=================== 开始SFT训练: {datetime.now()} ===================")
    print(f"🔥 GPU数量: {torch.cuda.device_count()}")
    print(f"🔥 当前设备: {torch.cuda.current_device()}")
    print(f"🔥 总训练样本: {len(split_dataset['train'])}")
    print(f"🔥 总验证样本: {len(split_dataset['test'])}")
    print(f"🔥 有效batch size: {batch_size * gradient_accumulation_steps * torch.cuda.device_count()}")

    try:
        trainer.train()
        print(f"=================== SFT训练成功: {datetime.now()} ===================")

        print(f"=================== 开始保存模型: {datetime.now()} ===================")
        save_model(model, tokenizer, args.output_dir)
        print(f"=================== 模型保存成功: {datetime.now()} ===================")

    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
