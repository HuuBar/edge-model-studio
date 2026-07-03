import os
import sys
import time
import math
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint

# 规范化标准服务器底色日志
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """解析集群/单机增量预训练超参数"""
    parser = argparse.ArgumentParser(description="Production-level LLM Incremental Pre-training (Mid-train) Pipeline")
    
    # 基础路径配置
    parser.add_argument("--model_path", type=str, required=True, help="Path to base pretrained model directory")
    parser.add_argument("--train_data_dir", type=str, required=True, help="Directory containing tokenized or raw jsonl files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store checkpoint and logs")
    
    # 超参数控制
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Target sequence length for text packing")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of updates steps to accumulate before backward")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Peak learning rate for adamw")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for regularization")
    parser.add_argument("--num_epochs", type=int, default=1, help="Total number of training epochs over the stream")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Linear warmup ratio over total training steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # 工程优化开关
    parser.add_argument("--flash_attn", action="store_true", help="Enable FlashAttention-2 for sequence acceleration")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable activation checkpointing to save VRAM")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision instead of float32/float16")
    
    # 周期监控
    parser.add_argument("--logging_steps", type=int, default=1, help="Log metrics every X steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate model every X steps")

    return parser.parse_args()


class ConstantLengthDataset(IterableDataset):
    """
    流式 Packing 数据集容器。
    它源源不断地从原始 jsonl 中读取文本并进行 Tokenize，然后将无数长短不一的句子用 <|endoftext|> (EOS) 拼接成一整条连续序列，
    再严格按照 max_seq_length 切块吐出。无任何 Padding 浪费，100% 榨干 GPU 算力。
    """
    def __init__(
        self, 
        tokenizer: AutoTokenizer, 
        dataset, 
        max_seq_length: int = 4096, 
        text_key: str = "text"
    ):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_seq_length = max_seq_length
        self.text_key = text_key
        self.eos_token_id = tokenizer.eos_token_id

    def __iter__(self):
        buffer = []
        # 利用流式迭代，内存开销保持在 O(1)
        for sample in self.dataset:
            text = sample.get(self.text_key, "")
            if not text.strip():
                continue
                
            # 增量训练只关心原始文本，无需拼接应用 Chat Template 模板
            token_ids = self.tokenizer(text, truncation=False)["input_ids"]
            # 显式追加 EOS 隔离两条不同的语义语料
            buffer.extend(token_ids + [self.eos_token_id])
            
            # 当缓冲区长度积累到可以切分出若干个完整窗口时
            while len(buffer) >= self.max_seq_length:
                chunk = buffer[:self.max_seq_length]
                buffer = buffer[self.max_seq_length:]
                
                # Mid-train 的 labels 矩阵即为 input_ids 的深拷贝（因底层有 causal mask 错位计算 loss）
                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                    "labels": torch.tensor(chunk, dtype=torch.long)
                }


def main():
    args = parse_args()
    set_seed(args.seed)

    # 1. 检测集群多卡 DDP 分布式环境环境
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main_process = local_rank in [-1, 0]

    if is_main_process:
        logger.info(f"Initializing Mid-train pipeline layout target: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)

    # 2. 加载 Tokenizer 
    logger.info(f"Loading pre-trained tokenizer skeleton from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 流式加载海量未加工语料 (支持大规模文本目录或单文件)
    logger.info(f"Configuring streaming dataset driver from: {args.train_data_dir}")
    data_files = [str(p) for p in Path(args.train_data_dir).glob("*.jsonl")]
    if not data_files:
        logger.critical(f"No jsonl target data format files detected inside: {args.train_data_dir}")
        sys.exit(1)

    # 开启 streaming=True 后，数据将以流式 Generator 吐出，不占用本地物理磁盘缓存与 RAM
    raw_dataset = load_dataset("json", data_files=data_files, split="train", streaming=True)
    
    # 将数据集打包送进 Packing 状态机
    train_dataset = ConstantLengthDataset(
        tokenizer=tokenizer,
        dataset=raw_dataset,
        max_seq_length=args.max_seq_length,
        text_key="text"
    )

    # 4. 配置基座模型
    logger.info(f"Compiling causal LM architecture from: {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path)
    
    # 动态注入 FlashAttention-2 算子加速
    if args.flash_attn:
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            config._attn_implementation = "flash_attention_2"
            logger.info("FlashAttention-2 kernel execution successfully injected.")
        else:
            logger.warning("FlashAttention-2 is only supported on Ampere or newer GPUs (e.g., A100, H100). Falling back.")

    # 载入原生权重
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        low_cpu_mem_usage=True
    )

    # 显存控制机制
    if args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing for VRAM footprint optimization.")
        model.gradient_checkpointing_enable()

    # 5. 构筑生产级 TrainingArguments 矩阵
    # 增量训练一般需要更小、更平稳的自适应衰减率
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_epochs,
        max_steps=int(os.environ.get("MAX_STEPS", -1)),  # 流式数据可显式限定绝对最大更新步数
        
        # 调度控制
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        evaluation_strategy="no",  # 增量阶段常不需要耗费算力做中途 Eval
        
        # 混合精度
        bf16=args.bf16,
        fp16=not args.bf16 and torch.cuda.is_available(),
        
        # 分布式 DDP / 算力控制优化
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        remove_unused_columns=False,  # ConstantLengthDataset 包含非默认列，需设为 False 规避被自动过滤
        ddp_find_unused_parameters=False,
        group_by_length=False,        # 已做 Packing 长度完全齐整，关闭长度聚类以节省排序开销
        report_to=["tensorboard"] if is_main_process else [],
        logging_dir=os.path.join(args.output_dir, "runs")
    )

    # 6. 初始化训练器
    # 利用 default_data_collator，因为它能高效、安全地处理我们在 Dataset 里已经转换好的固定张量
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )

    # 7. 断点续训自动接管防御（Resume from Checkpoint）
    last_checkpoint = None
    if os.path.isdir(args.output_dir):
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Detected previous checkpoint cluster: '{last_checkpoint}'. Resuming training stream...")

    # 8. 启动增量预训练
    logger.info("=== Mid-train Stream Execution Triggered ===")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    logger.info("=== Mid-train Stream Finished Successfully ===")

    # 9. 闭环保存最终成果模型
    if is_main_process:
        logger.info(f"Serializing final merged weights directory to: {args.output_dir}")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info("[SUCCESS] Pre-training flow accomplished clean.")


if __name__ == "__main__":
    main()