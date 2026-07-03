import torch
import time
import json
import random
import logging
from dataclasses import dataclass
from typing import List, Dict

from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================
# Config
# =========================
@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_test_samples: int = 5
    max_new_tokens: int = 80


cfg = Config()


# =========================
# Logger
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# =========================
# Utils
# =========================
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_prompt(instruction, input_text=""):
    return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""


# =========================
# Model Loader
# =========================
def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16 if cfg.device == "cuda" else torch.float32,
        device_map="auto"
    )
    model.eval()
    return model


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# =========================
# Sample Test Data
# =========================
TEST_CASES = [
    {
        "instruction": "解释什么是跑步",
        "input": "简单说明"
    },
    {
        "instruction": "什么是RAG模型",
        "input": "用简单语言解释"
    },
    {
        "instruction": "减脂为什么要有氧运动",
        "input": "说明原理"
    },
    {
        "instruction": "心率区间是什么意思",
        "input": "运动健康角度解释"
    },
    {
        "instruction": "VO2max是什么",
        "input": "简单科普"
    }
]


# =========================
# Generation Test
# =========================
@torch.no_grad()
def generate_once(model, tokenizer, instruction, input_text):
    prompt = build_prompt(instruction, input_text)

    inputs = tokenizer(prompt, return_tensors="pt").to(cfg.device)

    start = time.time()

    output = model.generate(
        **inputs,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    latency = time.time() - start

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    return text, latency


# =========================
# Batch Latency Test
# =========================
@torch.no_grad()
def batch_test(model, tokenizer, batch_size=4):
    logging.info("Running batch inference test...")

    prompts = [
        build_prompt("解释什么是跑步", "")
        for _ in range(batch_size)
    ]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(cfg.device)

    start = time.time()

    outputs = model.generate(
        **inputs,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=False
    )

    end = time.time()

    total_time = end - start

    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Total time: {total_time:.3f}s")
    logging.info(f"Per sample: {total_time / batch_size:.3f}s")


# =========================
# Forward Loss Test
# =========================
@torch.no_grad()
def forward_loss_test(model, tokenizer):
    logging.info("Running forward loss test...")

    text = build_prompt(
        "解释什么是跑步",
        "简单说明"
    )

    inputs = tokenizer(text, return_tensors="pt").to(cfg.device)

    labels = inputs["input_ids"].clone()

    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        labels=labels
    )

    logging.info(f"Loss: {outputs.loss.item():.4f}")


# =========================
# Accuracy-style heuristic eval
# =========================
def simple_eval(outputs: str):
    """
    非严格评估：
    看是否输出长度正常、是否包含关键词
    """
    score = 0

    if len(outputs) > 50:
        score += 1
    if "跑步" in outputs or "运动" in outputs:
        score += 1
    if "心率" in outputs or "RAG" in outputs:
        score += 1

    return score


# =========================
# Full Evaluation
# =========================
def run_eval(model, tokenizer):
    logging.info("Running full evaluation...")

    scores = []
    latencies = []

    for i, case in enumerate(TEST_CASES):
        output, latency = generate_once(
            model,
            tokenizer,
            case["instruction"],
            case["input"]
        )

        score = simple_eval(output)

        scores.append(score)
        latencies.append(latency)

        logging.info(f"\n--- Sample {i} ---")
        logging.info(f"Latency: {latency:.3f}s")
        logging.info(f"Score: {score}")
        logging.info(f"Output preview: {output[:120]}")

    logging.info("\n===== FINAL REPORT =====")
    logging.info(f"Avg score: {sum(scores)/len(scores):.2f}")
    logging.info(f"Avg latency: {sum(latencies)/len(latencies):.3f}s")


# =========================
# Stress Test
# =========================
def stress_test(model, tokenizer, rounds=3):
    logging.info("Running stress test...")

    for r in range(rounds):
        logging.info(f"Round {r+1}")

        for case in TEST_CASES:
            _, latency = generate_once(
                model,
                tokenizer,
                case["instruction"],
                case["input"]
            )

            logging.info(f"latency={latency:.3f}s")


# =========================
# Main
# =========================
def main():
    set_seed(42)

    logging.info(f"Device: {cfg.device}")
    logging.info("Loading model...")

    tokenizer = load_tokenizer()
    model = load_model()

    # 1. 单样本测试
    logging.info("\n=== SINGLE TEST ===")
    out, latency = generate_once(
        model,
        tokenizer,
        "解释什么是跑步",
        "简单说明"
    )

    logging.info(f"Latency: {latency:.3f}s")
    logging.info(f"Output:\n{out[:300]}")

    # 2. forward loss
    forward_loss_test(model, tokenizer)

    # 3. batch test
    batch_test(model, tokenizer, batch_size=4)

    # 4. full eval
    run_eval(model, tokenizer)

    # 5. stress test
    stress_test(model, tokenizer)


if __name__ == "__main__":
    main()