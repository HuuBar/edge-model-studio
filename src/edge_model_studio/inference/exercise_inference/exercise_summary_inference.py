#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import requests
from transformers import AutoTokenizer

# ----------------------------------------------------------------------
# 默认配置（可通过 CLI 覆盖）
# ----------------------------------------------------------------------
DEFAULT_MODEL_NAME = "qwen3_0.6b"
DEFAULT_API_URL = "http://localhost:8088/v1/completions"
DEFAULT_BATCH_SIZE = 64
DEFAULT_INPUT_FILE = "/data2/jwllm/datasets/exercise_generate/summary/exercise_summary_test_results.jsonl"
DEFAULT_OUTPUT_FILE = "/data2/jwllm/datasets/exercise_generate/summary/exercise_summary_test_results.jsonl"
DEFAULT_SERVED_MODEL_NAME = "qwen3-jwllm"  # vLLM API 中的 served_model_name
DEFAULT_MODEL_DIR_TEMPLATE = f"/data2/jwllm/model_process/exercise_summary/{DEFAULT_MODEL_NAME}"


# ----------------------------------------------------------------------
# 通用工具函数
# ----------------------------------------------------------------------
def parse_model_output(text: str):
    """解析 <think>…</think> 模型输出，拆分『思考』和『答案』两部分。"""
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip(), text[think_match.end():].strip()

    thinking = ""
    answer = (
        text.replace("<|im_start|>", "")
            .replace("<|im_end|>", "")
            .strip()
    )
    answer = re.sub(
        r"^(assistant|user|system):?", "", answer, flags=re.IGNORECASE
    ).strip()
    return thinking, answer


def extract_final_answer(text: str) -> str:
    """从完整输出中提取满足规则的最终答案段落。"""
    text = text.replace("<|im_start|>", "").replace("<|im_end|>", "")
    patterns = [
        r"#指标解析.*",
        r"指标解析.*",
        r"#分析结果.*",
        r"分析结果.*",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(0).strip()
    return text.strip()


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: str, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# 传入dataset，转变成encoding后的prompts
def build_prompt_list(dataset, tokenizer):
    prompt_list = []
    for item in dataset:
        # 拼接 prompt（仅 user 部分）
        user_prompt = item["prompt"]
        metrics = item["metrics"]
        messages = [
            {"role": "system", "content": user_prompt},
            {"role": "user", "content": metrics}
        ]
        prompt_str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_list.append(prompt_str)
    return prompt_list


def perform_batch_inference(
        prompts,
        api_url: str,
        batch_size: int,
        served_model_name: str,
):
    """发送批量请求到 vLLM API，返回解析后的结果列表。"""
    results = []
    for i in range(0, len(prompts), batch_size):
        print(f"[{datetime.now()}] 批次推理：{i} / {len(prompts)}")
        batch = prompts[i: i + batch_size]
        payload = {
            "model": served_model_name,
            "prompt": batch,
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.15,
        }
        try:
            response = requests.post(api_url, json=payload, timeout=120)
        except requests.RequestException as e:
            print(f"[错误] 请求失败：{e}")
            results.extend([""] * len(batch))
            continue

        if response.status_code == 200:
            for choice in response.json().get("choices", []):
                _, content = parse_model_output(choice["text"].strip())
                results.append(extract_final_answer(content))
        else:
            print(
                f"[错误] 状态码: {response.status_code}，响应: {response.text}"
            )
            results.extend([""] * len(batch))
    return results


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="批量调用 vLLM /completions API 进行推理并保存结果")

    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help=f"模型名称（默认: {DEFAULT_MODEL_NAME})")
    parser.add_argument("--model_dir", type=str, help="模型本地路径（默认根据 --model_name 拼接生成）",
                        default=DEFAULT_MODEL_DIR_TEMPLATE)
    parser.add_argument("--served_model_name", type=str, default=DEFAULT_SERVED_MODEL_NAME,
                        help=f"vLLM API 中的 served_model_name（默认: {DEFAULT_SERVED_MODEL_NAME})")
    parser.add_argument("--api_url", type=str, default=DEFAULT_API_URL,
                        help=f"vLLM /completions 接口 URL（默认: {DEFAULT_API_URL})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"单批推理样本数（默认: {DEFAULT_BATCH_SIZE})", )
    parser.add_argument("--input_file", type=str, default=DEFAULT_INPUT_FILE,
                        help=f"输入 JSONL 文件路径（默认: {DEFAULT_INPUT_FILE})", )
    parser.add_argument("--output_file", type=str, default=DEFAULT_OUTPUT_FILE,
                        help=f"输出 JSONL 文件路径（默认: {DEFAULT_OUTPUT_FILE})")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    print(f"🔑 加载 tokenizer（模型目录: {args.model_dir}）...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    # ------------------------------------------------------------------
    print("📥 正在读取原始数据...")
    dataset = read_jsonl(args.input_file)
    print("拼装prompt...")
    prompts = build_prompt_list(dataset, tokenizer)
    print(f"共 {len(prompts)} 条prompt，开始推理...")

    # ------------------------------------------------------------------
    print("🧠 正在进行模型推理...")
    outputs = perform_batch_inference(
        prompts,
        api_url=args.api_url,
        batch_size=args.batch_size,
        served_model_name=args.served_model_name,
    )

    # ------------------------------------------------------------------
    print("📦 正在写入带推理结果的新数据文件...")
    for item, result in zip(dataset, outputs):
        item[args.model_name] = result
    write_jsonl(args.output_file, dataset)
    print(f"✅ 推理完成，文件已保存至: {args.output_file}")


if __name__ == "__main__":
    main()
