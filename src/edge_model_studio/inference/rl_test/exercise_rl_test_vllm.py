import json
import re
from datetime import datetime

import requests
from transformers import AutoTokenizer

# 配置项
model_name = f"qwen3_0.6b_sft_valuehead_RL"
MODEL_ID = f"/data2/jwllm/model_process/exercise_models/{model_name}"
API_URL = "http://localhost:8088/v1/completions"
BATCH_SIZE = 32
# INPUT_FILE = "/data2/jwllm/datasets/exercise_generate/exercise_dataset_test.jsonl"
INPUT_FILE = "/data2/jwllm/scripts/inference/rl_test/exercise_dataset_test_output_20250625_095743.jsonl"
OUTPUT_FILE = f"exercise_dataset_test_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)


def parse_qwen3_output(text):
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip(), text[think_match.end():].strip()
    thinking = ""
    answer = text.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
    answer = re.sub(r"^(assistant|user|system):?", "", answer, flags=re.IGNORECASE).strip()
    return thinking, answer


def extract_final_answer(text: str) -> str:
    text = text.replace("<|im_start|>", "").replace("<|im_end|>", "")
    patterns = [r"#指标解析.*", r"指标解析.*", r"#分析结果.*", r"分析结果.*"]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(0).strip()
    return text.strip()


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f.readlines()]


def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def perform_batch_inference(prompts):
    results = []
    for i in range(0, len(prompts), BATCH_SIZE):
        print(f"[{datetime.now()}] 批次推理：{i} / {len(prompts)}")
        batch = prompts[i:i + BATCH_SIZE]
        payload = {
            "model": "qwen3-8B-jwllm",
            "prompt": batch,
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.15
        }
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            for choice in response.json()["choices"]:
                _, content = parse_qwen3_output(choice["text"].strip())
                content = extract_final_answer(content)
                results.append(content)
        else:
            print(f"[错误] 状态码: {response.status_code}，响应: {response.text}")
            results.extend([""] * len(batch))
    return results


def main():
    print("📥 正在读取原始数据...")
    dataset = read_jsonl(INPUT_FILE)
    prompts = [item["exercise_prompt"] for item in dataset]

    print("🧠 正在进行模型推理...")
    outputs = perform_batch_inference(prompts)

    print("📦 正在写入带推理结果的新数据文件...")
    for item, result in zip(dataset, outputs):
        item[f"{model_name}"] = result
    write_jsonl(OUTPUT_FILE, dataset)
    print(f"✅ 推理完成，文件保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
