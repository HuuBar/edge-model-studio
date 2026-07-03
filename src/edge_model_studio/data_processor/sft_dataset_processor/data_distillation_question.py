import json
import random
import re
from datetime import datetime, timedelta

import requests
from transformers import AutoTokenizer


# =========================
# Config
# =========================
TOTAL_CASES = 10
CYCLE_TIMES = 1
BATCH_SIZE = 2

MODEL_ID = "/home/l00495039/code/Qwen3-8B"
API_URL = "http://100.105.97.35:34512/v1/completions"
OUTPUT_FILE = "question_20.jsonl"

METRICS_POOL = [
    "年龄", "性别", "运动经验", "技能水平", "目标",
    "运动类型", "距离", "时长", "平均配速", "平均心率",
    "步频", "环境"
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)


# =========================
# Utils
# =========================
def rand_age():
    return random.randint(20, 50)


def rand_date(start=2020, end=2024):
    s = datetime(start, 1, 1)
    e = datetime(end, 12, 31)
    return (s + timedelta(days=random.randint(0, (e - s).days))).strftime("%Y年%m月%d日")


def chat(messages, thinking=False):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking
    )


def clean_output(text):
    text = text.replace("<|im_start|>", "").replace("<|im_end|>", "")
    think = re.search(r"<think>(.*?)</think>", text, re.S)
    if think:
        return think.group(1).strip(), text[think.end():].strip()
    return "", text.strip()


# =========================
# Prompt Builders
# =========================
def build_metrics_prompt():
    age = rand_age()

    system = f"""
你是专业跑步教练，生成真实运动数据。
年龄固定：{age}
技能水平：入门/进阶/精英
运动类型：跑步/骑行
环境：公园/小区
必须自然合理
"""

    user = "示例：运动类型：跑步；性别：男；年龄：32岁；目标：提升心肺耐力；技能水平：进阶；运动时长：58分钟；运动距离：11.9km；平均配速：4'53/km；最大心率：176bpm；平均心率：148bpm；有氧训练压力：3.6；无氧训练压力：0.9；上升速度：3.5km/h/s；下降速度：4.0km/h/s；消耗热量：770kcal；最大摄氧量：53ml/kg/min；平均步频：178步/分钟；平均步幅：1.14m；平均触地时间：240ms；平均振幅：7.8cm；生理因素：睡眠7.8小时。"

    return chat([
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ])


# =========================
# Pipeline
# =========================
def generate_metrics(n):
    return [build_metrics_prompt() for _ in range(n)]

# =========================
# Inference
# =========================
def batch_infer(prompts):
    results = []

    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i + BATCH_SIZE]

        payload = {
            "model": "Qwen3-235B-A22B-w8a8",
            "prompt": batch,
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.15
        }

        r = requests.post(API_URL, json=payload)

        if r.status_code != 200:
            print("API error:", r.text)
            continue

        for c in r.json()["choices"]:
            _, content = clean_output(c["text"])
            results.append(content)

    return results

# =========================
# Parsing
# =========================

def build_fix_prompt(unit, suggestion):
    system = "根据评价修改文本，不改变结构，只修复问题"
    return chat([
        {"role": "system", "content": system},
        {"role": "user", "content": f"{unit}\n修改建议：{suggestion}"}
    ])

def get_score(text):
    m = re.search(r"累计扣分.*?(\d+)", text)
    return int(m.group(1)) if m else -1


def get_suggestion(text):
    return text.split("不足及修改说明")[-1]

def build_eval_prompt(text):
    system = """
你是语言学专家，评分0-100。
必须输出：
1.【不足及修改说明】
2.【累计扣分】：xx分
"""
    return chat([
        {"role": "system", "content": system},
        {"role": "user", "content": f"文本：{text}"}
    ])

def main():
    print(f"Generating {TOTAL_CASES} cases...")

    # 1. generate prompts
    metric_prompts = generate_metrics(TOTAL_CASES)

    # 2. infer metrics
    metrics = batch_infer(metric_prompts)

    if len(metrics) != len(metric_prompts):
        raise ValueError("metrics size mismatch")

    dataset = []

    # 3. evaluation loop
    for i in range(len(metrics)):
        eval_prompt = build_eval_prompt(metrics[i])
        dataset.append({
            "instruction": eval_prompt,
            "input": metrics[i].replace(" ", "")
        })

    # 4. save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for d in dataset:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print("Saved to", OUTPUT_FILE)
    
if __name__ == "__main__":
    main()