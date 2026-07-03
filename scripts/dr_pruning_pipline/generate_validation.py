import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# === 配置路径 ===
json_path = "/home/shangyangyang/project/Dataset/clts_filter.json"

model_path = "/home/shangyangyang/models/qwen-0.6b-sft/Qwen3-0.6B-sft-round-0-sft-round-4-int8"
output_dir = "/home/shangyangyang/project/Dataset/compare_for_data_19819"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"{os.path.basename(model_path.rstrip('/'))}_summary_19000_19819.json")

# === 模型配置 ===
MAX_NEW_TOKENS = 80
BATCH_SIZE = 8

# === 加载模型与 tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).eval()

# 打印GPU信息
print(f"GPU 内存分配: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
print(f"GPU 缓存分配: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")

# === 封装推理函数 ===
def predict(messages_list, model, tokenizer, max_new_tokens=80):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        ) for messages in messages_list
    ]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding="longest"
    ).to(device)

    generation_config = {
        "max_new_tokens": max_new_tokens,
        "temperature": 1.0,
        "do_sample": False,
        "num_beams": 4,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.2,
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_config)

    input_len = inputs["input_ids"].shape[1]
    return tokenizer.batch_decode(
        [output[input_len:] for output in outputs],
        skip_special_tokens=True
    )

# === 加载数据 ===
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

data = data[19000:19820]
src_lines = [item["context"] for item in data]
tgt_lines = [item["ground_truth"] for item in data]
assert len(src_lines) == len(tgt_lines), "源文本与摘要数量不一致"

PROMPT = "你是一位专业新闻编辑，精通文章写作与摘要技巧，能够准确提炼关键信息，不得虚构、猜测或省略原文中的重要数据，确保摘要真实、客观、精炼。请基于上述内容，生成一段不超过70字的摘要。"
messages_list = [
    [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": context}
    ]
    for context in src_lines
]

# === 批量推理 ===
results = [None] * len(src_lines)
pbar = tqdm(total=len(src_lines), desc="生成摘要")

for start in range(0, len(messages_list), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(messages_list))
    batch_messages = messages_list[start:end]
    batch_outputs = predict(batch_messages, model, tokenizer, max_new_tokens=MAX_NEW_TOKENS)

    for i, output in enumerate(batch_outputs):
        idx = start + i
        results[idx] = {
            "context": src_lines[idx],
            "ground_truth": tgt_lines[idx],
            "llm_answer": output.strip()
        }

    pbar.update(end - start)

pbar.close()

# === 保存结果 ===
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"已保存结果到 {output_file}")
print(f"最终 GPU 内存占用: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")