import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# === 路径配置 ===
json_path = "/data2/jwllm/datasets/ocnli/ocnli_test.json"
# model_path = "/data2/jwllm/models_origin/Qwen3-0.6B"
model_path = "/data2/jwllm/model_process/Qwen3-0.6B-Base-sft-round-0"

output_dir = "/data2/jwllm/inference_output/nli_predictions_base_round_0.json"

MAX_NEW_TOKENS = 10
BATCH_SIZE = 8

# === 加载模型 ===
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).cuda().eval()

# === Prompt设定 ===
# PROMPT = (
#     "你是一位逻辑推理专家，擅长判断两个句子之间的逻辑关系。给定两个句子：\n"
#     "- 前提句（sentence1）\n- 假设句（sentence2）\n"
#     "请判断 sentence2 与 sentence1 的逻辑关系，关系包括："
#     "「entailment」「contradiction」「neutral」\n"
#     "请只输出其中一个类别作为回答。"
#     "例如：输入是：sentence1: 一点来钟时,张永红却来了,sentence2: 一点多钟,张永红来了, 输出为：entailment"  
# )


PROMPT = """你是一位逻辑推理专家，擅长判断两个句子之间的逻辑关系。从「entailment」「contradiction」「neutral」选择一个输出，不需要额外输出任何其他信息。"""

# === 推理函数 ===
def predict(messages_list, model, tokenizer, max_new_tokens=10):
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
        max_length=512,
        padding="longest"
    ).to(device)

    generation_config = {
        "max_new_tokens": max_new_tokens,
        "temperature": 0,
        "do_sample": False,
        "num_beams": 1,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.2  # 设置重复惩罚

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

messages_list = []
sents1 = []
sents2 = []
labels = []

for item in data:
    s1 = item["sentence1"]
    s2 = item["sentence2"]
    sents1.append(s1)
    sents2.append(s2)
    labels.append(item.get("label", ""))  # 真实标签（可选）

    messages_list.append([
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": f"sentence1：{s1}\nsentence2：{s2}"}
    ])

# messages_list = messages_list[:10]  # 限制推理数量，用于测试


# === 批量推理 ===
results = []
pbar = tqdm(total=len(messages_list), desc="推理逻辑关系")

for start in range(0, len(messages_list), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(messages_list))
    batch_messages = messages_list[start:end]
    batch_outputs = predict(batch_messages, model, tokenizer)

    for i, output in enumerate(batch_outputs):
        results.append({
            "sentence1": sents1[start + i],
            "sentence2": sents2[start + i],
            "predicted_label": output.strip(),
            "true_label": labels[start + i]
        })

    pbar.update(end - start)

pbar.close()

# === 保存结果 ===
with open(output_dir, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"✅ 已保存逻辑关系预测结果到：{output_dir}")