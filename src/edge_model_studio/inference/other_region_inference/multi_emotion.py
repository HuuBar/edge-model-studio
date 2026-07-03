# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# import json
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from tqdm import tqdm

# # === 配置参数 ===
# json_path = "/data2/jwllm/datasets/multi_emotion/multi_emotion_test.json"
# model_path = "/data2/jwllm/models_origin/Qwen3-0.6B-Base"

# output_dir = "/data2/jwllm/inference_output/emotion_prediction_base_10.json"

# MAX_NEW_TOKENS = 500
# BATCH_SIZE = 8

# # === 加载模型与tokenizer ===
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# tokenizer.padding_side = "left"

# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     trust_remote_code=True,
#     torch_dtype=torch.float16,
#     low_cpu_mem_usage=True
# ).cuda().eval()

# # === Prompt设定 ===
# # PROMPT = (
# #     "你是一位资深情感分析专家，擅长判断用户语句中所表达的主要情绪。情绪类别包括：「平淡」「开心」「悲伤」「愤怒」「惊奇」「厌恶」。"
# #     "请判断下方用户语句所表达的主要情绪，并只输出情绪类别，不要添加解释说明。"
# # )

# # PROMPT = (
# #     "你是一位资深情感分析专家，擅长判断用户语句中所表达的主要情绪。"
# #     "情绪类别：「平淡」「开心」「悲伤」「愤怒」「惊奇」「厌恶」「疑问」「关切」"
# #     "请判断下方用户语句所表达的主要情绪，仅输出情绪类别，不要包含任何其他文字或符号。"
# #     "比如：input:我很难过，output:悲伤。"
# # )
# PROMPT = """你是一位资深情感分析专家，请判断以下句子的情感类别：
# 「平淡」「开心」「悲伤」「愤怒」「惊奇」「厌恶」「疑问」「关切」
# 仅输出情感类别，不要包含任何其他内容或解释。
# """

# # === 推理函数 ===
# def predict(messages_list, model, tokenizer, max_new_tokens=500):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     prompts = [
#         tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True,
#             enable_thinking=False
#         ) for messages in messages_list
#     ]
#     inputs = tokenizer(
#         prompts,
#         return_tensors="pt",
#         truncation=True,
#         max_length=512,
#         padding="longest"
#     ).to(device)

#     generation_config = {
#         "max_new_tokens": max_new_tokens,
#         "temperature": 0,
#         "do_sample": False,
#         "num_beams": 1,
#         "pad_token_id": tokenizer.eos_token_id,
#     }

#     with torch.no_grad():
#         outputs = model.generate(**inputs, **generation_config)

#     input_len = inputs["input_ids"].shape[1]
#     return tokenizer.batch_decode(
#         [output[input_len:] for output in outputs],
#         skip_special_tokens=True
#     )


# # === 加载数据 ===
# with open(json_path, "r", encoding="utf-8") as f:
#     data = json.load(f)

# src_texts = [item["text"] for item in data]
# labels = [item["emotion"] for item in data]  # 仅用于对比

# messages_list = [
#     [
#         {"role": "system", "content": PROMPT},
#         {"role": "user", "content": text}
#     ]
#     for text in src_texts
# ]
# messages_list = messages_list[:10]  # 仅用于测试

# # === 推理处理 ===
# results = []
# pbar = tqdm(total=len(messages_list), desc="预测情绪")

# for start in range(0, len(messages_list), BATCH_SIZE):
#     end = min(start + BATCH_SIZE, len(messages_list))
#     batch_messages = messages_list[start:end]
#     batch_outputs = predict(batch_messages, model, tokenizer)

#     for i, output in enumerate(batch_outputs):
#         results.append({
#             "text": src_texts[start + i],
#             "predicted_emotion": output.strip(),
#             "true_emotion": labels[start + i]  # 可选，用于评估
#         })

#     pbar.update(end - start)

# pbar.close()

# # === 保存结果 ===
# with open(output_dir, "w", encoding="utf-8") as f:
#     json.dump(results, f, ensure_ascii=False, indent=2)

# print(f"✅ 已保存情绪分析结果到 {output_dir}")


import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# === 配置参数 ===
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
json_path = "/data2/jwllm/datasets/multi_emotion/multi_emotion_test.json"
# model_path = "/data2/jwllm/models_origin/Qwen3-0.6B"
model_path = "/data2/jwllm/model_process/Qwen3-0.6B-Base-sft-round-0"
output_dir = "/data2/jwllm/inference_output/emotion_prediction_base_round_0_newprompt.json"

MAX_NEW_TOKENS = 10  # 限制输出长度
BATCH_SIZE = 16

# === 加载模型与 tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).cuda().eval()

# === Prompt 设定 ===
# PROMPT = """请分析下文情感，仅输出以下情感类别之一：
# 「平淡」「开心」「悲伤」「愤怒」「惊奇」「厌恶」「疑问」「关切」
# 不添加任何解释或额外内容，只返回单个词，例如："开心"。
# """
PROMPT = """"你是一位资深情感分析专家，擅长判断用户语句中所表达的主要情绪。
情绪类别：「平淡」「开心」「悲伤」「愤怒」「惊奇」「厌恶」「疑问」「关切」。
输出情绪类别，不要包含任何其他文字或符号。
"""
# 示例：
# 用户输入："你要不要去吃午餐？"
# 模型输出："平淡"

# 用户输入："诶诶诶！我甄选上了！"
# 模型输出："开心"

# 用户输入："我几天身体好像有点不太舒服，肚子好痛"
# 模型输出："悲伤"

# 用户输入："我的小专题组员都不做事，干!超后悔跟他一组"
# 模型输出："愤怒"

# 用户输入："你知道吗?我们班有人交女朋友诶!"
# 模型输出："惊奇"

# 用户输入："干！他真的有够恶心，卫生习惯有够差。"
# 模型输出："厌恶"

# 用户输入："你觉得平行宇宙真的存在吗？"
# 模型输出："疑问"

# 用户输入："你有去看医生了吗?需不需要我陪你？"
# 模型输出："关切"

# 现在，请判断以下句子的情感类别：
# """

# === 清理输出文本（处理乱码和多余字符）===
def clean_output(text):
    return text.strip().replace("\n", "").replace(" ", "").replace("⃗", "").replace("ujące", "")

# === 推理函数 ===
def predict(messages_list, model, tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = [
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        ) for messages in messages_list
    ]

    inputs = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=512, padding="longest").to(device)

    generation_config = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": 0,  # 设为 0 让模型输出最确定的答案
        "do_sample": False,  # 禁用随机采样，确保稳定输出
        "num_beams": 1,  # 降低束搜索，避免模型生成额外文本
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.2  # 设置重复惩罚

    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_config)

    input_len = inputs["input_ids"].shape[1]
    return [clean_output(output) for output in tokenizer.batch_decode([output[input_len:] for output in outputs], skip_special_tokens=True)]

# === 加载数据 ===
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

src_texts = [item["text"] for item in data]
labels = [item["emotion"] for item in data]  # 仅用于对比

messages_list = [
    [{"role": "system", "content": PROMPT}, {"role": "user", "content": text}]
    for text in src_texts
]
messages_list = messages_list[:1000]  # 仅用于测试

# === 推理处理 ===
results = []
pbar = tqdm(total=len(messages_list), desc="预测情绪")

for start in range(0, len(messages_list), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(messages_list))
    batch_messages = messages_list[start:end]
    batch_outputs = predict(batch_messages, model, tokenizer)

    for i, output in enumerate(batch_outputs):
        results.append({
            "text": src_texts[start + i],
            "predicted_emotion": output.strip(),
            "true_emotion": labels[start + i]  # 可选，用于评估
        })

    pbar.update(end - start)

pbar.close()

# === 保存结果 ===
with open(output_dir, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"✅ 已保存情绪分析结果到 {output_dir}")
