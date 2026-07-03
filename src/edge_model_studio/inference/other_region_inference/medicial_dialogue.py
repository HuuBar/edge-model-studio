
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# === 配置路径 ===
json_path = "/data2/jwllm/datasets/medicial_dialogue/medical_dialogue_test.json"
# model_path = "/data2/jwllm/models_origin/Qwen3-0.6B"
model_path = "/data2/jwllm/model_process/Qwen3-0.6B-Base-sft-round-0"

output_dir = "/data2/jwllm/inference_output/medical_qa_results_base_round_0.json"

MAX_NEW_TOKENS = 350
BATCH_SIZE = 16

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
#     "你是一位经验丰富的儿科医生，擅长根据患儿病情描述给出专业、清晰、有同理心的医学建议。"
#     "请你认真阅读用户的提问，判断问题涉及的主要健康问题，并提供专业的诊疗建议。"
#     "答复应准确、客观，尽量避免重复、空泛，语言简洁清晰，建议就诊时请提醒去正规三甲医院。"
# )

        
PROMPT = """你是一位经验丰富的儿科医生，擅长根据患儿病情描述给出专业、清晰、有同理心的医学建议。不超过200字."""

# === 推理函数 ===
def predict(messages_list, model, tokenizer, max_new_tokens=256):
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
        max_length=1024,
        padding="longest"
    ).to(device)

    generation_config = {
        "max_new_tokens": max_new_tokens,
        "temperature": 0.7,
        "do_sample": False,
        "num_beams": 1,
        "pad_token_id": tokenizer.eos_token_id,
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

asks = [item["ask"] for item in data]
refs = [item["answer"] for item in data]  # 参考答案（可选）

messages_list = [
    [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": ask}
    ]
    for ask in asks
]

# messages_list = messages_list[:10]  # 限制推理数量，用于测试

# === 批量推理 ===
results = []
pbar = tqdm(total=len(messages_list), desc="生成医疗问答")

for start in range(0, len(messages_list), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(messages_list))
    batch_messages = messages_list[start:end]
    batch_outputs = predict(batch_messages, model, tokenizer)

    for i, output in enumerate(batch_outputs):
        results.append({
            "ask": asks[start + i],
            "llm_answer": output.strip(),
            "ref_answer": refs[start + i]
        })

    pbar.update(end - start)

pbar.close()

# === 保存结果 ===
with open(output_dir, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"✅ 医疗问答结果已保存至：{output_dir}")