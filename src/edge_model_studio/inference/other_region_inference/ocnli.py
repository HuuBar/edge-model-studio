import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# === 路径配置 ===
json_path = "/data2/jwllm/datasets/ocnli/ocnli_test.json"

# 需要推理的多个剪枝微调模型轮次
models = [
    "qwen3_0.6b_base",
    # "qwen3_0.6b_base_sft_round_0",
    # "qwen3_0.6b_base_sft_round_1",
    # "qwen3_0.6b_base_sft_round_2",
    # "qwen3_0.6b_base_sft_round_3",
    # "qwen3_0.6b_base_sft_round_4",
    # "qwen3_0.6b_base_sft_round_4_int8"
]

output_base = "/data2/jwllm/inference_output"

for local_model in models:
    model_path = f"/data2/jwllm/model_process/{local_model}"
    output_dir = os.path.join(output_base, local_model)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成文件路径
    output_file = os.path.join(output_dir, f"ocnli_{local_model}.json")

    MAX_NEW_TOKENS = 10
    BATCH_SIZE = 8

    if "int8" in local_model:
        print(f"识别到 {local_model} 为 INT8 量化模型，使用 device_map='auto'")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto"  # INT8 量化模型自动分配设备
        ).eval()
    else:
        print(f"加载 {local_model} 为 FP16 模型，使用 torch_dtype=torch.float16")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).cuda().eval()  # FP16 需要手动 .cuda()

    # === 加载模型与 tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"

    # === Prompt设定 ===
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
            # "temperature": 0,
            "do_sample": False,
            "num_beams": 1,
            "pad_token_id": tokenizer.eos_token_id,
            "repetition_penalty": 1.2
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
        labels.append(item.get("label", ""))

        messages_list.append([
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f"sentence1：{s1}\nsentence2：{s2}"}
        ])

    # === 批量推理 ===
    results = []
    # pbar = tqdm(total=len(messages_list), desc=f"推理逻辑关系：{model}")
    pbar = tqdm(total=len(messages_list), desc=f"推理逻辑关系：{local_model}")


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
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 已保存 {local_model} 逻辑关系预测结果到：{output_file}")
