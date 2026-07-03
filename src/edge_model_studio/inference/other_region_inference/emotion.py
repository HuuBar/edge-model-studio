import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# === 配置参数 ===
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
json_path = "/data2/jwllm/datasets/multi_emotion/multi_emotion_test.json"

# 需要推理的多个剪枝微调模型轮次
models = [
    "qwen3_0.6b_base",
    # "qwen3_0.6b_base_sft_round_1",
    # "qwen3_0.6b_base_sft_round_2",
    # "qwen3_0.6b_base_sft_round_3",
    # "qwen3_0.6b_base_sft_round_4",
    # "qwen3_0.6b_base_sft_round_4_int8"  # INT8 量化模型
]

output_base = "/data2/jwllm/inference_output"

for local_model in models:
    model_path = f"/data2/jwllm/model_process/{local_model}"
    output_dir = os.path.join(output_base, local_model)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成文件路径
    output_file = os.path.join(output_dir, f"emotion_prediction_{local_model}.json")

    MAX_NEW_TOKENS = 10
    BATCH_SIZE = 16

    # === 自动检测 INT8 或 FP16 ===
    if "int8" in local_model:
        print(f"⚡ 识别到 {local_model} 为 INT8 量化模型，使用 device_map='auto'")
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto"  # INT8 量化模型自动分配设备
        ).eval()
    else:
        print(f"🚀 加载 {local_model} 为 FP16 模型，使用 torch_dtype=torch.float16")
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).cuda().eval()  # FP16 需要手动 .cuda()

    # === 加载 Tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"

    # === Prompt 设定 ===
    PROMPT = """你是一位资深情感分析专家，擅长判断用户语句中所表达的主要情绪。
    情绪类别：「平淡」「开心」「悲伤」「愤怒」「惊奇」「厌恶」「疑问」「关切」。
    输出情绪类别，不要包含任何其他文字或符号。"""

    # === 清理输出文本（处理乱码和多余字符）===
    def clean_output(text):
        return text.strip().replace("\n", "").replace(" ", "").replace("⃗", "").replace("ujące", "")

    # === 推理函数 ===
    def predict(messages_list, llm_model, tokenizer):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        prompts = [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            ) for messages in messages_list
        ]

        inputs = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=512, padding="longest").to(device)

        generation_config = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0,
            "do_sample": False,
            "num_beams": 1,
            "pad_token_id": tokenizer.eos_token_id,
            "repetition_penalty": 1.2
        }

        with torch.no_grad():
            outputs = llm_model.generate(**inputs, **generation_config)

        input_len = inputs["input_ids"].shape[1]
        return [clean_output(output) for output in tokenizer.batch_decode([output[input_len:] for output in outputs], skip_special_tokens=True)]

    # === 加载数据 ===
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    src_texts = [item["text"] for item in data]
    labels = [item["emotion"] for item in data]  # 可选，用于评估

    messages_list = [
        [{"role": "system", "content": PROMPT}, {"role": "user", "content": text}]
        for text in src_texts
    ]

    # === 推理处理 ===
    results = []
    pbar = tqdm(total=len(messages_list), desc=f"预测情绪 - {local_model}")

    for start in range(0, len(messages_list), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(messages_list))
        batch_messages = messages_list[start:end]
        batch_outputs = predict(batch_messages, llm_model, tokenizer)

        for i, output in enumerate(batch_outputs):
            results.append({
                "text": src_texts[start + i],
                "predicted_emotion": output.strip(),
                "true_emotion": labels[start + i]
            })

        pbar.update(end - start)

    pbar.close()

    # === 保存结果 ===
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 已保存情绪分析结果到 {output_file}")
