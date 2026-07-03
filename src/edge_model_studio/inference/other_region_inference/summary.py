import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# === 配置路径 ===
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
json_path = "/data2/jwllm/datasets/clts_datasets/clts_filter_test.json"

# 需要推理的多个剪枝微调模型轮次
models = [
    # "qwen3_0.6b_base",
    "Qwen3-1.7B"
    # "qwen3_0.6b_base_sft_round_0",
    # "qwen3_0.6b_base_sft_round_1",
    # "qwen3_0.6b_base_sft_round_2",
    # "qwen3_0.6b_base_sft_round_3",
    # "qwen3_0.6b_base_sft_round_4",
    # "qwen3_0.6b_base_sft_round_4_int8"  # INT8 量化模型
]

output_base = "/data2/jwllm/inference_output"

for local_model in models:
    model_path = f"/data2/jwllm/models_origin/{local_model}"
    output_dir = os.path.join(output_base, local_model)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成文件路径
    output_file = os.path.join(output_dir, f"summary_{local_model}_test.json")

    MAX_NEW_TOKENS = 80
    BATCH_SIZE = 8

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

    # === 文章摘要 Prompt ===
    PROMPT = "你是一位专业新闻编辑，精通文章写作与摘要技巧，能够准确提炼关键信息，不得虚构、猜测或省略原文中的重要数据，确保摘要真实、客观、精炼。请基于上述内容，生成一段不超过70字的摘要。"

    # === 推理函数 ===
    def predict(messages_list, llm_model, tokenizer, max_new_tokens=80):
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
            "do_sample": False,
            "num_beams": 4,
            "pad_token_id": tokenizer.eos_token_id,
            "repetition_penalty": 1.2,
        }

        with torch.no_grad():
            outputs = llm_model.generate(**inputs, **generation_config)

        input_len = inputs["input_ids"].shape[1]
        return tokenizer.batch_decode(
            [output[input_len:] for output in outputs],
            skip_special_tokens=True
        )

    # === 加载数据 ===
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # data = data[:5]

    src_lines = [item["context"] for item in data]
    tgt_lines = [item["ground_truth"] for item in data]
    assert len(src_lines) == len(tgt_lines), "源文本与摘要数量不一致"

    messages_list = [
        [{"role": "system", "content": PROMPT}, {"role": "user", "content": context}]
        for context in src_lines
    ]

    # === 批量推理 ===
    results = []
    pbar = tqdm(total=len(src_lines), desc=f"生成摘要 - {local_model}")

    for start in range(0, len(messages_list), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(messages_list))
        batch_messages = messages_list[start:end]
        batch_outputs = predict(batch_messages, llm_model, tokenizer, max_new_tokens=MAX_NEW_TOKENS)

        for i, output in enumerate(batch_outputs):
            results.append({
                "context": src_lines[start + i],
                "ground_truth": tgt_lines[start + i],
                "llm_answer": output.strip()
            })

        pbar.update(end - start)

    pbar.close()

    # === 保存结果 ===
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 已保存摘要生成结果到 {output_file}")
