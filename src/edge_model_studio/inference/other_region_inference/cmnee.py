import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# === 配置路径 ===
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
json_path = "/data2/jwllm/datasets/cmnee/cmnee_test_simple.json"

# 需要推理的多个剪枝微调模型轮次
models = [
    "qwen3_0.6b_base",
    # "qwen3_0.6b_base_sft_round_0",
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
    output_file = os.path.join(output_dir, f"keywords_extraction_{local_model}.json")

    MAX_NEW_TOKENS = 180
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

    # === 关键词提取 Prompt ===
    PROMPT = """你是一位专业文本分析师，擅长从文章中提取关键词。
    关键词应该是文章中的核心概念、重要事件或关键人物，不需要包含普通词汇或无关信息。
    请确保关键词准确且不超过 10 个。"""

    # === 推理函数 ===
    def predict(messages_list, llm_model, tokenizer, max_new_tokens=300):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        prompts = [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
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
            "repetition_penalty": 1.2
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

    src_lines = [item["context"] for item in data]
    tgt_lines = [item["keywords"] for item in data]  # 参考人工标注

    assert len(src_lines) == len(tgt_lines), "源文本与关键词数量不一致"

    # === 构造对话格式的推理输入 ===
    messages_list = [
        [{"role": "system", "content": PROMPT}, {"role": "user", "content": context}]
        for context in src_lines
    ]

    # === 批量推理 ===
    results = []
    pbar = tqdm(total=len(messages_list), desc=f"提取关键词 - {local_model}")

    for start in range(0, len(messages_list), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(messages_list))
        batch_messages = messages_list[start:end]
        batch_outputs = predict(batch_messages, llm_model, tokenizer, max_new_tokens=MAX_NEW_TOKENS)

        for i, output in enumerate(batch_outputs):
            extracted_keywords = [kw.strip() for kw in output.strip().split(",")]  # 简单分割关键词

            results.append({
                "context": src_lines[start + i],
                "ground_truth": tgt_lines[start + i],  # 参考人工标注
                "llm_extracted_keywords": extracted_keywords  # LLM 提取的关键词
            })

        pbar.update(end - start)

    pbar.close()

    # === 保存结果 ===
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 已保存关键词提取结果到 {output_file}")
