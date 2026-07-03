import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# === 配置路径 ===
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
json_path = "/data2/jwllm/datasets/medicial_dialogue/medical_dialogue_test.json"

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
    output_file = os.path.join(output_dir, f"medical_qa_results_{local_model}.json")

    MAX_NEW_TOKENS = 350
    BATCH_SIZE = 32

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

    # === Prompt设定 ===
    PROMPT = """你是一位经验丰富的儿科医生，擅长根据患儿病情描述给出专业、清晰、有同理心的医学建议。不超过200字。"""

    # === 推理函数 ===
    def predict(messages_list, llm_model, tokenizer, max_new_tokens=256):
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
            "do_sample": True,
            "num_beams": 1,
            "pad_token_id": tokenizer.eos_token_id,
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

    asks = [item["ask"] for item in data]
    refs = [item["answer"] for item in data]  # 参考答案（可选）

    messages_list = [
        [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": ask}
        ]
        for ask in asks
    ]

    # === 批量推理 ===
    results = []
    pbar = tqdm(total=len(messages_list), desc=f"生成医疗问答 - {local_model}")

    for start in range(0, len(messages_list), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(messages_list))
        batch_messages = messages_list[start:end]
        batch_outputs = predict(batch_messages, llm_model, tokenizer)

        for i, output in enumerate(batch_outputs):
            results.append({
                "ask": asks[start + i],
                "llm_answer": output.strip(),
                "ref_answer": refs[start + i]
            })

        pbar.update(end - start)

    pbar.close()

    # === 保存结果 ===
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 医疗问答结果已保存至：{output_file}")
