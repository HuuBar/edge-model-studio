#!/bin/bash

# 先启动服务，示例：
# vllm serve Qwen3-30B-A3B \
#   --served-model-name "Qwen3-30B-A3B-w8a8" \
#   --max-model-len 8192 \
#   --tensor-parallel-size 1 \
#   --data-parallel-size 2 \
#   --gpu-memory-utilization 0.85 \
#   --host 0.0.0.0 \
#   --port 8000

# ================= 配置区域 =================

# vLLM OpenAI API 服务地址
BASE_URL="http://127.0.0.1:8000"

# 注意：这里必须和 vllm serve 的 --served-model-name 一致
MODEL_NAME="Qwen3-30B-A3B-w8a8-FDO"

# tokenizer 路径，用于本地统计 input/output token 数
TOKENIZER="tokenizer_path"

INPUT_LEN=2048
OUTPUT_LEN=400
NUM_PROMPTS=1000

# online 里没有 batch-size，改成 max-concurrency
# 你原来 bs40，这里对应可以先用 concurrency=40
MAX_CONCURRENCY=40

# request-rate=0 表示尽快提交请求，由 max-concurrency 控制最大并发
REQUEST_RATE=0

TEMPERATURE=0
TOP_P=1
HARDWARE="Ascend 910B"
TESTER="tester"

# 是否开启 streaming
# 开启后可以统计 TTFT / ITL / TPOT
STREAM_FLAG="--stream"

# 定义变化的参数
# 格式: "prompt_file|output_dir|max_concurrency"
TASKS=(
    "prompt_file_1|output_dir_1|max_concurrency_1"
    "prompt_file_2|output_dir_2|max_concurrency_2"
    "prompt_file_3|output_dir_3|max_concurrency_3"
)

# ================= 执行逻辑 =================

echo "=========================================="
echo "开始批量 online 基准测试，共 ${#TASKS[@]} 个任务"
echo "BASE_URL   : ${BASE_URL}"
echo "MODEL_NAME : ${MODEL_NAME}"
echo "=========================================="

# 先检查服务是否可用
echo ""
echo ">>> 检查 vLLM 服务 health"
curl -i "${BASE_URL}/health"

echo ""
echo ">>> 检查模型列表"
curl "${BASE_URL}/v1/models"
echo ""

for task in "${TASKS[@]}"; do
    IFS='|' read -r PROMPT_FILE OUTPUT_DIR MAX_CONCURRENCY <<< "$task"

    echo ""
    echo ">>> 正在启动 online 任务:"
    echo "    Prompt File     : $PROMPT_FILE"
    echo "    Output Dir      : $OUTPUT_DIR"
    echo "    Max Concurrency : $MAX_CONCURRENCY"
    echo "------------------------------------------"

    mkdir -p "$OUTPUT_DIR"

    python run_online_bench.py \
        --base-url "${BASE_URL}" \
        --model "${MODEL_NAME}" \
        --tokenizer "${TOKENIZER}" \
        --endpoint chat \
        ${STREAM_FLAG} \
        --num-prompts "${NUM_PROMPTS}" \
        --max-concurrency "${MAX_CONCURRENCY}" \
        --request-rate "${REQUEST_RATE}" \
        --input-len "${INPUT_LEN}" \
        --output-len "${OUTPUT_LEN}" \
        --prompt-file "${PROMPT_FILE}" \
        --normalize-file-prompts \
        --temperature "${TEMPERATURE}" \
        --top-p "${TOP_P}" \
        --ignore-eos \
        --hardware "${HARDWARE}" \
        --tester "${TESTER}" \
        --output-dir "${OUTPUT_DIR}"

    if [ $? -eq 0 ]; then
        echo ">>> online 任务完成: $OUTPUT_DIR"
    else
        echo ">>> [错误] online 任务失败: $OUTPUT_DIR，请检查日志"
    fi
    sleep 5
done

echo ""
echo "=========================================="
echo "所有 online 任务已执行完毕"
echo "=========================================="