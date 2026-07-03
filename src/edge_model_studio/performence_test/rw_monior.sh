#!/bin/bash
set -euo pipefail

MODEL_LIST=(
    ""
)
PROMPT_FILE=""
RESULTS_FILE=""

: > "$RESULTS_FILE"

for MODEL_PATH in "${MODEL_LIST[@]}"; do
    sync
    sudo bash -c 'echo 3 > /proc/sys/vm/drop_caches'
    sleep 5
    rm -rf "./tmp/tmp_$(basename "$MODEL_PATH")"
    
    MODEL_NAME=$(basename "$MODEL_PATH")
    echo "===============================" | tee -a "$RESULTS_FILE"
    echo "开始测试模型: $MODEL_NAME" | tee -a "$RESULTS_FILE"
    echo "路径: $MODEL_PATH" | tee -a "$RESULTS_FILE"
    echo "===============================" | tee -a "$RESULTS_FILE"

    TMP_DIR="./tmp/tmp_${MODEL_NAME}"
    IO_FILE="${TMP_DIR}/mole_io_proc_io.txt"

    mkdir -p "$TMP_DIR"

    START_TIME_NS=$(date +%s.%N)
    echo "start_monitor,$START_TIME_NS" >> "${TMP_DIR}/mole_times.txt"

    python inference.py "$START_TIME_NS" "$MODEL_PATH" "$PROMPT_FILE" "$IO_FILE" 2>&1 | tee "${TMP_DIR}/inference_output.txt"

    python data_parse.py "$TMP_DIR" | tee -a "$RESULTS_FILE"

    echo -e "\n\n" >> "$RESULTS_FILE"
done

echo "批量测试完成！统计结果已保存到 $RESULTS_FILE"