#!/bin/bash

echo "执行 /data2/jwllm/scripts/batch_inference/nli.py"
python /data2/jwllm/scripts/batch_inference/nli.py

echo "执行 /data2/jwllm/scripts/batch_inference/emotion.py"
python /data2/jwllm/scripts/batch_inference/emotion.py

echo "执行 summary.py"
python /data2/jwllm/scripts/batch_inference/summary.py

echo "执行 medical_qa.py"
python /data2/jwllm/scripts/batch_inference/medical_qa.py

echo "✅ 所有脚本已顺序执行完毕"
