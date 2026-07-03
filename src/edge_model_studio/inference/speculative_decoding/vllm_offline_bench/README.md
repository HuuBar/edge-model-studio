# vLLM Offline Baseline / EAGLE3 Benchmark 精简版

本目录只保留 5 个常用功能：

```text
4. 单独测试 baseline
5. 单独测试 EAGLE3 spec
6. 比较 baseline 和 spec
7. 一条命令跑 baseline + spec + compare
16. 只统计 baseline/spec 指标并输出 Excel，不计算性能对比
```

已移除批量矩阵、batch size sweep、示例 shell 等辅助功能，避免使用路径过多。

---

## 0. 先设置模型路径

进入脚本目录：

```bash
cd /workspace/../vllm_offline_bench
```

设置目标模型和草稿模型路径：

```bash
export TARGET_MODEL=/workspace/shangyangyang/models/Qwen2.5-14B-Instruct
export DRAFT_MODEL=/workspace/../vl_spec/speculators/scripts/checkpoints/checkpoint_best
```

确认脚本可执行：

```bash
chmod +x *.py
```

---

## 1. 输出目录说明

单独测试 baseline 或 spec 时，输出目录中会包含：

```text
*.summary.json      单次测试 summary，后续 compare 或 Excel 读取它
*.requests.jsonl    每条请求的输入/输出 token 数等记录
*.batches.jsonl     每个 batch 的耗时记录
summary.csv         当前目录下的 summary 汇总
```

`run_metrics_excel.py` 最终只输出：

```text
performance_metrics.xlsx
```

但它内部仍会保留 baseline/spec 的 summary 文件，用于生成 Excel。

---

## 2. 单独测试 baseline

用于只测试目标模型基线性能。

```bash
python run_offline_bench.py \
  --mode baseline \
  --target-model ${TARGET_MODEL} \
  --input-len 512 \
  --output-len 256 \
  --num-prompts 16 \
  --batch-size 1 \
  --warmup-batches 1 \
  --temperature 0 \
  --top-p 1 \
  --ignore-eos \
  --tensor-parallel-size 1 \
  --hardware "Ascend 910B" \
  --tester "tester" \
  --collect-npu-smi \
  --output-dir bench_results/synthetic_baseline_in512_out256_bs1
```

重点看 summary 中的字段：

```text
output_tokens_per_s
total_tokens_per_s
avg_ms_per_output_token
batch_latency_avg_ms
batch_latency_p50_ms
batch_latency_p90_ms
batch_latency_p99_ms
```

说明：

```text
不传 --prompt-file 时，脚本会自动生成 synthetic 固定长度输入。
--input-len 512 表示每条输入目标长度为 512 tokens。
--output-len 256 表示每条最多输出 256 tokens。
--ignore-eos 表示忽略 EOS，尽量固定输出长度，便于性能比较。
```

---

## 3. 单独测试 EAGLE3 spec

用于只测试目标模型挂载 EAGLE3 草稿模型后的性能。

```bash
python run_offline_bench.py \
  --mode spec \
  --target-model ${TARGET_MODEL} \
  --draft-model ${DRAFT_MODEL} \
  --spec-method eagle3 \
  --num-speculative-tokens 4 \
  --draft-tensor-parallel-size 1 \
  --input-len 512 \
  --output-len 256 \
  --num-prompts 16 \
  --batch-size 1 \
  --warmup-batches 1 \
  --temperature 0 \
  --top-p 1 \
  --ignore-eos \
  --tensor-parallel-size 1 \
  --hardware "Ascend 910B" \
  --tester "tester name" \
  --collect-npu-smi \
  --output-dir bench_results/synthetic_spec_in512_out256_bs1_k4
```

重点看 spec 相关字段：

```text
spec_metrics_available
spec_acceptance_rate
spec_accepted_tokens_per_draft
spec_mean_acceptance_length_including_bonus
spec_accepted_per_pos_rates
```

字段含义：

```text
spec_acceptance_rate
    被目标模型接受的草稿 token 数 / 草稿 token 总数。

spec_accepted_tokens_per_draft
    每轮 draft 平均被接受的草稿 token 数。

spec_mean_acceptance_length_including_bonus
    1 + spec_accepted_tokens_per_draft，粗略表示每轮验证平均推进 token 数。

spec_accepted_per_pos_rates
    每个草稿位置的接受率。比如 k=4 时，0/1/2/3 分别对应第 1/2/3/4 个草稿 token。
```

如果 `spec_metrics_available=false`，说明当前 vLLM 版本没有暴露 speculative metrics；此时仍可看吞吐和耗时，但不能分析接受率。

---

## 4. 比较 baseline 和 spec

### 4.1 自动比较两个目录下最新结果

推荐使用 `compare_latest.py`，不用手动找 summary 文件名。

```bash
python compare_latest.py \
  --baseline-dir bench_results/synthetic_baseline_in512_out256_bs1 \
  --spec-dir bench_results/synthetic_spec_in512_out256_bs1_k4 \
  --output-dir bench_results/synthetic_compare_in512_out256_bs1_k4
```

重点看：

```text
is_strictly_comparable
config_mismatches
output_tps_speedup
total_tps_speedup
ms_per_output_token_speedup
batch_latency_speedup
wall_time_speedup
spec_acceptance_rate
spec_accepted_tokens_per_draft
```

判断方式：

```text
output_tps_speedup > 1
    spec 输出 token 吞吐高于 baseline。

ms_per_output_token_speedup > 1
    spec 平均每输出 token 耗时低于 baseline。

batch_latency_speedup > 1
    spec batch 平均耗时低于 baseline。
```

如果 speedup 小于 1，表示挂草稿模型后变慢。

### 4.2 显式指定 summary 文件比较

如果你要指定某次运行结果：

```bash
python compare_runs.py \
  --baseline bench_results/synthetic_baseline_in512_out256_bs1/<baseline_xxx>.summary.json \
  --spec bench_results/synthetic_spec_in512_out256_bs1_k4/<spec_xxx>.summary.json \
  --output-dir bench_results/synthetic_compare_in512_out256_bs1_k4
```

---

## 5. 一条命令跑 baseline + spec + compare

如果你想一个 case 一条命令跑完 baseline、spec 和对比，用：

```bash
python run_pair_offline_bench.py \
  --target-model ${TARGET_MODEL} \
  --draft-model ${DRAFT_MODEL} \
  --input-len 512 \
  --output-len 256 \
  --num-prompts 16 \
  --batch-size 1 \
  --num-speculative-tokens 4 \
  --warmup-batches 1 \
  --temperature 0 \
  --top-p 1 \
  --ignore-eos \
  --tensor-parallel-size 1 \
  --hardware "Ascend 910B" \
  --tester "tester" \
  --collect-npu-smi \
  --output-root bench_results/pair_in512_out256_bs1_k4
```

输出结构：

```text
bench_results/pair_in512_out256_bs1_k4/
  baseline/
  spec_k4/
  compare/
```

说明：

```text
run_pair_offline_bench.py 会计算 speedup。
它适合快速判断 spec 是否比 baseline 更快。
它不生成 performance_metrics.xlsx。
```

---

## 6. 只统计 baseline/spec 指标并输出 Excel，不计算性能对比

如果你只想生成汇报用 Excel，并且不想在表格里计算 speedup，用：

```bash
python run_metrics_excel.py \
  --target-model ${TARGET_MODEL} \
  --draft-model ${DRAFT_MODEL} \
  --target-name "Qwen2.5-14B-Instruct" \
  --draft-name "draftmodel" \
  --spec-method eagle3 \
  --num-speculative-tokens 4 \
  --draft-tensor-parallel-size 1 \
  --input-len 512 \
  --output-len 256 \
  --num-prompts 16 \
  --batch-size 1 \
  --warmup-batches 1 \
  --temperature 0 \
  --top-p 1 \
  --ignore-eos \
  --tensor-parallel-size 1 \
  --hardware "Ascend 910B" \
  --tester "tester" \
  --collect-npu-smi \
  --output-root bench_results/metrics_excel_in512_out256_bs1_k4
```

最终输出：

```text
bench_results/metrics_excel_in512_out256_bs1_k4/performance_metrics.xlsx
```

Excel 中的测试对象显示为：

```text
Qwen2.5-14B-Instruct
Qwen2.5-14B-Instruct+draftmodel
```

说明：

```text
1. run_metrics_excel.py 不计算 speedup。
2. baseline 没有的 speculative 指标会写“无”。
3. spec 模式应产出但未产出的 speculative 指标会写“异常”。
4. 离线测试无法准确测 TTFT/ITL/TPOT，因此在线测试预留字段会写“无”。
5. 最终只输出 xlsx，不额外输出 csv/json。
```

如果你已经有 baseline/spec 的 summary 文件，只想生成 Excel：

```bash
python run_metrics_excel.py \
  --baseline-summary-json bench_results/synthetic_baseline_in512_out256_bs1/<baseline_xxx>.summary.json \
  --spec-summary-json bench_results/synthetic_spec_in512_out256_bs1_k4/<spec_xxx>.summary.json \
  --target-name "Qwen2.5-14B-Instruct" \
  --draft-name "draftmodel" \
  --output-root bench_results/metrics_excel_from_existing
```

---

## 常见判断逻辑

### spec 变快

```text
output_tps_speedup > 1
ms_per_output_token_speedup > 1
batch_latency_speedup > 1
```

### spec 变慢

```text
output_tps_speedup < 1
ms_per_output_token_speedup < 1
batch_latency_speedup < 1
```

优先看：

```text
spec_acceptance_rate
spec_accepted_tokens_per_draft
spec_accepted_per_pos_rates
```

如果 acceptance 很低，优先排查：

```text
1. 草稿模型是否基于同一个 target model 训练；
2. tokenizer/config 是否匹配；
3. num_speculative_tokens 是否过大；
4. 测试 prompt 分布是否和训练分布差异太大。
```
