# vllm_online_bench - 在线测试工具

通过 YAML 配置文件启动 vLLM 在线性能测试。

## 快速开始

### 1. CLI 模式

```bash
python run_online_bench.py \
    --base-url http://127.0.0.1:8000 \
    --model Qwen3-30B-A3B-w8a8 \
    --num-prompts 16 \
    --max-concurrency 1
```

### 2. YAML 配置模式

```bash
python run_online_bench.py --config example.yaml
```

#### Dry Run 验证配置

```bash
python run_online_bench.py --config example.yaml --dry-run
```

## YAML 配置文件格式

参考example.yaml

## 输出文件

测试完成后在 `output-dir` 目录下生成：

| 文件 | 说明 |
|------|------|
| `{case_id}_{run_id}.summary.json` | 测试汇总结果 |
| `{case_id}_{run_id}.requests.jsonl` | 每个请求的详细结果 |
| `summary.csv` | 所有测试的 CSV 格式汇总 |
| `summary.xlsx` | 所有测试的 Excel 格式汇总 |

## 指标说明

| 指标 | 说明 |
|------|------|
| `request_throughput_req_s` | 请求吞吐量 (req/s) |
| `output_tokens_per_s` | 输出 token 吞吐量 |
| `total_tokens_per_s` | 总 token 吞吐量 |
| `latency_avg_ms` | 端到端延迟均值 |
| `latency_p50/p90/p99_ms` | 端到端延迟分位数 |
| `ttft_avg_ms` | 首 token 时间均值（需开启流式） |
| `ttft_p50/p90/p99_ms` | 首 token 时间分位数 |
| `tpot_avg_ms` | 输出每个 token 时间均值（需开启流式） |
| `tpot_p50/p90/p99_ms` | 输出每个 token 时间分位数 |
| `itl_avg_ms` | token 间延迟均值（需开启流式） |
| `itl_p50/p90/p99_ms` | token 间延迟分位数 |

## 注意事项

1. **YAML 字段名与 CLI 参数名一致**：YAML 中的字段名直接对应 CLI 参数（加 `--` 前缀），无需映射。
2. **多客户端执行**：YAML 中 `client` 可以是单个对象或对象数组，会按顺序依次执行。
