import argparse
import re
import sys
import os
import numpy as np


def parse_log_file(file_path):
    """
    流式解析日志，修复原正则中 .*? 连环嵌套带来的回溯隐患。
    """
    results = []
    if not os.path.exists(file_path):
        print(f"[CRITICAL] Target log file not found: {file_path}", file=sys.stderr)
        return results

    # 优化正则结构：用 [^:]+ 或明确的非换行空白匹配替代 .*?，消除回溯灾难
    # 并且各字段独立提取，增强应对日志格式微调时的鲁棒性
    kv_pattern = re.compile(
        r"TotalTimeMs:(?P<TotalTimeMs>[\d.]+)[^\d]+"
        r"PrefillTimeMs:(?P<PrefillTimeMs>[\d.]+)[^\d]+"
        r"DecodeTimeMs:(?P<DecodeTimeMs>[\d.]+)[^\d]+"
        r"OutputTokenCount:(?P<OutputTokenCount>\d+)[^\d]+"
        r"DecodeNum:(?P<DecodeNum>\d+)"
    )

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = kv_pattern.search(line)
            if not match:
                continue

            try:
                total_time_ms = float(match.group('TotalTimeMs'))
                prefill_time_ms = float(match.group('PrefillTimeMs'))
                decode_time_ms = float(match.group('DecodeTimeMs'))
                output_tokens = int(match.group('OutputTokenCount'))
                decode_num = int(match.group('DecodeNum'))

                # 安全边界防御，防止分母为 0 导致崩溃
                mat = output_tokens / decode_num if decode_num > 0 else 0.0
                decode_tokens_per_sec = (output_tokens / decode_time_ms) * 1000.0 if decode_time_ms > 0 else 0.0
                full_tokens_per_sec = (output_tokens / total_time_ms) * 1000.0 if total_time_ms > 0 else 0.0
                decode_time_per_step = decode_time_ms / decode_num if decode_num > 0 else 0.0

                results.append({
                    'TotalTimeMs': total_time_ms,
                    'PrefillTimeMs': prefill_time_ms,
                    'DecodeTimeMs': decode_time_ms,
                    'OutputTokenCount': output_tokens,
                    'DecodeNum': decode_num,
                    'MAT': mat,
                    'DecodeTokensPerSec': decode_tokens_per_sec,
                    'FullTokensPerSec': full_tokens_per_sec,
                    'DecodeTimePerStep': decode_time_per_step
                })
            except (ValueError, ZeroDivisionError):
                continue
                
    return results


def print_metrics_summary(data):
    """
    打印详细指标矩阵与分位数统计。
    """
    if not data:
        print("[WARN] No valid metric data collected.")
        return

    # 1. 打印全量样本流水明细
    print("-" * 100)
    print(f"{'Idx':<6} {'MAT':<10} {'Decode(tok/s)':<18} {'Full(tok/s)':<18} {'DecMs/Step':<14} {'PrefillMs':<12}")
    print("-" * 100)
    for idx, entry in enumerate(data):
        print(
            f"{idx+1:<6}"
            f"{entry['MAT']:<10.4f}"
            f"{entry['DecodeTokensPerSec']:<18.2f}"
            f"{entry['FullTokensPerSec']:<18.2f}"
            f"{entry['DecodeTimePerStep']:<14.2f}"
            f"{entry['PrefillTimeMs']:<12.2f}"
        )

    # 2. 利用 Numpy 进行高维度统计分析（计算均值与尾部延迟）
    mats = np.array([x['MAT'] for x in data])
    dec_tps = np.array([x['DecodeTokensPerSec'] for x in data])
    full_tps = np.array([x['FullTokensPerSec'] for x in data])
    dec_step = np.array([x['DecodeTimePerStep'] for x in data])
    prefills = np.array([x['PrefillTimeMs'] for x in data])

    print("\n" + "=" * 100)
    print(f"统计总览 (样本数: {len(data)})")
    print("-" * 100)
    print(f"{'Metric':<25} | {'Average':<12} | {'P50':<12} | {'P95':<12} | {'P99':<12}")
    print("-" * 100)
    print(f"{'MAT (Mean Accept Toks)':<25} | {np.mean(mats):<12.4f} | {np.percentile(mats, 50):<12.4f} | {np.percentile(mats, 95):<12.4f} | {np.percentile(mats, 99):<12.4f}")
    print(f"{'Decode Throughput (tok/s)':<25} | {np.mean(dec_tps):<12.2f} | {np.percentile(dec_tps, 50):<12.2f} | {np.percentile(dec_tps, 95):<12.2f} | {np.percentile(dec_tps, 99):<12.2f}")
    print(f"{'Full Throughput (tok/s)':<25} | {np.mean(full_tps):<12.2f} | {np.percentile(full_tps, 50):<12.2f} | {np.percentile(full_tps, 95):<12.2f} | {np.percentile(full_tps, 99):<12.2f}")
    print(f"{'Decode Time/Step (ms)':<25} | {np.mean(dec_step):<12.2f} | {np.percentile(dec_step, 50):<12.2f} | {np.percentile(dec_step, 95):<12.2f} | {np.percentile(dec_step, 99):<12.2f}")
    print(f"{'Prefill Latency (ms)':<25} | {np.mean(prefills):<12.2f} | {np.percentile(prefills, 50):<12.2f} | {np.percentile(prefills, 95):<12.2f} | {np.percentile(prefills, 99):<12.2f}")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="LLM Inference Benchmark Log Analyzer")
    parser.add_argument(
        "--log_path", 
        type=str, 
        default="./logs/64/live_cpu_16nodes.txt", 
        help="Path to the inference log file"
    )
    args = parser.parse_args()

    print(f"[INFO] Profiling target: {args.log_path}")
    data = parse_log_file(args.log_path)
    print_metrics_summary(data)


if __name__ == '__main__':
    main()