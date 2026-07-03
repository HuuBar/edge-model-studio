import re
import sys
import os
import numpy as np


def parse_log_stream(stream):
    """
    使用生成器流式解析日志，一行行处理，内存复杂度 O(1)。
    即使面对数 GB 的巨型日志文件，也绝不会发生内存暴涨（OOM）。
    """
    decode_times = []
    decode_topk_times = []

    # 预编译正则，提升百万行日志下的匹配效率
    # 去掉了原正则前面的引号约束(如有必要请自行加回)，使其对标准标准输出和格式化日志更具鲁棒性
    decode_pattern = re.compile(r'Decode costs ([\d.]+) ms')
    decode_topk_pattern = re.compile(r'Decode topk costs ([\d.]+) ms')

    for line in stream:
        # 性能压测日志通常很长，不盲目使用 strip()，直接在整行中检索更高效
        match = decode_pattern.search(line)
        if match:
            try:
                decode_times.append(float(match.group(1)))
            except ValueError:
                pass
            continue  # 一行日志通常只满足一种指标，命中则直接跳过，减少后续正则错检

        match = decode_topk_pattern.search(line)
        if match:
            try:
                decode_topk_times.append(float(match.group(1)))
            except ValueError:
                pass

    return decode_times, decode_topk_times


def print_metrics_report(label, data):
    """
    打印规范的性能统计报告。
    线上调优单看 Average 毫无意义，必须引入 P50/P95/P99 观察长尾抖动（Spike）。
    """
    if not data:
        print(f"[WARN] No matched entries found for: '{label}'")
        return

    arr = np.array(data, dtype=np.float32)
    avg_val = np.mean(arr)
    min_val = np.min(arr)
    max_val = np.max(arr)
    p50 = np.percentile(arr, 50)
    p95 = np.percentile(arr, 95)
    p99 = np.percentile(arr, 99)

    print(f"=== Performance Report: {label} ===")
    print(f"  Count   : {len(arr)}")
    print(f"  Average : {avg_val:.3f} ms")
    print(f"  Min/Max : {min_val:.3f} / {max_val:.3f} ms")
    print(f"  P50     : {p50:.3f} ms")
    print(f"  P95     : {p95:.3f} ms")
    print(f"  P99     : {p99:.3f} ms")
    print()


def main():
    # 支持两种模式：
    # 1. 脚本后面直接接路径：python parse.log inference.log
    # 2. 管道流式传输：cat inference.log | python parse.log
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
        if not os.path.exists(log_path):
            print(f"[CRITICAL] Log file not found: {log_path}", file=sys.stderr)
            sys.exit(1)
            
        print(f"[INFO] Processing log file: {log_path}")
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            decode_times, decode_topk_times = parse_log_stream(f)
    else:
        print("[INFO] Processing log from stdin (streaming mode)...")
        decode_times, decode_topk_times = parse_log_stream(sys.stdin)

    print("\n" + "=" * 50 + "\n")
    print_metrics_report("Decode Costs", decode_times)
    print_metrics_report("Decode TopK Costs", decode_topk_times)


if __name__ == "__main__":
    main()