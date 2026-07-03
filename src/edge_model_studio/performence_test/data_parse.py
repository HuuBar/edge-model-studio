import os
import sys


def bytes_to_mb(b):
    return b / (1024.0 * 1024.0)


def read_time_events(times_file):
    te = {}
    try:
        with open(times_file, "r") as f:
            for line in f:
                if "," in line:
                    k, v = line.strip().split(",", 1)
                    try:
                        te[k] = float(v)
                    except ValueError:
                        te[k] = v
    except FileNotFoundError:
        raise SystemExit(f"错误：时间文件 {times_file} 不存在")
    return te


def print_memory_report(te):
    if all(k in te for k in ["init_cpu_mem", "loaded_cpu_mem", "init_gpu_mem", "loaded_gpu_mem"]):
        cpu_delta = te["loaded_cpu_mem"] - te["init_cpu_mem"]
        gpu_delta = te["loaded_gpu_mem"] - te["init_gpu_mem"]
        tokenizer_delta = te["token_cpu_mem"] - te["init_cpu_mem"]
        
        print("\n内存占用报告:")
        print(f"{'类型':<10} | {'加载前(MB)':>12} | {'加载后(MB)':>12} | {'增量(MB)':>12}")
        print("-" * 55)
        print(f"{'CPU':<10} | {te['init_cpu_mem']:>12.2f} | {te['loaded_cpu_mem']:>12.2f} | {cpu_delta:>12.2f}")
        print(f"{'GPU总':<10} | {te['init_gpu_mem']:>12.2f} | {te['loaded_gpu_mem']:>12.2f} | {gpu_delta:>12.2f}")
        
        if "model_weights_mem" in te:
            print(f"{'GPU权重':<10} | {'-':>12} | {te['model_weights_mem']:>12.2f} | {'-':>12}")
        if "token_cpu_mem" in te:
            print(f"{'tokenizer占用':<10} | {'-':>12} | {tokenizer_delta:>12.2f} | {'-':>12}")


def print_io_report(te):
    if all(k in te for k in ["load_read_mb", "load_write_mb", "start_load", "end_load"]):
        load_time = te["end_load"] - te["start_load"]
        load_read_speed = te["load_read_mb"] / load_time if load_time > 0 else 0
        load_write_speed = te["load_write_mb"] / load_time if load_time > 0 else 0
        
        print("\n磁盘IO报告 - 加载阶段:")
        print(f"{'读取总量(MB)':<15}: {te['load_read_mb']:.3f}")
        print(f"{'写入总量(MB)':<15}: {te['load_write_mb']:.3f}")
        print(f"{'平均读取速度(MB/s)':<15}: {load_read_speed:.2f}")
        print(f"{'平均写入速度(MB/s)':<15}: {load_write_speed:.2f}")
        print(f"{'持续时间(s)':<15}: {load_time:.3f}")

    if all(k in te for k in ["infer_read_mb", "infer_write_mb", "start_infer", "end_infer"]):
        infer_time = te["end_infer"] - te["start_infer"]
        infer_read_speed = te["infer_read_mb"] / infer_time if infer_time > 0 else 0
        infer_write_speed = te["infer_write_mb"] / infer_time if infer_time > 0 else 0
        
        print("\n磁盘IO报告 - 推理阶段:")
        print(f"{'读取总量(MB)':<15}: {te['infer_read_mb']:.3f}")
        print(f"{'写入总量(MB)':<15}: {te['infer_write_mb']:.3f}")
        print(f"{'平均读取速度(MB/s)':<15}: {infer_read_speed:.2f}")
        print(f"{'平均写入速度(MB/s)':<15}: {infer_write_speed:.2f}")
        print(f"{'持续时间(s)':<15}: {infer_time:.3f}")


def print_performance_report(te):
    if all(k in te for k in ["total_tokens", "total_time", "avg_speed"]):
        print("\n性能报告:")
        print(f"{'总Tokens':<12}: {te['total_tokens']}")
        print(f"{'总时间(s)':<12}: {te['total_time']:.3f}")
        print(f"{'速度(t/s)':<12}: {te['avg_speed']:.2f}")
    
    if "avg_ftl" in te:
        print(f"{'首Token延迟':<12}: {te['avg_ftl']:.4f}s")


if __name__ == "__main__":
    TMP_DIR = sys.argv[1] if len(sys.argv) > 1 else "./tmp"
    TIMES_FILE = os.path.join(TMP_DIR, "mole_times.txt")
    
    te = read_time_events(TIMES_FILE)
    
    print("="*50)
    print(f"{'模型性能分析报告':^50}")
    print("="*50)
    
    print_memory_report(te)
    print_io_report(te)  # 新增IO报告
    print_performance_report(te)
    
    print("\n时间统计:")
    print(f"{'模型加载':<12}: {te.get('end_load',0)-te.get('start_load',0):.3f}s")
    print(f"{'推理总时间':<12}: {te.get('end_infer',0)-te.get('start_infer',0):.3f}s")
    print("="*50)