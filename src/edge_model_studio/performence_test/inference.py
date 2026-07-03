import os
import sys
import time
import json
import torch
import subprocess
from tqdm import tqdm
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer


def get_cpu_memory():
    try:
        pid = os.getpid()
        cmd = f"ps -p {pid} -o rss="
        result = subprocess.check_output(cmd, shell=True).decode().strip()
        return round(float(result) / 1024, 2)  # KB → MB
    except Exception:
        return 0.0


def get_gpu_memory():
    try:
        pid = os.getpid()
        cmd = f"nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | grep {pid}"
        result = subprocess.check_output(cmd, shell=True).decode().strip()
        if result:
            return round(float(result.split(',')[1]), 2)
        return 0.0
    except Exception:
        return 0.0


def read_disk_io():
    io_data = {}
    try:
        with open("/proc/self/io", "r") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    io_data["read_bytes"] = int(line.split()[1])
                elif line.startswith("write_bytes:"):
                    io_data["write_bytes"] = int(line.split()[1])
    except FileNotFoundError:
        return {}
    return io_data


def write_io_point(label, io_file):
    io_vals = read_disk_io()
    with open(io_file, "a") as f:
        f.write(f"{label}_read_bytes,{io_vals.get('read_bytes', 0)}\n")
        f.write(f"{label}_write_bytes,{io_vals.get('write_bytes', 0)}\n")


def calculate_io_delta(start_io, end_io):
    return {
        "read_bytes": (end_io.get('read_bytes', 0) - start_io.get('read_bytes', 0)) / (1024*1024),
        "write_bytes": (end_io.get('write_bytes', 0) - start_io.get('write_bytes', 0)) / (1024*1024),
    }


START_TIME_NS = float(sys.argv[1])
MODEL_PATH = sys.argv[2]
PROMPT_FILE = sys.argv[3]
IO_FILE = sys.argv[4]

TMP_DIR = os.path.dirname(IO_FILE)
TIMES_FILE = os.path.join(TMP_DIR, "mole_times.txt")
os.makedirs(TMP_DIR, exist_ok=True)

with open(TIMES_FILE, "a") as f:
    f.write(f"start_monitor,{time.time()}\n")

with open(TIMES_FILE, "a") as f:
    f.write(f"start_load,{time.time()}\n")

load_start_io = read_disk_io()
write_io_point("load_start", IO_FILE)

init_cpu = get_cpu_memory()
init_gpu = get_gpu_memory()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer_cpu = get_cpu_memory()
tokenizer_gpu = get_gpu_memory()

print("start model_load")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="cuda",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    local_files_only=True
)

loaded_cpu = get_cpu_memory()
loaded_gpu = get_gpu_memory()
print("finish model_load")
print(next(model.parameters()).dtype)

load_end_io = read_disk_io()
write_io_point("load_end", IO_FILE)
load_io_delta = calculate_io_delta(load_start_io, load_end_io)

param_size = 0
for _, param in model.named_parameters():
    param_size += param.numel() * param.element_size()
param_size_mb = param_size / (1024 * 1024)

with open(TIMES_FILE, "a") as f:
    f.write(f"end_load,{time.time()}\n")
    f.write(f"init_cpu_mem,{init_cpu}\n")
    f.write(f"init_gpu_mem,{init_gpu}\n")
    f.write(f"token_cpu_mem,{tokenizer_cpu}\n")
    f.write(f"loaded_cpu_mem,{loaded_cpu}\n")
    f.write(f"loaded_gpu_mem,{loaded_gpu}\n")
    f.write(f"model_dtype,{model.dtype}\n")
    f.write(f"load_read_mb,{load_io_delta['read_bytes']:.3f}\n")
    f.write(f"load_write_mb,{load_io_delta['write_bytes']:.3f}\n")
    f.write(f"model_weights_mem,{param_size_mb}\n")

with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompts = [item["prompt"] for item in json.load(f)]

total_tokens = 0
total_time = 0.0
ftl_list = []

with open(TIMES_FILE, "a") as f:
    f.write(f"start_infer,{time.time()}\n")

infer_start_io = read_disk_io()
write_io_point("infer_start", IO_FILE)

for idx, prompt in enumerate(tqdm(prompts[:], desc="推理进度")):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.5,
        top_p=0.9,
        use_cache=True,
        streamer=streamer,
    )

    start_time = time.time()
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    output_text = ""
    ftl_recorded = False

    for new_text in streamer:
        if not ftl_recorded:
            ftl = time.time() - start_time
            ftl_list.append(ftl)
            with open(TIMES_FILE, "a") as f:
                f.write(f"ftl_{idx},{ftl}\n")
            ftl_recorded = True
        output_text += new_text

    thread.join()
    print(f"输出: {output_text}...")

    num_tokens = len(tokenizer.encode(output_text))
    total_tokens += num_tokens
    elapsed = time.time() - start_time
    total_time += elapsed

infer_end_io = read_disk_io()
write_io_point("infer_end", IO_FILE)
infer_io_delta = calculate_io_delta(infer_start_io, infer_end_io)

avg_ftl = sum(ftl_list) / len(ftl_list) if ftl_list else 0.0

with open(TIMES_FILE, "a") as f:
    f.write(f"end_infer,{time.time()}\n")
    f.write(f"total_tokens,{total_tokens}\n")
    f.write(f"total_time,{total_time}\n")
    f.write(f"avg_speed,{total_tokens/total_time if total_time > 0 else 0}\n")
    f.write(f"avg_ftl,{avg_ftl}\n")
    f.write(f"infer_read_mb,{infer_io_delta['read_bytes']:.3f}\n")
    f.write(f"infer_write_mb,{infer_io_delta['write_bytes']:.3f}\n")

torch.cuda.empty_cache()
