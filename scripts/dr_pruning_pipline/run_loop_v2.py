import os
import subprocess

# === 全局参数 ===
BASE_MODEL = "/data2/jwllm/models/Qwen3-0.6B-Base"
DATA_PATH = "/data2/jwllm/datasets/train_pairs.json"
SFT_SCRIPT = "./sft_v2.py"
PRUNE_SCRIPT = "./pruning.py"
NUM_ROUNDS = 4
OUTPUT_BASE_DIR = "/data2/jwllm/models/model_sft"

# 设置多卡 GPU 可见性
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# 获取基础模型名
base_model_name = os.path.basename(BASE_MODEL.rstrip("/"))

# 初始模型路径
model_path = BASE_MODEL

# === 第 0 轮：全量模型的首次微调 ===
sft_0_dir = os.path.join(OUTPUT_BASE_DIR, f"{base_model_name}-sft-round-0")
os.makedirs(sft_0_dir, exist_ok=True)

print(f"\n=== 第 0 轮：全量模型首次微调（→ {sft_0_dir}） ===\n")
subprocess.run([
    "torchrun", "--nproc_per_node=4", "--master_port=29500", SFT_SCRIPT,
    "--model_path", model_path,
    "--data_path", DATA_PATH,
    "--output_dir", sft_0_dir
], check=True)

# 把首次微调后的模型作为第 1 轮剪枝的起点
model_path = sft_0_dir

for i in range(1, NUM_ROUNDS + 1):
    pruned_dir = os.path.join(OUTPUT_BASE_DIR, f"{base_model_name}-pruned-round-{i}")
    sft_dir = os.path.join(OUTPUT_BASE_DIR, f"{base_model_name}-sft-round-{i}")

    os.makedirs(pruned_dir, exist_ok=True)
    os.makedirs(sft_dir, exist_ok=True)

    print(f"\n=== 第 {i} 轮：剪枝（→ {pruned_dir}） ===\n")
    subprocess.run([
        "python", PRUNE_SCRIPT,
        "--model", model_path,
        "--pruning_ratio", "0.2",
        "--max_seq_len", "1024",
        "--save_model", pruned_dir
    ], check=True)

    print(f"\n=== 第 {i} 轮：微调（→ {sft_dir}） ===\n")
    subprocess.run([
        "torchrun", "--nproc_per_node=4", "--master_port=29500", SFT_SCRIPT,
        "--model_path", pruned_dir,
        "--data_path", DATA_PATH,
        "--output_dir", sft_dir
    ], check=True)

    # 更新下轮输入模型路径
    model_path = sft_dir

# # 把首次微调后的模型作为第 1 轮剪枝的起点
# model_path = sft_0_dir

# # === 剪枝-微调循环 ===
# for i in range(1, NUM_ROUNDS + 1):
#     pruned_dir = os.path.join(OUTPUT_BASE_DIR, f"{base_model_name}-pruned-round-{i}")
#     sft_dir = os.path.join(OUTPUT_BASE_DIR, f"{base_model_name}-sft-round-{i}")

#     os.makedirs(pruned_dir, exist_ok=True)
#     os.makedirs(sft_dir, exist_ok=True)

#     # 剪枝（单卡运行）
#     print(f"\n=== 第 {i} 轮：剪枝（→ {pruned_dir}） ===")
#     subprocess.run([
#         "python", PRUNE_SCRIPT,
#         "--model", model_path,
#         "--pruning_ratio", "0.2",
#         "--max_seq_len", "1024",
#         "--save_model", pruned_dir
#     ], check=True)

#     # 微调（多卡分布式）
#     print(f"\n=== 第 {i} 轮：微调（→ {sft_dir}） ===")
#     run_distributed_training(
#         script_path=SFT_SCRIPT,
#         args=[
#             "--model_path", pruned_dir,
#             "--data_path", DATA_PATH,
#             "--output_dir", sft_dir
#         ],
#         gpus=4
#     )

#     # 更新下轮输入模型路径
#     model_path = sft_dir

# print("\n=== 所有轮次完成 ===")


# import os
# import subprocess
# import time
# import socket
# import logging

# # === 日志配置 ===
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[logging.StreamHandler()]
# )
# logger = logging.getLogger(__name__)

# # === 强制使用前4张4090显卡（设备0-3）===
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 关键设置：仅使用4张4090
# os.environ["OMP_NUM_THREADS"] = "1"  # 防止系统过载

# # === 全局参数 ===
# BASE_MODEL = "/data2/jwllm/models/Qwen3-0.6B-Base"
# DATA_PATH = "/data2/jwllm/datasets/train_pairs.json"
# SFT_SCRIPT = "./sft_v2.py"
# PRUNE_SCRIPT = "./pruning.py"
# NUM_ROUNDS = 4
# OUTPUT_BASE_DIR = "/data2/jwllm/models/model_sft"
# BASE_MASTER_PORT = 29501  # 基础端口，实际端口会动态计算

# def find_free_port(base_port):
#     """动态寻找可用端口"""
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         s.bind(("", 0))
#         return s.getsockname()[1]

# def run_distributed_training(script_path, args, gpus=4, round_idx=0):
#     """启动分布式训练（显式绑定GPU）"""
#     master_port = find_free_port(BASE_MASTER_PORT + round_idx)
#     cmd = [
#         "torchrun",
#         f"--nproc_per_node={gpus}",
#         f"--master_port={master_port}",
#         script_path,
#         *args
#     ]
#     logger.info(f"启动命令: {' '.join(cmd)}")
#     try:
#         subprocess.run(cmd, check=True)
#     except subprocess.CalledProcessError as e:
#         logger.error(f"分布式训练失败: {e}")
#         raise

# def run_pruning(script_path, args):
#     """运行剪枝脚本（强制单卡）"""
#     env = os.environ.copy()
#     env["CUDA_VISIBLE_DEVICES"] = "0"  # 剪枝只用第一张卡
#     cmd = ["python", script_path, *args]
#     logger.info(f"剪枝命令: {' '.join(cmd)}")
#     try:
#         subprocess.run(cmd, env=env, check=True)
#     except subprocess.CalledProcessError as e:
#         logger.error(f"剪枝失败: {e}")
#         raise

# def main():
#     # 获取基础模型名
#     base_model_name = os.path.basename(BASE_MODEL.rstrip("/"))

#     # === 第 0 轮：全量模型的首次微调 ===
#     sft_0_dir = os.path.join(OUTPUT_BASE_DIR, f"{base_model_name}-sft-round-0")
#     os.makedirs(sft_0_dir, exist_ok=True)

#     logger.info(f"\n=== 第 0 轮：全量模型首次微调（→ {sft_0_dir}） ===")
#     run_distributed_training(
#         script_path=SFT_SCRIPT,
#         args=[
#             "--model_path", BASE_MODEL,
#             "--data_path", DATA_PATH,
#             "--output_dir", sft_0_dir
#         ],
#         gpus=4,
#         round_idx=0
#     )
#     model_path = sft_0_dir

#     # === 剪枝-微调循环 ===
#     for i in range(1, NUM_ROUNDS + 1):
#         pruned_dir = os.path.join(OUTPUT_BASE_DIR, f"{base_model_name}-pruned-round-{i}")
#         sft_dir = os.path.join(OUTPUT_BASE_DIR, f"{base_model_name}-sft-round-{i}")
#         os.makedirs(pruned_dir, exist_ok=True)
#         os.makedirs(sft_dir, exist_ok=True)

#         # 剪枝（单卡运行）
#         logger.info(f"\n=== 第 {i} 轮：剪枝（→ {pruned_dir}） ===")
#         run_pruning(
#             script_path=PRUNE_SCRIPT,
#             args=[
#                 "--model", model_path,
#                 "--pruning_ratio", "0.2",
#                 "--max_seq_len", "1024",
#                 "--save_model", pruned_dir
#             ]
#         )

#         # 微调（多卡分布式）
#         logger.info(f"\n=== 第 {i} 轮：微调（→ {sft_dir}） ===")
#         run_distributed_training(
#             script_path=SFT_SCRIPT,
#             args=[
#                 "--model_path", pruned_dir,
#                 "--data_path", DATA_PATH,
#                 "--output_dir", sft_dir
#             ],
#             gpus=4,
#             round_idx=i
#         )
#         model_path = sft_dir

#     logger.info("\n=== 所有轮次完成 ===")

# if __name__ == "__main__":
#     main()