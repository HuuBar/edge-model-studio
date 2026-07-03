import torch.distributed as dist
import os

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    print(f"Rank {rank} 成功启动，使用GPU {os.environ['CUDA_VISIBLE_DEVICES']}")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()