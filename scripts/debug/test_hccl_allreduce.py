#!/usr/bin/env python3
"""
HCCL AllReduce 测试 - Python版
模拟C++样例的功能，测试多卡集合通信

用法:
    python test_hccl_allreduce.py          # 单机单进程测试（自动fork子进程）
    torchrun --nproc_per_node=8 test_hccl_allreduce.py  # torchrun方式

排查目标：确认HCCL通信域能正常初始化，AllReduce结果正确
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run_allreduce(rank, world_size):
    """每个rank执行的AllReduce测试"""
    # 初始化HCCL通信域
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    # 使用hccl后端初始化
    dist.init_process_group("hccl", rank=rank, world_size=world_size)

    # 每个rank创建不同的输入数据（模拟C++样例的0~7初始化）
    device = torch.device(f"npu:{rank}")
    data = torch.arange(world_size, dtype=torch.float32).to(device)
    # rank i 的第j个元素初始值为 j + i*0.1（不同rank有不同值）
    tensor = data + rank * 0.1

    print(f"[Rank {rank}] Before AllReduce: {tensor.cpu().tolist()} on {device}")

    # 执行AllReduce SUM
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"[Rank {rank}] After  AllReduce: {tensor.cpu().tolist()}")

    # 验证：每个位置应该是所有rank对应位置的和
    expected = sum(torch.arange(world_size, dtype=torch.float32) + r * 0.1 for r in range(world_size))
    diff = (tensor.cpu() - expected).abs().max().item()

    if diff < 1e-4:
        print(f"[Rank {rank}] ✅ AllReduce结果正确! max_diff={diff:.6f}")
    else:
        print(f"[Rank {rank}] ❌ AllReduce结果错误! expected={expected.tolist()}, diff={diff:.6f}")

    dist.destroy_process_group()


def test_single_process():
    """单进程模式：自动fork子进程测试"""
    if not torch.npu.is_available():
        print("❌ NPU不可用!")
        sys.exit(1)

    world_size = torch.npu.device_count()
    print(f"检测到 {world_size} 张NPU卡，启动AllReduce测试...\n")

    mp.spawn(run_allreduce, args=(world_size,), nprocs=world_size, join=True)
    print("\n✅ AllReduce测试全部通过!")


def test_torchrun():
    """torchrun模式：由外部torchrun启动"""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    run_allreduce(rank, world_size)


if __name__ == "__main__":
    # 检测是torchrun启动还是直接python启动
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        test_torchrun()
    else:
        test_single_process()
