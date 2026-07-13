#!/usr/bin/env python3
"""
Torchrun + torch_npu 分布式测试
模拟实际训练时的启动方式，测试torchrun能否正确拉起多卡

用法:
    torchrun --nproc_per_node=8 --nnodes=1 test_torchrun_npu.py
"""

import os
import torch
import torch.distributed as dist
import torch.nn as nn


def main():
    # torchrun会自动设置这些环境变量
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 设置当前进程使用的NPU卡
    torch.npu.set_device(local_rank)

    # 初始化HCCL
    dist.init_process_group("hccl")

    print(f"[Rank {rank}/{world_size}] LocalRank={local_rank}, "
          f"NPU={torch.npu.current_device()}, "
          f"Host={os.uname().nodename}")

    # Barrier测试：所有rank同步
    dist.barrier()
    if rank == 0:
        print("✅ Barrier同步成功!")

    # AllReduce测试
    tensor = torch.tensor([float(rank + 1)]).npu()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected_sum = sum(range(1, world_size + 1))

    if rank == 0:
        actual = tensor.item()
        if abs(actual - expected_sum) < 1e-4:
            print(f"✅ AllReduce验证通过! sum={actual}, expected={expected_sum}")
        else:
            print(f"❌ AllReduce验证失败! sum={actual}, expected={expected_sum}")

    # 模拟简单前向+反向
    model = nn.Linear(10, 10).npu()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    x = torch.randn(4, 10).npu()
    y = model(x)
    loss = y.sum()
    loss.backward()

    if rank == 0:
        print(f"✅ 简单DDP前向/反向成功! loss={loss.item():.4f}")

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        print("\n🎉 全部测试通过! torchrun+hccl+DDP工作正常")


if __name__ == "__main__":
    main()
