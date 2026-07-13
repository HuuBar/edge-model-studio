# NPU多卡训练排查指南

## 三层排查模型

```
┌─────────────────────────────────────┐
│  Layer 3: 框架层 (TRL/Accelerate)   │  ← 框架API调用问题
│  - SFTTrainer/GRPOTrainer配置       │     (如ddp_find_unused_parameters)
│  - accelerate config                │
├─────────────────────────────────────┤
│  Layer 2: 分布式层 (torchrun/HCCL)  │  ← 环境变量配置问题
│  - torchrun启动参数                 │     (如HCCL_INTRA_ROCE_ENABLE)
│  - HCCL通信域初始化                 │     (如MASTER_PORT冲突)
│  - DDP后端配置                      │
├─────────────────────────────────────┤
│  Layer 1: 硬件/驱动层 (CANN/NPU)    │  ← 物理连接问题
│  - NPU设备状态                      │     (如Xid错误、卡间链路断开)
│  - 驱动/CANN版本                    │
│  - 卡间互联拓扑                     │
└─────────────────────────────────────┘
```

---

## 问题1: 物理连接有问题

### 典型症状
- `npu-smi info` 看不到卡，或某些卡显示 **Not Active**
- HCCL初始化报错：**HCCL_E_OPEN_FILE** 或 **timeout**
- AllReduce结果不一致，某些rank卡住
- `npu-smi topo -m` 显示某些卡之间无连接

### 诊断命令
```bash
# 1. 查看NPU设备状态
npu-smi info
# 预期: 8张卡全部Normal/Active

# 2. 查看卡间拓扑
npu-smi topo -m
# 预期: 卡间互联矩阵完整

# 3. 查看健康状态
for i in $(seq 0 7); do npu-smi info -t health -i $i; done
# 预期: 全部Healthy，无Xid错误
```

### 常见原因
| 问题 | 现象 | 解决 |
|------|------|------|
| 驱动未加载 | `npu-smi` 报命令不存在 | 安装/加载Ascend驱动 |
| 某些卡故障 | `npu-smi info` 显示故障 | 联系运维更换NPU |
| 卡间HCCS链路断开 | `topo` 显示某些卡无连接 | 联系运维检查物理连接 |
| 散热问题 | 温度过高导致降频/掉卡 | 检查机房散热 |

---

## 问题2: 环境变量配置有问题

### 典型症状
- `import torch_npu` 报错
- HCCL初始化报错：**HCCL_E_ENV** 相关错误
- 单卡能跑，多卡hang住或报错
- `torchrun` 启动后子进程崩溃

### 关键环境变量检查清单

```bash
# CANN基础环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 必须设置的变量
echo $ASCEND_HOME_PATH           # /usr/local/Ascend/ascend-toolkit/latest
echo $ASCEND_TOOLKIT_HOME        # 同上
echo $ASCEND_OPP_PATH            # .../opp
echo $LD_LIBRARY_PATH            # 包含Ascend/lib64

# HCCL相关
echo $HCCL_INTRA_ROCE_ENABLE     # 单机多卡建议设为0 (用HCCS而非RoCE)
export HCCL_INTRA_ROCE_ENABLE=0

echo $MASTER_ADDR                # torchrun自动设置，或手动设为127.0.0.1
echo $MASTER_PORT                # 确保端口未被占用
```

### 常见原因
| 问题 | 现象 | 解决 |
|------|------|------|
| CANN未source | `ASCEND_HOME_PATH`为空 | `source set_env.sh` |
| HCCL_INTRA_ROCE_ENABLE错误 | 单机多卡走RoCE超时 | 单机设为0，多机设为1 |
| MASTER_PORT冲突 | 多用户同服务器冲突 | 改用随机端口：`--master_port $((29500 + RANDOM % 1000))` |
| LD_LIBRARY_PATH缺失 | 找不到libhccl.so | 检查CANN source是否完整 |

---

## 问题3: 框架里面对HCCL库API的调用有问题

### 典型症状
- HCCL测试通过，但transformers/trl训练报错
- 报错含 **find_unused_parameters**、**bucket_cap_mb** 等DDP参数
- 报错含 **accelerate**、**deepspeed** 配置相关
- 训练启动后立刻崩溃，或hang在某个rank

### 框架层常见问题

#### 3.1 transformers Trainer DDP配置
```python
# 错误示范
training_args = TrainingArguments(
    # ...
    ddp_backend="hccl",  # 必须显式指定！
)

# 如果使用torchrun启动，transformers会自动处理大部分配置
# 但要确保：
# 1. bf16=True 需要NPU支持bf16 (910B支持)
# 2. 不要在args里传 device_map="auto" (和DDP冲突)
# 3. 模型.to(device) 而不是 device_map
```

#### 3.2 accelerate配置
```yaml
# 如果使用accelerate，需要hccl兼容的配置
distributed_type: MULTI_NPU
# 或者
distributed_type: DEEPSPEED  # deepspeed会自动处理
```

#### 3.3 TRL SFTTrainer
```python
# TRL的SFTTrainer在NPU上可能需要调整
from trl import SFTTrainer

# 常见问题：SFTTrainer内部调用gather_for_metrics时NPU不支持
# 解决：设置 args.dispatch_batches=False
```

---

## 快速诊断流程

```bash
# 进入项目目录
cd edge-model-studio

# 1. 一键运行完整诊断
bash scripts/debug/run_full_diagnosis.sh

# 2. 或者分步手动执行：

# Step 1: 硬件
bash scripts/debug/check_hardware.sh

# Step 2: 环境变量
bash scripts/debug/check_env.sh

# Step 3: HCCL通信 (单机多卡自动fork)
python scripts/debug/test_hccl_allreduce.py

# Step 4: torchrun分布式
torchrun --nproc_per_node=8 scripts/debug/test_torchrun_npu.py

# Step 5: SFT训练 (可选)
torchrun --nproc_per_node=8 scripts/debug/test_sft_multi_npu.py
```

---

## 预期输出对照

### test_hccl_allreduce.py 正常输出
```
检测到 8 张NPU卡，启动AllReduce测试...

[Rank 0] Before AllReduce: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] on npu:0
[Rank 1] Before AllReduce: [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1] on npu:1
...
[Rank 0] After  AllReduce: [2.8, 10.8, 18.8, 26.8, 34.8, 42.8, 50.8, 58.8]
...
[Rank 0] ✅ AllReduce结果正确! max_diff=0.000000
[Rank 7] ✅ AllReduce结果正确! max_diff=0.000000

✅ AllReduce测试全部通过!
```

### test_torchrun_npu.py 正常输出
```
[Rank 0/8] LocalRank=0, NPU=0, Host=your-server
[Rank 1/8] LocalRank=1, NPU=1, Host=your-server
...
✅ Barrier同步成功!
✅ AllReduce验证通过! sum=36.0, expected=36
✅ 简单DDP前向/反向成功! loss=xxxx

🎉 全部测试通过! torchrun+hccl+DDP工作正常
```

---

## 典型错误速查

| 错误信息 | 所属层次 | 解决方案 |
|----------|----------|----------|
| `npu-smi: command not found` | Layer 1 | 驱动未安装或不在PATH |
| `ASCEND_HOME_PATH not set` | Layer 2 | source CANN set_env.sh |
| `libhccl.so: cannot open` | Layer 2 | LD_LIBRARY_PATH缺失 |
| `HCCL_E_OPEN_FILE` | Layer 2/3 | HCCN配置错误或权限问题 |
| `Connection refused` / timeout | Layer 2 | MASTER_PORT冲突/防火墙 |
| `RuntimeError: Distributed backend not available` | Layer 3 | torch_npu未正确安装 |
| `Expected all tensors to be on the same NPU` | Layer 3 | device_map与DDP冲突 |
| `find_unused_parameters`相关 | Layer 3 | model.gradient_checkpointing_enable() |
| `NaN in loss` | Layer 3 | 尝试fp32代替bf16 |
