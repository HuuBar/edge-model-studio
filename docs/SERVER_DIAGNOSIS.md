# 服务器诊断结果与下一步

## 当前状态

| 组件 | 状态 |
|------|------|
| Docker | ✅ 18.09.0 |
| NPU 910B4 x8 | ✅ 2张在用(vLLM), 6张空闲 |
| CANN 8.5.0 | ✅ 已安装 |
| edge_model_image | ❌ 不存在 |
| Python包(torch等) | ❌ 未安装(宿主机) |

## 关键发现：已有运行中的容器

```
CONTAINER ID   IMAGE          NAMES
ef96754d226e   831f3c77abea   cann_container          ← 可能是CANN环境
7b4a312f960f   d6a0e7f25bf0   huangyiming             ← 可能含Python环境
c5e42345c6fa   000a773cabd2   torch_npu_builder        ← 名字暗示torch-npu
36e6ff32b77d   d6a0e7f25bf0   triton_lzp              ← triton推理框架
```

## 请执行以下检查

### 1. 检查 cann_container（最可能是CANN+torch-npu环境）

```bash
docker exec -it cann_container bash -c "
  python3 -c 'import torch; import torch_npu; print(\"torch:\", torch.__version__); print(\"torch_npu:\", torch_npu.__version__); print(\"NPU count:\", torch.npu.device_count())' 2>/dev/null || echo 'torch not installed in container'
"
```

### 2. 检查 huangyiming 容器（有人在做LLM）

```bash
docker exec -it huangyiming bash -c "
  python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'no torch'
  pip list 2>/dev/null | grep -iE 'torch|transformers|vllm|trl|datasets' || echo 'pip not available'
"
```

### 3. 检查 torch_npu_builder（名字很直接）

```bash
docker exec -it torch_npu_builder bash -c "
  python3 -c 'import torch; import torch_npu; print(torch.__version__, torch_npu.__version__)' 2>/dev/null || echo 'not ready'
"
```

### 4. 查看所有容器的镜像信息

```bash
docker images --format "{{.ID}} {{.Repository}}:{{.Tag}}" | sort | uniq
```

### 5. 检查宿主机上是否有 conda 或其他 Python 环境

```bash
# 检查 conda
conda --version 2>/dev/null && conda env list

# 检查多个 python 路径
which -a python python3
ls /usr/bin/python* 2>/dev/null

# 检查是否有虚拟环境
find /home -name "activate" -path "*/bin/activate" 2>/dev/null | head -10
```

### 6. 检查 vLLM 服务怎么跑的

```bash
# 查看 vLLM 进程详情
ps aux | grep vllm

# 查看 NPU 0 上的进程
npu-smi info -t processes
```

## 根据检查结果判断

### 情况 A：某个容器已有完整环境
→ 直接用那个容器，或者基于它 commit 新镜像

### 情况 B：只有 CANN 基础环境，无 Python ML 包
→ 需要手动安装 torch-npu + transformers + vllm（文档已有步骤）

### 情况 C：什么都没有
→ 从 CANN 官方镜像从头构建

## 快速尝试：进入 cann_container

```bash
docker exec -it cann_container bash
# 然后里面试
python3 -c "import torch; import torch_npu; print(torch.npu.device_count())"
```

如果这个容器里 torch 能用，那大概率已经有基础环境了。
