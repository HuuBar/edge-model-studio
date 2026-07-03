# MindSpeed-LLM 与 Edge Model Studio 关系分析报告

## 一、MindSpeed-LLM 是什么？

**MindSpeed-LLM** 是华为昇腾官方的大语言模型分布式训练框架：

| 特性 | 说明 |
|------|------|
| **定位** | 昇腾生态的端到端大模型训练套件 |
| **核心能力** | 分布式预训练、指令微调、RLHF、推理、评估 |
| **技术栈** | Megatron-LM + MindSpeed(昇腾加速库) + PyTorch |
| **支持模型** | Qwen3、DeepSeek、GLM、MiniMax 等主流模型 |
| **硬件** | 昇腾 910B/A2/A3 |

### 关键入口脚本

```
MindSpeed-LLM/
├── pretrain_gpt.py           # 预训练
├── posttrain_gpt.py          # 后训练（SFT/DPO）
├── rlhf_gpt.py               # RLHF 强化学习
├── train_fsdp2.py            # FSDP2 训练
├── inference.py              # 推理
├── evaluation.py             # 评估
├── preprocess_data.py        # 数据预处理
├── convert_ckpt.py           # 权重转换
└── requirements.txt          # 依赖
```

---

## 二、与 Edge Model Studio 的关系

### 结论：两者是互补关系，不是替代关系

| 维度 | Edge Model Studio | MindSpeed-LLM |
|------|-------------------|---------------|
| **定位** | 端侧模型全生命周期平台 | 分布式大模型训练框架 |
| **核心关注** | 剪枝→量化→端侧部署 | 预训练→微调→RLHF |
| **训练方式** | 单机/小规模（transformers.Trainer） | 分布式大规模（Megatron-LM） |
| **剪枝** | DRPruning + DepGraph | 不支持 |
| **量化** | bitsandbytes INT8/INT4 | 不支持 |
| **投机解码** | EAGLE3 + FRSPEC | 不支持 |
| **推理部署** | vllm-ascend | vllm-ascend |
| **数据蒸馏** | 完整流程 | 需手动实现 |
| **预训练** | 简单实现 | 完整分布式预训练 |
| **SFT微调** | transformers.Trainer | 分布式SFT |
| **RLHF** | open-r1 + verl | RLHF |

### 代码层面：没有直接引用

Edge Model Studio 的代码中**没有 import mindspeed_llm**，两者是独立的项目。

**但可以在 MindSpeed-LLM 容器内运行 edge-model-studio 的代码**，利用 MindSpeed-LLM 提供的基础环境（torch-npu、CANN 等）。

---

## 三、推荐的 Docker 镜像策略

### 你的服务器环境

| 组件 | 版本 |
|------|------|
| NPU | 910B4 x8 |
| CANN | 8.5.0 |
| Docker | 18.09.0 |

### 可用镜像对比

| 镜像 | 更新日期 | CANN版本 | 适用场景 | 推荐度 |
|------|---------|---------|---------|--------|
| **mindspeed-llm** | 2026/05/12 | 8.5.2 | 预训练/微调/推理/评估 | 5星 |
| **verl_pt27_25rc4** | 2026/06/06 | - | RL + vLLM 强化学习 | 4星 |
| **mindspeed_rl_pt25_25rc3** | 2025/12/05 | - | MindSpeed RL | 3星 |
| **ascend-pytorch** | - | - | 基础 PyTorch 环境 | 2星 |

### 推荐方案

#### 方案 A：mindspeed-llm 镜像（推荐）

**优点：**
- 预装 torch-npu + transformers + vllm-ascend
- CANN 8.5.2（服务器是 8.5.0，通常向下兼容）
- 包含完整的 MindSpeed-LLM 训练框架
- 可以在容器内同时运行 edge-model-studio 的代码

**拉取命令：**
```bash
# 从 ascendhub 拉取
docker pull ascendhub/mindspeed-llm:26.0.0-910b-ubuntu22.04-py3.11-aarch64
```

**启动命令：**
```bash
docker run -d -ti \
  --entrypoint /bin/bash \
  --restart=always \
  --privileged \
  --net=host \
  --name=edge_model_studio \
  -e NPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e HCCL_WHITELIST_DISABLE=1 \
  -e VLLM_USE_ASCEND=True \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /home/:/home/ \
  ascendhub/mindspeed-llm:26.0.0-910b-ubuntu22.04-py3.11-aarch64
```

**进入后安装额外依赖：**
```bash
docker exec -it edge_model_studio bash

# 安装 edge-model-studio 需要的额外包
pip install swanlab torch-pruning bitsandbytes trl \
    openpyxl pandas jieba opencc-python-reimplemented \
    nltk rouge-score bert-score

# clone edge-model-studio
cd /home/edgeModelWorkspace/workspace/
git clone https://github.com/HuuBar/edge-model-studio.git
```

---

## 四、服务器操作步骤

```bash
# 1. 确认 CANN 版本兼容性
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg

# 2. 拉取 mindspeed-llm 镜像
docker pull ascendhub/mindspeed-llm:26.0.0-910b-ubuntu22.04-py3.11-aarch64

# 3. 启动容器
docker run -d -ti \
  --entrypoint /bin/bash \
  --restart=always \
  --privileged \
  --net=host \
  --name=edge_model_studio \
  -e NPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e HCCL_WHITELIST_DISABLE=1 \
  -e VLLM_USE_ASCEND=True \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /home/:/home/ \
  ascendhub/mindspeed-llm:26.0.0-910b-ubuntu22.04-py3.11-aarch64

# 4. 进入容器
docker exec -it edge_model_studio bash

# 5. 验证环境
python3 -c "import torch; import torch_npu; print(f'NPU: {torch.npu.device_count()}')"
python3 -c "import transformers; print(f'transformers: {transformers.__version__}')"
python3 -c "import vllm; print(f'vllm: {vllm.__version__}')"

# 6. 安装额外依赖并 clone 代码
pip install swanlab torch-pruning bitsandbytes trl
cd /home/edgeModelWorkspace/workspace/
git clone https://github.com/HuuBar/edge-model-studio.git

# 7. 准备数据并运行 SFT
# (使用 edge-model-studio 的 sft_exercise_multi.py)
```

---

*报告生成时间：2026-07-03*
