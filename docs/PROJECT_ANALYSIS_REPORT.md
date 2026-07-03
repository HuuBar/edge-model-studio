# Edge Model Studio — 项目完整分析报告

## 一、项目概述

Edge Model Studio 是一个**边缘端大模型全生命周期开发平台**，覆盖从数据处理到端侧部署的完整链路：

```
数据蒸馏 → 预训练 → SFT微调 → RL强化学习 → 剪枝 → 量化 → 投机解码 → 端侧部署
```

**代码统计**：118 个文件，涵盖 17 个功能模块 + 4 个脚本流水线 + 测试套件

---

## 二、训练链路完整性分析

### 2.1 整体链路评估

| 阶段 | 状态 | 入口脚本 | 可独立运行 | 缺失项 |
|------|------|----------|-----------|--------|
| **数据生成** | ✅ 完整 | `create_exercise_dataset.py` | ✅ 是 | 需 vLLM 服务提供 API |
| **数据蒸馏** | ✅ 完整 | `data_distillation.py` | ✅ 是 | 需大模型 API |
| **数据预处理** | ✅ 完整 | `data_processor/` 14个脚本 | ✅ 是 | 部分需 jieba/opencc |
| **预训练** | ⚠️ 可用 | `pretrain_run.py` | ✅ 是 | 需自建 GPTLike 模型 |
| **SFT 微调** | ✅ 完整 | `sft_exercise_multi.py` | ✅ 是 | **推荐首选入口** |
| **RL (PPO)** | ⚠️ 可用 | `change_value_head.py` | ✅ 是 | 需 trl 库 |
| **RL (GRPO)** | ❌ 缺失 | `rl/open-r1/` | ❌ 否 | open-r1 源码未上传 |
| **剪枝** | ✅ 完整 | `pruning.py` | ✅ 是 | 需 torch-pruning |
| **量化** | ✅ 完整 | `quantization.py` | ✅ 是 | 需 bitsandbytes |
| **推理 (vLLM)** | ⚠️ 可用 | `exercise_summary_inference.py` | ✅ 是 | 需 vLLM 服务 |
| **推理 (EAGLE3)** | ⚠️ 可用 | `cnets_static_tree.py` | ❌ 否 | 需 configs/utils_c/choices |
| **推理 (FRSPEC)** | ⚠️ 可用 | `frspec/fr.py` | ❌ 否 | 需预训练 tokenizer 频率 |
| **Token回收** | ⚠️ 可用 | `tk_start.py` | ❌ 否 | 需 token_recycling 模块 |
| **评测** | ⚠️ 可用 | `evaluate_reports.py` | ✅ 是 | 需评测 API |
| **记忆模块** | ⚠️ 可用 | `memory/main.py` | ❌ 否 | 需 mem0 + langgraph |
| **中间训练** | ⚠️ 可用 | `midtrain_run.py` | ✅ 是 | 需 pretrain_model 模块 |
| **性能测试** | ✅ 完整 | `performence_test/` | ✅ 是 | 需 MOLE 模型（可选） |

### 2.2 可端到端跑通的最短链路

**推荐链路（最小可行流程）**：
```bash
# Step 1: SFT 微调（最核心、最独立）
python src/edge_model_studio/finetune/sft_exercise_multi.py \
    --model_path Qwen/Qwen3-0.6B \
    --data_path data/exercise_train.jsonl \
    --output_dir output/sft_model

# Step 2: 添加 Value Head（为 RL 做准备）
python src/edge_model_studio/rl/PPO/change_value_head.py \
    --source output/sft_model \
    --target output/sft_valuehead

# Step 3: 量化（端侧部署）
python src/edge_model_studio/quantization/quantization.py 
# (需修改脚本中的硬编码路径)

# Step 4: 推理（调用 vLLM 服务）
python src/edge_model_studio/inference/exercise_inference/exercise_summary_inference.py \
    --model_name qwen3_0.6b_sft \
    --api_url http://localhost:8088/v1/completions
```

---

## 三、核心依赖清单

### 3.1 必须安装（核心框架）

```bash
pip install torch transformers datasets accelerate tqdm huggingface-hub safetensors numpy requests
```

| 包名 | 版本建议 | 用途 | 文件数 |
|------|---------|------|--------|
| torch | >=2.0 | 深度学习框架 | 47 |
| transformers | >=4.40 | HuggingFace 模型库 | 41 |
| datasets | >=2.14 | 数据集处理 | 13 |
| accelerate | >=0.25 | 分布式训练 | 多处 |
| numpy | >=1.24 | 数值计算 | 16 |
| tqdm | >=4.65 | 进度条 | 14 |
| requests | >=2.28 | HTTP 请求 | 9 |
| safetensors | >=0.4 | 安全序列化 | 1 |

### 3.2 SFT 阶段专用

```bash
pip install swanlab  # 实验日志追踪（5个文件使用）
```

> **注意**：`sft_exercise_multi.py` 中有 `swanlab.init()`，如果不需要日志追踪，可以注释掉相关代码使其成为可选依赖。

### 3.3 剪枝阶段专用

```bash
pip install torch-pruning  # 依赖图结构化剪枝
```

### 3.4 量化阶段专用

```bash
pip install bitsandbytes  # INT8/INT4 量化
```

### 3.5 RL 阶段专用

```bash
pip install trl  # Transformers Reinforcement Learning (PPO/GRPO)
```

### 3.6 推理阶段专用

```bash
pip install vllm  # 高性能推理引擎（API 服务）
```

### 3.7 评测阶段专用

```bash
pip install nltk rouge-score bert-score openpyxl
```

### 3.8 记忆模块专用

```bash
pip install mem0ai langchain langchain-community langchain-core langgraph rank-bm25 openai
```

### 3.9 数据处理专用

```bash
pip install jieba opencc-python-reimplemented
```

---

## 四、外部依赖库（需从 GitHub clone）

### 4.1 open-r1（GRPO 强化学习）

```bash
git clone https://github.com/huggingface/open-r1.git
# 然后将其 src/open_r1/ 目录链接到项目中
```

**状态**：README 已标记，源码缺失，GRPO 阶段不可运行

### 4.2 DRPruning（动态重参数化剪枝）

```bash
# 仓库地址待确认，可能为
git clone https://github.com/xxx/DRPruning.git
```

**状态**：README 已标记，项目中有 `sft_exercise_multi_drpruning.py` 引用但未包含库本身

### 4.3 emsevals（超轻量级评测框架）

```bash
# 仓库地址待确认
git clone https://github.com/xxx/emsevals.git
```

**状态**：README 已标记，评测模块 `evaluation/emsevals/` 为空目录

### 4.4 EAGLE3 配套模块

```bash
# cnets_static_tree.py 引用了以下模块，需一并获取：
# - configs.py (EConfig)
# - utils_c.py
# - choices.py
```

**状态**：核心文件 `cnets_static_tree.py` 已上传，但配套模块缺失

### 4.5 MOLE-ROM 模型定义

```python
# 以下模块在 4 个文件中被引用但缺失：
# - modeling_mole
# - modeling_mole_rep  
# - modeling_mole_rep_fp16
```

**影响范围**：`scripts/mole/mole_rom/` 和 `performence_test/` 中的 MOLE 相关脚本

---

## 五、NPU 适配注意事项

### 5.1 当前代码中的 NPU 相关配置

```python
# sft_exercise_multi.py
os.environ["NCCL_P2P_DISABLE"] = "1"  # 禁用 NCCL P2P
os.environ["NCCL_IB_DISABLE"] = "1"   # 禁用 InfiniBand
```

### 5.2 需要适配的点

| 问题 | 影响 | 解决方案 |
|------|------|----------|
| `torch.cuda` 调用 | 多处使用 CUDA API | 需替换为 `torch.npu` 或使用 `device_map` |
| `bitsandbytes` | 量化库可能不支持 NPU | 尝试使用 Ascend 原生量化工具 |
| `vllm` | 推理引擎默认 CUDA | 需使用 `vllm-ascend` 或 MindIE |
| `swanlab` | 日志追踪 | 可选，可注释掉 |
| `torch.compile()` | 预训练中使用 | NPU 支持情况需验证 |

### 5.3 NPU 安装建议

```bash
# 1. 首先确认 NPU 驱动和 CANN 已安装
npu-smi info

# 2. 安装 torch-npu（替代 torch）
pip install torch==2.1.0 torch-npu==2.1.0

# 3. 安装 transformers（通常兼容）
pip install transformers datasets accelerate

# 4. 其他依赖按需安装
pip install tqdm numpy requests safetensors
```

---

## 六、推荐的最小可运行 SFT 示例

### 6.1 步骤一：准备数据

创建 `sample_data.jsonl`（3-5 条示例数据即可跑通）：

```jsonl
{"prompt": "你是华为健康手环AI助手，擅长分析运动健康数据", "metrics": "用户今日跑步5公里，平均心率152bpm，最大心率175bpm，配速6:00/km", "summary": "#指标解析\n(1)平均心率：今日跑步平均心率152bpm，处于有氧训练区间，说明训练强度适中。\n(2)最大心率：最大心率175bpm，约为最大心率的85%，未超过安全阈值。\n(3)配速：6:00/km的配速适合基础有氧训练，建议保持。"}
{"prompt": "你是华为健康手环AI助手，擅长分析运动健康数据", "metrics": "用户今日睡眠7小时，深睡比例25%，REM睡眠20%，入睡时间23:30", "summary": "#指标解析\n(1)睡眠时长：7小时睡眠略低于建议的7-8小时，建议今晚提前入睡。\n(2)深睡比例：25%的深睡比例良好，说明身体恢复质量不错。\n(3)REM睡眠：20%的REM睡眠在正常范围内，有助于记忆巩固。"}
```

### 6.2 步骤二：运行 SFT

```bash
# 单卡运行（最简单）
python src/edge_model_studio/finetune/sft_exercise_multi.py \
    --model_path Qwen/Qwen3-0.6B \
    --data_path sample_data.jsonl \
    --output_dir ./output/sft_test
```

### 6.3 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | 必填 | 基座模型路径（Qwen3-0.6B） |
| `--data_path` | 必填 | 训练数据 JSONL 路径 |
| `--output_dir` | 必填 | 输出目录 |
| batch_size | 2 | 每卡 batch |
| gradient_accumulation_steps | 4 | 梯度累积 |
| num_train_epochs | 3 | 训练轮数 |
| learning_rate | 1e-5 | 学习率 |
| max_length | 2048 | 最大序列长度 |

### 6.4 使 swanlab 可选（无需安装）

修改 `sft_exercise_multi.py` 第 7 行和第 26 行：

```python
# 原代码
import swanlab
# 修改为
try:
    import swanlab
    HAS_SWANLAB = True
except ImportError:
    HAS_SWANLAB = False
    print("[WARN] swanlab not installed, logging disabled")

# init_env 函数中
if HAS_SWANLAB and local_rank in (-1, 0):
    swanlab.init(mode="local")

# build_training_args 中
report_to="swanlab" if HAS_SWANLAB else None,
```

---

## 七、下一步行动建议

### 优先级 P0（本周必做）

1. **在 NPU 服务器上安装核心依赖**：
   ```bash
   pip install transformers datasets accelerate tqdm numpy requests safetensors
   # NPU 需安装 torch-npu 替代 torch
   ```

2. **跑通最小 SFT 示例**：准备 5-10 条数据，验证 `sft_exercise_multi.py` 可正常训练

3. **解决 swanlab 依赖**：将其改为可选依赖或安装

### 优先级 P1（下周）

4. **安装剪枝依赖**：`pip install torch-pruning`，验证 `pruning.py` 可运行
5. **安装量化依赖**：`pip install bitsandbytes`（需确认 NPU 兼容性）
6. **搭建 vLLM 推理服务**：部署模型并提供 API

### 优先级 P2（后续）

7. **clone open-r1 仓库**：补全 GRPO 强化学习功能
8. **clone DRPruning 仓库**：补全动态重参数化剪枝
9. **clone emsevals 仓库**：补全评测框架
10. **获取 EAGLE3 配套模块**：configs/utils_c/choices

---

## 八、项目代码质量评估

### 优点
- ✅ 模块化设计清晰，17个模块职责分明
- ✅ SFT脚本可直接运行，参数配置完整
- ✅ 数据格式统一（JSONL），易于扩展
- ✅ 支持多种推理方式（vLLM/EAGLE3/FRSPEC）
- ✅ 完整的评测和日志追踪体系

### 待改进
- ⚠️ 多处硬编码绝对路径（`/data2/jwllm/...`）
- ⚠️ swanlab 等日志库未做可选处理
- ⚠️ 外部依赖库未明确列出 requirements.txt
- ⚠️ GRPO（open-r1）源码缺失
- ⚠️ EAGLE3 配套模块不完整
- ⚠️ CUDA 代码需适配 NPU

---

*报告生成时间：2026-07-03*
*分析范围：src/edge_model_studio/ + scripts/ + tests/ 共 118 个文件*
