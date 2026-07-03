# NPU 服务器操作指南

## 一、环境概览（已就绪）

服务器已有完整的 Docker 镜像，**无需安装任何依赖**：

| 组件 | 版本 | 状态 |
|------|------|------|
| Python | 3.11.13 | ✅ 已安装 |
| CANN | 7.8 | ✅ 已安装 |
| torch | 2.5.1 | ✅ 已安装 |
| torch-npu | 2.5.1.post1.dev20250619 | ✅ 已安装 |
| vllm | 0.9.2 | ✅ 已安装 |
| vllm-ascend | 0.9.2rc1 | ✅ 已安装 |
| transformers | 4.52.4 | ✅ 已安装 |
| trl | 0.19.1 | ✅ 已安装 |
| accelerate | 1.9.0 | ✅ 已安装 |
| datasets | 4.0.0 | ✅ 已安装 |

### 模型已下载

```
/home/edgeModelWorkspace/origin_model/
├── Qwen3-0.6B/          # 主模型（SFT用）
├── Qwen3-0.6B-Base/     # Base 版本
└── Qwen3-32B/           # 大模型
```

---

## 二、启动 Docker 容器

### 2.1 启动容器

```bash
docker run -d -ti \
  --entrypoint /bin/bash \
  --restart=always \
  --privileged \
  --net=host \
  --name=edge_model_studio \
  -h docker_name_host \
  -e NPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e HCCL_WHITELIST_DISABLE=1 \
  -e VLLM_USE_ASCEND=True \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -v /home/edgeModelWorkspace/:/home/edgeModelWorkspace/ \
  -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ \
  -v /usr/local/Ascend/firmware/:/usr/local/Ascend/firmware/ \
  -v /home/:/home/ \
  edge_model_image:vllm-ascend-v6
```

### 2.2 进入容器

```bash
docker exec -it edge_model_studio /bin/bash
```

### 2.3 验证 NPU 环境

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -c "import torch_npu; import torch; print(f'NPU count: {torch.npu.device_count()}')"
```

---

## 三、Clone 代码到服务器

```bash
cd /home/edgeModelWorkspace/workspace/
git clone https://github.com/HuuBar/edge-model-studio.git
cd edge-model-studio
```

---

## 四、跑通最小 SFT 示例（穿刺验证）

### 4.1 准备数据

创建 `sample_data.jsonl`（5 条数据即可跑通）：

```bash
cat > /home/edgeModelWorkspace/datasets/sample_sft_data.jsonl << 'JSONL'
{"prompt": "你是华为健康手环AI助手，擅长分析运动健康数据", "metrics": "用户今日跑步5公里，平均心率152bpm，最大心率175bpm，配速6:00/km", "summary": "#指标解析\n(1)平均心率：今日跑步平均心率152bpm，处于有氧训练区间，说明训练强度适中。\n(2)最大心率：最大心率175bpm，约为最大心率的85%，未超过安全阈值。\n(3)配速：6:00/km的配速适合基础有氧训练，建议保持。"}
{"prompt": "你是华为健康手环AI助手，擅长分析运动健康数据", "metrics": "用户今日睡眠7小时，深睡比例25%，REM睡眠20%，入睡时间23:30", "summary": "#指标解析\n(1)睡眠时长：7小时睡眠略低于建议的7-8小时，建议今晚提前入睡。\n(2)深睡比例：25%的深睡比例良好，说明身体恢复质量不错。\n(3)REM睡眠：20%的REM睡眠在正常范围内，有助于记忆巩固。"}
{"prompt": "你是华为健康手环AI助手，擅长分析运动健康数据", "metrics": "用户今日游泳1500米，平均配速2:30/100m，消耗热量380千卡，心率区间在有氧范围", "summary": "#指标解析\n(1)游泳距离：1500米属于中等强度训练量，有助于提升心肺耐力。\n(2)平均配速：2:30/100m的配速适中，建议逐步提升至2:15/100m以增强训练效果。\n(3)热量消耗：380千卡的消耗符合中等强度游泳的预期，有助于体重管理。"}
{"prompt": "你是华为健康手环AI助手，擅长分析运动健康数据", "metrics": "用户今日骑行20公里，平均速度22km/h，爬升150米，平均心率138bpm", "summary": "#指标解析\n(1)骑行距离：20公里属于良好训练量，有助于下肢力量提升。\n(2)平均速度：22km/h的巡航速度适合有氧耐力训练，可逐步提升至25km/h。\n(3)爬升高度：150米爬升增加了训练强度，有助于腿部肌肉力量发展。"}
{"prompt": "你是华为健康手环AI助手，擅长分析运动健康数据", "metrics": "用户今日步行10000步，爬楼梯15层，活跃时间45分钟，静息心率65bpm", "summary": "#指标解析\n(1)步数：10000步达到日常活动目标，有助于维持基础代谢水平。\n(2)爬楼：15层楼梯爬升有助于下肢力量和心肺功能锻炼。\n(3)静息心率：65bpm的静息心率良好，表明心肺功能处于健康状态。"}
JSONL
```

### 4.2 单卡 SFT 运行（最简单）

```bash
cd /home/edgeModelWorkspace/workspace/edge-model-studio

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

CUDA_VISIBLE_DEVICES=0 \
python src/edge_model_studio/finetune/sft_exercise_multi.py \
  --model_path /home/edgeModelWorkspace/origin_model/Qwen3-0.6B/ \
  --data_path /home/edgeModelWorkspace/datasets/sample_sft_data.jsonl \
  --output_dir /home/edgeModelWorkspace/process_model/sft_test_output/
```

### 4.3 多卡 DDP 运行（推荐）

```bash
cd /home/edgeModelWorkspace/workspace/edge-model-studio

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node 4 \
  src/edge_model_studio/finetune/sft_exercise_multi.py \
  --model_path /home/edgeModelWorkspace/origin_model/Qwen3-0.6B/ \
  --data_path /home/edgeModelWorkspace/datasets/sample_sft_data.jsonl \
  --output_dir /home/edgeModelWorkspace/process_model/sft_test_output/
```

### 4.4 关键参数说明

| 参数 | 脚本中的默认值 | 说明 |
|------|---------------|------|
| batch_size | 2 | 每卡 batch size |
| gradient_accumulation_steps | 4 | 梯度累积步数 |
| num_train_epochs | 3 | 训练轮数 |
| learning_rate | 1e-5 | 学习率 |
| max_length | 2048 | 最大序列长度 |
| bf16 | True | 使用 BF16 混合精度 |

---

## 五、启动 vLLM 推理服务

### 5.1 启动服务

```bash
export NCCL_P2P_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

CUDA_VISIBLE_DEVICES=0 \
python -m vllm.entrypoints.openai.api_server \
  --model /home/edgeModelWorkspace/process_model/sft_test_output/ \
  --served-model-name qwen3-0.6b-sft \
  --tensor-parallel-size 1 \
  --dtype float16 \
  --max-model-len 4096 \
  --block-size 32 \
  --swap-space 16 \
  --host 0.0.0.0 \
  --port 8088 \
  --max-num-seqs 32 \
  --seed 42
```

### 5.2 测试推理

```bash
curl http://localhost:8088/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b-sft",
    "prompt": "你是一个跑步专家，用户问：今日跑步5公里，平均心率152bpm，跑得怎么样？",
    "max_tokens": 256,
    "temperature": 0.6
  }'
```

---

## 六、SwanLab 日志查看（可选）

### 6.1 启动本地 Dashboard

```bash
swanlab watch --logdir /path/to/swanlog --host 0.0.0.0 --port 8089
```

### 6.2 本地访问

通过 MobaXterm tunneling 映射端口后访问：
```
http://localhost:8089/
```

---

## 七、完整训练链路（后续步骤）

```
Step 1: SFT 微调（上面已完成）
  ↓
Step 2: 添加 Value Head（PPO 准备）
  python src/edge_model_studio/rl/PPO/change_value_head.py \
    --source /home/edgeModelWorkspace/process_model/sft_test_output/ \
    --target /home/edgeModelWorkspace/process_model/sft_valuehead/
  ↓
Step 3: GRPO 强化学习
  # 使用 open-r1
  cd src/edge_model_studio/rl/open-r1/
  # 配置 recipes 和 reward 函数后执行
  ↓
Step 4: 剪枝
  # DepGraph 剪枝
  python src/edge_model_studio/pruning/dep_graph_pruning/pruning.py \
    --model /home/edgeModelWorkspace/process_model/rl_model/ \
    --pruning_ratio 0.2 \
    --save_model /home/edgeModelWorkspace/process_model/pruned_model/
  ↓
Step 5: 量化
  python src/edge_model_studio/quantization/quantization.py
  # 需修改脚本中的硬编码路径
  ↓
Step 6: 推理部署
  # 启动 vLLM（见 5.1）
```

---

## 八、常见问题

### 8.1 端口占用

```bash
# 查找占用 29500 端口的进程（DDP 默认端口）
lsof -i:29500
# 关闭对应 PID
kill -9 <PID>
```

### 8.2 关闭 vLLM

```bash
ps aux | grep vllm
kill -9 <PID>
```

### 8.3 容器时区

```bash
ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
echo "Asia/Shanghai" > /etc/timezone
```

---

*文档版本：2026-07-03*
*适用环境：华为昇腾 NPU + edge_model_image:vllm-ascend-v6*
