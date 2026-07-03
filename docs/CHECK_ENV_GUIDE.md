# NPU 服务器环境检查与部署指南

## 一、检查当前服务器环境

### 1.1 基础检查（在服务器上执行）

```bash
# ===== 1. 检查 Docker 是否安装 =====
docker --version
docker info 2>/dev/null | head -20
# 如果显示版本号，说明 Docker 已安装
# 如果没安装，看第四节：手动部署

# ===== 2. 查看现有 Docker 镜像 =====
docker images
# 重点看有没有这些：
# - edge_model_image:vllm-ascend-v6
# - edge_model_image 相关
# - cann:* (昇腾官方镜像)
# - vllm-ascend 相关

# ===== 3. 查看运行中的容器 =====
docker ps

# ===== 4. 查看所有容器（含停止的）=====
docker ps -a
```

### 1.2 检查 NPU 驱动（关键！）

```bash
# ===== 5. 检查昇腾 NPU 驱动 =====
npu-smi info
# 应该显示 NPU 卡信息，如：
# +-------------------+-----------------+-----------------+
# | NPU     Name      | Health          | Power(W)  Temp(C)
# +-------------------+-----------------+-----------------+

# ===== 6. 检查 CANN 是否安装 =====
ls /usr/local/Ascend/
# 应该看到：driver/ firmware/ ascend-toolkit/ 等目录

# ===== 7. 检查 CANN 版本 =====
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null
# 或者
cat /usr/local/Ascend/ascend-toolkit/latest/compiler/version.info 2>/dev/null

# ===== 8. 检查 torch-npu 是否可用 =====
python3 -c "import torch_npu; print('torch_npu OK:', torch_npu.__version__)" 2>/dev/null
# 或者
python3 -c "import torch; import torch_npu; print(f'NPU count: {torch.npu.device_count()}')" 2>/dev/null
```

### 1.3 检查 Python 环境和已装包

```bash
# ===== 9. Python 版本 =====
python3 --version

# ===== 10. 检查已安装的关键包 =====
python3 -c "
import importlib
packages = [
    ('torch', 'torch'), ('torch-npu', 'torch_npu'),
    ('transformers', 'transformers'), ('datasets', 'datasets'),
    ('vllm', 'vllm'), ('trl', 'trl'), ('accelerate', 'accelerate'),
    ('swanlab', 'swanlab'), ('torch_pruning', 'torch_pruning'),
    ('bitsandbytes', 'bitsandbytes'), ('huggingface_hub', 'huggingface_hub'),
]
for name, mod in packages:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, '__version__', 'unknown')
        print(f'  {name:25s} : {ver}')
    except ImportError:
        print(f'  {name:25s} : NOT INSTALLED')
"
```

---

## 二、根据检查结果判断情况

### 情况 A：已有 edge_model_image:vllm-ascend-v6 镜像

```bash
# 直接启动！
docker run -d -ti \
  --entrypoint /bin/bash \
  --restart=always --privileged --net=host \
  --name=edge_model_studio \
  -e NPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e HCCL_WHITELIST_DISABLE=1 \
  -e VLLM_USE_ASCEND=True \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -v /home/edgeModelWorkspace/:/home/edgeModelWorkspace/ \
  -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ \
  -v /usr/local/Ascend/firmware/:/usr/local/Ascend/firmware/ \
  edge_model_image:vllm-ascend-v6
```

### 情况 B：有 CANN 官方镜像但无魔改镜像

检查有没有以下官方镜像：
```bash
docker images | grep -i "cann\|ascend"
# 可能看到：
# cann:dev-2.0.RC2.B050-800I-A2-py311-ubuntu24.04-aarch64
```

如果有，需要**手动构建**魔改环境（见第四节）。

### 情况 C：没有任何相关镜像，但有 NPU 驱动和 CANN

基于官方镜像或手动安装，从头构建环境（见第四节）。

### 情况 D：连 NPU 驱动都没有

```bash
npu-smi info
# 报错：command not found
```

→ 需要先安装**昇腾驱动 + CANN**，这是前置条件。

---

## 三、Docker 镜像迁移方法（如果原服务器可用）

如果原服务器可以访问，有以下几种迁移方式：

### 方法 1：save/load 导出导入（推荐）

在原服务器上：
```bash
# 导出镜像为 tar 文件
docker save edge_model_image:vllm-ascend-v6 | gzip > edge_model_image_v6.tar.gz

# 传到新服务器（scp/ftp/任何方式）
scp edge_model_image_v6.tar.gz user@new_server:/path/
```

在新服务器上：
```bash
# 导入镜像
gunzip -c edge_model_image_v6.tar.gz | docker load

# 验证
docker images | grep edge_model
```

### 方法 2：导出 Dockerfile 重建

在原服务器上导出容器变更：
```bash
# 从运行中的容器创建新镜像
docker commit edge_model_zeyu edge_model_image:export_v1

# 然后 save
docker save edge_model_image:export_v1 | gzip > edge_model_export.tar.gz
```

### 方法 3：Registry 推送拉取（如果有私有仓库）

```bash
# 原服务器推送
docker tag edge_model_image:vllm-ascend-v6 your_registry/edge_model:v6
docker push your_registry/edge_model:v6

# 新服务器拉取
docker pull your_registry/edge_model:v6
```

---

## 四、手动部署环境（无现成镜像时）

### 4.1 前置条件检查

```bash
# 确认这些已安装，否则先安装：
npu-smi info          # NPU 驱动
ls /usr/local/Ascend/ # CANN 工具包
```

### 4.2 方案 1：基于 CANN 官方镜像构建

```bash
# 1. 拉取昇腾官方镜像
docker pull ascendai/cann:dev-2.0.RC2.B050-800I-A2-py311-ubuntu24.04-aarch64

# 2. 启动基础容器
docker run -d -ti \
  --entrypoint /bin/bash \
  --privileged --net=host \
  --name=edge_model_build \
  -e NPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/ \
  -v /usr/local/Ascend/firmware/:/usr/local/Ascend/firmware/ \
  ascendai/cann:dev-2.0.RC2.B050-800I-A2-py311-ubuntu24.04-aarch64

docker exec -it edge_model_build /bin/bash
```

在容器内安装依赖：
```bash
# 3. 设置 pip
export PIP_NO_PARALLEL=1
export PIP_PROGRESS_BAR=off

# 4. 安装基础工具
apt update && apt install -y libnuma-dev build-essential cmake ninja-build wget git curl jq net-tools

# 5. 安装 PyTorch + torch-npu
pip install torch==2.5.1 torchvision==0.20.1
# 下载 torch-npu whl 后安装：
pip install torch_npu-2.5.1.post1.dev20250619-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# 6. 安装 vllm + vllm-ascend
pip install vllm==0.9.2
pip install vllm-ascend==0.9.2rc1

# 7. 安装 transformers 生态
pip install transformers==4.52.4 datasets accelerate trl==0.19.1 \
    huggingface_hub safetensors tqdm numpy requests

# 8. 验证
python3 -c "import torch_npu; import torch; import vllm; import transformers; print('All OK!')"

# 9. 保存为新镜像
docker commit edge_model_build edge_model_image:vllm-ascend-v6
```

### 4.3 方案 2：直接在宿主机安装（无 Docker）

```bash
# 1. 创建虚拟环境
python3 -m venv /home/yourname/venv_edge
source /home/yourname/venv_edge/bin/activate

# 2. 安装基础包
pip install --upgrade pip setuptools wheel

# 3. 安装 PyTorch 生态
pip install torch==2.5.1 torchvision==0.20.1
# 安装 torch-npu（需下载对应版本 whl）
pip install torch_npu-*.whl

# 4. 安装 vllm
pip install vllm==0.9.2 --no-deps  # 避免依赖冲突
pip install vllm-ascend==0.9.2rc1

# 5. 安装 transformers 生态
pip install transformers==4.52.4 datasets==4.0.0 \
    accelerate==1.9.0 trl==0.19.1 \
    huggingface_hub safetensors tqdm numpy requests openpyxl pandas

# 6. 安装可选依赖（按需）
pip install swanlab bitsandbytes torch-pruning \
    nltk rouge-score bert-score jieba opencc-python-reimplemented

# 7. 验证环境
python3 << 'EOF'
import torch
import torch_npu
import transformers
import vllm
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
print(f"torch: {torch.__version__}")
print(f"torch_npu: {torch_npu.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"vllm: {vllm.__version__}")
print(f"NPU count: {torch.npu.device_count()}")
print("Environment OK!")
EOF
```

---

## 五、一键检查脚本

把下面保存为 `check_env.sh`，在服务器上直接执行：

```bash
#!/bin/bash

echo "=========================================="
echo "  Edge Model Studio - 环境检查脚本"
echo "=========================================="

# 1. Docker
echo ""
echo "[1/8] Docker 检查..."
if command -v docker &>/dev/null; then
    echo "  Docker: $(docker --version)"
    echo "  镜像列表:"
    docker images --format "  {{.Repository}}:{{.Tag}}" | grep -i "edge\|cann\|ascend\|vllm" || echo "  (无相关镜像)"
else
    echo "  Docker: 未安装"
fi

# 2. NPU 驱动
echo ""
echo "[2/8] NPU 驱动检查..."
if command -v npu-smi &>/dev/null; then
    npu-smi info 2>/dev/null | head -10
else
    echo "  npu-smi: 未安装"
fi

# 3. CANN
echo ""
echo "[3/8] CANN 检查..."
if [ -d "/usr/local/Ascend" ]; then
    echo "  CANN 目录存在: $(ls /usr/local/Ascend/)"
else
    echo "  CANN: 未安装"
fi

# 4. Python
echo ""
echo "[4/8] Python: $(python3 --version 2>/dev/null || echo '未安装')"

# 5. torch-npu
echo ""
echo "[5/8] torch-npu 检查..."
python3 -c "import torch_npu; print(f'  torch_npu: {torch_npu.__version__}')" 2>/dev/null || echo "  torch_npu: 未安装"

# 6. 关键包
echo ""
echo "[6/8] Python 包检查..."
python3 << 'PYEOF'
import importlib
packages = [
    'torch', 'torch_npu', 'transformers', 'datasets',
    'vllm', 'trl', 'accelerate', 'huggingface_hub'
]
for mod in packages:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, '__version__', 'unknown')
        print(f"  {mod:25s} : {ver}")
    except ImportError:
        print(f"  {mod:25s} : NOT INSTALLED")
PYEOF

# 7. NPU 可用性
echo ""
echo "[7/8] NPU 可用性..."
python3 -c "import torch, torch_npu; print(f'  NPU 数量: {torch.npu.device_count()}')" 2>/dev/null || echo "  无法检测 NPU"

# 8. 模型目录
echo ""
echo "[8/8] 模型检查..."
for path in /home/edgeModelWorkspace/origin_model/Qwen3-0.6B \
            /data2/jwllm/models_origin/qwen3_0.6b \
            /data/models/Qwen3-0.6B; do
    if [ -d "$path" ]; then
        echo "  找到模型: $path"
    fi
done

echo ""
echo "=========================================="
echo "  检查完成"
echo "=========================================="
```

执行方式：
```bash
chmod +x check_env.sh
./check_env.sh
```

---

## 六、常见问题

### Q: 找不到 docker 命令？
```bash
# 安装 Docker
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER
# 退出重新登录
docker --version
```

### Q: 如何找到模型的实际路径？
```bash
# 全局搜索
find / -name "config.json" -path "*/Qwen*" 2>/dev/null | head -5
find / -name "model.safetensors" 2>/dev/null | head -5

# 如果模型还没下载
# 在 HuggingFace 下载 Qwen3-0.6B
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-0.6B --local-dir /path/to/save
```

### Q: torch-npu 安装失败？
- 确认 CANN 版本和 torch-npu 版本匹配
- 从华为镜像站手动下载 whl: https://mirrors.huaweicloud.com/ascend/repos/pypi/torch-npu/
- 确认 Python 版本和 whl 的 cp 标记一致（cp311 = Python 3.11）

---

*文档版本：2026-07-03*
