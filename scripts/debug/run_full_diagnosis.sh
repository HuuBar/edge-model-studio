#!/bin/bash
# ============================================================
# NPU多卡训练问题 - 完整诊断流程
# 按顺序执行，每一步都有预期输出
# 在哪一步出错，问题就在哪一层
# ============================================================

set -e  # 遇到错误立即退出，方便定位

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "  NPU多卡训练问题诊断流程"
echo "========================================"
echo ""

# ============================================================
# Step 0: CANN环境
# ============================================================
echo -e "${YELLOW}>>> Step 0: 检查并加载CANN环境${NC}"
if [ -z "$ASCEND_HOME_PATH" ]; then
    echo "CANN环境未加载，尝试自动source..."
    # 常见路径
    for path in \
        "/usr/local/Ascend/ascend-toolkit/set_env.sh" \
        "/usr/local/Ascend/cann/latest/set_env.sh" \
        "/usr/local/Ascend/cann/set_env.sh" \
        "/home/$(whoami)/Ascend/ascend-toolkit/set_env.sh"
    do
        if [ -f "$path" ]; then
            echo "找到: $path"
            source "$path"
            echo "✅ CANN环境已加载"
            break
        fi
    done
else
    echo "✅ CANN环境已加载: $ASCEND_HOME_PATH"
fi

if [ -z "$ASCEND_HOME_PATH" ]; then
    echo -e "${RED}❌ CANN环境未找到！请手动指定set_env.sh路径${NC}"
    echo "   常见位置: /usr/local/Ascend/ascend-toolkit/set_env.sh"
    exit 1
fi

# ============================================================
# Step 1: 硬件层
# ============================================================
echo ""
echo -e "${YELLOW}>>> Step 1: 硬件/驱动层检查${NC}"
echo "--- 1.1 NPU设备 ---"
npu-smi info || { echo -e "${RED}❌ npu-smi不可用！驱动可能未安装${NC}"; exit 1; }

echo ""
echo "--- 1.2 驱动版本 ---"
cat /usr/local/Ascend/driver/version.info 2>/dev/null || echo "⚠️ 未找到driver/version.info"

echo ""
echo "--- 1.3 NPU拓扑 ---"
npu-smi topo -m 2>/dev/null || echo "⚠️ npu-smi topo不可用"

echo ""
echo -e "${GREEN}✅ Step 1 通过 - 硬件层正常${NC}"

# ============================================================
# Step 2: 环境变量层
# ============================================================
echo ""
echo -e "${YELLOW}>>> Step 2: 环境变量/软件配置检查${NC}"

bash "$SCRIPT_DIR/check_env.sh"

echo ""
echo -e "${GREEN}✅ Step 2 通过 - 环境变量配置正常${NC}"

# ============================================================
# Step 3: HCCL通信层 (核心!)
# ============================================================
echo ""
echo -e "${YELLOW}>>> Step 3: HCCL集合通信测试${NC}"
echo "--- 3.1 Python HCCL AllReduce ---"
python "$SCRIPT_DIR/test_hccl_allreduce.py" || {
    echo -e "${RED}❌ HCCL AllReduce失败！问题在通信层${NC}"
    echo "   可能原因:"
    echo "   - HCCN配置错误 (/etc/hccn.conf)"
    echo "   - 防火墙阻塞HCCL端口"
    echo "   - 卡间互联(HCCS/RoCE)物理连接问题"
    exit 1
}

echo ""
echo -e "${GREEN}✅ Step 3 通过 - HCCL通信正常${NC}"

# ============================================================
# Step 4: torchrun分布式层
# ============================================================
echo ""
echo -e "${YELLOW}>>> Step 4: torchrun + DDP测试${NC}"
NPUS=$(python -c "import torch; print(torch.npu.device_count())")
echo "使用 $NPUS 张NPU卡测试..."

torchrun --nproc_per_node=$NPUS --nnodes=1 "$SCRIPT_DIR/test_torchrun_npu.py" || {
    echo -e "${RED}❌ torchrun分布式测试失败！${NC}"
    echo "   可能原因:"
    echo "   - torch_npu与pytorch版本不匹配"
    echo "   - torchrun参数错误"
    exit 1
}

echo ""
echo -e "${GREEN}✅ Step 4 通过 - torchrun分布式正常${NC}"

# ============================================================
# Step 5: SFT训练层 (可选，需要下载模型)
# ============================================================
echo ""
echo -e "${YELLOW}>>> Step 5: SFT多卡训练测试${NC}"
echo "⚠️  这一步需要下载约1GB的模型，可能需要几分钟..."
echo "    模型: Qwen/Qwen2.5-0.5B"
read -p "是否执行? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    torchrun --nproc_per_node=$NPUS --nnodes=1 "$SCRIPT_DIR/test_sft_multi_npu.py" || {
        echo -e "${RED}❌ SFT多卡训练测试失败！${NC}"
        echo "   可能原因:"
        echo "   - transformers Trainer DDP配置问题"
        echo "   - 内存不足 (OOM)"
        echo "   - bf16不支持"
        exit 1
    }
    echo -e "${GREEN}✅ Step 5 通过 - SFT多卡训练正常${NC}"
else
    echo "⏭️  跳过Step 5 (手动执行: bash scripts/debug/test_sft_multi_npu.sh)"
fi

# ============================================================
# 总结
# ============================================================
echo ""
echo "========================================"
echo -e "  ${GREEN}🎉 全部诊断通过！${NC}"
echo "  硬件 → 环境变量 → HCCL → torchrun → SFT"
echo "  全部链路正常，问题在更高层框架（如TRL/Accelerate配置）"
echo "========================================"
