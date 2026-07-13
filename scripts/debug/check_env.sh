#!/bin/bash
# ============================================================
# Layer 2: 环境变量/软件配置排查
# 预期输出：HCCL相关变量正确设置，Python能导入torch_npu
# ============================================================

echo "===== 2.1 CANN环境变量 ====="
echo "ASCEND_HOME_PATH: $ASCEND_HOME_PATH"
echo "ASCEND_TOOLKIT_HOME: $ASCEND_TOOLKIT_HOME"
echo "ASCEND_OPP_PATH: $ASCEND_OPP_PATH"
echo "LD_LIBRARY_PATH contains ascend: $(echo $LD_LIBRARY_PATH | grep -o 'Ascend[^:]*' | head -5 | tr '\n' ', ')"

echo ""
echo "===== 2.2 HCCL关键环境变量 ====="
echo "HCCL_INTRA_ROCE_ENABLE: ${HCCL_INTRA_ROCE_ENABLE:-<not set>}"
echo "HCCL_OP_EXPANSION_MODE: ${HCCL_OP_EXPANSION_MODE:-<not set>}"
echo "HCCL_WHITELIST_DISABLE: ${HCCL_WHITELIST_DISABLE:-<not set>}"
echo "HCCL_CONNECT_TIMEOUT: ${HCCL_CONNECT_TIMEOUT:-<not set>}"
echo "HCCL_EXEC_TIMEOUT: ${HCCL_EXEC_TIMEOUT:-<not set>}"

echo ""
echo "===== 2.3 Python环境检查 ====="
which python
python --version

echo ""
echo "===== 2.4 关键Python包版本 ====="
python -c "
import torch; print(f'torch: {torch.__version__}')
import torch_npu; print(f'torch_npu: {torch_npu.__version__}')
print(f'npu available: {torch.npu.is_available()}')
print(f'npu device count: {torch.npu.device_count()}')
" 2>&1

echo ""
echo "===== 2.5 HCCL库文件检查 ====="
# 检查HCCL库是否存在
find /usr/local/Ascend -name "libhccl*" -type f 2>/dev/null | head -5
find /usr/local/Ascend -name "libascendcl*" -type f 2>/dev/null | head -5

echo ""
echo "===== 2.6 测试单卡torch_npu基础功能 ====="
python -c "
import torch
import torch_npu
x = torch.tensor([1.0, 2.0]).npu()
print(f'Single NPU tensor: {x}')
print(f'Device: {x.device}')
" 2>&1
