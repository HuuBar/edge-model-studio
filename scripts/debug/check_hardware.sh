#!/bin/bash
# ============================================================
# Layer 1: 硬件/物理连接排查
# 预期输出：8张NPU卡，状态Normal，无Xid错误
# ============================================================

echo "===== 1.1 NPU设备状态 ====="
npu-smi info

echo ""
echo "===== 1.2 驱动版本 ====="
cat /usr/local/Ascend/driver/version.info 2>/dev/null || cat /var/log/ascend_seclog/ascend_install.log 2>/dev/null | grep "Driver" | tail -5

echo ""
echo "===== 1.3 CANN版本 ====="
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null | head -5

echo ""
echo "===== 1.4 NPU健康检查（Xid错误） ====="
# Xid错误表示硬件故障
for i in $(seq 0 7); do
    npu-smi info -t health -i $i 2>/dev/null | grep -E "Health|Xid"
done

echo ""
echo "===== 1.5 查看NPU拓扑（确认卡间连接） ====="
npu-smi topo -m 2>/dev/null || echo "npu-smi topo not available"

echo ""
echo "===== 1.6 HCCN配置（多卡通信网络配置） ====="
cat /etc/hccn.conf 2>/dev/null || echo "No /etc/hccn.conf found"
