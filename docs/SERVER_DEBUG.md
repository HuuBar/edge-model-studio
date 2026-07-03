# 服务器问题排查 - Docker 存储路径问题

## 发现的问题

容器 `docker ps` 显示 `Up`，但 `docker exec` 报错：
```
Error response from daemon: stat /home/dockerlib/overlay2/xxx: no such file or directory
```

说明 Docker 的 `data-root` 被修改为 `/home/dockerlib/`（非默认的 `/var/lib/docker`），但当前这个路径可能：
- 是 NFS/网络挂载，暂时不可访问
- 被移动或删除了
- 权限问题

## 下一步排查命令

### 1. 检查 Docker 配置

```bash
# 查看 Docker 数据目录配置
cat /etc/docker/daemon.json 2>/dev/null
grep "data-root" /etc/docker/daemon.json 2>/dev/null

# 检查 /home/dockerlib 是否存在
ls -la /home/dockerlib/ 2>/dev/null | head -5
ls -la /home/dockerlib/overlay2/ 2>/dev/null | head -5

# 检查磁盘挂载
df -h | grep dockerlib
df -h | grep -E "home|var"
```

### 2. vLLM 到底怎么跑的？

```bash
# 查看 vLLM 进程的完整命令行
ps aux | grep vllm | grep -v grep

# 查看 vLLM 进程的工作目录
ls -la /proc/$(pgrep -f "vllm" | head -1)/cwd 2>/dev/null

# 查看这个进程属于哪个用户
ps aux | grep VLLMWorker | head -5
```

### 3. 宿主机 Python 环境深度检查

```bash
# 宿主机 Python 版本和包
/usr/bin/python3 --version
/usr/bin/python3 -m pip list 2>/dev/null | grep -iE "torch|transform|vllm|trl|data|accel" || echo "pip not available"

# 是否有其他 Python 路径
find / -name "python3" -type f 2>/dev/null | head -10
find / -name "pip3" -type f 2>/dev/null | head -10

# 是否有 virtualenv / venv
find / -name "activate" -path "*/bin/activate" 2>/dev/null | head -10

# 是否有 conda
find / -name "conda" -type f 2>/dev/null | head -5
```

### 4. 检查环境变量（可能被设置了 Python 路径）

```bash
env | grep -iE "python|path|conda|venv"
cat ~/.bashrc | grep -iE "python|conda|venv|activate" | head -20
cat ~/.bash_profile 2>/dev/null | grep -iE "python|conda|venv" | head -10
```

### 5. 直接找到模型文件

```bash
# 搜索 Qwen 模型
find / -name "config.json" -path "*Qwen*" 2>/dev/null | head -5
find / -name "model.safetensors" 2>/dev/null | head -5
find / -name "*.safetensors" 2>/dev/null | head -10

# 搜索 huggingface 缓存
ls -la ~/.cache/huggingface/hub/ 2>/dev/null | head -10
find / -name "models--Qwen*" -type d 2>/dev/null | head -5
```

### 6. 网络挂载检查

```bash
# 查看所有挂载点
mount | grep -iE "home|docker|nfs"
cat /etc/fstab | grep -v "^#" | grep -v "^$"

# 检查 /home 是否网络挂载
df -Th | grep -E "home|nfs|ceph"
```

---

## 根据检查结果的判断

### 如果 /home/dockerlib 不存在或不可访问

说明容器文件系统实际存储在其他地方（可能是共享存储故障）。**vLLM 进程是之前启动的，现在文件系统失联了**。

**解决方案**：
1. 重启 Docker 服务看能否恢复
2. 或者直接在宿主机上安装环境（绕过 Docker）

### 如果 vLLM 进程实际在宿主机上

检查 `ps aux` 的输出，看进程的启动用户和命令行。

### 如果宿主机有 pip 但没有 ML 包

直接在宿主机上安装：
```bash
/usr/bin/python3 -m pip install torch==2.5.1 transformers datasets accelerate --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 如果宿主机连 pip 都没有

先安装 pip：
```bash
/usr/bin/python3 -m ensurepip --upgrade
/usr/bin/python3 -m pip install --upgrade pip
```

---

## 快速决策流程

```
/home/dockerlib 存在？
├── YES → Docker 数据目录正常，检查权限
│         └── ls -la /home/dockerlib/overlay2/
├── NO  → 共享存储/网络挂载问题
          └── 方案 A: 修复挂载
          └── 方案 B: 直接在宿主机安装（绕过 Docker）
                  
vLLM 在宿主机还是容器？
├── 宿主机 → 说明有人直接在宿主机装了环境
│         └── 找到他的 Python 环境（conda/venv/全局）
└── 容器   → 容器文件系统可能损坏了
          └── 需要重建容器
```
