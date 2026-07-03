# EMSEvals - 超轻量级大模型评测框架

EMSEvals是一个**专注核心评测功能**的轻量级框架，提供完整的评测、打分、日志保存等核心逻辑。


| 功能 | 新API | 说明 |
|------|-------|------|
| 配置类 | `EvalConfig` | 评测配置（原TaskConfig） |
| 执行评测 | `execute_eval()` | 运行评测（原run_task） |
| 评测阶段 | `FULL/INFERENCE/SCORING` | 完整/推理/评审 |

## **支持的16个评测基准

| 类别 | 评测集 | 说明 |
|------|--------|------|
| 指令遵循 | IFEval | 指令遵循评测 |
| 中文能力 | C-Eval | 中文综合能力评测 |
| 中文能力 | CMMLU | 中文多任务语言理解 |
| 推理能力 | BBH | Big Bench Hard |
| 推理能力 | HellaSwag | 常识推理 |
| 数学能力 | GSM8K | 小学数学应用题 |
| 数学能力 | AIME | AIME数学竞赛 |
| 科学能力 | super_gpqa | 超级GPQA科学问答 |
| 语言理解 | Winogrande | 指代消歧 |
| 语言理解 | GLUE | 通用语言理解 |
| 信息抽取 | Universal-NER | 通用命名实体识别 |
| 文本生成 | CNN/DailyMail | 文本摘要 |
| 领域评测 | HealthBench | 健康领域（judge） |
| 领域评测 | HammerBench | Hammer评测 |
| 生成评测 | PersonaLens | 个性化评测（judge） |
| 生成评测 | WritingBench | 写作评测（judge） |

## 快速开始

### 安装

```bash
cd emsevals
pip install -e .
```

### 命令行评测

```bash
# 基础评测
emsevals eval \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --datasets gsm8k ceval \
  --limit 10

# 或使用简短命令
ems-eval run \
  --model your-model \
  --datasets gsm8k \
  --limit 10
```

### Python API（新API）

```python
from emsevals import EvalConfig, execute_eval

# 创建配置
config = EvalConfig(
    model='Qwen/Qwen2.5-0.5B-Instruct',
    datasets=['gsm8k', 'ceval', 'ifeval'],
    limit=10
)

# 执行评测
execute_eval(task_cfg=config)
```

##  评测流程控制

EMSEvals支持三个评测阶段：

### 完整流程（FULL）
```bash
emsevals eval --model your-model --datasets gsm8k --stage full
```

### 仅推理（INFERENCE）
只运行模型推理，生成预测结果：
```bash
emsevals eval --model your-model --datasets gsm8k --stage inference
```

### 仅评审（SCORING）
基于已有的推理结果进行评审：
```bash
emsevals eval \
  --model your-model \
  --datasets gsm8k \
  --stage scoring \
  --use-cache outputs/20241013_120000
```

## 高级配置

### 使用新API

```python
from emsevals import EvalConfig, execute_eval

# 完整配置示例
config = EvalConfig(
    model='your-model',
    model_args={
        'revision': 'master',
        'precision': 'torch.float16',
        'device_map': 'auto'
    },
    datasets=['gsm8k', 'ifeval'],
    generation_config={
        'do_sample': True,
        'temperature': 0.6,
        'max_new_tokens': 512
    },
    stage='full',  # full/inference/scoring
    limit=100
)

result = execute_eval(config)
```

### API服务评测

```bash
emsevals eval \
  --model qwen2.5 \
  --eval-type service \
  --api-url http://127.0.0.1:8801/v1 \
  --api-key EMPTY \
  --datasets gsm8k ifeval \
  --limit 10
```

## 输出结构

评测结果保存在`outputs/`目录下：

```
outputs/
└── 20241013_120000/          # 运行时间戳
    ├── configs/              # 配置文件
    ├── predictions/          # 模型预测结果
    ├── reviews/              # 评审结果
    ├── reports/              # 评测报告（JSON）
    └── logs/                 # 日志文件
```

## 常用示例

### 评测指令遵循（IFEval）
```bash
emsevals eval \
  --model Qwen/Qwen2.5-7B-Instruct \
  --datasets ifeval \
  --limit 100
```

### 综合评测
```bash
emsevals eval \
  --model your-model \
  --datasets ifeval ceval cmmlu gsm8k bbh \
  --limit 100
```

### Judge评测（PersonaLens）
```bash
emsevals eval \
  --model your-model \
  --datasets personalens \
  --dataset-args '{
    "personalens": {
      "local_path": "/path/to/data",
      "subset_list": ["single_domain"]
    }
  }' \
  --judge-model-args '{
    "api_url": "http://judge-api:8080/v1",
    "model_id": "Qwen3-32B"
  }'
```

## 项目结构

```
emsevals/
├── emsevals/                # 主包
│   ├── backend/            # 后端基类
│   ├── benchmarks/         # 16个评测基准
│   ├── cli/                # 命令行接口
│   ├── evaluator/          # 评测器
│   ├── metrics/            # 评测指标
│   ├── models/             # 模型加载
│   ├── report/             # 报告生成
│   └── utils/              # 工具函数
├── requirements/           # 依赖
└── setup.py               # 安装配置
```

### 重构映射表

| 原API | 新API | 说明 |
|-------|-------|------|
| `TaskConfig` | `EvalConfig` | 评测配置类 |
| `run_task()` | `execute_eval()` | 执行评测 |
| `DataAdapter` | `DatasetHandler` | 数据处理器 |
| `Benchmark` | `EvalTask` | 评测任务 |
| `EvalStage.ALL` | `EvalStage.FULL` | 完整评测 |
| `EvalStage.INFER` | `EvalStage.INFERENCE` | 推理阶段 |
| `EvalStage.REVIEW` | `EvalStage.SCORING` | 评审阶段 |

