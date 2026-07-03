#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
离线 vLLM speculative decoding benchmark 的公共工具函数。

本文件主要负责四类事情：
1. 生成/加载 prompt，并统计输入 token 数；
2. 从 vLLM offline LLM.generate() 的输出中统计输出 token 数；
3. 采集 speculative decoding 相关指标，例如 draft token 数、accepted token 数；
4. 汇总 benchmark summary，计算吞吐、平均每 token 耗时、batch latency 分位数等。

注意：
- 本脚本基于离线 LLM.generate()，因此可以准确测量端到端 batch 生成耗时、
  output tokens/s、total tokens/s、avg_ms_per_output_token。
- 离线 LLM.generate() 不是 streaming 接口，因此不能准确测 TTFT、ITL、TPOT。
  这些字段在 summary 中会显式置为 None，避免产生误导。
- speculative 指标依赖当前 vLLM 版本是否暴露 llm.get_metrics()。
  如果没有暴露，spec_metrics_available 会是 False。
"""

from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def now_iso() -> str:
    # 实验时间戳。写入 summary["experiment_date"]，用于后续结果追踪和报告整理。
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_version(pkg: str) -> str:
    # 读取 Python 包版本，例如 vllm/torch/transformers。
    # 如果包不存在或元信息不可读，返回 unknown，避免 benchmark 因环境字段失败。
    try:
        return importlib.metadata.version(pkg)
    except Exception:
        return "unknown"


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    # 计算分位数，例如 batch_latency_p50_ms / p90 / p99。
    # 使用线性插值：样本较少时比简单取下标更平滑。
    if not values:
        return None
    xs = sorted(float(x) for x in values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    # 计算平均值；空数组返回 None。
    # 用于 input_len_actual_avg、output_len_actual_avg、batch_latency_avg_ms 等字段。
    vals = [float(v) for v in values if v is not None]
    return statistics.mean(vals) if vals else None


def ms(x: Optional[float]) -> Optional[float]:
    # 秒 -> 毫秒。内部计时通常使用 time.perf_counter() 得到秒，
    # summary 对用户展示时统一输出 *_ms 字段。
    return None if x is None else x * 1000.0


def chunked(xs: Sequence[Any], n: int) -> Iterable[Sequence[Any]]:
    if n <= 0:
        raise ValueError("batch size must be positive")
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def load_json_maybe(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "summary" in obj:
        return obj["summary"]
    return obj


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_csv(path: str | Path, row: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    flat = {}
    for k, v in row.items():
        if isinstance(v, (dict, list, tuple)):
            flat[k] = json.dumps(v, ensure_ascii=False)
        else:
            flat[k] = v

    file_exists = path.exists()
    if file_exists:
        # Keep the first header stable. Extra keys are ignored when appending to
        # an existing CSV created by a previous run.
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                fieldnames = next(reader)
            except StopIteration:
                fieldnames = list(flat.keys())
    else:
        fieldnames = list(flat.keys())

    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(flat)


def environment_info(collect_npu_smi: bool = False) -> Dict[str, Any]:
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "vllm_version": safe_version("vllm"),
        "torch_version": safe_version("torch"),
        "transformers_version": safe_version("transformers"),
        "cwd": os.getcwd(),
    }
    if collect_npu_smi:
        info["npu_smi_info"] = get_npu_smi_info()
    return info


def get_npu_smi_info() -> str:
    try:
        proc = subprocess.run(
            ["npu-smi", "info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=8,
        )
        return proc.stdout[:8000]
    except Exception as exc:
        return f"npu-smi unavailable: {exc!r}"


def _encode(tokenizer: Any, text: str) -> List[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def make_synthetic_token_ids(
    tokenizer: Any,
    target_input_len: int,
    idx: int,
    language: str = "zh",
) -> List[int]:
    # 生成固定长度 synthetic prompt 的 token ids。
    #
    # 作用：
    # - 当没有传 --prompt-file 时，用它构造可复现、长度可控的输入；
    # - baseline/spec 使用同一 target tokenizer 和同一 input_len 时，生成的输入一致；
    # - 适合做 input_len/output_len 的纯性能矩阵测试。
    #
    # 注意：
    # - 这里直接返回 token ids，而不是先 decode 成文本再 encode；
    #   这样可以保证输入长度严格等于 target_input_len。
    if target_input_len <= 0:
        raise ValueError("target_input_len must be positive")

    if language == "en":
        text = (
            f"This is benchmark sample {idx}. Please analyze the following material "
            "and produce a clear, structured response. Large language model "
            "inference usually includes a prefill phase and a decode phase. "
            "Speculative decoding uses a draft model to propose multiple tokens "
            "and a target model to verify them, reducing target-model decode steps. "
        )
    else:
        text = (
            f"这是第 {idx} 个性能测试样本。请阅读下面的材料，并给出结构清晰、信息完整的回答。"
            "大语言模型推理通常包含预填充阶段和解码阶段。预填充阶段处理输入上下文，"
            "解码阶段逐 token 生成输出。投机解码会引入草稿模型提前生成若干候选 token，"
            "再由目标模型进行验证，从而减少目标模型的解码步数。"
        )

    base_ids = _encode(tokenizer, text)
    if not base_ids:
        raise RuntimeError("tokenizer produced empty token ids for synthetic prompt")

    ids: List[int] = []
    while len(ids) < target_input_len:
        ids.extend(base_ids)
    return ids[:target_input_len]


def normalize_token_ids(ids: List[int], target_len: int) -> List[int]:
    # 将已有 prompt 的 token ids 调整到固定长度。
    # 用于 --prompt-file + --normalize-file-prompts 的场景：
    # - 如果原 prompt 太长，则截断；
    # - 如果原 prompt 太短，则重复拼接到 target_len。
    # 这样可以在真实 prompt 分布基础上做固定输入长度测试。
    if target_len <= 0:
        return ids
    if not ids:
        raise ValueError("cannot normalize empty prompt token ids")
    if len(ids) >= target_len:
        return ids[:target_len]
    out: List[int] = []
    while len(out) < target_len:
        out.extend(ids)
    return out[:target_len]


def load_prompt_token_ids(
    tokenizer: Any,
    num_prompts: int,
    input_len: int,
    prompt_file: Optional[str] = None,
    synthetic_language: str = "zh",
    normalize_file_prompts: bool = False,
) -> Tuple[List[List[int]], List[str]]:
    """Return prompt token ids and light metadata strings.

    Default synthetic prompts are exactly input_len tokens by construction. File
    prompts are kept as-is unless normalize_file_prompts is enabled.
    """
    if num_prompts <= 0:
        raise ValueError("num_prompts must be positive")

    token_ids_list: List[List[int]] = []
    sources: List[str] = []

    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("prompt") or obj.get("text") or obj.get("content")
                    if text is None:
                        raise ValueError("JSON line has no prompt/text/content")
                except json.JSONDecodeError:
                    text = line

                ids = _encode(tokenizer, str(text))
                if normalize_file_prompts:
                    ids = normalize_token_ids(ids, input_len)
                token_ids_list.append(ids)
                sources.append(f"file:{prompt_file}:{line_no}")
                if len(token_ids_list) >= num_prompts:
                    break
        if not token_ids_list:
            raise RuntimeError(f"no prompts loaded from {prompt_file}")
        return token_ids_list, sources

    for i in range(num_prompts):
        token_ids_list.append(
            make_synthetic_token_ids(
                tokenizer=tokenizer,
                target_input_len=input_len,
                idx=i,
                language=synthetic_language,
            )
        )
        sources.append("synthetic")
    return token_ids_list, sources


def make_vllm_prompts(prompt_token_ids: Sequence[Sequence[int]], use_token_ids: bool) -> List[Any]:
    # vLLM offline generate 支持直接传 prompt_token_ids。
    # 这里优先使用 token ids，而不是文本 prompt，原因是：
    # - 可以精确控制 input_len；
    # - 避免 chat template、空格、Unicode decode/encode 等导致长度漂移；
    # - baseline/spec 输入完全一致，更适合性能比较。
    if use_token_ids:
        return [{"prompt_token_ids": list(ids)} for ids in prompt_token_ids]
    raise ValueError(
        "use_token_ids=False is intentionally not implemented in this first version. "
        "Using prompt_token_ids keeps input length exact and avoids chat-template variance."
    )


def get_first_completion(output: Any) -> Any:
    outs = getattr(output, "outputs", None)
    if not outs:
        return None
    return outs[0]


def output_token_count(output: Any) -> int:
    # 统计单个请求实际生成的输出 token 数。
    #
    # 优先使用 vLLM completion.token_ids：
    # - 这是最准确的输出 token 数；
    # - 用于 total_output_tokens、output_tokens_per_s、avg_ms_per_output_token。
    #
    # 如果当前 vLLM 输出对象没有 token_ids，则退化为文本长度。
    # 退化分支只作为兼容兜底，不建议用于正式 token/s 报告。
    comp = get_first_completion(output)
    if comp is None:
        return 0
    token_ids = getattr(comp, "token_ids", None)
    if token_ids is not None:
        return len(token_ids)
    text = getattr(comp, "text", "") or ""
    return len(text)


def output_text_preview(output: Any, limit: int = 200) -> str:
    comp = get_first_completion(output)
    if comp is None:
        return ""
    text = getattr(comp, "text", "") or ""
    return text[:limit]


def metric_value(metric: Any) -> float:
    # vLLM metric 对象的 value 字段可能是 int/float/其他数值包装类型。
    # 这里统一转为 float，避免不同 vLLM 版本的 metric 类型差异影响统计。
    v = getattr(metric, "value", None)
    if v is None:
        return 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


def collect_spec_metrics(llm: Any, num_speculative_tokens: int) -> Dict[str, Any]:
    """Best-effort collection of vLLM offline speculative metrics.

    Newer vLLM examples expose llm.get_metrics() and metric objects with names
    like vllm:spec_decode_num_drafts. This function avoids importing vLLM
    internal metric classes so that it remains version-tolerant.
    """
    result = {
        # available:
        #   是否成功从 vLLM metrics 中发现 speculative decoding 指标。
        #   False 不代表 spec 没有运行，只代表当前 vLLM 版本/路径没有暴露这些指标。
        "available": False,

        # num_drafts:
        #   草稿轮数。一次 draft round 通常会提出 num_speculative_tokens 个候选 token。
        #   对 EAGLE3 来说，可理解为草稿模型被调用并提出候选 token 的轮数。
        "num_drafts": 0.0,

        # num_draft_tokens:
        #   草稿模型总共提出的候选 token 数。
        #   理想情况下大约等于 num_drafts * num_speculative_tokens，
        #   但实际可能受 EOS、调度、最后一轮不足等因素影响。
        "num_draft_tokens": 0.0,

        # num_accepted_tokens:
        #   目标模型验证后接受的草稿 token 总数。
        #   这是计算 acceptance_rate 和 accepted_tokens_per_draft 的核心分子。
        "num_accepted_tokens": 0.0,

        # accepted_per_pos_counts:
        #   每个草稿位置被接受的次数。
        #   例如 k=4 时，pos=0/1/2/3 分别表示第 1/2/3/4 个草稿 token。
        #   这个指标能判断是第一个 token 都不准，还是后面位置快速衰减。
        "accepted_per_pos_counts": {},
    }

    get_metrics = getattr(llm, "get_metrics", None)
    if get_metrics is None:
        return result

    try:
        metrics = get_metrics()
    except Exception as exc:
        result["error"] = repr(exc)
        return result

    for metric in metrics:
        name = str(getattr(metric, "name", ""))
        norm_name = name[:-6] if name.endswith("_total") else name
        if "spec_decode" not in norm_name:
            continue

        result["available"] = True
        if norm_name.endswith("spec_decode_num_accepted_tokens_per_pos"):
            values = getattr(metric, "values", None)
            if values is not None:
                for pos, val in enumerate(values):
                    try:
                        result["accepted_per_pos_counts"][str(pos)] = (
                            result["accepted_per_pos_counts"].get(str(pos), 0.0)
                            + float(val)
                        )
                    except Exception:
                        pass
        elif norm_name.endswith("spec_decode_num_draft_tokens"):
            result["num_draft_tokens"] += metric_value(metric)
        elif norm_name.endswith("spec_decode_num_drafts"):
            result["num_drafts"] += metric_value(metric)
        elif norm_name.endswith("spec_decode_num_accepted_tokens"):
            result["num_accepted_tokens"] += metric_value(metric)

    # Guarantee dense positions for easier CSV/JSON reading.
    for pos in range(max(num_speculative_tokens, 0)):
        result["accepted_per_pos_counts"].setdefault(str(pos), 0.0)

    return result


def diff_spec_metrics(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    # 计算 measured benchmark 阶段的 speculative 指标增量。
    #
    # 为什么要 diff：
    # - benchmark 前后会分别采集一次 llm.get_metrics()；
    # - vLLM metrics 多数是累计计数器；
    # - 直接读取 after 会混入模型加载、warmup 或之前请求的数据；
    # - after - before 才是本轮 measured run 的统计结果。
    available = bool(before.get("available") or after.get("available"))
    num_drafts = float(after.get("num_drafts", 0.0)) - float(before.get("num_drafts", 0.0))
    num_draft_tokens = float(after.get("num_draft_tokens", 0.0)) - float(before.get("num_draft_tokens", 0.0))
    num_accepted_tokens = float(after.get("num_accepted_tokens", 0.0)) - float(before.get("num_accepted_tokens", 0.0))

    before_pos = before.get("accepted_per_pos_counts", {}) or {}
    after_pos = after.get("accepted_per_pos_counts", {}) or {}
    positions = sorted(set(before_pos.keys()) | set(after_pos.keys()), key=lambda x: int(x))
    pos_counts: Dict[str, float] = {}
    pos_rates: Dict[str, Optional[float]] = {}
    for p in positions:
        val = float(after_pos.get(p, 0.0)) - float(before_pos.get(p, 0.0))
        pos_counts[p] = val
        pos_rates[p] = (val / num_drafts) if num_drafts > 0 else None

    # acceptance_rate:
    #   被接受的草稿 token 数 / 草稿 token 总数。
    #   例如 0.10 表示草稿模型提出的 token 只有 10% 被目标模型接受。
    #   注意：acceptance_rate 高不必然意味着端到端加速，还要看草稿模型开销。
    acceptance_rate = (num_accepted_tokens / num_draft_tokens) if num_draft_tokens > 0 else None

    # accepted_tokens_per_draft:
    #   每一轮 draft 平均被接受的草稿 token 数。
    #   例如 k=4 但 accepted_tokens_per_draft=0.4，说明每轮平均只接受 0.4 个 token，
    #   通常很难覆盖草稿模型 forward 和验证开销。
    accepted_tokens_per_draft = (num_accepted_tokens / num_drafts) if num_drafts > 0 else None

    # mean_acceptance_length_including_bonus:
    #   vLLM 示例里常用的“平均推进长度”，定义为 1 + accepted_tokens_per_draft。
    #   这里的 +1 表示即使草稿 token 全部拒绝，目标模型验证后通常也会产生/确认一个 token。
    mean_acceptance_length = 1.0 + accepted_tokens_per_draft if accepted_tokens_per_draft is not None else None

    return {
        "available": available,
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "acceptance_rate": acceptance_rate,
        "accepted_tokens_per_draft": accepted_tokens_per_draft,
        "mean_acceptance_length_including_bonus": mean_acceptance_length,
        "accepted_per_pos_counts": pos_counts,
        "accepted_per_pos_rates": pos_rates,
    }


def summarize_offline_run(
    *,
    args: Any,
    run_id: str,
    case_id: str,
    load_time_s: float,
    measured_wall_time_s: float,
    batch_records: Sequence[Dict[str, Any]],
    request_records: Sequence[Dict[str, Any]],
    spec_metrics_delta: Dict[str, Any],
    env_info: Dict[str, Any],
) -> Dict[str, Any]:
    # 成功/失败请求拆分：
    # - success_count / failed_count / success_rate 用于判断测试是否稳定；
    # - 失败请求不参与 token 吞吐和长度统计，避免把异常请求算入性能结论。
    successful_requests = [r for r in request_records if r.get("success", True)]
    failed_requests = [r for r in request_records if not r.get("success", True)]

    # 每个 batch 的端到端耗时，单位为秒。
    # 后续转换成 batch_latency_avg_ms / p50 / p90 / p99。
    batch_latencies = [float(b["latency_s"]) for b in batch_records]

    # total_input_tokens:
    #   所有成功请求的输入 token 总数。
    #   用于 total_tokens_per_s，也用于核对 input_len_actual_avg 是否符合预期。
    input_tokens = sum(int(r.get("input_tokens", 0)) for r in successful_requests)

    # total_output_tokens:
    #   所有成功请求实际生成的输出 token 总数。
    #   用于 output_tokens_per_s 和 avg_ms_per_output_token。
    output_tokens = sum(int(r.get("output_tokens", 0)) for r in successful_requests)

    # output_tokens_per_s:
    #   输出 token 吞吐 = 总输出 token 数 / measured 阶段墙钟时间。
    #   这是比较 baseline/spec 端到端生成性能的最核心指标。
    output_tps = output_tokens / measured_wall_time_s if measured_wall_time_s > 0 else None

    # total_tokens_per_s:
    #   总 token 吞吐 = (输入 token + 输出 token) / measured 阶段墙钟时间。
    #   更偏整体吞吐，但对 decode 加速来说，output_tokens_per_s 通常更关键。
    total_tps = (input_tokens + output_tokens) / measured_wall_time_s if measured_wall_time_s > 0 else None

    # requests_per_s:
    #   离线 generate 每秒处理的请求数。
    #   注意它不是在线 QPS，因为这里没有 HTTP server、排队和流式返回。
    requests_per_s = len(successful_requests) / measured_wall_time_s if measured_wall_time_s > 0 else None

    # avg_ms_per_output_token:
    #   平均每个输出 token 的端到端耗时近似值。
    #   计算方式是 measured 总耗时 / 总输出 token 数。
    #   它不是严格 TPOT/ITL；严格 TPOT/ITL 需要在线 streaming 或低层 engine step 打点。
    avg_ms_per_output_token = (measured_wall_time_s * 1000.0 / output_tokens) if output_tokens > 0 else None

    summary = {
        # -------------------------
        # 1. 实验标识与元信息
        # -------------------------
        # run_id:
        #   单次运行的唯一 ID，通常包含时间戳和随机后缀。
        # case_id:
        #   可读的实验配置摘要，例如 baseline_in512_out256_bs1_k0。
        # bench_mode:
        #   当前为 offline，表示使用 vLLM.LLM.generate()，不是 server 压测。
        "run_id": run_id,
        "case_id": case_id,
        "bench_mode": "offline",
        "mode": args.mode,
        "experiment_date": now_iso(),
        "tester": args.tester,
        "hardware": args.hardware,
        "hostname": env_info.get("hostname"),

        # -------------------------
        # 2. 模型与 speculative 配置
        # -------------------------
        # target_model:
        #   被验证的目标模型路径或名称。
        # draft_model:
        #   草稿模型路径；baseline 模式为空。
        # spec_method:
        #   speculative 方法，例如 eagle3、dflash、draft_model、mtp 等。
        # num_speculative_tokens:
        #   每轮草稿模型尝试生成/提出的候选 token 数，也常记作 k。
        "target_model": args.target_model,
        "tokenizer": args.tokenizer or args.target_model,
        "draft_model": args.draft_model if args.mode == "spec" else "",
        "spec_method": args.spec_method if args.mode == "spec" else "",
        "num_speculative_tokens": args.num_speculative_tokens if args.mode == "spec" else 0,
        "draft_tensor_parallel_size": args.draft_tensor_parallel_size if args.mode == "spec" else 0,

        # -------------------------
        # 3. vLLM/运行时配置
        # -------------------------
        # tensor_parallel_size:
        #   目标模型 tensor parallel 大小。
        # max_num_seqs / max_num_batched_tokens:
        #   vLLM 调度和批处理相关限制；对显存和吞吐都有影响。
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,

        # -------------------------
        # 4. 请求规模与输入来源
        # -------------------------
        # num_prompts:
        #   measured 阶段总请求数。
        # batch_size:
        #   离线每次 llm.generate() 提交的 prompt 数，不等同于在线并发数。
        # warmup_batches:
        #   正式计时前的 warmup batch 数，不计入 measured_wall_time_s。
        # prompt_file:
        #   为空时使用 synthetic 固定长度输入；非空时从文件读取 prompt。
        "num_prompts": args.num_prompts,
        "batch_size": args.batch_size,
        "warmup_batches": args.warmup_batches,
        "prompt_file": args.prompt_file,
        "synthetic_language": args.synthetic_language,
        "normalize_file_prompts": args.normalize_file_prompts,

        # -------------------------
        # 5. 输入/输出长度统计
        # -------------------------
        # input_len_target / output_len_target:
        #   命令行指定的目标输入/输出长度。
        # input_len_actual_*:
        #   实际输入 token 长度统计。使用 synthetic token ids 时通常精确等于 input_len_target；
        #   使用 prompt_file 且不 normalize 时，它反映真实问题集的平均/最小/最大长度。
        # output_len_actual_*:
        #   实际输出 token 长度统计。若设置 ignore_eos=True，通常接近 output_len_target；
        #   若允许 EOS，可能明显小于 output_len_target。
        "input_len_target": args.input_len,
        "output_len_target": args.output_len,
        "input_len_actual_avg": mean_or_none([r["input_tokens"] for r in successful_requests]),
        "input_len_actual_min": min([r["input_tokens"] for r in successful_requests], default=None),
        "input_len_actual_max": max([r["input_tokens"] for r in successful_requests], default=None),
        "output_len_actual_avg": mean_or_none([r["output_tokens"] for r in successful_requests]),
        "output_len_actual_min": min([r["output_tokens"] for r in successful_requests], default=None),
        "output_len_actual_max": max([r["output_tokens"] for r in successful_requests], default=None),

        # -------------------------
        # 6. 采样参数
        # -------------------------
        # temperature/top_p/top_k:
        #   采样配置。为了做 baseline/spec 可复现比较，第一轮通常建议 temperature=0、top_p=1。
        # ignore_eos:
        #   是否忽略 EOS，强制生成到 output_len_target；做固定长度性能测试时建议打开。
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "ignore_eos": args.ignore_eos,
        "seed": args.seed,

        # -------------------------
        # 7. 运行耗时与稳定性
        # -------------------------
        # load_time_s:
        #   LLM 对象初始化/模型加载耗时，不计入 measured throughput。
        # measured_wall_time_s:
        #   正式 measured 阶段总墙钟时间，不包含 warmup。
        # success_count/failed_count/success_rate:
        #   请求成功和失败情况，用于判断本轮结果是否可信。
        "load_time_s": load_time_s,
        "measured_wall_time_s": measured_wall_time_s,
        "success_count": len(successful_requests),
        "failed_count": len(failed_requests),
        "success_rate": len(successful_requests) / len(request_records) if request_records else 0.0,
        "num_batches": len(batch_records),

        # -------------------------
        # 8. 吞吐与平均耗时指标
        # -------------------------
        # total_input_tokens:
        #   成功请求的输入 token 总数。
        # total_output_tokens:
        #   成功请求的输出 token 总数。
        # output_tokens_per_s:
        #   输出 token 吞吐。对 speculative decoding 加速最重要。
        # total_tokens_per_s:
        #   输入+输出 token 总吞吐。长输入场景下会受 prefill 影响更大。
        # requests_per_s:
        #   离线每秒处理请求数，不等同于在线 QPS。
        # avg_ms_per_output_token:
        #   measured 总耗时 / 输出 token 总数，是离线端到端 ms/token 近似指标。
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "output_tokens_per_s": output_tps,
        "total_tokens_per_s": total_tps,
        "requests_per_s": requests_per_s,
        "avg_ms_per_output_token": avg_ms_per_output_token,

        # -------------------------
        # 9. batch latency 分位数
        # -------------------------
        # batch_latency_*:
        #   每次离线 llm.generate(batch_prompts) 调用的端到端耗时统计。
        #   bs=1 时近似单请求端到端 latency；
        #   bs>1 时表示一个 batch 的整体耗时，不是单请求 latency。
        "batch_latency_avg_ms": ms(mean_or_none(batch_latencies)),
        "batch_latency_p50_ms": ms(percentile(batch_latencies, 50)),
        "batch_latency_p90_ms": ms(percentile(batch_latencies, 90)),
        "batch_latency_p99_ms": ms(percentile(batch_latencies, 99)),

        # Offline LLM.generate does not expose streaming TTFT/ITL. Keep these
        # explicit instead of silently reporting misleading values.
        "ttft_avg_ms": None,
        "tpot_avg_ms": None,
        "itl_avg_ms": None,
        "latency_metric_note": (
            "offline LLM.generate measures end-to-end batch latency; "
            "TTFT/ITL/TPOT require server streaming or lower-level engine stepping"
        ),

        # -------------------------
        # 10. speculative decoding 指标
        # -------------------------
        # spec_metrics_available:
        #   是否成功采集到 vLLM speculative metrics。
        # spec_num_drafts:
        #   measured 阶段 draft round 总数。
        # spec_num_draft_tokens:
        #   草稿模型提出的候选 token 总数。
        # spec_num_accepted_tokens:
        #   被目标模型接受的草稿 token 总数。
        # spec_acceptance_rate:
        #   spec_num_accepted_tokens / spec_num_draft_tokens。
        # spec_accepted_tokens_per_draft:
        #   每轮 draft 平均接受 token 数。
        # spec_mean_acceptance_length_including_bonus:
        #   1 + accepted_tokens_per_draft，粗略表示每轮验证平均推进长度。
        # spec_accepted_per_pos_counts/rates:
        #   每个草稿位置的接受次数/接受率，用于判断接受率是否随位置快速衰减。
        "spec_metrics_available": spec_metrics_delta.get("available"),
        "spec_num_drafts": spec_metrics_delta.get("num_drafts"),
        "spec_num_draft_tokens": spec_metrics_delta.get("num_draft_tokens"),
        "spec_num_accepted_tokens": spec_metrics_delta.get("num_accepted_tokens"),
        "spec_acceptance_rate": spec_metrics_delta.get("acceptance_rate"),
        "spec_accepted_tokens_per_draft": spec_metrics_delta.get("accepted_tokens_per_draft"),
        "spec_mean_acceptance_length_including_bonus": spec_metrics_delta.get(
            "mean_acceptance_length_including_bonus"
        ),
        "spec_accepted_per_pos_counts": spec_metrics_delta.get("accepted_per_pos_counts"),
        "spec_accepted_per_pos_rates": spec_metrics_delta.get("accepted_per_pos_rates"),

        # -------------------------
        # 11. 软件环境
        # -------------------------
        # 这些字段用于复现实验环境；vLLM/torch/transformers 版本变化可能显著影响结果。
        "python_version": env_info.get("python_version"),
        "python_executable": env_info.get("python_executable"),
        "platform": env_info.get("platform"),
        "vllm_version": env_info.get("vllm_version"),
        "torch_version": env_info.get("torch_version"),
        "transformers_version": env_info.get("transformers_version"),
    }
    return summary
