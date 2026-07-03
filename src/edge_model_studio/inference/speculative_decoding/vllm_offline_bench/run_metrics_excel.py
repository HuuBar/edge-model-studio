#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run target baseline and draft/spec benchmark, then export raw performance metrics to one Excel file.

特点：
1. 只统计 baseline 和 spec 两类原始性能指标，不计算两者 speedup，不做性能对比结论；
2. baseline/spec 分别独立进程执行，适合 Ascend/NPU 环境；
3. 只输出一个 xlsx，包含“性能指标”和“字段说明”两个 sheet；
4. baseline 没有的投机推理指标写为“无”；
5. spec 应该产出但未产出的指标写为“异常”；
6. 预留在线测试字段，例如首 token 时延、ITL、TPOT、在线 E2E latency；离线测试默认写“无”；
7. “测试对象”直接使用模型形态，例如 Qwen2.5-14B-Instruct 和 Qwen2.5-14B-Instruct+draftmodel。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from xml.sax.saxutils import escape

from bench_common import load_json_maybe

NA = "无"
ERR = "异常"
OK = "正常"


@dataclass(frozen=True)
class ColumnSpec:
    key: str
    title: str
    description: str
    spec_only: bool = False
    config_field: bool = False
    optional_field: bool = False
    width: float = 16.0


# 字段顺序按“吞吐速率/端到端耗时/投机加速相关指标优先”排列。
COLUMNS: List[ColumnSpec] = [
    ColumnSpec("test_object", "测试对象", "具体模型或模型+草稿模型形态，例如 Qwen2.5-14B-Instruct 或 Qwen2.5-14B-Instruct+draftmodel。", width=30),

    # 吞吐与端到端效率：放在最前面，便于直接判断模型生成效率。
    ColumnSpec("output_tokens_per_s", "输出token吞吐(tokens/s)", "总输出 token 数 / measured 阶段墙钟时间。评估生成吞吐最重要。", width=22),
    ColumnSpec("total_tokens_per_s", "总token吞吐(tokens/s)", "输入+输出 token 总数 / measured 阶段墙钟时间。长输入场景会受 prefill 影响。", width=21),
    ColumnSpec("avg_ms_per_output_token", "平均每输出token耗时(ms)", "measured 总耗时 / 总输出 token 数。离线端到端 ms/token 近似指标，不等同严格 TPOT/ITL。", width=24),
    ColumnSpec("requests_per_s", "请求吞吐(req/s)", "成功请求数 / measured 阶段墙钟时间。离线指标，不等同在线 QPS。", width=18),
    ColumnSpec("measured_wall_time_s", "正式测试总耗时(s)", "不含 warmup 的 measured 阶段总墙钟时间。", width=20),
    ColumnSpec("batch_latency_avg_ms", "batch平均耗时(ms)", "每次 llm.generate(batch) 调用的平均端到端耗时。", width=20),
    ColumnSpec("batch_latency_p50_ms", "batch P50耗时(ms)", "batch 端到端耗时 P50。", width=18),
    ColumnSpec("batch_latency_p90_ms", "batch P90耗时(ms)", "batch 端到端耗时 P90。", width=18),
    ColumnSpec("batch_latency_p99_ms", "batch P99耗时(ms)", "batch 端到端耗时 P99。", width=18),

    # 在线服务字段预留：当前 run_metrics_excel.py 使用离线 LLM.generate()，这些字段默认写“无”。
    ColumnSpec("ttft_avg_ms", "首token时延TTFT平均(ms)", "在线 streaming 测试字段。离线测试无法准确测量，填无。", optional_field=True, width=24),
    ColumnSpec("ttft_p50_ms", "首token时延TTFT P50(ms)", "在线 streaming 测试字段。离线测试无法准确测量，填无。", optional_field=True, width=24),
    ColumnSpec("ttft_p90_ms", "首token时延TTFT P90(ms)", "在线 streaming 测试字段。离线测试无法准确测量，填无。", optional_field=True, width=24),
    ColumnSpec("ttft_p99_ms", "首token时延TTFT P99(ms)", "在线 streaming 测试字段。离线测试无法准确测量，填无。", optional_field=True, width=24),
    ColumnSpec("itl_avg_ms", "增量时延ITL平均(ms)", "在线 streaming 测试字段。表示相邻输出 token 间隔。离线测试填无。", optional_field=True, width=22),
    ColumnSpec("itl_p50_ms", "增量时延ITL P50(ms)", "在线 streaming 测试字段。离线测试填无。", optional_field=True, width=22),
    ColumnSpec("itl_p90_ms", "增量时延ITL P90(ms)", "在线 streaming 测试字段。离线测试填无。", optional_field=True, width=22),
    ColumnSpec("itl_p99_ms", "增量时延ITL P99(ms)", "在线 streaming 测试字段。离线测试填无。", optional_field=True, width=22),
    ColumnSpec("tpot_avg_ms", "TPOT平均(ms)", "在线测试字段。通常表示除首 token 外每个输出 token 的平均耗时。离线测试填无。", optional_field=True, width=18),
    ColumnSpec("tpot_p50_ms", "TPOT P50(ms)", "在线测试字段。离线测试填无。", optional_field=True, width=18),
    ColumnSpec("tpot_p90_ms", "TPOT P90(ms)", "在线测试字段。离线测试填无。", optional_field=True, width=18),
    ColumnSpec("tpot_p99_ms", "TPOT P99(ms)", "在线测试字段。离线测试填无。", optional_field=True, width=18),
    ColumnSpec("e2e_avg_ms", "在线E2E平均时延(ms)", "在线服务端到端请求时延平均值。离线测试填无。", optional_field=True, width=22),
    ColumnSpec("e2e_p50_ms", "在线E2E P50时延(ms)", "在线服务端到端请求时延 P50。离线测试填无。", optional_field=True, width=22),
    ColumnSpec("e2e_p90_ms", "在线E2E P90时延(ms)", "在线服务端到端请求时延 P90。离线测试填无。", optional_field=True, width=22),
    ColumnSpec("e2e_p99_ms", "在线E2E P99时延(ms)", "在线服务端到端请求时延 P99。离线测试填无。", optional_field=True, width=22),

    # 投机推理配置与接受率：baseline 行写“无”。
    ColumnSpec("num_speculative_tokens", "草稿预测token配置(k)", "--num-speculative-tokens，每轮草稿模型预测/提出的候选 token 数；baseline 行写为无。", spec_only=True, config_field=True, width=22),
    ColumnSpec("spec_metrics_available", "投机指标是否可用", "vLLM 是否暴露 speculative metrics。baseline 无该指标。", spec_only=True, width=18),
    ColumnSpec("spec_acceptance_rate", "草稿token接受率", "被接受草稿 token 数 / 草稿 token 总数。仅投机推理产出。", spec_only=True, width=18),
    ColumnSpec("spec_accepted_tokens_per_draft", "每轮平均接受草稿token数", "被接受草稿 token 数 / draft round 数。仅投机推理产出。", spec_only=True, width=26),
    ColumnSpec("spec_mean_acceptance_length_including_bonus", "平均推进长度(含bonus)", "1 + 每轮平均接受草稿 token 数。粗略表示每轮验证平均推进 token 数。", spec_only=True, width=24),
    ColumnSpec("spec_num_drafts", "草稿轮数", "measured 阶段 draft round 总数。仅投机推理产出。", spec_only=True, width=14),
    ColumnSpec("spec_num_draft_tokens", "草稿token总数", "草稿模型提出的候选 token 总数。仅投机推理产出。", spec_only=True, width=16),
    ColumnSpec("spec_num_accepted_tokens", "接受草稿token总数", "目标模型接受的草稿 token 总数。仅投机推理产出。", spec_only=True, width=20),
    ColumnSpec("spec_accepted_per_pos_rates", "各位置接受率", "每个草稿位置的接受率，例如第1/2/3/4个草稿 token。仅投机推理产出。", spec_only=True, width=34),

    ColumnSpec("success_count", "成功请求数", "成功生成的请求数。", width=14),
    ColumnSpec("failed_count", "失败请求数", "失败请求数；非 0 时关键指标可能标记为异常。", width=14),
    ColumnSpec("success_rate", "成功率", "成功请求数 / 总请求数。", width=12),
    ColumnSpec("total_output_tokens", "输出token总数", "成功请求实际生成的输出 token 总数。", width=16),
    ColumnSpec("total_input_tokens", "输入token总数", "成功请求输入 token 总数。", width=16),
    ColumnSpec("input_len_actual_avg", "实际平均输入长度", "实际输入 token 平均长度。", width=18),
    ColumnSpec("output_len_actual_avg", "实际平均输出长度", "实际输出 token 平均长度。", width=18),

    ColumnSpec("target_model", "目标模型路径", "目标模型路径或名称。", config_field=True, width=42),
    ColumnSpec("draft_model", "草稿模型路径", "草稿模型路径；baseline 行写为无。", spec_only=True, config_field=True, width=42),
    ColumnSpec("spec_method", "投机方法", "例如 eagle3、dflash、draft_model、mtp 等；baseline 行写为无。", spec_only=True, config_field=True, width=14),
    ColumnSpec("draft_tensor_parallel_size", "草稿TP", "草稿模型 tensor parallel size；baseline 行写为无。", spec_only=True, config_field=True, width=12),

    ColumnSpec("batch_size", "离线batch_size", "离线每次 llm.generate 提交的 prompt 数，不等同在线并发。", config_field=True, width=16),
    ColumnSpec("num_prompts", "请求总数", "measured 阶段请求总数。", config_field=True, width=12),
    ColumnSpec("input_len_target", "目标输入长度", "命令行指定的 input_len。", config_field=True, width=14),
    ColumnSpec("output_len_target", "目标输出长度", "命令行指定的 output_len / max_tokens。", config_field=True, width=14),
    ColumnSpec("temperature", "temperature", "采样 temperature。", config_field=True, width=13),
    ColumnSpec("top_p", "top_p", "采样 top_p。", config_field=True, width=10),
    ColumnSpec("top_k", "top_k", "采样 top_k；未设置写为无。", config_field=True, optional_field=True, width=10),
    ColumnSpec("ignore_eos", "ignore_eos", "是否忽略 EOS，固定长度测试通常为 True。", config_field=True, width=12),
    ColumnSpec("prompt_file", "prompt文件", "使用的 prompt 文件；synthetic 模式写为无。", config_field=True, optional_field=True, width=34),
    ColumnSpec("normalize_file_prompts", "文件prompt定长化", "是否把 prompt_file 中 prompt 重复/截断到 input_len。", config_field=True, width=16),

    ColumnSpec("hardware", "硬件机器", "实验记录中的硬件描述。", config_field=True, width=18),
    ColumnSpec("tester", "实验人", "实验人。", config_field=True, width=14),
    ColumnSpec("summary_json", "summary路径", "该行对应的 summary JSON 文件路径；异常时可能为空。", config_field=True, optional_field=True, width=48),
    ColumnSpec("error_message", "异常信息", "执行失败、summary 缺失或指标异常时的说明。", config_field=True, optional_field=True, width=42),
]

@dataclass
class BenchResult:
    mode: str
    test_object: str
    status: str
    summary: Optional[Dict[str, Any]]
    summary_json: str
    error_message: str = ""


def latest_summary_json(result_dir: Path) -> Optional[Path]:
    candidates = sorted(result_dir.glob("*.summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_summary(path: str | Path) -> Dict[str, Any]:
    summary = load_json_maybe(str(path))
    if not isinstance(summary, dict):
        raise RuntimeError(f"summary JSON invalid: {path}")
    return summary


def add_common_args(cmd: List[str], args: argparse.Namespace) -> None:
    cmd += [
        "--target-model", args.target_model,
        "--input-len", str(args.input_len),
        "--output-len", str(args.output_len),
        "--num-prompts", str(args.num_prompts),
        "--batch-size", str(args.batch_size),
        "--warmup-batches", str(args.warmup_batches),
        "--temperature", str(args.temperature),
        "--top-p", str(args.top_p),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--dtype", args.dtype,
        "--hardware", args.hardware,
        "--tester", args.tester,
        "--synthetic-language", args.synthetic_language,
    ]
    if args.tokenizer:
        cmd += ["--tokenizer", args.tokenizer]
    if args.prompt_file:
        cmd += ["--prompt-file", args.prompt_file]
    if args.normalize_file_prompts:
        cmd.append("--normalize-file-prompts")
    if args.top_k is not None:
        cmd += ["--top-k", str(args.top_k)]
    if args.ignore_eos:
        cmd.append("--ignore-eos")
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.skip_special_tokens is not None:
        cmd += ["--skip-special-tokens", str(args.skip_special_tokens).lower()]
    if args.max_model_len is not None:
        cmd += ["--max-model-len", str(args.max_model_len)]
    if args.gpu_memory_utilization is not None:
        cmd += ["--gpu-memory-utilization", str(args.gpu_memory_utilization)]
    if args.max_num_seqs is not None:
        cmd += ["--max-num-seqs", str(args.max_num_seqs)]
    if args.max_num_batched_tokens is not None:
        cmd += ["--max-num-batched-tokens", str(args.max_num_batched_tokens)]
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.enable_chunked_prefill:
        cmd.append("--enable-chunked-prefill")
    if args.disable_log_stats:
        cmd.append("--disable-log-stats")
    if args.extra_llm_kwargs_json:
        cmd += ["--extra-llm-kwargs-json", args.extra_llm_kwargs_json]
    if args.collect_npu_smi:
        cmd.append("--collect-npu-smi")
    if args.use_tqdm:
        cmd.append("--use-tqdm")
    if args.print_output:
        cmd.append("--print-output")



def default_model_display_name(model_path: str, fallback: str) -> str:
    # 从模型路径中提取可读名称，例如 /path/Qwen2.5-14B-Instruct -> Qwen2.5-14B-Instruct。
    # 如果路径为空，则使用 fallback。
    if not model_path:
        return fallback
    name = Path(model_path.rstrip("/")).name
    return name or fallback


def build_test_object_name(mode: str, args: argparse.Namespace, summary: Optional[Dict[str, Any]] = None) -> str:
    # Excel 中“测试对象”的显示规则：
    # - baseline 行：直接写目标模型具体名称；
    # - spec 行：写“目标模型+草稿模型形态”。
    #
    # 可以通过 --target-name / --draft-name 覆盖显示名称。
    # 未指定时，目标模型名从路径 basename 推导，草稿模型默认显示为 draftmodel，
    # 避免 checkpoint_best 这类目录名出现在报告里。
    summary = summary or {}
    target_source = args.target_name or summary.get("target_model") or args.target_model
    target_name = args.target_name or default_model_display_name(str(target_source or ""), "target_model")

    if mode == "baseline":
        return target_name

    draft_name = args.draft_name or "draftmodel"
    return f"{target_name}+{draft_name}"

def run_one_mode(mode: str, args: argparse.Namespace, output_dir: Path) -> BenchResult:
    output_dir.mkdir(parents=True, exist_ok=True)

    explicit_summary = args.baseline_summary_json if mode == "baseline" else args.spec_summary_json
    test_object = build_test_object_name(mode, args)

    if explicit_summary:
        try:
            summary = load_summary(explicit_summary)
            test_object = build_test_object_name(mode, args, summary)
            status = OK if int(summary.get("failed_count") or 0) == 0 and int(summary.get("success_count") or 0) > 0 else ERR
            err = "" if status == OK else "summary 中存在失败请求或成功请求数为 0"
            return BenchResult(mode, test_object, status, summary, str(explicit_summary), err)
        except Exception as exc:
            return BenchResult(mode, test_object, ERR, None, str(explicit_summary), f"读取 summary 失败: {exc!r}")

    script_dir = Path(__file__).resolve().parent
    run_bench_py = script_dir / "run_offline_bench.py"

    cmd = [sys.executable, str(run_bench_py), "--mode", mode, "--output-dir", str(output_dir)]
    if mode == "spec":
        if args.speculative_config_json:
            cmd += ["--speculative-config-json", args.speculative_config_json]
        else:
            cmd += [
                "--draft-model", args.draft_model,
                "--spec-method", args.spec_method,
                "--num-speculative-tokens", str(args.num_speculative_tokens),
            ]
            if args.draft_tensor_parallel_size > 0:
                cmd += ["--draft-tensor-parallel-size", str(args.draft_tensor_parallel_size)]
        if args.extra_speculative_config_json:
            cmd += ["--extra-speculative-config-json", args.extra_speculative_config_json]
        if args.disable_padded_drafter_batch:
            cmd.append("--disable-padded-drafter-batch")
        if args.parallel_drafting:
            cmd.append("--parallel-drafting")

    add_common_args(cmd, args)

    print("\n========== RUN BENCH ==========")
    print(" ".join(cmd))
    proc = subprocess.run(cmd)

    summary_path = latest_summary_json(output_dir)
    if summary_path is None:
        return BenchResult(mode, test_object, ERR, None, "", f"benchmark returncode={proc.returncode}，未找到 *.summary.json")

    try:
        summary = load_summary(summary_path)
        test_object = build_test_object_name(mode, args, summary)
    except Exception as exc:
        return BenchResult(mode, test_object, ERR, None, str(summary_path), f"读取 summary 失败: {exc!r}")

    failed = int(summary.get("failed_count") or 0)
    success = int(summary.get("success_count") or 0)
    if proc.returncode != 0:
        return BenchResult(mode, test_object, ERR, summary, str(summary_path), f"benchmark returncode={proc.returncode}")
    if failed > 0 or success <= 0:
        return BenchResult(mode, test_object, ERR, summary, str(summary_path), f"failed_count={failed}, success_count={success}")
    return BenchResult(mode, test_object, OK, summary, str(summary_path), "")


def is_missing_value(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return False


def stringify_complex(v: Any) -> Any:
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False)
    return v


def value_for_column(result: BenchResult, col: ColumnSpec) -> Any:
    mode = result.mode
    if col.key == "test_object":
        return result.test_object
    if col.key == "status":
        return result.status
    if col.key == "summary_json":
        return result.summary_json or ERR
    if col.key == "error_message":
        return result.error_message or NA

    if col.spec_only and mode != "spec":
        return NA

    if result.summary is None:
        return ERR if not col.config_field else (NA if col.optional_field or col.spec_only else ERR)

    if mode == "spec" and col.key.startswith("spec_") and col.key != "spec_metrics_available":
        if result.summary.get("spec_metrics_available") is not True:
            return ERR

    v = result.summary.get(col.key)

    if col.key == "spec_metrics_available" and mode == "spec":
        return True if v is True else ERR

    if is_missing_value(v):
        if col.optional_field:
            return NA
        if col.spec_only and mode != "spec":
            return NA
        return ERR

    return stringify_complex(v)


def build_rows(results: Sequence[BenchResult]) -> List[Dict[str, Any]]:
    return [{col.key: value_for_column(res, col) for col in COLUMNS} for res in results]


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[c.key for c in COLUMNS])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def col_letter(n: int) -> str:
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def xml_text(v: Any) -> str:
    return escape(str(v), {"\"": "&quot;", "'": "&apos;"})


def xlsx_cell(ref: str, value: Any, style_id: int = 0) -> str:
    if isinstance(value, bool):
        value = "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isnan(float(value)):
        return f'<c r="{ref}" s="{style_id}"><v>{value}</v></c>'
    return f'<c r="{ref}" s="{style_id}" t="inlineStr"><is><t>{xml_text(value)}</t></is></c>'


def make_sheet_xml(rows: Sequence[Sequence[Any]], widths: Sequence[float], freeze_first_row: bool = True, autofilter: bool = True) -> str:
    max_row = len(rows)
    max_col = max((len(r) for r in rows), default=1)
    last_cell = f"{col_letter(max_col)}{max_row}"
    cols_xml = "".join(
        f'<col min="{i}" max="{i}" width="{widths[i-1] if i-1 < len(widths) else 14}" customWidth="1"/>'
        for i in range(1, max_col + 1)
    )
    pane_xml = ""
    if freeze_first_row:
        pane_xml = (
            '<sheetViews><sheetView workbookViewId="0">'
            '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
            '</sheetView></sheetViews>'
        )
    row_xml_parts = []
    for r_idx, row in enumerate(rows, start=1):
        cells = []
        for c_idx, value in enumerate(row, start=1):
            ref = f"{col_letter(c_idx)}{r_idx}"
            if r_idx == 1:
                style = 1
            elif value == ERR:
                style = 3
            elif value == NA:
                style = 4
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                style = 2
            else:
                style = 0
            cells.append(xlsx_cell(ref, value, style))
        row_xml_parts.append(f'<row r="{r_idx}">' + "".join(cells) + '</row>')
    auto_filter_xml = f'<autoFilter ref="A1:{last_cell}"/>' if autofilter and max_row >= 1 else ""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'{pane_xml}'
        f'<cols>{cols_xml}</cols>'
        '<sheetData>' + "".join(row_xml_parts) + '</sheetData>'
        f'{auto_filter_xml}'
        '</worksheet>'
    )


def workbook_xml(sheet_names: Sequence[str]) -> str:
    sheets = []
    for i, name in enumerate(sheet_names, start=1):
        sheets.append(f'<sheet name="{xml_text(name)}" sheetId="{i}" r:id="rId{i}"/>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets>' + "".join(sheets) + '</sheets>'
        '</workbook>'
    )


def workbook_rels(sheet_count: int) -> str:
    rels = []
    for i in range(1, sheet_count + 1):
        rels.append(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>')
    rels.append(f'<Relationship Id="rId{sheet_count + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rels) + '</Relationships>'
    )


def content_types(sheet_count: int) -> str:
    overrides = [
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for i in range(1, sheet_count + 1):
        overrides.append(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        + "".join(overrides) + '</Types>'
    )


def root_rels() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )


def styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="4">
    <font><sz val="11"/><name val="Calibri"/></font>
    <font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font>
    <font><color rgb="FF111827"/><sz val="11"/><name val="Calibri"/></font>
    <font><color rgb="FFB91C1C"/><b/><sz val="11"/><name val="Calibri"/></font>
  </fonts>
  <fills count="5">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E79"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFFEE2E2"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFF3F4F6"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border><left style="thin"><color rgb="FFD1D5DB"/></left><right style="thin"><color rgb="FFD1D5DB"/></right><top style="thin"><color rgb="FFD1D5DB"/></top><bottom style="thin"><color rgb="FFD1D5DB"/></bottom><diagonal/></border>
  </borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="5">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>
    <xf numFmtId="4" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1" applyBorder="1"/>
    <xf numFmtId="0" fontId="3" fillId="3" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1"/>
    <xf numFmtId="0" fontId="2" fillId="4" borderId="1" xfId="0" applyFill="1" applyBorder="1"/>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>"""


def write_xlsx(path: Path, metric_rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [c.title for c in COLUMNS]
    rows = [header]
    for row in metric_rows:
        rows.append([row.get(c.key, "") for c in COLUMNS])
    desc_rows = [["字段", "字段名", "是否投机专属", "说明"]]
    for c in COLUMNS:
        desc_rows.append([c.title, c.key, "是" if c.spec_only else "否", c.description])
    sheets = [("性能指标", rows, [c.width for c in COLUMNS]), ("字段说明", desc_rows, [22, 34, 16, 86])]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types(len(sheets)))
        zf.writestr("_rels/.rels", root_rels())
        zf.writestr("xl/workbook.xml", workbook_xml([s[0] for s in sheets]))
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels(len(sheets)))
        zf.writestr("xl/styles.xml", styles_xml())
        for i, (_, sheet_rows, widths) in enumerate(sheets, start=1):
            zf.writestr(f"xl/worksheets/sheet{i}.xml", make_sheet_xml(sheet_rows, widths))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run baseline/spec metrics and export them to one Excel workbook without comparing speedups.")
    p.add_argument("--target-model", required=False, default="")
    p.add_argument("--draft-model", default="")
    p.add_argument("--target-name", default="", help="Excel 测试对象显示用目标模型名称；默认取 target model 路径 basename")
    p.add_argument("--draft-name", default="draftmodel", help="Excel 测试对象显示用草稿模型名称；spec 行显示为 target+draft，默认 draftmodel")
    p.add_argument("--tokenizer", default="")
    p.add_argument("--spec-method", default="eagle3", choices=["eagle", "eagle3", "draft_model", "ngram", "mtp", "suffix", "dflash"])
    p.add_argument("--num-speculative-tokens", type=int, default=4)
    p.add_argument("--draft-tensor-parallel-size", type=int, default=1)
    p.add_argument("--speculative-config-json", default="")
    p.add_argument("--extra-speculative-config-json", default="")
    p.add_argument("--disable-padded-drafter-batch", action="store_true")
    p.add_argument("--parallel-drafting", action="store_true")
    p.add_argument("--num-prompts", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup-batches", type=int, default=1)
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--prompt-file", default="")
    p.add_argument("--normalize-file-prompts", action="store_true")
    p.add_argument("--synthetic-language", choices=["zh", "en"], default="zh")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--ignore-eos", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--skip-special-tokens", type=lambda x: x.lower() == "true", default=None)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=None)
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--max-num-batched-tokens", type=int, default=None)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--enable-chunked-prefill", action="store_true")
    p.add_argument("--disable-log-stats", action="store_true")
    p.add_argument("--extra-llm-kwargs-json", default="")
    p.add_argument("--tester", default=os.getenv("USER", "unknown"))
    p.add_argument("--hardware", default="Ascend 910B")
    p.add_argument("--collect-npu-smi", action="store_true")
    p.add_argument("--use-tqdm", action="store_true")
    p.add_argument("--print-output", action="store_true")
    p.add_argument("--baseline-summary-json", default="", help="use existing baseline summary instead of running baseline")
    p.add_argument("--spec-summary-json", default="", help="use existing spec summary instead of running spec")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--skip-spec", action="store_true")
    p.add_argument("--output-root", default="")
    p.add_argument("--excel-name", default="performance_metrics.xlsx")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.target_model and not args.baseline_summary_json and not args.spec_summary_json:
        raise SystemExit("--target-model is required unless results are supplied by summary json")
    if not args.skip_spec and not args.spec_summary_json and not args.speculative_config_json and not args.draft_model:
        raise SystemExit("--draft-model is required for spec unless --speculative-config-json or --spec-summary-json is provided")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    output_root = Path(args.output_root or f"bench_results/metrics_excel_{timestamp}")
    output_root.mkdir(parents=True, exist_ok=True)
    results: List[BenchResult] = []
    if not args.skip_baseline:
        results.append(run_one_mode("baseline", args, output_root / "baseline"))
    if not args.skip_spec:
        results.append(run_one_mode("spec", args, output_root / f"spec_k{args.num_speculative_tokens}"))
    rows = build_rows(results)
    xlsx_path = output_root / args.excel_name
    write_xlsx(xlsx_path, rows)
    print("\n========== METRICS EXCEL ==========")
    for row in rows:
        print(
            f"{row.get('test_object')}: "
            f"output_tokens_per_s={row.get('output_tokens_per_s')}, "
            f"avg_ms_per_output_token={row.get('avg_ms_per_output_token')}, "
            f"num_speculative_tokens={row.get('num_speculative_tokens')}, "
            f"spec_acceptance_rate={row.get('spec_acceptance_rate')}"
        )
    print(f"\nexcel: {xlsx_path}")


if __name__ == "__main__":
    main()
