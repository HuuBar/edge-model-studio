#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run one offline baseline/spec benchmark pair and compare automatically.

This wrapper intentionally starts baseline and spec as separate Python processes.
That is safer on Ascend/NPU environments because model state, graph cache, and
device memory are less likely to pollute the next case.

Example:
  python run_pair_offline_bench.py \
    --target-model /../models/Qwen2.5-14B-Instruct \
    --draft-model /../vl_spec/speculators/scripts/checkpoints/checkpoint_best \
    --input-len 512 \
    --output-len 256 \
    --num-prompts 16 \
    --batch-size 1 \
    --num-speculative-tokens 4 \
    --hardware "Ascend 910B" \
    --tester "your_name" \
    --output-root bench_results/pair_in512_out256_bs1_k4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    print("\n========== RUN ==========")
    print(" ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def latest_summary_json(result_dir: Path) -> Path:
    candidates = sorted(
        result_dir.glob("*.summary.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no *.summary.json found in {result_dir}")
    return candidates[0]


def add_common_bench_args(cmd: list[str], args: argparse.Namespace) -> None:
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
        "--hardware", args.hardware,
        "--tester", args.tester,
    ]

    if args.prompt_file:
        cmd += ["--prompt-file", args.prompt_file]
    if args.normalize_file_prompts:
        cmd += ["--normalize-file-prompts"]
    if args.synthetic_language:
        cmd += ["--synthetic-language", args.synthetic_language]
    if args.ignore_eos:
        cmd += ["--ignore-eos"]
    if args.collect_npu_smi:
        cmd += ["--collect-npu-smi"]
    if args.use_tqdm:
        cmd += ["--use-tqdm"]
    if args.top_k is not None:
        cmd += ["--top-k", str(args.top_k)]
    if args.tokenizer:
        cmd += ["--tokenizer", args.tokenizer]
    if args.dtype:
        cmd += ["--dtype", args.dtype]
    if args.max_model_len is not None:
        cmd += ["--max-model-len", str(args.max_model_len)]
    if args.gpu_memory_utilization is not None:
        cmd += ["--gpu-memory-utilization", str(args.gpu_memory_utilization)]
    if args.max_num_seqs is not None:
        cmd += ["--max-num-seqs", str(args.max_num_seqs)]
    if args.max_num_batched_tokens is not None:
        cmd += ["--max-num-batched-tokens", str(args.max_num_batched_tokens)]
    if args.enforce_eager:
        cmd += ["--enforce-eager"]
    if args.enable_chunked_prefill:
        cmd += ["--enable-chunked-prefill"]
    if args.extra_llm_kwargs_json:
        cmd += ["--extra-llm-kwargs-json", args.extra_llm_kwargs_json]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--target-model", required=True)
    p.add_argument("--draft-model", required=True)
    p.add_argument("--tokenizer", default="")

    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--num-prompts", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup-batches", type=int, default=1)
    p.add_argument("--prompt-file", default="")
    p.add_argument("--normalize-file-prompts", action="store_true")
    p.add_argument("--synthetic-language", default="zh", choices=["zh", "en"])

    p.add_argument("--spec-method", default="eagle3")
    p.add_argument("--num-speculative-tokens", type=int, default=4)
    p.add_argument("--draft-tensor-parallel-size", type=int, default=1)
    p.add_argument("--extra-speculative-config-json", default="")
    p.add_argument("--speculative-config-json", default="")

    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--ignore-eos", action="store_true", default=True)

    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--gpu-memory-utilization", type=float, default=None)
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--max-num-batched-tokens", type=int, default=None)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--enable-chunked-prefill", action="store_true")
    p.add_argument("--extra-llm-kwargs-json", default="")

    p.add_argument("--hardware", default="Ascend 910B")
    p.add_argument("--tester", default="unknown")
    p.add_argument("--collect-npu-smi", action="store_true")
    p.add_argument("--use-tqdm", action="store_true")

    p.add_argument("--output-root", required=True)
    p.add_argument("--reuse-baseline", action="store_true", help="reuse newest baseline summary if baseline dir already has one")
    p.add_argument("--skip-spec", action="store_true", help="only run/reuse baseline")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    run_bench_py = script_dir / "run_offline_bench.py"
    compare_py = script_dir / "compare_runs.py"

    output_root = Path(args.output_root)
    baseline_dir = output_root / "baseline"
    spec_dir = output_root / f"spec_k{args.num_speculative_tokens}"
    compare_dir = output_root / "compare"

    baseline_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_baseline and list(baseline_dir.glob("*.summary.json")):
        print(f"[INFO] reuse baseline from {baseline_dir}")
    else:
        baseline_cmd = [
            sys.executable, str(run_bench_py),
            "--mode", "baseline",
            "--output-dir", str(baseline_dir),
        ]
        add_common_bench_args(baseline_cmd, args)
        run_cmd(baseline_cmd)

    if args.skip_spec:
        print("[INFO] skip spec by request")
        return

    spec_cmd = [
        sys.executable, str(run_bench_py),
        "--mode", "spec",
        "--draft-model", args.draft_model,
        "--spec-method", args.spec_method,
        "--num-speculative-tokens", str(args.num_speculative_tokens),
        "--draft-tensor-parallel-size", str(args.draft_tensor_parallel_size),
        "--output-dir", str(spec_dir),
    ]
    if args.extra_speculative_config_json:
        spec_cmd += ["--extra-speculative-config-json", args.extra_speculative_config_json]
    if args.speculative_config_json:
        spec_cmd += ["--speculative-config-json", args.speculative_config_json]

    add_common_bench_args(spec_cmd, args)
    run_cmd(spec_cmd)

    baseline_json = latest_summary_json(baseline_dir)
    spec_json = latest_summary_json(spec_dir)

    compare_cmd = [
        sys.executable, str(compare_py),
        "--baseline", str(baseline_json),
        "--spec", str(spec_json),
        "--output-dir", str(compare_dir),
        "--name", f"compare_in{args.input_len}_out{args.output_len}_bs{args.batch_size}_k{args.num_speculative_tokens}",
    ]
    run_cmd(compare_cmd)

    print("\n========== DONE ==========")
    print(f"baseline_dir: {baseline_dir}")
    print(f"spec_dir    : {spec_dir}")
    print(f"compare_dir : {compare_dir}")
    print(f"baseline_json: {baseline_json}")
    print(f"spec_json    : {spec_json}")


if __name__ == "__main__":
    main()
