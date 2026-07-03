#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run one offline vLLM benchmark case.

This script supports two modes:
  1. baseline: target model only
  2. spec:     target model + EAGLE/EAGLE3/draft model speculative decoding

The offline metric is end-to-end LLM.generate latency. It does not report real
TTFT/ITL because no token stream is exposed by LLM.generate.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from bench_common import (
    append_csv,
    chunked,
    collect_spec_metrics,
    diff_spec_metrics,
    environment_info,
    load_prompt_token_ids,
    make_vllm_prompts,
    output_text_preview,
    output_token_count,
    summarize_offline_run,
    write_json,
)


def add_optional(d: Dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        d[key] = value


def parse_extra_json(raw: Optional[str], name: str) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON for {name}: {exc}") from exc
    if not isinstance(obj, dict):
        raise SystemExit(f"{name} must be a JSON object")
    return obj


def build_speculative_config(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    if args.mode != "spec":
        return None

    if args.speculative_config_json:
        cfg = parse_extra_json(args.speculative_config_json, "--speculative-config-json")
    else:
        if not args.draft_model:
            raise SystemExit("--draft-model is required when --mode spec")
        if args.num_speculative_tokens <= 0:
            raise SystemExit("--num-speculative-tokens must be positive when --mode spec")
        cfg = {
            "method": args.spec_method,
            "model": args.draft_model,
            "num_speculative_tokens": args.num_speculative_tokens,
        }

    if args.draft_tensor_parallel_size > 0:
        cfg.setdefault("draft_tensor_parallel_size", args.draft_tensor_parallel_size)
    if args.disable_padded_drafter_batch:
        cfg.setdefault("disable_padded_drafter_batch", True)
    if args.parallel_drafting:
        cfg.setdefault("parallel_drafting", True)

    extra = parse_extra_json(args.extra_speculative_config_json, "--extra-speculative-config-json")
    cfg.update(extra)
    return cfg


def build_llm_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    llm_kwargs: Dict[str, Any] = {
        "model": args.target_model,
        "tokenizer": args.tokenizer or args.target_model,
        "trust_remote_code": args.trust_remote_code,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
    }

    add_optional(llm_kwargs, "max_model_len", args.max_model_len)
    add_optional(llm_kwargs, "gpu_memory_utilization", args.gpu_memory_utilization)
    add_optional(llm_kwargs, "max_num_seqs", args.max_num_seqs)
    add_optional(llm_kwargs, "max_num_batched_tokens", args.max_num_batched_tokens)

    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    if args.enable_chunked_prefill:
        llm_kwargs["enable_chunked_prefill"] = True
    if args.disable_log_stats:
        llm_kwargs["disable_log_stats"] = True
    else:
        # Needed by some versions to retain metrics; harmless when accepted.
        llm_kwargs["disable_log_stats"] = False

    speculative_config = build_speculative_config(args)
    if speculative_config:
        llm_kwargs["speculative_config"] = speculative_config

    extra = parse_extra_json(args.extra_llm_kwargs_json, "--extra-llm-kwargs-json")
    llm_kwargs.update(extra)
    return llm_kwargs


def build_sampling_params(args: argparse.Namespace) -> SamplingParams:
    # SamplingParams changes across versions. Only pass parameters supported by
    # the installed version.
    wanted: Dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.output_len,
        "ignore_eos": args.ignore_eos,
    }
    if args.top_k is not None:
        wanted["top_k"] = args.top_k
    if args.seed is not None:
        wanted["seed"] = args.seed
    if args.skip_special_tokens is not None:
        wanted["skip_special_tokens"] = args.skip_special_tokens

    try:
        sig = inspect.signature(SamplingParams)
        supported = set(sig.parameters.keys())
        kwargs = {k: v for k, v in wanted.items() if k in supported}
    except Exception:
        kwargs = wanted
    return SamplingParams(**kwargs)


def call_generate(llm: LLM, prompts: Sequence[Any], sampling_params: SamplingParams, use_tqdm: bool) -> List[Any]:
    try:
        return llm.generate(prompts, sampling_params=sampling_params, use_tqdm=use_tqdm)
    except TypeError:
        return llm.generate(prompts, sampling_params=sampling_params)


def run_batches(
    *,
    llm: LLM,
    vllm_prompts: Sequence[Any],
    prompt_token_ids: Sequence[Sequence[int]],
    prompt_sources: Sequence[str],
    sampling_params: SamplingParams,
    batch_size: int,
    use_tqdm: bool,
    print_output: bool,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
    batch_records: List[Dict[str, Any]] = []
    request_records: List[Dict[str, Any]] = []

    measured_start = time.perf_counter()
    global_request_idx = 0

    for batch_idx, idxs in enumerate(chunked(list(range(len(vllm_prompts))), batch_size)):
        batch_prompts = [vllm_prompts[i] for i in idxs]
        t0 = time.perf_counter()
        try:
            outputs = call_generate(llm, batch_prompts, sampling_params, use_tqdm=use_tqdm)
            success = True
            err = None
        except Exception as exc:
            outputs = []
            success = False
            err = repr(exc)
        t1 = time.perf_counter()
        latency_s = t1 - t0

        batch_output_tokens = 0
        if success:
            for local_idx, output in enumerate(outputs):
                src_idx = idxs[local_idx]
                out_tokens = output_token_count(output)
                batch_output_tokens += out_tokens
                preview = output_text_preview(output) if print_output else ""
                record = {
                    "request_idx": global_request_idx,
                    "source_idx": src_idx,
                    "batch_idx": batch_idx,
                    "success": True,
                    "error": None,
                    "prompt_source": prompt_sources[src_idx],
                    "input_tokens": len(prompt_token_ids[src_idx]),
                    "output_tokens": out_tokens,
                    "batch_latency_s": latency_s,
                    "output_preview": preview,
                }
                request_records.append(record)
                global_request_idx += 1
        else:
            for src_idx in idxs:
                record = {
                    "request_idx": global_request_idx,
                    "source_idx": src_idx,
                    "batch_idx": batch_idx,
                    "success": False,
                    "error": err,
                    "prompt_source": prompt_sources[src_idx],
                    "input_tokens": len(prompt_token_ids[src_idx]),
                    "output_tokens": 0,
                    "batch_latency_s": latency_s,
                    "output_preview": "",
                }
                request_records.append(record)
                global_request_idx += 1

        batch_records.append(
            {
                "batch_idx": batch_idx,
                "batch_size": len(idxs),
                "latency_s": latency_s,
                "success": success,
                "error": err,
                "input_tokens": sum(len(prompt_token_ids[i]) for i in idxs),
                "output_tokens": batch_output_tokens,
                "output_tokens_per_s": batch_output_tokens / latency_s if latency_s > 0 else None,
            }
        )

    measured_wall_time_s = time.perf_counter() - measured_start
    return batch_records, request_records, measured_wall_time_s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline vLLM baseline/speculative benchmark")

    p.add_argument("--mode", choices=["baseline", "spec"], required=True)
    p.add_argument("--target-model", required=True, help="target model path/name")
    p.add_argument("--draft-model", default="", help="EAGLE/EAGLE3/draft model path/name")
    p.add_argument("--tokenizer", default="", help="default: target model")

    p.add_argument("--spec-method", default="eagle3", choices=["eagle", "eagle3", "draft_model", "ngram", "mtp", "suffix", "dflash"])
    p.add_argument("--num-speculative-tokens", type=int, default=0)
    p.add_argument("--draft-tensor-parallel-size", type=int, default=0)
    p.add_argument("--speculative-config-json", default="", help="override full speculative_config JSON")
    p.add_argument("--extra-speculative-config-json", default="", help="merge extra keys into speculative_config")
    p.add_argument("--disable-padded-drafter-batch", action="store_true")
    p.add_argument("--parallel-drafting", action="store_true")

    p.add_argument("--num-prompts", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=1, help="offline micro-batch size; use this instead of online concurrency")
    p.add_argument("--warmup-batches", type=int, default=1)
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--prompt-file", default="", help="txt/jsonl file; jsonl field prompt/text/content")
    p.add_argument("--normalize-file-prompts", action="store_true", help="truncate/repeat file prompts to exactly --input-len tokens")
    p.add_argument("--synthetic-language", choices=["zh", "en"], default="zh")

    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--ignore-eos", action="store_true", help="recommended for fixed output length")
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
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--extra-llm-kwargs-json", default="", help="extra LLM kwargs as JSON object")

    p.add_argument("--run-id", default="")
    p.add_argument("--case-id", default="")
    p.add_argument("--tester", default=os.getenv("USER", "unknown"))
    p.add_argument("--hardware", default="Ascend 910B")
    p.add_argument("--output-dir", default="bench_results")
    p.add_argument("--collect-npu-smi", action="store_true")
    p.add_argument("--use-tqdm", action="store_true")
    p.add_argument("--print-output", action="store_true")

    args = p.parse_args()

    if not args.run_id:
        args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    if not args.case_id:
        args.case_id = (
            f"{args.mode}_in{args.input_len}_out{args.output_len}"
            f"_bs{args.batch_size}_k{args.num_speculative_tokens if args.mode == 'spec' else 0}"
        )
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("========== vLLM offline benchmark ==========")
    print(f"mode: {args.mode}")
    print(f"target_model: {args.target_model}")
    if args.mode == "spec":
        print(f"draft_model: {args.draft_model}")
        print(f"spec_method: {args.spec_method}")
        print(f"num_speculative_tokens: {args.num_speculative_tokens}")
    print(f"num_prompts: {args.num_prompts}, batch_size: {args.batch_size}")
    print(f"input_len: {args.input_len}, output_len: {args.output_len}")

    tokenizer_path = args.tokenizer or args.target_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    prompt_token_ids, prompt_sources = load_prompt_token_ids(
        tokenizer=tokenizer,
        num_prompts=args.num_prompts,
        input_len=args.input_len,
        prompt_file=args.prompt_file or None,
        synthetic_language=args.synthetic_language,
        normalize_file_prompts=args.normalize_file_prompts,
    )
    vllm_prompts = make_vllm_prompts(prompt_token_ids, use_token_ids=True)

    sampling_params = build_sampling_params(args)
    llm_kwargs = build_llm_kwargs(args)

    print("\n[INFO] LLM kwargs:")
    printable_kwargs = dict(llm_kwargs)
    if "speculative_config" in printable_kwargs:
        printable_kwargs["speculative_config"] = dict(printable_kwargs["speculative_config"])
    print(json.dumps(printable_kwargs, ensure_ascii=False, indent=2, default=str))

    print("\n[INFO] loading LLM ...")
    load_t0 = time.perf_counter()
    llm = LLM(**llm_kwargs)
    load_time_s = time.perf_counter() - load_t0
    print(f"[INFO] load_time_s: {load_time_s:.3f}")

    if args.warmup_batches > 0:
        warmup_n = min(args.num_prompts, args.warmup_batches * args.batch_size)
        print(f"[INFO] warmup prompts: {warmup_n}")
        warmup_prompts = vllm_prompts[:warmup_n]
        try:
            call_generate(llm, warmup_prompts, sampling_params, use_tqdm=args.use_tqdm)
        except Exception as exc:
            raise RuntimeError(f"warmup failed: {exc!r}") from exc

    metrics_before = collect_spec_metrics(llm, args.num_speculative_tokens)

    print("[INFO] running measured batches ...")
    batch_records, request_records, measured_wall_time_s = run_batches(
        llm=llm,
        vllm_prompts=vllm_prompts,
        prompt_token_ids=prompt_token_ids,
        prompt_sources=prompt_sources,
        sampling_params=sampling_params,
        batch_size=args.batch_size,
        use_tqdm=args.use_tqdm,
        print_output=args.print_output,
    )
    metrics_after = collect_spec_metrics(llm, args.num_speculative_tokens)
    spec_metrics_delta = diff_spec_metrics(metrics_before, metrics_after)

    env_info = environment_info(collect_npu_smi=args.collect_npu_smi)
    summary = summarize_offline_run(
        args=args,
        run_id=args.run_id,
        case_id=args.case_id,
        load_time_s=load_time_s,
        measured_wall_time_s=measured_wall_time_s,
        batch_records=batch_records,
        request_records=request_records,
        spec_metrics_delta=spec_metrics_delta,
        env_info=env_info,
    )

    stem = f"{args.case_id}_{args.run_id}"
    summary_path = output_dir / f"{stem}.summary.json"
    requests_path = output_dir / f"{stem}.requests.jsonl"
    batches_path = output_dir / f"{stem}.batches.jsonl"
    csv_path = output_dir / "summary.csv"

    write_json(summary_path, {"summary": summary, "env": env_info, "llm_kwargs": printable_kwargs})
    with open(requests_path, "w", encoding="utf-8") as f:
        for r in request_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(batches_path, "w", encoding="utf-8") as f:
        for r in batch_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    append_csv(csv_path, summary)

    print("\n========== SUMMARY ==========")
    for key in [
        "run_id",
        "case_id",
        "mode",
        "bench_mode",
        "num_prompts",
        "batch_size",
        "input_len_target",
        "output_len_target",
        "success_count",
        "failed_count",
        "measured_wall_time_s",
        "total_input_tokens",
        "total_output_tokens",
        "output_tokens_per_s",
        "total_tokens_per_s",
        "avg_ms_per_output_token",
        "batch_latency_avg_ms",
        "batch_latency_p50_ms",
        "batch_latency_p90_ms",
        "batch_latency_p99_ms",
        "spec_metrics_available",
        "spec_acceptance_rate",
        "spec_accepted_tokens_per_draft",
        "spec_mean_acceptance_length_including_bonus",
    ]:
        print(f"{key}: {summary.get(key)}")

    print("\n[INFO] outputs:")
    print(f"summary : {summary_path}")
    print(f"requests: {requests_path}")
    print(f"batches : {batches_path}")
    print(f"csv     : {csv_path}")


if __name__ == "__main__":
    main()
