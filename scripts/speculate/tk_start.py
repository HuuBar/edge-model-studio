import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np

from token_recycling import TokenRecycling


def convert_turns_to_messages(turns: List[str], system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
    """
    将原生 turns 列表自适应转换为标准的 ChatML 消息流。
    不仅兼容单轮，也优雅支持了多轮 [User, Assistant, User...] 交叉对话格式。
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
        
    for idx, content in enumerate(turns):
        # 奇数步为 User，偶数步为 Assistant (基 0 索引)
        role = "user" if idx % 2 == 0 else "assistant"
        messages.append({"role": role, "content": content})
    return messages


def run_one_chat_sample(
    token_recycle: TokenRecycling,
    turns: List[str],
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 1024,
    silent: bool = True,
    save_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    单样本推理流水线。
    增加上下文越界防御，改用高精度 perf_counter 计时。
    """
    try:
        messages = convert_turns_to_messages(turns, system_prompt)

        # 1. 渲染模版，结尾自动补全助理启始标记
        chat_prompt = token_recycle.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # 2. 预计算上下文长度并做防御性截断/丢弃检查
        input_ids = token_recycle.tokenizer(
            chat_prompt, return_tensors="pt"
        ).input_ids.to(token_recycle.device)
        prompt_len = input_ids.shape[-1]

        # 边界卡点：防止某些超长样本在端侧触发硬件极限导致 OOM
        max_context_limit = getattr(token_recycle.model.config, "max_position_embeddings", 4096)
        if prompt_len + max_new_tokens > max_context_limit:
            print(f"[WARN] Sample skipped to avoid OOM. Context length ({prompt_len}) + New tokens ({max_new_tokens}) exceeds limit ({max_context_limit}).")
            return None

        # 3. 核心推理
        torch_start = time.perf_counter()
        outputs = token_recycle.generate(
            chat_prompt,
            max_new_tokens=max_new_tokens,
            silent=silent,
            stop_on_eos=True,
            save_dir=save_dir,
        )
        torch_end = time.perf_counter()

        # 4. 指标统计与解码
        generated_ids = outputs.output_ids[0]
        total_len = generated_ids.shape[-1]
        new_tokens = max(total_len - prompt_len, 0)

        result_text = token_recycle.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        latency = torch_end - torch_start
        tokens_per_second = new_tokens / latency if latency > 0 and new_tokens > 0 else 0.0

        return {
            "answer": result_text,
            "latency": latency,
            "prompt_len": int(prompt_len),
            "new_tokens": int(new_tokens),
            "tokens_per_second": tokens_per_second,
            "total_steps": int(outputs.total_steps),
            "mean_accepted_tokens": float(outputs.mean_accepted_tokens),
            "prompt_tokens_per_sec": float(outputs.prompt_tokens_per_sec),
            "generation_tokens_per_sec": float(outputs.generation_tokens_per_sec),
        }
    except Exception as e:
        print(f"[ERROR] Inference engine encountered an exception: {e}", file=sys.stderr)
        return None


def run_benchmark_jsonl(
    model_name_or_path: str,
    input_jsonl_path: str,
    output_json_path: Optional[str] = None,
    max_new_tokens: int = 1024,
    save_dir: Optional[str] = None,
):
    """
    批量压测主控循环。具备断点防御机制，支持大样本流式双写。
    """
    input_path = Path(input_jsonl_path)
    if not input_path.exists():
        print(f"[CRITICAL] Benchmark input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Initializing Core Engine: {model_name_or_path}")
    token_recycle = TokenRecycling.from_pretrained(model_name_or_path)

    # 建立流式实时追写文件，防止中途断电或杀进程导致全盘皆输
    results = []
    backup_path = None
    if output_json_path:
        out_file = Path(output_json_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        backup_path = out_file.with_suffix(".working.jsonl")
        print(f"[INFO] Live caching progress into: {backup_path}")

    print("[INFO] Starting pipeline execution...")
    wall_clock_start = time.perf_counter()

    metrics_pool = {
        "latency": [], "new_tokens": [], "tps": [],
        "mat": [], "prompt_tps": [], "gen_tps": []
    }

    with input_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Line {line_idx} is corrupted JSON syntax. Skipping.")
                continue

            turns = item.get("turns", [])
            if not turns:
                print(f"[WARN] Line {line_idx}: Empty 'turns' block. Skipping.")
                continue

            question_id = item.get("question_id", line_idx)
            category = item.get("category", "generic")

            # 触发单样本处理（自适应处理多轮/单轮）
            res = run_one_chat_sample(
                token_recycle,
                turns=turns,
                system_prompt=None,
                max_new_tokens=max_new_tokens,
                silent=True,
                save_dir=save_dir,
            )

            if res is None:
                continue

            # 全量压测指标回收
            metrics_pool["latency"].append(res["latency"])
            metrics_pool["new_tokens"].append(res["new_tokens"])
            metrics_pool["tps"].append(res["tokens_per_second"])
            metrics_pool["mat"].append(res["mean_accepted_tokens"])
            metrics_pool["prompt_tps"].append(res["prompt_tokens_per_sec"])
            metrics_pool["gen_tps"].append(res["generation_tokens_per_sec"])

            out_item = {
                "question_id": question_id,
                "category": category,
                "turns": turns,
                **res
            }
            results.append(out_item)

            # 增量安全写入，规避大批量压测时闪退风险
            if backup_path:
                with backup_path.open("a", encoding="utf-8") as bf:
                    bf.write(json.dumps(out_item, ensure_ascii=False) + "\n")

            print(
                f"[SAMPLE] #{len(results)} | qid={question_id} | "
                f"Time={res['latency']:.2f}s | Gen={res['new_tokens']}toks | "
                f"TPS={res['tokens_per_second']:.1f} | MAT={res['mean_accepted_tokens']:.2f} | "
                f"P_TPS={res['prompt_tokens_per_sec']:.1f} | G_TPS={res['generation_tokens_per_sec']:.1f}"
            )

    wall_clock_end = time.perf_counter()
    total_samples = len(results)

    if total_samples == 0:
        print("[CRITICAL] No valid data point extracted. Pipeline aborted.", file=sys.stderr)
        return

    # 5. 高维度分位数聚合输出
    lats = np.array(metrics_pool["latency"])
    toks = np.array(metrics_pool["new_tokens"])
    tpss = np.array(metrics_pool["tps"])
    mats = np.array(metrics_pool["mat"])
    p_tps = np.array(metrics_pool["prompt_tps"])
    g_tps = np.array(metrics_pool["gen_tps"])

    pure_inference_time = np.sum(lats)
    total_gen_tokens = np.sum(toks)
    e2e_total_time = wall_clock_end - wall_clock_start

    print("\n" + "=" * 90)
    print(f"Benchmark Analytics Summary (Total Valid Samples: {total_samples})")
    print("-" * 90)
    print(f"  E2E Wall-Clock Run Time     : {e2e_total_time:.2f} s")
    print(f"  Pure Engine Inference Time  : {pure_inference_time:.2f} s")
    print(f"  Aggregated Generated Tokens : {total_gen_tokens} tokens")
    print(f"  Pure Engine Total Throughput: {total_gen_tokens / pure_inference_time:.2f} tokens/s")
    print("-" * 90)
    print(f"{'Performance Metric':<30} | {'Average':<12} | {'P50 (Median)':<12} | {'P95':<12}")
    print("-" * 90)
    print(f"{'Latency per Sample (s)':<30} | {np.mean(lats):<12.3f} | {np.percentile(lats, 50):<12.3f} | {np.percentile(lats, 95):<12.3f}")
    print(f"{'New Tokens per Sample':<30} | {np.mean(toks):<12.1f} | {np.percentile(toks, 50):<12.1f} | {np.percentile(toks, 95):<12.1f}")
    print(f"{'Generation Speed (tokens/s)':<30} | {np.mean(tpss):<12.2f} | {np.percentile(tpss, 50):<12.2f} | {np.percentile(tpss, 95):<12.2f}")
    print(f"{'Mean Accepted Tokens (MAT)':<30} | {np.mean(mats):<12.2f} | {np.percentile(mats, 50):<12.2f} | {np.percentile(mats, 95):<12.2f}")
    print(f"{'Prefill Phase Speed (tps)':<30} | {np.mean(p_tps):<12.2f} | {np.percentile(p_tps, 50):<12.2f} | {np.percentile(p_tps, 95):<12.2f}")
    print(f"{'Decode Phase Speed (tps)':<30} | {np.mean(g_tps):<12.2f} | {np.percentile(g_tps, 50):<12.2f} | {np.percentile(g_tps, 95):<12.2f}")
    print("=" * 90)

    # 保存最终结构化的标准 JSON 文件，并清理临时增量缓存
    if output_json_path:
        final_report = Path(output_json_path)
        with final_report.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[SUCCESS] Final consolidated log generated: {final_report}")
        
        if backup_path and backup_path.exists():
            os.remove(backup_path)


def main():
    parser = argparse.ArgumentParser(description="TokenRecycling Multi-Turn Performance Profiler")
    parser.add_argument("--model_path", type=str, required=True, help="HF hub stub or directory path")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to evaluation jsonl target")
    parser.add_argument("--output_json", type=str, default="./benchmark_report.json", help="Final static analysis report file")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Budget limitation for new tokens generation")
    parser.add_argument("--save_dir", type=str, default=None, help="Inference artifacts store target")
    args = parser.parse_args()

    run_benchmark_jsonl(
        model_name_or_path=args.model_path,
        input_jsonl_path=args.input_jsonl,
        output_json_path=args.output_json,
        max_new_tokens=args.max_new_tokens,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()