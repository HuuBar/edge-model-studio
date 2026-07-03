"""
example:
python scripts/regenerate_train_data.py \
     --model edgemodelstudio_inference \
    --concurrency 32 \
        --max-tokens 4096 \
        --server-address 10.136.183.8:7088 \
        --temperature 0 \
        --input-file-path ./cache/dataset/ultrachat_train.jsonl \
        --output-file-path ./cache/dataset/ultrachat_train_regen_7b.jsonl
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from openai import OpenAI
from tqdm import tqdm


# =========================================================
# Argument parsing
# =========================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Re-generate training data using vLLM model server"
    )

    # model
    parser.add_argument("--model", type=str, required=True)

    parser.add_argument("--is-reasoning-model", action="store_true")
    parser.add_argument("--is-gpt-oss", action="store_true")

    # sampling
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=None, dest="top_p")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=1024)

    # performance
    parser.add_argument("--concurrency", type=int, default=64)

    # data
    parser.add_argument("--input-file-path", type=str, required=True)
    parser.add_argument("--output-file-path", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=None)

    # vLLM server (支持多个，语义与 sglang 一致)
    parser.add_argument(
        "--server-address",
        type=str,
        nargs="+",
        required=True,
        help="vLLM server address(es), e.g. 127.0.0.1:7088",
    )

    return parser.parse_args()


# =========================================================
# Utils
# =========================================================

def get_random_reasoning_effort() -> str:
    return random.choices(
        ["low", "medium", "high"],
        weights=[4, 4, 2],
        k=1,
    )[0]


def compute_context_length(conversations: List[Dict[str, Any]]) -> int:
    length = 0
    for msg in conversations:
        content = msg.get("content")
        if isinstance(content, str):
            length += len(content.split())
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        length += len(text.split())
    return length


# =========================================================
# Build chat.completions kwargs (对齐 sglang)
# =========================================================

def build_query_kwargs(args, messages, max_tokens=None):
    effective_max_tokens = max_tokens or args.max_tokens

    kwargs = dict(
        model=args.model,
        messages=messages,
        max_tokens=effective_max_tokens,
        temperature=args.temperature,
    )

    if args.top_p is not None:
        kwargs["top_p"] = args.top_p

    if args.repetition_penalty is not None:
        kwargs["presence_penalty"] = args.repetition_penalty

    extra_body = {}
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k

    if args.is_gpt_oss:
        # vLLM 会忽略，但为了行为对齐保留
        kwargs["reasoning_effort"] = get_random_reasoning_effort()

    if extra_body:
        kwargs["extra_body"] = extra_body

    return kwargs


# =========================================================
# Single sample regeneration (vLLM)
# =========================================================

def call_vllm(
    args,
    server_address: str,
    data: Dict[str, Any],
    max_tokens=None,
):
    client = OpenAI(
        base_url=f"http://{server_address}/v1",
        api_key="EMPTY",
    )

    messages = data["conversations"]
    regenerated_messages = []

    if messages[0]["role"] == "assistant":
        data["status"] = "error"
        data["error"] = "Data starts with an assistant message"
        return data

    for msg in messages:
        if msg["role"] == "system":
            regenerated_messages.append(msg)

        elif msg["role"] == "assistant":
            continue

        elif msg["role"] == "user":
            regenerated_messages.append(msg)

            query_kwargs = build_query_kwargs(
                args, regenerated_messages, max_tokens
            )

            try:
                resp = client.chat.completions.create(**query_kwargs)
            except Exception as e:
                data["status"] = "error"
                data["error"] = str(e)
                return data

            assistant_content = resp.choices[0].message.content

            resp_msg = {
                "role": "assistant",
                "content": assistant_content,
            }

            # reasoning placeholder（vLLM 目前不会返回）
            if args.is_reasoning_model:
                resp_msg["thinking"] = None

            regenerated_messages.append(resp_msg)

        else:
            data["status"] = "error"
            data["error"] = f"Invalid role: {msg['role']}"
            return data

    data["conversations"] = regenerated_messages
    data["status"] = "success"
    return data


# =========================================================
# Main
# =========================================================

def main():
    args = parse_arguments()

    print("Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Servers: {args.server_address}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max tokens: {args.max_tokens}")
    print("-" * 60)

    total_lines = sum(1 for _ in open(args.input_file_path))
    error_path = args.output_file_path.replace(".jsonl", "_error.jsonl")

    # test servers
    valid_servers = []
    for addr in args.server_address:
        try:
            test_client = OpenAI(
                base_url=f"http://{addr}/v1",
                api_key="EMPTY",
            )
            test_client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1,
                temperature=0.0,
            )
            valid_servers.append(addr)
        except Exception as e:
            print(f"Server {addr} not available: {e}")

    if not valid_servers:
        raise RuntimeError("No valid vLLM servers available")

    print(f"Using servers: {valid_servers}")
    print("-" * 60)

    success, error_count = 0, 0
    ctx_sum, ctx_min, ctx_max = 0, None, 0

    with (
        open(args.input_file_path, "r") as fin,
        open(args.output_file_path, "w") as fout,
        open(error_path, "w") as ferr,
    ):
        executor = ThreadPoolExecutor(
            max_workers=args.concurrency * len(valid_servers)
        )
        waiting_queue = {addr: [] for addr in valid_servers}
        pbar = tqdm(total=total_lines, desc="Processing")

        server_idx = 0

        for line in fin:
            if args.num_samples and (success + error_count) >= args.num_samples:
                break

            data = json.loads(line.strip())
            server = valid_servers[server_idx]
            server_idx = (server_idx + 1) % len(valid_servers)

            while len(waiting_queue[server]) >= args.concurrency:
                for fut in waiting_queue[server]:
                    if fut.done():
                        result = fut.result()
                        waiting_queue[server].remove(fut)

                        if result["status"] == "success":
                            ctx = compute_context_length(result["conversations"])
                            ctx_sum += ctx
                            ctx_min = ctx if ctx_min is None else min(ctx_min, ctx)
                            ctx_max = max(ctx_max, ctx)
                            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                            success += 1
                        else:
                            ferr.write(json.dumps(result, ensure_ascii=False) + "\n")
                            error_count += 1
                        pbar.update(1)
                        break

            fut = executor.submit(call_vllm, args, server, data)
            waiting_queue[server].append(fut)

        # flush remaining
        for server, futures in waiting_queue.items():
            for fut in futures:
                result = fut.result()
                if result["status"] == "success":
                    ctx = compute_context_length(result["conversations"])
                    ctx_sum += ctx
                    ctx_min = ctx if ctx_min is None else min(ctx_min, ctx)
                    ctx_max = max(ctx_max, ctx)
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    success += 1
                else:
                    ferr.write(json.dumps(result, ensure_ascii=False) + "\n")
                    error_count += 1
                pbar.update(1)

        pbar.close()

    print("\nDone.")
    print(f"Success: {success}, Errors: {error_count}")
    if success:
        print(f"Context length avg: {ctx_sum / success:.2f}")
        print(f"Min: {ctx_min}, Max: {ctx_max}")


if __name__ == "__main__":
    main()
