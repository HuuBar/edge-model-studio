#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compare one baseline offline benchmark result with one speculative result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from bench_common import append_csv, load_json_maybe, now_iso, write_json


def ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den == 0:
        return None
    return num / den


def load_summary(path: str) -> Dict[str, Any]:
    obj = load_json_maybe(path)
    if obj is None:
        raise RuntimeError(f"cannot load summary from {path}")
    return obj


def compare(baseline: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "compare_date": now_iso(),
        "baseline_run_id": baseline.get("run_id"),
        "spec_run_id": spec.get("run_id"),
        "baseline_case_id": baseline.get("case_id"),
        "spec_case_id": spec.get("case_id"),
        "target_model": spec.get("target_model") or baseline.get("target_model"),
        "draft_model": spec.get("draft_model"),
        "spec_method": spec.get("spec_method"),
        "num_speculative_tokens": spec.get("num_speculative_tokens"),
        "input_len_target": spec.get("input_len_target"),
        "output_len_target": spec.get("output_len_target"),
        "batch_size": spec.get("batch_size"),
        "num_prompts": spec.get("num_prompts"),
        "temperature": spec.get("temperature"),
        "top_p": spec.get("top_p"),
        "top_k": spec.get("top_k"),
        "ignore_eos": spec.get("ignore_eos"),

        "baseline_output_tokens_per_s": baseline.get("output_tokens_per_s"),
        "spec_output_tokens_per_s": spec.get("output_tokens_per_s"),
        "output_tps_speedup": ratio(spec.get("output_tokens_per_s"), baseline.get("output_tokens_per_s")),

        "baseline_total_tokens_per_s": baseline.get("total_tokens_per_s"),
        "spec_total_tokens_per_s": spec.get("total_tokens_per_s"),
        "total_tps_speedup": ratio(spec.get("total_tokens_per_s"), baseline.get("total_tokens_per_s")),

        "baseline_avg_ms_per_output_token": baseline.get("avg_ms_per_output_token"),
        "spec_avg_ms_per_output_token": spec.get("avg_ms_per_output_token"),
        "ms_per_output_token_speedup": ratio(
            baseline.get("avg_ms_per_output_token"), spec.get("avg_ms_per_output_token")
        ),

        "baseline_batch_latency_avg_ms": baseline.get("batch_latency_avg_ms"),
        "spec_batch_latency_avg_ms": spec.get("batch_latency_avg_ms"),
        "batch_latency_speedup": ratio(
            baseline.get("batch_latency_avg_ms"), spec.get("batch_latency_avg_ms")
        ),

        "baseline_wall_time_s": baseline.get("measured_wall_time_s"),
        "spec_wall_time_s": spec.get("measured_wall_time_s"),
        "wall_time_speedup": ratio(baseline.get("measured_wall_time_s"), spec.get("measured_wall_time_s")),

        "spec_metrics_available": spec.get("spec_metrics_available"),
        "spec_num_drafts": spec.get("spec_num_drafts"),
        "spec_num_draft_tokens": spec.get("spec_num_draft_tokens"),
        "spec_num_accepted_tokens": spec.get("spec_num_accepted_tokens"),
        "spec_acceptance_rate": spec.get("spec_acceptance_rate"),
        "spec_accepted_tokens_per_draft": spec.get("spec_accepted_tokens_per_draft"),
        "spec_mean_acceptance_length_including_bonus": spec.get(
            "spec_mean_acceptance_length_including_bonus"
        ),
        "spec_accepted_per_pos_rates": spec.get("spec_accepted_per_pos_rates"),
    }

    comparable_keys = [
        "target_model",
        "input_len_target",
        "output_len_target",
        "batch_size",
        "num_prompts",
        "temperature",
        "top_p",
        "top_k",
        "ignore_eos",
    ]
    mismatches = {}
    for key in comparable_keys:
        if baseline.get(key) != spec.get(key):
            mismatches[key] = {"baseline": baseline.get(key), "spec": spec.get(key)}
    out["config_mismatches"] = mismatches
    out["is_strictly_comparable"] = len(mismatches) == 0
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, help="baseline *.summary.json")
    p.add_argument("--spec", required=True, help="spec *.summary.json")
    p.add_argument("--output-dir", default="bench_results")
    p.add_argument("--name", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_summary(args.baseline)
    spec = load_summary(args.spec)
    result = compare(baseline, spec)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"compare_{baseline.get('run_id')}_vs_{spec.get('run_id')}"
    json_path = out_dir / f"{name}.json"
    csv_path = out_dir / "compare.csv"

    write_json(json_path, {"comparison": result})
    append_csv(csv_path, result)

    print("========== COMPARE ==========")
    for key in [
        "is_strictly_comparable",
        "config_mismatches",
        "output_tps_speedup",
        "total_tps_speedup",
        "ms_per_output_token_speedup",
        "batch_latency_speedup",
        "wall_time_speedup",
        "spec_acceptance_rate",
        "spec_accepted_tokens_per_draft",
        "spec_mean_acceptance_length_including_bonus",
    ]:
        print(f"{key}: {result.get(key)}")
    print(f"\ncomparison json: {json_path}")
    print(f"comparison csv : {csv_path}")


if __name__ == "__main__":
    main()
