#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare latest baseline/spec offline benchmark summaries.

Usage:
  python compare_latest.py \
    --baseline-dir bench_results/synthetic_baseline_in512_out256_bs1 \
    --spec-dir bench_results/synthetic_spec_in512_out256_bs1_k4 \
    --output-dir bench_results/synthetic_compare_in512_out256_bs1_k4
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bench_common import append_csv, write_json
from compare_runs import compare, load_summary


def latest_summary_json(result_dir: str) -> Path:
    d = Path(result_dir)
    if not d.exists():
        raise FileNotFoundError(f"directory does not exist: {d}")

    candidates = sorted(
        d.glob("*.summary.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no *.summary.json found in: {d}")

    return candidates[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-dir", required=True, help="directory containing baseline *.summary.json")
    p.add_argument("--spec-dir", required=True, help="directory containing spec *.summary.json")
    p.add_argument("--baseline-json", default="", help="optional explicit baseline summary json")
    p.add_argument("--spec-json", default="", help="optional explicit spec summary json")
    p.add_argument("--output-dir", default="bench_results/compare_latest")
    p.add_argument("--name", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    baseline_path = Path(args.baseline_json) if args.baseline_json else latest_summary_json(args.baseline_dir)
    spec_path = Path(args.spec_json) if args.spec_json else latest_summary_json(args.spec_dir)

    baseline = load_summary(str(baseline_path))
    spec = load_summary(str(spec_path))
    result = compare(baseline, spec)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    name = args.name or (
        f"compare_"
        f"{baseline.get('case_id', 'baseline')}_"
        f"vs_"
        f"{spec.get('case_id', 'spec')}"
    )

    json_path = out_dir / f"{name}.json"
    csv_path = out_dir / "compare.csv"

    payload = {
        "baseline_summary_json": str(baseline_path),
        "spec_summary_json": str(spec_path),
        "comparison": result,
    }
    write_json(json_path, payload)
    append_csv(csv_path, result)

    print("========== AUTO COMPARE LATEST ==========")
    print(f"baseline_json: {baseline_path}")
    print(f"spec_json    : {spec_path}")
    print()
    for key in [
        "is_strictly_comparable",
        "config_mismatches",
        "output_tps_speedup",
        "total_tps_speedup",
        "ms_per_output_token_speedup",
        "batch_latency_speedup",
        "wall_time_speedup",
        "spec_metrics_available",
        "spec_acceptance_rate",
        "spec_accepted_tokens_per_draft",
        "spec_mean_acceptance_length_including_bonus",
    ]:
        print(f"{key}: {result.get(key)}")

    print(f"\ncomparison json: {json_path}")
    print(f"comparison csv : {csv_path}")


if __name__ == "__main__":
    main()
