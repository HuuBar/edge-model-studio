#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for bench_common module - utility and speculative metrics functions."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import bench_common as bc
import pytest


# ============================================================================
# Helper functions
# ============================================================================

def create_mock_metric(name: str, value: Any) -> MagicMock:
    """Create a mock metric object."""
    metric = MagicMock()
    metric.name = name
    metric.value = value
    return metric


# ============================================================================
# Tests for now_iso
# ============================================================================

class TestNowIso:
    def test_now_iso_returns_string(self):
        result = bc.now_iso()
        assert isinstance(result, str)

    def test_now_iso_contains_t(self):
        assert "T" in bc.now_iso()


# ============================================================================
# Tests for safe_version
# ============================================================================

class TestSafeVersion:
    def test_safe_version_existing_package(self):
        result = bc.safe_version("pytest")
        assert result != "unknown" and isinstance(result, str)

    def test_safe_version_nonexistent_package(self):
        assert bc.safe_version("nonexistent-package-xyz") == "unknown"


# ============================================================================
# Tests for percentile
# ============================================================================

class TestPercentile:
    @pytest.mark.parametrize("p,expected_min,expected_max", [(50, 4.5, 5.5), (99, 9.0, 10.0)])
    def test_percentile_normal(self, p, expected_min, expected_max):
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        assert expected_min <= bc.percentile(values, p) <= expected_max

    def test_percentile_empty_list(self):
        assert bc.percentile([], 50) is None

    def test_percentile_single_value(self):
        assert bc.percentile([5.0], 50) == 5.0

    def test_percentile_p0(self):
        assert bc.percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0) == 1.0

    def test_percentile_p100(self):
        assert bc.percentile([1.0, 2.0, 3.0, 4.0, 5.0], 100) == 5.0


# ============================================================================
# Tests for mean_or_none
# ============================================================================

class TestMeanOrNone:
    def test_mean_or_none_normal(self):
        assert bc.mean_or_none([1.0, 2.0, 3.0, 4.0, 5.0]) == 3.0

    def test_mean_or_none_empty(self):
        assert bc.mean_or_none([]) is None

    def test_mean_or_none_with_none_values(self):
        assert bc.mean_or_none([1.0, None, 3.0, None, 5.0]) == 3.0


# ============================================================================
# Tests for ms (seconds to milliseconds)
# ============================================================================

class TestMs:
    def test_ms_normal(self):
        assert bc.ms(1.0) == 1000.0
        assert bc.ms(0.5) == 500.0

    def test_ms_none(self):
        assert bc.ms(None) is None


# ============================================================================
# Tests for chunked
# ============================================================================

class TestChunked:
    def test_chunked_normal(self):
        assert list(bc.chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]

    def test_chunked_exact_division(self):
        assert list(bc.chunked([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]

    def test_chunked_empty(self):
        assert list(bc.chunked([], 2)) == []

    def test_chunked_invalid_size(self):
        with pytest.raises(ValueError):
            list(bc.chunked([1, 2, 3], 0))


# ============================================================================
# Tests for write_json
# ============================================================================

class TestWriteJson:
    def test_write_json_new_file(self, temp_dir: Path):
        path = temp_dir / "test.json"
        bc.write_json(path, {"key": "value", "number": 42})
        assert path.exists()
        with open(path) as f:
            assert json.load(f) == {"key": "value", "number": 42}

    def test_write_json_creates_parent_dir(self, temp_dir: Path):
        path = temp_dir / "subdir" / "test.json"
        bc.write_json(path, {"data": 123})
        assert path.exists()


# ============================================================================
# Tests for append_csv
# ============================================================================

class TestAppendCsv:
    def test_append_csv_new_file(self, temp_dir: Path):
        path = temp_dir / "test.csv"
        bc.append_csv(path, {"col1": "a", "col2": 1})
        with open(path) as f:
            assert len(f.readlines()) == 2  # header + row

    def test_append_csv_append_mode(self, temp_dir: Path):
        path = temp_dir / "test.csv"
        bc.append_csv(path, {"col1": "a", "col2": 1})
        bc.append_csv(path, {"col1": "b", "col2": 2})
        with open(path) as f:
            assert len(f.readlines()) == 3  # header + 2 rows

    def test_append_csv_ignores_extra_keys(self, temp_dir: Path):
        path = temp_dir / "test.csv"
        bc.append_csv(path, {"col1": "a", "col2": 1, "extra": 999})
        with open(path) as f:
            assert "extra" not in f.readlines()[1]


# ============================================================================
# Tests for normalize_token_ids
# ============================================================================

class TestNormalizeTokenIds:
    def test_normalize_token_ids_shorter(self):
        result = bc.normalize_token_ids([1, 2, 3], 10)
        assert len(result) == 10 and result[:3] == [1, 2, 3]

    def test_normalize_token_ids_longer(self):
        assert bc.normalize_token_ids([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5) == [1, 2, 3, 4, 5]

    def test_normalize_token_ids_exact(self):
        assert bc.normalize_token_ids([1, 2, 3, 4, 5], 5) == [1, 2, 3, 4, 5]

    def test_normalize_token_ids_invalid_zero(self):
        assert bc.normalize_token_ids([1, 2, 3], 0) == [1, 2, 3]

    def test_normalize_token_ids_empty(self):
        with pytest.raises(ValueError):
            bc.normalize_token_ids([], 10)


# ============================================================================
# Tests for make_synthetic_token_ids
# ============================================================================

class TestMakeSyntheticTokenIds:
    @pytest.mark.parametrize("lang", ["zh", "en"])
    def test_make_synthetic_token_ids(self, mock_tokenizer, lang):
        result = bc.make_synthetic_token_ids(mock_tokenizer, 20, 0, lang)
        assert isinstance(result, list) and len(result) == 20

    def test_make_synthetic_token_ids_invalid_len(self, mock_tokenizer):
        with pytest.raises(ValueError):
            bc.make_synthetic_token_ids(mock_tokenizer, 0, 0)


# ============================================================================
# Tests for load_prompt_token_ids
# ============================================================================

class TestLoadPromptTokenIds:
    def test_load_prompt_token_ids_synthetic(self, mock_tokenizer):
        token_ids, sources = bc.load_prompt_token_ids(tokenizer=mock_tokenizer, num_prompts=5, input_len=32)
        assert len(token_ids) == 5 and len(sources) == 5
        assert all(s == "synthetic" for s in sources) and all(len(ids) == 32 for ids in token_ids)

    def test_load_prompt_token_ids_invalid_num(self, mock_tokenizer):
        with pytest.raises(ValueError):
            bc.load_prompt_token_ids(mock_tokenizer, 0, 32)


# ============================================================================
# Tests for make_vllm_prompts
# ============================================================================

class TestMakeVllmPrompts:
    def test_make_vllm_prompts_with_token_ids(self):
        result = bc.make_vllm_prompts([[1, 2, 3], [4, 5, 6]], use_token_ids=True)
        assert len(result) == 2 and result[0] == {"prompt_token_ids": [1, 2, 3]}

    def test_make_vllm_prompts_without_token_ids_raises(self):
        with pytest.raises(ValueError):
            bc.make_vllm_prompts([[1, 2, 3]], use_token_ids=False)


# ============================================================================
# Tests for output_token_count
# ============================================================================

class TestOutputTokenCount:
    def test_output_token_count_with_token_ids(self):
        output = MagicMock()
        output.outputs = [MagicMock(token_ids=[1, 2, 3, 4, 5])]
        assert bc.output_token_count(output) == 5

    def test_output_token_count_with_text_fallback(self):
        output = MagicMock()
        output.outputs = [MagicMock(token_ids=None, text="hello")]
        assert bc.output_token_count(output) == 5

    def test_output_token_count_no_outputs(self):
        output = MagicMock()
        output.outputs = []
        assert bc.output_token_count(output) == 0


# ============================================================================
# Tests for output_text_preview
# ============================================================================

class TestOutputTextPreview:
    def test_output_text_preview_normal(self):
        output = MagicMock()
        output.outputs = [MagicMock(text="Hello World")]
        assert bc.output_text_preview(output) == "Hello World"

    def test_output_text_preview_truncated(self):
        output = MagicMock()
        output.outputs = [MagicMock(text="a" * 300)]
        assert len(bc.output_text_preview(output, limit=200)) == 200

    def test_output_text_preview_no_outputs(self):
        output = MagicMock()
        output.outputs = []
        assert bc.output_text_preview(output) == ""


# ============================================================================
# Tests for load_json_maybe
# ============================================================================

class TestLoadJsonMaybe:
    @pytest.mark.parametrize("input_val", [None, ""])
    def test_load_json_maybe_none_or_empty(self, input_val):
        assert bc.load_json_maybe(input_val) is None

    def test_load_json_maybe_valid_file(self, temp_dir: Path):
        path = temp_dir / "test.json"
        with open(path, "w") as f:
            json.dump({"key": "value"}, f)
        assert bc.load_json_maybe(str(path)) == {"key": "value"}

    def test_load_json_maybe_with_summary_key(self, temp_dir: Path):
        path = temp_dir / "test.json"
        with open(path, "w") as f:
            json.dump({"summary": {"key": "value"}, "extra": 123}, f)
        assert bc.load_json_maybe(str(path)) == {"key": "value"}


# ============================================================================
# Tests for metric_value (speculative metrics)
# ============================================================================

class TestMetricValue:
    @pytest.mark.parametrize("input_val,expected", [(42.5, 42.5), (42, 42.0), (None, 0.0)])
    def test_metric_value(self, input_val, expected):
        metric = MagicMock()
        metric.value = input_val
        assert bc.metric_value(metric) == expected

    def test_metric_value_without_value_attr(self):
        assert bc.metric_value(MagicMock(spec=[])) == 0.0


# ============================================================================
# Tests for collect_spec_metrics
# ============================================================================

class TestCollectSpecMetrics:
    def test_collect_spec_metrics_no_get_metrics(self):
        llm = MagicMock()
        llm.get_metrics = None
        result = bc.collect_spec_metrics(llm, 4)
        assert result["available"] is False and result["num_drafts"] == 0.0

    def test_collect_spec_metrics_with_metrics(self):
        llm = MagicMock()
        llm.get_metrics.return_value = [
            create_mock_metric("vllm:spec_decode_num_drafts_total", 100.0),
            create_mock_metric("vllm:spec_decode_num_draft_tokens_total", 400.0),
            create_mock_metric("vllm:spec_decode_num_accepted_tokens_total", 200.0),
        ]
        result = bc.collect_spec_metrics(llm, 4)
        assert result["available"] is True
        assert result["num_drafts"] == 100.0 and result["num_draft_tokens"] == 400.0

    def test_collect_spec_metrics_with_position_counts(self):
        llm = MagicMock()
        metric = MagicMock()
        metric.name = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
        metric.values = [80.0, 60.0, 40.0, 20.0]
        llm.get_metrics.return_value = [metric]
        result = bc.collect_spec_metrics(llm, 4)
        assert result["available"] is True
        assert result["accepted_per_pos_counts"]["0"] == 80.0 and result["accepted_per_pos_counts"]["1"] == 60.0

    def test_collect_spec_metrics_exception(self):
        llm = MagicMock()
        llm.get_metrics.side_effect = Exception("test error")
        result = bc.collect_spec_metrics(llm, 4)
        assert result["available"] is False and "error" in result


# ============================================================================
# Tests for diff_spec_metrics
# ============================================================================

class TestDiffSpecMetrics:
    def test_diff_spec_metrics_basic(self, sample_spec_metrics_before, sample_spec_metrics_after):
        result = bc.diff_spec_metrics(sample_spec_metrics_before, sample_spec_metrics_after)
        assert result["available"] is True
        assert result["num_drafts"] == 100.0 and result["num_draft_tokens"] == 400.0

    def test_diff_spec_metrics_acceptance_rate(self, sample_spec_metrics_before, sample_spec_metrics_after):
        result = bc.diff_spec_metrics(sample_spec_metrics_before, sample_spec_metrics_after)
        assert result["acceptance_rate"] == 0.5  # 200 / 400

    def test_diff_spec_metrics_accepted_per_draft(self, sample_spec_metrics_before, sample_spec_metrics_after):
        result = bc.diff_spec_metrics(sample_spec_metrics_before, sample_spec_metrics_after)
        assert result["accepted_tokens_per_draft"] == 2.0  # 200 / 100

    def test_diff_spec_metrics_mean_acceptance_length(self, sample_spec_metrics_before, sample_spec_metrics_after):
        result = bc.diff_spec_metrics(sample_spec_metrics_before, sample_spec_metrics_after)
        assert result["mean_acceptance_length_including_bonus"] == 3.0  # 1 + 2.0

    def test_diff_spec_metrics_no_spec(self, sample_spec_metrics_before):
        after = {"available": False, "num_drafts": 0.0, "num_draft_tokens": 0.0, "num_accepted_tokens": 0.0, "accepted_per_pos_counts": {}}
        result = bc.diff_spec_metrics(sample_spec_metrics_before, after)
        assert result["available"] is False and result["acceptance_rate"] is None

    def test_diff_spec_metrics_zero_draft_tokens(self):
        zero_metrics = {"available": True, "num_drafts": 0.0, "num_draft_tokens": 0.0, "num_accepted_tokens": 0.0, "accepted_per_pos_counts": {}}
        result = bc.diff_spec_metrics(zero_metrics, zero_metrics)
        assert result["acceptance_rate"] is None and result["accepted_tokens_per_draft"] is None


# ============================================================================
# Tests for summarize_offline_run
# ============================================================================

class TestSummarizeOfflineRun:
    def test_summarize_offline_run_basic(self, mock_args, mock_env_info, sample_batch_records, sample_request_records, sample_spec_metrics_before, sample_spec_metrics_after):
        spec_delta = bc.diff_spec_metrics(sample_spec_metrics_before, sample_spec_metrics_after)
        result = bc.summarize_offline_run(
            args=mock_args, run_id="test_run", case_id="baseline_in512_out256_bs1_k0",
            load_time_s=10.0, measured_wall_time_s=5.0, batch_records=sample_batch_records,
            request_records=sample_request_records, spec_metrics_delta=spec_delta, env_info=mock_env_info,
        )
        assert result["run_id"] == "test_run" and result["mode"] == "baseline"
        assert result["success_count"] == 2 and result["failed_count"] == 1
        assert result["total_input_tokens"] == 1024 and result["total_output_tokens"] == 512

    def test_summarize_offline_run_with_spec_mode(self, mock_args, mock_env_info, sample_batch_records, sample_request_records, sample_spec_metrics_before, sample_spec_metrics_after):
        mock_args.mode = "spec"
        mock_args.draft_model = "/path/to/draft"
        mock_args.num_speculative_tokens = 4
        spec_delta = bc.diff_spec_metrics(sample_spec_metrics_before, sample_spec_metrics_after)
        result = bc.summarize_offline_run(
            args=mock_args, run_id="test_run", case_id="spec_in512_out256_bs1_k4",
            load_time_s=10.0, measured_wall_time_s=5.0, batch_records=sample_batch_records,
            request_records=sample_request_records, spec_metrics_delta=spec_delta, env_info=mock_env_info,
        )
        assert result["mode"] == "spec" and result["draft_model"] == "/path/to/draft"
        assert result["num_speculative_tokens"] == 4 and result["spec_acceptance_rate"] == 0.5
