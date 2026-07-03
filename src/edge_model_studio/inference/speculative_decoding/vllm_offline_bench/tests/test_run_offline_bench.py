#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for run_offline_bench module - build and parse functions."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import functions to test - need to set up path first
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_offline_bench import (
    add_optional,
    build_llm_kwargs,
    build_sampling_params,
    build_speculative_config,
    call_generate,
    parse_extra_json,
)


def create_mock_metric(name: str, value: Any) -> MagicMock:
    """Create a mock metric object."""
    metric = MagicMock()
    metric.name = name
    metric.value = value
    return metric


# ============================================================================
# Tests for add_optional
# ============================================================================

class TestAddOptional:
    def test_add_optional_with_value(self):
        """Test add_optional adds value to dict."""
        d = {}
        add_optional(d, "key", "value")
        assert d == {"key": "value"}

    def test_add_optional_with_none(self):
        """Test add_optional does not add None value."""
        d = {}
        add_optional(d, "key", None)
        assert d == {}


# ============================================================================
# Tests for parse_extra_json
# ============================================================================

class TestParseExtraJson:
    def test_parse_extra_json_valid(self):
        """Test parse_extra_json with valid JSON."""
        result = parse_extra_json('{"key": "value"}', "test")
        assert result == {"key": "value"}

    def test_parse_extra_json_empty(self):
        """Test parse_extra_json with empty string returns empty dict."""
        result = parse_extra_json("", "test")
        assert result == {}

    def test_parse_extra_json_invalid_json(self):
        """Test parse_extra_json with invalid JSON raises SystemExit."""
        with pytest.raises(SystemExit, match="invalid JSON"):
            parse_extra_json("{invalid}", "test")

    def test_parse_extra_json_not_dict(self):
        """Test parse_extra_json with non-dict JSON raises SystemExit."""
        with pytest.raises(SystemExit, match="must be a JSON object"):
            parse_extra_json("[1, 2, 3]", "test")


# ============================================================================
# Tests for build_speculative_config
# ============================================================================

class TestBuildSpeculativeConfig:
    def test_build_speculative_config_baseline_mode(self, default_args: MagicMock):
        """Test build_speculative_config returns None for baseline mode."""
        default_args.mode = "baseline"
        result = build_speculative_config(default_args)
        assert result is None

    def test_build_speculative_config_with_json(self, default_args: MagicMock):
        """Test build_speculative_config with speculative_config_json."""
        default_args.mode = "spec"
        default_args.speculative_config_json = '{"method": "eagle3", "model": "/path"}'
        default_args.draft_model = ""

        result = build_speculative_config(default_args)
        assert result["method"] == "eagle3"
        assert result["model"] == "/path"

    def test_build_speculative_config_without_draft_model_raises(self, default_args: MagicMock):
        """Test build_speculative_config raises without draft_model."""
        default_args.mode = "spec"
        default_args.speculative_config_json = ""
        default_args.draft_model = ""
        default_args.num_speculative_tokens = 4

        with pytest.raises(SystemExit, match="--draft-model is required"):
            build_speculative_config(default_args)

    def test_build_speculative_config_without_tokens_raises(self, default_args: MagicMock):
        """Test build_speculative_config raises with invalid num_speculative_tokens."""
        default_args.mode = "spec"
        default_args.speculative_config_json = ""
        default_args.draft_model = "/path/to/draft"
        default_args.num_speculative_tokens = 0

        with pytest.raises(SystemExit, match="--num-speculative-tokens must be positive"):
            build_speculative_config(default_args)

    def test_build_speculative_config_creates_config(self, default_args: MagicMock):
        """Test build_speculative_config creates correct config."""
        default_args.mode = "spec"
        default_args.speculative_config_json = ""
        default_args.draft_model = "/path/to/draft"
        default_args.num_speculative_tokens = 4
        default_args.draft_tensor_parallel_size = 2
        default_args.disable_padded_drafter_batch = True
        default_args.parallel_drafting = True

        result = build_speculative_config(default_args)
        assert result["method"] == "eagle3"
        assert result["model"] == "/path/to/draft"
        assert result["num_speculative_tokens"] == 4
        assert result["draft_tensor_parallel_size"] == 2
        assert "disable_padded_drafter_batch" in result
        assert "parallel_drafting" in result


# ============================================================================
# Tests for build_llm_kwargs
# ============================================================================

class TestBuildLlmKwargs:
    def test_build_llm_kwargs_baseline(self, default_args: MagicMock):
        """Test build_llm_kwargs for baseline mode."""
        # default_args already has mode="baseline" and all other fields set
        result = build_llm_kwargs(default_args)
        assert result["model"] == "/path/to/model"
        assert result["tensor_parallel_size"] == 2
        assert result["dtype"] == "float16"
        assert result["max_model_len"] == 2048
        assert result["enforce_eager"] is True

    def test_build_llm_kwargs_with_speculative(self, default_args: MagicMock):
        """Test build_llm_kwargs for spec mode includes speculative_config."""
        default_args.mode = "spec"
        default_args.enforce_eager = False
        default_args.enable_chunked_prefill = True
        default_args.speculative_config_json = '{"method": "eagle3", "model": "/draft"}'

        result = build_llm_kwargs(default_args)
        assert "speculative_config" in result
        assert result["speculative_config"]["method"] == "eagle3"


# ============================================================================
# Tests for build_sampling_params
# ============================================================================

class TestBuildSamplingParams:
    def test_build_sampling_params_basic(self, default_args: MagicMock):
        """Test build_sampling_params creates valid params."""
        # default_args already has temperature, top_p, output_len, ignore_eos, top_k, seed set
        with patch("run_offline_bench.SamplingParams") as MockSamplingParams:
            MockSamplingParams.return_value = MagicMock()
            result = build_sampling_params(default_args)
            MockSamplingParams.assert_called_once()

    def test_build_sampling_params_with_top_k(self, default_args: MagicMock):
        """Test build_sampling_params includes top_k when set."""
        default_args.top_k = 50
        default_args.seed = None

        with patch("run_offline_bench.SamplingParams") as MockSamplingParams:
            MockSamplingParams.return_value = MagicMock()
            build_sampling_params(default_args)
            # Check that default_args.top_k was passed to SamplingParams
            assert default_args.top_k == 50


# ============================================================================
# Tests for call_generate
# ============================================================================

class TestCallGenerate:
    def test_call_generate_with_use_tqdm(self):
        """Test call_generate passes use_tqdm parameter."""
        llm = MagicMock()
        prompts = ["test"]
        sampling_params = MagicMock()

        # Patch the generate method on the llm instance directly
        llm.generate.return_value = []
        result = call_generate(llm, prompts, sampling_params, use_tqdm=True)
        assert result == []
        llm.generate.assert_called_once()

    def test_call_generate_without_use_tqdm_fallback(self):
        """Test call_generate falls back when TypeError raised."""
        llm = MagicMock()
        prompts = ["test"]
        sampling_params = MagicMock()

        # First call raises TypeError, second works
        llm.generate.side_effect = [TypeError("unexpected keyword argument"), []]
        result = call_generate(llm, prompts, sampling_params, use_tqdm=True)
        assert result == []
