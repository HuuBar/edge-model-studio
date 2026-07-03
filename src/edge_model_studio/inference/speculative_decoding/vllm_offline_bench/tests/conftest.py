#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared pytest fixtures for vllm_offline_bench tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def temp_dir() -> Path:
    """Create a temporary directory."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    import shutil
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_request_records() -> List[Dict[str, Any]]:
    """Sample request records from offline benchmark."""
    return [
        {
            "request_idx": 0,
            "source_idx": 0,
            "batch_idx": 0,
            "success": True,
            "error": None,
            "prompt_source": "synthetic",
            "input_tokens": 512,
            "output_tokens": 256,
            "batch_latency_s": 0.5,
            "output_preview": "这是测试输出",
        },
        {
            "request_idx": 1,
            "source_idx": 1,
            "batch_idx": 0,
            "success": True,
            "error": None,
            "prompt_source": "synthetic",
            "input_tokens": 512,
            "output_tokens": 256,
            "batch_latency_s": 0.5,
            "output_preview": "这是测试输出2",
        },
        {
            "request_idx": 2,
            "source_idx": 2,
            "batch_idx": 1,
            "success": False,
            "error": "CUDA out of memory",
            "prompt_source": "synthetic",
            "input_tokens": 512,
            "output_tokens": 0,
            "batch_latency_s": 1.0,
            "output_preview": "",
        },
    ]


@pytest.fixture
def sample_batch_records() -> List[Dict[str, Any]]:
    """Sample batch records from offline benchmark."""
    return [
        {
            "batch_idx": 0,
            "batch_size": 2,
            "latency_s": 0.5,
            "success": True,
            "error": None,
            "input_tokens": 1024,
            "output_tokens": 512,
            "output_tokens_per_s": 1024.0,
        },
        {
            "batch_idx": 1,
            "batch_size": 1,
            "latency_s": 1.0,
            "success": False,
            "error": "CUDA out of memory",
            "input_tokens": 512,
            "output_tokens": 0,
            "output_tokens_per_s": None,
        },
    ]


@pytest.fixture
def sample_spec_metrics_before() -> Dict[str, Any]:
    """Sample speculative metrics before benchmark."""
    return {
        "available": False,
        "num_drafts": 0.0,
        "num_draft_tokens": 0.0,
        "num_accepted_tokens": 0.0,
        "accepted_per_pos_counts": {},
    }


@pytest.fixture
def sample_spec_metrics_after() -> Dict[str, Any]:
    """Sample speculative metrics after benchmark."""
    return {
        "available": True,
        "num_drafts": 100.0,
        "num_draft_tokens": 400.0,
        "num_accepted_tokens": 200.0,
        "accepted_per_pos_counts": {
            "0": 80.0,
            "1": 60.0,
            "2": 40.0,
            "3": 20.0,
        },
    }

@pytest.fixture
def default_args() -> MagicMock:
    """提供一个包含完整默认字段的 MagicMock 命令行参数对象."""
    args = MagicMock()
    # Speculative config 默认相关字段
    args.mode = "baseline"
    args.speculative_config_json = ""
    args.draft_model = ""
    args.num_speculative_tokens = 0
    args.spec_method = "eagle3"
    args.draft_tensor_parallel_size = 0
    args.disable_padded_drafter_batch = False
    args.parallel_drafting = False
    args.extra_speculative_config_json = ""

    # LLM kwargs 默认相关字段
    args.target_model = "/path/to/model"
    args.tokenizer = ""
    args.trust_remote_code = True
    args.tensor_parallel_size = 2
    args.dtype = "float16"
    args.max_model_len = 2048
    args.gpu_memory_utilization = 0.9
    args.max_num_seqs = 40
    args.max_num_batched_tokens = 4096
    args.enforce_eager = True
    args.enable_chunked_prefill = False
    args.disable_log_stats = False
    args.extra_llm_kwargs_json = ""

    # Sampling 默认相关字段
    args.temperature = 0.0
    args.top_p = 1.0
    args.output_len = 256
    args.ignore_eos = True
    args.top_k = None
    args.seed = 42
    args.skip_special_tokens = None
    return args

@pytest.fixture
def mock_args(default_args: MagicMock) -> MagicMock:
    """Mock argparse.Namespace for summarize_offline_run.

    Inherits from default_args and overrides fields that differ for this test.
    """
    args = default_args
    # Override fields that differ from default_args
    args.tensor_parallel_size = 1
    args.dtype = "auto"
    args.enforce_eager = False
    args.num_prompts = 32
    args.batch_size = 1
    args.warmup_batches = 1
    args.prompt_file = ""
    args.synthetic_language = "zh"
    args.normalize_file_prompts = False
    args.input_len = 512
    args.tester = "tester"
    args.hardware = "Ascend 910B"
    return args


@pytest.fixture
def mock_env_info() -> Dict[str, Any]:
    """Mock environment info."""
    return {
        "hostname": "test-host",
        "platform": "Linux-x86_64",
        "python_version": "3.11.0",
        "python_executable": "/usr/bin/python",
        "vllm_version": "0.4.0",
        "torch_version": "2.0.0",
        "transformers_version": "4.30.0",
        "cwd": "/home/test",
    }


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    """Mock tokenizer for testing prompt generation."""
    tokenizer = MagicMock()
    # Mock encode to return list of token IDs
    tokenizer.encode.return_value = list(range(10, 50))
    return tokenizer
