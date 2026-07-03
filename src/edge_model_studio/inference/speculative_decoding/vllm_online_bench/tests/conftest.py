# -*- coding: utf-8 -*-

"""Shared pytest fixtures for vllm_online_bench tests."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

# Mock heavy dependencies before importing the module
mock_openpyxl = MagicMock()
mock_openpyxl.styles = MagicMock()
mock_openpyxl.styles.Font = MagicMock()
mock_openpyxl.styles.Alignment = MagicMock()
mock_openpyxl.Workbook = MagicMock()
mock_openpyxl.load_workbook = MagicMock()
sys.modules['openpyxl'] = mock_openpyxl
sys.modules['openpyxl.styles'] = mock_openpyxl.styles
sys.modules['transformers'] = MagicMock()
sys.modules['requests'] = MagicMock()


@pytest.fixture
def temp_dir() -> Path:
    """Create a temporary directory."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    import shutil
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_client_config() -> Dict[str, Any]:
    """Sample client config for YAML parsing tests."""
    return {
        "base-url": "http://127.0.0.1:8000",
        "model": "Qwen3-30B-A3B-w8a8",
        "endpoint": "chat",
        "stream": True,
        "num-prompts": 16,
        "max-concurrency": 4,
        "request-rate": 0.0,
        "input-len": 512,
        "output-len": 256,
        "temperature": 0.0,
        "top-p": 1.0,
        "prompt-file": "",
        "dataset-name": "custom",
    }


@pytest.fixture
def mock_args() -> MagicMock:
    """Mock argparse.Namespace for online benchmark."""
    args = MagicMock()
    args.base_url = "http://127.0.0.1:8000"
    args.model = "Qwen3-30B-A3B-w8a8"
    args.tokenizer = ""
    args.api_key = ""
    args.endpoint = "chat"
    args.stream = True
    args.include_usage = False
    args.prompt_file = ""
    args.dataset_name = "custom"
    args.num_prompts = 16
    args.max_concurrency = 4
    args.request_rate = 0.0
    args.input_len = 512
    args.output_len = 256
    args.normalize_file_prompts = False
    args.synthetic_language = "zh"
    args.temperature = 0.0
    args.top_p = 1.0
    args.top_k = None
    args.seed = None
    args.ignore_eos = False
    args.timeout = 600.0
    args.output_dir = "./bench_results"
    args.result_filename = ""
    args.save_result = True
    args.print_output = False
    args.preview_chars = 300
    args.run_id = "test_run_001"
    args.case_id = "online_chat_in512_out256_n16_c4"
    args.tester = "tester"
    args.hardware = "Ascend 910B"
    args.skip_health_check = True
    args.skip_list_models = True
    return args


@pytest.fixture
def sample_request_records() -> List[Dict[str, Any]]:
    """Sample request records from online benchmark."""
    return [
        {
            "request_idx": 0,
            "success": True,
            "status_code": 200,
            "error": None,
            "prompt_source": "synthetic",
            "input_tokens": 512,
            "output_tokens": 256,
            "latency_s": 1.5,
            "ttft_s": 0.1,
            "tpot_s": 0.005,
            "itl_avg_s": 0.004,
            "itl_p50_s": 0.003,
            "itl_p90_s": 0.006,
            "itl_p99_s": 0.01,
            "num_stream_chunks": 50,
            "output_preview": "测试输出",
        },
        {
            "request_idx": 1,
            "success": True,
            "status_code": 200,
            "error": None,
            "prompt_source": "synthetic",
            "input_tokens": 512,
            "output_tokens": 256,
            "latency_s": 1.4,
            "ttft_s": 0.09,
            "tpot_s": 0.005,
            "itl_avg_s": 0.004,
            "itl_p50_s": 0.003,
            "itl_p90_s": 0.006,
            "itl_p99_s": 0.01,
            "num_stream_chunks": 50,
            "output_preview": "测试输出2",
        },
        {
            "request_idx": 2,
            "success": False,
            "status_code": 500,
            "error": "Internal server error",
            "prompt_source": "synthetic",
            "input_tokens": 512,
            "output_tokens": 0,
            "latency_s": 0.5,
            "ttft_s": None,
            "tpot_s": None,
            "itl_avg_s": None,
            "itl_p50_s": None,
            "itl_p90_s": None,
            "itl_p99_s": None,
            "num_stream_chunks": 0,
            "output_preview": "",
        },
    ]
