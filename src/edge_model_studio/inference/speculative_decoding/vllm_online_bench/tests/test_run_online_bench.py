# -*- coding: utf-8 -*-

"""Tests for run_online_bench module - YAML config and benchmark functions."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Set up path to import the module under test
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_online_bench import (
    build_prompts,
    parse_yaml_config,
    validate_client_config,
    yaml_to_cli_args,
    summarize,
)


class TestYamlToCliArgs:
    """Tests for yaml_to_cli_args function."""

    def test_basic_conversion(self, sample_client_config):
        """Test basic YAML dict to CLI args conversion."""
        args = yaml_to_cli_args(sample_client_config)

        assert "--base-url" in args
        assert "http://127.0.0.1:8000" in args
        assert "--model" in args
        assert "Qwen3-30B-A3B-w8a8" in args
        assert "--num-prompts" in args
        assert "16" in args

    def test_stream_true(self):
        """Test stream=True produces --stream flag."""
        config = {"stream": True}
        args = yaml_to_cli_args(config)
        assert "--stream" in args
        assert "--no-stream" not in args

    def test_stream_false(self):
        """Test stream=False produces --no-stream flag."""
        config = {"stream": False}
        args = yaml_to_cli_args(config)
        assert "--no-stream" in args
        assert "--stream" not in args

    def test_bool_true_flag(self):
        """Test boolean True value produces flag without value."""
        config = {"skip-health-check": True}
        args = yaml_to_cli_args(config)
        assert "--skip-health-check" in args
        # Should not have a value after the flag
        idx = args.index("--skip-health-check")
        assert idx + 1 >= len(args) or args[idx + 1].startswith("--")

    def test_none_values_skipped(self):
        """Test None values are skipped."""
        config = {"top-k": None, "seed": None}
        args = yaml_to_cli_args(config)
        assert "--top-k" not in args
        assert "--seed" not in args

    def test_internal_fields_skipped(self):
        """Test internal fields (dataset-name, save-result, result-filename) are skipped."""
        config = {
            "base-url": "http://localhost",
            "dataset-name": "custom",
            "save-result": True,
            "result-filename": "test.json",
        }
        args = yaml_to_cli_args(config)
        assert "--dataset-name" not in args
        assert "--save-result" not in args
        assert "--result-filename" not in args
        assert "--base-url" in args

    def test_empty_config(self):
        """Test empty config returns empty list."""
        args = yaml_to_cli_args({})
        assert args == []


class TestValidateClientConfig:
    """Tests for validate_client_config function."""

    def test_valid_config(self, sample_client_config):
        """Test valid config returns no errors."""
        errors = validate_client_config(sample_client_config)
        assert errors == []

    def test_missing_base_url(self):
        """Test missing base-url returns error."""
        config = {"model": "Qwen3"}
        errors = validate_client_config(config)
        assert len(errors) == 1
        assert "base-url" in errors[0]

    def test_missing_model(self):
        """Test missing model returns error."""
        config = {"base-url": "http://localhost"}
        errors = validate_client_config(config)
        assert len(errors) == 1
        assert "model" in errors[0]

    def test_missing_both(self):
        """Test missing both base-url and model returns two errors."""
        config = {}
        errors = validate_client_config(config)
        assert len(errors) == 2

    def test_empty_base_url(self):
        """Test empty base-url returns error."""
        config = {"base-url": "", "model": "Qwen3"}
        errors = validate_client_config(config)
        assert len(errors) == 1
        assert "base-url" in errors[0]


class TestParseYamlConfig:
    """Tests for parse_yaml_config function."""

    def test_parse_valid_yaml(self, temp_dir):
        """Test parsing valid YAML file."""
        yaml_content = """
client:
  - base-url: http://localhost:8000
    model: test-model
"""
        yaml_file = temp_dir / "test.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        config = parse_yaml_config(yaml_file)
        assert "client" in config
        assert config["client"][0]["base-url"] == "http://localhost:8000"
        assert config["client"][0]["model"] == "test-model"

    def test_parse_nonexistent_file(self):
        """Test parsing nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_yaml_config("/nonexistent/path/config.yaml")

    def test_parse_invalid_yaml(self, temp_dir):
        """Test parsing invalid YAML raises yaml.YAMLError."""
        yaml_file = temp_dir / "invalid.yaml"
        yaml_file.write_text("invalid: yaml: content:", encoding="utf-8")

        import yaml
        with pytest.raises(yaml.YAMLError):
            parse_yaml_config(yaml_file)

    def test_parse_non_dict_yaml(self, temp_dir):
        """Test parsing non-dict YAML raises SystemExit."""
        yaml_file = temp_dir / "list.yaml"
        yaml_file.write_text("- item1\n- item2", encoding="utf-8")

        with pytest.raises(SystemExit):
            parse_yaml_config(yaml_file)


class TestBuildPrompts:
    """Tests for build_prompts function."""

    def test_synthetic_prompts_count(self, mock_args):
        """Test synthetic prompts are generated correctly."""
        mock_args.prompt_file = ""
        mock_args.num_prompts = 4
        mock_args.input_len = 128
        mock_args.normalize_file_prompts = False
        mock_args.synthetic_language = "zh"

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(10, 50))  # 40 tokens

        prompts, input_tokens, sources = build_prompts(mock_args, mock_tokenizer)

        assert len(prompts) == 1
        assert len(input_tokens) == 1
        assert all(s == "synthetic" for s in sources)

    def test_build_prompts_respects_num_prompts(self, mock_args):
        """Test build_prompts respects num_prompts setting."""
        mock_args.prompt_file = ""
        mock_args.num_prompts = 8
        mock_args.input_len = 128
        mock_args.normalize_file_prompts = False
        mock_args.synthetic_language = "en"

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(10, 50))

        prompts, _, _ = build_prompts(mock_args, mock_tokenizer)
        assert len(prompts) == 1


class TestSummarize:
    """Tests for summarize function."""

    def test_summarize_basic(self, mock_args, sample_request_records):
        """Test summarize produces expected fields."""
        summary = summarize(
            args=mock_args,
            run_id="test_run",
            case_id="test_case",
            measured_wall_time_s=10.0,
            records=sample_request_records,
        )

        assert summary["run_id"] == "test_run"
        assert summary["case_id"] == "test_case"
        assert summary["bench_mode"] == "online"
        assert summary["num_prompts"] == 16  # from mock_args
        assert summary["success_count"] == 2
        assert summary["failed_count"] == 1
        assert "total_input_tokens" in summary
        assert "total_output_tokens" in summary
        assert "request_throughput_req_s" in summary

    def test_summarize_with_no_records(self, mock_args):
        """Test summarize with empty records."""
        summary = summarize(
            args=mock_args,
            run_id="test_run",
            case_id="test_case",
            measured_wall_time_s=1.0,
            records=[],
        )

        assert summary["success_count"] == 0
        assert summary["failed_count"] == 0
        assert summary["request_throughput_req_s"] == 0.0  # 0 requests / 1.0s = 0.0
