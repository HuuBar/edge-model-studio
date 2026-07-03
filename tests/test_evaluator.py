"""Tests for evaluator.py."""
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock


def _jsonl_to_list(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def _build_stub_modules():
    """Build minimal stub modules required to import evaluator.py."""

    class DummyLogger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

    emsevals_pkg = types.ModuleType("emsevals")
    emsevals_pkg.__path__ = []

    utils_pkg = types.ModuleType("emsevals.utils")
    utils_pkg.__path__ = []

    benchmarks = types.ModuleType("emsevals.benchmarks")
    benchmarks.DatasetHandler = object

    config = types.ModuleType("emsevals.config")
    config.EvalConfig = object

    constants = types.ModuleType("emsevals.constants")
    constants.AnswerKeys = types.SimpleNamespace(
        INDEX="index",
        MODEL_SPEC="model_spec",
        ANSWER_ID="answer_id",
        SUBSET_NAME="subset_name",
        RAW_INPUT="raw_input",
        CHOICES="choices",
    )
    constants.DumpMode = types.SimpleNamespace(APPEND="append")
    constants.EvalStage = types.SimpleNamespace(
        INFERENCE="infer",
        SCORING="score",
    )
    constants.EvalType = types.SimpleNamespace(SERVICE="service")
    constants.JudgeStrategy = types.SimpleNamespace(
        RULE="rule",
        LLM="llm",
        AUTO="auto",
        LLM_RECALL="llm_recall",
    )
    constants.ReviewKeys = types.SimpleNamespace(
        REVIEWED="reviewed",
        REVIEW_ID="review_id",
        REVIEWER_SPEC="reviewer_spec",
        REVIEW_TIME="review_time",
        MESSAGE="message",
        CONTENT="content",
        REVIEW="review",
        GOLD="gold",
        PRED="pred",
        RESULT="result",
    )

    report = types.ModuleType("emsevals.report")
    report.Report = object
    report.gen_table = lambda *args, **kwargs: ""

    io_utils = types.ModuleType("emsevals.utils.io_utils")
    io_utils.OutputsStructure = object
    io_utils.dump_jsonl_data = lambda *args, **kwargs: None
    io_utils.gen_hash = lambda value: str(abs(hash(value)))
    io_utils.jsonl_to_list = _jsonl_to_list

    logger = types.ModuleType("emsevals.utils.logger")
    logger.get_logger = lambda: DummyLogger()

    model_utils = types.ModuleType("emsevals.utils.model_utils")
    model_utils.dict_torch_dtype_to_str = lambda value: value

    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable=None, *args, **kwargs: iterable if iterable is not None else []

    return {
        "emsevals": emsevals_pkg,
        "emsevals.utils": utils_pkg,
        "emsevals.benchmarks": benchmarks,
        "emsevals.config": config,
        "emsevals.constants": constants,
        "emsevals.report": report,
        "emsevals.utils.io_utils": io_utils,
        "emsevals.utils.logger": logger,
        "emsevals.utils.model_utils": model_utils,
        "tqdm": tqdm_module,
    }


def _load_evaluator_module():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    evaluator_path = os.path.abspath(
        os.path.join(
            test_dir,
            "..",
            "src",
            "edge_model_studio",
            "evaluation",
            "emsevals",
            "emsevals",
            "evaluator",
            "evaluator.py",
        )
    )

    spec = importlib.util.spec_from_file_location("evaluator_under_test", evaluator_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


with mock.patch.dict(sys.modules, _build_stub_modules()):
    evaluator_module = _load_evaluator_module()

Evaluator = evaluator_module.Evaluator
AnswerKeys = evaluator_module.AnswerKeys


class TestEvaluatorFilterAnswer(unittest.TestCase):
    """Test Evaluator.filter_answer static method."""

    def setUp(self):
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_obj.name

    def tearDown(self):
        self.tmpdir_obj.cleanup()

    def test_filter_answer_no_cache(self):
        """use_cache=False should return empty answers and all prompts."""
        prompts = [
            {AnswerKeys.INDEX: 0, "text": "q1"},
            {AnswerKeys.INDEX: 1, "text": "q2"},
        ]
        pred_file = os.path.join(self.tmpdir, "pred.jsonl")

        answers, remaining = Evaluator.filter_answer(False, prompts, pred_file)

        self.assertEqual(answers, [])
        self.assertEqual(remaining, prompts)

    def test_filter_answer_cache_not_exists(self):
        """use_cache=True but prediction file does not exist."""
        prompts = [
            {AnswerKeys.INDEX: 0, "text": "q1"},
        ]
        pred_file = os.path.join(self.tmpdir, "not_exists.jsonl")

        answers, remaining = Evaluator.filter_answer(True, prompts, pred_file)

        self.assertEqual(answers, [])
        self.assertEqual(remaining, prompts)

    def test_filter_answer_cache_with_answers(self):
        """use_cache=True should filter prompts already answered."""
        prompts = [
            {AnswerKeys.INDEX: 0, "text": "q1"},
            {AnswerKeys.INDEX: 1, "text": "q2"},
            {AnswerKeys.INDEX: 2, "text": "q3"},
        ]
        pred_file = os.path.join(self.tmpdir, "pred.jsonl")

        with open(pred_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({AnswerKeys.INDEX: 0, "answer": "a1"}) + "\n")
            f.write(json.dumps({AnswerKeys.INDEX: 2, "answer": "a3"}) + "\n")

        answers, remaining = Evaluator.filter_answer(True, prompts, pred_file)

        self.assertEqual(len(answers), 2)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0][AnswerKeys.INDEX], 1)


if __name__ == "__main__":
    unittest.main()