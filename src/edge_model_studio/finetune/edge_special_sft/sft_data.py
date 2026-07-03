from __future__ import annotations

import glob
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from pretrain_utils import Logger, dump_json, human_bytes, human_num, now_str


class ByteTokenizer:
    vocab_size: int = 256

    def encode(self, text: str) -> List[int]:
        b = text.encode("utf-8", errors="replace")
        return list(b)

    def decode(self, ids: List[int]) -> str:
        b = bytes(int(x) & 0xFF for x in ids)
        return b.decode("utf-8", errors="replace")


# -----------------------------
# Synthetic SFT dataset generator
# -----------------------------

_SYN_INSTRUCTIONS = [
    "把下面句子翻译成英文",
    "把下面句子翻译成中文",
    "总结下面内容，输出 1 句话",
    "解释下面代码在做什么",
    "从下面描述中抽取关键信息，输出 JSON",
    "给下面问题一个直接答案",
    "改写下面文字，让它更口语化",
]

_SYN_TOPICS = [
    "运动健康", "跑步训练", "心率区间", "睡眠恢复", "营养摄入", "模型剪枝", "蒸馏训练",
    "数据清洗", "性能测试", "GPU 吞吐", "推理加速", "日志分析", "异常定位"
]

_SYN_TEXTS = [
    "今天我跑了 5 公里，平均配速 5 分 10 秒每公里，心率主要在 2 区。",
    "训练过程中出现了梯度爆炸，需要检查学习率与混合精度设置。",
    "我们把数据切分成 train/val，并记录 tokens/s 作为吞吐指标。",
    "模型输出太长，想要更短更直接的回答。",
    "需要在不升级依赖的情况下提高推理吞吐。",
]

_SYN_CODE = [
    "def f(x):\n    s=0\n    for i in range(x):\n        s+=i\n    return s\n",
    "import torch\nx=torch.randn(2,3)\ny=x.mean(dim=1)\nprint(y)\n",
]


def maybe_generate_synthetic_sft(data_dir: str, logger: Logger) -> None:
    os.makedirs(data_dir, exist_ok=True)
    existing = glob.glob(os.path.join(data_dir, "**", "*.*"), recursive=True)
    existing = [p for p in existing if os.path.isfile(p)]
    if existing:
        logger.log0(f"sft data_dir has {len(existing)} files; skip synthetic generation.")
        return

    logger.log0("sft data_dir empty; generating synthetic SFT jsonl dataset.")
    rng = random.Random(20250102)
    out_path = os.path.join(data_dir, "synthetic_sft.jsonl")

    def make_sample(i: int) -> Dict:
        inst = rng.choice(_SYN_INSTRUCTIONS)
        topic = rng.choice(_SYN_TOPICS)
        text = rng.choice(_SYN_TEXTS)
        code = rng.choice(_SYN_CODE)

        if "代码" in inst:
            inp = code
            out = (
                "这段代码定义了一个函数或执行了一个张量运算，用来演示基本的循环/聚合操作。"
                "如果是 torch 片段，它在生成随机张量后沿维度求均值并打印结果。"
            )
        elif "JSON" in inst:
            inp = f"主题：{topic}\n内容：{text}\n指标：pace=5:10, hr_zone=2"
            out = json.dumps(
                {
                    "topic": topic,
                    "summary": "跑步训练概况与关键指标",
                    "metrics": {"pace": "5:10", "hr_zone": 2},
                    "note": "示例输出为合成数据",
                },
                ensure_ascii=False,
            )
        elif "翻译成英文" in inst:
            inp = f"{text}（主题：{topic}）"
            out = "I ran 5 kilometers today at an average pace of 5:10 per kilometer, mostly staying in zone 2."
        elif "翻译成中文" in inst:
            inp = "We split the dataset into train/val and use tokens/s as the throughput metric."
            out = "我们把数据集切分为训练集和验证集，并用 tokens/s 作为吞吐指标。"
        elif "总结" in inst:
            inp = f"{text} 另外，{rng.choice(_SYN_TEXTS)}"
            out = "这段内容概括了训练/运动过程中的关键指标与优化方向。"
        elif "直接答案" in inst:
            inp = "SFT 和 pretrain 的区别是什么？"
            out = "Pretrain 用海量无标注数据学通用语言能力；SFT 用指令-答案对让模型更贴近期望行为。"
        else:
            inp = f"{text}（主题：{topic}）"
            out = "更口语一点：我今天跑了 5 公里，配速差不多 5 分 10 秒一公里，心率基本在 2 区。"

        return {
            "id": i,
            "instruction": inst,
            "input": inp,
            "output": out,
            "meta": {"source": "synthetic", "topic": topic, "ts": int(time.time())},
        }

    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(6000):
            f.write(json.dumps(make_sample(i), ensure_ascii=False) + "\n")

    logger.log0(f"synthetic SFT dataset generated: {out_path}")


# -----------------------------
# Reading / Formatting
# -----------------------------

def list_sft_files(data_dir: str, logger: Logger) -> List[str]:
    patterns = ["**/*.jsonl", "**/*.json", "**/*.txt"]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(data_dir, pat), recursive=True))
    files = [p for p in files if os.path.isfile(p)]
    files = sorted(set(files))
    logger.log0(f"discovered {len(files)} sft files under {data_dir}")
    return files


def _guess_sample_from_obj(obj: Dict) -> Tuple[str, str, str]:
    inst = ""
    inp = ""
    out = ""

    if isinstance(obj.get("instruction", None), str) and isinstance(obj.get("output", None), str):
        inst = obj.get("instruction", "")
        inp = obj.get("input", "") if isinstance(obj.get("input", ""), str) else ""
        out = obj.get("output", "")
        return inst, inp, out

    if isinstance(obj.get("prompt", None), str) and isinstance(obj.get("response", None), str):
        inst = obj.get("prompt", "")
        inp = ""
        out = obj.get("response", "")
        return inst, inp, out

    msgs = obj.get("messages", None)
    if isinstance(msgs, list) and msgs:
        parts = []
        last_assistant = ""
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", "")).lower()
            content = m.get("content", "")
            if not isinstance(content, str):
                continue
            if role == "assistant":
                last_assistant = content
            else:
                parts.append(f"{role}: {content}")
        inst = "\n".join(parts).strip()
        out = last_assistant.strip()
        return inst, "", out

    # fallback
    s = json.dumps(obj, ensure_ascii=False)
    return "请处理下面输入", s, "已处理（示例输出为随机生成）"


def iter_sft_samples(files: List[str], logger: Logger) -> Iterator[Dict[str, str]]:
    for fp in files:
        ext = os.path.splitext(fp)[1].lower()
        if ext == ".jsonl":
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    inst, inp, out = _guess_sample_from_obj(obj)
                    if inst.strip() and out.strip():
                        yield {"instruction": inst, "input": inp, "output": out}
        elif ext == ".json":
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                try:
                    obj = json.load(f)
                except Exception:
                    continue
            if isinstance(obj, list):
                for it in obj:
                    if not isinstance(it, dict):
                        continue
                    inst, inp, out = _guess_sample_from_obj(it)
                    if inst.strip() and out.strip():
                        yield {"instruction": inst, "input": inp, "output": out}
            elif isinstance(obj, dict):
                inst, inp, out = _guess_sample_from_obj(obj)
                if inst.strip() and out.strip():
                    yield {"instruction": inst, "input": inp, "output": out}
        else:
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line.strip():
                        continue
                    if "\t" in line:
                        a, b = line.split("\t", 1)
                        yield {"instruction": a.strip(), "input": "", "output": b.strip()}
                    else:
                        yield {"instruction": line.strip(), "input": "", "output": "（示例输出：该输入已读取）"}


def format_sft_text(instruction: str, input_text: str, output_text: str) -> Tuple[str, str]:
    inst = instruction.strip()
    inp = (input_text or "").strip()
    out = output_text.strip()

    prefix = f"### Instruction:\n{inst}\n"
    if inp:
        prefix += f"### Input:\n{inp}\n"
    prefix += "### Response:\n"

    full = prefix + out + "\n"
    return prefix, full


# -----------------------------
# Build binary + index
# -----------------------------

@dataclass
class SFTMeta:
    tokenizer: str
    vocab_size: int
    dtype: str
    train_samples: int
    val_samples: int
    train_tokens: int
    val_tokens: int
    created_at: str
    source_dir: str
    files: List[str]
    val_ratio: float
    template: str  # short description


def load_sft_meta(out_dir: str) -> Optional[SFTMeta]:
    p = os.path.join(out_dir, "sft_meta.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return SFTMeta(**obj)


def build_sft_bins(
    files: List[str],
    out_dir: str,
    tokenizer: ByteTokenizer,
    val_ratio: float,
    logger: Logger,
    seed: int = 20250103,
) -> SFTMeta:
    os.makedirs(out_dir, exist_ok=True)
    train_bin = os.path.join(out_dir, "sft_train.bin")
    val_bin = os.path.join(out_dir, "sft_val.bin")
    train_idx = os.path.join(out_dir, "sft_train_index.npy")
    val_idx = os.path.join(out_dir, "sft_val_index.npy")
    meta_path = os.path.join(out_dir, "sft_meta.json")

    for p in [train_bin, val_bin, train_idx, val_idx, meta_path]:
        if os.path.exists(p):
            os.remove(p)

    rng = random.Random(seed)

    train_records: List[List[int]] = []  # [offset, length, resp_start]
    val_records: List[List[int]] = []

    train_pos = 0
    val_pos = 0
    train_tokens = 0
    val_tokens = 0
    train_samples = 0
    val_samples = 0
    bytes_in = 0

    t0 = time.time()
    last_log = time.time()
    last_total_toks = 0

    # Open once for speed
    with open(train_bin, "wb") as ft, open(val_bin, "wb") as fv:
        for fp in files:
            try:
                bytes_in += os.stat(fp).st_size
            except Exception:
                pass

        for sample in iter_sft_samples(files, logger):
            prefix, full = format_sft_text(sample["instruction"], sample["input"], sample["output"])
            prefix_ids = tokenizer.encode(prefix)
            full_ids = tokenizer.encode(full)
            resp_start = len(prefix_ids)

            if len(full_ids) < 2:
                continue

            ids_u8 = np.asarray(full_ids, dtype=np.uint8)
            if rng.random() < val_ratio:
                offset = val_pos
                fv.write(ids_u8.tobytes())
                val_pos += int(ids_u8.size)
                val_records.append([offset, int(ids_u8.size), int(resp_start)])
                val_tokens += int(ids_u8.size)
                val_samples += 1
            else:
                offset = train_pos
                ft.write(ids_u8.tobytes())
                train_pos += int(ids_u8.size)
                train_records.append([offset, int(ids_u8.size), int(resp_start)])
                train_tokens += int(ids_u8.size)
                train_samples += 1

            if time.time() - last_log >= 2.0:
                total = train_tokens + val_tokens
                dt = time.time() - t0
                tok_s = total / max(dt, 1e-6)
                delta = total - last_total_toks
                last_total_toks = total
                logger.log0(
                    f"sft preprocess: samples={train_samples+val_samples}, tokens={human_num(total)}, "
                    f"tok/s={human_num(tok_s)}, recent+{human_num(delta)}"
                )
                last_log = time.time()

    np.save(train_idx, np.asarray(train_records, dtype=np.int64))
    np.save(val_idx, np.asarray(val_records, dtype=np.int64))

    dt = time.time() - t0
    logger.log0(
        f"sft preprocess done: input={human_bytes(bytes_in)}, "
        f"samples={train_samples+val_samples} (train={train_samples}, val={val_samples}), "
        f"tokens={human_num(train_tokens+val_tokens)} (train={human_num(train_tokens)}, val={human_num(val_tokens)}), "
        f"time={dt:.2f}s, tok/s={human_num((train_tokens+val_tokens)/max(dt,1e-6))}"
    )

    meta = SFTMeta(
        tokenizer="byte_utf8",
        vocab_size=tokenizer.vocab_size,
        dtype="uint8",
        train_samples=int(train_samples),
        val_samples=int(val_samples),
        train_tokens=int(train_tokens),
        val_tokens=int(val_tokens),
        created_at=now_str(),
        source_dir=os.path.abspath(os.path.dirname(files[0])) if files else "",
        files=[os.path.basename(f) for f in files],
        val_ratio=float(val_ratio),
        template="### Instruction / ### Input (optional) / ### Response",
    )
    dump_json(meta_path, meta.__dict__)
    return meta
