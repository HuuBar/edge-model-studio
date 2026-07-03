from __future__ import annotations

import glob
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np

from pretrain_utils import Logger, human_bytes, human_num, now_str, dump_json


class ByteTokenizer:
    vocab_size: int = 256

    def encode(self, text: str) -> List[int]:
        b = text.encode("utf-8", errors="replace")
        return list(b)

    def decode(self, ids: List[int]) -> str:
        b = bytes(int(x) & 0xFF for x in ids)
        return b.decode("utf-8", errors="replace")


_SYN_WORDS = [
    "runner", "heart", "pace", "stride", "oxygen", "recovery", "interval", "cadence",
    "training", "sleep", "hydration", "protein", "carbs", "fat", "watch", "sensor",
    "signal", "noise", "baseline", "trend", "threshold", "zone2", "zone5", "warmup",
    "cooldown", "metabolic", "stress", "adaptation", "consistency", "discipline",
    "metrics", "insight", "analysis", "coach", "plan", "progress", "habit",
    "dataset", "token", "model", "gradient", "optimizer", "batch", "throughput",
]


def maybe_generate_synthetic_dataset(data_dir: str, logger: Logger) -> None:
    os.makedirs(data_dir, exist_ok=True)
    existing = glob.glob(os.path.join(data_dir, "**", "*.*"), recursive=True)
    existing = [p for p in existing if os.path.isfile(p)]
    if existing:
        logger.log0(f"data_dir has {len(existing)} files; skip synthetic generation.")
        return

    logger.log0("data_dir empty; generating synthetic raw dataset.")
    rng = random.Random(12345)

    for i in range(4):
        path = os.path.join(data_dir, f"synthetic_{i:02d}.txt")
        with open(path, "w", encoding="utf-8") as f:
            for _ in range(8000):
                n = rng.randint(8, 24)
                words = rng.choices(_SYN_WORDS, k=n)
                hr = rng.randint(95, 188)
                pace = rng.uniform(3.5, 7.5)
                dist = rng.uniform(0.2, 18.0)
                s = f"session hr={hr} pace={pace:.2f}min/km dist={dist:.2f}km | " + " ".join(words)
                f.write(s + "\n")

    jpath = os.path.join(data_dir, "synthetic_records.jsonl")
    with open(jpath, "w", encoding="utf-8") as f:
        for i in range(3000):
            rec = {
                "id": i,
                "text": f"record {i} " + " ".join(rng.choices(_SYN_WORDS, k=rng.randint(10, 30))),
                "ts": int(time.time()) - rng.randint(0, 86400 * 30),
                "tag": rng.choice(["run", "sleep", "meal", "workout", "stress"]),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.log0(f"synthetic dataset generated in: {data_dir}")


def list_input_files(data_dir: str, logger: Logger) -> List[str]:
    patterns = ["**/*.txt", "**/*.text", "**/*.jsonl", "**/*.json", "**/*.log"]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(data_dir, pat), recursive=True))
    files = [p for p in files if os.path.isfile(p)]
    files = sorted(set(files))
    logger.log0(f"discovered {len(files)} raw files under {data_dir}")
    return files


def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t\f\v]+", " ", s)
    s = "".join(ch for ch in s if (ch == "\n" or ord(ch) >= 32))
    return s


def iter_text_from_file(path: str, jsonl_field: str = "text") -> Iterator[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".txt", ".text", ".log"]:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip("\n")
                if line.strip():
                    yield line
    elif ext == ".jsonl":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                val = obj.get(jsonl_field, "")
                if isinstance(val, str) and val.strip():
                    yield val
                else:
                    yield json.dumps(obj, ensure_ascii=False)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            try:
                obj = json.load(f)
            except Exception:
                return
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict) and isinstance(item.get(jsonl_field, None), str):
                    yield item[jsonl_field]
                else:
                    yield json.dumps(item, ensure_ascii=False)
        else:
            yield json.dumps(obj, ensure_ascii=False)
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip("\n")
                if line.strip():
                    yield line


@dataclass
class DataMeta:
    tokenizer: str
    vocab_size: int
    dtype: str
    train_tokens: int
    val_tokens: int
    created_at: str
    source_dir: str
    files: List[str]
    jsonl_field: str
    val_ratio: float


def load_meta(out_dir: str) -> Optional[DataMeta]:
    p = os.path.join(out_dir, "meta.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return DataMeta(**obj)


def file_n_tokens(bin_path: str, dtype: np.dtype) -> int:
    if not os.path.exists(bin_path):
        return 0
    size = os.path.getsize(bin_path)
    return int(size // np.dtype(dtype).itemsize)


def build_bins(
    files: List[str],
    out_dir: str,
    tokenizer: ByteTokenizer,
    jsonl_field: str,
    val_ratio: float,
    logger: Logger,
) -> DataMeta:
    os.makedirs(out_dir, exist_ok=True)
    train_bin = os.path.join(out_dir, "train.bin")
    val_bin = os.path.join(out_dir, "val.bin")
    meta_path = os.path.join(out_dir, "meta.json")

    for p in [train_bin, val_bin, meta_path]:
        if os.path.exists(p):
            os.remove(p)

    rng = random.Random(20250101)
    t0 = time.time()
    bytes_in = 0
    docs = 0
    train_tokens = 0
    val_tokens = 0

    chunk_train: List[int] = []
    chunk_val: List[int] = []
    train_flush = 1_000_000
    val_flush = max(100_000, int(train_flush * val_ratio))

    def append_tokens(path: str, ids: List[int]) -> int:
        if not ids:
            return 0
        arr = np.asarray(ids, dtype=np.uint8)
        with open(path, "ab") as f:
            arr.tofile(f)
        return int(arr.size)

    last_log = time.time()
    last_total = 0

    for fp in files:
        try:
            bytes_in += os.stat(fp).st_size
        except Exception:
            pass

        for text in iter_text_from_file(fp, jsonl_field=jsonl_field):
            docs += 1
            text = normalize_text(text) + "\n"
            ids = tokenizer.encode(text)

            if rng.random() < val_ratio:
                chunk_val.extend(ids)
            else:
                chunk_train.extend(ids)

            if len(chunk_train) >= train_flush:
                train_tokens += append_tokens(train_bin, chunk_train)
                chunk_train.clear()

            if len(chunk_val) >= val_flush:
                val_tokens += append_tokens(val_bin, chunk_val)
                chunk_val.clear()

            if time.time() - last_log >= 2.0:
                total = train_tokens + val_tokens + len(chunk_train) + len(chunk_val)
                dt = time.time() - t0
                tok_s = total / max(dt, 1e-6)
                delta = total - last_total
                last_total = total
                logger.log0(
                    f"preprocess: docs={docs}, tokens={human_num(total)}, tok/s={human_num(tok_s)}, recent+{human_num(delta)}"
                )
                last_log = time.time()

    train_tokens += append_tokens(train_bin, chunk_train)
    val_tokens += append_tokens(val_bin, chunk_val)
    chunk_train.clear()
    chunk_val.clear()

    dt = time.time() - t0
    total = train_tokens + val_tokens
    logger.log0(
        f"preprocess done: input={human_bytes(bytes_in)}, docs={docs}, tokens={human_num(total)} "
        f"(train={human_num(train_tokens)}, val={human_num(val_tokens)}), time={dt:.2f}s, tok/s={human_num(total/max(dt,1e-6))}"
    )

    meta = DataMeta(
        tokenizer="byte_utf8",
        vocab_size=tokenizer.vocab_size,
        dtype="uint8",
        train_tokens=int(train_tokens),
        val_tokens=int(val_tokens),
        created_at=now_str(),
        source_dir=os.path.abspath(os.path.dirname(files[0])) if files else "",
        files=[os.path.relpath(f, os.path.abspath(os.path.dirname(files[0]))) if files else f for f in files],
        jsonl_field=jsonl_field,
        val_ratio=float(val_ratio),
    )
    dump_json(meta_path, meta.__dict__)
    return meta
