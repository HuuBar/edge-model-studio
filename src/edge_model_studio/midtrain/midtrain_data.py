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
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, ids: List[int]) -> str:
        return bytes(int(x) & 0xFF for x in ids).decode("utf-8", errors="replace")


# -----------------------------
# Synthetic mid-train data
# -----------------------------

_SYN_HEALTH = [
    "今天跑步 8 公里，配速 5:05，心率主要在 2 区，恢复良好。",
    "心率漂移偏高，建议降低强度，增加补水，并保证睡眠。",
    "训练计划：周二间歇，周四节奏跑，周末 LSD，注意热身和拉伸。",
]
_SYN_CODE = [
    "def topk(xs,k):\n    return sorted(xs, reverse=True)[:k]\n",
    "import torch\nx=torch.randn(4,8)\ny=x.softmax(dim=-1)\nprint(y[0])\n",
    "for i in range(10):\n    print(i*i)\n",
]
_SYN_GENERAL = [
    "我们把数据按域拆分，设置采样权重，并记录 tokens/s 作为吞吐指标。",
    "mid-train 用来做域适配：在不改变架构的情况下提升垂域表现。",
    "为了更稳定，需要 warmup、梯度裁剪和合适的 batch/accum。",
]


def maybe_generate_synthetic_midtrain(data_dir: str, logger: Logger) -> None:
    os.makedirs(data_dir, exist_ok=True)
    existing = glob.glob(os.path.join(data_dir, "**", "*.*"), recursive=True)
    existing = [p for p in existing if os.path.isfile(p)]
    if existing:
        logger.log0(f"midtrain data_dir has {len(existing)} files; skip synthetic generation.")
        return

    logger.log0("midtrain data_dir empty; generating synthetic multi-domain dataset.")
    rng = random.Random(20250105)

    domains = {
        "sports_health": _SYN_HEALTH,
        "code": _SYN_CODE,
        "general": _SYN_GENERAL,
    }

    for dn, pool in domains.items():
        dpath = os.path.join(data_dir, dn)
        os.makedirs(dpath, exist_ok=True)
        # text
        tpath = os.path.join(dpath, f"{dn}.txt")
        with open(tpath, "w", encoding="utf-8") as f:
            for _ in range(12000):
                s = rng.choice(pool)
                # inject a little randomness
                if dn == "sports_health":
                    km = rng.uniform(2.0, 20.0)
                    hr = rng.randint(95, 190)
                    s = f"{s} 距离={km:.2f}km, avg_hr={hr}bpm."
                elif dn == "general":
                    w = rng.uniform(0.2, 3.0)
                    s = f"{s} weight={w:.3f}"
                f.write(s + "\n")

        # jsonl
        jpath = os.path.join(dpath, f"{dn}.jsonl")
        with open(jpath, "w", encoding="utf-8") as f:
            for i in range(2500):
                obj = {
                    "id": i,
                    "text": rng.choice(pool),
                    "domain": dn,
                    "ts": int(time.time()) - rng.randint(0, 86400 * 60),
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    logger.log0(f"synthetic midtrain dataset generated in: {data_dir}")


# -----------------------------
# Discover domains / read text
# -----------------------------

def list_domain_files(domain_dir: str) -> List[str]:
    pats = ["**/*.txt", "**/*.text", "**/*.log", "**/*.jsonl", "**/*.json"]
    files: List[str] = []
    for pat in pats:
        files.extend(glob.glob(os.path.join(domain_dir, pat), recursive=True))
    files = [p for p in files if os.path.isfile(p)]
    return sorted(set(files))


def discover_domains(data_dir: str, logger: Logger) -> Dict[str, List[str]]:
    # If has subdirs -> each subdir a domain
    subdirs = [p for p in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(p)]
    dom_map: Dict[str, List[str]] = {}

    if subdirs:
        for sd in sorted(subdirs):
            dn = os.path.basename(sd)
            fs = list_domain_files(sd)
            if fs:
                dom_map[dn] = fs
    else:
        fs = list_domain_files(data_dir)
        if fs:
            dom_map["default"] = fs

    logger.log0(f"discovered domains={list(dom_map.keys())}")
    for dn, fs in dom_map.items():
        logger.log0(f"  domain={dn}, files={len(fs)}")
    return dom_map


def iter_text_from_file(path: str, jsonl_field: str = "text") -> Iterator[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".txt", ".text", ".log"]:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
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
            for it in obj:
                if isinstance(it, dict) and isinstance(it.get(jsonl_field, ""), str):
                    yield it[jsonl_field]
                else:
                    yield json.dumps(it, ensure_ascii=False)
        else:
            yield json.dumps(obj, ensure_ascii=False)
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if line.strip():
                    yield line


def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s


# -----------------------------
# Build per-domain bins
# -----------------------------

@dataclass
class MidMeta:
    tokenizer: str
    vocab_size: int
    dtype: str
    created_at: str
    source_dir: str
    val_ratio: float
    jsonl_field: str
    domain_tokens_train: Dict[str, int]
    domain_tokens_val: Dict[str, int]
    domain_docs: Dict[str, int]
    domain_prefix: bool


def load_mid_meta(out_dir: str) -> Optional[MidMeta]:
    p = os.path.join(out_dir, "mid_meta.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return MidMeta(**obj)


def _bin_paths(out_dir: str, domain: str) -> Tuple[str, str]:
    tr = os.path.join(out_dir, f"mid_train_{domain}.bin")
    va = os.path.join(out_dir, f"mid_val_{domain}.bin")
    return tr, va


def build_mid_bins(
    dom_map: Dict[str, List[str]],
    out_dir: str,
    tokenizer: ByteTokenizer,
    val_ratio: float,
    logger: Logger,
    jsonl_field: str = "text",
    domain_prefix: bool = True,
    seed: int = 20250106,
) -> MidMeta:
    os.makedirs(out_dir, exist_ok=True)

    # cleanup old
    for dn in dom_map.keys():
        tr, va = _bin_paths(out_dir, dn)
        for p in [tr, va]:
            if os.path.exists(p):
                os.remove(p)
    meta_path = os.path.join(out_dir, "mid_meta.json")
    if os.path.exists(meta_path):
        os.remove(meta_path)

    rng = random.Random(seed)
    t0 = time.time()

    domain_tokens_train: Dict[str, int] = {}
    domain_tokens_val: Dict[str, int] = {}
    domain_docs: Dict[str, int] = {}

    bytes_in = 0
    total_tokens = 0
    last_log = time.time()
    last_total = 0

    for dn, files in dom_map.items():
        tr_path, va_path = _bin_paths(out_dir, dn)
        docs = 0
        tok_tr = 0
        tok_va = 0

        with open(tr_path, "wb") as ftr, open(va_path, "wb") as fva:
            for fp in files:
                try:
                    bytes_in += os.stat(fp).st_size
                except Exception:
                    pass

                for text in iter_text_from_file(fp, jsonl_field=jsonl_field):
                    docs += 1
                    text = normalize_text(text).strip()
                    if not text:
                        continue

                    if domain_prefix:
                        text = f"<domain:{dn}>\n" + text

                    # add separator newline
                    text = text + "\n"
                    ids = tokenizer.encode(text)
                    if len(ids) < 2:
                        continue

                    arr = np.asarray(ids, dtype=np.uint8)
                    if rng.random() < val_ratio:
                        fva.write(arr.tobytes())
                        tok_va += int(arr.size)
                    else:
                        ftr.write(arr.tobytes())
                        tok_tr += int(arr.size)

                    total_tokens += int(arr.size)
                    if time.time() - last_log >= 2.0:
                        dt = time.time() - t0
                        tok_s = total_tokens / max(dt, 1e-6)
                        delta = total_tokens - last_total
                        last_total = total_tokens
                        logger.log0(
                            f"mid preprocess: tokens={human_num(total_tokens)}, tok/s={human_num(tok_s)}, recent+{human_num(delta)}"
                        )
                        last_log = time.time()

        domain_tokens_train[dn] = int(tok_tr)
        domain_tokens_val[dn] = int(tok_va)
        domain_docs[dn] = int(docs)
        logger.log0(
            f"domain done: {dn} docs={docs}, train_tok={human_num(tok_tr)}, val_tok={human_num(tok_va)}"
        )

    dt = time.time() - t0
    logger.log0(
        f"mid preprocess done: input={human_bytes(bytes_in)}, tokens={human_num(total_tokens)}, "
        f"time={dt:.2f}s, tok/s={human_num(total_tokens/max(dt,1e-6))}"
    )

    meta = MidMeta(
        tokenizer="byte_utf8",
        vocab_size=tokenizer.vocab_size,
        dtype="uint8",
        created_at=now_str(),
        source_dir=os.path.abspath(out_dir),
        val_ratio=float(val_ratio),
        jsonl_field=str(jsonl_field),
        domain_tokens_train=domain_tokens_train,
        domain_tokens_val=domain_tokens_val,
        domain_docs=domain_docs,
        domain_prefix=bool(domain_prefix),
    )
    dump_json(meta_path, meta.__dict__)
    return meta
