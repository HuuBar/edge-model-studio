from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch


class SFTDataset(torch.utils.data.Dataset):
    """
    Reads:
      - sft_*.bin: concatenated uint8 tokens
      - sft_*_index.npy: int64 array [offset, length, resp_start]
    Returns fixed-length:
      x: (block_size,) int64
      y: (block_size,) int64 with prompt/pad masked as -100
    """
    def __init__(self, bin_path: str, index_path: str, block_size: int, pad_id: int = 0):
        super().__init__()
        if not os.path.exists(bin_path):
            raise FileNotFoundError(bin_path)
        if not os.path.exists(index_path):
            raise FileNotFoundError(index_path)

        self.bin_path = bin_path
        self.index_path = index_path
        self.block_size = int(block_size)
        self.pad_id = int(pad_id)

        self.index = np.load(index_path).astype(np.int64)
        if self.index.ndim != 2 or self.index.shape[1] != 3:
            raise RuntimeError(f"bad index shape: {self.index.shape}, expected (N,3)")

        self.data = np.memmap(bin_path, dtype=np.uint8, mode="r")

        # We need seq_len = block_size + 1 for shift
        self.max_seq = self.block_size + 1

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        offset, length, resp_start = self.index[i].tolist()
        offset = int(offset)
        length = int(length)
        resp_start = int(resp_start)

        # Load raw sequence
        seq = np.asarray(self.data[offset : offset + length], dtype=np.int64)
        if seq.size < 2:
            # Extremely rare; fabricate minimal seq
            seq = np.asarray([self.pad_id, self.pad_id], dtype=np.int64)
            resp_start = 0

        if seq.size > self.max_seq:
            cut = seq.size - self.max_seq
            seq = seq[cut:]
            resp_start = max(0, resp_start - cut)

        seq_len = int(seq.size)
        if seq_len < self.max_seq:
            pad_len = self.max_seq - seq_len
            seq = np.concatenate([seq, np.full((pad_len,), self.pad_id, dtype=np.int64)], axis=0)
            attn = np.concatenate([np.ones((seq_len,), dtype=np.int64), np.zeros((pad_len,), dtype=np.int64)], axis=0)
        else:
            attn = np.ones((self.max_seq,), dtype=np.int64)

        # Shift
        x = seq[:-1].copy()
        y = seq[1:].copy()
        attn_y = attn[1:].copy()  # labels alignment

        y[attn_y == 0] = -100

        cut_prompt = max(0, resp_start - 1)
        if cut_prompt > 0:
            y[:cut_prompt] = -100

        return torch.from_numpy(x), torch.from_numpy(y)
