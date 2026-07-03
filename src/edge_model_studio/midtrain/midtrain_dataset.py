from __future__ import annotations

import os
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch


def file_n_tokens(bin_path: str, dtype: np.dtype = np.uint8) -> int:
    if not os.path.exists(bin_path):
        return 0
    return int(os.path.getsize(bin_path) // np.dtype(dtype).itemsize)


class MultiDomainRandomBlockDataset(torch.utils.data.IterableDataset):
    """
    Infinite dataset:
      - choose a domain by weights
      - sample random contiguous block from that domain's token bin
    """
    def __init__(
        self,
        domain_bins: Dict[str, str],
        domain_weights: Dict[str, float],
        block_size: int,
        seed: int,
        dtype: np.dtype = np.uint8,
    ):
        super().__init__()
        self.domain_bins = dict(domain_bins)
        self.domain_weights = dict(domain_weights)
        self.block_size = int(block_size)
        self.seed = int(seed)
        self.dtype = dtype

        # validate
        self.domains: List[str] = sorted(self.domain_bins.keys())
        if not self.domains:
            raise RuntimeError("no domains provided")

        w = []
        for d in self.domains:
            wv = float(self.domain_weights.get(d, 1.0))
            if wv < 0:
                wv = 0.0
            w.append(wv)
        s = sum(w)
        if s <= 0:
            w = [1.0 for _ in w]
            s = float(len(w))
        self.weights = np.asarray([x / s for x in w], dtype=np.float64)

        # token counts
        self.n_tokens: Dict[str, int] = {}
        need = self.block_size + 2
        for d, p in self.domain_bins.items():
            n = file_n_tokens(p, self.dtype)
            if n < need:
                raise RuntimeError(f"domain={d} token file too small: {p} has {n}, need >= {need}")
            self.n_tokens[d] = int(n)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        worker = torch.utils.data.get_worker_info()
        wid = 0 if worker is None else int(worker.id)
        rng = np.random.default_rng(self.seed + 10_000 * wid)

        # open memmaps lazily per worker
        memmaps: Dict[str, np.memmap] = {d: np.memmap(self.domain_bins[d], dtype=self.dtype, mode="r") for d in self.domains}

        max_starts = {d: self.n_tokens[d] - (self.block_size + 2) for d in self.domains}

        while True:
            dn = rng.choice(self.domains, p=self.weights)
            start = int(rng.integers(0, max_starts[dn]))
            data = memmaps[dn]
            chunk = np.asarray(data[start : start + self.block_size + 1], dtype=np.int64)
            x = torch.from_numpy(chunk[:-1].copy())
            y = torch.from_numpy(chunk[1:].copy())
            yield x, y


def make_loader(
    ds: torch.utils.data.IterableDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
