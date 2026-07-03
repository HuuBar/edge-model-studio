from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import torch

from pretrain_data import file_n_tokens


class RandomBlockDataset(torch.utils.data.IterableDataset):
    def __init__(self, bin_path: str, block_size: int, seed: int, dtype: np.dtype = np.uint8):
        super().__init__()
        self.bin_path = bin_path
        self.block_size = int(block_size)
        self.seed = int(seed)
        self.dtype = dtype

        n = file_n_tokens(bin_path, dtype)
        need = self.block_size + 2
        if n < need:
            raise RuntimeError(f"token file too small: {bin_path} has {n} tokens; need >= {need}.")
        self.n_tokens = n

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        worker = torch.utils.data.get_worker_info()
        wid = 0 if worker is None else int(worker.id)
        rng = np.random.default_rng(self.seed + 10_000 * wid)
        data = np.memmap(self.bin_path, dtype=self.dtype, mode="r")

        max_start = self.n_tokens - (self.block_size + 2)
        while True:
            start = int(rng.integers(0, max_start))
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
