from __future__ import annotations

import datetime as _dt
import json
import math
import os
import random
import time
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def now_str() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def human_bytes(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0:
            return f"{x:.2f}{u}"
        x /= 1024.0
    return f"{x:.2f}PB"


def human_num(n: float) -> str:
    units = ["", "K", "M", "B", "T"]
    x = float(n)
    for u in units:
        if abs(x) < 1000.0:
            return f"{x:.2f}{u}"
        x /= 1000.0
    return f"{x:.2f}P"


class Logger:
    def __init__(self, rank: int = 0):
        self.rank = rank

    def log(self, msg: str) -> None:
        print(f"[{now_str()}][rank={self.rank}] {msg}", flush=True)

    def log0(self, msg: str) -> None:
        if self.rank == 0:
            self.log(msg)


def try_import_psutil():
    try:
        import psutil  # type: ignore

        return psutil
    except Exception:
        return None


def get_cpu_mem_stats() -> Dict[str, str]:
    psutil = try_import_psutil()
    if psutil is None:
        return {"cpu": "psutil_not_installed", "ram": "psutil_not_installed"}
    vm = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=None)
    return {
        "cpu": f"{cpu:.1f}%",
        "ram_used": human_bytes(vm.used),
        "ram_total": human_bytes(vm.total),
        "ram_pct": f"{vm.percent:.1f}%",
    }


def get_gpu_mem_stats(device: torch.device) -> Dict[str, str]:
    if device.type != "cuda":
        return {"gpu": "cpu_only"}
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return {
        "gpu_free": human_bytes(free),
        "gpu_total": human_bytes(total),
        "gpu_alloc": human_bytes(torch.cuda.memory_allocated(device)),
        "gpu_reserved": human_bytes(torch.cuda.memory_reserved(device)),
    }


def set_seed(seed: int, rank: int = 0) -> None:
    s = seed + 1000 * rank
    random.seed(s)
    np.random.seed(s % (2**32 - 1))
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def get_device_and_ddp(enable_ddp: bool, force_cpu: bool) -> Tuple[torch.device, Dict[str, int], bool]:
    ddp = False
    rank = 0
    local_rank = 0
    world_size = 1
    if enable_ddp:
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        ddp = world_size > 1

    if torch.cuda.is_available() and (not force_cpu):
        device = torch.device("cuda", local_rank if ddp else 0)
        if ddp:
            torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    return device, {"rank": rank, "local_rank": local_rank, "world_size": world_size}, ddp


def init_distributed_if_needed(ddp: bool, device: torch.device) -> None:
    if not ddp:
        return
    backend = "nccl" if device.type == "cuda" else "gloo"
    torch.distributed.init_process_group(backend=backend)


def dist_barrier(ddp: bool) -> None:
    if ddp and torch.distributed.is_initialized():
        torch.distributed.barrier()


def cleanup_distributed(ddp: bool) -> None:
    if ddp and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def autocast_context(device: torch.device, dtype_str: str):
    if device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    if dtype_str == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
    if dtype_str == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    return torch.autocast(device_type="cuda", enabled=False)


class CosineWithWarmup:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
        self.opt = optimizer
        self.warmup = int(max(1, warmup_steps))
        self.total = int(max(self.warmup + 1, total_steps))
        self.min_lr_ratio = float(min_lr_ratio)
        self.step_num = 0
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self) -> None:
        self.step_num += 1
        t = self.step_num
        for i, g in enumerate(self.opt.param_groups):
            base = self.base_lrs[i]
            if t <= self.warmup:
                lr = base * (t / self.warmup)
            else:
                p = (t - self.warmup) / max(1, (self.total - self.warmup))
                p = min(max(p, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * p))
                lr = base * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine)
            g["lr"] = lr

    def state_dict(self) -> Dict[str, int]:
        return {"step_num": self.step_num}

    def load_state_dict(self, sd: Dict[str, int]) -> None:
        self.step_num = int(sd.get("step_num", 0))


def save_checkpoint(
    out_dir: str,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineWithWarmup,
    args_dict: Dict[str, Any],
    data_meta: Any,
    logger: Logger,
    ddp: bool,
    rank: int,
) -> str:
    if ddp and rank != 0:
        return ""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ckpt_step_{step}.pt")
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": args_dict,
        "data_meta": asdict(data_meta) if hasattr(data_meta, "__dataclass_fields__") else data_meta,
        "saved_at": now_str(),
    }
    torch.save(payload, path)
    logger.log0(f"checkpoint saved: {path}")
    return path


def load_checkpoint(
    ckpt_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineWithWarmup,
    device: torch.device,
    logger: Logger,
) -> int:
    obj = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    scheduler.load_state_dict(obj.get("scheduler", {}))
    step = int(obj.get("step", 0))
    logger.log(f"resumed: {ckpt_path}, step={step}")
    return step


def dump_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
