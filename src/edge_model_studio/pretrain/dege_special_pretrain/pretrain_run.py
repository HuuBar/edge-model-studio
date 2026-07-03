from __future__ import annotations

import argparse
import os
import signal
import time
from typing import Dict, Tuple

import torch

from pretrain_data import (
    ByteTokenizer,
    build_bins,
    list_input_files,
    load_meta,
    maybe_generate_synthetic_dataset,
)
from pretrain_dataset import RandomBlockDataset, make_loader
from pretrain_model import GPTLike
from pretrain_utils import (
    CosineWithWarmup,
    Logger,
    autocast_context,
    cleanup_distributed,
    dist_barrier,
    get_cpu_mem_stats,
    get_device_and_ddp,
    get_gpu_mem_stats,
    human_num,
    init_distributed_if_needed,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--data_dir", type=str, default="./raw_data")
    p.add_argument("--out_dir", type=str, default="./out_pretrain")
    p.add_argument("--reprocess", action="store_true")
    p.add_argument("--jsonl_field", type=str, default="text")

    p.add_argument("--val_ratio", type=float, default=0.01)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--pin_memory", action="store_true")

    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_embd", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--clip_grad", type=float, default=1.0)

    p.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--compile", action="store_true")

    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--resume", type=str, default="")

    p.add_argument("--benchmark_only", action="store_true")
    p.add_argument("--dl_bench_steps", type=int, default=200)
    p.add_argument("--train_bench_steps", type=int, default=100)

    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--ddp", action="store_true")
    return p.parse_args()


@torch.no_grad()
def estimate_loss(model: torch.nn.Module, loader, device: torch.device, iters: int, dtype_str: str) -> float:
    model.eval()
    it = iter(loader)
    losses = []
    for _ in range(iters):
        x, y = next(it)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast_context(device, dtype_str):
            _, loss = model(x, y)
        losses.append(float(loss.item()))
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def benchmark_dataloader(loader, steps: int, block_size: int, batch_size: int, logger: Logger) -> None:
    t0 = time.time()
    it = iter(loader)
    for _ in range(steps):
        _ = next(it)
    dt = time.time() - t0
    samples = steps * batch_size
    tokens = samples * block_size
    logger.log0(f"dataloader bench: samples/s={human_num(samples/max(dt,1e-6))}, tokens/s={human_num(tokens/max(dt,1e-6))}")


def benchmark_train_steps(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader,
    device: torch.device,
    dtype_str: str,
    steps: int,
    grad_accum: int,
    logger: Logger,
) -> None:
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and dtype_str == "fp16"))
    it = iter(loader)

    # warmup
    for _ in range(5):
        x, y = next(it)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast_context(device, dtype_str):
            _, loss = model(x, y)
        loss = loss / max(1, grad_accum)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    t0 = time.time()
    total_tokens = 0
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            x, y = next(it)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast_context(device, dtype_str):
                _, loss = model(x, y)
            loss = loss / max(1, grad_accum)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total_tokens += int(x.numel())

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dt = time.time() - t0
    logger.log0(f"train-step bench: tokens/s={human_num(total_tokens/max(dt,1e-6))}, gpu={get_gpu_mem_stats(device)}")


def main() -> int:
    args = parse_args()
    device, ddp_info, ddp = get_device_and_ddp(args.ddp, args.cpu)
    rank = ddp_info["rank"]
    logger = Logger(rank=rank)

    logger.log0(f"device={device}, ddp={ddp}, world_size={ddp_info['world_size']}")
    logger.log0(f"cpu/mem: {get_cpu_mem_stats()}")

    stop = {"flag": False}

    def _handle(sig, frame):
        stop["flag"] = True
        logger.log(f"signal {sig} received; stop after current step.")

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    init_distributed_if_needed(ddp, device)
    set_seed(args.seed, rank=rank)

    maybe_generate_synthetic_dataset(args.data_dir, logger)
    files = list_input_files(args.data_dir, logger)
    if not files:
        logger.log0("no raw files found; exit.")
        cleanup_distributed(ddp)
        return 2

    tok = ByteTokenizer()
    train_bin = os.path.join(args.out_dir, "train.bin")
    val_bin = os.path.join(args.out_dir, "val.bin")

    meta = load_meta(args.out_dir)
    need = args.reprocess or (meta is None) or (not os.path.exists(train_bin)) or (not os.path.exists(val_bin))
    if need and rank == 0:
        logger.log0("building train/val bins...")
        _ = build_bins(
            files=files,
            out_dir=args.out_dir,
            tokenizer=tok,
            jsonl_field=args.jsonl_field,
            val_ratio=args.val_ratio,
            logger=logger,
        )
    dist_barrier(ddp)

    meta = load_meta(args.out_dir)
    if meta is None:
        logger.log0("meta.json missing; exit.")
        cleanup_distributed(ddp)
        return 3

    train_ds = RandomBlockDataset(train_bin, args.block_size, args.seed + 1_000_000 * rank)
    val_ds = RandomBlockDataset(val_bin, args.block_size, args.seed + 2_000_000 * rank)

    pin = bool(args.pin_memory and device.type == "cuda")
    train_loader = make_loader(train_ds, args.batch_size, args.num_workers, pin, args.prefetch_factor)
    val_loader = make_loader(val_ds, args.batch_size, max(1, min(args.num_workers, 2)), pin, args.prefetch_factor)

    if rank == 0:
        benchmark_dataloader(train_loader, args.dl_bench_steps, args.block_size, args.batch_size, logger)

    model = GPTLike(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    ).to(device)

    if args.compile:
        try:
            model = torch.compile(model)  # type: ignore
            logger.log0("torch.compile enabled.")
        except Exception as e:
            logger.log0(f"torch.compile failed: {repr(e)}")

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[ddp_info["local_rank"]] if device.type == "cuda" else None,
            output_device=ddp_info["local_rank"] if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scheduler = CosineWithWarmup(optimizer, args.warmup_steps, args.max_steps, args.min_lr_ratio)

    if rank == 0:
        benchmark_train_steps(
            model=(model.module if ddp else model),
            optimizer=optimizer,
            loader=train_loader,
            device=device,
            dtype_str=args.dtype,
            steps=args.train_bench_steps,
            grad_accum=args.grad_accum,
            logger=logger,
        )

    if args.benchmark_only:
        logger.log0("benchmark_only; exit.")
        cleanup_distributed(ddp)
        return 0

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        start_step = load_checkpoint(args.resume, (model.module if ddp else model), optimizer, scheduler, device, logger)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.dtype == "fp16"))
    train_iter = iter(train_loader)

    logger.log0(
        f"train start: steps={args.max_steps}, bs={args.batch_size}, T={args.block_size}, accum={args.grad_accum}, dtype={args.dtype}"
    )

    step_times = []
    tok_seen = 0

    for step in range(start_step, args.max_steps):
        if stop["flag"]:
            logger.log0("stop flag set; break.")
            break

        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        loss_accum = 0.0
        for _ in range(args.grad_accum):
            x, y = next(train_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast_context(device, args.dtype):
                _, loss = model(x, y)
            loss = loss / max(1, args.grad_accum)
            loss_accum += float(loss.item())

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            tok_seen += int(x.numel())

        if args.clip_grad > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = time.time() - t0
        step_times.append(dt)

        if (step + 1) % args.log_interval == 0 and rank == 0:
            window = step_times[-args.log_interval :]
            avg_dt = sum(window) / max(1, len(window))
            tok_s = (args.batch_size * args.block_size * args.grad_accum) / max(avg_dt, 1e-6)
            lr = optimizer.param_groups[0]["lr"]
            logger.log0(
                f"step={step+1} loss={loss_accum:.4f} lr={lr:.3e} dt={avg_dt:.4f}s tok/s={human_num(tok_s)} gpu={get_gpu_mem_stats(device)}"
            )

        if (step + 1) % args.eval_interval == 0 and rank == 0:
            base_model = model.module if ddp else model
            tr = estimate_loss(base_model, train_loader, device, max(5, args.eval_iters // 5), args.dtype)
            va = estimate_loss(base_model, val_loader, device, args.eval_iters, args.dtype)
            logger.log0(f"eval step={step+1} train_loss={tr:.4f} val_loss={va:.4f}")

        if (step + 1) % args.save_interval == 0:
            save_checkpoint(
                out_dir=args.out_dir,
                step=step + 1,
                model=(model.module if ddp else model),
                optimizer=optimizer,
                scheduler=scheduler,
                args_dict=vars(args),
                data_meta=meta,
                logger=logger,
                ddp=ddp,
                rank=rank,
            )

    if rank == 0:
        save_checkpoint(
            out_dir=args.out_dir,
            step=step + 1,
            model=(model.module if ddp else model),
            optimizer=optimizer,
            scheduler=scheduler,
            args_dict=vars(args),
            data_meta=meta,
            logger=logger,
            ddp=ddp,
            rank=rank,
        )
        logger.log0(f"done. tokens_seen={human_num(tok_seen)}")

    cleanup_distributed(ddp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
