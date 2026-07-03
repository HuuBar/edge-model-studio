from __future__ import annotations

import argparse
import os
import signal
import time
from typing import Dict, Tuple

import torch

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
    save_checkpoint,
    set_seed,
)
from midtrain_data import (
    ByteTokenizer,
    build_mid_bins,
    discover_domains,
    load_mid_meta,
    maybe_generate_synthetic_midtrain,
)
from midtrain_dataset import MultiDomainRandomBlockDataset, make_loader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--data_dir", type=str, default="./mid_data")
    p.add_argument("--out_dir", type=str, default="./out_mid")
    p.add_argument("--reprocess", action="store_true")
    p.add_argument("--val_ratio", type=float, default=0.01)
    p.add_argument("--jsonl_field", type=str, default="text")
    p.add_argument("--domain_prefix", action="store_true")

    # domain weights: "general=1,sports_health=3,code=1"
    p.add_argument("--domain_weights", type=str, default="")

    # data
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--pin_memory", action="store_true")

    # model
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_embd", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)

    # train
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--clip_grad", type=float, default=1.0)

    p.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--compile", action="store_true")

    # load base weights
    p.add_argument("--base_ckpt", type=str, default="", help="load pretrained weights from ckpt (pretrain output).")

    # log/ckpt
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--save_interval", type=int, default=500)

    # bench
    p.add_argument("--benchmark_only", action="store_true")
    p.add_argument("--dl_bench_steps", type=int, default=200)
    p.add_argument("--train_bench_steps", type=int, default=100)

    # system
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--ddp", action="store_true")
    return p.parse_args()


def parse_domain_weights(s: str) -> Dict[str, float]:
    # "a=1,b=2"
    out: Dict[str, float] = {}
    if not s.strip():
        return out
    parts = [x.strip() for x in s.split(",") if x.strip()]
    for it in parts:
        if "=" not in it:
            continue
        k, v = it.split("=", 1)
        k = k.strip()
        try:
            out[k] = float(v.strip())
        except Exception:
            out[k] = 1.0
    return out


def build_domain_bins_paths(out_dir: str, doms: Dict[str, list], split: str) -> Dict[str, str]:
    # split: "train" or "val"
    m: Dict[str, str] = {}
    for d in doms.keys():
        m[d] = os.path.join(out_dir, f"mid_{split}_{d}.bin")
    return m


def make_weights_from_meta(meta, split: str = "train") -> Dict[str, float]:
    # token-proportional weights as default
    dct = meta.domain_tokens_train if split == "train" else meta.domain_tokens_val
    w: Dict[str, float] = {}
    for d, n in dct.items():
        w[d] = float(max(1, int(n)))
    return w


def load_base_weights_flexible(model: torch.nn.Module, ckpt_path: str, logger: Logger) -> None:
    if not ckpt_path or (not os.path.exists(ckpt_path)):
        logger.log0("base_ckpt not provided or not found; skip loading base weights.")
        return

    obj = torch.load(ckpt_path, map_location="cpu")
    sd = obj.get("model", obj)
    # strip "module." just in case
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    # handle pos_emb mismatch
    if "pos_emb.weight" in sd:
        src = sd["pos_emb.weight"]
        dst = model.pos_emb.weight.data
        if src.shape != dst.shape:
            n = min(src.shape[0], dst.shape[0])
            logger.log0(f"pos_emb resize: ckpt={tuple(src.shape)} -> model={tuple(dst.shape)}, copy={n}")
            dst[:n].copy_(src[:n])
            if dst.shape[0] > n:
                # init remaining like normal(0,0.02)
                torch.nn.init.normal_(dst[n:], mean=0.0, std=0.02)
            sd["pos_emb.weight"] = dst.clone()

    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.log0(f"loaded base weights: {ckpt_path}")
    logger.log0(f"state_dict missing={len(missing)} unexpected={len(unexpected)}")


@torch.no_grad()
def estimate_loss(model: torch.nn.Module, loader, device: torch.device, dtype_str: str, steps: int) -> float:
    model.eval()
    it = iter(loader)
    losses = []
    for _ in range(steps):
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


def benchmark_train_steps(model, optimizer, loader, device, dtype_str, steps, grad_accum, logger: Logger) -> None:
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

    # Data discovery + synthetic fallback
    maybe_generate_synthetic_midtrain(args.data_dir, logger)
    dom_map = discover_domains(args.data_dir, logger)
    if not dom_map:
        logger.log0("no domains/files found; exit.")
        cleanup_distributed(ddp)
        return 2

    tok = ByteTokenizer()
    meta = load_mid_meta(args.out_dir)

    need = args.reprocess or (meta is None)
    if need and rank == 0:
        logger.log0("building midtrain bins...")
        _ = build_mid_bins(
            dom_map=dom_map,
            out_dir=args.out_dir,
            tokenizer=tok,
            val_ratio=args.val_ratio,
            logger=logger,
            jsonl_field=args.jsonl_field,
            domain_prefix=args.domain_prefix,
        )
    dist_barrier(ddp)
    meta = load_mid_meta(args.out_dir)
    if meta is None:
        logger.log0("mid_meta.json missing; exit.")
        cleanup_distributed(ddp)
        return 3

    # domain bins
    train_bins = build_domain_bins_paths(args.out_dir, dom_map, "train")
    val_bins = build_domain_bins_paths(args.out_dir, dom_map, "val")

    # weights: user override > meta token-proportional
    w_user = parse_domain_weights(args.domain_weights)
    if w_user:
        weights = {d: float(w_user.get(d, 1.0)) for d in train_bins.keys()}
    else:
        weights = make_weights_from_meta(meta, "train")

    if rank == 0:
        logger.log0(f"domain weights: {weights}")

    train_ds = MultiDomainRandomBlockDataset(train_bins, weights, args.block_size, args.seed + 1_000_000 * rank)
    val_ds = MultiDomainRandomBlockDataset(val_bins, weights, args.block_size, args.seed + 2_000_000 * rank)

    pin = bool(args.pin_memory and device.type == "cuda")
    train_loader = make_loader(train_ds, args.batch_size, args.num_workers, pin, args.prefetch_factor)
    val_loader = make_loader(val_ds, args.batch_size, max(1, min(args.num_workers, 2)), pin, args.prefetch_factor)

    if rank == 0:
        benchmark_dataloader(train_loader, args.dl_bench_steps, args.block_size, args.batch_size, logger)

    # Model
    model = GPTLike(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    ).to(device)

    # Load base weights BEFORE DDP wrap
    if rank == 0:
        logger.log0(f"base_ckpt={args.base_ckpt or '(none)'}")
    load_base_weights_flexible(model, args.base_ckpt, logger)
    dist_barrier(ddp)

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

    # Bench train steps
    if rank == 0:
        base = model.module if ddp else model
        benchmark_train_steps(base, optimizer, train_loader, device, args.dtype, args.train_bench_steps, args.grad_accum, logger)

    if args.benchmark_only:
        logger.log0("benchmark_only; exit.")
        cleanup_distributed(ddp)
        return 0

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.dtype == "fp16"))
    it = iter(train_loader)

    logger.log0(
        f"midtrain start: steps={args.max_steps}, bs={args.batch_size}, T={args.block_size}, accum={args.grad_accum}, dtype={args.dtype}"
    )

    step_times = []

    for step in range(0, args.max_steps):
        if stop["flag"]:
            logger.log0("stop flag set; break.")
            break

        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        loss_accum = 0.0
        for _ in range(args.grad_accum):
            x, y = next(it)
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
            base = model.module if ddp else model
            va = estimate_loss(base, val_loader, device, args.dtype, args.eval_steps)
            logger.log0(f"eval step={step+1} val_loss={va:.4f}")

        if (step + 1) % args.save_interval == 0:
            save_checkpoint(
                out_dir=args.out_dir,
                step=step + 1,
                model=(model.module if ddp else model),
                optimizer=optimizer,
                scheduler=scheduler,
                args_dict=vars(args),
                data_meta=meta.__dict__,
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
            data_meta=meta.__dict__,
            logger=logger,
            ddp=ddp,
            rank=rank,
        )
        logger.log0("done.")

    cleanup_distributed(ddp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
