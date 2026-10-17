"""Boucle d'entraînement BlackCode (CPU ou T4).

Exemple Colab T4 :

    python -m blackcode.train \
      --config configs/t4_small.json \
      --data data/code \
      --out runs/t4_small \
      --batch-size 8 --grad-accum 4 --seq-len 1024 \
      --max-steps 30000 --lr 6e-4 --grad-checkpoint

Le script :
- utilise l'AMP fp16 sur CUDA (indispensable sur T4) ;
- calcule la perte par morceaux de logits (la tête vocabulaire est le poste
  mémoire dominant quand le vocabulaire fait 32k) ;
- sauvegarde régulièrement pour survivre aux coupures Colab ;
- reprend automatiquement avec ``--resume``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig
from .data import TokenDataset
from .model import BlackCode, build_model


def get_lr(step: int, warmup: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * (step + 1) / max(1, warmup)
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * ratio))


def build_optimizer(model: BlackCode, weight_decay: float, lr: float, betas):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8)


@torch.no_grad()
def estimate_loss(model: BlackCode, dataset: TokenDataset, batch_size: int,
                  seq_len: int, iters: int, device: str, logit_chunk: int) -> dict:
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(iters)
        for i in range(iters):
            x, y = dataset.get_batch(split, batch_size, seq_len)
            loss = model.forward_loss(x, y, logit_chunk=logit_chunk)
            losses[i] = loss.detach().float().cpu()
        out[split] = losses.mean().item()
    model.train()
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entraîne BlackCode")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr", type=float, default=6e-5)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--logit-chunk", type=int, default=256)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(int(os.environ.get("BLACKCODE_THREADS", os.cpu_count() or 1)))

    cfg = ModelConfig.from_json(args.config)
    seq_len = args.seq_len or cfg.max_seq_len
    if seq_len > cfg.max_seq_len:
        raise ValueError(f"seq_len {seq_len} > max_seq_len {cfg.max_seq_len}")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if args.dtype == "auto":
        dtype = "float16" if device.startswith("cuda") else "float32"
    else:
        dtype = args.dtype
    if device == "cpu":
        dtype = "float32"
    torch.set_float32_matmul_precision("high")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {cfg.name}: {n_params/1e6:.2f} M paramètres, "
          f"{cfg.n_attention_layers} couches d'attention / {cfg.n_layers}")
    print(f"[device] {device} | dtype {dtype} | seq_len {seq_len} | "
          f"batch {args.batch_size} × accum {args.grad_accum}")

    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            print("[compile] torch.compile activé")
        except Exception as exc:  # pragma: no cover
            print(f"[compile] ignoré : {exc}")

    dataset = TokenDataset(args.data, device=device)
    print(f"[data] train {dataset.train_tokens/1e6:.2f} M tokens | "
          f"val {dataset.val_tokens/1e6:.2f} M tokens")

    optimizer = build_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2))
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "float16"))

    start_step = 0
    best_val = float("inf")
    ckpt_path = out_dir / "ckpt.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"] + 1
        best_val = ckpt.get("best_val", best_val)
        print(f"[resume] reprise à l'étape {start_step}")

    warmup = args.warmup if args.warmup is not None else max(10, int(0.02 * args.max_steps))
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
                    if dtype == "float16" else nullcontext())
    log_path = out_dir / "train_log.jsonl"
    tokens_per_step = args.batch_size * args.grad_accum * seq_len

    model.train()
    t0 = time.time()
    last_log_time = t0
    last_log_tokens = start_step * tokens_per_step

    for step in range(start_step, args.max_steps):
        lr = get_lr(step, warmup, args.max_steps, args.lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(args.grad_accum):
            x, y = dataset.get_batch("train", args.batch_size, seq_len)
            with autocast_ctx:
                loss = model.forward_loss(x, y, logit_chunk=args.logit_chunk,
                                          checkpoint=args.grad_checkpoint)
            loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            loss_accum += loss.detach().item()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % args.log_interval == 0:
            now = time.time()
            elapsed = max(now - last_log_time, 1e-9)
            tokens = (step + 1) * tokens_per_step
            tok_s = (tokens - last_log_tokens) / elapsed
            remaining = (args.max_steps - step - 1) * tokens_per_step / max(tok_s, 1e-9)
            mem = ""
            if device.startswith("cuda"):
                mem = f" | VRAM {torch.cuda.max_memory_allocated()/1e9:.2f} Go"
                torch.cuda.reset_peak_memory_stats()
            print(f"step {step:6d} | loss {loss_accum:.4f} | lr {lr:.2e} | "
                  f"{tok_s/1e3:.1f} k tok/s | ETA {remaining/3600:.2f} h{mem}")
            record = {"step": step, "loss": loss_accum, "lr": lr,
                      "tokens_per_s": tok_s, "elapsed_s": now - t0}
            with log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            last_log_time, last_log_tokens = now, tokens

        if (step + 1) % args.eval_interval == 0 or step == args.max_steps - 1:
            metrics = estimate_loss(model, dataset, args.batch_size, seq_len,
                                    args.eval_iters, device, args.logit_chunk)
            print(f"[eval] step {step} | train {metrics['train']:.4f} | val {metrics['val']:.4f}")
            with log_path.open("a") as f:
                f.write(json.dumps({"step": step, "eval": metrics}) + "\n")
            if metrics["val"] < best_val:
                best_val = metrics["val"]
                torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(),
                            "step": step, "best_val": best_val}, out_dir / "best.pt")
            model.train()

        if (step + 1) % args.save_interval == 0 or step == args.max_steps - 1:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if dtype == "float16" else None,
                "step": step,
                "best_val": best_val,
                "cfg": cfg.to_dict(),
            }, ckpt_path)
            print(f"[ckpt] sauvegarde {ckpt_path} (step {step})")

    final_state = {"model": model.state_dict(), "cfg": cfg.to_dict(),
                   "step": args.max_steps - 1, "best_val": best_val}
    torch.save(final_state, out_dir / "final.pt")
    if not (out_dir / "best.pt").exists():
        torch.save(final_state, out_dir / "best.pt")
    print(f"[done] terminé en {(time.time()-t0)/3600:.2f} h | meilleure val {best_val:.4f}")


if __name__ == "__main__":
    main()
