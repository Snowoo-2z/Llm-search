"""Boucle d'entraînement BlackCode (CPU ou T4).

Objectif : **réduire le temps à taille de modèle fixée**. Le script expose donc
tous les leviers de ``blackcode.optim`` et ``blackcode.timing`` :

- ``--optim {adamw,muon,ademamix,adafactor,adamw8bit,paged_adamw8bit}``
- ``--schedule {cosine,wsd,constant}`` (WSD = MiniCPM, interruptible)
- ``--checkpoint-policy {none,selective,full}`` (sélectif = 1 bloc sur 2)
- ``--seq-len-warmup-frac`` : contexte court au début, long à la fin
- ``--optimizer-in-fp32`` / ``--fused``
- ``--mirror-dir`` : copie asynchrone des checkpoints (Drive lent)
- ``--log-flops`` : MFU réel à chaque log (le seul chiffre fiable)

Exemple Colab T4 :

    python -m blackcode.train \
      --config configs/t4_small.json --data data/code --out runs/t4_small \
      --batch-size 8 --grad-accum 4 --seq-len 1024 --max-steps 30000 \
      --optim muon --muon-lr 0.02 --lr 6e-4 \
      --schedule wsd --stable-frac 0.8 \
      --checkpoint-policy selective --compile --log-flops \
      --mirror-dir /content/drive/MyDrive/blackcode/runs/t4_small
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig
from .data import TokenDataset
from .model import BlackCode, build_model
from .optim import make_optimizer, schedule_multiplier
from .timing import MFU_PRESETS, flops_per_token


def _async_copy(src: Path, dst: Path) -> threading.Thread:
    """Copie un fichier dans un thread (les écritures Drive coûtent 10–60 s)."""

    def _run():
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(dst.suffix + ".tmp")
            with src.open("rb") as fsrc, tmp.open("wb") as fdst:
                while True:
                    chunk = fsrc.read(1 << 24)
                    if not chunk:
                        break
                    fdst.write(chunk)
            tmp.replace(dst)
        except Exception as exc:  # pragma: no cover
            print(f"[mirror] échec {dst}: {exc}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


@torch.no_grad()
def estimate_loss(model: BlackCode, dataset: TokenDataset, batch_size: int,
                  seq_len: int, iters: int, logit_chunk: int) -> dict:
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
    p.add_argument("--seq-len-warmup-frac", type=float, default=0.0,
                   help="fraction des pas où la longueur monte linéairement "
                        "(ex. 0.2 : 512 → seq_len sur les 20 %% premiers pas)")
    p.add_argument("--seq-len-start", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=1000)

    # Optimisation
    p.add_argument("--optim", type=str, default="adamw",
                   choices=["adamw", "muon", "ademamix", "adafactor",
                            "adamw8bit", "paged_adamw8bit"])
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr", type=float, default=6e-5)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--adafactor-lr", type=float, default=None,
                   help="par défaut 30× --lr (normalisation interne d'Adafactor)")
    p.add_argument("--muon-ns-steps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--schedule", type=str, default="cosine",
                   choices=["cosine", "wsd", "constant"])
    p.add_argument("--stable-frac", type=float, default=0.8,
                   help="part de la phase stable pour --schedule wsd")
    p.add_argument("--decay-shape", type=str, default="sqrt",
                   choices=["sqrt", "linear", "cosine"])
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--fused", action="store_true", help="AdamW fused (CUDA)")

    # Mémoire / débit
    p.add_argument("--logit-chunk", type=int, default=256)
    p.add_argument("--checkpoint-policy", type=str, default="none",
                   choices=["none", "selective", "full"])
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="alias de --checkpoint-policy full")
    p.add_argument("--compile", action="store_true")

    # Suivi / reprise
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--log-flops", action="store_true")
    p.add_argument("--peak-tflops", type=float, default=65.0,
                   help="pic fp16 du GPU (T4 = 65)")
    p.add_argument("--mirror-dir", type=str, default=None,
                   help="copie asynchrone des checkpoints (ex. Google Drive)")
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

    policy = "full" if args.grad_checkpoint else args.checkpoint_policy

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {cfg.name}: {n_params/1e6:.2f} M paramètres, "
          f"{cfg.n_attention_layers} couches d'attention / {cfg.n_layers}")
    print(f"[device] {device} | dtype {dtype} | seq_len {seq_len} | "
          f"batch {args.batch_size} × accum {args.grad_accum} | "
          f"optim {args.optim} | schedule {args.schedule} | ckpt {policy}")

    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            print("[compile] torch.compile activé")
        except Exception as exc:  # pragma: no cover
            print(f"[compile] ignoré : {exc}")

    dataset = TokenDataset(args.data, device=device)
    print(f"[data] train {dataset.train_tokens/1e6:.2f} M tokens | "
          f"val {dataset.val_tokens/1e6:.2f} M tokens")

    optimizer, base_lrs = make_optimizer(
        model, name=args.optim, lr=args.lr, muon_lr=args.muon_lr,
        weight_decay=args.weight_decay, betas=(args.beta1, args.beta2),
        fused=args.fused, alpha_warmup=max(1, args.max_steps // 2),
        adafactor_lr=args.adafactor_lr,
    )
    # Muon n'expose pas ns_steps dans make_optimizer : on le règle ici.
    for opt in getattr(optimizer, "optimizers", [optimizer]):
        for group in opt.param_groups:
            if "ns_steps" in group:
                group["ns_steps"] = args.muon_ns_steps
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "float16"))
    min_ratio = args.min_lr / max(args.lr, 1e-12)

    start_step = 0
    best_val = float("inf")
    ckpt_path = out_dir / "ckpt.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except (KeyError, ValueError) as exc:
            print(f"[resume] optimiseur incompatible, réinitialisé : {exc}")
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"] + 1
        best_val = ckpt.get("best_val", best_val)
        print(f"[resume] reprise à l'étape {start_step}")

    warmup = args.warmup if args.warmup is not None else max(10, int(0.02 * args.max_steps))
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
                    if dtype == "float16" else nullcontext())
    log_path = out_dir / "train_log.jsonl"
    mirror_dir = Path(args.mirror_dir) if args.mirror_dir else None
    mirror_threads: list[threading.Thread] = []

    def current_seq_len(step: int) -> int:
        if args.seq_len_warmup_frac <= 0:
            return seq_len
        ramp = max(1, int(args.max_steps * args.seq_len_warmup_frac))
        if step >= ramp:
            return seq_len
        ratio = step / ramp
        value = args.seq_len_start + (seq_len - args.seq_len_start) * ratio
        # multiple de 8 pour rester compatible avec les kernels
        return max(8, int(value // 8 * 8))

    fpt_final = flops_per_token(cfg, seq_len)["total"]
    model.train()
    t0 = time.time()
    last_log_time = t0
    last_log_tokens = 0
    tokens_seen = start_step * args.batch_size * args.grad_accum * seq_len

    for step in range(start_step, args.max_steps):
        mult = schedule_multiplier(
            step, args.schedule, warmup, args.max_steps, min_ratio,
            args.stable_frac, args.decay_shape,
        )
        for group, base in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base * mult

        step_seq = current_seq_len(step)
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(args.grad_accum):
            x, y = dataset.get_batch("train", args.batch_size, step_seq)
            with autocast_ctx:
                loss = model.forward_loss(x, y, logit_chunk=args.logit_chunk,
                                          checkpoint=policy)
            loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            loss_accum += loss.detach().item()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        tokens_seen += args.batch_size * args.grad_accum * step_seq

        if step % args.log_interval == 0:
            now = time.time()
            elapsed = max(now - last_log_time, 1e-9)
            tok_s = (tokens_seen - last_log_tokens) / elapsed
            remaining = (args.max_steps - step - 1) * args.batch_size * args.grad_accum * seq_len
            remaining_s = remaining / max(tok_s, 1e-9)
            mem = ""
            if device.startswith("cuda"):
                mem = f" | VRAM {torch.cuda.max_memory_allocated()/1e9:.2f} Go"
                torch.cuda.reset_peak_memory_stats()
            flops_s = fpt_final * (tokens_seen - last_log_tokens) / elapsed
            mfu = flops_s / (args.peak_tflops * 1e12)
            if not args.log_flops:
                flops_txt = ""
            elif device.startswith("cuda"):
                flops_txt = f" | {flops_s/1e12:5.1f} TFLOPS ({mfu:4.0%} MFU)"
            else:
                flops_txt = f" | {flops_s/1e9:5.1f} GFLOP/s"
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"step {step:6d} | loss {loss_accum:.4f} | lr {current_lr:.2e} | "
                  f"seq {step_seq} | {tok_s/1e3:5.1f} k tok/s | "
                  f"ETA {remaining_s/3600:.2f} h{flops_txt}{mem}")
            record = {"step": step, "loss": loss_accum, "lr_mult": mult,
                      "seq_len": step_seq, "tokens_per_s": tok_s,
                      "flops_per_s": flops_s, "mfu": mfu,
                      "elapsed_s": now - t0}
            with log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            last_log_time, last_log_tokens = now, tokens_seen

        if (step + 1) % args.eval_interval == 0 or step == args.max_steps - 1:
            metrics = estimate_loss(model, dataset, args.batch_size, seq_len,
                                    args.eval_iters, args.logit_chunk)
            print(f"[eval] step {step} | train {metrics['train']:.4f} | val {metrics['val']:.4f}")
            with log_path.open("a") as f:
                f.write(json.dumps({"step": step, "eval": metrics}) + "\n")
            if metrics["val"] < best_val:
                best_val = metrics["val"]
                best_path = out_dir / "best.pt"
                torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(),
                            "step": step, "best_val": best_val}, best_path)
                if mirror_dir:
                    mirror_threads.append(_async_copy(best_path, mirror_dir / "best.pt"))
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
            if mirror_dir:
                mirror_threads.append(_async_copy(ckpt_path, mirror_dir / "ckpt.pt"))

    final_state = {"model": model.state_dict(), "cfg": cfg.to_dict(),
                   "step": args.max_steps - 1, "best_val": best_val}
    torch.save(final_state, out_dir / "final.pt")
    if not (out_dir / "best.pt").exists():
        torch.save(final_state, out_dir / "best.pt")
    for thread in mirror_threads:
        thread.join(timeout=60)
    print(f"[done] terminé en {(time.time()-t0)/3600:.2f} h | meilleure val {best_val:.4f}")


if __name__ == "__main__":
    main()
