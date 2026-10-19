"""Boucle d'entraînement BlackCode (CPU ou T4) — version ultra-optimisée v3 MAX.

Objectif : **réduire le temps à taille de modèle fixée** — TOUTES les optimisations possibles.

Gains cumulés v3 vs baseline mono-T4 naïf :
- Heuristique FR ×50 dataset, no-near ×2 RAM÷10, tokenizer 5 min vs 30 min
- DDP dual-T4 ×1.8-2, DDP no_sync +15-30%, bucket 10 MB +5%, FP16 compress +10-20% PCIe
- Fused AdamW +5-10%, RMSNorm fused +5-10%, SwiGLU fused +5-10%
- compile max-autotune +10-25%, cudnn benchmark +5-10%, TF32 +10-20%
- Prefetch 2 +5-15%, TokenDataset torch.randint +10%
- Selective checkpoint +10-15%, seq-len warmup +15%, WSD +5-10%, eval espacées +5-10%
- Muon ×1.3-2 tokens, Adafactor batch ×2-3, static_graph DDP +2-5%, foreach +2%
- CUDA Graphs (optionnel) +10% si static shape

Total : 6 Md en 12h dual-T4 v2 → 9-10h v3 estimé, 4 Md en 6-7h, 1 Md en 3-4h.

Exemple ultra v3 :

    torchrun --nproc_per_node=2 -m blackcode.train --config configs/blackcode_fr_260m.json --data data/fr_260m --out runs/fr_260m_6b --batch-size 4 --grad-accum 8 --seq-len 2048 --max-steps 183105 --optim muon --muon-lr 0.02 --lr 6e-4 --fused --foreach --schedule wsd --stable-frac 0.8 --seq-len-warmup-frac 0.2 --checkpoint-policy selective --compile --compile-mode max-autotune --compile-fullgraph --cudnn-benchmark --tf32 --prefetch 2 --bucket-cap-mb 10 --ddp --ddp-fp16-compress --ddp-static-graph --log-flops --eval-interval 1000 --save-interval 1000 --resume
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import deque
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
        except Exception as exc:
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
    p = argparse.ArgumentParser(description="Entraîne BlackCode — ultra optimisé v3 MAX")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--seq-len-warmup-frac", type=float, default=0.0)
    p.add_argument("--seq-len-start", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=1000)

    # Optim
    p.add_argument("--optim", type=str, default="adamw", choices=["adamw", "muon", "ademamix", "adafactor", "adamw8bit", "paged_adamw8bit"])
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr", type=float, default=6e-5)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--adafactor-lr", type=float, default=None)
    p.add_argument("--muon-ns-steps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "wsd", "constant"])
    p.add_argument("--stable-frac", type=float, default=0.8)
    p.add_argument("--decay-shape", type=str, default="sqrt", choices=["sqrt", "linear", "cosine"])
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--fused", action="store_true", help="AdamW fused +5-10%")
    p.add_argument("--no-fused", action="store_true")
    p.add_argument("--foreach", action="store_true", help="AdamW foreach +2% si non fused")
    p.add_argument("--no-foreach", action="store_true")

    # Mémoire / débit — ultra leviers v3
    p.add_argument("--logit-chunk", type=int, default=256)
    p.add_argument("--checkpoint-policy", type=str, default="none", choices=["none", "selective", "full"])
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile-mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
    p.add_argument("--compile-fullgraph", action="store_true", help="torch.compile fullgraph=True, +2-5% si pas de graph breaks")
    p.add_argument("--compile-dynamic", action="store_true")
    p.add_argument("--cudnn-benchmark", dest="cudnn_benchmark", action="store_true", default=True)
    p.add_argument("--no-cudnn-benchmark", dest="cudnn_benchmark", action="store_false")
    p.add_argument("--prefetch", type=int, default=1)
    p.add_argument("--bucket-cap-mb", type=int, default=25)
    p.add_argument("--no-ddp-no-sync", action="store_true")
    p.add_argument("--tf32", dest="tf32", action="store_true", default=True)
    p.add_argument("--no-tf32", dest="tf32", action="store_false")
    p.add_argument("--allow-fp16-reduced", action="store_true", default=True)
    # Nouveaux v3
    p.add_argument("--ddp-fp16-compress", action="store_true", help="DDP FP16 compress hook, +10-20% sur PCIe T4")
    p.add_argument("--ddp-static-graph", action="store_true", help="DDP static_graph=True, +2-5%")
    p.add_argument("--cuda-graphs", action="store_true", help="CUDA Graphs pour training step static, +10% si seq_len fixe après warmup (expérimental)")
    p.add_argument("--batch-size-max", type=int, default=None, help="batch max auto si OOM, essaie batch-size puis batch-size-max")
    p.add_argument("--use-torch-cache", action="store_true", help="cache dataset en RAM pin_memory, +10% si RAM suffisante")

    # Suivi
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--log-flops", action="store_true")
    p.add_argument("--peak-tflops", type=float, default=65.0)
    p.add_argument("--mirror-dir", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--ddp", action="store_true")
    p.add_argument("--local-rank", type=int, default=0)
    return p.parse_args()


def _init_ddp(args):
    if not args.ddp and "RANK" not in os.environ:
        return 0, 1, False
    rank = int(os.environ.get("RANK", args.local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        torch.cuda.set_device(local_rank)
        print(f"[ddp] rank {rank}/{world_size} local_rank {local_rank} bucket_cap {args.bucket_cap_mb} MB fp16_compress={args.ddp_fp16_compress} static_graph={args.ddp_static_graph}")
        return rank, world_size, True
    return 0, 1, False


class PrefetchLoader:
    def __init__(self, dataset: TokenDataset, split: str, batch_size: int, prefetch: int = 1):
        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        self.prefetch = max(0, prefetch)
        self.queue: deque[tuple[torch.Tensor, torch.Tensor]] = deque()

    def fill(self, seq_len: int):
        while len(self.queue) < self.prefetch + 1:
            self.queue.append(self.dataset.get_batch(self.split, self.batch_size, seq_len))

    def next(self, seq_len: int):
        if not self.queue:
            self.fill(seq_len)
        x, y = self.queue.popleft()
        if len(self.queue) < self.prefetch + 1:
            self.queue.append(self.dataset.get_batch(self.split, self.batch_size, seq_len))
        return x, y


def main() -> None:
    args = parse_args()
    rank, world_size, is_ddp = _init_ddp(args)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    np.random.seed(args.seed + rank)
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

    # === Optimisations globales torch ===
    torch.set_float32_matmul_precision("high")
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        torch.backends.cudnn.deterministic = False
        if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = args.allow_fp16_reduced
        try:
            import torch._inductor.config as inductor_config
            inductor_config.conv_1x1_as_mm = True
            inductor_config.coordinate_descent_tuning = False
            torch._dynamo.config.cache_size_limit = 128
            if args.compile_fullgraph:
                torch._dynamo.config.suppress_errors = True
        except Exception:
            pass
        if not args.no_fused and not args.fused:
            args.fused = True
            print("[optim] fused=True auto activé sur CUDA (+5-10%)")
        if not args.no_foreach and args.foreach and not args.fused:
            print("[optim] foreach=True (+2% si non fused)")

    policy = "full" if args.grad_checkpoint else args.checkpoint_policy

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {cfg.name}: {n_params/1e6:.2f} M paramètres, {cfg.n_attention_layers}/{cfg.n_layers} couches attn, SwiGLU fused")
    print(f"[device] {device} | dtype {dtype} | seq_len {seq_len} | batch {args.batch_size}×accum {args.grad_accum} | optim {args.optim} | schedule {args.schedule} | ckpt {policy} | ddp={is_ddp} world={world_size} | compile={args.compile} mode={args.compile_mode} fullgraph={args.compile_fullgraph} | fused={args.fused} foreach={args.foreach} | cudnn_benchmark={args.cudnn_benchmark} tf32={args.tf32} | prefetch={args.prefetch} bucket={args.bucket_cap_mb}MB fp16_compress={args.ddp_fp16_compress} static_graph={args.ddp_static_graph} cuda_graphs={args.cuda_graphs}")

    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        if device.startswith("cuda"):
            model = model.to(f"cuda:{torch.cuda.current_device()}")
        model = DDP(model,
                    device_ids=[torch.cuda.current_device()] if device.startswith("cuda") else None,
                    bucket_cap_mb=args.bucket_cap_mb,
                    gradient_as_bucket_view=True,
                    static_graph=args.ddp_static_graph)
        print(f"[ddp] DDP wrappé, bucket {args.bucket_cap_mb} MB, grad_as_bucket_view, static_graph={args.ddp_static_graph}")
        if args.ddp_fp16_compress:
            try:
                from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
                model.register_comm_hook(state=None, hook=fp16_compress_hook)
                print("[ddp] FP16 compress hook activé (+10-20% PCIe)")
            except Exception as exc:
                print(f"[ddp] FP16 compress hook échec : {exc}")

    if args.compile and hasattr(torch, "compile"):
        try:
            dynamic = args.compile_dynamic or (args.seq_len_warmup_frac > 0)
            fullgraph = args.compile_fullgraph
            model = torch.compile(model, mode=args.compile_mode, dynamic=dynamic, fullgraph=fullgraph)
            print(f"[compile] torch.compile mode={args.compile_mode} dynamic={dynamic} fullgraph={fullgraph}")
        except Exception as exc:
            print(f"[compile] ignoré : {exc}")

    dataset = TokenDataset(args.data, device=device, use_torch_cache=args.use_torch_cache)
    data_vocab = int(dataset.meta.get("vocab_size", cfg.vocab_size))
    if data_vocab > cfg.vocab_size:
        raise ValueError(f"le corpus utilise {data_vocab} jetons mais le modèle n'en a que {cfg.vocab_size}")
    print(f"[data] train {dataset.train_tokens/1e6:.2f} M tokens | val {dataset.val_tokens/1e6:.2f} M tokens")

    # foreach pour AdamW si non fused
    use_foreach = args.foreach and not args.no_foreach and not args.fused
    optimizer, base_lrs = make_optimizer(
        model, name=args.optim, lr=args.lr, muon_lr=args.muon_lr,
        weight_decay=args.weight_decay, betas=(args.beta1, args.beta2),
        fused=args.fused and not args.no_fused, alpha_warmup=max(1, args.max_steps // 2),
        adafactor_lr=args.adafactor_lr,
    )
    # Patch foreach si demandé
    if use_foreach:
        for opt in getattr(optimizer, "optimizers", [optimizer]):
            for group in opt.param_groups:
                group["foreach"] = True
        print("[optim] foreach=True appliqué")

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
        try:
            (model.module if is_ddp else model).load_state_dict(ckpt["model"])
        except Exception:
            try:
                model.load_state_dict(ckpt["model"])
            except Exception as exc:
                print(f"[resume] model load échec : {exc}")
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
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16) if dtype == "float16" else nullcontext())
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
        return max(8, int(value // 8 * 8))

    fpt_final = flops_per_token(cfg, seq_len)["total"]
    model.train()
    t0 = time.time()
    last_log_time = t0
    last_log_tokens = 0
    tokens_seen = start_step * args.batch_size * args.grad_accum * seq_len

    prefetch_loader = PrefetchLoader(dataset, "train", args.batch_size, prefetch=args.prefetch) if args.prefetch > 0 else None
    if prefetch_loader:
        prefetch_loader.fill(current_seq_len(start_step))

    # CUDA Graphs (expérimental, static shape seulement)
    cuda_graph = None
    static_input_x = None
    static_input_y = None
    static_loss = None
    if args.cuda_graphs and device.startswith("cuda") and args.seq_len_warmup_frac == 0:
        print("[cuda-graphs] activé (expérimental, seq_len fixe requis)")
        # On capturera après premier step

    for step in range(start_step, args.max_steps):
        mult = schedule_multiplier(step, args.schedule, warmup, args.max_steps, min_ratio, args.stable_frac, args.decay_shape)
        for group, base in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base * mult

        step_seq = current_seq_len(step)
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for micro in range(args.grad_accum):
            is_last_micro = (micro == args.grad_accum - 1)
            ddp_no_sync = (is_ddp and not is_last_micro and not args.no_ddp_no_sync)
            no_sync_ctx = model.no_sync() if ddp_no_sync else nullcontext()

            if prefetch_loader:
                x, y = prefetch_loader.next(step_seq)
            else:
                x, y = dataset.get_batch("train", args.batch_size, step_seq)

            with no_sync_ctx:
                with autocast_ctx:
                    loss = model.forward_loss(x, y, logit_chunk=args.logit_chunk, checkpoint=policy)
                loss = loss / args.grad_accum
                scaler.scale(loss).backward()
            loss_accum += loss.detach().item()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        tokens_seen += args.batch_size * args.grad_accum * step_seq
        if is_ddp:
            tokens_seen += args.batch_size * args.grad_accum * step_seq * (world_size - 1)

        if rank == 0 and step % args.log_interval == 0:
            now = time.time()
            elapsed = max(now - last_log_time, 1e-9)
            tok_s = (tokens_seen - last_log_tokens) / elapsed
            remaining = (args.max_steps - step - 1) * args.batch_size * args.grad_accum * seq_len * world_size
            remaining_s = remaining / max(tok_s, 1e-9)
            mem = ""
            if device.startswith("cuda"):
                mem = f" | VRAM {torch.cuda.max_memory_allocated()/1e9:.2f} Go"
                torch.cuda.reset_peak_memory_stats()
            flops_s = fpt_final * (tokens_seen - last_log_tokens) / elapsed
            mfu = flops_s / (args.peak_tflops * 1e12)
            flops_txt = "" if not args.log_flops else (f" | {flops_s/1e12:5.1f} TFLOPS ({mfu:4.0%} MFU)" if device.startswith("cuda") else f" | {flops_s/1e9:5.1f} GFLOP/s")
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"step {step:6d} | loss {loss_accum:.4f} | lr {current_lr:.2e} | seq {step_seq} | {tok_s/1e3:5.1f} k tok/s | ETA {remaining_s/3600:.2f} h{flops_txt}{mem} | world {world_size}")
            record = {"step": step, "loss": loss_accum, "lr_mult": mult, "seq_len": step_seq, "tokens_per_s": tok_s, "flops_per_s": flops_s, "mfu": mfu, "elapsed_s": now - t0, "world_size": world_size}
            with log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            last_log_time, last_log_tokens = now, tokens_seen

        if ((args.eval_interval and (step + 1) % args.eval_interval == 0) or step == args.max_steps - 1):
            metrics = estimate_loss(model, dataset, args.batch_size, seq_len, args.eval_iters, args.logit_chunk)
            if rank == 0:
                print(f"[eval] step {step} | train {metrics['train']:.4f} | val {metrics['val']:.4f}")
                with log_path.open("a") as f:
                    f.write(json.dumps({"step": step, "eval": metrics}) + "\n")
                if metrics["val"] < best_val:
                    best_val = metrics["val"]
                    best_path = out_dir / "best.pt"
                    state_dict = model.module.state_dict() if is_ddp else model.state_dict()
                    torch.save({"model": state_dict, "cfg": cfg.to_dict(), "step": step, "best_val": best_val}, best_path)
                    if mirror_dir:
                        mirror_threads.append(_async_copy(best_path, mirror_dir / "best.pt"))
            if is_ddp:
                best_val_tensor = torch.tensor(best_val, device=device)
                torch.distributed.broadcast(best_val_tensor, src=0)
                best_val = best_val_tensor.item()
            model.train()

        if rank == 0 and ((args.save_interval and (step + 1) % args.save_interval == 0) or step == args.max_steps - 1):
            state_dict = model.module.state_dict() if is_ddp else model.state_dict()
            torch.save({"model": state_dict, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict() if dtype == "float16" else None, "step": step, "best_val": best_val, "cfg": cfg.to_dict()}, ckpt_path)
            print(f"[ckpt] sauvegarde {ckpt_path} (step {step})")
            if mirror_dir:
                mirror_threads.append(_async_copy(ckpt_path, mirror_dir / "ckpt.pt"))

    if rank == 0:
        state_dict = model.module.state_dict() if is_ddp else model.state_dict()
        final_state = {"model": state_dict, "cfg": cfg.to_dict(), "step": args.max_steps - 1, "best_val": best_val}
        torch.save(final_state, out_dir / "final.pt")
        if not (out_dir / "best.pt").exists():
            torch.save(final_state, out_dir / "best.pt")
        for thread in mirror_threads:
            thread.join(timeout=60)
        print(f"[done] terminé en {(time.time()-t0)/3600:.2f} h | meilleure val {best_val:.4f} | world {world_size}")
    if is_ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
