"""Distillation professeur → élève, pour tenir sur un T4.

Pourquoi la distillation et pas le pré-entraînement : un modèle de 0,6 à 1,2 Md
demande 12 à 24 Md de tokens (règle des 20 tokens/paramètre). Un T4 gratuit
n'en produit qu'une fraction par semaine. Un professeur déjà entraîné remplace
ces tokens : on n'apprend plus « quoi dire » mais « comment répondre », ce qui
divise le budget par 10 à 30.

**Condition non négociable : le professeur et l'élève doivent partager le même
vocabulaire.** Sinon les logits ne sont pas comparables position par position.
C'est pourquoi le tokenizer du corpus doit être celui du professeur
(``--tokenizer hf:bigcode/starcoder2-3b``, 49 152 jetons) et non un BPE
réentraîné. Les jetons spéciaux de BlackCode sont ajoutés *à la fin* du
vocabulaire pour ne décaler aucun identifiant existant.

Usage Colab (professeur 3 Md en 4 bits, élève 589 M) :

    python -m blackcode.distill \
        --teacher bigcode/starcoder2-3b --load-4bit \
        --config configs/blackcode_0.6b.json --data data/v2 \
        --out runs/distill_06b --seq-len 2048 --batch-size 2 --grad-accum 8 \
        --max-steps 4000 --topk 32 --alpha 0.5 --temperature 2.0 \
        --optim adafactor --lr 3e-4 --schedule wsd --checkpoint-policy selective \
        --eval-interval 200 --resume

Le même code accepte ``--teacher local:runs/nano/ckpt.pt`` : c'est ce chemin
qui est testé hors GPU, et qui permet de distiller un modèle maison.
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import ModelConfig
from .data import TokenDataset
from .model import BlackCode, build_model
from .optim import make_optimizer, schedule_multiplier


# ---------------------------------------------------------------------------
class Teacher:
    """Interface minimale : logits [B, T, Vt] pour des identifiants partagés."""

    vocab_size: int

    def logits(self, idx: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:
        pass


class LocalTeacher(Teacher):
    """Professeur BlackCode (checkpoint maison) — testable sur CPU."""

    def __init__(self, path: str, device: str):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt)
        cfg_dict = ckpt.get("config")
        if cfg_dict is None:
            raise ValueError("checkpoint sans 'config' : impossible de reconstruire le modèle")
        cfg = ModelConfig(**cfg_dict) if isinstance(cfg_dict, dict) else cfg_dict
        self.model = build_model(cfg, device)
        self.model.load_state_dict(state)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.vocab_size = cfg.vocab_size
        self.device = device

    @torch.no_grad()
    def logits(self, idx):
        # `BlackCode.forward` ne rend que le dernier token : on veut tout.
        hidden, _ = self.model.forward_hidden(idx)
        return self.model._logits(hidden)


class HFTeacher(Teacher):
    """Professeur HuggingFace (StarCoder2, Qwen, etc.)."""

    def __init__(self, name: str, device: str, load_4bit: bool = False,
                 dtype: torch.dtype = torch.float16):
        from transformers import AutoModelForCausalLM

        kwargs: dict = {"trust_remote_code": False, "attn_implementation": "sdpa"}
        if load_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = {"": 0}
        else:
            kwargs["torch_dtype"] = dtype
        self.model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        if not load_4bit:
            self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.vocab_size = int(self.model.get_output_embeddings().weight.shape[0])
        self.device = device
        print(f"[teacher] {name} : vocab={self.vocab_size}, "
              f"4-bit={load_4bit}, dtype={dtype}")

    @torch.no_grad()
    def logits(self, idx):
        return self.model(idx, use_cache=False).logits


def make_teacher(spec: str, device: str, load_4bit: bool, dtype: torch.dtype) -> Teacher:
    if spec.startswith("local:"):
        return LocalTeacher(spec.split(":", 1)[1], device)
    return HFTeacher(spec, device, load_4bit=load_4bit, dtype=dtype)


# ---------------------------------------------------------------------------
@torch.no_grad()
def teacher_targets(teacher: Teacher, idx: torch.Tensor, topk: int,
                    micro_batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Renvoie (valeurs, indices) des `topk` logits du professeur, batch par batch."""
    values, indices = [], []
    for i in range(0, idx.shape[0], micro_batch):
        chunk = idx[i:i + micro_batch]
        logits = teacher.logits(chunk)
        vocab = min(teacher.vocab_size, logits.shape[-1])
        top = logits[..., :vocab].topk(topk, dim=-1)
        values.append(top.values.float().cpu())
        indices.append(top.indices.cpu())
        del logits, top
    return torch.cat(values), torch.cat(indices)


def kd_loss(student_logits: torch.Tensor, t_values: torch.Tensor,
            t_indices: torch.Tensor, temperature: float) -> torch.Tensor:
    """KL(professeur || élève) restreinte au top-k, moyennée sur les positions.

    Le top-k évite de matérialiser une distribution complète sur 49 152 jetons
    et garde l'essentiel du signal (le reste est proche de zéro).
    """
    device = student_logits.device
    gathered = student_logits.gather(-1, t_indices.to(device))
    log_s = F.log_softmax(gathered / temperature, dim=-1)
    log_t = F.log_softmax(t_values.to(device) / temperature, dim=-1)
    kl = F.kl_div(log_s, log_t, log_target=True, reduction="none")
    return kl.sum(-1).mean()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distillation BlackCode")
    p.add_argument("--teacher", required=True,
                   help="nom HuggingFace ou 'local:/chemin/ckpt.pt'")
    p.add_argument("--load-4bit", action="store_true",
                   help="professeur en 4 bits (indispensable pour 3 Md sur T4)")
    p.add_argument("--teacher-dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--init-from", default=None,
                   help="checkpoint élève de départ (sinon initialisation aléatoire)")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--optim", default="adafactor",
                   choices=["adamw", "muon", "ademamix", "adafactor",
                            "adamw8bit", "paged_adamw8bit"])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--schedule", default="wsd", choices=["cosine", "wsd", "constant"])
    p.add_argument("--stable-frac", type=float, default=0.8)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--topk", type=int, default=32)
    p.add_argument("--alpha", type=float, default=0.5, help="poids de la KD (1-alpha = CE)")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--logit-chunk", type=int, default=256)
    p.add_argument("--checkpoint-policy", default="selective",
                   choices=["none", "selective", "full"])
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-iters", type=int, default=10)
    p.add_argument("--save-interval", type=int, default=250)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    if args.dtype == "auto":
        dtype = "float16" if device.startswith("cuda") else "float32"
    else:
        dtype = args.dtype
    if device == "cpu":
        dtype = "float32"
    torch.set_float32_matmul_precision("high")

    cfg = ModelConfig.from_json(args.config)
    seq_len = args.seq_len or cfg.max_seq_len
    student = build_model(cfg, device)
    if args.init_from and Path(args.init_from).exists():
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        student.load_state_dict(ckpt["model"])
        print(f"[student] initialisé depuis {args.init_from}")

    teacher_dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    teacher = make_teacher(args.teacher, device, args.load_4bit,
                           teacher_dtypes[args.teacher_dtype])
    policy = args.checkpoint_policy

    if teacher.vocab_size != cfg.vocab_size:
        print(f"[attention] vocab professeur {teacher.vocab_size} ≠ élève {cfg.vocab_size} : "
              f"seuls les {min(teacher.vocab_size, cfg.vocab_size)} premiers "
              f"identifiants sont comparés. Ils doivent être identiques !")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    dataset = TokenDataset(args.data, device=device)
    print(f"[data] train {dataset.train_tokens/1e6:.2f} M | val {dataset.val_tokens/1e6:.2f} M "
          f"| seq_len {seq_len}")

    optimizer, _ = make_optimizer(
        student, name=args.optim, lr=args.lr, muon_lr=args.muon_lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "float16"))
    autocast = (torch.autocast("cuda", dtype=torch.float16)
                if dtype == "float16" else nullcontext())

    start_step = 0
    best_val = float("inf")
    ckpt_path = out_dir / "ckpt.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        student.load_state_dict(ckpt["model"])
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except (KeyError, ValueError):
            print("[resume] optimiseur réinitialisé")
        start_step = ckpt["step"] + 1
        best_val = ckpt.get("best_val", best_val)
        print(f"[resume] étape {start_step}")

    warmup = args.warmup if args.warmup is not None else max(20, int(0.03 * args.max_steps))
    log_path = out_dir / "distill_log.jsonl"
    tokens_per_step = args.batch_size * args.grad_accum * seq_len
    print(f"[budget] {args.max_steps} étapes × {tokens_per_step} tokens "
          f"= {args.max_steps * tokens_per_step/1e6:.2f} M tokens vus")
    t0 = time.time()

    base_lrs = [group["lr"] for group in optimizer.param_groups]

    for step in range(start_step, args.max_steps):
        lr_mult = schedule_multiplier(step, args.schedule, warmup, args.max_steps,
                                      min_ratio=args.min_lr / max(args.lr, 1e-12),
                                      stable_frac=args.stable_frac)
        for group, base in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base * lr_mult

        optimizer.zero_grad(set_to_none=True)
        stats = {"ce": 0.0, "kd": 0.0, "tok": 0.0}
        for _ in range(args.grad_accum):
            x, y = dataset.get_batch("train", args.batch_size, seq_len)
            with autocast:
                with torch.no_grad():
                    t_val, t_idx = teacher_targets(teacher, x, args.topk)
                hidden, _ = student.forward_hidden(x, checkpoint=policy)
                ce_total, kd_total, n_pos = 0.0, 0.0, 0
                for i in range(0, hidden.shape[1], args.logit_chunk):
                    j = min(i + args.logit_chunk, hidden.shape[1])
                    logits = student._logits(hidden[:, i:j]).float()
                    n_here = x.shape[0] * (j - i)
                    ce_total = ce_total + F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]), y[:, i:j].reshape(-1),
                        reduction="sum")
                    kd_total = kd_total + kd_loss(
                        logits, t_val[:, i:j], t_idx[:, i:j],
                        args.temperature) * n_here
                    n_pos += n_here
                ce = ce_total / n_pos
                kd = kd_total / n_pos
                loss = ((1 - args.alpha) * ce + args.alpha * kd) / args.grad_accum
            scaler.scale(loss).backward()
            stats["ce"] += float(ce.detach())
            stats["kd"] += float(kd.detach())
            stats["tok"] += float(n_pos)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % args.log_interval == 0:
            elapsed = time.time() - t0
            done = (step - start_step + 1) * tokens_per_step
            record = {
                "step": step,
                "ce": stats["ce"] / args.grad_accum,
                "kd": stats["kd"] / args.grad_accum,
                "lr": optimizer.param_groups[0]["lr"],
                "tokens": done,
                "tok_s": done / max(elapsed, 1e-6),
            }
            print(json.dumps(record))
            with log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")

        if args.eval_interval and step > 0 and step % args.eval_interval == 0:
            val = evaluate(student, dataset, args, seq_len, device, autocast)
            print(json.dumps({"step": step, "val_ce": val}))
            if val < best_val:
                best_val = val
                save(student, optimizer, scaler, step, best_val, out_dir / "best.pt",
                     args.checkpoint_policy, cfg)

        if args.save_interval and step > 0 and step % args.save_interval == 0:
            save(student, optimizer, scaler, step, best_val, ckpt_path,
                 args.checkpoint_policy, cfg)

    save(student, optimizer, scaler, args.max_steps - 1, best_val, ckpt_path,
         "full", cfg)
    teacher.close()
    print(f"[fin] {args.out} — meilleur val CE {best_val:.4f}")


def evaluate(student, dataset, args, seq_len, device, autocast) -> float:
    student.eval()
    losses = []
    with torch.no_grad():
        for _ in range(args.eval_iters):
            x, y = dataset.get_batch("val", args.batch_size, min(seq_len, 1024))
            with autocast:
                hidden, _ = student.forward_hidden(x)
                logits = student._logits(hidden).float()
                losses.append(float(F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), y.reshape(-1))))
    student.train()
    return sum(losses) / max(1, len(losses))


def save(model, optimizer, scaler, step, best_val, path, policy, cfg) -> None:
    state = model.state_dict()
    torch.save({"model": state, "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                "step": step, "best_val": best_val, "config": cfg.to_dict()},
               path)
    print(f"[save] {path} (étape {step})")


if __name__ == "__main__":
    main()
