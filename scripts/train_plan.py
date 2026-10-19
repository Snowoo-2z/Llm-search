#!/usr/bin/env python
"""Produit le plan d'entraînement exact (étapes, heures, sessions, commandes).

Le but n'est pas d'estimer « à peu près » : chaque chiffre sort de la config,
du corpus réellement présent dans `--data`, et d'un débit que vous pouvez
remplacer par une mesure (`--measured-tok-s`, obtenu avec `--log-flops`).

    python scripts/train_plan.py --config configs/blackcode_0.6b.json \
        --data data/v2 --stage distill --session-hours 10 --out plan.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blackcode.config import ModelConfig  # noqa: E402
from blackcode.timing import MFU_PRESETS, flops_per_token


def human_hours(h: float) -> str:
    if h < 1:
        return f"{h * 60:.0f} min"
    if h < 48:
        return f"{h:.1f} h"
    return f"{h / 24:.1f} j"


def main() -> None:
    p = argparse.ArgumentParser(description="Plan d'entraînement BlackCode")
    p.add_argument("--config", required=True)
    p.add_argument("--data", default=None, help="répertoire avec meta.json")
    p.add_argument("--stage", choices=["pretrain", "distill"], default="pretrain")
    p.add_argument("--tokens", type=float, default=None,
                   help="tokens cible (défaut : 20×params en pré-entraînement, 30 M en distillation)")
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--device", default="t4", choices=list(MFU_PRESETS))
    p.add_argument("--mfu", type=float, default=None)
    p.add_argument("--measured-tok-s", type=float, default=None)
    p.add_argument("--teacher-params", type=float, default=3e9,
                   help="taille du professeur (distillation)")
    p.add_argument("--session-hours", type=float, default=10.0,
                   help="durée utile d'une session Colab avant coupure")
    p.add_argument("--overhead", type=float, default=0.15,
                   help="surcoût checkpointing/compile (0,15 = +15 %)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = ModelConfig.from_json(args.config)
    seq_len = args.seq_len or min(cfg.max_seq_len, 2048)
    params = cfg.num_params
    tokens = args.tokens or (params * 20 if args.stage == "pretrain" else 30e6)
    tokens_per_step = args.batch_size * args.grad_accum * seq_len
    steps = max(1, math.ceil(tokens / tokens_per_step))

    detail = flops_per_token(cfg, seq_len)
    flops = detail["total"]
    if args.stage == "distill":
        # Le professeur ne fait que du forward : 2·N par token.
        flops += 2 * args.teacher_params
    flops *= 1.0 + args.overhead

    peak, mfu = MFU_PRESETS[args.device]
    tflops = peak * (args.mfu if args.mfu is not None else mfu)
    tok_s = args.measured_tok_s or (tflops * 1e12 / flops)
    hours = tokens / tok_s / 3600
    sessions = max(1, math.ceil(hours / args.session_hours))

    data_tokens = None
    data_mb = None
    if args.data and (Path(args.data) / "meta.json").exists():
        meta = json.loads((Path(args.data) / "meta.json").read_text())
        data_tokens = meta.get("train_tokens")
        data_mb = meta.get("stats", {}).get("bytes", 0) / 1e6 if meta.get("stats") else None
    epochs = (tokens / data_tokens) if data_tokens else None

    L: list[str] = []
    L.append(f"# Plan d'entraînement — {cfg.name}")
    L.append("")
    L.append(f"- **Étape** : {args.stage}")
    L.append(f"- **Paramètres** : {params/1e6:.1f} M "
             f"({cfg.n_attention_layers} couches d'attention / {cfg.n_layers})")
    L.append(f"- **Contexte** : {seq_len} (max {cfg.max_seq_len})")
    L.append(f"- **Budget** : {tokens/1e6:.1f} M tokens à "
             f"{flops/1e9:.2f} GFLOP/token")
    L.append(f"- **Batch effectif** : {args.batch_size} × {args.grad_accum} × "
             f"{seq_len} = **{tokens_per_step} tokens/step**")
    L.append(f"- **Étapes** : {steps}")
    L.append(f"- **Débit** : {tok_s:,.0f} tokens/s "
             f"({'mesuré' if args.measured_tok_s else f'estimé {args.device} à MFU {tflops/peak:.0%}'})")
    L.append(f"- **Durée** : **{human_hours(hours)}**")
    L.append(f"- **Sessions Colab ({args.session_hours:g} h)** : **{sessions}**")
    if epochs is not None:
        L.append(f"- **Corpus** : {data_tokens/1e6:.1f} M tokens"
                 + (f" ({data_mb:.0f} Mo brut)" if data_mb else "")
                 + f" → {epochs:.2f} époque(s)")
        if epochs > 4:
            L.append("  - ⚠️ plus de 4 époques : risque de surapprentissage, "
                     "élargir le corpus.")
        if args.stage == "pretrain" and epochs < 0.9:
            L.append("  - ⚠️ moins d'une époque : le modèle ne verra chaque "
                     "document qu'une fraction de fois.")
    L.append("")

    L.append("## Séquence recommandée")
    L.append("")
    if args.stage == "pretrain":
        L.append("| phase | contexte | part des steps | but |")
        L.append("|---|---|---:|---|")
        L.append("| échauffement | 512 | 30 % | apprendre vite, peu de FLOPs |")
        L.append(f"| montée | {seq_len} | 70 % | tenir le contexte final |")
        L.append("")
        L.append("`--seq-len-warmup-frac 0.3 --seq-len-start 512` applique ce "
                 "programme sans script supplémentaire : les premiers steps "
                 "coûtent 4× moins cher.")
    else:
        L.append("| phase | données | but |")
        L.append("|---|---|---|")
        L.append("| distillation | 30 M tokens | imiter le professeur sur du code réel |")
        L.append("| SFT agentique | 2–5 M tokens | format outils, français/anglais |")
        L.append("| ternarisation + QAT | 1–2 M tokens | passer en poids ternaires |")
    L.append("")

    L.append("## Commande prête à coller")
    L.append("")
    L.append("```bash")
    if args.stage == "pretrain":
        L.append(f"python -m blackcode.train --config {args.config} --data {args.data or 'data/v2'} \\")
        L.append(f"  --out runs/{cfg.name} --batch-size {args.batch_size} "
                 f"--grad-accum {args.grad_accum} --seq-len {seq_len} \\")
        L.append(f"  --max-steps {steps} --optim adafactor --lr 3e-4 --schedule wsd \\")
        L.append("  --seq-len-warmup-frac 0.3 --seq-len-start 512 \\")
        L.append("  --checkpoint-policy selective --compile --log-flops --resume \\")
        L.append(f"  --eval-interval {max(100, steps//20)} --save-interval {max(100, steps//10)} \\")
        L.append(f"  --mirror-dir /content/drive/MyDrive/blackcode/runs/{cfg.name}")
    else:
        L.append(f"python -m blackcode.distill --teacher bigcode/starcoder2-3b --load-4bit \\")
        L.append(f"  --config {args.config} --data {args.data or 'data/v2'} \\")
        L.append(f"  --out runs/distill_{cfg.name} --seq-len {seq_len} \\")
        L.append(f"  --batch-size {args.batch_size} --grad-accum {args.grad_accum} \\")
        L.append(f"  --max-steps {steps} --topk 32 --alpha 0.5 --temperature 2.0 \\")
        L.append("  --optim adafactor --lr 3e-4 --schedule wsd --resume \\")
        L.append("  --checkpoint-policy selective --eval-interval 200 --save-interval 200")
    L.append("```")
    L.append("")
    L.append("Chaque session : relancer exactement la même commande "
             "(`--resume` reprend au dernier checkpoint).")
    L.append("")

    L.append("## Points de contrôle")
    L.append("")
    L.append(f"- checkpoints tous les {max(100, steps//10)} steps "
             f"(≈ toutes les {human_hours(hours/10)})")
    L.append("- `best.pt` conservé séparément : c'est lui qui sera exporté en `.bcpkg`")
    L.append("- après chaque session, vérifier `val` : si elle monte, réduire le LR")
    L.append("")
    L.append("*Débit estimé, pas mesuré : `--log-flops` affiche le MFU réel. "
             "Relancer ce script avec `--measured-tok-s <valeur>` recalcule tout.*")

    text = "\n".join(L) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"[plan] écrit {args.out}")
    print(text)


if __name__ == "__main__":
    main()
