"""Calculateur de temps d'entraînement et de leviers.

    python -m blackcode.timing --config configs/blackcode_1b.json \
        --tokens 1e9 --seq-len 4096 --tflops 25

Le modèle de coût est explicite :

    FLOPs ≈ 6 · N_hors_emb · tokens                     (corps dense)
          + 12 · L_attn · d_model · T · tokens          (attention, T² en T)
          + 6 · V · d_model · tokens                    (tête de sortie)

Le facteur 6 = 2 (forward) × 3 (forward + backward). C'est l'approximation
standard ; elle ignore les normes et les convolutions (négligeables).
"""
from __future__ import annotations

import argparse

from .config import ModelConfig


# Efficacité réaliste (MFU) observée sur quelques accélérateurs, en fp16.
MFU_PRESETS = {
    "t4": (65.0, 0.35),        # 65 TFLOPS fp16 crête, ~35 % atteignables
    "v100": (125.0, 0.40),
    "a100": (312.0, 0.45),
    "h100": (990.0, 0.40),
    "cpu": (0.2, 0.30),
}


def flops_per_token(cfg: ModelConfig, seq_len: int) -> dict:
    parts = cfg.param_breakdown()
    body = 6 * parts["non_embedding"]
    attn = 12 * cfg.n_attention_layers * cfg.d_model * seq_len
    logits = 6 * cfg.vocab_size * cfg.d_model
    return {"body": body, "attention": attn, "logits": logits,
            "total": body + attn + logits}


def estimate(cfg: ModelConfig, tokens: float, seq_len: int,
             tflops_effective: float, extra_overhead: float = 0.0) -> dict:
    per_token = flops_per_token(cfg, seq_len)["total"]
    flops = per_token * tokens * (1.0 + extra_overhead)
    seconds = flops / (tflops_effective * 1e12)
    return {
        "flops": flops,
        "hours": seconds / 3600,
        "days": seconds / 86400,
        "tokens_per_s": tokens / max(seconds, 1e-9),
        "flops_per_token": per_token,
    }


def format_hours(hours: float) -> str:
    if hours < 1:
        return f"{hours*60:.0f} min"
    if hours < 48:
        return f"{hours:.1f} h"
    return f"{hours/24:.1f} j"


def main() -> None:
    p = argparse.ArgumentParser(description="Estime le temps d'entraînement")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--tokens", type=float, default=1e9)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--device", type=str, default="t4", choices=list(MFU_PRESETS))
    p.add_argument("--mfu", type=float, default=None, help="remplace le preset")
    p.add_argument("--tflops", type=float, default=None, help="TFLOPS effectifs")
    p.add_argument("--overhead", type=float, default=0.0,
                   help="surcoût (checkpointing 0.3, Muon 0.15, …)")
    p.add_argument("--memory", action="store_true",
                   help="affiche la mémoire d'entraînement par optimiseur")
    p.add_argument("--activation-mb", type=float, default=1500.0)
    args = p.parse_args()

    cfg = ModelConfig.from_json(args.config)
    seq_len = args.seq_len or cfg.max_seq_len
    peak, mfu = MFU_PRESETS[args.device]
    tflops = args.tflops if args.tflops is not None else peak * (args.mfu or mfu)

    detail = flops_per_token(cfg, seq_len)
    print(f"{cfg.name} : {cfg.num_params/1e6:.1f} M paramètres, "
          f"{cfg.n_attention_layers}/{cfg.n_layers} couches d'attention")
    print(f"contexte {seq_len}, {args.tokens/1e9:.2f} Md tokens, "
          f"{tflops:.1f} TFLOPS effectifs ({args.device})")
    print()
    print("Répartition des FLOPs par token :")
    total = detail["total"]
    for key in ("body", "attention", "logits"):
        print(f"  {key:10s} {detail[key]/1e9:7.3f} GFLOP  ({detail[key]/total:5.1%})")
    print(f"  {'total':10s} {total/1e9:7.3f} GFLOP")
    print()

    est = estimate(cfg, args.tokens, seq_len, tflops, args.overhead)
    print(f"Temps estimé : **{format_hours(est['hours'])}** "
          f"({est['tokens_per_s']:.0f} tokens/s)")
    print()

    # Leviers cumulés, avec hypothèses explicites.
    # (nom, facteur de débit, facteur de tokens, note)
    steps = [
        ("référence", 1.00, 1.00, "—"),
        ("+ distillation", 1.00, 0.10,
         "Ternary Mamba : 102 M tokens ≈ 105 Md"),
        ("+ pruning de données", 1.00, 0.59,
         "Sorscher et al., phi-1 : 1,5–3×"),
        ("+ Muon", 0.71, 1.00, "arXiv:2502.16982 : 1,3–2×"),
        ("+ compile / 8-bit / ckpt sélectif", 0.78, 1.00,
         "10–25 % de débit + batch plus grand"),
        ("+ sessions parallèles (k=4)", 0.25, 1.00,
         "DiLoCo : communication ÷500"),
    ]
    print("| scénario cumulé | temps | gain | hypothèse du pas |")
    print("|---|---:|---:|---|")
    time_f, token_f = 1.0, 1.0
    for name, dt, dk, note in steps:
        time_f *= dt
        token_f *= dk
        h = est["hours"] * time_f * token_f
        print(f"| {name} | {format_hours(h)} | ×{est['hours']/max(h,1e-9):.1f} | {note} |")

    if args.memory:
        from .optim import training_memory_mb
        params = cfg.num_params
        print("Mémoire d'entraînement (fp32, activations "
              f"{args.activation_mb:.0f} Mo) :")
        print("| optimiseur | poids | grads | états | total | tient sur 16 Go |")
        print("|---|---:|---:|---:|---:|:---:|")
        for opt_name in ("adamw", "ademamix", "muon", "adamw8bit", "adafactor"):
            m = training_memory_mb(params, opt_name, args.activation_mb)
            ok = "oui" if m["fits_16gb"] else "**non**"
            print(f"| {opt_name} | {m['weights_mb']:.0f} | {m['grads_mb']:.0f} | "
                  f"{m['optimizer_mb']:.0f} | **{m['total_mb']:.0f}** | {ok} |")
        print()

    print()
    print("Rappel : ces facteurs sont des ordres de grandeur issus de la "
          "littérature, pas des mesures sur ce projet. Le seul chiffre "
          "fiable est celui que vous mesurez vous-même (--log-flops).")


if __name__ == "__main__":
    main()
