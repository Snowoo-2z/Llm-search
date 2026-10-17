"""Planificateur de budget mémoire.

    python -m blackcode.planner --config configs/blackcode_1b.json --context 8192
    python -m blackcode.planner --config configs/blackcode_1b.json --context 8192 --scan --ram 1024

``--scan`` cherche le meilleur couple (taille, bits) sous un budget de RAM donné,
avec la loi de précision de Kumar et al. (2024) et les constantes Chinchilla.
"""
from __future__ import annotations

import argparse

from .config import ModelConfig
from .memory import budget, format_table, scan_bits


def main() -> None:
    p = argparse.ArgumentParser(description="Budget RAM d'un modèle BlackCode")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--weight-bits", type=float, default=2.0)
    p.add_argument("--embed-bits", type=float, default=8.0)
    p.add_argument("--runtime-mb", type=float, default=180.0)
    p.add_argument("--ram", type=float, default=1024.0)
    p.add_argument("--gamma", type=float, default=6.0)
    p.add_argument("--scan", action="store_true")
    args = p.parse_args()

    cfg = ModelConfig.from_json(args.config)
    print(cfg.name, "|", cfg.description)
    print(f"paramètres : {cfg.num_params/1e6:.2f} M "
          f"({cfg.param_breakdown()['non_embedding']/1e6:.2f} M hors embeddings)")
    print(f"couches : {cfg.n_layers} dont {cfg.n_attention_layers} attention "
          f"({cfg.n_attention_layers/cfg.n_layers:.0%})")
    print()

    b = budget(cfg, args.context, args.weight_bits, args.embed_bits, args.runtime_mb)
    print(f"Budget à {args.context} tokens, poids {args.weight_bits} bits, "
          f"embeddings {args.embed_bits} bits :")
    for key, value in b.as_dict().items():
        print(f"  {key:16s} {value:9.1f}")
    print(f"  {'marge/1Go':16s} {1024 - b.total_mb:9.1f}")
    print(f"  KV par token     {b.kv_per_token_kb:.2f} Ko")
    print()

    if args.scan:
        print(f"Balayage sous {args.ram:.0f} Mo de RAM, contexte {args.context}, "
              f"gamma={args.gamma} :")
        rows = scan_bits(cfg, args.ram, args.context, args.runtime_mb, args.gamma)
        print(format_table(rows))
        best = min(rows, key=lambda r: r["predicted_loss"])
        print(f"\n→ optimum prédit : {best['bits']} bits, "
              f"{best['params']/1e6:.0f} M paramètres, perte {best['predicted_loss']:.4f}")
        print("  (loi d'échelle : extrapolation à valider empiriquement, "
              "surtout sous 4 bits)")


if __name__ == "__main__":
    main()
