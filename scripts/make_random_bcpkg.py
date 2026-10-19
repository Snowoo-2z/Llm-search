#!/usr/bin/env python
"""Génère un paquet .bcpkg de la bonne taille, sans entraîner ni allouer le modèle.

Pourquoi : la RAM consommée par l'inférence ne dépend **pas** des valeurs des
poids, seulement de leur nombre, de leur encodage et du contexte. On peut donc
mesurer dès aujourd'hui la RAM réelle d'un BlackCode-1B avec des poids
aléatoires, avant même de l'entraîner.

    python scripts/make_random_bcpkg.py --config configs/blackcode_1b.json \
        --out artifacts/random_1b.bcpkg --context 8192
    ./runtime/bcrun --pkg artifacts/random_1b.bcpkg --ids 1 --max-new 0 \
        --context 8192 --mem-report
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blackcode.config import ModelConfig  # noqa: E402
from blackcode.export import specs_from_meta_model, write_package  # noqa: E402


def make_tensor(spec, gen: torch.Generator) -> torch.Tensor:
    """Poids plausibles : normes à 1, matrices ~N(0, 0.02), conv ~N(0, 0.05)."""
    shape = tuple(spec.shape)
    if spec.name.endswith("norm.weight") or spec.name == "norm_f.weight":
        return torch.ones(shape)
    if spec.name.endswith("conv.weight"):
        return torch.randn(shape, generator=gen) * 0.05
    if spec.kind == "embedding":
        return torch.randn(shape, generator=gen) * 0.02
    if spec.kind == "ternary":
        return torch.randn(shape, generator=gen) * 0.02
    return torch.randn(shape, generator=gen) * 0.02


def main() -> None:
    p = argparse.ArgumentParser(description="Paquet .bcpkg aléatoire de taille réelle")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--embed-bits", type=int, default=8, choices=[8, 16])
    p.add_argument("--runtime-mb", type=float, default=16.0,
                   help="estimation du runtime C (mesurée : ~8 Mo)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = ModelConfig.from_json(args.config)
    specs = specs_from_meta_model(cfg, args.embed_bits)
    gen = torch.Generator().manual_seed(args.seed)
    total = sum(int(torch.tensor(s.shape).prod()) for s in specs)
    print(f"[random] {cfg.name} : {total/1e6:.1f} M paramètres, "
          f"{len(specs)} tenseurs, contexte {args.context}")

    header = write_package(
        cfg, specs, args.out, args.context, args.embed_bits, args.runtime_mb,
        provider=lambda spec: make_tensor(spec, gen),
        extra_meta={"note": "poids aléatoires — mémoire identique à un modèle entraîné"},
    )
    print(f"[random] {args.out} : {header['packed_mb']:.1f} Mo "
          f"({header['effective_bits_per_param']:.2f} bits/paramètre effectifs)")
    cat = header["category_mb"]
    print(f"[random] ternaire {cat['ternary']:.1f} | embeddings {cat['embedding']:.1f} | "
          f"autres {cat['other']:.2f} Mo")
    print(f"[random] RAM estimée : {header['estimated_runtime_ram_mb']:.0f} Mo "
          f"(mesurer avec : ./runtime/bcrun --mem-report)")


if __name__ == "__main__":
    main()
