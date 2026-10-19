"""Mesure honnête de la RAM et de la vitesse de génération.

    python -m blackcode.bench --ckpt runs/nano/best.pt --tokenizer data/code/tokenizer.json

``ru_maxrss`` est le pic de RSS du processus. Il inclut Python, PyTorch et les
allocations : c'est volontaire. La différence entre « après import » et « après
chargement du modèle » isole le coût du modèle.
"""
from __future__ import annotations

import argparse
import resource
import time

import torch

from .corpus import load_tokenizer
from .generate import load_checkpoint


def rss_mb() -> float:
    # Linux : ru_maxrss est en kilo-octets.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark RAM / vitesse")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--tokenizer", type=str, required=True)
    p.add_argument("--prompt", type=str, default="def solve(x):\n")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    base = rss_mb()
    t0 = time.time()
    model, cfg = load_checkpoint(args.ckpt, args.device)
    load_s = time.time() - t0
    after_load = rss_mb()
    tok = load_tokenizer(args.tokenizer)
    ids = torch.tensor([tok.encode(args.prompt, add_special_tokens=False).ids],
                       dtype=torch.long, device=args.device)

    # échauffement
    model.generate(ids, max_new_tokens=2, temperature=0.0)
    t0 = time.time()
    out = model.generate(ids, max_new_tokens=args.max_new_tokens, temperature=0.0)
    gen_s = time.time() - t0
    peak = rss_mb()
    n_new = out.shape[1] - ids.shape[1]

    print(f"modèle          : {cfg.name} ({cfg.num_params/1e6:.2f} M paramètres)")
    print(f"RSS après import: {base:7.1f} Mo")
    print(f"RSS après charg.: {after_load:7.1f} Mo  (+{after_load-base:.1f})")
    print(f"RSS pic         : {peak:7.1f} Mo  (+{peak-base:.1f} par rapport à l'import)")
    print(f"chargement      : {load_s:.2f} s")
    print(f"génération      : {n_new} tokens en {gen_s:.2f} s "
          f"({n_new/max(gen_s,1e-9):.1f} tok/s)")
    print("note            : ce chiffre inclut Python + PyTorch ; un runtime C "
          "dédié vise <80 Mo hors poids.")


if __name__ == "__main__":
    main()
