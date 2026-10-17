#!/usr/bin/env python
"""Audit d'un corpus tokenisé : est-il prêt pour tel modèle ?

    python scripts/corpus_report.py --data data/code --config configs/t4_small.json

Répond à trois questions :
  1. combien de tokens, de documents, quel taux de déduplication ;
  2. combien d'époques pour le modèle visé, et est-ce raisonnable ;
  3. combien de temps d'entraînement sur T4, et quelle taille de corpus viser.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blackcode.config import ModelConfig  # noqa: E402
from blackcode.timing import MFU_PRESETS, estimate  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Audit du corpus")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--device", type=str, default="t4")
    p.add_argument("--target-tokens-per-param", type=float, default=20.0)
    args = p.parse_args()

    data_dir = Path(args.data)
    meta = json.loads((data_dir / "meta.json").read_text())
    cfg = ModelConfig.from_json(args.config)
    seq = args.seq_len or cfg.max_seq_len

    train, val = meta["train_tokens"], meta["val_tokens"]
    tokens = train + val
    print(f"Corpus        : {data_dir}")
    print(f"  train       : {train/1e6:.2f} M tokens")
    print(f"  val         : {val/1e6:.2f} M tokens "
          f"({100*val/max(tokens,1):.2f} %)")
    print(f"  documents   : {meta.get('documents', '?')}")
    print(f"  vocab       : {meta.get('vocab_size', '?')}")

    # distribution des longueurs de documents
    arr = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    eos = meta.get("eos_id", 0)
    if len(arr) > 0:
        chunk = 2_000_000
        sample = np.asarray(arr[: min(len(arr), chunk)])
        lens = np.diff(np.flatnonzero(sample == eos))
        if lens.size:
            print(f"  longueur médiane d'un document : {np.median(lens):.0f} tokens "
                  f"(p10 {np.percentile(lens,10):.0f}, p90 {np.percentile(lens,90):.0f})")

    # échantillon de tokens pour repérer les caractères dominants
    if len(arr) > 0:
        sample = np.asarray(arr[: min(len(arr), 200_000)])
        print(f"  premiers tokens : {Counter(sample[:200].tolist()).most_common(3)}")

    print()
    params = cfg.num_params
    target = params * args.target_tokens_per_param
    epochs = tokens / max(target, 1)
    print(f"Modèle        : {cfg.name} ({params/1e6:.1f} M paramètres)")
    print(f"  cible       : {args.target_tokens_per_param:.0f} tokens/paramètre "
          f"= {target/1e9:.2f} Md tokens")
    print(f"  corpus actuel couvre {100*tokens/target:.1f} % de la cible "
          f"→ {epochs:.1f} époque(s)")

    peak, mfu = MFU_PRESETS[args.device]
    est_full = estimate(cfg, target, seq, peak * mfu)
    est_have = estimate(cfg, tokens, seq, peak * mfu)
    print(f"  temps {args.device} pour la cible ({target/1e9:.2f} Md) : "
          f"{est_full['hours']:.1f} h")
    print(f"  temps {args.device} pour ce corpus ({tokens/1e6:.0f} M) : "
          f"{est_have['hours']*60:.0f} min")

    print()
    if epochs >= 4:
        print("VERDICT : corpus trop petit pour ce modèle "
              "(surapprentissage et mémorisation probables).")
        print("  → augmenter le corpus, réduire le modèle, "
              "ou passer par la distillation.")
    elif epochs < 0.3:
        print("VERDICT : corpus insuffisant — le modèle verrait moins de 30 % "
              "d'une seule passe sur la cible.")
        print("  → bon pour valider le pipeline et faire des ablations ; "
              "pas pour produire le modèle final.")
        print(f"  → viser au moins {0.3*target/1e9:.2f} Md tokens, "
              f"ou distiller (20-30 M tokens suffisent alors).")
    else:
        print("VERDICT : corpus utilisable pour ce modèle "
              f"({epochs:.1f} époques sur la cible).")
    print(f"  rappel : '{args.device}' = {peak*mfu:.1f} TFLOPS effectifs ; "
          "mesurer avec --log-flops.")


if __name__ == "__main__":
    main()
