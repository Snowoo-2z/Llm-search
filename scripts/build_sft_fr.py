#!/usr/bin/env python3
"""
Construit un petit corpus SFT français à partir de sources locales ou HF,
avec filtrage qualité + français + licence.

Exemple d'usage (local) :
  python scripts/build_sft_fr.py --src data/sft_seed --out data/sft_fr --max-mb 100

Exemple HF (nécessite datasets) :
  python scripts/build_sft_fr.py --hf OpenLLM-France/Claire-Dialogue-French-0.1 --hf-field content --out data/sft_fr
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blackcode.data import build, build_parser

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", action="append", default=["data/fr_seed"], help="répertoires locaux")
    parser.add_argument("--hf", action="append", default=None)
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-field", default="content")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-mb", type=int, default=100)
    parser.add_argument("--vocab", type=int, default=24576)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--french-only", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    # Construit un parser compatible avec blackcode.data
    # On réutilise build_parser mais on force agentic_frac=0, fim_rate=0 pour SFT
    from blackcode.data import build as build_fn, Deduper, Doc
    import re, random, json
    from pathlib import Path as P

    # Simule args pour build()
    class Args:
        src = args.src
        exts = ".jsonl,.json,.md,.txt"
        min_bytes = 100
        max_bytes = 400_000
        max_mb = args.max_mb
        no_near = False
        licence = "all"
        keep_if = None
        french_only = args.french_only
        hf = args.hf
        hf_split = args.hf_split
        hf_field = args.hf_field
        hf_license_field = "license"
        require_hf_license = False
        allowed_hf_license = None
        seed = args.seed
        agentic_frac = 0.0
        agentic_exec = False
        max_spawns = 0
        min_quality_score = 0.0

    result = build_fn(Args)
    docs = result["docs"]
    print(f"[sft] {len(docs)} docs, {result['stats']['bytes']/1e6:.2f} Mo")

    # Pack avec vocab
    from blackcode.data import train_tokenizer, pack, load_tokenizer_size
    out = P(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_tokenizer(docs, args.vocab, out, sample_chars=20_000_000, seed=args.seed)
    meta = pack(docs, out / "tokenizer.json", out, args.seq_len, fim_rate=0, val_frac=0.05, seed=args.seed)
    meta["stats"]=result["stats"]
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False)+"\n")
    print(f"[sft] écrit dans {out}, train_tokens={meta['train_tokens']}")

if __name__=="__main__":
    main()
