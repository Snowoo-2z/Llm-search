"""Tokenisation et préparation du corpus.

Deux commandes :

    python -m blackcode.corpus prepare --src /chemin/vers/code --out data/code
    python -m blackcode.corpus prepare --jsonl data.jsonl --out data/code

La première accepte une arborescence de fichiers (``.py`` par défaut), la
seconde un JSONL avec un champ texte (pratique pour un dataset HuggingFace
téléchargé hors ligne sur Colab).

Le tokenizer est un BPE byte-level : aucun token inconnu, donc tout programme
reste représentable, y compris les langages rares.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
from tokenizers import ByteLevelBPETokenizer, Tokenizer

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|pad|>",
    "<|user|>",
    "<|assistant|>",
    "<|code|>",
    "<|test|>",
]
EOS_TOKEN = "<|endoftext|>"


# ---------------------------------------------------------------------------
def iter_files(src: Path, suffixes: tuple[str, ...], min_bytes: int, max_bytes: int):
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        if any(part.startswith(".") or part == "__pycache__" for part in path.parts):
            continue
        if suffixes and path.suffix not in suffixes:
            continue
        size = path.stat().st_size
        if size < min_bytes or size > max_bytes:
            continue
        yield path


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def iter_jsonl_texts(path: Path, field: str):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get(field)
            if isinstance(text, str) and text:
                yield text


def dedupe_and_split(paths: list[Path], val_frac: float, seed: int):
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        text = read_text(path)
        if text is None or not text.strip():
            continue
        digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(path)
    rng = random.Random(seed)
    rng.shuffle(unique)
    n_val = max(1, int(len(unique) * val_frac)) if len(unique) > 10 else 0
    return unique[n_val:], unique[:n_val]


def train_tokenizer(files: list[Path], vocab_size: int, out: Path,
                    sample_mb: int = 64, seed: int = 0) -> None:
    print(f"[tokenizer] entraînement BPE byte-level, vocab={vocab_size}, "
          f"{len(files)} fichiers, échantillon≤{sample_mb} Mo")
    budget = sample_mb * 1024 * 1024

    def iterator():
        used = 0
        for path in files:
            text = read_text(path)
            if text is None:
                continue
            yield text
            used += len(text.encode("utf-8"))
            if used >= budget:
                break

    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(
        iterator(),
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
    )
    tok.save(str(out / "tokenizer.json"))
    (out / "tokenizer_config.json").write_text(json.dumps({
        "type": "ByteLevelBPE",
        "vocab_size": vocab_size,
        "special_tokens": SPECIAL_TOKENS,
        "eos_token": EOS_TOKEN,
    }, indent=2))
    print(f"[tokenizer] écrit {out/'tokenizer.json'} "
          f"({(out/'tokenizer.json').stat().st_size/1024:.0f} Ko)")


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def tokenize_split(files: list[Path], tok: Tokenizer, out_bin: Path,
                   batch_docs: int = 64) -> int:
    eos_id = tok.token_to_id(EOS_TOKEN)
    assert eos_id is not None, "token EOS absent du tokenizer"
    total = 0
    with out_bin.open("wb") as f:
        batch: list[str] = []
        for path in files:
            text = read_text(path)
            if text is None:
                continue
            batch.append(text)
            if len(batch) >= batch_docs:
                total += _flush(batch, tok, eos_id, f)
                batch.clear()
        if batch:
            total += _flush(batch, tok, eos_id, f)
    return total


def _flush(batch: list[str], tok: Tokenizer, eos_id: int, f) -> int:
    encodings = tok.encode_batch(batch, add_special_tokens=False)
    count = 0
    for enc in encodings:
        ids = np.asarray(enc.ids, dtype=np.uint16)
        ids.tofile(f)
        np.asarray([eos_id], dtype=np.uint16).tofile(f)
        count += len(ids) + 1
    return count


def tokenize_texts(texts, tok: Tokenizer, out_bin: Path, batch_docs: int = 64) -> int:
    eos_id = tok.token_to_id(EOS_TOKEN)
    total = 0
    with out_bin.open("wb") as f:
        batch: list[str] = []
        for text in texts:
            batch.append(text)
            if len(batch) >= batch_docs:
                total += _flush(batch, tok, eos_id, f)
                batch.clear()
        if batch:
            total += _flush(batch, tok, eos_id, f)
    return total


# ---------------------------------------------------------------------------
def cmd_prepare(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.jsonl:
        texts = list(iter_jsonl_texts(Path(args.jsonl), args.text_field))
        if not texts:
            sys.exit("Aucun texte trouvé dans le JSONL.")
        rng = random.Random(args.seed)
        rng.shuffle(texts)
        n_val = max(1, int(len(texts) * args.val_frac))
        train_texts, val_texts = texts[n_val:], texts[:n_val]
        print(f"[corpus] {len(texts)} documents JSONL (champ '{args.text_field}')")
        print("[tokenizer] échantillon de documents pour l'entraînement")
        # Tokenizer entraîné sur un échantillon déterministe du train.
        sample = train_texts[: max(1, min(len(train_texts), 20_000))]
        tok = ByteLevelBPETokenizer()
        tok.train_from_iterator(sample, vocab_size=args.vocab, min_frequency=2,
                                special_tokens=SPECIAL_TOKENS, show_progress=False)
        tok.save(str(out / "tokenizer.json"))
        tokenizer = load_tokenizer(out / "tokenizer.json")
        n_train = tokenize_texts(train_texts, tokenizer, out / "train.bin", args.batch_docs)
        n_val = tokenize_texts(val_texts, tokenizer, out / "val.bin", args.batch_docs)
        n_docs = len(texts)
    else:
        src = Path(args.src)
        if not src.exists():
            sys.exit(f"Source introuvable : {src}")
        suffixes = tuple(args.suffixes.split(","))
        paths = list(iter_files(src, suffixes, args.min_bytes, args.max_bytes))
        print(f"[corpus] {len(paths)} fichiers candidats dans {src}")
        train_files, val_files = dedupe_and_split(paths, args.val_frac, args.seed)
        print(f"[corpus] après déduplication : {len(train_files)} train / {len(val_files)} val")
        if not train_files:
            sys.exit("Aucun fichier exploitable.")
        train_tokenizer(train_files, args.vocab, out, args.tokenizer_sample_mb, args.seed)
        tokenizer = load_tokenizer(out / "tokenizer.json")
        n_train = tokenize_split(train_files, tokenizer, out / "train.bin", args.batch_docs)
        n_val = tokenize_split(val_files, tokenizer, out / "val.bin", args.batch_docs)
        n_docs = len(train_files) + len(val_files)

    meta = {
        "vocab_size": args.vocab,
        "train_tokens": n_train,
        "val_tokens": n_val,
        "documents": n_docs,
        "eos_id": load_tokenizer(out / "tokenizer.json").token_to_id(EOS_TOKEN),
        "tokenizer": str(out / "tokenizer.json"),
        "source": args.src or args.jsonl,
        "seed": args.seed,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[corpus] train : {n_train/1e6:.3f} M tokens | val : {n_val/1e6:.3f} M tokens")
    print(f"[corpus] métadonnées : {out/'meta.json'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prépare un corpus de code tokenisé")
    p.add_argument("command", nargs="?", default="prepare", choices=["prepare"])
    p.add_argument("--src", type=str, default=None, help="répertoire de fichiers")
    p.add_argument("--jsonl", type=str, default=None, help="fichier JSONL")
    p.add_argument("--text-field", type=str, default="content")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--vocab", type=int, default=8192)
    p.add_argument("--suffixes", type=str, default=".py,.pyi,.md,.txt")
    p.add_argument("--min-bytes", type=int, default=500)
    p.add_argument("--max-bytes", type=int, default=200_000)
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--batch-docs", type=int, default=64)
    p.add_argument("--tokenizer-sample-mb", type=int, default=64)
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if not args.src and not args.jsonl:
        parser.error("fournir --src ou --jsonl")
    cmd_prepare(args)
