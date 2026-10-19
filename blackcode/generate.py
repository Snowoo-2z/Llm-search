"""Génération de texte / code à partir d'un checkpoint BlackCode."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import ModelConfig
from .corpus import load_tokenizer
from .model import build_model


def load_checkpoint(path: str | Path, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["cfg"])
    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def main() -> None:
    p = argparse.ArgumentParser(description="Génère du code avec BlackCode")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--tokenizer", type=str, required=True)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--prompt-file", type=str, default=None)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    prompt = args.prompt or ""
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text()
    if not prompt:
        prompt = "# "

    model, cfg = load_checkpoint(args.ckpt, device)
    tok = load_tokenizer(args.tokenizer)
    ids = tok.encode(prompt, add_special_tokens=False).ids
    if not ids:
        ids = [tok.token_to_id("<|endoftext|>")]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    eos_id = tok.token_to_id("<|endoftext|>")
    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=eos_id,
        repetition_penalty=args.repetition_penalty,
    )
    text = tok.decode(out[0].tolist(), skip_special_tokens=False)
    print("=" * 72)
    print(text)
    print("=" * 72)
    if args.out:
        Path(args.out).write_text(text)
        print(f"[generate] écrit {args.out}")


if __name__ == "__main__":
    main()
