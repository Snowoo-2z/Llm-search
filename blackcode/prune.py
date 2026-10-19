"""Élagage de données par bande de perte (« learnable band »).

Idée : à budget de calcul fixe, tous les tokens ne valent pas le même prix.

- les documents **très faciles** (perte basse) sont déjà appris : les répéter
  n'apprend presque rien ;
- les documents **très durs ou bruités** (perte haute) sont des outliers,
  des fichiers générés, des duplications déguisées : ils coûtent cher et
  n'améliorent pas la généralisation.

On mesure la perte d'un petit modèle de référence sur chaque fichier, puis on
garde la bande de quantiles ``[--lo, --hi]``. La littérature sur l'élagage
de données (Sorscher et al., 2022 ; DCLM, 2024) rapporte 1,5 à 3× moins de
tokens à qualité égale. **À valider sur votre corpus** : garder la bande est
un pari, pas une garantie.

    python -m blackcode.prune \
        --ckpt runs/t4_small/best.pt --tokenizer data/code/tokenizer.json \
        --src /content/corpus --out /content/corpus_pruned \
        --lo 0.05 --hi 0.85 --copy-mode symlink
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .corpus import iter_files, load_tokenizer, read_text
from .generate import load_checkpoint


def select_band(losses, lo: float = 0.0, hi: float = 0.9):
    """Retourne (indices gardés, seuils bas/haut) selon des quantiles."""
    values = np.asarray(losses, dtype=np.float64)
    if values.size == 0:
        return np.array([], dtype=int), float("nan"), float("nan")
    lo_thr = float(np.quantile(values, lo))
    hi_thr = float(np.quantile(values, hi))
    keep = np.where((values >= lo_thr) & (values <= hi_thr))[0]
    return keep, lo_thr, hi_thr


@torch.no_grad()
def document_losses(model, tokenizer, paths: list[Path], max_len: int,
                    batch_size: int, device: str) -> np.ndarray:
    """Perte moyenne par document, en lots avec padding masqué."""
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    pad_id = eos_id if eos_id is not None else 0
    losses: list[float] = []
    buffer: list[list[int]] = []

    def flush():
        if not buffer:
            return
        width = max(len(ids) for ids in buffer)
        x = torch.full((len(buffer), width), pad_id, dtype=torch.long, device=device)
        y = torch.full((len(buffer), width), -100, dtype=torch.long, device=device)
        for row, ids in enumerate(buffer):
            if len(ids) < 2:
                continue
            x[row, :len(ids)] = torch.tensor(ids, device=device)
            y[row, :len(ids) - 1] = torch.tensor(ids[1:], device=device)
        hidden, _ = model.forward_hidden(x)
        logits = model._logits(hidden).float()
        token_loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), y.view(-1),
            ignore_index=-100, reduction="none",
        ).view(y.shape)
        valid = (y != -100).sum(dim=1).clamp_min(1)
        per_doc = token_loss.sum(dim=1) / valid
        losses.extend(per_doc.cpu().tolist())
        buffer.clear()

    for path in paths:
        text = read_text(path)
        if text is None:
            continue
        ids = tokenizer.encode(text[: max_len * 8], add_special_tokens=False).ids[:max_len]
        if len(ids) < 2:
            continue
        buffer.append(ids)
        if len(buffer) >= batch_size:
            flush()
    flush()
    return np.asarray(losses, dtype=np.float64)


def main() -> None:
    p = argparse.ArgumentParser(description="Élague un corpus par bande de perte")
    p.add_argument("--ckpt", type=str, required=True,
                   help="modèle de référence (petit modèle suffit)")
    p.add_argument("--tokenizer", type=str, required=True)
    p.add_argument("--src", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--suffixes", type=str, default=".py,.pyi")
    p.add_argument("--min-bytes", type=int, default=500)
    p.add_argument("--max-bytes", type=int, default=200_000)
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lo", type=float, default=0.0, help="quantile bas à écarter")
    p.add_argument("--hi", type=float, default=0.9, help="quantile haut à écarter")
    p.add_argument("--copy-mode", type=str, default="symlink",
                   choices=["symlink", "hardlink", "copy", "list"])
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    model, cfg = load_checkpoint(args.ckpt, args.device)
    tokenizer = load_tokenizer(args.tokenizer)
    suffixes = tuple(args.suffixes.split(","))
    paths = list(iter_files(Path(args.src), suffixes, args.min_bytes, args.max_bytes))
    if args.max_files:
        paths = paths[: args.max_files]
    print(f"[prune] {len(paths)} fichiers candidats")

    losses = document_losses(model, tokenizer, paths, args.max_len,
                             args.batch_size, args.device)
    if len(losses) != len(paths):
        print(f"[prune] ⚠ {len(paths) - len(losses)} fichiers ignorés (trop courts ou illisibles)")
        paths = paths[: len(losses)]
    if len(losses) == 0:
        raise SystemExit("aucun document exploitable")

    keep_idx, lo_thr, hi_thr = select_band(losses, args.lo, args.hi)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    kept_tokens = 0
    kept = []
    for i in keep_idx:
        src = paths[int(i)]
        kept.append({"path": str(src), "loss": float(losses[int(i)])})
        if args.copy_mode == "list":
            continue
        dst = out_dir / src.name
        if dst.exists():
            dst = out_dir / f"{src.stem}_{i}{src.suffix}"
        try:
            if args.copy_mode == "symlink":
                os.symlink(src.resolve(), dst)
            elif args.copy_mode == "hardlink":
                os.link(src, dst)
            else:
                shutil.copy2(src, dst)
        except OSError as exc:
            print(f"[prune] {src}: {exc}")
        kept_tokens += src.stat().st_size

    total_bytes = sum(pth.stat().st_size for pth in paths)
    report = {
        "files_in": len(paths),
        "files_out": len(kept),
        "bytes_in": total_bytes,
        "bytes_out": kept_tokens,
        "kept_fraction": len(kept) / max(len(paths), 1),
        "loss_quantiles": {
            "lo": args.lo, "hi": args.hi,
            "lo_threshold": lo_thr, "hi_threshold": hi_thr,
            "median": float(np.median(losses)),
        },
        "assumption": "gain de 1,5 à 3× en tokens à qualité égale, à valider",
    }
    (out_dir / "prune_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.copy_mode == "list":
        (out_dir / "kept.txt").write_text("\n".join(k["path"] for k in kept) + "\n")
        (out_dir / "kept_losses.json").write_text(json.dumps(kept[:10000], indent=2) + "\n")

    print(f"[prune] bande [{args.lo:.2f}, {args.hi:.2f}] : "
          f"seuils {lo_thr:.3f} – {hi_thr:.3f} (médiane {np.median(losses):.3f})")
    print(f"[prune] {len(kept)}/{len(paths)} fichiers conservés "
          f"({report['kept_fraction']:.1%}), "
          f"{kept_tokens/1e6:.1f}/{total_bytes/1e6:.1f} Mo")
    print(f"[prune] rapport : {out_dir/'prune_report.json'}")
    print("[prune] relancer ensuite : python -m blackcode.corpus prepare "
          f"--src {out_dir} --out data/code_pruned")


if __name__ == "__main__":
    main()
