"""Export d'un checkpoint vers le format compact ``.bcpkg``.

Format (little-endian) :
    magic  : b"BCPK1\\n"                  (6 octets)
    u32    : longueur du header JSON
    header : JSON utf-8 (config, table des tenseurs, offsets)
    data   : blocs de tenseurs concaténés

Types :
- ``ternary2`` : matrices ternaires, 2 bits/poids + échelle fp16 par ligne ;
- ``int8row``  : embeddings int8 + échelle fp16 par ligne ;
- ``fp16``     : normes, convolutions, biais.

Deux usages :
- ``python -m blackcode.export --ckpt runs/.../best.pt --out m.bcpkg``
- ``write_package(cfg, specs, ...)`` depuis ``scripts/make_random_bcpkg.py``,
  qui génère un paquet de la bonne taille **sans matérialiser tout le modèle**
  (utile pour mesurer la RAM réelle avant l'entraînement).
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig
from .memory import conv_state_mb, kv_cache_mb
from .model import TernaryLinear, build_model
from .quant import int8_quantize_rows, pack_ternary, ternary_codes

MAGIC = b"BCPK1\n"


@dataclass
class TensorSpec:
    name: str
    kind: str          # "ternary" | "embedding" | "fp16"
    shape: tuple[int, ...]


def tensor_specs(model, embed_bits: int = 8) -> list[TensorSpec]:
    """Liste ordonnée des tenseurs et de leur encodage."""
    ternary = {name for name, mod in model.named_modules()
               if isinstance(mod, TernaryLinear) and mod.ternary}
    specs: list[TensorSpec] = []
    for name, p in model.named_parameters():
        shape = tuple(p.shape)
        layer = name.rsplit(".", 1)[0]
        if layer in ternary and p.ndim == 2:
            kind = "ternary"
        elif name == "tok_emb.weight" and embed_bits == 8:
            kind = "embedding"
        else:
            kind = "fp16"
        specs.append(TensorSpec(name=name, kind=kind, shape=shape))
    return specs


def specs_from_meta_model(cfg: ModelConfig, embed_bits: int = 8) -> list[TensorSpec]:
    """Mêmes spécifications, sans allouer les poids (device « meta »)."""
    with torch.device("meta"):
        model = build_model(cfg, "meta")
    return tensor_specs(model, embed_bits)


# ---------------------------------------------------------------------------
def _write_payload(tmp, arr: np.ndarray, spec: TensorSpec, offset: int,
                   category_bytes: dict) -> dict:
    entry = {"name": spec.name, "shape": list(arr.shape)}

    if spec.kind == "ternary":
        tensor = torch.from_numpy(arr)
        codes, scales = ternary_codes(tensor)
        packed = pack_ternary(codes)
        data = packed.tobytes()
        entry.update(dtype="ternary2", offset=offset, nbytes=len(data),
                     packed_cols=int(packed.shape[1]),
                     in_features=int(arr.shape[1]))
        tmp.write(data)
        offset += len(data)
        scale_data = scales.astype(np.float16).tobytes()
        entry.update(scale_dtype="float16", scale_shape=[int(arr.shape[0])],
                     scale_offset=offset, scale_nbytes=len(scale_data))
        tmp.write(scale_data)
        offset += len(scale_data)
        category_bytes["ternary"] += len(data) + len(scale_data)
        return entry

    if spec.kind == "embedding":
        q, scales = int8_quantize_rows(arr)
        data = q.tobytes()
        entry.update(dtype="int8row", offset=offset, nbytes=len(data))
        tmp.write(data)
        offset += len(data)
        scale_data = scales.tobytes()
        entry.update(scale_dtype="float16", scale_shape=[int(arr.shape[0])],
                     scale_offset=offset, scale_nbytes=len(scale_data))
        tmp.write(scale_data)
        offset += len(scale_data)
        category_bytes["embedding"] += len(data) + len(scale_data)
        return entry

    data = arr.astype(np.float16).tobytes()
    entry.update(dtype="fp16", offset=offset, nbytes=len(data))
    tmp.write(data)
    category_bytes["other"] += len(data)
    return entry


def write_package(cfg: ModelConfig, specs: list[TensorSpec], out_path: str,
                  context: int, embed_bits: int = 8, runtime_mb: float = 180.0,
                  provider=None, extra_meta: dict | None = None) -> dict:
    """Écrit un ``.bcpkg``. ``provider(spec)`` doit retourner un tenseur numpy."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    category_bytes = {"ternary": 0, "embedding": 0, "other": 0}
    raw_fp16_bytes = 0.0

    with tempfile.NamedTemporaryFile(delete=False, dir=out.parent) as tmp:
        tmp_path = Path(tmp.name)
        offset = 0
        for spec in specs:
            if provider is None:
                raise ValueError("provider requis")
            arr = np.asarray(torch.as_tensor(provider(spec)).detach().cpu().float().numpy(),
                             dtype=np.float32)
            raw_fp16_bytes += arr.size * 2
            entry = _write_payload(tmp, arr, spec, offset, category_bytes)
            offset = entry.get("scale_offset", entry["offset"]) + entry.get(
                "scale_nbytes", entry["nbytes"])
            entries.append(entry)
            del arr

        tmp.flush()
        tmp_size = tmp_path.stat().st_size
        packed_mb = tmp_size / (1024 ** 2)
        kv_mb = kv_cache_mb(cfg, context)
        conv_mb = conv_state_mb(cfg)
        estimated_ram = packed_mb + kv_mb + conv_mb + runtime_mb
        total_params = sum(int(np.prod(e["shape"])) for e in entries)

        header = {
            "format": "blackcode-packed-v1",
            "created_by": "blackcode.export",
            "config": cfg.to_dict(),
            "quantization": {
                "ternary_weights": "2 bits + scale fp16 par ligne",
                "embeddings": f"{embed_bits} bits + scale fp16 par ligne",
                "other": "float16",
            },
            "context": context,
            "params": total_params,
            "effective_bits_per_param": 8.0 * tmp_size / max(total_params, 1),
            "raw_fp16_mb": raw_fp16_bytes / (1024 ** 2),
            "packed_mb": packed_mb,
            "category_mb": {k: v / (1024 ** 2) for k, v in category_bytes.items()},
            "estimated_runtime_ram_mb": estimated_ram,
            "tensors": entries,
        }
        if extra_meta:
            header.update(extra_meta)
        header_bytes = json.dumps(header).encode("utf-8")

        with out.open("wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<I", len(header_bytes)))
            f.write(header_bytes)
            with tmp_path.open("rb") as src:
                shutil.copyfileobj(src, f)

    tmp_path.unlink(missing_ok=True)

    manifest = {k: header[k] for k in
                ("format", "params", "effective_bits_per_param", "raw_fp16_mb",
                 "packed_mb", "category_mb", "context", "estimated_runtime_ram_mb")}
    manifest["config_name"] = cfg.name
    manifest["file"] = str(out)
    (out.parent / (out.name + ".json")).write_text(json.dumps(manifest, indent=2) + "\n")
    return header


def export(ckpt_path: str, out_path: str, context: int, embed_bits: int = 8,
           runtime_mb: float = 180.0) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["cfg"])
    model = build_model(cfg, "cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    specs = tensor_specs(model, embed_bits)
    state = model.state_dict()

    header = write_package(cfg, specs, out_path, context, embed_bits,
                           runtime_mb, provider=lambda spec: state[spec.name])

    print(f"[export] {out_path} : {header['packed_mb']:.1f} Mo "
          f"({header['effective_bits_per_param']:.2f} bits/paramètre effectifs)")
    print(f"[export] fp16 brut : {header['raw_fp16_mb']:.1f} Mo "
          f"→ compression ×{header['raw_fp16_mb']/max(header['packed_mb'],1e-9):.1f}")
    cat = header["category_mb"]
    print(f"[export] ternaire {cat['ternary']:.1f} Mo | "
          f"embeddings {cat['embedding']:.1f} Mo | autres {cat['other']:.2f} Mo")
    print(f"[export] RAM estimée à {context} tokens : "
          f"{header['estimated_runtime_ram_mb']:.0f} Mo "
          f"(dont KV {kv_cache_mb(cfg, context):.0f} Mo + runtime {runtime_mb:.0f} Mo)")
    return header


def main() -> None:
    p = argparse.ArgumentParser(description="Exporte un checkpoint BlackCode")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--embed-bits", type=int, default=8, choices=[8, 16])
    p.add_argument("--runtime-mb", type=float, default=180.0)
    args = p.parse_args()
    export(args.ckpt, args.out, args.context, args.embed_bits, args.runtime_mb)


if __name__ == "__main__":
    main()
