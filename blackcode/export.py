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

Exemple :
    python -m blackcode.export --ckpt runs/t4_small/best.pt \
        --out artifacts/blackcode_t4.bcpkg --context 4096
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig
from .memory import conv_state_mb, kv_cache_mb
from .model import TernaryLinear, build_model
from .quant import int8_quantize_rows, pack_ternary, ternary_codes


def _ternary_param_names(model) -> set[str]:
    names = set()
    for mod_name, mod in model.named_modules():
        if isinstance(mod, TernaryLinear) and mod.ternary:
            names.add(f"{mod_name}.weight" if mod_name else "weight")
    return names


def export(ckpt_path: str, out_path: str, context: int, embed_bits: int = 8,
           runtime_mb: float = 180.0) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["cfg"])
    model = build_model(cfg, "cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    state = model.state_dict()
    ternary_names = _ternary_param_names(model)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    category_bytes = {"ternary": 0, "embedding": 0, "other": 0}
    raw_fp16_bytes = 0

    with tempfile.NamedTemporaryFile(delete=False, dir=out.parent) as tmp:
        tmp_path = Path(tmp.name)
        offset = 0
        for name, tensor in state.items():
            arr = tensor.detach().cpu().float().numpy()
            raw_fp16_bytes += arr.size * 2
            entry = {"name": name, "shape": list(arr.shape)}
            is_embedding = name == "tok_emb.weight"

            if name in ternary_names and arr.ndim == 2:
                codes, scales = ternary_codes(tensor)
                packed = pack_ternary(codes)
                data = packed.tobytes()
                tmp.write(data)
                entry.update(dtype="ternary2", offset=offset, nbytes=len(data),
                             packed_cols=int(packed.shape[1]), in_features=int(arr.shape[1]))
                offset += len(data)
                scale_data = scales.astype(np.float16).tobytes()
                tmp.write(scale_data)
                entry.update(scale_dtype="float16", scale_shape=[int(arr.shape[0])],
                             scale_offset=offset, scale_nbytes=len(scale_data))
                offset += len(scale_data)
                category_bytes["ternary"] += len(data) + len(scale_data)

            elif is_embedding and embed_bits == 8:
                q, scales = int8_quantize_rows(arr)
                data = q.tobytes()
                tmp.write(data)
                entry.update(dtype="int8row", offset=offset, nbytes=len(data))
                offset += len(data)
                scale_data = scales.tobytes()
                tmp.write(scale_data)
                entry.update(scale_dtype="float16", scale_shape=[int(arr.shape[0])],
                             scale_offset=offset, scale_nbytes=len(scale_data))
                offset += len(scale_data)
                category_bytes["embedding"] += len(data) + len(scale_data)

            else:
                data = arr.astype(np.float16).tobytes()
                tmp.write(data)
                entry.update(dtype="fp16", offset=offset, nbytes=len(data))
                offset += len(data)
                category_bytes["other"] += len(data)

            entries.append(entry)

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
        header_bytes = json.dumps(header).encode("utf-8")

        with out.open("wb") as f:
            f.write(b"BCPK1\n")
            f.write(struct.pack("<I", len(header_bytes)))
            f.write(header_bytes)
            with tmp_path.open("rb") as src:
                shutil.copyfileobj(src, f)

    tmp_path.unlink(missing_ok=True)

    manifest = {
        k: header[k] for k in
        ("format", "params", "effective_bits_per_param", "raw_fp16_mb",
         "packed_mb", "category_mb", "context", "estimated_runtime_ram_mb")
    }
    manifest["config_name"] = cfg.name
    manifest["file"] = str(out)
    (out.parent / (out.name + ".json")).write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"[export] {out} : {packed_mb:.1f} Mo "
          f"({header['effective_bits_per_param']:.2f} bits/paramètre effectifs)")
    print(f"[export] fp16 brut : {raw_fp16_bytes/(1024**2):.1f} Mo "
          f"→ compression ×{raw_fp16_bytes/max(tmp_size,1):.1f}")
    print(f"[export] ternaire {category_bytes['ternary']/(1024**2):.1f} Mo | "
          f"embeddings {category_bytes['embedding']/(1024**2):.1f} Mo | "
          f"autres {category_bytes['other']/(1024**2):.2f} Mo")
    print(f"[export] RAM estimée à {context} tokens : {estimated_ram:.0f} Mo "
          f"(dont KV {kv_mb:.0f} Mo + runtime {runtime_mb:.0f} Mo)")
    if estimated_ram > 1024:
        print("[export] ⚠ au-delà de 1 Go : réduire le contexte, les couches "
              "d'attention, ou passer le KV cache en int8.")
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
