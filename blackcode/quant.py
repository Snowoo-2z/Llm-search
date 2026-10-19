"""Quantification ternaire (BitNet b1.58) et outils de packaging.

Deux usages :
1. ``ternary_quantize`` : utilisé à l'entraînement (straight-through estimator)
   et à l'export, avec une échelle par ligne de sortie.
2. ``pack_ternary`` / ``unpack_ternary`` : format 2 bits compact utilisé par
   l'export ``.bcpkg``. Quatre poids tiennent dans un octet.

Codes : 0 -> -1, 1 -> 0, 2 -> +1 (jamais 3).
"""
from __future__ import annotations

import numpy as np
import torch


def ternary_quantize(weight: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantifie en {-1, 0, +1} avec une échelle absmean par ligne.

    Retourne (poids quantifié déquantifié, échelles de forme [out, 1]).
    """
    scale = weight.abs().mean(dim=1, keepdim=True).clamp_min(eps)
    q = torch.clamp(torch.round(weight / scale), -1.0, 1.0)
    return q * scale, scale


def ternary_ste(weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Poids ternaire avec straight-through estimator (gradient identité)."""
    wq, _ = ternary_quantize(weight, eps)
    return weight + (wq - weight).detach()


@torch.no_grad()
def ternary_codes(weight: torch.Tensor, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Retourne les codes {0,1,2} et les échelles fp32 d'une matrice."""
    scale = weight.abs().mean(dim=1, keepdim=True).clamp_min(eps)
    q = torch.clamp(torch.round(weight / scale), -1.0, 1.0)
    codes = (q + 1.0).to(torch.uint8).cpu().numpy()
    return codes, scale.squeeze(1).cpu().numpy()


def pack_ternary(codes: np.ndarray) -> np.ndarray:
    """[out, in] uint8 dans {0,1,2} -> [out, ceil(in/4)] uint8 (4 par octet)."""
    if codes.ndim != 2:
        raise ValueError("codes doit être 2D")
    out_f, in_f = codes.shape
    padded = np.zeros((out_f, (in_f + 3) // 4 * 4), dtype=np.uint8)
    padded[:, :in_f] = codes
    padded = padded.reshape(out_f, -1, 4)
    packed = (
        padded[:, :, 0]
        | (padded[:, :, 1] << 2)
        | (padded[:, :, 2] << 4)
        | (padded[:, :, 3] << 6)
    )
    return packed.astype(np.uint8)


def unpack_ternary(packed: np.ndarray, in_features: int) -> np.ndarray:
    """[out, ceil(in/4)] -> [out, in] uint8 dans {0,1,2}."""
    out_f = packed.shape[0]
    out = np.empty((out_f, packed.shape[1] * 4), dtype=np.uint8)
    for j in range(4):
        out[:, j::4] = (packed >> (2 * j)) & 0b11
    return out[:, :in_features]


def dequantize_ternary(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return (codes.astype(np.float32) - 1.0) * scales[:, None]


def int8_quantize_rows(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantification int8 symétrique par ligne (pour les embeddings)."""
    max_abs = np.abs(weight).max(axis=1, keepdims=True)
    scale = np.maximum(max_abs / 127.0, 1e-8)
    q = np.clip(np.round(weight / scale), -127, 127).astype(np.int8)
    return q, scale.squeeze(1).astype(np.float16)


def packing_efficiency(in_features: int, out_features: int) -> float:
    """Bits effectifs par poids, échelles incluses (fp16 par ligne)."""
    packed_bytes = out_features * ((in_features + 3) // 4)
    scale_bytes = out_features * 2
    return 8.0 * (packed_bytes + scale_bytes) / (in_features * out_features)
