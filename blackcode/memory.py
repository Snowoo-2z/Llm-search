"""Budget mémoire / RAM d'un modèle BlackCode.

Toutes les formules sont explicites et vérifiables. Le but est de répondre à
la question : « avec X Mo de RAM, quel couple (N paramètres, P bits) choisir ? »
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import ModelConfig


BYTES_PER_ELEM = {1.58: 0.25, 2: 0.25, 3: 0.375, 4: 0.5, 8: 1.0, 16: 2.0, 32: 4.0}


@dataclass
class MemoryBudget:
    weights_mb: float
    embedding_mb: float
    kv_cache_mb: float
    conv_state_mb: float
    runtime_mb: float
    total_mb: float
    kv_per_token_kb: float

    def as_dict(self) -> dict:
        return {
            "weights_mb": round(self.weights_mb, 1),
            "embedding_mb": round(self.embedding_mb, 1),
            "kv_cache_mb": round(self.kv_cache_mb, 1),
            "conv_state_mb": round(self.conv_state_mb, 1),
            "runtime_mb": round(self.runtime_mb, 1),
            "total_mb": round(self.total_mb, 1),
            "kv_per_token_kb": round(self.kv_per_token_kb, 2),
        }


def weight_mb(params: int, bits: float) -> float:
    """Poids + échelles (≈ 0,1 % pour des échelles fp16 par ligne)."""
    if bits not in BYTES_PER_ELEM:
        raise ValueError(f"bits non tabulé : {bits}")
    bytes_per_param = BYTES_PER_ELEM[bits]
    # échelle par ligne fp16 : 2 octets / largeur moyenne réaliste (~1024)
    scale_overhead = 2.0 / 1024.0
    return params * (bytes_per_param + scale_overhead) / (1024 ** 2)


def kv_cache_mb(cfg: ModelConfig, context: int, batch: int = 1,
                kv_bytes: int = 2) -> float:
    """Cache KV des seules couches d'attention (GQA)."""
    n_attn = cfg.n_attention_layers
    per_token = n_attn * 2 * cfg.n_kv_heads * cfg.head_dim * kv_bytes
    return batch * context * per_token / (1024 ** 2)


def kv_per_token_bytes(cfg: ModelConfig, kv_bytes: int = 2) -> float:
    return cfg.n_attention_layers * 2 * cfg.n_kv_heads * cfg.head_dim * kv_bytes


def conv_state_mb(cfg: ModelConfig, batch: int = 1, state_bytes: int = 2) -> float:
    """k-1 vecteurs par couche de convolution, indépendant du contexte."""
    return batch * cfg.n_conv_layers * cfg.d_model * (cfg.conv_kernel - 1) * state_bytes / (1024 ** 2)


def budget(cfg: ModelConfig, context: int, weight_bits: float = 2.0,
           embed_bits: float = 8.0, runtime_mb: float = 180.0,
           batch: int = 1, kv_bytes: int = 2) -> MemoryBudget:
    parts = cfg.param_breakdown()
    w = weight_mb(parts["non_embedding"], weight_bits)
    e = weight_mb(parts["embeddings"], embed_bits)
    kv = kv_cache_mb(cfg, context, batch, kv_bytes)
    cv = conv_state_mb(cfg, batch)
    return MemoryBudget(
        weights_mb=w,
        embedding_mb=e,
        kv_cache_mb=kv,
        conv_state_mb=cv,
        runtime_mb=runtime_mb,
        total_mb=w + e + kv + cv + runtime_mb,
        kv_per_token_kb=kv_per_token_bytes(cfg, kv_bytes) / 1024,
    )


# ---------------------------------------------------------------------------
# Choix (N, bits) sous contrainte de RAM.
# ---------------------------------------------------------------------------
# Loi de mise à l'échelle « précision » de Kumar et al. (ICLR 2025) :
#   N_eff = N * (1 - exp(-P / gamma))
#   L = A * N_eff^-alpha + B * D^-beta + E
# Constantes de Chinchilla (A, B, alpha, beta, E). gamma est un paramètre de
# sensibilité : nous le faisons varier au lieu de prétendre à une valeur exacte.
CHINCHILLA = dict(A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69)


def predicted_loss(params: int, bits: float, tokens_per_param: float = 20.0,
                   gamma: float = 6.0) -> float:
    c = CHINCHILLA
    n_eff = params * (1.0 - math.exp(-bits / gamma))
    tokens = max(tokens_per_param * params, 1.0)
    return c["A"] * n_eff ** (-c["alpha"]) + c["B"] * tokens ** (-c["beta"]) + c["E"]


def scan_bits(cfg_template: ModelConfig, ram_mb: float, context: int,
              runtime_mb: float = 180.0, gamma: float = 6.0,
              bits_grid=(1.58, 2, 3, 4, 8, 16),
              min_scale: float = 0.2, max_scale: float = 8.0,
              n_points: int = 96) -> list[dict]:
    """Pour chaque précision, trouve le plus grand modèle qui tient en RAM.

    Balayage géométrique plutôt que bissection : la mise à l'échelle est
    discrète (n_heads multiples de 8), une bissection peut rester bloquée.
    """
    results = []
    for bits in bits_grid:
        best = None
        for i in range(n_points):
            scale = min_scale * (max_scale / min_scale) ** (i / (n_points - 1))
            cfg = _scaled_config(cfg_template, scale)
            b = budget(cfg, context, bits, 8.0, runtime_mb)
            if b.total_mb <= ram_mb and (best is None or cfg.num_params > best[0].num_params):
                best = (cfg, b)
        if best is None:
            continue
        cfg, b = best
        results.append({
            "bits": bits,
            "params": cfg.num_params,
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "predicted_loss": predicted_loss(cfg.num_params, bits, gamma=gamma),
            "memory": b.as_dict(),
            "config": cfg,
        })
    return results


def best_config_under_budget(ram_mb: float, cfg_template: ModelConfig,
                             context: int, runtime_mb: float = 180.0,
                             bits_grid=(1.58, 2, 3, 4, 8, 16),
                             gamma: float = 6.0) -> dict:
    rows = scan_bits(cfg_template, ram_mb, context, runtime_mb, gamma, bits_grid)
    if not rows:
        raise ValueError("aucune configuration ne tient dans ce budget")
    return min(rows, key=lambda r: r["predicted_loss"])


def _scaled_config(cfg: ModelConfig, scale: float) -> ModelConfig:
    """Met à l'échelle largeur et profondeur.

    N ∝ d²·L. On choisit d ∝ S^0,4 et L ∝ S^0,2, donc N ∝ S^0,8·S^0,2 = S :
    on grandit un peu plus en largeur qu'en profondeur, sans devenir plat.
    """
    import copy

    new = copy.deepcopy(cfg)
    width_scale = scale ** 0.4
    depth_scale = scale ** 0.2
    span = cfg.head_dim
    # n_heads multiple de 8 : évite les nombres premiers (KV heads divisibles)
    raw_heads = max(1, round(cfg.d_model * width_scale / span))
    new.n_heads = max(8, int(round(raw_heads / 8)) * 8)
    new.d_model = new.n_heads * span
    new.n_kv_heads = min(cfg.n_kv_heads, new.n_heads)
    while new.n_heads % new.n_kv_heads != 0:
        new.n_kv_heads -= 1
    new.ffn_hidden = max(64, int(round(cfg.ffn_hidden * width_scale / 64)) * 64)
    new.n_layers = max(4, min(64, int(round(cfg.n_layers * depth_scale))))
    new.__post_init__()
    return new


def format_table(rows: list[dict]) -> str:
    head = "| bits | params | d_model | couches | poids+emb (Mo) | KV (Mo) | total (Mo) | perte prédite |"
    sep = "|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [head, sep]
    for r in rows:
        m = r["memory"]
        lines.append(
            f"| {r['bits']} | {r['params']/1e6:.0f} M | {r['d_model']} | {r['n_layers']} | "
            f"{m['weights_mb']+m['embedding_mb']:.0f} | {m['kv_cache_mb']:.0f} | "
            f"{m['total_mb']:.0f} | {r['predicted_loss']:.4f} |"
        )
    return "\n".join(lines)
