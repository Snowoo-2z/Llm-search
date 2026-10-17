"""Architecture BlackCode.

Briques :
- ``TernaryLinear``  : matrice ternaire {-1,0,+1}, échelle absmean par ligne, STE.
- ``Attention``      : GQA + RoPE + QK-norm, cache KV O(T).
- ``ConvBlock``      : convolution courte doublement portée (LFM2) + SwiGLU.
- ``BlackCode``      : alternance 3 convolutions / 1 attention, embeddings liés,
  partage de couches optionnel (``loop``), perte chunkée pour économiser la VRAM.

Le choix « convolution courte + attention » donne un cache mémoire constant sur
75 % des couches : c'est ce qui rend le budget < 1 Go réaliste même à 8k de
contexte. Les convolutions sont aussi rapides sur CPU que des GEMM, alors que
l'attention est limitée par la bande passante mémoire.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .quant import ternary_ste


# ---------------------------------------------------------------------------
# Normalisation / quantification
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class TernaryLinear(nn.Module):
    """Linear dont la matrice est ternaire pendant le forward.

    Les poids restent en fp32 (poids « maîtres ») pour l'optimiseur ; le
    gradient passe par l'identité (STE). À l'export, seuls 2 bits/poids et une
    échelle fp16 par ligne sont stockés.
    """

    def __init__(self, in_features: int, out_features: int, ternary: bool = True,
                 bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ternary = ternary
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = ternary_ste(self.weight) if self.ternary else self.weight
        return F.linear(x, w, self.bias)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int, ternary: bool):
        super().__init__()
        self.w1 = TernaryLinear(d_model, hidden, ternary)
        self.w3 = TernaryLinear(d_model, hidden, ternary)
        self.w2 = TernaryLinear(hidden, d_model, ternary)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------
def precompute_rope(head_dim: int, max_seq_len: int, base: float = 10_000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(max_seq_len).float(), inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x : [B, H, T, D] ; cos/sin : [T, D/2]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ---------------------------------------------------------------------------
# Blocs
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        d = cfg.d_model
        self.q_proj = TernaryLinear(d, cfg.n_heads * cfg.head_dim, cfg.ternary)
        self.k_proj = TernaryLinear(d, cfg.n_kv_heads * cfg.head_dim, cfg.ternary)
        self.v_proj = TernaryLinear(d, cfg.n_kv_heads * cfg.head_dim, cfg.ternary)
        self.o_proj = TernaryLinear(cfg.n_heads * cfg.head_dim, d, cfg.ternary)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)

    def forward(self, x, cos, sin, cache=None, use_cache=False):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v) if use_cache else None
        # is_causal n'est correct que si l'on traite tout le préfixe d'un coup.
        causal = t > 1
        try:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=causal, dropout_p=0.0,
                enable_gqa=(self.n_kv_heads != self.n_heads),
            )
        except TypeError:  # torch < 2.5
            if self.n_kv_heads != self.n_heads:
                rep = self.n_heads // self.n_kv_heads
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=causal, dropout_p=0.0)
        y = y.transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(y), new_cache


class AttentionBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden, cfg.ternary)

    def forward(self, x, cos, sin, cache=None, use_cache=False):
        y, new_cache = self.attn(self.attn_norm(x), cos, sin, cache, use_cache)
        x = x + y
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class ConvBlock(nn.Module):
    """Bloc à convolution courte, doublement portée (style LFM2).

    y = x + out_proj( conv(in_proj_a(x)) * in_proj_b(x) )
    puis FFN SwiGLU. Le cache ne stocke que k-1 vecteurs par couche.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.kernel = cfg.conv_kernel
        self.norm = RMSNorm(d, cfg.norm_eps)
        self.in_proj = TernaryLinear(d, 2 * d, cfg.ternary)
        self.conv = nn.Conv1d(d, d, kernel_size=self.kernel, groups=d, bias=False)
        self.out_proj = TernaryLinear(d, d, cfg.ternary)
        self.ffn_norm = RMSNorm(d, cfg.norm_eps)
        self.ffn = SwiGLU(d, cfg.ffn_hidden, cfg.ternary)
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))

    def forward(self, x, cos=None, sin=None, cache=None, use_cache=False):
        b, t, _ = x.shape
        a, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        a_t = a.transpose(1, 2)                                   # [B, d, T]
        prev = cache if cache is not None else a_t.new_zeros(b, a_t.shape[1], self.kernel - 1)
        seq = torch.cat([prev, a_t], dim=-1)                      # [B, d, T+k-1]
        conv = self.conv(seq)[..., :t]                            # [B, d, T]
        new_cache = seq[..., -(self.kernel - 1):].detach() if use_cache else None
        x = x + self.out_proj(conv.transpose(1, 2) * gate)
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


# ---------------------------------------------------------------------------
# Modèle complet
# ---------------------------------------------------------------------------
class BlackCode(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [AttentionBlock(cfg) if cfg.is_attention_layer(i) else ConvBlock(cfg)
             for i in range(cfg.n_layers)]
        )
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = None if cfg.tie_embeddings else nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        self._scale_residual_projections()

    # ------------------------------------------------------------------
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, TernaryLinear)):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def _scale_residual_projections(self) -> None:
        std = self.cfg.init_std / math.sqrt(2 * self.cfg.n_layers * self.cfg.loop)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("ffn.w2.weight") \
               or name.endswith("out_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=std)

    # ------------------------------------------------------------------
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        weight = self.tok_emb.weight
        return F.linear(hidden, weight)

    def forward_hidden(self, idx, kv_caches=None, start_pos: int = 0,
                       use_cache: bool = False, checkpoint=False):
        """``checkpoint`` : False/"none", "selective" (1 bloc sur 2) ou "full"."""
        b, t = idx.shape
        if start_pos + t > self.cfg.max_seq_len:
            raise ValueError("séquence plus longue que max_seq_len")
        cos = self.rope_cos[start_pos:start_pos + t]
        sin = self.rope_sin[start_pos:start_pos + t]
        x = self.tok_emb(idx)
        new_caches = [None] * (self.cfg.loop * self.cfg.n_layers) if use_cache else None
        slot = 0
        for _ in range(self.cfg.loop):
            for block in self.blocks:
                cache = kv_caches[slot] if kv_caches is not None else None
                if checkpoint in (True, "full"):
                    do_checkpoint = True
                elif checkpoint == "selective":
                    do_checkpoint = (slot % 2 == 0)  # schéma √L
                else:
                    do_checkpoint = False
                if do_checkpoint and self.training:
                    x, new_cache = torch.utils.checkpoint.checkpoint(
                        block, x, cos, sin, cache, use_cache, use_reentrant=False
                    )
                else:
                    x, new_cache = block(x, cos, sin, cache, use_cache)
                if use_cache:
                    new_caches[slot] = new_cache
                slot += 1
        return self.norm_f(x), new_caches

    def forward(self, idx, kv_caches=None, start_pos: int = 0, use_cache: bool = False):
        """Retourne les logits du dernier token (utile pour la génération)."""
        hidden, new_caches = self.forward_hidden(idx, kv_caches, start_pos, use_cache)
        return self._logits(hidden[:, -1, :]), new_caches

    def forward_loss(self, idx, targets, logit_chunk: int = 256,
                     checkpoint=False):
        """Perte en calculant les logits par morceaux (VRAM quasi constante)."""
        hidden, _ = self.forward_hidden(idx, checkpoint=checkpoint)
        b, t, _ = hidden.shape
        vocab = self.cfg.vocab_size
        total = hidden.new_zeros(())
        for i in range(0, t, logit_chunk):
            logits = self._logits(hidden[:, i:i + logit_chunk]).float()
            total = total + F.cross_entropy(
                logits.reshape(-1, vocab),
                targets[:, i:i + logit_chunk].reshape(-1),
                reduction="sum",
            )
        return total / (b * t)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8,
                 top_k: int | None = 50, eos_id: int | None = None,
                 repetition_penalty: float = 1.0):
        self.eval()
        device = idx.device
        caches = [None] * (self.cfg.loop * self.cfg.n_layers)
        logits, caches = self.forward(idx, caches, start_pos=0, use_cache=True)
        pos = idx.shape[1]
        out = idx
        for _ in range(max_new_tokens):
            logits = logits[0].float()
            if repetition_penalty != 1.0:
                for token in set(out[0].tolist()):
                    if logits[token] > 0:
                        logits[token] /= repetition_penalty
                    else:
                        logits[token] *= repetition_penalty
            if temperature <= 1e-6:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    values, _ = torch.topk(logits, min(top_k, logits.numel()))
                    logits[logits < values[-1]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_id.view(1, 1).to(device)], dim=1)
            if eos_id is not None and next_id.item() == eos_id:
                break
            if pos >= self.cfg.max_seq_len:
                break
            logits, caches = self.forward(
                next_id.view(1, 1).to(device), caches, start_pos=pos, use_cache=True
            )
            pos += 1
        return out


# ---------------------------------------------------------------------------
def build_model(cfg: ModelConfig, device: str = "cpu") -> BlackCode:
    model = BlackCode(cfg).to(device)
    return model


def param_report(cfg: ModelConfig) -> str:
    parts = cfg.param_breakdown()
    total = parts["total"]
    lines = [f"{cfg.name}: {total/1e6:.2f} M paramètres "
             f"({parts['non_embedding']/1e6:.2f} M hors embeddings)"]
    for key in ("embeddings", "attention", "convolution", "norms"):
        lines.append(f"  {key:12s} {parts[key]/1e6:8.2f} M  ({100*parts[key]/total:5.1f} %)")
    return "\n".join(lines)
