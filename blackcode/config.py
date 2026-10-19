"""Configuration des modèles BlackCode.

Un seul fichier décrit l'architecture : profondeur, largeur, type de blocs,
poids ternaires, partage de couches, etc. Les configs concrètes vivent dans
``configs/*.json`` et sont chargées par :func:`ModelConfig.from_json`.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    # --- Identité -------------------------------------------------------
    name: str = "nano"
    description: str = "Modèle de démonstration pour tests CPU"

    # --- Vocabulaire / séquence -----------------------------------------
    vocab_size: int = 8192
    max_seq_len: int = 256

    # --- Largeur ---------------------------------------------------------
    d_model: int = 128
    n_layers: int = 6
    n_heads: int = 4
    n_kv_heads: int = 2
    head_dim: int = 32
    ffn_hidden: int = 352
    conv_kernel: int = 4

    # --- Motif des blocs -------------------------------------------------
    # Un bloc d'attention tous les ``attn_every`` blocs (le dernier du groupe).
    # 4 => 3 convolutions + 1 attention, comme LFM2 / Qwen3-Next.
    attn_every: int = 3

    # --- Astuces de paramètres ------------------------------------------
    tie_embeddings: bool = True
    # Répète la pile de blocs ``loop`` fois SANS ajouter de paramètres
    # (MobileLLM « repeat-all-over » / TRM). Coût : plus de calcul.
    loop: int = 1

    # --- Quantification --------------------------------------------------
    ternary: bool = True
    ternary_embeddings: bool = False

    # --- Divers ----------------------------------------------------------
    norm_eps: float = 1e-5
    rope_base: float = 10_000.0
    # Extension YaRN pour les fenêtres longues. `none` conserve le RoPE historique.
    rope_scaling: str = "none"
    rope_original_max_seq_len: int = 16_384
    rope_factor: float = 1.0
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.d_model != self.n_heads * self.head_dim:
            raise ValueError("d_model doit valoir n_heads * head_dim")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads doit être un multiple de n_kv_heads")
        if self.attn_every < 1:
            raise ValueError("attn_every doit être >= 1")
        if self.loop < 1:
            raise ValueError("loop doit être >= 1")
        if self.vocab_size >= 65_536:
            raise ValueError("vocab_size doit rester < 65536 (stockage uint16)")
        if self.rope_scaling not in {"none", "yarn"}:
            raise ValueError("rope_scaling doit être 'none' ou 'yarn'")
        if self.rope_scaling == "yarn":
            if self.rope_original_max_seq_len < 1 or self.rope_factor <= 1.0:
                raise ValueError("YaRN exige original_max_seq_len > 0 et factor > 1")
            if self.max_seq_len > self.rope_original_max_seq_len * self.rope_factor:
                raise ValueError("max_seq_len dépasse le facteur d'extension YaRN")
            if self.rope_beta_fast <= self.rope_beta_slow or self.rope_beta_slow <= 0:
                raise ValueError("YaRN exige beta_fast > beta_slow > 0")

    # ------------------------------------------------------------------
    def is_attention_layer(self, layer_idx: int) -> bool:
        return (layer_idx + 1) % self.attn_every == 0

    @property
    def n_attention_layers(self) -> int:
        return sum(self.is_attention_layer(i) for i in range(self.n_layers))

    @property
    def n_conv_layers(self) -> int:
        return self.n_layers - self.n_attention_layers

    # ------------------------------------------------------------------
    def param_breakdown(self) -> dict[str, int]:
        """Compte exactement les paramètres, sans instancier le modèle."""
        d, ffn, v = self.d_model, self.ffn_hidden, self.vocab_size
        kv_dim = self.n_kv_heads * self.head_dim

        attn = (
            d * d                      # q_proj
            + 2 * d * kv_dim           # k_proj + v_proj
            + d * d                    # o_proj
            + 3 * d * ffn              # SwiGLU
            + 2 * self.head_dim        # q_norm + k_norm
        )
        conv = (
            2 * d * d                  # in_proj (x et gate)
            + d * self.conv_kernel     # depthwise conv
            + d * d                    # out_proj
            + 3 * d * ffn              # SwiGLU
        )
        norms_per_block = 2 * d        # norm + ffn_norm
        embeddings = v * d

        n_attn = self.n_attention_layers
        n_conv = self.n_conv_layers
        parts = {
            "embeddings": embeddings if self.tie_embeddings else 2 * embeddings,
            "attention": n_attn * attn,
            "convolution": n_conv * conv,
            "norms": self.n_layers * norms_per_block + d,
        }
        parts["total"] = sum(parts.values())
        parts["non_embedding"] = parts["total"] - parts["embeddings"]
        return parts

    @property
    def num_params(self) -> int:
        return self.param_breakdown()["total"]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))
