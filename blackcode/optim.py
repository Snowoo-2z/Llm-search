"""Optimiseurs et schedules pour réduire le temps d'entraînement **à taille fixe**.

Le temps d'entraînement vaut :

    temps ≈ (tokens nécessaires) × (temps par token)

Ce module agit sur les deux facteurs :

1. **Tokens nécessaires** — ``Muon`` et ``AdEMAMix`` améliorent l'efficacité
   par token ; ``wsd_multiplier`` permet de brancher/étendre un entraînement
   sans refaire la décroissance.
2. **Temps par token** — les mêmes optimiseurs coûtent peu ou pas de calcul
   supplémentaire ; ``make_optimizer`` choisit aussi des variantes économes
   en mémoire (8-bit, Adafactor) qui autorisent un plus gros batch.

Toutes les classes sont des ``torch.optim.Optimizer`` standard, donc
compatibles avec ``GradScaler``, le gradient clipping et ``state_dict``.
"""
from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------
def cosine_multiplier(step: int, warmup: int, max_steps: int,
                      min_ratio: float = 0.1) -> float:
    """Warmup linéaire puis décroissance cosinus. Retourne un multiplicateur."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    if step >= max_steps:
        return min_ratio
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))


def wsd_multiplier(step: int, warmup: int, max_steps: int,
                   stable_frac: float = 0.8, min_ratio: float = 0.02,
                   decay: str = "sqrt") -> float:
    """Warmup–Stable–Decay (MiniCPM, arXiv:2404.06395).

    - phase *stable* : LR constant. On peut y sauvegarder, brancher, comparer,
      ou prolonger l'entraînement sans refaire la décroissance ;
    - phase *decay* : courte, resserre la LR jusqu'à ``min_ratio``.

    Avantage pratique : un entraînement interrompu à 60 % n'a rien perdu, il
    suffit de faire la décroissance à partir de là.
    """
    if step < warmup:
        return (step + 1) / max(1, warmup)
    decay_start = int(max_steps * stable_frac)
    if step < decay_start:
        return 1.0
    progress = (step - decay_start) / max(1, max_steps - decay_start)
    progress = min(max(progress, 0.0), 1.0)
    if decay == "sqrt":
        shape = 1.0 - math.sqrt(progress)
    elif decay == "linear":
        shape = 1.0 - progress
    elif decay == "cosine":
        shape = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        raise ValueError(f"décroissance inconnue : {decay}")
    return min_ratio + (1.0 - min_ratio) * shape


def schedule_multiplier(step: int, name: str, warmup: int, max_steps: int,
                        min_ratio: float = 0.1, stable_frac: float = 0.8,
                        decay: str = "sqrt") -> float:
    if name == "cosine":
        return cosine_multiplier(step, warmup, max_steps, min_ratio)
    if name == "wsd":
        return wsd_multiplier(step, warmup, max_steps, stable_frac, min_ratio, decay)
    if name == "constant":
        return (step + 1) / max(1, warmup) if step < warmup else 1.0
    raise ValueError(f"schedule inconnu : {name}")


# ---------------------------------------------------------------------------
# Muon
# ---------------------------------------------------------------------------
def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5,
                                eps: float = 1e-7) -> torch.Tensor:
    """Orthogonalise approximativement ``g`` par itération de Newton–Schulz.

    Les coefficients sont ceux de l'implémentation de référence de Muon : le
    spectre de sortie est borné (typiquement 0,68–1,13) plutôt qu'exactement
    unitaire. C'est suffisant pour l'usage en descente de gradient, et c'est
    ce qui a été validé à grande échelle.

    Coût : ~``steps`` produits matriciels de la taille de ``g``. En pratique
    l'overhead est de 10–30 % sur un petit modèle, contre 1,3 à 2× de tokens
    économisés (Muon is Scalable, arXiv:2502.16982).
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float()
    x = x / (x.norm() + eps)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    """Momentum + orthogonalisation du gradient (Keller Jordan, 2024).

    À utiliser sur les matrices 2D cachées (FFN, projections). Les embeddings,
    têtes de sortie, normes et convolutions doivent rester sous AdamW.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5,
                 weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    raise ValueError("Muon ne gère que les tenseurs 2D")
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                update = zeropower_via_newtonschulz5(update, group["ns_steps"])
                # Normalisation d'échelle : garde une mise à jour RMS comparable
                # quelle que soit la forme de la matrice.
                update = update * (0.2 * max(1.0, p.size(0) / p.size(1)) ** 0.5)
                if group["weight_decay"]:
                    p.mul_(1.0 - lr * group["weight_decay"])
                p.add_(update, alpha=-lr)
        return loss


# ---------------------------------------------------------------------------
# AdEMAMix
# ---------------------------------------------------------------------------
class AdEMAMix(torch.optim.Optimizer):
    """AdamW + second EMA très lent (arXiv:2409.03137).

    Deux moyennes mobiles du gradient (rapide et lente) sont combinées :
    la lente retient le signal de long terme, ce qui améliore la qualité à
    budget de tokens fixé, surtout sur les longs entraînements. Aucun coût
    de calcul supplémentaire, ~1 état de plus par paramètre.
    """

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 beta3: float = 0.9999, alpha: float = 5.0, alpha_warmup: int = 0,
                 eps: float = 1e-8, weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, beta3=beta3, alpha=alpha,
                        alpha_warmup=alpha_warmup, eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        self._step_count += 1
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            beta3 = group["beta3"]
            alpha_max = group["alpha"]
            warm = group["alpha_warmup"]
            alpha = alpha_max * min(1.0, self._step_count / warm) if warm > 0 else alpha_max
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m1"] = torch.zeros_like(p)
                    state["m2"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                m1, m2, v = state["m1"], state["m2"], state["v"]
                m1.mul_(beta1).add_(g, alpha=1 - beta1)
                m2.mul_(beta3).add_(g, alpha=1 - beta3)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                denom = (v / bias2).sqrt_().add_(group["eps"])
                update = (m1 / bias1 + alpha * m2) / denom
                if group["weight_decay"]:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])
        return loss


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def _is_hidden_matrix(name: str, p: torch.Tensor) -> bool:
    return (p.ndim == 2
            and "emb" not in name
            and p.shape[0] > 1 and p.shape[1] > 1
            and not name.endswith("conv.weight"))


def make_optimizer(model, name: str = "adamw", lr: float = 6e-4,
                   muon_lr: float = 0.02, weight_decay: float = 0.1,
                   betas=(0.9, 0.95), fused: bool = False,
                   alpha_warmup: int = 0,
                   adafactor_lr: float | None = None):
    """Retourne ``(optimizer, base_lrs)``.

    ``base_lrs`` est la liste des LR de référence par groupe ; le scheduler
    applique ensuite un multiplicateur commun, ce qui permet de mélanger
    Muon (LR ~0,02) et AdamW (LR ~6e-4) dans le même entraînement.
    """
    decay, no_decay, muon = [], [], []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name == "muon" and _is_hidden_matrix(pname, p):
            muon.append(p)
        elif p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)

    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay,
                       "base_lr": lr, "tag": "adamw_decay"})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0,
                       "base_lr": lr, "tag": "adamw_nodecay"})

    if name == "muon":
        opt_muon = Muon([{"params": muon, "weight_decay": weight_decay,
                          "base_lr": muon_lr}],
                        lr=muon_lr, momentum=0.95, nesterov=True,
                        weight_decay=weight_decay)
        if groups:
            opt_rest = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8,
                                         fused=fused and torch.cuda.is_available())
            opt = _CombinedOptimizer([opt_muon, opt_rest])
        else:
            opt = opt_muon
    elif name == "adafactor":
        # La mise à jour d'Adafactor est normalisée par sqrt(v) : à LR égale
        # elle est ~30× plus petite que celle d'AdamW. On compense par défaut.
        ada_lr = adafactor_lr if adafactor_lr is not None else lr * 30.0
        for group in groups:
            group["base_lr"] = ada_lr
        opt = torch.optim.Adafactor(groups, lr=ada_lr, weight_decay=weight_decay,
                                    beta2_decay=-0.8)
        print(f"[optim] Adafactor : lr={ada_lr:.2e} (×30 par rapport à {lr:.2e})")
    elif name == "ademamix":
        opt = AdEMAMix(groups, lr=lr, betas=betas, alpha=5.0,
                       alpha_warmup=alpha_warmup, weight_decay=weight_decay)
    elif name in ("adamw8bit", "paged_adamw8bit"):
        try:
            import bitsandbytes as bnb  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "bitsandbytes requis : pip install bitsandbytes"
            ) from exc
        cls = (bnb.optim.PagedAdamW8bit if name == "paged_adamw8bit"
               else bnb.optim.AdamW8bit)
        opt = cls(groups, lr=lr, betas=betas, eps=1e-8,
                  weight_decay=weight_decay)
    elif name == "adamw":
        opt = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8,
                                fused=fused and torch.cuda.is_available())
    else:
        raise ValueError(f"optimiseur inconnu : {name}")

    base_lrs = [g["base_lr"] for g in opt.param_groups]
    return opt, base_lrs


class _CombinedOptimizer:
    """Expose plusieurs optimiseurs comme un seul (Muon + AdamW).

    Ce n'est volontairement pas une sous-classe de ``torch.optim.Optimizer`` :
    GradScaler n'exige que ``param_groups`` et ``step()``, et l'héritage
    entrerait en conflit avec la propriété ``param_groups``.
    """

    def __init__(self, optimizers):
        self.optimizers = optimizers
        self.defaults = {}
        self.state = {}

    @property
    def param_groups(self):
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        losses = [opt.step(closure) for opt in self.optimizers]
        return losses[0] if losses else None

    def state_dict(self):
        return {"sub": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict):
        for opt, sub in zip(self.optimizers, state_dict["sub"]):
            opt.load_state_dict(sub)

    def __repr__(self):
        return f"CombinedOptimizer({self.optimizers})"


def training_memory_mb(params: int, optimizer: str, activation_mb: float = 1500.0,
                       param_bytes: int = 4, grad_bytes: int = 4) -> dict:
    """Mémoire d'entraînement complète : poids + gradients + états + activations.

    Hypothèse : poids et gradients en fp32 (comportement PyTorch par défaut,
    l'AMP ne change que le calcul). ``activation_mb`` doit être mesuré ; 1,5 Go
    est une valeur prudente avec checkpointing sur un T4.
    """
    weights = params * param_bytes / (1024 ** 2)
    grads = params * grad_bytes / (1024 ** 2)
    opt = estimate_optimizer_memory_mb(params, optimizer)["optimizer_state_mb"]
    total = weights + grads + opt + activation_mb
    return {
        "weights_mb": round(weights, 1),
        "grads_mb": round(grads, 1),
        "optimizer_mb": round(opt, 1),
        "activations_mb": round(activation_mb, 1),
        "total_mb": round(total, 1),
        "fits_16gb": total < 15_000,
    }


def estimate_optimizer_memory_mb(params: int, name: str,
                                 master_weights: bool = True) -> dict:
    """Empreinte mémoire approximative des états d'optimiseur."""
    fp32 = 4
    weights = params * fp32 / (1024 ** 2) if master_weights else 0.0
    if name in ("adamw", "ademamix", "adamw8bit", "paged_adamw8bit"):
        scale = 8 / 32 if name.endswith("8bit") else 1.0
        moments = 2 * params * fp32 * scale / (1024 ** 2)
        if name == "ademamix":
            moments += params * fp32 / (1024 ** 2)
    elif name == "adafactor":
        moments = params * fp32 * 2 * 0.1 / (1024 ** 2)  # facteurs ≈ 10 %
    elif name == "muon":
        moments = params * fp32 / (1024 ** 2) * 0.75  # momentum seul sur ~75 %
    else:
        raise ValueError(name)
    return {"weights_master_mb": round(weights, 1),
            "optimizer_state_mb": round(moments, 1),
            "total_mb": round(weights + moments, 1)}
