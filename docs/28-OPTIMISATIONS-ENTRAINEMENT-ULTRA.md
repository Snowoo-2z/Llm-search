# Optimisations ultra pour entraînement FR 260M — 18 oct 2026

**Objectif : gagner du temps sur entraînement, cible dual-T4 Kaggle 6 Md tokens en <20h.**

## Résumé gains

| Optimisation | Gain temps | Implémenté | Comment |
|---|---|---|---|
| **Heuristique français** (0.1 ms vs lingua 5 ms) | ×50 dataset | ✅ `data.py` `_french_heuristic_score` | 3435 docs/s mesuré, 6849 docs/s parallel 4 workers |
| **--no-near** (exact dedup seulement) | ×2 + RAM÷10 | ✅ `build_fr_optimized_v2.sh` | SimHash LSH coûteux, exact SHA1 suffit première passe |
| **quality_score** filtre | qualité + débit | ✅ `quality_score` + `--min-quality-score` | trie et coupe 30% mauvais docs |
| **licence ligne** Common Corpus | conformité | ✅ `record_license` | filtre allowlist |
| **Tokenizer 20M chars** vs 120M | 5 min vs 30 min | ✅ `TOKENIZER_CHARS=20M` + `TOKENIZERS_PARALLELISM=true` | ×6 pour itération |
| **Manifest + disque tar** | 10-30 min vs 25 min bloqué cp -a | ✅ `build_fr_optimized_v2.sh` tar | `cp -a` vers Drive bloque |
| **DDP dual-T4** | ×1.8-2 | ✅ `train.py` `_init_ddp()` + `torchrun --nproc_per_node=2` | 2× T4 = 130 TFLOPS crête |
| **DDP no_sync grad-accum** | +15-30% dual-T4 | ✅ **NOUVEAU** `model.no_sync()` pour micro-steps sauf dernier | évite all-reduce NCCL à chaque micro-step |
| **Fused AdamW** | +5-10% | ✅ **NOUVEAU** `fused=True` auto sur CUDA + `--fused`/`--no-fused` | kernel fused CUDA |
| **RMSNorm fused** `F.rms_norm` | +5-10% | ✅ **NOUVEAU** `model.py` utilise `F.rms_norm` si dispo (torch 2.4+) | Triton/CUDA kernel |
| **torch.compile max-autotune** | +10-25% | ✅ **NOUVEAU** `--compile-mode max-autotune` | autotune après warmup |
| **cudnn benchmark** | +5-10% | ✅ **NOUVEAU** `--cudnn-benchmark` défaut True | benchmark conv kernels |
| **TF32** | +10-20% si fallback fp32 | ✅ **NOUVEAU** `--tf32` défaut True | `allow_tf32=True` |
| **Prefetch 2** | +5-15% | ✅ **NOUVEAU** `PrefetchLoader` queue 2 batches | CPU→GPU async |
| **bucket_cap_mb 10** | +5% NCCL T4 PCIe | ✅ **NOUVEAU** `--bucket-cap-mb 10` + `gradient_as_bucket_view` | -30% VRAM + NCCL |
| **TokenDataset torch cache** | +10% si RAM | ✅ `data.py` `torch.randint` + boucle torch, évite GIL | `np.random` → `torch.randint` |
| **Adafactor** | batch ×2-3 | ✅ `optim.py` | états 10% vs Adam 200%, permet batch plus grand |
| **Selective checkpoint** | +10-15% | ✅ `--checkpoint-policy selective` | 1 bloc sur 2 checkpointé |
| **Seq-len warmup 0.2** | +15% tokens/s début | ✅ `--seq-len-warmup-frac 0.2` | 256→2048 sur 20% steps |
| **WSD schedule** | +5-10% eval | ✅ `--schedule wsd` | stable phase évite eval inutile |
| **Eval espacées 1000** | +5-10% | ✅ `--eval-interval 1000` vs 250 | moins d'eval |
| **Muon** | ×1.3-2 tokens | ✅ `--optim muon` | orthogonalisation Newton-Schulz |
| **Async mirror** | évite blocage Drive | ✅ `mirror-dir` thread | tar 10-60s async |

**Total cumulé vs baseline mono-T4 naïf (batch 8, seq 1024, AdamW, no compile, eval 250) :**
- Baseline 1 Md : 20.5h
- + Adafactor + selective + compile : 2.2j → 1.1j pour 4 Md (pruning ×2) → 18h avec Muon ×1.5
- + DDP dual-T4 : ~10h pour 4 Md
- + DDP no_sync + fused + rms_norm fused + max-autotune + cudnn benchmark + TF32 + prefetch + bucket 10 : **~8h pour 4 Md, ~12h pour 6 Md** (estimation, à mesurer)

## Détails implémentation 18 oct 2026 v2

### 1. DDP no_sync (gain majeur dual-T4)

Avant :
```python
for _ in range(grad_accum):
    x,y = dataset.get_batch(...)
    loss = model.forward_loss(...)
    scaler.scale(loss).backward()  # all-reduce à chaque fois !
```

Après :
```python
for micro in range(grad_accum):
    is_last = micro == grad_accum-1
    no_sync_ctx = model.no_sync() if is_ddp and not is_last and not no_ddp_no_sync else nullcontext()
    with no_sync_ctx:
        x,y = prefetch_loader.next(seq_len) if prefetch else dataset.get_batch(...)
        loss = model.forward_loss(...)
        scaler.scale(loss).backward()  # pas de sync sauf dernier
```

→ Évite NCCL all-reduce sur `grad_accum-1` micro-steps. Sur 2× T4 PCIe, NCCL est coûteux, gain 15-30% mesuré dans littérature.

### 2. Fused AdamW + RMSNorm fused

- `torch.optim.AdamW(..., fused=True)` : kernel CUDA fused pour Adam, +5-10%
- `F.rms_norm(x, (dim,), weight, eps)` : kernel Triton depuis torch 2.4, remplace `weight * x * rsqrt(mean(x^2)+eps)` manuel, +5-10%
- Auto activé sur CUDA sauf `--no-fused`

### 3. torch.compile max-autotune

```bash
--compile --compile-mode max-autotune --cudnn-benchmark --tf32
```

- `max-autotune` : autotune kernels après warmup, +10-25% vs `default`
- `cudnn.benchmark=True` : benchmark conv kernels (ConvBlock LFM2), +5-10%
- `allow_tf32=True` : TF32 pour matmul fp32 fallback, +10-20%
- `compile_dynamic` : si seq-len warmup, dynamic=True pour éviter recompilation à chaque seq_len

### 4. PrefetchLoader

```python
class PrefetchLoader:
    queue = deque()
    fill(seq_len) -> précharge N+1 batches
    next(seq_len) -> pop + refill
```

→ Cache CPU→GPU en parallèle du forward, +5-15% si dataset memmap lent (Colab disque).

### 5. DDP bucket tuning

```python
DDP(model, bucket_cap_mb=10, gradient_as_bucket_view=True)
```

- `bucket_cap_mb=10` pour T4 PCIe (vs 25 défaut) : plus petits buckets = NCCL plus fréquent mais moins de latence PCIe
- `gradient_as_bucket_view=True` : -30% VRAM (grads = vues buckets, pas copies) + +5% débit

### 6. TokenDataset optimisé

- `torch.randint` au lieu de `np.random.randint` : évite GIL, +10%
- Boucle `for b,i in enumerate(ix): x[b]=data[i:i+seq]` au lieu de `np.stack([...])` : évite copie intermédiaire
- Option `use_torch_cache` : charge tout en RAM pin_memory si RAM suffisante (désactivé par défaut pour T4 16 Go)

### 7. Tokenizer parallel

```bash
export TOKENIZERS_PARALLELISM=true
--tokenizer-chars 20000000  # 20M vs 120M défaut
```

→ 5 min vs 30 min pour itération, ×6.

## Commandes copiables — ultra optimisées

### Build dataset 4 Go (1h30 → 45 min avec v2)

```bash
# v2 ultra optimisé : 20M chars tokenizer, no-near, heuristique, parallel
MAX_MB=4000 MIN_QUALITY_SCORE=0.6 VOCAB=24576 SEQ_LEN=2048 TOKENIZER_CHARS=20000000 bash scripts/build_fr_optimized_v2.sh

# Avec SMS abréviations (88milSMS CC BY 4.0)
INCLUDE_SMS=1 MAX_MB=4000 bash scripts/build_fr_optimized_v2.sh
```

### Entraînement mono-T4 ultra optimisé (4 Md → ~16h → ~12h)

```bash
python -m blackcode.train --config configs/blackcode_fr_260m.json --data data/fr_260m --out runs/fr_260m_4b --batch-size 2 --grad-accum 16 --seq-len 2048 --max-steps 122070 --optim muon --muon-lr 0.02 --lr 6e-4 --fused --schedule wsd --stable-frac 0.8 --seq-len-warmup-frac 0.2 --checkpoint-policy selective --compile --compile-mode max-autotune --cudnn-benchmark --tf32 --prefetch 2 --log-flops --eval-interval 1000 --save-interval 1000 --mirror-dir /content/drive/MyDrive/blackcode/runs --resume
```

### Entraînement dual-T4 Kaggle ultra optimisé (6 Md → ~19h → ~12h estimé)

```bash
torchrun --nproc_per_node=2 -m blackcode.train --config configs/blackcode_fr_260m.json --data data/fr_260m --out runs/fr_260m_6b --batch-size 2 --grad-accum 8 --seq-len 2048 --max-steps 183105 --optim muon --muon-lr 0.02 --lr 6e-4 --fused --schedule wsd --stable-frac 0.8 --seq-len-warmup-frac 0.2 --checkpoint-policy selective --compile --compile-mode max-autotune --cudnn-benchmark --tf32 --prefetch 2 --bucket-cap-mb 10 --ddp --log-flops --eval-interval 1000 --save-interval 1000 --mirror-dir /kaggle/working/drive/runs --resume
```

### Benchmark qualité (déjà mesuré)

```bash
python scripts/benchmark_quality.py
# quality_reasons 3435 docs/s (10000 docs 2.91s)
# french_heuristic 1.93s
# parallel 4 workers 1.46s → ×2
```

## Prochaines optimisations (non implémentées, 7 leviers futurs)

1. **FastText filter** : classifieur binaire FR vs non-FR entraîné sur 1k docs, 0.05 ms/doc vs 0.1 ms heuristique, +10% précision.
2. **Bloom filter dedup** : pré-filtre exact avant SHA1, RAM÷100, +20% débit build.
3. **xFormers memory-efficient attention** : T4 pas FA2, mais xFormers `memory_efficient_attention` → -30% VRAM attention, permet batch 4 vs 2.
4. **8-bit AdamW** : `bitsandbytes` AdamW8bit, états 0.5 Go vs 2 Go, batch ×2, mais overhead 5% sur T4.
5. **DiLoCo** : 2 sessions Colab indépendantes, sync toutes les 500 steps via Drive → ×2 sans DDP, pour dépasser quota 30h.
6. **CUDA Graphs** : `torch.cuda.graph` pour training step statique (batch, seq_len fixes après warmup), -10% Python overhead sur petit modèle 260M.
7. **Fused SwiGLU** : kernel Triton pour `silu(w1(x)) * w3(x)` + `w2`, -5% temps FFN.

Ces 7 leviers pourraient encore gagner ×1.5-2 sur total, mais demandent implémentation et test.

## Estimation temps finale avec ultra optimisations

| Taille | Baseline mono-T4 naïf | Mono-T4 opti v1 (Muon+DDP) | Mono-T4 opti v2 (no_sync+fused+rms+max-autotune+cudnn+prefetch) | Dual-T4 v2 |
|---|---|---|---|---|
| 1 Md | 20.5h | 12h | 10h | 5.5h |
| 4 Md (Chinchilla 5.16 Md) | 3.4j | 1.1j (18h Muon) → 10h DDP | 8h | **~8h** |
| 6 Md (recommandé good+) | 5.1j | 1.6j → 15h DDP | 12h | **~12h** |
| 10 Md | 8.5j | 2.7j → 30h dépasse quota | 20h | **~20h** |

**Recommandation 6 Md : 1h30 build v2 + 20 min transfert + 12h train dual-T4 v2 + 2-3h SFT = ~16h total** (vs 19-20h v1, vs 30h+ baseline).

## Fichiers modifiés 18 oct 2026

- `blackcode/train.py` : DDP no_sync, fused auto, compile-mode, cudnn benchmark, TF32, prefetch, bucket_cap_mb, RMSNorm fused, async mirror, WSD, seq-len warmup
- `blackcode/model.py` : RMSNorm fused `F.rms_norm` si dispo
- `blackcode/data.py` : TokenDataset torch.randint, prefetch, tokenizer parallel
- `scripts/build_fr_optimized_v2.sh` : 20M chars tokenizer, no-near, heuristique, tar, INCLUDE_SMS
- `colab/train_fr_260m_dual_t4.ipynb` : à mettre à jour avec v2 args
- `docs/28-...` : ce doc
- Tests : 36 OK

## Mesures à faire sur T4 réel

```bash
# Mesure débit avec et sans no_sync
torchrun --nproc_per_node=2 -m blackcode.train ... --bucket-cap-mb 10 --prefetch 2 --log-flops --max-steps 100 --eval-interval 1000

# Compare compile modes
--compile --compile-mode default vs max-autotune vs reduce-overhead

# Mesure RMSNorm fused vs manual (via torch.profiler)
python -m torch.profiler -- python -m blackcode.train ... --max-steps 10
```

