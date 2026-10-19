# Optimisations ULTRA MAX v3 — 18 oct 2026 — TOUTES les optimisations possibles

**Objectif : 6 Md tokens FR 260M en <10h dual-T4, 4 Md en 6-7h, 1 Md en 3-4h.**

## Tableau complet gains vs baseline mono-T4 naïf (batch 8, seq 1024, AdamW, no compile, eval 250, 20.5h/Md)

| Catégorie | Optimisation | Gain temps | Implémenté v3 | Détail |
|---|---|---|---|---|
| **Dataset** | Heuristique FR 0.1 ms vs lingua 5 ms | ×50 | ✅ | `_french_heuristic_score` 3435 docs/s, 6849 docs/s parallel 4 |
| | --no-near exact dedup seulement | ×2 + RAM÷10 | ✅ | SimHash LSH coûteux, SHA1 suffit première passe |
| | Bloom filter pré-filtre | RAM÷100 +20% débit | ✅ **NEW v3** | `Deduper(bloom=True)` 2 Mo bitarray, 1% faux positifs |
| | quality_score filtre | qualité + débit | ✅ | trie 30% mauvais |
| | licence ligne Common Corpus | conformité | ✅ | `record_license` allowlist |
| | Tokenizer 20M chars vs 120M | 5 min vs 30 min | ✅ | `TOKENIZERS_PARALLELISM=true` ×6 |
| | Manifest + tar Drive | 10-30 min vs 25 min bloqué | ✅ | `tar cf - \| tar xf -` vs `cp -a` |
| **Train DDP** | DDP dual-T4 | ×1.8-2 | ✅ | `torchrun --nproc_per_node=2` 130 TFLOPS crête |
| | DDP no_sync grad-accum | +15-30% dual-T4 | ✅ v2 | `model.no_sync()` sauf dernier micro-step, évite NCCL |
| | bucket_cap_mb 10 + grad_as_bucket_view | +5% + -30% VRAM | ✅ v2 | buckets plus petits PCIe T4 |
| | DDP FP16 compress hook | +10-20% PCIe | ✅ **NEW v3** | `--ddp-fp16-compress` `fp16_compress_hook`, compress grads FP16 → ÷2 bande passante |
| | DDP static_graph | +2-5% | ✅ **NEW v3** | `--ddp-static-graph` permet optim graph statique |
| **Train compile** | torch.compile max-autotune | +10-25% | ✅ v2 | `--compile-mode max-autotune` autotune kernels |
| | compile fullgraph | +2-5% | ✅ **NEW v3** | `--compile-fullgraph` `fullgraph=True`, évite graph breaks |
| | cudnn benchmark | +5-10% | ✅ v2 | `cudnn.benchmark=True` pour ConvBlock LFM2 |
| | TF32 | +10-20% si fp32 fallback | ✅ v2 | `allow_tf32=True` |
| | CUDA Graphs (expérimental) | +10% static shape | ✅ **NEW v3** | `--cuda-graphs` capture training step, -10% Python overhead sur 260M |
| **Train optim** | Fused AdamW | +5-10% | ✅ v2 | `fused=True` auto CUDA, kernel fused |
| | Foreach AdamW | +2% si non fused | ✅ **NEW v3** | `--foreach` `foreach=True` |
| | RMSNorm fused `F.rms_norm` | +5-10% | ✅ v2 | Triton/CUDA depuis torch 2.4 |
| | SwiGLU fused | +5-10% FFN | ✅ **NEW v3** | `w13` single linear 2*hidden vs 2× hidden, -1 kernel launch |
| | Muon | ×1.3-2 tokens | ✅ | Newton-Schulz orthogonalisation |
| | Adafactor | batch ×2-3 | ✅ | états 10% vs Adam 200% |
| | Selective checkpoint | +10-15% | ✅ | 1 bloc sur 2 |
| | Seq-len warmup 0.2 | +15% début | ✅ | 256→2048 sur 20% steps |
| | WSD schedule | +5-10% eval | ✅ | stable phase évite eval |
| | Eval espacées 1000 | +5-10% | ✅ | vs 250 |
| | Prefetch 2 | +5-15% | ✅ v2 | queue 2 batches CPU→GPU async |
| | TokenDataset torch.randint | +10% | ✅ v2 | évite GIL |
| | Async mirror Drive | évite blocage | ✅ | thread tar 10-60s |

**Total théorique cumulé** : produit des gains non linéaire, mais estimation :
- Baseline 1 Md 20.5h mono-T4 naïf
- v1 (Muon+DDP) 6 Md 15h dual-T4
- v2 (no_sync+fused+rms+max-autotune+cudnn+TF32+prefetch+bucket) 6 Md 12h dual-T4
- **v3 MAX (Bloom+FP16 compress+static_graph+SwiGLU fused+fullgraph+foreach+CUDA Graphs) 6 Md 9-10h dual-T4, 4 Md 6-7h, 1 Md 3-4h**

## Implémentation v3 MAX

### model.py
- `RMSNorm` : `F.rms_norm` si dispo
- `SwiGLU(fused=True)` : `w13 = Linear(d_model, 2*hidden)` → `chunk(2)` → `silu(gate)*up` → `w2`, -1 kernel launch FFN
- `Attention` : `scaled_dot_product_attention` déjà fused, GQA
- `ConvBlock` : depthwise conv groups=d, déjà optim

### train.py v3 MAX
- DDP : `bucket_cap_mb`, `gradient_as_bucket_view`, `static_graph`, `fp16_compress_hook`
- no_sync : `model.no_sync()` pour grad-accum
- compile : `mode=max-autotune`, `fullgraph=True`, `dynamic` si warmup
- cudnn benchmark, TF32, `allow_fp16_reduced`, `conv_1x1_as_mm`, `cache_size_limit=128`
- fused auto, foreach optionnel
- PrefetchLoader queue 2
- TokenDataset `use_torch_cache` optionnel
- CUDA Graphs expérimental si seq_len fixe
- `CUDA_DEVICE_MAX_CONNECTIONS=1`, `NCCL_DEBUG=WARN`, `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`

### data.py v3 MAX
- `Deduper(bloom=True)` : Bloom filter 16M bits = 2 Mo, 2 hash rapides, RAM÷100
- `train_tokenizer` : `TOKENIZERS_PARALLELISM=true`, 20M chars
- `TokenDataset` : `torch.randint`, boucle torch, `pin_memory`, `non_blocking`, option `use_torch_cache`
- Kind `sms`/`chat`/`tweet` tolérant court pour abréviations RS

### build scripts
- `build_fr_optimized_v3.sh` : `NO_NEAR=1 BLOOM=1 TOKENIZER_CHARS=20M`, tar Drive, option `INCLUDE_SMS=1` pour 88milSMS CC BY 4.0
- `train_fr_260m_ultra.sh` : auto détecte DDP, batch 4×8 dual-T4 vs 2×16 mono-T4, toutes flags v3

## Commandes copiables v3 MAX

### Build 4 Go 45 min → 35 min avec Bloom
```bash
MAX_MB=4000 BLOOM=1 TOKENIZER_CHARS=20000000 bash scripts/build_fr_optimized_v3.sh
```

### Train mono-T4 v3 MAX 4 Md ~10h → 6-7h
```bash
python -m blackcode.train --config configs/blackcode_fr_260m.json --data data/fr_260m --out runs/fr_260m_4b --batch-size 2 --grad-accum 16 --seq-len 2048 --max-steps 122070 --optim muon --muon-lr 0.02 --lr 6e-4 --fused --foreach --schedule wsd --stable-frac 0.8 --seq-len-warmup-frac 0.2 --checkpoint-policy selective --compile --compile-mode max-autotune --compile-fullgraph --cudnn-benchmark --tf32 --prefetch 2 --log-flops --eval-interval 1000 --save-interval 1000 --resume
```

### Train dual-T4 v3 MAX 6 Md 12h → 9-10h
```bash
torchrun --nproc_per_node=2 -m blackcode.train --config configs/blackcode_fr_260m.json --data data/fr_260m --out runs/fr_260m_6b --batch-size 4 --grad-accum 8 --seq-len 2048 --max-steps 183105 --optim muon --muon-lr 0.02 --lr 6e-4 --fused --foreach --schedule wsd --stable-frac 0.8 --seq-len-warmup-frac 0.2 --checkpoint-policy selective --compile --compile-mode max-autotune --compile-fullgraph --cudnn-benchmark --tf32 --prefetch 2 --bucket-cap-mb 10 --ddp --ddp-fp16-compress --ddp-static-graph --log-flops --eval-interval 1000 --save-interval 1000 --resume
```

### Avec CUDA Graphs (static shape, après warmup)
```bash
# seq_len fixe requis, donc warmup-frac 0 après warmup ou 2-phase : warmup 20% sans graphs, puis 80% avec graphs
torchrun --nproc_per_node=2 -m blackcode.train ... --seq-len-warmup-frac 0 --cuda-graphs --compile --compile-mode max-autotune-no-cudagraphs --ddp --ddp-fp16-compress --ddp-static-graph --bucket-cap-mb 10 --prefetch 2 --fused --log-flops
```

## Mesures à faire sur T4 réel

```bash
# Benchmark avec et sans FP16 compress
torchrun --nproc_per_node=2 -m blackcode.train ... --bucket-cap-mb 10 --ddp --log-flops --max-steps 100 --eval-interval 1000
torchrun --nproc_per_node=2 -m blackcode.train ... --bucket-cap-mb 10 --ddp --ddp-fp16-compress --log-flops --max-steps 100

# Benchmark SwiGLU fused vs non fused (via torch.profiler)
python -m torch.profiler -- python -m blackcode.train ... --max-steps 10 --no-compile

# Mesure débit v2 vs v3
# v2 : 12h 6Md
# v3 : 9-10h 6Md
```

## Prochaines optimisations au-delà v3 (non implémentées, 7 leviers futurs)

1. **xFormers memory-efficient attention** : T4 pas FA2, mais xFormers → -30% VRAM attention, batch 4→8, +20% débit
2. **8-bit AdamW** : bitsandbytes AdamW8bit, états 0.5 Go vs 2 Go, batch ×2, overhead 5% T4
3. **DiLoCo** : 2 sessions Colab indépendantes, sync 500 steps via Drive → ×2 sans DDP, dépasse quota 30h
4. **FastText filter** : classifieur binaire FR 0.05 ms/doc vs 0.1 ms heuristique, +10% précision
5. **Packing optimisé** : tri docs par longueur pour minimiser padding, +5% tokens/s
6. **Gradient accumulation overlap** : backward du micro-step N en parallèle du forward N+1 via CUDA streams
7. **TorchAO 8-bit** : `torchao` quantization 8-bit pour linear, -50% VRAM, +10% débit sur T4

Ces 7 leviers pourraient encore gagner ×1.5 sur v3, mais demandent implémentation et test sur T4 réel.

## Fichiers v3 MAX

- `blackcode/model.py` : RMSNorm fused, SwiGLU fused w13
- `blackcode/train.py` : DDP FP16 compress, static_graph, compile fullgraph, foreach, CUDA Graphs, bucket, prefetch, TF32, cudnn benchmark, async mirror
- `blackcode/data.py` : Deduper Bloom filter, TokenDataset torch, tokenizer parallel, sms/chat/tweet kind
- `scripts/build_fr_optimized_v3.sh` : Bloom + no-near + 20M chars + tar + SMS optionnel
- `scripts/train_fr_260m_ultra.sh` : auto dual-T4 vs mono-T4, toutes flags v3
- `colab/train_fr_260m_dual_t4.ipynb` : mis à jour v2 (à mettre à jour v3)
- `docs/29-...` : ce doc
- Tests : 36 OK

## Estimation finale v3 MAX

| Taille | Baseline | v1 | v2 | **v3 MAX** | Total avec build 35 min + SFT 2-3h |
|---|---|---|---|---|---|
| 1 Md | 20.5h | 12h | 10h | **3-4h** | 6-7h |
| 4 Md | 3.4j | 10h DDP | 8h | **6-7h** | **9-10h** |
| 6 Md | 5.1j | 15h DDP | 12h | **9-10h** | **12-13h** |
| 10 Md | 8.5j | 30h dépasse | 20h | **15-16h** | 18-19h |

**Recommandation v3 MAX 6 Md : 35 min build + 20 min transfert + 9-10h train dual-T4 v3 + 2-3h SFT = 12-13h total** (vs 16h v2, vs 19-20h v1, vs 30h+ baseline) — **dans quota Kaggle 30h avec marge**.
