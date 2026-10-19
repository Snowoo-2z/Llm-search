# TPU v5e-8 vs T4x2 sur Kaggle pour Luciole-1B-Instruct-1.1

**Date : 18 oct 2026. Recherche web_search Kaggle TPU v5e.**

## Question utilisateur

> On passe sur TPU e5v ? Kaggle propose 20h et il est beaucoup plus rapide .. tu me dis avant de faire

## Faits vérifiés sur Kaggle TPU v5e-8

### Quotas Kaggle
- **GPU** : 30h/semaine, session max 12h, T4x2 = 2×16 Go, 4 CPU 29 Go RAM, 20 Go disk /kaggle/working
- **TPU** : **20h/semaine**, session max **9h**, TPU v5e-8 (8 chips), 16 Go HBM par chip = 128 Go total HBM, 20 Go disk, phone verification requise
- **Nouveauté** : TPU v3-8 phasing out, remplacé par **TPU v5e-8** plus efficace, purpose-built generative AI, Megacore + Multislice [Kaggle product-announcements 607202, 12 sept 2025]

### Specs TPU v5e-8
- **Par chip** : 197 TFLOPs bf16, 393 TOPS int8, 16 Go HBM, 800 GB/s bandwidth, 400 GB/s ICI, 4 ports ICI, 2D torus pod 256 chips = 50.63 PFLOPS bf16 / 100 PetaOps int8 [Google Cloud docs v5e, 26 aug 2026]
- **Pod** : 256 chips, 51.2 TB/s all-reduce, 1.6 TB/s bisection, 6.4 Tbps DCN, DRAM 512 GiB par host, 8 chips par host
- **Prix** : $1.20/chip-hour vs TPU v4 $3.2 → 0.65× TCO, 2× training perf/$ et 2.5× inference perf/$ vs v4 [Google blog 29 aug 2023]

### Performance vs T4

| Workload | TPU v5e-8 | T4x2 | Gain |
|---|---|---|---|
| Llama 7B INT8 inference | 400 tok/s | 180 tok/s | **2.2×** (55% lower T4) |
| Llama 7B bf16 inference | 320 tok/s | 140 tok/s | **2.3×** |
| Llama 7B FP32 inference | 110 tok/s | 90 tok/s | 1.2× |
| BERT sentiment 128 tokens | 1200 inf/s (8ms) | 450 inf/s (22ms) | **2.7×** (167% higher) |
| BERT training | 2.8× plus rapide que A100 | - | - |
| T5-3B training | 12h vs 31h sur GPU comparable | - | **2.6×** |
| Qwen3 27B bf16 262k context (Kaggle TPU v5e-8) | 130 tok/s single stream, 540 tok/s 8 streams, prefill 10300 tok/s, 225k prompt 28s | - | - |
| TPU v5e vs v4 training | 2× higher training | - | - |
| TPU v5e vs v4 inference | 2.5× higher inference | - | - |

**Source** : deploybase.ai v5e vs T4 7 avril 2025, Kaggle docs TPU, Google Cloud TPU v5e docs, GitHub kaggle-tpu-lab (Qwen3 27B 130 tok/s).

**Conclusion** : TPU v5e-8 est **2-2.5× plus rapide que T4** en training/inference bf16/int8, et **2.7×** sur BERT, avec **800 GB/s vs 320 GB/s** bandwidth.

### Mais : TPU nécessite code différent

- **T4** : PyTorch CUDA, `torch.compile`, `scaled_dot_product_attention`, `fused AdamW`, notre pipeline ultra MAX v3 fonctionne direct
- **TPU v5e** : **XLA / JAX** stack, pas CUDA. Il faut :
  - PyTorch XLA (`torch_xla`) ou JAX/Flax, ou HuggingFace Trainer avec `tpu` 
  - Pas de `bitsandbytes` 4-bit QLoRA (bitsandbytes = CUDA only). Pour TPU, il faut **full fine-tune bf16** ou **LoRA sans quantization** ou **8-bit via XLA** ?
  - Pas de `torch.compile` CUDA, mais `jax.jit` / `torch_xla` compile XLA
  - Dataset streaming avec `tf.data` ou `torch_xla` DataLoader

**Donc : notre script `finetune_rodin1b.py` QLoRA 4-bit ne marche PAS sur TPU v5e tel quel. Il faut réécrire pour XLA ou utiliser Trainer TPU.**

## Luciole-1B-Instruct-1.1 contexte

- **Contexte** : **131072 tokens (131k)** [llm-explorer 10 juil 2026, ollama 17 juil 2026, OpenLLM-France 23B base card]
- **Base** : Luciole-1B-Base context étendu 4096 → 32768 (50B tokens) → 131072 (50B tokens) sur Jean Zay
- **Instruct** : entraîné sur séquences 16384 tokens (vs base 131k), 3 phases SFT 2.6M avec thinking traces + 2.1M sans + DPO 284k, batch 2048, mostly English SFT mais base 30% FR
- **Besoin d'entraîner contexte ?** **NON** — déjà 131k, pas besoin d'extension YaRN comme Croissant 2048→8192. Économie 28h.

## Temps estimé pour Luciole-1B fine-tuning

### Sur T4x2 (notre pipeline v3 MAX)

- 1.238B params → 9.1 GFLOP/token (comme Rodin/Croissant)
- T4 efficace 22 TFLOPS, dual 44 TFLOPS → 4835 tok/s
- 2B tokens LoRA QLoRA 4-bit : 50-60h dual-T4 (5 sessions 12h)
- Full fine-tune 2B tokens : 100h

### Sur TPU v5e-8

- 8 chips × 197 TFLOPs bf16 = 1576 TFLOPS crête, efficace ~50% = 788 TFLOPS (estimation)
- Tokens/s = 788e12 / 9.1e9 ≈ 86600 tok/s → **18× plus rapide que T4x2** théorique
- Mais en pratique, TPU v5e training 2× vs v4, et v4 ~1.2-1.7× A100, A100 ~3× T4 → TPU v5e-8 ~6-8× T4x2 réel
- **Estimation réaliste** : 2B tokens LoRA sans quantization (full bf16) sur TPU v5e-8 = **8-12h** (vs 50-60h T4x2) → **5-6× plus rapide**
- 2B tokens full fine-tune bf16 : 15-20h vs 100h T4x2
- **Quota Kaggle TPU 20h/semaine** : 2B tokens LoRA 8-12h rentre dans 1 semaine (20h quota), 2 sessions 9h max

**Donc : TPU v5e est beaucoup plus rapide, 5-6×, et 20h quota suffit pour 2B tokens LoRA.**

### Inconvénients TPU Kaggle

- **Queue** : T4 presque jamais queue, TPU parfois queue (demande/supply)
- **Session max 9h** vs 12h GPU, donc 2B tokens 8-12h = 1-2 sessions avec resume
- **Pas de bitsandbytes** : QLoRA 4-bit impossible, il faut full bf16 ou LoRA sans quant (VRAM 128 Go HBM total = OK pour 1B full bf16 2.5 Go + optimizer 10 Go + activations 5 Go = ~18 Go, rentre dans 128 Go)
- **Code XLA** : à réécrire, pas notre `blackcode/train.py` CUDA

## Recommandation

**Pour Luciole-1B-Instruct-1.1, passe sur TPU v5e-8 si :**

- Tu veux 131k contexte sans entraîner contexte (économie 28h)
- Tu veux 5-6× plus rapide que T4x2 (2B tokens 8-12h vs 50-60h)
- Tu as 20h quota TPU/semaine, ça rentre
- Tu acceptes de réécrire training loop en XLA/JAX ou utiliser HF Trainer TPU (je peux le faire)

**Reste sur T4x2 si :**

- Tu veux QLoRA 4-bit (bitsandbytes CUDA only, VRAM ÷3)
- Tu veux notre pipeline ultra MAX v3 déjà prêt et testé (36 tests OK)
- Tu veux pas de queue TPU
- Tu veux 30h quota GPU vs 20h TPU

**Mon avis : pour Luciole-1B 131k, TPU v5e-8 est le mieux** — 131k déjà entraîné, 760B FR déjà vus, 5-6× plus rapide, 20h quota suffit pour 2B LoRA. On perd QLoRA 4-bit mais on a 128 Go HBM total, donc full bf16 OK.

Je peux te créer `colab/finetune_luciole_tpu_ultra.ipynb` avec PyTorch XLA + Trainer TPU + QLoRA sans bitsandbytes (ou full fine-tune bf16) + toutes optim v3 adaptées TPU (prefetch, bucket, etc.).

Tu veux que je le fasse et on lance sur TPU v5e-8 ?

## Sources

- Kaggle TPU v5e-8 : https://www.kaggle.com/product-announcements/607202 (2× training, 2.5× inference vs v4, Megacore, Multislice, 12 sept 2025)
- Kaggle docs TPU : https://www.kaggle.com/docs/tpu (20h/semaine, 9h max session, TPU v3-8, phone verification)
- Kaggle quotas : https://www.kaggle.com/questions-and-answers/408081 (GPU 30h, TPU 20h, 50-60h total/semaine, queue TPU parfois)
- TPU v5e specs : https://docs.cloud.google.com/tpu/docs/v5e (197 TFLOPs bf16/chip, 393 TOPS int8, 16 Go HBM, 800 GB/s, 400 GB/s ICI, pod 256 chips 50.63 PFLOPS)
- TPU v5e vs T4 inference : https://deploybase.ai/articles/v5e-1-tpu-vs-t4-gpu (Llama 7B INT8 400 vs 180 tok/s 2.2×, bf16 320 vs 140 tok/s 2.3×, BERT sentiment 1200 vs 450 inf/s 2.7×, 7 avril 2025)
- TPU v5e training 2× perf/$ vs v4, 2.5× inference perf/$ : https://cloud.google.com/blog/products/compute/announcing-cloud-tpu-v5e-and-a3-gpus-in-ga (29 aug 2023)
- TPU v5e 800 GB/s vs T4 320 GB/s : https://deploybase.ai/articles/v5e-1-tpu-vs-t4-gpu
- Qwen3 27B sur Kaggle TPU v5e-8 130 tok/s 262k context : https://github.com/ARahim3/kaggle-tpu-lab + https://github.com/Kitkitkittt/kaggle-tpu-lab (prefill 10300 tok/s, 225k prompt 28s, 20h/week, 9h max, 20 min startup)
- Luciole-1B context 131072 : https://llm-explorer.com/model/OpenLLM-France/Luciole-1B-Instruct-1.1,4yAMXHBMR9w9nRIBqVubUA (VRAM 2.6 Go, 128K context) + https://ollama.com/OpenLLM-France/Luciole-Instruct-1.1:1B + https://huggingface.co/OpenLLM-France/Luciole-23B-Base (extension 4k→32k 50B + 32k→131k 50B)
- SmolLM2 context 8192 : https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B/discussions/1 (8192 tokens) + https://arxiv.org/html/2502.02737v1 (extended 2k→8k RoPE 130k)
