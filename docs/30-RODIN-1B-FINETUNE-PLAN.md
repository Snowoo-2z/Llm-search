# Rodin-1B Base — Plan de fine-tuning pour 1B puissant parfaitement français

**Date : 18 oct 2026. Recherche web_search + fetch HuggingFace 18 oct 2026.**

## Recherche Rodin-1B

### Fiche officielle
- **ID** : `rodin-llm/rodin-1b` https://huggingface.co/rodin-llm/rodin-1b
- **Licence** : **Apache 2.0** (permissive, commercial OK)
- **Paramètres** : **1.238 B** (1.24B annoncé)
- **Architecture** : LLaMA-style : RoPE, RMSNorm, SwiGLU, causal SDPA attention, 22 layers, hidden 2048, 16 heads (no GQA, n_kv_heads = n_heads), FFN 5461, vocab 64k custom SentencePiece BPE français, context 2048, RoPE theta 10k, RMSNorm eps 1e-5, weight tying yes, bfloat16 training
- **Tokenizer** : SentencePiece BPE 64k maison entraîné sur français (pas notre BPE 24k byte-level)
- **Entraînement original** : **32B tokens français** uniquement (HPLT CC0 packaging, CC100 permissive, Wikipedia FR CC BY-SA, Wikisource FR, Pleias books/news open/public domain, Légifrance Open license), 478k steps sur B200 spot + RTX 3090 local, bfloat16
- **Variante instruct** : `rodin-llm/rodin-1b-instruct` existe pour chat
- **Évaluation FrenchBench 3-shot** : Vocab 0.773, Grammaire 0.765, Compréhension écrite 0.606, BoolQA 0.573, NLI thématique 0.498, HellaSwag 0.424, FQuAD bool 0.500, XNLI 0.333, ARC challenge 0.265, Trivia f1 0.245, FQuAD genq f1 0.165, FQuAD f1 0.095. Catégories : Linguistique 0.68, Raisonnement MCQ 0.40, QA générative 0.16. **Lecture** : grammaire/vocab excellents (tokenizer FR dédié + corpus qualité), raisonnement/QA faible attendu pour 1B/32B tokens.
- **Limitations** : base model, pas instruction-tuné, pas safety tuning, hallucine, français uniquement, contexte 2048
- **Source code complet** : https://github.com/rodin-llm/rodin (data, tokenizer, pretraining, SFT, export, spot orchestration)
- **Taille fichiers** : model.safetensors 2.79 Go, GGUF Q4_K_M 860 Mo, Q8_0 1.48 Go, f16 2.79 Go

### Pourquoi c'est une excellente base pour ton objectif

- **Français natif de zéro**, pas un fine-tune Llama multilingue. Tokenizer 64k FR = meilleure efficacité que Llama 32k multilingue
- **Apache 2.0** = tu peux fine-tuner, redistribuer, commercial
- **1.238B** = 5× plus grand que 260M, mais encore entraînable sur T4 avec LoRA/QLoRA + gradient checkpoint + 8-bit
- **32B déjà vus** = il a déjà une bonne base linguistique FR, tu n'as pas besoin de refaire 32B, tu continues avec 10-20B de haute qualité
- **Architecture LLaMA** = compatible Transformers, FlashAttention, PEFT, TRL, vLLM, llama.cpp

## Objectif : 1B puissant très intelligent parfaitement français

Tu veux : beaucoup plus de tokens + beaucoup plus de taille (1B) + beaucoup plus de connaissance + parfaitement français.

**Traduction en plan technique :**
1. **Continue pretraining** (pas juste SFT) sur 10-20B tokens FR haute qualité pour ajouter connaissance
2. **SFT instruction** sur français pur (Aya FR, French Instruct filtré, Claire AN, 88milSMS transcodé, etc.)
3. **DPO / ORPO** optionnel pour alignement
4. **Évaluation** FrenchBench + agent_fr_v2.jsonl

## Données pour ajouter connaissance (10-20B tokens)

Mélange recommandé, 100% français, licences vérifiées :

| Source | Tokens | Licence | Pourquoi |
|---|---|---|---|
| FineWeb2-HQ fra_Latn filtré qualité ≥0.6 | 6-8B | ODC-By 1.0 packaging, CC à vérifier par page | Web français haute qualité, top 10% FineWeb2 |
| Pleias Common Corpus FR filtré licence permissive | 2-3B | par ligne (MIT, CC0, CC-BY, Licence Ouverte) | traçabilité ligne par ligne, livres/journaux FR open |
| Wikipedia FR + Wikisource FR | 0.5B | CC BY-SA | encyclopédie + domaine public |
| Légifrance + Open data FR | 0.5B | Licence Ouverte | juridique/administratif FR |
| French-PD-Books (Gallica) filtré OCR | 1-2B | domaine public (vérifier BNF conditions) | littérature FR |
| 88milSMS + smslareunion (CC BY 4.0) | 0.01B | CC BY 4.0 | abréviations RS, 5% du mélange SFT pas pretrain massif |
| Claire AN (dialogues) | 0.1B | Licence Ouverte | dialogue oral FR |

**Total : ~10-15B tokens FR haute qualité** = 3-5× plus que Rodin original 32B ? Non, 32B + 10-15B = 42-47B total, mais avec meilleure qualité et diversité. Pour vraiment "beaucoup plus de connaissance", il faudrait 100B, mais sur dual-T4 c'est 500h, impossible. 10-15B est réaliste en 50-100h dual-T4 avec LoRA.

## Temps d'entraînement estimé pour 1B (fait, pas promesse)

Formule : FLOPs/token ≈ 6×N (1.238B) + attention 12×L×d×T. Pour T=2048, L=22, d=2048 → attention = 12×22×2048×2048 ≈ 1.1B, body = 6×1.2B = 7.2B, logits = 6×64k×2048 ≈ 0.8B → total ~9.1 GFLOP/token.

T4 = 65 TFLOPS crête fp16, efficace ~22 TFLOPS avec nos optim (fused, compile max-autotune, cudnn benchmark, TF32, prefetch, bucket 10, FP16 compress). Dual-T4 = 44 TFLOPS efficaces.

Tokens/s = 44e12 / 9.1e9 ≈ 4835 tokens/s dual-T4.

- 10B tokens : 10e9 / 4835 = 2 068 000 s = 574h = 24j dual-T4 → **trop long** pour Kaggle 30h/semaine
- 2B tokens : 2e9 / 4835 = 114h = 4.7j dual-T4 → **faisable en 10 sessions Kaggle 12h avec resume**
- Avec LoRA (r=64, ~50M params entraînables au lieu de 1.2B) : forward identique mais backward 10× moins cher + optimizer 20× moins cher → ~2× plus rapide tokens/s + batch plus grand possible → 2B tokens en ~50-60h dual-T4 → **5 sessions Kaggle**

**Donc : full fine-tune 1B sur 2B tokens = ~100h dual-T4, LoRA 2B tokens = ~50-60h.**

Si tu veux vraiment 100B tokens, il faut B200/H100, pas T4.

## Stratégie recommandée : 2 phases

### Phase 1 — Continue pretraining LoRA (connaissance)

- **Méthode** : QLoRA 4-bit (bitsandbytes) + PEFT LoRA r=64 alpha=128 dropout 0.05, target_modules = q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj (tous les linears LLaMA)
- **Données** : 2B tokens FR haute qualité (FineWeb2-HQ 1.5B + Pleias 0.3B + Wikipedia 0.2B)
- **Batch** : 2×8×2048 = 32768 tokens/step dual-T4 (batch 2, grad_accum 8, seq 2048, world 2)
- **Steps** : 2e9 / 32768 ≈ 61035 steps
- **Optim** : AdamW fused, lr 2e-4 (10× plus petit que pretrain), cosine decay, warmup 2%, weight_decay 0.1, grad_clip 1.0, bf16
- **Temps** : ~50-60h dual-T4 → 5 sessions Kaggle 12h avec --resume
- **Résultat** : Rodin-1B + 2B tokens = 34B total, plus de connaissance, même tokenizer

### Phase 2 — SFT instruction français (intelligence conversationnelle)

- **Méthode** : SFTTrainer TRL + LoRA (ou full si VRAM ok avec Deepspeed ZeRO), chat template français
- **Données** : Aya FR 1.4k + French Instruct filtré 50k + Claire AN 30k + 88milSMS transcodé 1k (mapping abréviation→standard) + 30 tâches agent_fr_v2.jsonl
- **Mélange** : 50% instructions formelles FR + 30% dialogues + 20% SMS/abréviations (comprendre mais répondre formel)
- **Batch** : 4×4×2048, lr 1e-4, 3 epochs
- **Temps** : 2-3h dual-T4

### Phase 3 optionnelle — DPO

- Préférences françaises, 1h

## Commandes copiables

### Installation
```bash
pip install transformers accelerate peft bitsandbytes trl datasets -q
```

### Fine-tune Rodin-1B QLoRA continue pretraining (script à créer)

Voir `scripts/finetune_rodin1b.py` (à venir) et `colab/finetune_rodin1b_ultra.ipynb`

```bash
# Mono-T4
python scripts/finetune_rodin1b.py --model rodin-llm/rodin-1b --data data/fr_10b --out runs/rodin1b_cont_2b --batch-size 2 --grad-accum 8 --seq-len 2048 --max-steps 61035 --lora-r 64 --lora-alpha 128 --lr 2e-4 --bf16 --gradient-checkpointing --compile --eval-interval 1000 --save-interval 1000 --resume

# Dual-T4 Kaggle
torchrun --nproc_per_node=2 scripts/finetune_rodin1b.py --model rodin-llm/rodin-1b --data data/fr_10b --out runs/rodin1b_cont_2b --batch-size 2 --grad-accum 8 --seq-len 2048 --max-steps 61035 --lora-r 64 --lora-alpha 128 --lr 2e-4 --bf16 --gradient-checkpointing --compile --ddp --ddp-fp16-compress --bucket-cap-mb 10 --eval-interval 1000 --save-interval 1000 --resume
```

### SFT
```bash
python scripts/sft_rodin1b.py --model runs/rodin1b_cont_2b --data data/sft_fr --out runs/rodin1b_instruct_fr --lora-r 64 --lr 1e-4 --epochs 3
```

## Évaluation

- FrenchBench : comparer base 0.773 vocab / 0.765 grammaire vs après continue pretrain
- agent_fr_v2.jsonl : 30 tâches FR
- Perplexité sur Wikipedia FR held-out
- Test abréviations : "slt cv ?" → doit comprendre et répondre formel si consigne

## Limites et honnêteté

- 1.238B reste petit vs 7B, 70B. Même avec 10B tokens de plus, il restera limité en raisonnement complexe (ARC 0.265 → peut monter à 0.35-0.40 max avec plus de données, pas à 0.80)
- Contexte 2048 natif, pas 128k. Étendre à 8k via RoPE scaling (theta 10k→100k) possible mais qualité à valider
- Pas de safety tuning dans base, à ajouter en SFT si déploiement public
- Temps : 2B tokens LoRA = 50-60h dual-T4 = 5 sessions Kaggle 12h = 2 semaines de quota 30h/semaine. 10B tokens = 250-300h = impossible sur T4, besoin B200/H100
- Si tu veux vraiment "beaucoup plus de tokens" (100B), il faut changer de hardware ou utiliser DiLoCo (2 sessions indépendantes sync 500 steps)

## Prochaines étapes

1. Créer `scripts/finetune_rodin1b.py` avec QLoRA + compile + DDP optim v3
2. Créer `colab/finetune_rodin1b_ultra.ipynb` avec pipeline complet
3. Build dataset FR 2B tokens avec `build_fr_optimized_v3.sh` MAX_MB=2000
4. Lancer continue pretraining LoRA 2B tokens sur Kaggle dual-T4 (5 sessions)
5. SFT français
6. Évaluer FrenchBench

## Sources

- Rodin-1B base : https://huggingface.co/rodin-llm/rodin-1b (Apache 2.0, 1.238B, 32B tokens, 22 layers, 2048 hidden, 64k vocab, 2048 context, FrenchBench 0.773 vocab)
- Rodin GitHub : https://github.com/rodin-llm/rodin
- Rodin-1B instruct : https://huggingface.co/rodin-llm/rodin-1b-instruct
- FrenchBench : https://huggingface.co/datasets/ (évaluation EleutherAI lm-evaluation-harness)
- HPLT, CC100, Pleias, Wikipedia FR, Légifrance : sources Rodin originales
- Notre optimisations v3 : docs/29-OPTIMISATIONS-ULTRA-MAX-V3.md
