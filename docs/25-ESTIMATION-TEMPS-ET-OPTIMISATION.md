# Estimation temps dataset + entraînement + optimisations max

## Contexte modèle

- Config : `blackcode_fr_260m.json` — 258 578 048 paramètres, 20 blocs (15 conv / 5 attention), d_model 1024, vocab 24 576, embeddings liés.
- Contexte cible : 2048 (phase 1), puis 8192 (phase 2), expérimental 128k YaRN.
- FLOPs/token (seq_len 2048) : 1,677 GFLOP (body 1,4 GFLOP 83,5%, attention 0,126 GFLOP 7,5%, logits 0,151 GFLOP 9%).
- T4 : 65 TFLOPS crête fp16, MFU réaliste 35% → 22,8 TFLOPS effectifs, ~13 563 tokens/s pour ce modèle.

## 1. Formation du corpus français

### Sources et volumes

- FineWeb2-HQ `fra_Latn` : 32,2M docs / 483 Go annoncés (fiche). Filtré top 10% qualité XLM-RoBERTa.
- Common Corpus FR : ~275B tokens FR annoncés (à vérifier), métadonnées par ligne.
- Objectif lot initial : 4 000 Mo (MAX_MB=4000) → ~1 Md tokens estimés (4,2 bytes/token).

### Étapes et temps estimés (sur Colab/Kaggle T4, 1 GPU, streaming HF)

| Étape | Opération | Temps estimé | Goulot | Optimisation |
|---|---|---:|---|---|
| Audit 2000 lignes | `audit_fr_source.py --limit 2000` | 10-20 min | réseau HF + lingua | sans lingua (fallback heuristique) → 5 min, cache lingua models |
| Build 4 Go FineWeb2-HQ | streaming, french-only, quality, dedup | 60-120 min | réseau + déduplication SimHash | `--no-near` pour première passe → 40 min, puis dedup séparée |
| Build 4 Go Common Corpus avec allowlist | + filtre licence par ligne | 90-150 min | parsing licence + français | pré-filtrage `keep-if` regex français + licence |
| Tokenizer 24k | BPE byte-level 120M chars | 15-30 min | CPU | `--tokenizer-chars 20M` pour test rapide → 5 min |
| Pack train/val + manifest | écriture bin + manifest.jsonl | 5-10 min | disque | SSD local `/content` vs Drive |
| **Total 4 Go** | | **~2h30 à 4h** | | **Optimisé ~1h30** |

**Détail build** :
- `iter_hf_docs` streaming : ~300-600 docs/s selon réseau, 2M docs pour 4 Go → 60-110 min.
- `is_french_prose` avec lingua : ~5 ms/doc → 2M docs = 2,7h ! D'où fallback heuristique 0,45 seuil : ~0,1 ms/doc → 3 min.
- `Deduper` : SimHash 64 bits ~0,5 ms/doc, LSH 8 bandes → ~15 min pour 2M docs, RAM <1 Go pour 200k docs, mais 2M → ~10 Go si tout en mémoire. Solution : shard dedup par dump ou `--no-near` puis dedup batch.

### Optimisations max pour dataset

1. **Sans lingua pour première passe** : `use_heuristic_fallback=True` déjà implémenté, divise temps par ~50.
2. **Streaming + budget** : `--max-mb` coupe tôt, évite de parcourir 483 Go.
3. **Pré-filtre qualité** : `MIN_QUALITY_SCORE=0.6` (après audit p10/p20) rejette 40-60% tôt, avant français/dedup.
4. **Allowlist licence** : `--require-hf-license --allowed-hf-license` pour Common Corpus évite de traiter lignes non autorisées.
5. **Dedup en deux passes** : build avec `--no-near` (exact seulement, rapide), puis seconde passe SimHash sur manifest trié par hash.
6. **Disque local** : travailler dans `/content` ou `/kaggle/working`, pas Drive (Drive lent 10-60s/écriture). Copie finale vers Drive en thread async (`--mirror-dir`).
7. **Tokenizer léger pour test** : vocab 1024 + 20M chars pour valider pipeline en 5 min, puis 24k complet.
8. **Parallélisme** : sharder dataset par `dump` ou `collection`, lancer 2 builds en parallèle sur 2 T4 (Kaggle dual-T4) → ~2× débit si DDP implémenté ou manuel.

### Drive → Kaggle

| Transfert | Taille | Temps | Méthode optimisée |
|---|---:|---:|---|
| Local → Drive | 4 Go (train.bin+val.bin+tokenizer+manifest ~4,2 Go) | 10-30 min (upload Drive 5-20 Mo/s) | tar + `dd` ou `rclone`, pas `cp -a` (25 min échoué précédemment). `tar cf - data/fr_260m | dd bs=4M` → Drive |
| Drive → Kaggle | 4 Go | 5-15 min (download Drive → Kaggle ~20-50 Mo/s) | `gdown` ou Kaggle Dataset (plus rapide, versionné). Créer dataset Kaggle privé `blackcode-fr-260m` avec `kaggle datasets create` → 2 min d'indexation |
| **Total** | | **15-45 min** | Kaggle Dataset recommandé pour reprise |

**Astuce** : Ne pas mettre `data/` dans Git (gitignore). Utiliser `artifacts/` ou Drive. Le manifest permet de vérifier intégrité après transfert (hash).

## 2. Entraînement

### Calculs de base (T4, 22,8 TFLOPS effectifs, seq_len 2048)

| Tokens | Temps 1×T4 | Tokens/s | Sessions Colab 10h | Commentaire |
|---|---:|---:|---:|---|
| 1 Md | **20,5 h** | 13 563 | 3 | pilote qualité |
| 4 Md | **3,4 j** | 13 563 | 9 | Chinchilla ~15 tokens/param pour 258M |
| 10 Md | **8,5 j** | 13 563 | 21 | ~38 tokens/param |
| 20 Md | **17 j** | 13 563 | 41 | limite haute from scratch |

Chinchilla optimal pour 258M : ~5,16 Md tokens (20 tokens/param). Donc **4-6 Md** est cible réaliste.

### Mémoire entraînement (fp32, activations 1,5 Go)

| Optimiseur | Total | Tient 16 Go | Commentaire |
|---|---:|---|---|
| AdamW | ~19,6 Go (extrapolé 1,2Md) → ~6-7 Go pour 258M | oui | mais 19,6 Go pour 1,2Md = non |
| Adafactor | ~11,5 Go pour 1,2Md → ~3-4 Go pour 258M | oui | marge 12 Go, batch plus grand |
| AdamW 8-bit | ~12,8 Go pour 1,2Md → ~4 Go pour 258M | oui | nécessite bitsandbytes |
| Muon | ~14 Go pour 1,2Md → ~4,5 Go pour 258M | oui | 1,3-2× efficacité tokens |

Pour 258M, tous tiennent sur T4 16 Go, mais Adafactor/Muon permettent batch 2-4× plus grand → meilleur MFU.

### Optimisations max entraînement

| Levier | Gain | État actuel | Commande |
|---|---|---:|---|---|
| **Adafactor** | permet batch 4-8 vs 1 | implémenté | `--optim adafactor --lr 3e-4` (LR 30× plus haute auto) |
| **Checkpoint sélectif** | +10-15% débit vs complet | implémenté | `--checkpoint-policy selective` (1 bloc sur 2) |
| **torch.compile** | +10-25% sur T4, 0% CPU | implémenté | `--compile` |
| **Batch + grad-accum** | MFU ↑ jusqu'à batch critique 0,5-4M tokens | à tuner | `--batch-size 2 --grad-accum 8` → 32k tokens/pas |
| **Seq-len warmup** | +0-10% (attention peu coûteuse 7,5%) | implémenté | `--seq-len-warmup-frac 0.2 --seq-len-start 512` |
| **WSD schedule** | évite perte session coupée, +0-30% mural | implémenté | `--schedule wsd --stable-frac 0.8` |
| **Évals espacées** | +5-10% mural | à faire | `--eval-interval 1000 --eval-iters 4` |
| **Mirror async** | 10-60s sauvés / checkpoint | implémenté | `--mirror-dir /content/drive/...` |
| **Données en local** | variable, Drive lent | à faire | copier `data/` → `/content` |
| **Pruning données** | ×1,5-3 moins tokens à qualité égale | `blackcode.prune` dispo | entraîner petit modèle 51M, garder bande perte [0,05;0,85] |
| **Muon** | ×1,3-2 efficacité tokens | implémenté | `--optim muon --muon-lr 0.02 --lr 6e-4` (2 groupes) |
| **Dual-T4 DDP** | ×1,8-2 (si implémenté) | **non implémenté** actuellement | `blackcode.train` actuel mono-GPU, besoin DDP |

**Cumul réaliste sans distillation (from scratch demandé)** :
- Référence 4 Md : 3,4 j
- + Adafactor + selective + compile : 3,4j ×0,78×0,85 ≈ **2,2 j** (×1,5)
- + pruning ×2 : **1,1 j**
- + Muon ×1,5 : **0,75 j** (~18h)
- + dual-T4 ×1,8 : **10h** pour 4 Md tokens

Donc **10-24h pour 4 Md tokens** optimisé sur Kaggle dual-T4, vs 3,4j non optimisé mono-T4.

### Estimation complète Drive+Kaggle+train

| Scénario | Dataset 4 Go | Drive | Kaggle | Train 4 Md (from scratch) | Total |
|---|---:|---:|---:|---:|---:|
| **Non optimisé mono-T4** | 4h | 30 min | 15 min | 3,4 j | **~4 j** |
| **Optimisé mono-T4** (adafactor+selective+compile+pruning+Muon) | 1,5h | 15 min | 5 min | 18h | **~20h** |
| **Optimisé dual-T4 Kaggle** | 1,5h (parallèle) | 15 min | 5 min | 10h | **~12h** |
| **Avec distillation** (si autorisé, non demandé) | 1,5h | 15 min | 5 min | 2-4h pour 30-50M tokens distillés ≈ 11,8 Md from scratch | **~4-6h** |

**Coût Kaggle** : 2×T4, 30h/semaine gratuites. 4 Md from scratch optimisé dual-T4 = ~10h → 1/3 quota hebdo. 10 Md = ~25h → quota presque plein.

## 3. Recommandation pipeline optimisé

```bash
# 1. Audit rapide sans lingua (5 min)
python scripts/audit_fr_source.py --dataset epfml/FineWeb2-HQ:fra_Latn --limit 2000 --samples-out data/audits/fineweb2_hq.jsonl

# 2. Build 4 Go avec filtres qualité + french heuristic + licence (1,5h)
MAX_MB=4000 MIN_QUALITY_SCORE=0.6 OUT=data/fr_260m HF_DATASET=epfml/FineWeb2-HQ:fra_Latn VOCAB=24576 SEQ_LEN=2048 bash scripts/build_fr_corpus.sh

# 3. Vérif pureté (2 min)
python scripts/eval_french_purity.py --corpus-dir data/fr_260m --limit 100

# 4. Tokenizer déjà fait dans build, sinon :
# python -m blackcode.data build --src data/fr_seed --out data/fr_260m --vocab 24576 --seq-len 2048 --french-only --max-mb 4000

# 5. Copie vers Drive optimisée (tar + dd, 10 min)
tar cf - data/fr_260m | (cd /content/drive/MyDrive/blackcode && tar xf -)

# 6. Kaggle : créer dataset privé (5 min) ou gdown
# kaggle datasets create -p data/fr_260m --dir data/fr_260m -p ...

# 7. Train pilote 200 steps (10 min) pour mesurer MFU
python -m blackcode.train --config configs/blackcode_fr_260m.json --data data/fr_260m --out runs/fr_260m_pilot --batch-size 2 --grad-accum 8 --seq-len 2048 --max-steps 200 --optim adafactor --lr 3e-4 --checkpoint-policy selective --compile --eval-interval 100 --save-interval 100 --resume --mirror-dir /content/drive/MyDrive/blackcode/runs/fr_260m_pilot --log-flops

# 8. Train complet 4 Md tokens (~10h dual-T4 optimisé)
# max-steps = tokens / (batch*grad-accum*seq_len) = 4e9 / (2*8*2048) = 122k steps
python -m blackcode.train --config configs/blackcode_fr_260m.json --data data/fr_260m --out runs/fr_260m_4b --batch-size 2 --grad-accum 8 --seq-len 2048 --max-steps 122070 --optim muon --muon-lr 0.02 --lr 6e-4 --schedule wsd --stable-frac 0.8 --checkpoint-policy selective --compile --eval-interval 1000 --save-interval 1000 --resume --mirror-dir /content/drive/MyDrive/blackcode/runs/fr_260m_4b --log-flops
```

## 4. Limites

- Chiffres FLOPs = modèle théorique, pas mesure T4 partagé (MFU peut tomber à 15%).
- Pruning et Muon gains issus littérature, à valider par ablation 0,5 Md tokens.
- Dual-T4 DDP non implémenté dans `blackcode.train` actuel → besoin d'implémenter ou lancer 2 sessions manuelles avec DiLoCo.

