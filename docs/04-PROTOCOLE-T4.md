# Protocole expérimental sur Colab T4

> T4 : 16 Go de VRAM, ~65 TFLOPS fp16 théoriques (Tensor Cores), **pas de bf16 natif**, ~12 Go de RAM système sur Colab gratuit, sessions limitées et interrompues.
> Toutes les commandes supposent le dépôt cloné et `pip install -r requirements.txt`.

---

## 0. Installation

```bash
!nvidia-smi
!git clone https://github.com/<votre-compte>/Llm-search.git
%cd Llm-search
!pip install -q torch tokenizers numpy
!pip install -q evalplus  # pour HumanEval+/MBPP+
```

Vérifier que le GPU est bien utilisé :

```python
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0).total_memory / 1e9, "Go")
```

Sur T4, utiliser **fp16 + GradScaler** (`--dtype float16`, comportement par défaut sur CUDA). Ne pas utiliser bf16 : le T4 n'a pas les instructions natives, le calcul serait plus lent et moins stable.

---

## 1. Données

### 1.1 Choix des corpus

| Corpus | Taille | Usage | Remarque |
|---|---:|---|---|
| `bigcode/the-stack-smol` | ~1 Go | mise au point | petit, téléchargeable |
| `bigcode/the-stack-v2` | > 10 To | production | filtrage par langage nécessaire |
| `codeparrot/codeparrot-clean` | ~50 Go | production | Python majoritaire |
| `bigcode/starcoderdata` | ~250 Go | production | 86 langages |
| `ise-uiuc/Magicoder-OSS-Instruct-75K` | ~100 Mo | SFT | instructions code |
| `bigcode/self-oss-instruct-sc2-exec-filter-50k` | ~100 Mo | SFT | filtré par exécution |
| `openai/openai_humaneval`, `mbpp` | ~1 Mo | évaluation | à garder hors entraînement |

Pour un premier passage sur T4, viser **1 à 5 Go de code** (Python + JS/TS + Rust + Go), pas 50 Go. Le temps de tokenisation et de téléchargement domine sinon.

### 1.2 Préparer le corpus

Deux options.

**Option A — arborescence locale** (simple, reproductible) :

```bash
!git clone --depth 1 https://github.com/TheAlgorithms/Python /content/corpus/python
!git clone --depth 1 https://github.com/psf/requests /content/corpus/requests
!python -m blackcode.corpus prepare \
    --src /content/corpus --out data/code \
    --vocab 32000 --suffixes .py,.pyi,.md --max-bytes 200000
```

**Option B — dataset HuggingFace téléchargé en JSONL** :

```python
from datasets import load_dataset
ds = load_dataset("bigcode/the-stack-smol", split="train", streaming=True)
with open("/content/code.jsonl", "w") as f:
    for row in ds:
        if row.get("lang") in {"Python", "JavaScript", "Rust", "Go"}:
            f.write(json.dumps({"content": row["content"]}) + "\n")
```

```bash
!python -m blackcode.corpus prepare --jsonl /content/code.jsonl --out data/code \
    --vocab 32000 --text-field content
```

Le script déduplique par SHA-1, sépare train/val par fichier (pas de fuite), entraîne un BPE byte-level et écrit `train.bin`/`val.bin` en uint16.

---

## 2. Étape 0 — valider le pipeline (5 minutes)

```bash
!python -m blackcode.train --config configs/nano.json --data data/code \
    --out runs/nano --batch-size 8 --seq-len 128 --max-steps 300 \
    --lr 1.5e-3 --eval-interval 100 --save-interval 100
!python -m blackcode.generate --ckpt runs/nano/best.pt \
    --tokenizer data/code/tokenizer.json --prompt "def add(a, b):" --max-new-tokens 60
!python -m blackcode.export --ckpt runs/nano/best.pt --out artifacts/nano.bcpkg --context 256
```

Critère de succès : la perte descend, la génération produit des tokens valides, l'export affiche un budget cohérent.

---

## 3. Étape 1 — véhicule de recherche `t4_small` (51 M)

### 3.1 Entraînement

```bash
!python -m blackcode.train --config configs/t4_small.json --data data/code \
    --out runs/t4_small --batch-size 8 --grad-accum 4 --seq-len 1024 \
    --max-steps 30000 --lr 6e-4 --min-lr 6e-5 --warmup 600 \
    --eval-interval 500 --save-interval 500 --grad-checkpoint
```

- **Tokens par pas** : 8 × 4 × 1024 = 32 768.
- **30 000 pas** = ~1 Md de tokens.
- **Durée** : 3 à 6 h sur T4 selon le débit (compter 15–25 TFLOPS effectifs).
- **VRAM** : ~6–9 Go avec checkpointing. Si OOM, réduire `--batch-size 4` et augmenter `--grad-accum 8`.

### 3.2 Reprise après coupure

```bash
!python -m blackcode.train ... --resume
```

Sauvegarder sur Google Drive toutes les 500 pas :

```python
from google.colab import drive
drive.mount('/content/drive')
# puis utiliser --out /content/drive/MyDrive/blackcode/runs/t4_small
```

### 3.3 Ablations A1–A6

Chaque ablation = un `--config` différent, même nombre de tokens, même seed.
Pour créer une variante, copier le JSON et modifier **un seul champ** :

| Ablation | Champ | Variante |
|---|---|---|
| A1 fp16 | `ternary: false` | comparer la perte à mémoire égale |
| A2 motif | `attn_every: 1` / `2` | tout-attention / 1:1 |
| A3 forme | `d_model`, `n_layers` | 12×768 au lieu de 24×512 (à params égaux) |
| A4 embeddings | `tie_embeddings: false` | +67 M de paramètres |
| A5 loop | `loop: 2` | même mémoire, plus de calcul |
| A6 vocab | `vocab_size: 16000/64000` | retokeniser le corpus |

**Protocole** : 1 Md de tokens par variante, perte de validation + pass@1 sur 50 problèmes unitaires. Une ablation qui ne tient pas dans une session se découpe avec `--resume`.

---

## 4. Étape 2 — distillation ternaire (le cœur)

### 4.1 Pourquoi pas from scratch

Un étudiant de 0,6 Md entraîné de zéro sur 100 Md de tokens demanderait ~100 000 TFLOPS-jours ; le T4 n'en fournit que ~0,5 par jour. La distillation réduit le nombre de tokens nécessaires d'un facteur 10 à 50.

### 4.2 Professeur

| Professeur | VRAM en 4 bits | Débit T4 | Usage |
|---|---:|---|---|
| Qwen2.5-Coder-7B | ~4,5 Go | ~15–25 tok/s | qualité, petits volumes |
| Qwen2.5-Coder-3B | ~2 Go | ~40–60 tok/s | volume moyen |
| Qwen3-1.7B | ~1,2 Go | ~80–120 tok/s | passe rapide, ablations |

Le T4 peut faire tourner le professeur 7B et l'étudiant 0,6 Md **en même temps** si l'étudiant utilise Adafactor + checkpointing (~10–11 Go au total). Sinon, précalculer les top-k du professeur sur disque.

### 4.3 Recette TAD (M1)

```
1. Étudiant fp16 : init aléatoire, distillation KL(top-50) + CE sur 20–50 M tokens.
2. Ternarisation : remplacer les matrices par TernaryLinear, annealing de quantification
   sur 20 % des pas, distillation + CE sur 10–20 M tokens.
3. Récupération on-policy : 5–10 M tokens générés par l'étudiant et notés par le professeur.
4. RLVR (étape 3 séparée) sur tests unitaires.
```

Sur T4, l'étape 1 prend ~10–20 h pour 30 M tokens, l'étape 2 ~5–10 h, l'étape 3 ~5–10 h. C'est un projet de **plusieurs semaines de sessions Colab**, pas d'une nuit blanche.

### 4.4 Précalcul des logits

```python
# pseudo-code : une passe professeur, stockage top-k
for batch in loader:
    logits = teacher(batch.input_ids)          # [B, T, V]
    topk = logits.topk(50, dim=-1)
    save(topk.values.half(), topk.indices.int32())
```

Compter ~2× le temps d'un forward par token, et ~100 octets par token pour k=50 (valeurs fp16 + indices int32). 30 M tokens ≈ 3 Go.

---

## 5. Étape 3 — RLVR sur tests unitaires

- **Données** : MBPP (train), APPS, ou des problèmes synthétiques avec tests générés et filtrés par exécution.
- **Algorithme** : GRPO ou GSPO, groupe de 8 rollouts, récompense = tests passés / total + petite pénalité Ruff, sans KL (ou KL faible).
- **Mémoire** : LoRA sur l'étudiant ternaire est délicat ; le plus simple est de faire le RLVR **avant** la ternarisation finale, ou d'entraîner les poids maîtres fp32 avec QAT.
- **Durée** : 10 000 problèmes × 8 × 256 tokens ≈ 20 M tokens générés, ~10–20 h sur T4.
- **Attendu** : +5 à +15 points de pass@1 (la littérature rapporte +13 sur Qwen3-0.6B).

---

## 6. Étape 4 — export, mesure, évaluation

```bash
!python -m blackcode.export --ckpt runs/t4_small/best.pt \
    --out artifacts/blackcode_t4.bcpkg --context 4096
```

Le manifeste donne la taille réelle, les bits effectifs et la RAM estimée. Pour une mesure **honnête** de la RAM :

```bash
/usr/bin/time -v python -m blackcode.bench --pkg artifacts/blackcode_t4.bcpkg
```

`bench.py` (à venir avec le runtime C) charge le paquet, effectue 100 tokens et rapporte `Maximum resident set size`. Tant que le runtime C n'existe pas, la mesure PyTorch sera gonflée de 300–900 Mo : le dire explicitement dans les résultats.

### 6.1 Évaluation code

```bash
!pip install -q evalplus
!python -m evalplus.evaluate --model <hf-id-ou-api> --dataset humaneval
!python -m evalplus.evaluate --model <hf-id-ou-api> --dataset mbpp
```

Pour un modèle local non-HF, utiliser `blackcode/eval_code.py` (format HumanEval JSONL, exécution en sous-processus avec timeout, pass@1 et pass@k).

### 6.2 Évaluation générale

`lm-evaluation-harness` pour MMLU, HellaSwag, ARC ; `IFEval` pour le suivi d'instructions. Un modèle de 1 Md ne sera pas bon partout : l'objectif est « généraliste conversationnel correct + fort en code », pas SOTA sur MMLU.

---

## 7. Budget temps réaliste

| Étape | Durée T4 | Cumul |
|---|---:|---:|
| Pipeline nano + tests | 0,2 h | 0,2 h |
| `t4_small` 1 Md tokens | 4–6 h | 6 h |
| Ablations A1–A6 (6 × 0,5 Md) | 12–18 h | 24 h |
| Distillation étudiant 0,6 Md (30 M tokens) | 15–25 h | 49 h |
| RLVR | 10–20 h | 69 h |
| Export + évaluations | 5 h | 74 h |
| Marge (coupures, debug) | 20 h | ~94 h |

Soit **4 à 8 semaines** avec le quota Colab gratuit, ou 2–3 semaines en payant un peu de calcul. C'est le vrai coût du projet ; il n'y a pas de raccourci magique.

---

## 8. Pièges spécifiques au T4

1. **fp16, pas bf16.** Le T4 n'a pas de bf16 natif.
2. **Les logits explosent la VRAM.** Utiliser `--logit-chunk 256` (défaut) et ne jamais matérialiser `[B, T, 32k]`.
3. **Les sessions Colab se coupent.** Sauvegarder toutes les 500 pas sur Drive, toujours utiliser `--resume`.
4. **Le téléchargement HuggingFace est lent.** Mettre en cache sur Drive (`HF_HOME`).
5. **Le tokenizer doit être entraîné sur le train seulement.** Sinon, la val perd son sens.
6. **Ne pas évaluer sur les données d'entraînement.** HumanEval et MBPP doivent être tenus à l'écart, et vérifier la contamination (LiveCodeBench est plus récent).
7. **Un modèle de 0,6 Md qui atteint 15 % de pass@1 est déjà une réussite** : la moyenne des modèles de cette taille est basse. Comparer à Qwen3-0.6B et Qwen2.5-Coder-1.5B, pas à GPT-5.
