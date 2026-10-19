# Le dataset est-il prêt ? — état des lieux et marche à suivre

**Réponse courte : non, pas pour entraîner le modèle final.** Un corpus de démarrage réel existe désormais (86,7 M tokens), suffisant pour valider le pipeline et faire tourner `t4_small` une fois. Il manque un facteur 10 à 100 pour produire le modèle déployable.

---

## 1. Ce qui existe maintenant

Construit dans ce dépôt avec `scripts/fetch_corpus.py` (sources accessibles publiquement, sans compte HuggingFace) :

| | |
|---|---|
| Sources | 97 paquets PyPI (sdists) + 7 dépôts GitHub |
| Volume brut | **904 Mo** de code |
| Fichiers candidats | 23 376 |
| Après déduplication | 22 686 (3 % de doublons éliminés) |
| **Tokens train** | **86,74 M** |
| Tokens validation | 1,61 M (1,83 %) |
| Tokenizer | BPE byte-level, 32 000 tokens |
| Longueur médiane d'un document | 1 742 tokens (p10 363, p90 10 792) |
| Temps de construction | ~44 s de téléchargement + ~2 min de tokenisation |

Fichiers : `data/code/train.bin`, `val.bin` (uint16), `tokenizer.json`, `meta.json`, `data/raw/manifest.json` (licences et provenance).

### Ce que ce corpus permet

| Usage | Faisable ? | Temps T4 |
|---|---|---|
| Valider le pipeline de bout en bout | oui | quelques minutes |
| Ablations sur `nano` (2 M) | oui | minutes |
| Une passe sur `t4_small` (51 M) | oui, mais sous-entraîné | **~22 min** |
| Entraîner `t4_small` correctement (1 Md tokens) | non, il manque 92 % | 4,3 h avec le bon corpus |
| Entraîner BC-0.6B from scratch (12 Md tokens) | non, il manque 99,3 % | ~6 jours |
| Distiller BC-0.6B (20–30 M tokens) | **oui** | ~4–6 h |

---

## 2. Ce qu'il faut, par objectif

| Objectif | Tokens nécessaires | Pourquoi |
|---|---:|---|
| Pipeline + ablations | 50–100 M | vérifier que tout apprend |
| `t4_small` correct | 1 Md (20 tok/param) | Chinchilla + surentraînement raisonnable |
| BC-0.6B from scratch | 12 Md | 20 tok/param |
| BC-1.2B from scratch | 24 Md | 20 tok/param |
| **BC-0.6B distillé** | **20–30 M** | le professeur remplace les tokens |
| BC-1.2B distillé | 30–50 M | idem |

C'est la raison pour laquelle le projet ne pré-entraîne pas 1,2 Md : il n'existe pas de corpus raisonnable à cette échelle pour un T4. La distillation est le chemin.

---

## 3. Comment passer de 87 M à 1 Md de tokens

Le rendement mesuré est de **~96 M tokens par Go de code brut** (904 Mo → 86,7 M). Pour 1 Md, il faut donc **~10 Go de sources**. Trois options, de la plus simple à la plus efficace.

### Option A — élargir PyPI + GitHub (simple, sans compte)

```bash
# sur Colab, disque ~100 Go disponibles
python scripts/fetch_corpus.py --out data/raw --pypi popular --github big --max-mb 9000
python -m blackcode.corpus prepare --src data/raw --out data/code --vocab 32000 \
    --suffixes .py,.pyi,.md
```

Rendement attendu : ~850 M tokens. Durée : 10–20 min de téléchargement + ~20 min de tokenisation sur Colab (le réseau y est plus rapide qu'ici). Limite : **Python uniquement**, et beaucoup de bibliothèques sont des wrappers plutôt que du code dense.

### Option B — datasets HuggingFace (recommandé, multilingue)

```bash
pip install datasets
python scripts/fetch_corpus.py --out data/raw --pypi quick \
    --hf bigcode/the-stack-smol --hf-lang Python --max-mb 4000
# autres langages : relancer avec --hf-lang JavaScript / Rust / Go
python -m blackcode.corpus prepare --src data/raw --out data/code \
    --vocab 32000 --suffixes .py,.js,.ts,.rs,.go,.md
```

`the-stack-smol` est petit (~1 Go, licence permissive par langage) : idéal pour un premier vrai run. Pour 10 Go+, `bigcode/starcoderdata` ou `the-stack-v2` (accès conditionnel, opt-out). Attention : `the-stack-v2` demande de télécharger les fichiers un par un via S3 ; `datasets` le fait mais c'est long.

### Option C — ajouter d'autres langages à la main

```bash
python scripts/fetch_corpus.py --out data/raw --local /content/mon_code_js
```
ou cloner des dépôts JS/TS/Rust/Go dans `data/raw/github/`. Un modèle « fort en code » uniquement en Python est un choix à assumer.

---

## 4. Avant de lancer : la checklist

1. **Compter les tokens, pas les octets.** `python scripts/corpus_report.py --data data/code --config configs/t4_small.json` affiche le nombre d'époques et le temps T4. Sans ce chiffre, on entraîne à l'aveugle.
2. **Vérifier l'absence de fuite train/val.** Le split se fait par fichier, pas par ligne : OK. Ne jamais mettre HumanEval ou MBPP dans `data/raw`.
3. **Vérifier la contamination des benchmarks.** Les dépôts GitHub contiennent parfois des solutions de HumanEval. Pour une évaluation honnête, tester sur LiveCodeBench (plus récent) et sur des tests cachés.
4. **Regarder les licences.** Certains paquets PyPI sont GPL ; le manifeste enregistre ce qui est détectable (beaucoup de `?`). Pour publier des poids, filtrer vers MIT / Apache-2.0 / BSD.
5. **Contrôler la qualité.** Taux de déduplication (ici 3 %), longueur médiane (ici 1 742 tokens), proportion de fichiers générés. Un corpus de 1 Md tokens bruité vaut moins que 200 M propres : `blackcode.prune` existe pour ça.
6. **Sauvegarder `tokenizer.json` et `meta.json`** avec les poids : sans eux, le modèle est illisible.

---

## 5. Ce qui est prêt, ce qui ne l'est pas

| Élément | État |
|---|---|
| Script de collecte (PyPI, GitHub, HF, local) | ✅ `scripts/fetch_corpus.py` |
| Tokenizer 32k entraîné | ✅ `data/code/tokenizer.json` |
| Corpus de 87 M tokens | ✅ `data/code/` (non versionné, se régénère en 3 min) |
| Audit du corpus | ✅ `scripts/corpus_report.py` |
| Dataset de 1 Md tokens | ❌ à générer sur Colab (option A ou B) |
| Corpus multilingue | ❌ Python seulement pour l'instant |
| Filtrage qualité / licences | ⚠️ `blackcode.prune` existe, à exécuter |
| Filtrage de contamination | ❌ à faire avant toute évaluation publique |

---

## 6. La commande à lancer sur Colab aujourd'hui

```bash
# 1. données (10-20 min)
python scripts/fetch_corpus.py --out data/raw --pypi popular --github big --max-mb 9000
python -m blackcode.corpus prepare --src data/raw --out data/code --vocab 32000 \
    --suffixes .py,.pyi,.md

# 2. vérifier
python scripts/corpus_report.py --data data/code --config configs/t4_small.json

# 3. entraîner (reprise auto, ~4 h pour 1 Md)
OUT=/content/drive/MyDrive/blackcode/runs/t4_small ./scripts/train_t4.sh
```

Si `corpus_report` annonce moins de 0,3 époque : soit compléter le corpus, soit réduire la cible, soit passer directement à la distillation. Ne pas lancer 4 h d'entraînement sur 87 M tokens en espérant un modèle utile : le résultat sera un modèle qui récite.
