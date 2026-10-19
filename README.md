# BlackCode

> **BlackCode : modèle entraîné de zéro, uniquement en français, avec une cible de 200–300 millions de paramètres.**
> La configuration retenue compte exactement 258,6 M de paramètres; 4 Go de RAM minimum et 128k de contexte restent les objectifs de déploiement à valider.

---

## L'idée en 30 secondes

> **Nouvelle cible active :** pré-entraîner de zéro un modèle dense francophone de **258,6 M de paramètres**. La configuration 1,2 Md et les tableaux 1 Go ci-dessous sont historiques, pas la cible actuelle. Le contexte 128k et l'inférence sur 4 Go sont des objectifs non validés. Le prototype Qwen précédent est une référence facultative, pas la base du modèle BlackCode. Voir [`docs/16-MODELE-FR-260M.md`](docs/16-MODELE-FR-260M.md).

On ne fait pas un « petit modèle ».  On fait un **gros modèle qui coûte peu** :

| Levier | Effet |
|---|---|
| Poids **ternaires** {-1, 0, +1} (2 bits) | 8× moins de mémoire que le fp16 |
| **Hybride 3 convolutions / 1 attention** | cache KV constant sur 75 % des couches |
| **Embeddings liés** + vocabulaire 49 152 | −100 M de paramètres, −100 Mo |
| **Distillation + QAT** au lieu du from scratch | **×390** sur le budget de calcul |
| **Tests unitaires** à l'inférence | un vérificateur gratuit qui multiplie la capacité |
| **Runtime C** dédié | tient réellement sous 1 Go (PyTorch, non) |

Résultat calculé pour la cible `blackcode_1b` à 8k de contexte :

```
1 187 M paramètres (1 120 M hors embeddings, 24 couches dont 6 attention)
poids ternaires    269 Mo
embeddings int8     64 Mo
cache KV 8k         96 Mo
runtime C estimé   180 Mo
-----------------------------
total              609 Mo      marge : 415 Mo sous 1 Go
```

**Mesuré, pas estimé** (runtime C, `./runtime/bcrun --mem-report`) :

| Contexte | RSS pic | Marge / 1 Go |
|---:|---:|---:|
| 2 048 | 356 Mo | 668 Mo |
| 4 096 | **385 Mo** | 639 Mo |
| 8 192 | **431 Mo** | 593 Mo |
| 16 384 | 536 Mo | 488 Mo |

Le runtime seul pèse **1,6 Mo**. PyTorch, lui, consomme ~500 Mo avant de charger quoi que ce soit : avec lui, la promesse « 1 Go » est intenable. Détail et protocole : [`docs/08-RAM-MESUREE.md`](docs/08-RAM-MESUREE.md).

Débit du moteur C pour ce même modèle (2 cœurs, 8k de contexte) : **9,2 tok/s**, après déroulage de la boucle ternaire et `-ffast-math` (contre 2,6 tok/s pour la première version). Le moteur est limité par la bande passante mémoire — 267 Mo de poids lus par token — pas par le calcul flottant.

---

## Démarrage rapide (CPU, ~5 minutes)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Un petit corpus de code (exemple : TheAlgorithms/Python)
git clone --depth 1 https://github.com/TheAlgorithms/Python /tmp/corpus
python -m blackcode.corpus prepare --src /tmp/corpus --out data/algorithms \
    --vocab 8192 --suffixes .py

# 2. Entraîner le modèle de démonstration (2,2 M paramètres)
python -m blackcode.train --config configs/nano.json --data data/algorithms \
    --out runs/nano --batch-size 8 --seq-len 128 --max-steps 1500 \
    --lr 1.5e-3 --eval-interval 250 --save-interval 250

# 3. Générer, exporter, mesurer
python -m blackcode.generate --ckpt runs/nano/best.pt \
    --tokenizer data/algorithms/tokenizer.json --prompt "def add(a, b):"
python -m blackcode.export --ckpt runs/nano/best.pt --out artifacts/nano.bcpkg --context 256
python -m blackcode.planner --config configs/blackcode_1b.json --context 8192

# 4. Mesurer la RAM réelle du modèle cible (sans l'entraîner)
make -C runtime
python scripts/make_random_bcpkg.py --config configs/blackcode_1b.json \
    --out /tmp/rss.bcpkg --context 8192
./runtime/bcrun --pkg /tmp/rss.bcpkg --ids 1 --max-new 3 --context 8192 --mem-report
```

Tests : `python -m unittest discover -s tests -v` (38 tests : parité génération incrémentale / forward complet, quantification, export, budget, optimiseurs, schedules, calculateur de temps).

---

## Ce que contient le dépôt

```
blackcode/
  config.py        configurations d'architecture et comptage exact des paramètres
  model.py         hybride conv/attention, ternaire STE, cache KV + cache convolution
  quant.py         quantification ternaire, packing 2 bits, int8 par ligne
  memory.py        budget RAM, scan (taille, bits), loi de précision
  planner.py       CLI du planificateur
  data.py          pipeline v2 : qualité, dédup, FIM, traces agentiques françaises et filtre français
  corpus.py        pipeline v1 (BPE maison, split, tokenisation uint16)
  distill.py       distillation professeur → élève (top-k KL, professeur 4 bits)
  train.py         AMP fp16, perte chunkée, checkpoint/resume, gradient checkpointing
  generate.py      génération avec cache
  agent.py         assistant français, boucle d'outils bornée et outils sûrs en lecture seule
  export.py        format .bcpkg (2 bits + échelles fp16)
runtime/
  bcrun.c          moteur d'inférence C99 (mmap, GEMV ternaire, GQA, mesure RSS)
  optim.py         Muon, AdEMAMix, WSD, weighted decay, mémoire d'entraînement
  timing.py        calculateur FLOPs / heures / MFU / mémoire
  prune.py         élagage de données par bande de perte
configs/           blackcode_fr_260m (258,6 M), variante 128k YaRN, nano et configs historiques
scripts/           scripts de corpus historiques + build_fr_corpus.sh, audit_fr_source.py
benchmarks/        tâches agentiques françaises (jeu de départ, évaluation)
docs/              recherche historique; plan FR-260M, cahier de sources et prompts d'agents
colab/             notebook prêt à exécuter sur T4
tests/             65 tests unitaires
```

---

## Résultat de la démonstration

Le pipeline complet a été exécuté dans ce dépôt, sur CPU (2 cœurs), avec un corpus de 1,32 M tokens de Python :

| | |
|---|---|
| Modèle | 2,16 M paramètres, 6 couches (4 conv + 2 attention), ternaire |
| Entraînement | 1 500 pas, 1,54 M tokens, **4 min 48** sur 2 cœurs CPU |
| Perte | 9,03 → **3,64** (validation) |
| Débit | ~5 500 tokens/s en entraînement, ~144 tok/s en génération |
| Export | 1,3 Mo en `.bcpkg` (5,05 bits/paramètre effectifs) |
| RAM PyTorch mesurée | **537 Mo**, dont ~500 Mo de runtime avant même le modèle |

Le chiffre de 5,05 bits/paramètre peut surprendre : à 2,16 M de paramètres, l'embedding 8192×128 en int8 (1,0 Mo) domine les 0,28 Mo de poids ternaires. À l'échelle de 1,19 Md, l'embedding ne représente que 5,7 % et les bits effectifs tombent vers 2,3.

La mesure de RAM est le résultat le plus important de cette démo : **PyTorch consomme ~500 Mo avant de charger le modèle**. Un modèle de 1,19 Md (430 Mo de poids + KV) frôlerait donc la limite de 1 Go avec PyTorch, alors qu'un moteur C dédié viserait ~510 Mo au total. C'est pourquoi le runtime est un livrable de première classe.

Ce modèle n'est **pas** utile pour coder : c'est un test de bout en bout. Il prouve que le corpus, le tokenizer, l'architecture, l'entraînement, la génération et l'export fonctionnent ensemble.

---

## Sur Colab T4

**Le plan d'entraînement complet, chiffré, est dans [`docs/11-PLAN-ENTRAINEMENT.md`](docs/11-PLAN-ENTRAINEMENT.md).** Pour le stockage gratuit et la reprise, suivre [`docs/12-STOCKAGE-SAUVEGARDES.md`](docs/12-STOCKAGE-SAUVEGARDES.md). Pour Kaggle et ses deux T4, voir le smoke test DDP et le lancement multi-GPU dans [`docs/13-MULTI-GPU-KAGGLE.md`](docs/13-MULTI-GPU-KAGGLE.md).
Il répond à une question simple : un T4 ne peut pas pré-entraîner 0,6–1,2 Md
depuis zéro (26 à 90 jours). La distillation le ramène à **une session de 4 h**.

```bash
# 1. préparer la session
bash scripts/colab_setup.sh

# 2. construire le corpus (tokenizer StarCoder2, FIM, agentique exécuté)
python -m blackcode.data build --src data/raw --out data/v2 \
  --tokenizer hf:bigcode/starcoder2-3b --seq-len 8192 \
  --fim-rate 0.5 --agentic-frac 0.10 --agentic-exec

# 3. connaître le budget exact avant de dépenser du GPU
python scripts/train_plan.py --config configs/blackcode_0.6b.json --stage distill

# 4. distiller (reprise automatique après coupure)
python -m blackcode.distill --teacher bigcode/starcoder2-3b --load-4bit \
  --config configs/blackcode_0.6b.json --data data/v2 --out runs/distill_06b \
  --seq-len 2048 --batch-size 2 --grad-accum 8 --max-steps 916 \
  --topk 32 --alpha 0.5 --temperature 2.0 --optim adafactor --lr 3e-4 \
  --schedule wsd --checkpoint-policy selective --resume
```

Ouvrir `colab/BlackCode_T4.ipynb`. Le notebook :

1. vérifie le GPU et installe les dépendances ;
2. prépare un corpus de code ;
3. lance `t4_small` (51 M) avec sauvegarde sur Drive et reprise après coupure ;
4. exporte au format `.bcpkg` et affiche le budget RAM ;
5. génère un échantillon et lance les tests.

Le protocole complet (corpus, distillation, RLVR, évaluation, temps estimés) est dans [`docs/04-PROTOCOLE-T4.md`](docs/04-PROTOCOLE-T4.md).

---

## Réduire le temps d'entraînement (sans réduire la taille)

Le temps vaut **tokens nécessaires × temps par token ÷ parallélisme**. Les leviers implémentés :

```bash
# débit : +20 à 40 % sur T4, et rend 1,2 Md possible en mémoire
python -m blackcode.train ... --optim adafactor --checkpoint-policy selective     --compile --schedule wsd --mirror-dir /content/drive/MyDrive/blackcode/runs/t4

# efficacité par token : Muon (1,3–2× selon la littérature)
python -m blackcode.train ... --optim muon --muon-lr 0.02

# moins de tokens : élagage des données par bande de perte
python -m blackcode.prune --ckpt runs/t4_small/best.pt     --tokenizer data/code/tokenizer.json --src /content/corpus     --out /content/corpus_pruned --lo 0.05 --hi 0.85

# savoir combien de temps et de mémoire il faut, avant de lancer
python -m blackcode.timing --config configs/blackcode_1b.json     --tokens 1e9 --seq-len 4096 --memory
```

Sur un T4, **AdamW fp32 ne permet pas d'entraîner 1,2 Md** (19,6 Go de poids + gradients + états pour 16 Go de VRAM). Adafactor (11,5 Go) et AdamW 8-bit (12,8 Go) le rendent possible. Le détail, les mesures CPU et le plan d'action sont dans [`docs/07-TEMPS-D-ENTRAINEMENT.md`](docs/07-TEMPS-D-ENTRAINEMENT.md).

---

## La recherche

| Document | Contenu |
|---|---|
| [`docs/00-SYNTHESE.md`](docs/00-SYNTHESE.md) | les 10 constats et les décisions en une page |
| [`docs/07-TEMPS-D-ENTRAINEMENT.md`](docs/07-TEMPS-D-ENTRAINEMENT.md) | réduire le temps à taille fixée : leviers, chiffres, plan |
| [`docs/08-RAM-MESUREE.md`](docs/08-RAM-MESUREE.md) | la RAM réellement mesurée : 431 Mo à 8k, parité C/PyTorch |
| [`docs/09-DATASET.md`](docs/09-DATASET.md) | le dataset est-il prêt ? 87 M tokens construits, comment aller à 1 Md |
| [`docs/01-ETAT-DE-L-ART.md`](docs/01-ETAT-DE-L-ART.md) | revue : BitNet, ParetoQ, lois de précision, LFM2, Kimi Linear, MobileLLM, memory layers, TRM, RLVR, SLM agents |
| [`docs/02-ARCHITECTURE.md`](docs/02-ARCHITECTURE.md) | architecture, budget détaillé, ablations, runtime |
| [`docs/03-METHODES-NOUVELLES.md`](docs/03-METHODES-NOUVELLES.md) | TAD, UTB, REN, PKM-Code, LDD, MAP, routeur |
| [`docs/04-PROTOCOLE-T4.md`](docs/04-PROTOCOLE-T4.md) | commandes, corpus, distillation, RLVR, pièges |
| [`docs/05-FEUILLE-DE-ROUTE.md`](docs/05-FEUILLE-DE-ROUTE.md) | phases, jalons, critères d'arrêt, budget |

### Les trois paris

1. **À mémoire fixe, 2 bits + plus de paramètres bat 4 bits + moins de paramètres.** Le planificateur prédit 2,6 Md à 2 bits dans 1 Go, contre 336 M à 8 bits. À valider empiriquement — c'est l'expérience MAP.
2. **Un modèle de 1,2 Md peut être distillé vers le ternaire sans s'effondrer**, si l'on utilise annealing de quantification + distillation on-policy (TAD).
3. **Le calcul à l'inférence (tests, réparation) remplace une partie des paramètres.** Un 0,6 Md + 16 candidats + tests peut battre un modèle bien plus gros en pass@1.

---

## Honnêteté intellectuelle

- **Pas de « Claude dans 1 Go ».** Un 1,2 Md ternaire ne raisonnera pas comme un 100 Md. Il peut être excellent sur des tâches de code courtes et répétitives, augmenté par les tests, la réparation et les outils.
- **Pas de pré-entraînement 1 Md sur T4.** Le calcul manque d'un facteur ~100. La voie est distillation + ternarisation + RLVR.
- **Le ternaire natif n'a pas prouvé sa parité à 1,2 Md.** BitNet la montre à partir de ~3 Md. C'est l'hypothèse centrale, et elle peut échouer. Le plan B (export 4 bits, cible 0,8 Md) est documenté.
- **PyTorch ne tient pas sous 1 Go** (~500 Mo de runtime avant les poids). Avec le moteur C de ce dépôt, c'est mesuré : **431 Mo à 8k**. En revanche, la vitesse du moteur C reste faible (2,6 tok/s sur 2 cœurs) : c'est un chantier d'optimisation, pas une promesse tenue.
- **Les benchmarks de code sont contaminés.** Utiliser LiveCodeBench, des tests cachés et une revue manuelle.

---

## Références principales

BitNet b1.58 2B4T (`arXiv:2504.12285`), Scaling Laws for Precision (`arXiv:2411.04330`), ParetoQ (`arXiv:2502.02631`), MobileLLM (`arXiv:2402.14905`), LFM2 (`arXiv:2511.23404`), Kimi Linear (`arXiv:2510.26692`), Memory Layers at Scale (`arXiv:2412.09764`), Tiny Recursive Model (`arXiv:2510.04871`), SLM agents (`arXiv:2506.02153`).

La liste complète est dans `docs/01-ETAT-DE-L-ART.md`.

---

## Licence

Code : MIT. Les poids et données produits devront respecter les licences des corpus et des professeurs utilisés (Apache 2.0 pour Qwen, notamment).

---

## Modèle français 260M — nouveautés 18 octobre 2026

Voir `docs/16-MODELE-FR-260M.md` pour la config 258,6 M paramètres dense, vocab 24 576 français, contexte 8k (expérimental 128k).

**Pipeline qualité** :

- `blackcode/data.py` : filtre PII/secrets (email, tel FR, NIR, IBAN, CB), boilerplate, urls, ponctuation répétée, répétitif, minifié, python invalide + heuristique française sans lingua (stopwords + accents) + score qualité 0-1 + licence par enregistrement (`record_license`) + manifest traçable.
- `scripts/audit_fr_source.py` v2 : taux français, licences brutes/normalisées, collections, open_types, hosts, PII, reasons.
- `scripts/build_fr_corpus.sh` v2 et `scripts/build_common_corpus_fr.sh` : build avec allowlist licences.
- `scripts/build_sft_fr.py` : build SFT français local/HF.
- `scripts/validate_agent_tasks.py` + `scripts/eval_agent_benchmark.py` + `scripts/eval_french_purity.py` : validation benchmarks.

**Benchmarks** :

- `benchmarks/agent_fr.jsonl` (13) + `benchmarks/agent_fr_v2.jsonl` (30 tâches originales) couvrant lecture, extraction, code, outil, planification, sécurité, fraîcheur.

**Docs** :

- `docs/17-CAHIER-SOURCES-FR.md` : registre sources avec sections K/L améliorations.
- `docs/18-PROMPTS-AGENTS-DATA-FR.md` : prompts 1-4 + exécution.
- `docs/19-VEILLE-ACTUALITE-IA-ET-EVALUATION-AGENTS-FR.md` : veille 18 oct 2026 (Gemini 4 Argon, Muse Spark Open, DeepSeek Flash, Mistral Large 4, GPT-6) + architecture RAG + 30 tâches.
- `docs/20-QUALITE-DATASET-FR.md` : détails qualité.
- `docs/21-EXECUTION-PROMPTS-FR.md` : synthèse prompts.
- `docs/22-PIPELINE-COMPLET-FR.md` : pipeline end-to-end.
- `docs/23-SFT-FR.md` : préparation SFT.

**Tests** : 33 tests data OK (`test_data.py` avec QualityV2 + QualityScore).

**Commandes rapides** :

```bash
python scripts/audit_fr_source.py --dataset epfml/FineWeb2-HQ:fra_Latn --limit 2000 --samples-out data/audits/fineweb2_hq.jsonl
MAX_MB=4000 MIN_QUALITY_SCORE=0.6 OUT=data/fr_260m bash scripts/build_fr_corpus.sh
.venv/bin/python -m blackcode.data build --src data/fr_seed --out /tmp/test_fr --vocab 1024 --seq-len 128 --max-mb 1 --french-only
.venv/bin/python scripts/validate_agent_tasks.py
.venv/bin/python scripts/eval_agent_benchmark.py
```

Aucun corpus HF téléchargé dans ce dépôt lors des sessions locales ; chiffres tailles/licences à vérifier sur fiches officielles et par audit Colab/Kaggle.

