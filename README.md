# BlackCode

> **Un LLM généraliste et fort en code dans ≤ 1 Go de RAM — sans réduire sa taille.**
> Recherche, architecture et pipeline d'entraînement pour un modèle ternaire de 1,19 Md de paramètres, entraînable sur un T4 Colab.

---

## L'idée en 30 secondes

On ne fait pas un « petit modèle ». On fait un **gros modèle qui coûte peu** :

| Levier | Effet |
|---|---|
| Poids **ternaires** {-1, 0, +1} (2 bits) | 8× moins de mémoire que le fp16 |
| **Hybride 3 convolutions / 1 attention** | cache KV constant sur 75 % des couches |
| **Embeddings liés** + vocabulaire 32k | −67 M de paramètres, −64 Mo |
| **Distillation + QAT** au lieu du from scratch | 10–50× moins de tokens d'entraînement |
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

# variante 4k de contexte + runtime C de 80 Mo :
# 269 + 64 + 48 + 0,2 + 80 = 461 Mo  →  sous 0,5 Go
```

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
```

Tests : `python -m unittest discover -s tests -v` (15 tests, dont la parité entre génération incrémentale et forward complet).

---

## Ce que contient le dépôt

```
blackcode/
  config.py        configurations d'architecture et comptage exact des paramètres
  model.py         hybride conv/attention, ternaire STE, cache KV + cache convolution
  quant.py         quantification ternaire, packing 2 bits, int8 par ligne
  memory.py        budget RAM, scan (taille, bits), loi de précision
  planner.py       CLI du planificateur
  corpus.py        BPE byte-level, déduplication, split, tokenisation uint16
  data.py          chargement memmap
  train.py         AMP fp16, perte chunkée, checkpoint/resume, gradient checkpointing
  generate.py      génération avec cache
  export.py        format .bcpkg (2 bits + échelles fp16)
configs/           nano (2,2 M), t4_small (51 M), t4_base (127 M), blackcode_1b (1,19 Md)
docs/              recherche, architecture, méthodes, protocole T4, feuille de route
colab/             notebook prêt à exécuter sur T4
tests/             15 tests unitaires
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

Ouvrir `colab/BlackCode_T4.ipynb`. Le notebook :

1. vérifie le GPU et installe les dépendances ;
2. prépare un corpus de code ;
3. lance `t4_small` (51 M) avec sauvegarde sur Drive et reprise après coupure ;
4. exporte au format `.bcpkg` et affiche le budget RAM ;
5. génère un échantillon et lance les tests.

Le protocole complet (corpus, distillation, RLVR, évaluation, temps estimés) est dans [`docs/04-PROTOCOLE-T4.md`](docs/04-PROTOCOLE-T4.md).

---

## La recherche

| Document | Contenu |
|---|---|
| [`docs/00-SYNTHESE.md`](docs/00-SYNTHESE.md) | les 10 constats et les décisions en une page |
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
- **PyTorch ne tient pas sous 1 Go.** Le budget de 609 Mo suppose un runtime C. Avec Python + PyTorch, compter +300 à 900 Mo. Le moteur d'inférence est un livrable, pas un détail.
- **Les benchmarks de code sont contaminés.** Utiliser LiveCodeBench, des tests cachés et une revue manuelle.

---

## Références principales

BitNet b1.58 2B4T (`arXiv:2504.12285`), Scaling Laws for Precision (`arXiv:2411.04330`), ParetoQ (`arXiv:2502.02631`), MobileLLM (`arXiv:2402.14905`), LFM2 (`arXiv:2511.23404`), Kimi Linear (`arXiv:2510.26692`), Memory Layers at Scale (`arXiv:2412.09764`), Tiny Recursive Model (`arXiv:2510.04871`), SLM agents (`arXiv:2506.02153`).

La liste complète est dans `docs/01-ETAT-DE-L-ART.md`.

---

## Licence

Code : MIT. Les poids et données produits devront respecter les licences des corpus et des professeurs utilisés (Apache 2.0 pour Qwen, notamment).
