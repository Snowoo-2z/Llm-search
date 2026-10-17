# Résultats de la démonstration (CPU)

> But : prouver que le pipeline complet fonctionne, pas produire un modèle utile.
> Environnement : 2 vCPU Xeon 2,6 GHz, 3,8 Go de RAM, **aucun GPU**, PyTorch 2.14 CPU.

---

## 1. Corpus

| | |
|---|---|
| Source | `TheAlgorithms/Python` (GitHub) |
| Fichiers candidats | 1 366 |
| Après déduplication | 1 338 train / 27 val |
| Tokens train | **1 317 248** |
| Tokens val | 24 465 |
| Tokenizer | BPE byte-level, 8 192 tokens, 532 Ko |
| Temps de préparation | 2,6 s |

## 2. Modèle et entraînement

| | |
|---|---|
| Config | `configs/nano.json` |
| Paramètres | 2 158 336 (1,11 M hors embeddings) |
| Couches | 6 : 4 convolutions + 2 attentions (motif 3:1) |
| Poids | ternaires, échelle absmean par ligne |
| Embeddings | liés, 8 192 × 128 |
| Séquence | 128 |
| Batch | 8 (1 024 tokens/pas) |
| Pas | 1 500 (1,54 M tokens, ~1,2 époque) |
| Learning rate | 1,5e-3, warmup 50, cosinus |
| Durée | **4 min 48** |
| Débit | ~5 500 tokens/s |

### Courbe de perte

| Pas | Perte train (batch) | Perte train (eval) | Perte val |
|---:|---:|---:|---:|
| 0 | 9,037 | — | — |
| 250 | 4,540 | 4,838 | 4,835 |
| 500 | 4,378 | 4,199 | 4,264 |
| 750 | 3,317 | 3,743 | 4,209 |
| 1 000 | 3,436 | 3,831 | 3,895 |
| 1 250 | 3,696 | 3,559 | 3,873 |
| 1 500 | — | 3,392 | **3,644** |

La perte de validation descend de 9,0 à 3,64. Le modèle a mémorisé une partie du corpus (1,2 époque) ; ce n'est pas une mesure de généralisation.

## 3. Export `.bcpkg`

| | |
|---|---|
| Fichier | 1,4 Mo |
| Données packées | 1,30 Mo |
| Poids ternaires | 0,28 Mo |
| Embedding int8 | 1,02 Mo |
| Normes / convolutions | 0,01 Mo |
| fp16 brut équivalent | 4,12 Mo (compression ×3,2) |
| Bits effectifs | **5,05 bits/paramètre** |

À cette échelle, l'embedding (1,02 Mo) domine les poids ternaires (0,28 Mo) : c'est pourquoi les bits effectifs sont à 5,05 et non à 2,3. Sur `blackcode_1b`, l'embedding ne représente que 5,7 % des paramètres, et les bits effectifs tombent vers 2,3.

## 4. Mesure de RAM (PyTorch)

```
RSS après import PyTorch : 499,7 Mo
RSS après chargement      : 524,0 Mo  (+24,2)
RSS pic (génération)      : 537,3 Mo  (+37,5)
Vitesse                   : 143,8 tok/s (CPU, modèle 2,16 M)
```

**Le résultat clé n'est pas la taille du modèle, c'est le coût du runtime.** PyTorch consomme ~500 Mo avant même de charger les poids. Le moteur C `runtime/bcrun.c` (phase 4, livré) ramène ce coût à **1,6 Mo** : le modèle de 1,19 Md tient alors en **431 Mo à 8k** de contexte. Voir `docs/08-RAM-MESUREE.md`.

## 5. Tests

```
Ran 20 tests in 0.9s
OK
```

Couverture :

- parité entre génération incrémentale (cache KV + cache convolution) et forward complet ;
- comptage exact des paramètres vs formule ;
- gradient non nul à travers le STE ternaire ;
- partage de paramètres avec `loop` ;
- packing/dépacking 2 bits ;
- budget mémoire : le modèle 1,19 Md tient sous 1 Go ;
- export : magic, header, nombre de tenseurs, bits effectifs ;
- harnais d'évaluation : exécution, timeout, pass@k, stop sequences.

## 6. Ce que cette démo prouve

- Le corpus, le tokenizer, l'architecture, l'entraînement, la génération, l'export et le budget mémoire **s'articulent sans erreur**.
- Le ternaire STE apprend (la perte baisse) et se sauvegarde dans un format 2 bits.
- Le cache incrémental est **numériquement identique** au forward complet, y compris avec les convolutions.
- Le coût du runtime Python/PyTorch est mesuré, et il est incompatible avec une promesse « 1 Go » sans moteur dédié.

## 7. Ce que cette démo ne prouve pas

- Qu'un modèle de 2,16 M code. Il produit des fragments syntaxiquement plausibles mais incorrects.
- Que le ternaire à 1,2 Md atteint la parité avec le fp16. C'est l'hypothèse de la phase 2.
- Que le runtime C tiendra en 80 Mo. Il reste à écrire.
- Que la distillation sur T4 tient dans le quota Colab. Les estimations sont dans `docs/04-PROTOCOLE-T4.md`, mais elles n'ont pas encore été mesurées sur T4.

## 8. Reproductibilité

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./scripts/smoke.sh
```

Les données, checkpoints et artefacts sont ignorés par Git (`.gitignore`) ; seuls le code et les documents sont versionnés.
