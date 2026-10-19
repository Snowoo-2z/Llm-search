# Feuille de route historique — cible 1,2 Md

> **Cette feuille de route est historique et ne décrit plus la cible active.** La décision actuelle est un modèle dense de 200–300 M paramètres, entraîné de zéro et parlant uniquement français, avec un contexte 128k et un budget RAM minimal de 4 Go. Voir [`docs/16-MODELE-FR-260M.md`](16-MODELE-FR-260M.md).
>
> Les phases et estimations ci-dessous peuvent encore éclairer les anciennes expériences 1,2 Md, mais ne doivent pas être suivies comme plan du projet courant.

---

## Vue d'ensemble

| Phase | Durée | Livrable | Métrique de sortie |
|---|---|---|---|
| **0. Pipeline** | 1 semaine | dépôt, tests, nano entraîné, planner | pipeline reproductible, 15/15 tests |
| **1. Véhicule de recherche** | 2–3 semaines | `t4_small` 51 M + ablations | perte, pass@1, KV, tok/s |
| **2. BlackCode-1** | 4–6 semaines | étudiant 0,6–1,2 Md ternaire | pass@1 vs Qwen3-0.6B |
| **3. Capacités** | 3–4 semaines | RLVR, UTB, REN | +10 pts pass@1, MEM |
| **4. Runtime + agent** | 3–4 semaines | moteur C, harnais outils | RSS < 1 Go mesuré, coût/tâche |

Durée totale réaliste : **3 à 5 mois** à temps partiel sur T4, ou 2 mois à plein temps avec un peu de calcul payant.

---

## Phase 0 — Pipeline (terminée dans ce dépôt)

**Livrables**
- `blackcode/model.py` : architecture hybride, ternaire STE, cache KV + cache convolution.
- `blackcode/train.py` : AMP fp16, perte chunkée, checkpoint/resume, gradient checkpointing.
- `blackcode/corpus.py` : BPE byte-level, déduplication, split train/val.
- `blackcode/export.py` : format `.bcpkg` 2 bits + échelles.
- `blackcode/memory.py` / `planner.py` : budget RAM et scan (taille, bits).
- `configs/nano.json`, `t4_small.json`, `t4_base.json`, `blackcode_1b.json`.
- 15 tests unitaires (cache incrémental = forward complet, export, quantification, budget).

**Résultat mesuré** : `nano` (2,16 M) entraîné sur 1,3 M tokens de code en ~4 minutes CPU ; perte 9,0 → ~3,4 ; génération et export fonctionnels.

**Critère de sortie** : un nouvel arrivant peut cloner, installer et entraîner en <10 minutes. ✅

---

## Phase 1 — Véhicule de recherche (2–3 semaines)

**Objectif** : comprendre quels leviers marchent à petite échelle, avant de dépenser du calcul sur 1 Md.

**Expériences**
1. `t4_small` (51 M) ternaire sur 1 Md tokens de code → perte de référence, pass@1 sur 50 problèmes.
2. Ablations A1–A6 (`docs/02-ARCHITECTURE.md`) à budget mémoire égal.
3. Mesure du débit et de la VRAM réels ; ajustement des configs.

**Métriques**
- perte de validation par Mo de poids ;
- pass@1 et taux de syntaxe valide ;
- cache KV par token ; tokens/s CPU (T4 pour l'entraînement, CPU pour l'inférence) ;
- stabilité (courbes de perte, gradients).

**Critère de décision (G1)**
- Si le ternaire natif 51 M est >0,3 de perte au-dessus du fp16 **à mémoire égale**, le pari « 2 bits » est fragilisé → passer à un export 4 bits et une cible de 0,8 Md.
- Si le motif 3:1 conv:attn perd >0,15 de perte face à tout-attention, augmenter la part d'attention (2:1).
- Si `loop=2` ne gagne pas ≥0,5 point, abandonner la récursion.

**Livrable** : un rapport de 5 pages avec les courbes et une config recommandée pour la phase 2.

---

## Phase 2 — BlackCode-1 (4–6 semaines)

**Objectif** : produire le premier modèle réellement déployable.

**Deux étudiants en parallèle**
- **BC-0.6B** : 24 couches, d=1024, 16 têtes / 4 KV, ffn=2816, vocab 32k → ~0,62 Md. Plus facile à entraîner sur T4, sert de preuve.
- **BC-1.2B** : la cible finale (config `blackcode_1b.json`), si le budget le permet.

**Recette**
1. Distillation fp16 du professeur (Qwen2.5-Coder-7B en 4 bits, ou 3B si le débit est trop faible) sur 20–30 M tokens de code + instructions.
2. Ternarisation avec annealing + KD sur 10–20 M tokens (TAD).
3. Récupération on-policy courte (5 M tokens).
4. Export `.bcpkg`, mesure de la RAM, évaluation.

**Métriques de sortie (G2)**
- pass@1 HumanEval+ ≥ 25 %, MBPP+ ≥ 35 % (seuils planchers ; Qwen3-0.6B est autour de 20–30 % selon les versions, Qwen2.5-Coder-1.5B autour de 40–45 % sur HumanEval de base).
- perte de validation à moins de 0,15 du fp16 équivalent.
- RAM estimée ≤ 700 Mo à 8k, mesurée ≤ 1 Go avec le runtime C.
- taux de syntaxe valide ≥ 98 % sur les générations.

**Plan B** : si le ternaire plafonne, exporter en 4 bits (poids ~0,6 Go) et réduire la cible à 0,8 Md ; ou garder 1,2 Md en 2,5–3 bits (0,45 Go).

---

## Phase 3 — Capacités (3–4 semaines)

**Objectif** : gagner en performance sans ajouter de mémoire.

1. **RLVR** (M6) : GRPO/GSPO sur MBPP/APPS avec tests, récompense combinée tests + Ruff. Attendu +5 à +15 pts.
2. **UTB** (M2) : décodage best-of-N avec tests, budget adaptatif, métrique MEM.
3. **REN** (M3) : réparateur 30 M entraîné sur bugs synthétiques + deep supervision. Attendu +5 pts à mémoire quasi constante.
4. **PKM-Code** (M4) : seulement si le temps le permet et si la bande passante CPU le justifie.

**Critère G3** : au moins deux des trois premiers leviers apportent chacun ≥3 points de pass@1, ou améliorent le coût par tâche de ≥30 %.

**Livrable** : une fiche de résultats « avant/après » avec RAM, latence, pass@1 et coût.

---

## Phase 4 — Runtime et agent (3–4 semaines)

**Objectif** : tenir la promesse des 1 Go **mesurés**, et rendre le modèle utile en pratique.

1. **Moteur C99** (`runtime/`) : mmap du `.bcpkg`, GEMV ternaire, conv1d, GQA, sampling. Parité 1e-3 avec PyTorch. Objectif <80 Mo de RSS hors poids.
2. **Harnais agent** : boucle générer → compiler → tester → réparer ; outils (lecture de fichiers, grep, exécuteur, retrieval).
3. **Routeur** (M7) : petit modèle qui choisit local / gros modèle / outil.
4. **Évaluation honnête** : HumanEval+, MBPP+, LiveCodeBench, MMLU, IFEval ; mesure RSS avec `/usr/bin/time -v` ; comparaison à Qwen3-0.6B, Qwen2.5-Coder-1.5B, LFM2-1.2B, BitNet 2B.

**Critère G4** : BlackCode résout plus de tâches **par euro** que Qwen3-0.6B en local, et tient sous 1 Go mesuré.

**Livrable** : binaire + paquet + carte de modèle + rapport de benchmark reproductible.

---

## Ce qu'on fait cette semaine

1. `git push` du dépôt et lancement du notebook Colab fourni.
2. Préparer 1 Go de code (Python/JS/Rust/Go) avec `blackcode.corpus`.
3. Lancer `t4_small` sur 1 Md tokens (4–6 h), sauvegarde Drive.
4. Pendant l'entraînement, écrire le harnais d'évaluation (`blackcode/eval_code.py`) et les 50 problèmes de test.
5. À la fin, remplir le tableau G1 et décider de la config de la phase 2.

---

## Budget et ressources

| Poste | Besoin | Coût |
|---|---|---|
| Entraînement | ~100 h T4 | Colab gratuit (plusieurs semaines) ou ~50–100 € |
| Stockage | 50 Go (corpus + checkpoints + logits) | Google Drive / GitHub Releases |
| Évaluation | CPU | gratuit |
| Modèles professeurs | Qwen2.5-Coder-7B/3B, 4 bits | gratuit (poids ouverts) |
| Temps humain | 3–5 mois à temps partiel | — |

**À ne pas faire** : louer un A100 « pour aller plus vite » avant d'avoir validé G1. Le projet peut échouer pour des raisons scientifiques (ternaire à 1,2 Md), pas seulement de confort.

---

## Risques majeurs et parades

| Risque | Signal d'alerte | Parade |
|---|---|---|
| Le ternaire à 1,2 Md déçoit | perte >0,3 au-dessus du fp16 après 2× budget | export 4 bits, cible 0,8 Md |
| La distillation est trop lente sur T4 | <10 tok/s professeur | professeur 1,5–3 B, logits précalculés |
| Le runtime C dépasse 1 Go | RSS > 900 Mo | mmap, KV int8, contexte 4k |
| Les benchmarks sont contaminés | pass@1 élevé mais revue manuelle mauvaise | LiveCodeBench, tests cachés |
| Le projet s'essouffle | pas de jalon G1 après 6 semaines | réduire à BC-0.6B, publier les résultats négatifs |

---

## Définition du succès

**Minimum viable** : un modèle de 0,6 Md ternaire, distillé, qui atteint ~30 % sur HumanEval+ et tourne sous 1 Go, avec un pipeline reproductible et des résultats honnêtes. C'est déjà rare.

**Cible** : 1,2 Md ternaire, ~40 % HumanEval+, 8k de contexte, <700 Mo de poids, agent de réparation, runtime C <80 Mo. C'est un excellent résultat pour un projet T4.

**Pari scientifique** : démontrer qu'à 1 Go, le ternaire natif à 1,2 Md bat le 4 bits à 0,6 Md et le fp16 à 0,35 Md. Si c'est vrai, c'est une contribution publiable.
