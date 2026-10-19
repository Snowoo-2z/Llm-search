# Exécution des prompts 1-5 — synthèse finale 18 octobre 2026

Ce document fait le point sur l'exécution des 4 prompts restants (le 5 avait déjà été traité dans docs/19) pendant la session de travail d'une heure.

## Prompt 1 — Corpus français de pré-entraînement

**Objectif** : jusqu'à 50 sources distinctes, fiche officielle, licence dataset + droits sous-jacents, audit 10-15 sources.

**Réalisé dans cette session** :

- Pipeline `blackcode/data.py` amélioré pour gérer licence par enregistrement (`record_license`), collection, open_type, etc. — indispensable pour Common Corpus (275B tokens français annoncés, à vérifier).
- Audit v2 (`scripts/audit_fr_source.py`) compte licences brutes/normalisées, collections, open_types, language_field, missing_license, hosts, PII, etc.
- Build scripts v2 avec variables env et manifest traçable.
- Cahier `docs/17` enrichi section K avec améliorations qualité.
- `docs/20` détaille protocole recommandé : audit d'abord, choix seuil qualité après distribution, filtrage licence explicite.

**Sources candidates prioritaires** (déjà listées en docs/17) :

- FineWeb2-HQ `fra_Latn` (32M docs / 483 Go annoncés, top 10% FineWeb2, ODC-By + Common Crawl conditions) — candidate prioritaire, script branché.
- PleIAs Common Corpus (métadonnées par ligne language/license/collection) — grande valeur mais filtrage par licence obligatoire.
- French-PD-Books (Gallica/BNF, OCR, conditions commerciales à clarifier).
- Wikipédia FR / Wikisource (CC BY-SA, attribution).
- DILA LEGI/JORF/DOLE (Licence Ouverte, PII noms).
- HPLT, OSCAR, CulturaX, mC4 comme pistes comparatives, pas libres par licence packaging.

**Limites** : aucun dataset HF téléchargé dans cette session locale ; tailles/licences restent déclarées par fiches, pas mesurées. Rapport externe `AUDIT_RAPPORT_PRETRAIN_FR.md` (50 sources, 15 échantillons) reste à réconcilier, pas importé tel quel.

## Prompt 2 — Livres, sciences, éducation, données publiques

**Réalisé** :

- Cahier `docs/17` section B liste Gallica/BNF, Wikisource, LEGI, HAL/thèses, Insee, etc. avec distinction domaine public vs conditions d'accès bibliothèque, Licence Ouverte vs CC BY-SA/NC-SA.
- Pipeline qualité : filtre OCR implicite via `alnum_ratio`, `boilerplate`, `répétitif`.
- `build_common_corpus_fr.sh` montre comment filtrer par licence pour corpus patrimoniaux.

**À faire** : audit réel Gallica (289k livres annoncés), HALvest, thèses — vérifier OCR, date, droits, PII.

## Prompt 3 — Conversations et SFT français

**Réalisé** :

- Cahier section B : Aya Dataset (204k annotations humaines, 1 422 FR selon Lucie), French Instruct (276k dialogues, 29% humains), Luth-SFT (571k lignes, splits hétérogènes), French Alpaca 110K (généré GPT-3.5, fiabilité faible), Légitus SFT (27k conversations juridiques, abstention/citations/RAG), Claire (37k conversations, licences hétérogènes, accès conditionné), FQuAD (QA, CC BY-NC-SA).
- Distinction humain/traduit/synthétique, licence par sous-source.
- Agentique : `blackcode/data.py` génère trajectoires françaises uniquement (`Tu es BlackCode...`), jamais inventées sans exécution si `--agentic-exec`.

**Décision** : pas de SFT automatique sur mélanges complets ; shortlist autorisable après audit split par split, séparation train/val/test.

## Prompt 4 — Code, tâches agentiques, évaluations françaises

**Réalisé** :

- Veille `docs/19` : FrenchBench, COLE (23 tâches FR), AILuminate FR, HumanEval-XL (FR→code), xCodeEval (CC BY-NC), SWE-bench multilingual, Terminal-Bench, BFCL v4, Toolathlon (108 tâches), etc. Matrice avec licence/biais/décision.
- 30 tâches françaises originales spécifiées dans `docs/19` avec objectif, outils, résultat observable, critère.
- Nouveau `benchmarks/agent_fr_v2.jsonl` : 30 tâches exécutables couvrant lecture, extraction, code, sélection outil, planification, sécurité, fraîcheur.
- `blackcode/agent.py` reste prototype lecture seule, sans shell.

**Protocole** : mesurer séparément modèle brut vs agent complet (modèle + outils + scaffold), garder fixtures déterministes, évaluer réussite stricte, choix outil, citations, violations permission, langue française.

## Prompt 5 — Veille actualité IA et informations fraîches

Déjà exécuté et consigné dans `docs/19` le 18 octobre 2026 :

- Veille datée : Gemini 4 Argon (Google, 13 oct), Muse Spark Open (Meta, 15 oct, 816B tot / 42B actifs, licence Muse), DeepSeek V4.1 Flash-1017 (17 oct), Mistral Large 4 (6 oct, 1T tot / 52B actifs, poids à venir), GPT-6 (OpenAI, 7 oct), Claude Sonnet 5.5 (Anthropic, 28 sept).
- Conclusion : modèle entraîné ne connaît pas automatiquement les nouvelles ; RAG/recherche avec citations datées ou réentraînement sur nouveau corpus. Architecture proposée en 8 étapes (filtrage sources officielles, récupération, réponse fondée, sécurité injection, mise à jour/retrait, évaluation fraîcheur).
- Sources primaires citées, distinction faits vérifiés / claims fournisseurs / recommandations.

## Améliorations qualité transverses (session 1h)

Voir `docs/20-QUALITE-DATASET-FR.md` pour détails :

- Qualité : `trop d'urls`, `ponctuation répétée`, `boilerplate`, `répétitif` prose.
- Licences : `normalize_licence_label`, `PERMISSIVE` restreint, `require-hf-license`, `allowed-hf-license`, manifest.
- Audit v2 : licences, collections, open_types, language_field.
- Build v2 : variables env, manifest, stats enrichies.
- Tests : 24 tests data OK, 52 tests totaux (6 erreurs torch attendues).

## Livrables de cette session

- `blackcode/data.py` amélioré
- `scripts/audit_fr_source.py` v2
- `scripts/build_fr_corpus.sh` v2
- `scripts/build_common_corpus_fr.sh` nouveau
- `benchmarks/agent_fr_v2.jsonl` 30 tâches
- `docs/20-QUALITE-DATASET-FR.md` nouveau
- `docs/21-EXECUTION-PROMPTS-FR.md` (ce fichier)
- `docs/17-CAHIER-SOURCES-FR.md` section K
- `docs/19` déjà existant (veille + 30 tâches)
- Tests verts

Aucun corpus HF téléchargé localement dans cette session ; aucun entraînement lancé. Les chiffres de taille/licence restent à vérifier sur fiches officielles et par audit local sur Colab/Kaggle.

