# Lots de recherche à déléguer à des agents Arena

Ce document contient quatre prompts indépendants. Ils peuvent être envoyés à quatre agents différents. Ensemble, ils visent **jusqu'à 200 sources candidates** (environ 50 par agent), mais seules les sources munies d'une fiche et d'un audit crédibles entreront au cahier comme vérifiées. Une recherche rapide n'est pas une validation.

Demander à chaque agent de fournir :

- environ **50 sources uniques** dans son périmètre; si la quantité n'est pas atteignable sans sources douteuses, fournir moins et le dire;
- pour chacune, l'URL de la fiche officielle, l'identifiant exact, la version/configuration, la langue, la licence du dataset et la licence/provenance des œuvres sous-jacentes;
- 10 à 15 sources priorisées avec une vraie lecture de fiche et d'échantillons; distinguer ces sources des simples résultats de recherche;
- les risques PII/mineurs, contenu sensible, conditions de retrait, données traduites ou synthétiques et fuites de benchmark;
- verdict argumenté : **pré-entraînement**, **SFT**, **évaluation seulement**, **en attente**, ou **à exclure**;
- aucune donnée brute, conversation privée ou contenu sous accès non autorisé ne doit être collecté ou recopié.

## Prompt 1 — Corpus français de pré-entraînement

```text
Tu es chargé d'auditer des sources de texte pour entraîner de zéro un petit modèle de langage (200–300 M paramètres) qui devra répondre uniquement en français. Le corpus Discord privé est hors périmètre et ne doit jamais être demandé ni utilisé.

Recherche jusqu'à 50 sources distinctes de pré-entraînement en français : corpus Web français filtrés, encyclopédies, archives, textes publics, livres avec droits établis, articles scientifiques, documents éducatifs, administrations publiques. Recherche prioritairement des sources avec provenance et sous-licences traçables. N'inclus pas les modèles préentraînés ni les datasets d'instructions sauf s'ils ont aussi un corpus pré-entraînement distinct.

Pour chaque source, donne une ligne JSONL contenant exactement ces clés :
{"dataset":"","dataset_id":"","version_or_date":"","official_card_url":"","upstream_sources":"","language_claim":"","french_share_or_subset":"","size_docs_tokens_bytes":"","license_dataset":"","underlying_content_rights":"","access_terms":"","pii_minor_sensitive_risks":"","quality_and_filtering_evidence":"","known_failures":"","sample_inspected":"yes/no; how many and what fields","recommendation":"pretrain / limited / hold / reject","reason":"","confidence":"high/medium/low"}

Pour les 10–15 meilleures sources, lis la fiche officielle et inspecte une petite sélection représentative de textes sans republier leur contenu. Note les langues réellement observées, erreurs OCR, boilerplate, répétitions, dates, noms de personnes, courriels/téléphones, sujets sensibles, pages de faible qualité et licences incertaines. Si tu n'as pas ouvert la fiche ou inspecté un échantillon, indique clairement « non vérifié »; ne déduis pas une licence du seul nom du dataset.

Termine par : (1) shortlist top 10, (2) sources à exclure et pourquoi, (3) mélange recommandé avec quotas par domaine/langue/date, (4) étapes d'audit reproductibles et URL exactes. Cite des liens directs. Ne prétends pas qu'une licence du conteneur rend libres toutes les œuvres sources.
```

## Prompt 2 — Livres, sciences, éducation et données publiques

```text
Recherche des ressources textuelles en français de bonne qualité pour un modèle de langage entraîné de zéro (200–300 M paramètres), français uniquement. N'utilise aucune conversation Discord ou donnée privée.

Ton périmètre : livres et patrimoine, Gallica/BNF, Wikisource/Wikimedia, textes parlementaires et gouvernementaux, justice/droit, Insee/données publiques, HAL/thèses/articles, cours/ressources éducatives, examens et manuels ouverts. Cherche jusqu'à 50 ressources/datasets, puis audite vraiment 10–15 fiches/échantillons.

Pour chaque ressource, renvoie une ligne JSONL avec :
{"dataset":"","dataset_id":"","official_url":"","version_date":"","type":"book / law / parliamentary / science / education / stats / other","language":"","size":"","original_source":"","license_of_dataset":"","license_per_work_or_source":"","commercial_use":"allowed / prohibited / unclear / conditional","attribution_sharealike_nc_terms":"","access_and_contact_terms":"","ocr_quality_or_source_accuracy":"","pii_or_named_speaker_risk":"","sample_method_and_n":"","sample_findings":"","suitable_stage":"pretrain / sft / evaluation / none","verdict":"use / limited / hold / reject","confidence":"high/medium/low"}

Examine particulièrement la différence entre domaine public d'une œuvre et conditions d'accès/réutilisation d'une bibliothèque; distinguer Licence Ouverte et CC BY-SA/NC-SA; noter toute redevance ou interdiction commerciale; pour les entretiens oraux, vérifier consentement, identifiabilité et âge. Signale les mélanges de langues et les périodes historiques, les erreurs OCR et le risque de conversations avec des mineurs. N'invente pas de droits : si la fiche ou la source primaire ne tranche pas, marque « inconnu ».

Termine par une shortlist qui couvre des styles complémentaires, tout en séparant les corpus d'évaluation de ceux d'entraînement. Fournis les URL officielles et une méthode de réutilisation licite.
```

## Prompt 3 — Conversations et SFT français

```text
Audite les datasets publics d'instructions, de dialogues et de conversations en français pour le SFT d'un modèle entraîné de zéro, 200–300 M paramètres, qui doit répondre uniquement en français. Les conversations Discord privées sont strictement hors périmètre.

Trouve jusqu'à 50 datasets candidats. Pour 10–15 candidats importants, ouvre la fiche officielle et examine un petit échantillon stratifié par sous-source/split. Sépare impérativement : contenu écrit par des humains dans la langue cible; contenu traduit; réponses générées par modèle; données synthétiques à partir de documents sources; données de préférence; dialogues audio/transcrits; jeux spécialisés.

Format JSONL par source :
{"dataset":"","dataset_id":"","official_card_url":"","version_date":"","rows_tokens":"","conversation_format":"","french_only_or_filter":"","human_translated_synthetic_shares":"","upstream_datasets_and_models":"","license_dataset":"","underlying_data_rights":"","terms_and_restrictions":"","pii_minors_consent":"","sample_n_and_sampling":"","observed_language_quality":"","factuality_and_staleness":"","template_repetition_and_duplicates":"","suitable_use":"sft / preference / eval / hold / reject","verdict":"","confidence":"high/medium/low"}

Repère particulièrement les licences manquantes, les sous-datasets à licences hétérogènes, les traductions anglaises de faible qualité, les faits datés, les réponses inventées, les prompts non français, les données extraites de communautés/personnes sans consentement documenté et les fuites de benchmarks. Ne confonds pas « dataset marqué French » et « chaque message/réponse est en français ». Ne cite pas seulement un billet de blog si une fiche officielle existe.

Termine par une shortlist réellement autorisable, des tests de qualité et un plan de séparation train/validation/test. Si moins de 50 sources robustes existent, dis-le au lieu de remplir la liste avec des copies ou des repos sans provenance.
```

## Prompt 4 — Code, tâches agentiques et évaluations françaises

```text
Recherche des corpus de code/documentation française, traces d'utilisation d'outils, tâches d'agent et benchmarks francophones pour évaluer un modèle entraîné de zéro (200–300 M paramètres) qui répond uniquement en français. Les conversations Discord privées sont exclues.

Trouve jusqu'à 50 sources candidates dans ce périmètre. Audite 10–15 fiches et échantillons; sépare strictement jeu d'entraînement, jeu de test public et évaluation tenue à l'écart. Inclure code open-source seulement quand sa licence par dépôt/fichier est connue. Identifier les tool traces synthétiques vs exécutées, les appels corrects et vérifiés, le français des explications et les scripts d'évaluation.

Format JSONL :
{"dataset":"","dataset_id":"","official_url":"","version_date":"","type":"code / docs / tool-trace / task / benchmark","language_of_prose":"","french_only_evidence":"","size":"","license_and_attribution":"","per_repo_or_record_provenance":"","synthetic_or_executed":"","execution_and_verification_protocol":"","benchmark_contamination_risk":"","pii_or_secret_risk":"","sample_n_and_findings":"","train_or_eval":"","verdict":"use / limited / hold / reject","confidence":"high/medium/low"}

Ne recommande pas d'exécuter aveuglément des dépôts. Distingue la licence du dataset de celle des dépôts de code. Signale secrets, exemples malveillants, tests non isolés, fonctions hallucinées, benchmarks traduits et contamination des sets publics. Une trace n'est « vérifiée » que si le résultat d'outil/test a réellement été exécuté et conservé avec sa provenance.

Termine par une shortlist et propose au moins 30 tâches françaises indépendantes (pas les réponses), chacune avec objectif, outils nécessaires, résultats observables et critères de réussite, pour notre jeu de test non contaminé. N'inclus pas ces nouvelles tâches dans les datasets publics sans vérifier leurs licences.
```

## Format de retour conseillé à l'utilisateur

Demande aux agents de renvoyer les tableaux JSONL en fichiers texte ou dans le chat, plus un résumé priorisé. Ajouter dans le cahier principal uniquement les lignes dont les URLs, licence et niveau de vérification sont explicites. Les autres restent en « repérée » ou sont rejetées.

## Exécution réalisée le 18 octobre 2026

Les 4 prompts ci-dessus + veille ont été exécutés dans cette session :

- **Prompt 1** : pipeline amélioré pour licence par ligne, audit v2, build scripts v2, cahier section K.
- **Prompt 2** : distinction domaine public / conditions bibliothèque, Licence Ouverte vs CC BY-SA/NC, filtre OCR.
- **Prompt 3** : audit Aya, French Instruct, Luth-SFT, Alpaca, Légitus, Claire, FQuAD ; pas de SFT automatique.
- **Prompt 4** : veille benchmarks (FrenchBench, COLE, HumanEval-XL, xCodeEval, SWE-bench, Terminal-Bench, BFCL, Toolathlon), 30 tâches originales dans docs/19 et benchmarks/agent_fr_v2.jsonl (30 tâches).
- **Prompt 5** : déjà traité dans docs/19 avec veille datée au 18 octobre 2026 et architecture RAG.

Détails : docs/20-QUALITE-DATASET-FR.md et docs/21-EXECUTION-PROMPTS-FR.md
