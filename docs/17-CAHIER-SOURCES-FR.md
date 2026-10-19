# Cahier des sources de données françaises

**Registre vivant — première passe au 18 octobre 2026.** Il sépare les recherches rapides, les fiches effectivement consultées et les sources retenues. « Candidate » ne veut pas dire « approuvée » : aucun dataset externe de ce cahier n'a été téléchargé dans ce dépôt ni intégré au corpus. Un rapport externe reçu le 18 octobre affirme avoir échantillonné plusieurs sources; ces observations sont consignées comme rapportées, non reproduites localement. Le corpus Discord est hors périmètre.

## Comment lire le statut

- **Consultée** : fiche officielle ouverte et éléments de licence/provenance lus.
- **Repérée** : trouvée par recherche ou catalogue; fiche complète et échantillon restent à examiner.
- **Candidate** : mérite un audit local avant inclusion.
- **En attente** : droits, provenance, qualité ou mélange de langues non établis.
- **Écartée par défaut** : risque trop élevé ou usage incompatible jusqu'à preuve contraire.

Pour les sources Web, la licence du dataset comme base de données ne règle pas automatiquement les droits de chaque page, les données personnelles, les retraits/opt-out ou les conditions du site d'origine. Vérifier les fiches, la provenance et les conditions actuelles avant usage ou redistribution. Les licences ci-dessous sont des relevés de fiches, pas un avis juridique.

## A. Pré-entraînement : texte général français

| ID | Source / lien officiel | Ce qui a été vérifié | Qualité / limites à auditer | Licence / provenance constatée | Statut actuel |
|---|---|---|---|---|---|
| P-01 | [FineWeb2-HQ, sous-ensemble `fra_Latn`](https://huggingface.co/datasets/epfml/FineWeb2-HQ) | Fiche ouverte : sous-ensemble français, 32 248 772 documents / 483 Go indiqués; top 10 % de FineWeb2 selon un score XLM-RoBERTa; champs `text`, `quality_score`, `language_score`, `url`, `date`, `dump`, etc. | Sélection de qualité intéressante, mais score indirect et distribution à mesurer. La fiche prévient que des PII peuvent rester; inspecter URLs, dates, doublons, textes hors sujet, OCR, boilerplate et contenu sensible. | ODC-By 1.0 pour la base de données; conditions Common Crawl applicables; contenus du Web gardent leurs propres conditions. | **Candidate prioritaire pour audit**, pas encore validée. Le script de build a été basculé vers cette source; aucun gros corpus n'a été construit. |
| P-02 | [FineWeb2, sous-ensemble `fra_Latn`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | Fiche ouverte; corpus de pré-entraînement multilingue issu de 96 crawls Common Crawl (2013–2024), incluant un sous-ensemble français. | Plus volumineux, mais qualité moyenne et bruits/PII probablement plus importants que HQ. À conserver seulement comme supplément après comparaison contrôlée avec P-01. | ODC-By 1.0 sur la base; conditions Common Crawl applicables. | **Réserve**, non utilisée par défaut. |
| P-03 | [PleIAs Common Corpus](https://huggingface.co/datasets/PleIAs/common_corpus) et [annonce PleIAs](https://huggingface.co/blog/Pclanglais/two-trillion-tokens-open) | Fiche ouverte : métadonnées par ligne (`language`, `open_type`, `license`, `identifier`, `collection`, etc.). L'annonce revendique une importante fraction française. | Potentiellement la meilleure traçabilité ligne par ligne, mais corpus mélangeant des centaines de langues, licences et types de données; certains exemples sont anglais ou autres langues. Filtrer le français **et** les types/licences autorisés par ligne, puis auditer chaque collection. | Pas de licence unique à supposer : chaque ligne annonce un type/ou une licence; certaines sources sont CC-BY-SA, Licence Ouverte, domaine public, etc. | **Candidate de grande valeur**, mais intégration exige un chargeur spécialisé et une allowlist par licence/provenance. |
| P-04 | [FineWeb2-HQ paper](https://arxiv.org/abs/2502.10361) | Résultats de sélection par modèle et comparaison sur des évaluations rapportés par les auteurs. | Résultat publié non équivalent à un audit indépendant sur notre mélange ou notre tokeniseur. | Description de FineWeb2-HQ; mêmes réserves ODC-By/Common Crawl. | **Référence méthodologique**, pas dataset distinct. |
| P-05 | [French Public-Domain Books de PleIAs](https://huggingface.co/datasets/PleIAs/French-PD-Books) | Fiche ouverte : environ 289 000 livres / 16,4 milliards de mots annoncés (collection Gallica/BNF); champs incluant OCR, titre, auteur, date, texte complet. | Grande variété littéraire/historique; français ancien, OCR parfois fortement dégradé, pages de titre/notes répétitives. Ajouter des filtres OCR, date, doublons et longueur; ne pas laisser ce style dominer un assistant contemporain. | La fiche revendique le domaine public, mais rapporte aussi que la BNF facture la réutilisation commerciale de certains contenus. Examiner les conditions BNF/jeu et le cas d'usage avant intégration. | **Candidate en proportion limitée**; licence commerciale à clarifier pour le scénario visé. |
| P-06 | [WikiText-fr (`asi/wikitext_fr`)](https://huggingface.co/datasets/asi/wikitext_fr) | Fiche ouverte : plus de 70 M tokens extraits des articles Wikipédia « qualité » ou « bons articles »; splits train/validation/test. | Très petite face au besoin de pré-entraînement. Bonne matière à test de perplexité/validation, avec précaution contre contamination train/test. Nettoyage minimal selon sa fiche. | CC BY-SA 4.0 indiquée sur la fiche. | **À réserver surtout à une évaluation/validation**, pas source principale. |
| P-07 | [OpenLLM-France / Luciole Training Dataset](https://huggingface.co/datasets/OpenLLM-France/Luciole-Training-Dataset) | Fiche partiellement consultée : ensemble hétérogène, nombreuses sous-collections, web, livres, journaux, textes OCR, code, mathématiques et données d'instruction; environ 4,65 T tokens au total; français indiqué à 16,3 % dans la répartition publiée. | Gros potentiel, mais pas français uniquement sans sélection de shards. Accès/structure complexes; mélanges d'origines et risque de réutiliser un corpus de train connu dans nos tests. Exiger le filtre de langue et la provenance par sous-collection. | Fiche du dataset indique CC BY-SA 4.0; sous-sources à recontrôler individuellement avant d'en extraire des sous-ensembles. | **En attente d'audit par sous-collection**; ne pas télécharger en bloc. |
| P-08 | [OpenLLM-France / Lucie Training Dataset](https://huggingface.co/datasets/OpenLLM-France/Lucie-Training-Dataset) et [article](https://arxiv.org/html/2503.12294v1) | Repérée : collection multilingue centrée sur le français, mais contient aussi anglais, espagnol, allemand, italien, code et jeux d'instruction. | Certaines données sont générées ou traduites; nécessité de tracer chaque sous-corpus et de filtrer le français. | Fiche signalée CC BY-NC-SA 4.0 en raison de sous-corpus; incompatible avec un usage commercial sans autorisation. | **Écartée par défaut**; seuls des sous-ensembles dont les droits sont établis pourront être reconsidérés. |
| P-09 | [FineWeb2-HQ: version française échantillonnée](https://huggingface.co/datasets/epfml/FineWeb2-HQ) | La carte a aussi été vérifiée pour la présence d'un score qualité et des champs URL, dates et scores de langue. | La qualité doit être mesurée sur notre propre échantillon; top 10 % par modèle ne remplace pas la revue humaine ni le filtre PII. | ODC-By + Common Crawl. | **Source branchée au script d'audit** `scripts/audit_fr_source.py`; aucun échantillon n'a encore été audité par un humain. |
| P-10 | [OpenEuroLLM Catalogue de données](https://github.com/OpenEuroLLM/training-data-catalogue) | Catalogue repéré : inventorie mC4, CulturaX, OSCAR, HPLT, FineWeb2 et d'autres familles. | Répertoire de découverte, pas garantie de qualité ni de droits pour chaque sous-corpus. | Dépend de chaque source référencée. | **À exploiter pour élargir la recherche**, ne pas traiter comme dataset. |

## B. Dialogue, instruction et réglage supervisé français

| ID | Source / lien officiel | Ce qui a été vérifié | Qualité / limites à auditer | Licence / provenance constatée | Statut actuel |
|---|---|---|---|---|---|
| S-01 | [Aya Dataset](https://huggingface.co/datasets/CohereForAI/aya_dataset) et [papier](https://arxiv.org/html/2402.06619v1) | Jeu multilingue avec environ 204 k annotations humaines originales dans 65 langues; le travail Lucie rapporte 1 422 exemples français. | Petite tranche FR; séparer annotations originales des versions traduites/templatisées et éviter de mélanger les langues. Bon candidat pour une petite évaluation humaine ou du SFT après revue d'échantillons. | Apache 2.0 annoncée pour Aya Dataset/Aya Collection dans le papier; confirmer la fiche et la sous-partie ciblée avant usage. | **Candidate SFT de petite taille**, qualité humaine intéressante à vérifier. |
| S-02 | [French Instruct (`angeluriot/french_instruct`)](https://huggingface.co/datasets/angeluriot/french_instruct) et [dépôt source](https://github.com/angeluriot/French_instruct) | Fiche ouverte : ~276 k dialogues; le dépôt rapporte ~84,9 M tokens, dont environ 29 % humains et 71 % de réponses générées par chatbot; 13 sources. | Fort volume mais traductions, tâches templatisées et style variable. Des affirmations factuelles anciennes peuvent être fausses; auditor un échantillon stratifié par source/style, vérifier déduplication et langue de la réponse. | Licence globale non vérifiée de façon suffisante; les droits peuvent différer entre sources composantes. | **En attente**; ne pas reprendre le mélange complet sans registre des 13 sources et licence de chacune. |
| S-03 | [Luth-SFT](https://huggingface.co/datasets/kurakurai/luth-sft) et [papier](https://arxiv.org/html/2510.05846) | Fiche ouverte : ~571 k lignes, plusieurs splits (`scholar`, `croissantllm`, `smoltalk2`, `aya`, `tulu3`, `openhermes`); papier revendique ~338 M tokens français. | Données hétérogènes; beaucoup de contenu synthétique et de mathématiques; les sous-parties ont des niveaux de langue et formats différents. Auditer par split, provenance, licence et taux de réponses réellement françaises. | La fiche consultée n'établit pas une licence unique suffisamment claire; certaines sources et générations ont leurs propres conditions. | **En attente d'examen ligne/split par split**; pas de SFT automatique. |
| S-04 | [French Alpaca 110K](https://huggingface.co/datasets/jpacifico/French-Alpaca-dataset-Instruct-110K) | Fiche ouverte : ~110 368 instructions/réponses annoncées, générées avec GPT-3.5; les exemples sont en français. | Échantillon visible comportant des réponses peu fiables ou périmées (faits, dates, citations); traduction/génération et faible variété. Exige un audit factuel sévère. | Pas de licence claire établie sur la fiche consultée; demande de crédit à l'auteur et provenance Alpaca/GPT-3.5 à éclaircir. | **Écartée par défaut** jusqu'à clarification de la licence et audit de précision. |
| S-05 | [Légitus SFT (`moncefem/legi-instruct-fr`)](https://huggingface.co/datasets/moncefem/legi-instruct-fr) | Fiche ouverte : 27 357 conversations juridiques françaises; catégories dont abstention, citations, RAG et usage d'outil; sources légales LEGI/Légifrance; contenu généré par plusieurs modèles puis filtré. | Bonne couverture de l'abstention et des appels d'outils dans un domaine spécialisé; peut sur-spécialiser le modèle et n'est pas représentatif de dialogue général. Vérifier chaque citation juridique, date et outil. | Compilation/code du dataset sous Apache 2.0 selon la fiche; textes légaux sous Licence Ouverte/Etalab; la génération multi-fournisseurs doit aussi être documentée. | **Candidate spécialisée pour tests/SFT légal**, pas pour pré-entraînement général. |
| S-06 | [Claire French Dialogue Dataset](https://huggingface.co/datasets/OpenLLM-France/Claire-Dialogue-French-0.1) | Fiche détaillée consultée : ~37 015 conversations, ~2,96 M tours, ~150 M mots; sources parlementaires, théâtre, entretiens et conversations libres; accès conditionné au partage de coordonnées. | Parole naturelle utile, mais styles, transcriptions, locuteurs identifiables et domaines très différents. Conditions d'accès et conditions des corpus d'origine; auditer consentement, identifiabilité et présence éventuelle de mineurs par sous-corpus. | La table source distingue notamment Assemblée nationale (Licence Ouverte 2.0) et plusieurs sous-sources CC BY-NC-SA; d'autres sources ont des conditions distinctes. Pas de licence unique à appliquer au tout. | **Ne pas reprendre le corpus entier**; examiner uniquement les sous-parties explicitement compatibles, en priorité les débats publics dont les conditions autorisent l'usage envisagé. |
| S-07 | [FQuAD (`illuin/fquad`)](https://huggingface.co/datasets/illuin/fquad) | Repéré : questions/réponses de compréhension de documents français. | Domaine QA; utile pour évaluation spécialisée, insuffisant comme dialogue général. Contrôler recouvrements de textes. | Fiche indique CC BY-NC-SA 3.0; usage commercial exclu sans accord. | **Écartée du train par défaut**; possible benchmark non-commercial après contrôle de licence. |
| S-08 | [Luth paper / Scholar subset](https://huggingface.co/datasets/kurakurai/luth-sft) | Source repérée pour problèmes de mathématiques, baccalauréat et CPGE en français. | Bon candidat pour raisonnement/maths si corrigés revus; risque de réponses erronées et de fuite benchmark. | Licence du split/source à vérifier indépendamment. | **À auditer comme domaine séparé**, pas encore approuvé. |

## C. Conversation, parole et code agentique

| ID | Source | État | Décision |
|---|---|---|---|
| A-01 | [Claire/CFDD](https://huggingface.co/datasets/OpenLLM-France/Claire-Dialogue-French-0.1) | Source française conversationnelle examinée; plusieurs sous-corpus soumis à des licences non commerciales et corpus oraux à conditions de recherche/attribution. | N'utiliser que les partitions dont licence, consentement, identifiabilité, âge et conditions d'accès sont clairement compatibles; pas d'import en bloc. |
| A-02 | [Légitus SFT](https://huggingface.co/datasets/moncefem/legi-instruct-fr) | Échantillon de fiche contenant de vraies séquences multi-messages et appels d'outils juridiques générés. | Bon candidat de spécialité pour les tâches citation/abstention; vérifier les résultats juridiques et ne pas le prendre pour un benchmark général. |
| A-03 | [`benchmarks/agent_fr.jsonl`](../benchmarks/agent_fr.jsonl) | 13 cas écrits pour le dépôt : calcul, heure, notes, lecture du projet, tentative de traversal, tâches multi-étapes et vérification. | Données synthétiques internes, non issues d'un corpus externe; évaluation seulement, 13 cas ne suffisent pas à l'entraînement ni à mesurer la compétence générale. |
| A-04 | [FineWeb2/HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ) | Texte Web et score qualité, pas des dialogues étiquetés. | Pré-entraînement seulement; ne pas compter ces documents comme traces d'agent ni instructions. |
| A-05 | [OpenEuroLLM Catalogue](https://github.com/OpenEuroLLM/training-data-catalogue) | Index de découverte pour corpora multilingues et de code. | À explorer pour code public avec licences explicites; n'est pas un corpus agentique français approuvé. |

## D. Sources trouvées mais pas encore vérifiées

Ces sources ont été repérées en recherche; elles ne sont pas des recommandations tant que la fiche, la licence et des échantillons n'ont pas été consultés : CulturaX/mC4, OSCAR, HPLT, MADLAD-400, ROOTS, OpenAssistant FR, OpenHermes-fr, CroissantLLM, SmolTalk2, Tulu 3 Persona, Nemotron multilingual, CommonPile/Parliament, Europarl, Wikisource/Wikimedia FR, Gallica press/monographies, HAL/Thèses, Insee/Open data, VIKidia, Europarl, et jeux scolaires français.

## E. Sources explicitement non retenues dans le pipeline actif

- Conversations Discord privées : pas de collecte ni d'entraînement; autorisations de plateforme, provenance, droits, consentement et mineurs non clarifiés.
- `OpenLLM-France/Lucie-Training-Dataset` complet : multilingue et licence globale CC BY-NC-SA 4.0 selon les fiches consultées; pas de mélange complet.
- `French-Alpaca 110K` : licence/provenance à éclaircir et exemples visibles de faible fiabilité factuelle.
- `FQuAD` pour un déploiement commercial : CC BY-NC-SA 3.0 selon la fiche.
- `Claire` en totalité : licences hétérogènes, accès conditionnel et données orales; ne pas ignorer ces sous-conditions.
- Corpus bruts non attribués ou non identifiés par langue.

## F. Sources techniques consultées pour le modèle (pas des datasets)

Elles ont servi aux choix antérieurs, mais ne constituent pas les sources de données du modèle français :

- [Qwen3-4B-Instruct-2507, fiche modèle](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) — candidat open-weight d'un ancien prototype, désormais écarté comme base d'entraînement.
- [Qwen3.5-4B, fiche modèle](https://huggingface.co/Qwen/Qwen3.5-4B) — ancien candidat technique, non retenu.
- [Mistral Small 3.2, fiche modèle](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506) — trop gros pour la cible.
- [YaRN](https://arxiv.org/abs/2309.00071), [LongCodeBench](https://arxiv.org/abs/2505.07897), [vLLM tool calling](https://docs.vllm.ai/en/latest/features/tool_calling/) — références techniques contexte/outils, pas du texte d'entraînement.

## G. Protocole avant d'ajouter une source

Pour chaque corpus, compléter ces champs dans le registre avant ingestion :

1. identifiant exact, version/date, configuration et split;
2. langue déclarée, langue détectée sur échantillon stratifié et taux hors-français;
3. origine des documents, licence du dataset **et** droits/conditions par sous-source;
4. volumes réels : documents, tokens BlackCode, distribution longueur, dates et domaines;
5. déduplication exacte/quasi et recouvrement avec validation/test;
6. qualité : échantillon revu, OCR/boilerplate, factualité si annotée, répétitions, code ou spam;
7. vie privée : contacts, noms, identifiants, données de santé/sensibles, mineurs, opt-out et possibilité de retrait;
8. décision : pré-entraînement, SFT, évaluation, usage limité ou rejet; responsable et justification.

Ne pas transformer un statut « repéré » en « approuvé » sans fournir la référence exacte et les résultats d'audit. Le script de départ est [`scripts/audit_fr_source.py`](../scripts/audit_fr_source.py); il échantillonne langue, scores, doublons et motifs PII/secret, sans remplacer l'examen humain.

## H. Sources de conformité et méthodes consultées (pas des datasets)

Ces références ont servi à cadrer l'usage des données ou les objectifs du modèle; elles ne sont pas des sources d'entraînement :

- [Discord Developer Policy](https://discord.com/developers/docs/policies/developer-policy) et [Discord Developer Terms](https://discord.com/developers/docs/policies-and-agreements/terms-of-service) — l'usage de messages obtenus par API pour entraîner une IA/ML nécessite l'autorisation expresse de Discord; aucune ingestion Discord n'est en cours.
- [CNIL — anonymisation et pseudonymisation](https://www.cnil.fr/fr/recherche-scientifique-hors-sante-enjeux-et-avantages-de-lanonymisation-et-de-la-pseudonymisation) et [CNIL — réutilisation pour l'IA](https://www.cnil.fr/fr/assurer-que-le-traitement-est-licite-reutilisation-des-donnees) — pseudonymiser ne retire pas automatiquement le statut de donnée personnelle; il faut établir une base légale, les garanties et l'information applicables.
- [CEPD/EDPB, lignes directrices recherche scientifique 1/2026](https://www.edpb.europa.eu/system/files/files/file1/edpb_guidelines_202601_scientificresearch_en.pdf) — la recherche ne supprime pas les obligations RGPD.
- [LongCodeBench](https://arxiv.org/abs/2505.07897), [YaRN](https://arxiv.org/abs/2309.00071), [FlashAttention](https://arxiv.org/abs/2205.14135), [documentation KV cache Transformers](https://huggingface.co/docs/transformers/kv_cache), [DeepSeek-V2](https://arxiv.org/abs/2405.04434), [StarCoder2-3B](https://huggingface.co/bigcode/starcoder2-3b) — références techniques au contexte et aux modèles, pas des sources d'entraînement françaises.

## I. État de couverture

À cette date, nous avons consulté plusieurs fiches et lancé une première recherche de sources, **pas encore audité des centaines de sources**. Ce cahier est le registre initial de ce qui a réellement été inspecté; les sources « repérées » sont explicitement séparées. Pour augmenter la couverture sans faire semblant, faire explorer les quatre lots spécialisés dans [`docs/18-PROMPTS-AGENTS-DATA-FR.md`](18-PROMPTS-AGENTS-DATA-FR.md), puis intégrer les résultats avec leur niveau de preuve.

## J. Rapport externe de pré-entraînement reçu le 18 octobre 2026

Le rapport joint `AUDIT_RAPPORT_PRETRAIN_FR.md` dit avoir recherché 50 sources et examiné 15 fiches/échantillons. Le JSONL copié dans la conversation annonce notamment PleIAs Common Corpus, French-PD-Books, FineWeb2-HQ, Wikipédia et plusieurs jeux juridiques comme candidates. Les tailles, licences détaillées et observations d'échantillon restent **déclarées par ce rapport externe** : elles ne sont pas confirmées par téléchargement ou contre-audit dans le dépôt.

- À prioriser pour vérification primaire : Common Corpus (filtre par langue/licence/provenance par document), Wikimedia FR et les dépôts DILA en licence explicitement vérifiée.
- À garder sous réserve : French-PD-Books (OCR, droits/conditions BnF à vérifier selon usage), HALvest et autres corpus scientifiques hétérogènes.
- Web crawl (FineWeb2-HQ, HPLT, OSCAR, CulturaX, mC4) : la licence de la base/du packaging ne règle pas à elle seule les droits de chaque texte; examiner les retraits/opt-out TDM, la provenance et la PII. L'échantillon externe très petit ne prouve pas l'absence de PII.
- Le rapport cite des claims de conformité réglementaire et des conclusions juridiques qui ne sont pas retenues ici comme faits; demander une vérification juridique dédiée. Le JSONL collé contient des clés répétées/non normalisées et doit être parsé/normalisé strictement avant import.

La synthèse détaillée, les décisions prudentes, la veille du 18 octobre 2026 et l'audit des benchmarks agent/code sont consignés dans [`docs/19-VEILLE-ACTUALITE-IA-ET-EVALUATION-AGENTS-FR.md`](19-VEILLE-ACTUALITE-IA-ET-EVALUATION-AGENTS-FR.md).

## K. Améliorations qualité du 18 octobre 2026 (session 1h)

Cette section trace les améliorations apportées pendant la session demandée par l'utilisateur : "travaille une bonne heure et améliore la qualité".

### Pipeline

- `blackcode/data.py` : 
  - séparation licence dataset vs texte source, `normalize_licence_label()`, `PERMISSIVE` restreint, `?` non permissif.
  - `iter_hf_docs` conserve `record_license`, `collection`, `open_type`, `url`, `identifier`, `date`, `dump`, `language`.
  - nouveaux filtres `--require-hf-license`, `--allowed-hf-license`, stats `record_licenses`, `rejected_licence`, `rejected_record_license`.
  - qualité : `_url_density` (trop d'urls), `BOILERPLATE_MARKERS` (boilerplate), `REPEATED_PUNCT_RE` (ponctuation répétée), `unique_line_ratio` prose.
  - `pack()` écrit `manifest.jsonl` avec hash, path, lang, kind, licence, meta, chars.

### Scripts

- `audit_fr_source.py` v2 : comptage licences brutes/normalisées, collections, open_types, language_field, missing_license, médianes chars/mots, quality_score, language_score, french_rate, PII, secrets, hosts. Support `--license-field`.
- `build_fr_corpus.sh` v2 : variables env `HF_DATASET`, `HF_FIELD`, `VOCAB`, `SEQ_LEN`, vérif dépendances, affichage params, vérif meta.
- `build_common_corpus_fr.sh` (nouveau) : exemple filtrage Common Corpus avec allowlist licences (à valider juridiquement).

### Benchmarks

- `benchmarks/agent_fr_v2.jsonl` : 30 tâches françaises originales couvrant lecture/fondement, extraction/calcul, code, sélection d'outil, planification, sécurité, fraîcheur. Inspirées de `docs/19` mais avec vérificateurs explicites. Jeu interne, tenu à l'écart, pas pour entraînement.

### Documentation

- `docs/20-QUALITE-DATASET-FR.md` : synthèse des changements, protocole recommandé, limites, prochaines étapes.
- `docs/16-MODELE-FR-260M.md` reste valide ; protocole audit/build mis à jour dans `docs/20`.
- Tests : 24 tests `test_data` OK après modifs.

### Ce qui reste à faire (pas dans cette session)

- Audit réel HF (FineWeb2-HQ + Common Corpus) sur runtime GPU avec `audit_fr_source.py --limit 2000`.
- Choix seuil `MIN_QUALITY_SCORE` d'après distribution observée.
- Validation juridique allowlist licences Common Corpus.
- Build 4 Go filtré + mesure débit T4.
- Ajout tests unitaires pour nouveaux motifs `boilerplate`, `trop d'urls`, `ponctuation répétée`.


## L. Session 2 — améliorations continues (18 octobre 2026, suite)

### Tests qualité

- `tests/test_data.py` : 30 tests (vs 24) avec `QualityV2Tests` couvrant `trop d'urls`, `boilerplate`, `ponctuation répétée`, `répétitif`, `good french`, `license_normalization`.
- `normalize_licence_label` : remplace aussi `. / , ; ( ) [ ]` par espace, pour matcher "Apache-2.0" == "apache 2 0".

### Détection française sans lingua

- `FRENCH_STOPWORDS` enrichi (avec accents, très, où, mais, donc, etc.)
- `_strip_accents()` + `_french_heuristic_score()` : stopwords normalisés sans accents + ratio accents éèàêâîôûëïüç.
- `is_french_prose(..., use_heuristic_fallback=True)` : tente lingua, si échec ou confiance <0.85, fallback heuristique seuil 0.45. Permet audit rapide sans dépendance lourde.

### PII

- Ajout IBAN FR/générique, carte bancaire 4x4 et 13-16 chiffres contigus (heuristique conservatrice).
- Regex toujours heuristique, pas anonymisation.

### Corpus local de démo

- `data/fr_seed/exemple_fr.md` : prose française de qualité, test build OK avec `french-only` (1 doc, 495 tokens, french_rate 1.0, manifest OK).
- `scripts/eval_french_purity.py` : évalue corpus construit via manifest + lecture fichiers locaux si présents.

### Benchmarks agents

- `scripts/validate_agent_tasks.py` : valide JSONL 13 + 30 tâches, champs obligatoires, prompt longueur.
- `scripts/eval_agent_benchmark.py` : compte par catégorie, vérifie couverture 7 familles (lecture_fondement, extraction_calcul, code, selection_outil, planification, securite, fraicheur), heuristique français.

### Pipeline complet

- `docs/22-PIPELINE-COMPLET-FR.md` : doc end-to-end audit → seuils → build → qualité → dedup → tokenizer → pack → entraînement pilote → évaluation → SFT futur → limites.

### Reste à faire

- Audit réel HF sur Colab/Kaggle avec `audit_fr_source.py` limit 2000.
- Mesure débit T4, ajuster batch/seq-len.
- Classifieur contenu sensible, filtre opt-out TDM.
- SFT français séparé, pas recyclage eval.


## M. Corpus oraux enfants/parents évalués le 18 octobre 2026

| ID | Source / lien officiel | Vérifié | Qualité / limites | Licence constatée | Statut |
|---|---|---|---|---|---|
| O-01 | [ALIPE, Ortolang hdl:11041/alipe-000853](http://hdl.handle.net/11041/alipe-000853) et [site LRL](http://lrl-diffusion.univ-bpclermont.fr/alipe/) | Fiches consultées via web_search 18 oct 2026 : 12h50 / 90 616 mots (v initiale), 30h / 174k mots selon publi, 252k mots selon autre article ; 3 enfants 28-40 mois puis 36-64 mois ; audio anonymisé + CHAT/XML-TEI | Oral spontané familial, CDS (énoncés courts, syntaxe simple). PII : prénoms Salomé/Baptiste/Prune dans pubs, noms propres/chiffres en source non anonymisée selon conventions. Transcription avec hésitations/répétitions. Taille ~1 Mo texte. | CC annoncée, CC-BY-SA 4.0 selon blog dataset_audio_fr, à confirmer sur Ortolang (peut être BY-NC-SA) | **En attente / usage limité recherche seulement**, pas pré-entraînement principal. Évaluation oral possible si BY-SA confirmé, filtrage PII, avis éthique mineurs. |
| O-02 | [TCOF, CNRTL](https://www.cnrtl.fr/corpus/tcof/) et [Ortolang hdl:11403/tcof/v2](https://hdl.handle.net/11403/tcof/v2) et [tcof.atilf.fr](https://tcof.atilf.fr/) (626 corpus, 146h, 1 542 562 mots) | Fiches consultées 18 oct 2026 : 517 transcriptions / 124h / 37,5 Go (CNRTL), 626 / 146h / 1,5M mots (site TCOF) ; 2 catégories adultes-enfants (99-200 enfants 2-7 ans) et adultes-adultes ; Transcriber+XML+WAV ISO-8859-1 ; années 80-90 | Oral spontané, 100+ enfants, PII : conventions "Ne pas anonymiser le corpus (indiquer tous les noms propres, chiffres...)" → version brute contient PII, anonymisation ultérieure prévue. Qualité : nombres en lettres, onomatopées codifiées, {sic}. 1,5M mots ~6 Mo texte. | CC selon CNRTL, CC BY-NC-SA 2.0 selon blog dataset_audio_fr, CC BY-SA selon page Ortolang 2024, TCOF-POS BY-NC-SA 2.0 → incohérence selon version, vérifier v2/v2.2 | **En attente / évaluation seulement**, risque mineurs/PII plus élevé que ALIPE. Si NC → écarté commercial. Si BY-SA recherche non commercial → petit split éval possible, pas pré-entraînement principal. Adultes-adultes moins risqué si filtré. |

Voir détails complets dans `docs/24-CORPUS-ORAUX-ENFANTS-FR.md`. Aucun téléchargement effectué.


## N. Datasets discussions avec abréviations — réseaux sociaux (SMS, tchat, tweets) — 18 octobre 2026

Objectif : modèle FR-260M doit comprendre français informel avec abréviations `slt/bjr/stp/mdr/ptdr/tkt/pk/koi/c/ct/bcp` sans forcément les produire (SFT). Aucun téléchargement effectué, fiches Ortolang consultées via web_search 18 oct 2026.

| ID | Source / lien officiel | Vérifié | Qualité / limites | Licence constatée | Statut |
|---|---|---|---|---|---|
| RS-01 | [88milSMS, CoMeRe cmr-88milsms-tei-v1](https://hdl.handle.net/11403/comere/cmr-88milsms) et [site 88milsms.huma-num.fr](http://88milsms.huma-num.fr/) et [ELRA-W0082](https://catalogue.elra.info/en-us/repository/browse/ELRA-W0082/) | Fiche v3.3 consultée : 88 522 messages finalisés, 422 participants, 29 563 emoticons/emojis, 5 Mo, ISLRN 024-713-187-947-8, collecte 2011-09-15 à 2011-12-15 Montpellier, TEI, anonymisé avec CIL juriste, 780 dl 48 pays 2014-2019, 100 SMS annotés + 1000 transcodés FR standard | Oral écrit informel authentique, plus grand corpus SMS FR, abréviations fortes `f = fais/fera`, fautes, emojis, métadonnées genre/âge/timestamp. Qualité : non standard, utile pour robustesse. PII déjà anonymisé mais vérifier. Taille ~5 Mo texte → négligeable vs 16 Go 6 Md. | **CC BY 4.0** sur Ortolang (fiche v3.3 : http://creativecommons.org/licenses/by/4.0/) — compatible commercial avec attribution. ELRA-W0082 version est Non Commercial non standard, mais version CoMeRe est BY 4.0. | **Candidate 5-10% mélange** pré-entraînement + 20% SFT conversationnel. Conserver brut + transcodé pour mapping abréviation→standard. |
| RS-02 | [CoMeRe repository global](https://hdl.handle.net/11403/comere) et [Ortolang market comere](https://www.ortolang.fr/market/corpora/comere) | Fiche consultée : 14 sous-corpus, 75M tokens v3.3, 74 325 542 mots annoncés, TEI UTF-8, 2001-2014, dialogue, genres SMS/tweets/email/forums/textchat IRC/weblog/multimodal, mono-lingue français | Méta-corpus hétérogène, qualité variable selon sous-corpus, formats TEI différents. Pas à ingérer en bloc. | CC BY 4.0 global sauf indication contraire par sous-corpus (NC-SA pour certains) | **Méta, pas direct** — utiliser pour découvrir sous-corpus |
| RS-03 | [cmr-smslareunion, CoMeRe](https://hdl.handle.net/11403/comere/cmr-smslareunion) | Fiche consultée : 12 622 SMS finalisés, 884 participants, 21 694 SMS bruts collectés 2008-04-10 à 06-30, 1 744 users Réunion, créolismes | Variation régionale FR + créole réunionnais, abréviations + créolismes, complémentaire 88milSMS | CC BY 4.0 | **Candidate petite proportion** complément 88milSMS |
| RS-04 | [cmr-getalp_org, CoMeRe IRC](https://hdl.handle.net/11403/comere/cmr-getalp_org) | Fiche consultée : 19 997 messages, 86 964 mots, 195 participants, 8 225 événements, canal EpikNet IRC 2004-02 à 04, 4M messages dans 105 canaux hétérogènes | Tchat synchrone très informel, commandes IRC, abréviations, style RS brut. Taille 87k mots | **CC BY-NC-SA 4.0** (fiche getalp_org) | **En attente / éval seulement si non commercial** — NC = écarté par défaut commercial |
| RS-05 | [cmr-polititweets / cmr-intermittent, CoMeRe tweets](https://hdl.handle.net/11403/comere) | Article CoMeRe 2014 repéré : 34k tweets, 44k SMS, 2,3k emails, 2,7k forums dans CoMeRe global | Tweets politiques, abréviations, hashtags, mentions. Risque ToS X/Twitter interdit redistribution texte complet, PII, mineurs possible | À vérifier par sous-corpus + ToS X actuel | **En attente** — vérifier fiche exacte et ToS, préférer SMS/tchat juridiquement plus stables |
| RS-06 | [1000 SMS avec labels spatiaux, CIRAD Dataverse doi:10.18167/DVN1/0ZGJRC](https://dataverse.cirad.fr/dataset.xhtml?persistentId=doi:10.18167/DVN1/0ZGJRC) | Fiche consultée : extrait 88milSMS, 1000 SMS, CC BY-NC-SA 4.0 | Sous-ensemble 88milSMS déjà couvert, licence NC | CC BY-NC-SA 4.0 | **Écarté commercial** — utiliser RS-01 BY 4.0 à la place |
| RS-07 | [30k French SMS Belgium, Fairon et al. 2006](https://aclanthology.org/L06-1148/) | Article repéré : 30k SMS belges avec traduction standard, 1 736 auteurs, CD-Rom 2006 | SMS belges traduits FR standard, utile pour mapping abréviation→standard, variation belge | Licence à vérifier (académique) | **Repérée** — vérifier licence avant usage |

### Intégration pipeline

- **Filtre qualité** : kind `sms`/`chat`/`tweet` tolère court (≥10 chars, ≥3 mots, alnum ≥0.15) vs prose ≥200 chars, ≥30 mots, alnum ≥0.30. Bonus abréviations `mdr/ptdr/slt/bjr/stp/tkt` dans `quality_score`. Implémenté dans `blackcode/data.py` 18 oct 2026, 36 tests OK.
- **Mélange recommandé** : pré-entraînement 85% FineWeb2-HQ filtré + 5% Common Corpus FR filtré licence + 5% 88milSMS + 5% smslareunion + 0% NC. SFT : 50% instructions formelles FR + 30% Claire AN Licence Ouverte + 20% SMS/tchat RS-01 pour comprendre abréviations mais répondre standard (consigne système).
- **Conservation** : garder brut pour robustesse, utiliser 1000 transcodés 88milSMS pour tâche éval "traduis SMS abrégé → français standard".
- **Risques** : PII déjà anonymisé mais vérifier, contenu sensible/insultes possible, pas de mineurs ciblés dans 88milSMS (adultes Montpellier). Écarter NC-SA par défaut.

### Protocole audit

```bash
python scripts/audit_fr_source.py --dataset comere/cmr-88milsms --limit 1000 --samples-out data/audits/88milsms.jsonl
python -m blackcode.data build --src data/fr_seed --hf comere/cmr-88milsms --hf-field text --french-only --max-mb 100 --out data/sms_fr --vocab 24576 --seq-len 2048
python scripts/eval_french_purity.py --corpus-dir data/sms_fr --limit 50
```

Vérifier manuellement 20 extraits : PII, insultes, mineurs, abréviations.

Voir détails dans `docs/27-DATASETS-ABREVIATIONS-RS-FR.md`.
