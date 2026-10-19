# Corpus oraux francophones spécialisés — ALIPE et TCOF

**Date d'évaluation : 18 octobre 2026. Sources consultées : Ortolang, CNRTL, LRL, ATILF, HAL, blog dataset_audio_fr. Les licences et tailles sont relevées de fiches, pas mesurées localement. Aucun de ces corpus n'a été téléchargé dans ce dépôt.**

## 1. ALIPE — Acquisition de la Liaison et Interactions Parents-Enfants

**Fiche officielle** : http://lrl-diffusion.univ-bpclermont.fr/alipe/ et permalien Ortolang http://hdl.handle.net/11041/alipe-000853
**Laboratoire** : LRL (Laboratoire de Recherche sur le Langage), Clermont, projet ANR Phonlex, Thierry Chanier, Loïc Liégeois, Damien Chabanal.

**Contenu** :
- 3 enfants (2 fillettes et 1 garçon) enregistrés à domicile en interaction naturelle avec parents.
- Protocole : 1h/jour pendant 1 semaine, répété 8 mois plus tard → dense.
- Taille annoncée : **12h50 de dialogue, 90 616 mots** (version initiale), **30h, 174 209 mots** selon publication 2023 (CoLaJE/ALIPE comparaison), **252 579 mots** selon autre article. Variation selon version ; à vérifier sur Ortolang v finale.
- Formats : audio anonymisé + transcriptions CHAT et XML-TEI, alignement texte-son.
- Âge : 28, 36, 40 mois à T1, puis 36,44,64 mois à T2 (2;4 à 5;4 ans).

**Licence** : Creative Commons annoncée (CC-BY-SA 4.0 selon blog dataset_audio_fr [source](https://lbourdois.github.io/blog/audio/dataset_audio_fr/), CC selon fiche Ortolang). Vérifier fiche exacte Ortolang : souvent CC BY-NC-SA ou CC BY-SA pour ce type de corpus. Le blog indique CC-BY-SA 4.0, mais la fiche Ortolang peut préciser BY-NC-SA. **À confirmer sur https://www.ortolang.fr/market/corpora/alipe**.

**Intérêt pour BlackCode FR-260M** :
- Français oral spontané, interactions naturelles, Child Directed Speech (CDS) — discours parental adapté (énoncés plus courts, syntaxe simple, débit lent).
- Utile pour étudier variation phonologique (liaison, schwa) et style oral familier, complémentaire au Web écrit.

**Risques et limites** :
- **Mineurs** : 3 enfants identifiables par prénom (Salomé, Baptiste, Prune) dans publications, même si audio anonymisé. Données sensibles au sens RGPD, consentement parental nécessaire, mais réutilisation pour entraînement LLM non explicitement prévue à l'origine (corpus de recherche linguistique). CNIL et CEPD/EDPB 1/2026 : recherche ne supprime pas obligations RGPD ; pseudonymisation ≠ anonymisation.
- **PII** : noms propres, chiffres, contexte familial enregistrés à domicile → risque d'identification malgré anonymisation. Conventions TCOF disent explicitement "Ne pas anonymiser le corpus (indiquer tous les noms propres...) Le balisage... et l'anonymisation se fera ultérieurement" — donc version brute contient PII.
- **Licence commerciale** : CC BY-SA autorise commercial avec attribution et share-alike, mais BY-NC-SA interdit commercial. Si NC, **écarté par défaut** pour usage commercial.
- **Taille** : 12-30h, 90k-250k mots = ~0,5-1 Mo texte → négligeable vs 4 Go cible, mais style oral très spécifique, risque de sur-spécialisation.
- **Qualité transcription** : orthographe usuelle, hésitations, répétitions, codage spécifique CHAT/TEI → besoin de normalisation.

**Verdict** : **En attente / usage limité recherche seulement**, pas pré-entraînement général. Possible pour évaluation compréhension oral spontané ou SFT très limité si licence CC BY-SA confirmée et usage recherche non commercial, avec filtrage PII renforcé et exclusion noms. **Ne pas inclure dans corpus principal 260M sans avis juridique et éthique dédié sur données enfants**.

## 2. TCOF — Traitement de Corpus Oraux en Français

**Fiches** : https://www.cnrtl.fr/corpus/tcof/ et Ortolang https://hdl.handle.net/11403/tcof/v2 et https://tcof.atilf.fr/ (626 corpus, 146h, 1 542 562 mots selon site TCOF).

**Laboratoire** : ATILF UMR 7118 CNRS & Université de Lorraine, Virginie André et al.

**Contenu** :
- 2 catégories : interactions adultes-enfants (enfants jusqu'à 7 ans, 99 enfants selon article HAL) et adultes-adultes.
- Taille : **517 transcriptions, 124h, ~37,5 Go** (CNRTL), **626 corpus, 146h, 1 542 562 mots** (tcof.atilf.fr), **146h** (blog dataset_audio_fr). Versions 1,2,2.1,2.2 sur Ortolang.
- Formats : Transcriber + XML métadonnées + WAV, ISO-8859-1, alignement texte-son.
- Contexte : années 80-90, situations naturelles, durées 5-45 min, à terme pathologie et natifs/non natifs.

**Licence** :
- CNRTL : "Les corpus du projet TCOF sont utilisables sous licence Creative Commons" sans version précise.
- Blog dataset_audio_fr : **CC BY-NC-SA 2.0** pour TCOF (146h) [source](https://lbourdois.github.io/blog/audio/dataset_audio_fr/).
- Ortolang : CC BY-SA selon page colaf.huma-num.fr (CCBY-SA) pour TCOF 2024.
- TCOF-POS (annoté morphosyntaxe) : CC BY-NC-SA 2.0 selon ResearchGate.
→ **Incohérence licences selon version** : v2 sur Ortolang peut être CC BY-SA, mais POS et blog indiquent NC-SA. **Vérifier fiche exacte v2/v2.2**.

**Intérêt** :
- Plus grande collection oral français avec enfants, 99-200 enfants selon sources, complémentaire ALIPE.
- Interactions adultes-enfants et adultes-adultes, style informel spontané.
- Disponible sur Ortolang, accès libre.

**Risques et limites** :
- **Mineurs** : ~100-200 enfants, 2-7 ans, enregistrés en contexte naturel, potentiellement identifiable. Même réserves RGPD/mineurs que ALIPE, mais à plus grande échelle. Consentement recherche linguistique ≠ consentement entraînement LLM.
- **PII** : conventions explicites "Ne pas anonymiser le corpus (indiquer tous les noms propres, chiffres etc. entendus, même ceux permettant l'identification)". Anonymisation ultérieure prévue mais version brute contient PII. Risque élevé.
- **Licence NC** : si BY-NC-SA 2.0 confirmé, **écarté par défaut** pour usage commercial (comme FQuAD, Lucie). BY-SA autoriserait commercial avec share-alike, mais NC interdit.
- **Qualité** : ISO-8859-1, transcriptions avec hésitations, répétitions, {sic}, onomatopées codifiées, nombres en lettres → besoin normalisation lourde. Oral ≠ écrit, style très différent du français cible assistant.
- **Taille** : 1,5M mots = ~6 Mo texte → petit vs 4 Go, mais 37,5 Go avec audio (inutile pour modèle texte).

**Verdict** : **En attente / évaluation seulement**, même conclusion que ALIPE mais avec risque PII/mineurs plus élevé (100+ enfants). Si licence NC, rejet commercial. Si BY-SA et usage recherche non commercial, possible petit split évaluation oral, mais pas pré-entraînement général. Exiger audit par sous-corpus (adultes-enfants vs adultes-adultes), filtrage PII, anonymisation vérifiée, et avis éthique.

## 3. Comparaison et recommandation pour BlackCode FR-260M

| Critère | ALIPE | TCOF | Décision projet |
|---|---|---|---|
| Taille texte | 90k-250k mots (~1 Mo) | 1,5M mots (~6 Mo) | négligeable vs 4 Go |
| Enfants | 3 | 99-200 | risque mineurs élevé |
| Licence annoncée | CC (BY-SA 4.0 selon blog) | CC BY-NC-SA 2.0 selon blog, BY-SA selon Ortolang | vérifier fiche exacte, NC = écarté commercial |
| PII | noms propres présents, anonymisation ultérieure | explicitement non anonymisé en source | filtrage renforcé nécessaire |
| Style | oral spontané familial, CDS | oral spontané, adulte-enfant + adulte-adulte | complémentaire mais très différent écrit |
| Usage possible | éval oral / SFT limité recherche | éval oral / SFT limité recherche | pas pré-entraînement principal |

**Recommandation** :

- **Ne pas intégrer ALIPE/TCOF dans le corpus de pré-entraînement principal** (FineWeb2-HQ / Common Corpus) sans :
  1. Vérification fiche Ortolang exacte pour licence (BY-SA vs BY-NC-SA) et conditions commerciales.
  2. Vérification anonymisation effective et absence PII (noms, adresses, etc.).
  3. Avis juridique/éthique sur données enfants (consentement, base légale, information, durée conservation, droit retrait, RGPD art. 8).
  4. Filtrage PII avec nos regex + revue humaine.

- **Usage possible restreint** : petit jeu d'évaluation compréhension oral spontané (si licence compatible), ou SFT très limité recherche non commerciale, avec exclusion prénoms et PII, et documentation provenance.

- **Alternative recommandée pour oral** : corpus adultes-adultes de TCOF (sans enfants) si licence BY-SA et PII filtré, ou corpus Claire-Dialogue adultes (Assemblée nationale Licence Ouverte 2.0) — moins risqué.

- **Pour dynamique familiale / jeunes locuteurs** : si besoin réel, créer données synthétiques inspirées du style CDS (phrases courtes, syntaxe simple) mais sans reprendre verbatim enfants, ou utiliser Claire sous-parties adultes.

## 4. Protocole si audit approfondi souhaité

```bash
# Audit ALIPE (si disponible sur HF ou Ortolang)
python scripts/audit_fr_source.py --dataset ortolang/alipe --limit 500 --samples-out data/audits/alipe.jsonl

# Audit TCOF
python scripts/audit_fr_source.py --dataset atilf/tcof --limit 1000 --samples-out data/audits/tcof.jsonl
# Vérifier champs : age, prénom, noms propres, PII, licence
```

Vérifier manuellement 20 extraits masqués pour PII/mineurs, et consigner dans docs/17.

## 5. Sources

- ALIPE : http://lrl-diffusion.univ-bpclermont.fr/alipe/ , http://hdl.handle.net/11041/alipe-000853 , https://lrl.uca.fr/projet_du_labo/alipe-fr/ [recherche 18 oct 2026]
- TCOF : https://www.cnrtl.fr/corpus/tcof/ , https://hdl.handle.net/11403/tcof/v2 , https://tcof.atilf.fr/ (626 corpus, 146h, 1 542 562 mots), https://lbourdois.github.io/blog/audio/dataset_audio_fr/ (TCOF 146H CC BY-NC-SA 2.0)
- Conventions transcription TCOF : https://repository.ortolang.fr/api/content/tcof/12/TCOFConventions2017.pdf ("Ne pas anonymiser...")
- HAL ALIPE/TCOF : https://hal.science/hal-00523397 , https://hal.science/hal-00591427

