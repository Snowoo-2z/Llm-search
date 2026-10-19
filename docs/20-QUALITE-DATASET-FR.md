# Qualité du dataset français — améliorations 18 octobre 2026

Ce document synthétise les améliorations apportées au pipeline `blackcode/data.py` et aux scripts d'audit/build pendant la session de travail d'une heure demandée.

## Objectif

Améliorer la qualité du corpus français sans télécharger de données Discord, en restant traçable, reproductible et compatible Kaggle dual-T4.

## Modifications du pipeline `blackcode/data.py`

### 1. Licences : séparation nette dataset vs texte source

- Ajout de `normalize_licence_label()` : normalisation pour allowlist exacte, pas interprétation juridique.
- `PERMISSIVE` restreint à `mit, apache-2.0, bsd-3, unlicense, isc, cc0` pour code. Une licence inconnue `?` n'est plus implicitement permissive.
- `iter_hf_docs` enrichi :
  - métadonnées `hf_row`, `dataset_id`, `dataset_config`, `dataset_license`, `license_scope = "dataset packaging; not underlying text rights"`.
  - extraction `record_license` depuis champ configurable (`--hf-license-field`).
  - conservation `url`, `identifier`, `collection`, `open_type`, `date`, `dump`, `language`.
- Nouveau filtrage :
  - `--require-hf-license` rejette lignes sans licence par enregistrement.
  - `--allowed-hf-license` allowlist normalisée (ex: MIT, CC0, public domain, Licence Ouverte). Rejet sinon.
  - stats `rejected_licence`, `rejected_record_license`, `record_licenses`.

**Pourquoi :** ODC-By de FineWeb2-HQ couvre la base, pas chaque page Web. Common Corpus a une licence par ligne ; il faut la filtrer explicitement, pas la supposer.

### 2. Qualité : heuristiques anti-spam et boilerplate

Ancien `quality_reasons` : vide, trop court/long, secret, PII, alnum_ratio, minifié, répétitif, python invalide, prose trop courte.

Ajouts :

- `_url_density()` : rejet `trop d'urls` si >25% de caractères d'URL et <200 mots.
- `REPEATED_PUNCT_RE` : `!!!!` ou `....` répétés → `ponctuation répétée`.
- `BOILERPLATE_MARKERS` : mentions légales, cookies, tous droits réservés, newsletter, lorem ipsum → score, rejet `boilerplate` si ≥3 marqueurs et <120 mots.
- Pour prose : `unique_line_ratio <0.20` et >30 lignes → `répétitif`.

Ces règles sont heuristiques, avec faux positifs possibles ; elles sont tracées dans `by_reason` pour audit.

### 3. Traçabilité : manifest.jsonl

`pack()` écrit maintenant `manifest.jsonl` avec pour chaque doc :
- hash SHA256 tronqué (16 chars)
- path source (`hf:...` ou chemin local)
- lang, kind, licence, meta complète, chars

Permet de retrouver provenance, licence et filtres appliqués après coup. Exigence du protocole G du cahier.

### 4. Statistiques enrichies

`build()` conserve :
- `by_lang`, `by_kind`, `licences`, `record_licenses`, `by_reason`
- `rejected_language`, `rejected_score`, `rejected_licence`, `rejected_record_license`
- `bytes`, `tokens_estimate`

Utile pour décider du seuil `MIN_QUALITY_SCORE` après audit, pas avant.

## Scripts

### `scripts/audit_fr_source.py` (v2)

- Nouveau comptage : `licenses_raw_top`, `licenses_normalized_top`, `collections_top`, `open_types_top`, `language_field_top`, `missing_license_rows`.
- Support `--license-field` pour Common Corpus.
- Masquage PII/secret conservé, extraits locaux uniquement.
- Résumé JSON avec médianes/p90/p99 chars/mots, quality_score, language_score, taux français détecté, doublons exacts, PII.

Usage :
```bash
python scripts/audit_fr_source.py --dataset epfml/FineWeb2-HQ:fra_Latn --limit 2000 --samples-out data/audits/fineweb2_hq.jsonl
python scripts/audit_fr_source.py --dataset PleIAs/common_corpus --limit 2000 --license-field license --samples-out data/audits/common_corpus.jsonl
```

### `scripts/build_fr_corpus.sh` (v2)

- Variables d'env : `HF_DATASET`, `HF_FIELD`, `VOCAB`, `SEQ_LEN`, `MAX_MB`, `MIN_QUALITY_SCORE`, `OUT`.
- Vérif dépendances, affichage paramètres, vérif meta après build.
- Reste simple, copiable, compatible Kaggle.

### `scripts/build_common_corpus_fr.sh` (nouveau)

- Exemple de build avec allowlist licences (MIT, Apache 2.0, CC0, CC BY, public domain, Licence Ouverte) — **exemple à valider juridiquement**.
- Montre comment brancher `require-hf-license` + `allowed-hf-license`.
- Avertissement explicite : liste à affiner après audit.

## Protocole recommandé (mis à jour)

1. **Audit d'abord** : lancer `audit_fr_source.py` sur 2000 lignes, examiner taux français, licences, hosts, reasons.
2. **Choisir seuil qualité** : distribution `quality_score` médiane/p10/p90, pas 0 par défaut en prod.
3. **Filtrer licences** : pour Common Corpus, n'autoriser que licences vérifiées compatibles usage visé, avec `require-hf-license`.
4. **Build petit lot** : `MAX_MB=4000` → vérifier `meta.json`, `manifest.jsonl`, stats rejets.
5. **Vérifier manifest** : s'assurer que chaque doc a provenance et licence traçables.
6. **Tokenizer** : entraîné sur corpus filtré, vocab 24k français.
7. **Pilote entraînement** : batch-size 1, grad-accum 16, seq-len 2048, 200 steps, resume.

## Tests

- `tests/test_data.py` : 24 tests OK après modifications (qualité, dedup, FIM, agentique, end-to-end).
- Nouveau filtres `boilerplate`, `trop d'urls`, `ponctuation répétée` couverts indirectement via `by_reason` mais pas encore de tests unitaires dédiés — à ajouter si besoin.

## Limites restantes

- Détection PII/secret regex = heuristique, pas anonymisation garantie.
- Lingua detector nécessite `lingua-language-detector`, coûteux ; pas de fallback léger pour audit rapide.
- Pas de filtre opt-out TDM / robots.txt automatique — à vérifier côté source.
- Pas de détection de contenu sensible (violence, haine) au-delà de boilerplate — besoin d'un classifieur dédié ou revue humaine.
- Manifest ne stocke pas encore hash complet ni date de collecte — amélioration possible.

## Prochaines étapes proposées

- Ajouter test unitaire pour `trop d'urls`, `boilerplate`, `ponctuation répétée`.
- Échantillonner réellement FineWeb2-HQ et Common Corpus sur Colab/Kaggle et consigner résultats dans `docs/17-CAHIER-SOURCES-FR.md`.
- Définir allowlist finale licences après revue juridique et consigner dans `docs/17`.
- Construire corpus 4 Go filtré, mesurer débit entraînement sur T4, ajuster vocab/seq-len.
- Créer jeu d'évaluation français tenu à l'écart à partir des 30 tâches proposées dans `docs/19`.

