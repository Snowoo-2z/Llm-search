# SFT français — préparation

## Objectif

Préparer un corpus de réglage supervisé (SFT) en français uniquement, sans Discord, avec traçabilité licence et qualité.

## Sources candidates (voir docs/17)

- **Claire** (OpenLLM-France/Claire-Dialogue-French-0.1) : ~37k conversations, sources parlementaires, théâtre, entretiens. Licences hétérogènes (Licence Ouverte 2.0 pour AN, CC BY-NC-SA pour certains). Accès conditionné. Ne pas importer en bloc.
- **Légitus SFT** (moncefem/legi-instruct-fr) : 27k conversations juridiques, abstention, citations, RAG, outils. Apache 2.0 pour compilation, textes légaux Licence Ouverte. Bon pour tests citation/abstention, pas généraliste.
- **Aya Dataset** : 204k annotations humaines multilingues, ~1 422 FR selon Lucie. Apache 2.0 annoncé, qualité humaine intéressante.
- **Luth-SFT** : 571k lignes, splits scholar, croissantllm, smoltalk2, aya, tulu3, openhermes. Hétérogène, beaucoup synthétique.
- **French Instruct** : 276k dialogues, 29% humains, 71% générés, 13 sources. Licence à vérifier par sous-source.

**Décision** : pas de mélange complet sans audit split par split. Séparer humain / traduit / synthétique, vérifier licence, PII, mineurs, consentement.

## Pipeline SFT

### 1. Collecte locale

Placer des fichiers `.jsonl` avec format `{"prompt": "...", "response": "..."}` ou `.md` dans `data/sft_seed/`.

Exemple :
```jsonl
{"prompt": "Explique en français la différence entre sauvegarde et synchronisation.", "response": "Une sauvegarde copie..."}
```

### 2. Filtrage qualité

- `blackcode/data.py` : `quality_reasons` + `quality_score` + `is_french_prose`
- Rejet : PII, secrets, boilerplate, urls, ponctuation répétée, prose trop courte
- `french-only` : lingua si dispo, sinon fallback heuristique (stopwords + accents)

### 3. Build

```bash
python scripts/build_sft_fr.py --src data/sft_seed --out data/sft_fr --max-mb 100
# ou HF
python scripts/build_sft_fr.py --hf OpenLLM-France/Claire-Dialogue-French-0.1 --hf-field content --out data/sft_fr
```

Sortie : `train.bin`, `val.bin`, `tokenizer.json`, `meta.json`, `manifest.jsonl` avec provenance.

### 4. Trajectoires agentiques françaises

`blackcode/data.py` génère 3 exemples par fonction Python retenue :
- explication rôle
- version lisible
- création fichier + lecture + tests (outils `write_file`, `read_file`)

Système : `Tu es BlackCode, un assistant qui parle uniquement en français...`

Pour traces réellement exécutées :
```bash
python -m blackcode.data build --src data/fr_seed --out data/v2 --agentic-frac 0.12 --agentic-exec --max-spawns 200
```

`mutated_failure_example` mute opérateur comparaison AST, exécute, ne garde trace que si échec réel.

### 5. Évaluation SFT

- Séparer train/val/test, ne pas recycler `benchmarks/agent_fr_v2.jsonl`
- Mesurer : langue réponse (français exclusif), exactitude, abstention quand contexte manque, respect consignes, choix outil, arguments, ordre
- Benchmarks : `scripts/validate_agent_tasks.py`, `scripts/eval_agent_benchmark.py`

## Limites

- Traductions anglaises de faible qualité, faits datés, réponses inventées
- Licences manquantes / hétérogènes
- Données extraites sans consentement documenté, mineurs
- Fuites benchmarks si SFT contient tests

## Prochaines étapes

- Auditer Claire par sous-corpus (AN Licence Ouverte OK, CC BY-NC-SA KO pour commercial)
- Auditer Légitus : vérifier citations juridiques, dates, outils
- Créer jeu SFT interne 10k exemples français de qualité, avec sources traçables
- Fine-tuner modèle 260M après pré-entraînement, mesurer perplexité FR et taux français

