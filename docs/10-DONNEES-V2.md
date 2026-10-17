# Dataset v2 — bilingue français/anglais, spécialisé code et agentique

Objectif : un corpus qui rend un modèle de 0,6–1,2 Md **compétitif dans son
segment**, pas un modèle généraliste moyen. Trois choix structurent tout le
reste : le tokenizer, la composition, et la traçabilité.

---

## 1. La décision qui commande les autres : le tokenizer

**Adopter le tokenizer de StarCoder2 (49 152 jetons) au lieu d'entraîner le
nôtre.**

Raison : pour distiller un professeur, les logits de l'élève et du professeur
doivent être comparés **position par position**. Cela n'est possible que si les
deux partagent exactement le même vocabulaire. Un BPE réentraîné, même à 49 152
jetons, a des identifiants différents : la distillation au niveau des logits
devient impossible, et il ne reste que la distillation « texte », beaucoup plus
coûteuse (il faut faire générer le professeur, token par token).

Conséquences :

| | BPE maison (v1) | Tokenizer StarCoder2 |
|---|---|---|
| Distillation logits | impossible | **possible** |
| Temps d'entraînement du tokenizer | ~2 min | 0 |
| Vocabulaire | 32 000 | 49 152 |
| Embeddings (d=1536) | 49 M | 75 M |
| Longueur moyenne d'un fichier | référence | ~8–12 % plus court (mieux découpé) |

Les jetons spéciaux de BlackCode (`<|user|>`, `<|tool|>`, FIM…) sont **ajoutés à
la fin** du vocabulaire : aucun identifiant existant ne bouge, donc les 49 152
premiers logits du professeur restent valides. Le reste est ignoré par la KL.

### À vérifier une fois sur Colab

```python
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("bigcode/starcoder2-3b")
print("taille :", len(t))                 # attendu 49152
for s in ["<|endoftext|>", "<fim_prefix>", "<fim_middle>", "<fim_suffix>"]:
    print(s, t.convert_tokens_to_ids(s))
```

Si les identifiants FIM diffèrent de ceux supposés, `blackcode/data.py` les
détecte automatiquement (`resolve_fim_tokens`) — mais la vérification manuelle
reste la bienvenue : **copier la sortie dans le chat**.

---

## 2. Composition cible

Un corpus de code n'est pas un corpus de conversation : le modèle doit savoir
*écrire* du code, *lire* un dépôt, *appeler des outils* et *expliquer* en
français comme en anglais.

| Catégorie | Part | 1 Md tokens | Rôle |
|---|---:|---:|---|
| Python | 30 % | 300 M | langage principal, le plus dense en logique |
| JavaScript + TypeScript | 15 % | 150 M | web, le plus demandé en agentique |
| Rust + Go + C + C++ + Java | 15 % | 150 M | systèmes, structures différentes |
| Tests, CI, configuration (YAML, TOML, Dockerfile) | 5 % | 50 M | comportement de dépôt réel |
| Documentation technique anglaise | 12 % | 120 M | explications, README, docstrings |
| **Documentation technique française** | **8 %** | **80 M** | **la partie bilingue** |
| Agentique : trajectoires d'outils, corrections | 10 % | 100 M | format `<|tool|>`, boucles de test |
| Divers (SQL, shell, HTML/CSS) | 5 % | 50 M | couverture |

La part française est volontairement à 8 % et non à 50 % : un modèle bilingue
n'a pas besoin d'autant de français que d'anglais pour *répondre* en français,
mais il en a besoin pour ne pas traduire mot à mot. En dessous de 5 %, le
français se dégrade nettement.

---

## 3. Ce que le pipeline fait réellement

`blackcode/data.py` (nouveau, remplace `corpus.py` pour le v2) :

1. **Filtre qualité** — refuse : fichiers générés (`DO NOT EDIT`), minifiés
   (ligne moyenne > 160 caractères), Python qui ne compile pas, secrets (clés
   privées, `sk-…`, `AKIA…`, `ghp_…`), code répétitif, prose de moins de 30 mots.
2. **Déduplication exacte** (SHA-1 après normalisation) **et quasi-dupliquée**
   (SimHash 64 bits sur shingles de lignes, LSH 4 × 16 bits, seuil de Hamming 3).
3. **Détection de licence** par SPDX ou fichier `LICENSE` du dépôt.
4. **FIM** (`<fim_prefix>…<fim_suffix>…<fim_middle>`) sur 50 % des documents :
   c'est ce qui rend le modèle utilisable en complétion dans un IDE.
5. **Agentique** construit sur du code réel :
   - tâches d'implémentation à partir des signatures et docstrings ;
   - scénarios d'outils (`write_file`, `read_file`, `run_tests`) au format exact
     utilisé à l'inférence ;
   - **scénarios de correction exécutés** : on mute un opérateur de comparaison
     au niveau AST, on exécute original et mutant sur des entrées déduites des
     valeurs par défaut, et on n'écrit la trace **que si les résultats diffèrent
     réellement**.

### Mesures faites dans cet environnement

Construction complète de `data/v2` sur 1,2 Go de sources (97 sdists PyPI,
7 dépôts GitHub, 2 dépôts français) :

| Mesure | Valeur |
|---|---|
| Volume brut lu | 1,2 Go |
| Volume retenu après filtres | **661 Mo** (55 %) |
| Documents | **51 233** |
| **Tokens train** | **174,3 M** |
| Tokens validation | 3,54 M (2 %) |
| Documents avec FIM | 24 529 (48 %) |
| Exemples agentiques | 1 212 |
| Longueur médiane d'exemple | 1 657 tokens (p90 8 193) |
| Répartition | code 39 912 / prose 10 109 / agentique 1 212 |
| Durée totale | **10 min** sur 2 cœurs (filtre, dédup, tokenizer, tokenisation) |

Le corpus v2 fait donc **2× le v1** (174 M contre 87 M tokens), à volume brut
comparable : le v1 ne gardait que du Python de 97 paquets, le v2 ajoute les
dépôts GitHub complets, la documentation et les deux dépôts français.

| Autre mesure | Valeur |
|---|---|
| Taux de rejet qualité | ~19 % des fichiers |
| Taux de doublons | ~13 % (exacts + empreinte + SimHash) |
| Rendement | ~145 M tokens par Go de sources (vocab 49 152) |
| Fonctions produisant une **vraie** trace d'échec | **6,5 %** (13 / 200 exécutées) |
| Exemples agentiques par fonction | 3 (2 rédactions + 1 outil) |

Le taux de 6,5 % paraît faible, mais il est *cumulatif* : sur 40 000 fichiers
Python, cela représente environ 13 000 trajectoires de débogage réellement
exécutées — impossibles à obtenir par génération synthétique sans halluciner.

---

## 4. D'où vient le français

Le français technique est la partie la plus difficile à sourcer proprement.
Trois gisements, par ordre de qualité :

| Source | Volume | Comment |
|---|---:|---|
| Documentation de projets français (beta.gouv, Etalab, CNIL, OpenClassrooms) | 5–50 M tokens | dépôts GitHub, `--src` |
| Wikipédia FR filtrée sur l'informatique | 100–200 M tokens | `wikimedia/wikipedia`, config `20231101.fr`, filtre `--keep-if` |
| Web français filtré | 200 M+ | `HuggingFaceFW/fineweb-2`, config `fra`, filtre mots-clés |

Exemple de filtre pour ne garder que le français *technique* :

```bash
python -m blackcode.data build --src data/raw --out data/v2 \
  --hf "HuggingFaceFW/fineweb-2:fra" --hf-field text \
  --keep-if "fonction|variable|erreur|compil|algorithme|données|protocole" \
  --tokenizer hf:bigcode/starcoder2-3b
```

⚠️ Le filtre `--keep-if` s'applique à **toutes** les sources de l'appel. Pour
mélanger code sans filtre et prose filtrée, faire deux corpus puis les
concaténer (`--src data/code --src data/fr_prose`).

---

## 5. Sources de code à grande échelle

Le corpus actuel (97 paquets PyPI + 7 dépôts GitHub, 1,2 Go) est un bon
démarrage mais reste **mono-langage à 95 %**. Pour un modèle réellement
spécialisé, viser :

| Dataset HuggingFace | Langages | Volume | Accès |
|---|---|---:|---|
| `bigcode/the-stack-v2-dedup` | 600+ | 32 To | accord requis (gated) |
| `bigcode/the-stack-smol` | 30 | 1 Go | libre |
| `bigcode/starcoderdata` | 86 | ~250 Md tokens | gated |
| `codeparrot/codeparrot-clean` | Python | 50 Go | libre |
| `HuggingFaceTB/finemath` (non-code) | — | — | à éviter ici |

Sans accord HuggingFace, la voie réaliste est **PyPI + npm + crates.io +
Go modules**, qui sont publics et sans authentification. `fetch_corpus.py` gère
déjà PyPI et GitHub ; ajouter npm est trivial (registre JSON public).

---

## 6. Licences et contamination

- **Licences** : `--licence permissive` ne garde que MIT, Apache-2.0, BSD, MPL,
  ISC, Unlicense. Un modèle open source peut être entraîné sur du GPL, mais
  publier les poids d'un modèle entraîné sur du GPL est un terrain juridique
  incertain : le choix par défaut du projet est **permissif uniquement pour la
  publication**, avec conservation des autres sources hors du corpus publié.
- **Contamination** : HumanEval/MBPP sont massivement présents dans GitHub.
  Toute évaluation sur ces benchmarks après entraînement sur `data/raw` est
  suspecte. Utiliser **LiveCodeBench** (daté) et des tests cachés maison.
- **Traçabilité** : `manifest.json` enregistre source et licence ; `meta.json`
  enregistre la composition finale (`by_kind`, `by_lang`, `licences`).

---

## 7. Commandes

```bash
# 1. audit d'un répertoire, sans rien écrire
python -m blackcode.data stats --src data/raw --max-mb 400

# 2. construction complète (tokenizer StarCoder2 + FIM + agentique exécuté)
#    --sync-config met à jour les configs avec la taille réelle du vocabulaire
python -m blackcode.data build --src data/raw --out data/v2 \
  --tokenizer hf:bigcode/starcoder2-3b \
  --vocab 49152 --seq-len 8192 --fim-rate 0.5 \
  --agentic-frac 0.10 --agentic-exec --max-spawns 2000 \
  --sync-config configs/blackcode_0.6b.json \
  --sync-config configs/blackcode_1.2b.json

# 3. vérification
cat data/v2/meta.json
```

Le fichier `meta.json` contient : tokens train/val, documents, part FIM, part
agentique, répartition par langage et par type, licences, longueur médiane des
exemples. **C'est ce fichier qu'il faut coller dans le chat** après une
construction sur Colab.

---

## 8. Ce qui reste à faire

| Priorité | Action | Pourquoi |
|---|---|---|
| 1 | vérifier le tokenizer StarCoder2 sur Colab | condition de la distillation |
| 2 | ajouter npm/crates.io à `fetch_corpus.py` | sortir du tout-Python |
| 3 | construire la part française (wikipédia + fineweb-2 filtré) | bilingue réel |
| 4 | filtrer par exécution (tests qui passent) | qualité > quantité |
| 5 | décontaminer HumanEval/MBPP | évaluation honnête |
