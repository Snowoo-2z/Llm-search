# Plan d'entraînement historique — cibles 51 M à 1,2 Md

> Ce document ne pilote plus le modèle actif. La nouvelle cible française de 258,6 M entraînée de zéro est décrite dans [`docs/16-MODELE-FR-260M.md`](16-MODELE-FR-260M.md). Les estimations ci-dessous restent des références historiques, pas des promesses de durée pour cette nouvelle architecture/dataset.

Tous les chiffres ci-dessous sortent de `blackcode.timing` (FLOPs de la config
réelle) et des débits mesurés sur ce projet. Les hypothèses sont explicites :
**MFU 35 %**, T4 à 65 TFLOPS fp16. Le seul chiffre qui les remplacera est un
`--log-flops` en début de session.

---

## 1. Le constat qui décide de tout

| Modèle | Tokens | Contexte | Durée estimée | Sessions Colab (10 h) |
|---|---:|---:|---:|---:|
| `t4_small` (51 M) | 1 Md | 1024 | **4,0 h** | 1 |
| `t4_base` (127 M) | 3 Md | 2048 | **30,7 h** | 4 |
| `blackcode_0.6b` (589 M) | 1 Md | 1024 | **44,3 h** | 5 |
| `blackcode_0.6b` (589 M) | 11,8 Md (20 tok/param) | 2048 | **~26 j** | 62 |
| `blackcode_1.2b` (1,22 Md) | 1 Md | 2048 | **3,9 j** | 10 |
| `blackcode_1.2b` (1,22 Md) | 24 Md (20 tok/param) | 2048 | **~90 j** | — |

**Conclusion : le pré-entraînement from scratch de 0,6 à 1,2 Md est hors de
portée d'un T4 gratuit.** Ce n'est pas une question d'optimisation : c'est un
facteur 60 à 90 sur le temps disponible. Toute stratégie qui prétend le
contraire se heurtera au mur.

Deux chemins restent, et ils se combinent :

1. **Entraîner petit et bien** (`t4_small`, 51 M) pour valider toute la chaîne ;
2. **Distiller** un modèle plus gros depuis un professeur déjà entraîné.

| Étape | Tokens | Durée estimée |
|---|---:|---:|
| Distillation `blackcode_0.6b` depuis StarCoder2-3B | 30 M | **4,1 h** |
| Distillation `blackcode_1.2b` depuis StarCoder2-3B | 50 M | **9,6 h** |

C'est le levier principal : **30 M tokens distillés ≈ 11,8 Md tokens de
pré-entraînement**, soit un facteur ~390 sur le budget de calcul, au prix d'un
plafond : l'élève ne dépassera pas son professeur.

---

## 2. Pourquoi StarCoder2 comme professeur

Deux conditions sont nécessaires pour distiller au niveau des logits :

1. le professeur doit tenir sur un T4 (16 Go) → 3 Md en 4 bits ≈ 2 Go ;
2. le professeur doit **partager notre tokenizer** → StarCoder2 utilise 49 152
   jetons, exactement la taille choisie pour BlackCode.

Le cours du raisonnement est dans `docs/10-DONNEES-V2.md` §1. Si la
vérification du tokenizer sur Colab échoue, le plan de secours est :

- **secours A** : tokenizer StarCoder2 quand même, mais distillation « texte »
  (le professeur génère, on réentraîne sur le texte) → 10× plus lent ;
- **secours B** : entraîner notre propre professeur (impossible sur T4) ;
- **secours C** : renoncer aux logits et faire de l'apprentissage supervisé sur
  données synthétiques de qualité (Qwen2.5-Coder-32B via API) → coût financier.

---

## 3. Les cinq phases, avec critères de passage

### Phase 0 — terrain (aujourd'hui, 1 session courte)

```bash
bash scripts/colab_setup.sh
python -c "from transformers import AutoTokenizer as A; t=A.from_pretrained('bigcode/starcoder2-3b'); print(len(t))"
python -m blackcode.data stats --src data/raw --max-mb 200
```

**Critère de passage** : `len(t) == 49152`, et le pipeline tourne sans erreur.

### Phase 1 — la chaîne complète sur petit modèle (1 session)

```bash
python -m blackcode.data build --src data/raw --out data/v2 \
  --tokenizer hf:bigcode/starcoder2-3b --seq-len 2048 \
  --fim-rate 0.5 --agentic-frac 0.05 --agentic-exec --max-spawns 500

python -m blackcode.train --config configs/t4_small.json --data data/v2 \
  --out runs/p1_small --batch-size 8 --grad-accum 4 --seq-len 1024 \
  --max-steps 3000 --optim muon --schedule wsd \
  --checkpoint-policy selective --compile --log-flops --resume \
  --mirror-dir /content/drive/MyDrive/blackcode/runs/p1_small
```

3 000 pas × 32 768 tokens ≈ 98 M tokens, ~30 min sur T4.
**Critère de passage** : `val` baisse régulièrement, le MFU affiché est > 20 %,
et `runs/p1_small/best.pt` se charge pour générer du code.

### Phase 2 — distillation de `blackcode_0.6b` (1 session)

```bash
python -m blackcode.distill --teacher bigcode/starcoder2-3b --load-4bit \
  --config configs/blackcode_0.6b.json --data data/v2 \
  --out runs/p2_distill_06b --seq-len 2048 --batch-size 2 --grad-accum 8 \
  --max-steps 916 --topk 32 --alpha 0.5 --temperature 2.0 \
  --optim adafactor --lr 3e-4 --schedule wsd --checkpoint-policy selective \
  --eval-interval 200 --save-interval 200 --resume
```

**Critère de passage** : la KL descend, `val_ce` sous celle de `t4_small` du
même corpus, et une génération manuelle produit du Python syntaxiquement
correct sur 10 prompts.

### Phase 3 — agentique et FIM (1 session)

Reconstruire `data/v2` avec `--agentic-frac 0.10 --agentic-exec --max-spawns 4000`,
puis reprendre la distillation sur 10–20 M tokens supplémentaires en baissant
`--alpha` à 0,3 (plus de CE sur les trajectoires d'outils).

**Critère de passage** : sur 20 tâches à deux outils, le modèle produit le bon
format `tool(...)` et le bon nom de fichier au moins 15 fois.

### Phase 4 — ternarisation, export, mesure (30 min, CPU)

```bash
python -m blackcode.export --ckpt runs/p2_distill_06b/best.pt \
  --out artifacts/blackcode06.bcpkg --context 8192
./runtime/bcrun --pkg artifacts/blackcode06.bcpkg --ids 1 --max-new 32 \
  --context 8192 --mem-report
```

**Critère de passage** : RSS pic < 700 Mo à 8k de contexte, et sortie
cohérente. C'est la seule mesure qui compte pour la cible « ≤ 1 Go ».

---

## 4. Mécanique des sessions Colab

| Contrainte | Conséquence pratique |
|---|---|
| 12 h max, coupure réelle ~11 h | `--session-hours 10` dans le plan, 1 h de marge |
| déconnexion aléatoire en gratuit | `--resume` + `--save-interval` court (200 pas) |
| Drive lent (10–60 s par écriture) | `--mirror-dir` copie en thread, l'entraînement ne bloque pas |
| RAM système 12 Go | le dataset reste sur disque (memmap), pas en RAM |
| GPU partagé | mesurer `--log-flops` en début de session : un T4 partagé peut tomber à 15 % de MFU |

**Règle d'or** : une session = une commande identique, `--resume` fait le reste.
Ne jamais changer `--lr`, `--max-steps` ou `--batch-size` entre deux sessions :
le schedule WSD est calculé sur `max-steps`, le changer invalide la reprise.

---

## 5. Leviers de temps, classés par gain réel

| Levier | Gain | État |
|---|---:|---|
| Distillation (30 M au lieu de 11,8 Md) | **×390** | implémenté (`blackcode/distill.py`) |
| Contexte court au début (`--seq-len-warmup-frac 0.3`) | ×1,3–1,5 | implémenté |
| Muon plutôt qu'AdamW | ×1,3–1,8 (littérature) | implémenté |
| Tokenizer 49 k au lieu de 32 k | ×1,1 (séquences plus courtes) | décidé |
| `--compile` + checkpointing sélectif | +10–25 % de débit | implémenté |
| WSD + reprise | évite de perdre une session | implémenté |
| Deux sessions en parallèle (deux comptes) | ×2 | possible, non automatisé |

Les 3 premiers sont mesurables dans la première session. Le reste est du
paramétrage.

---

## 6. Ce que je fais, ce que tu fais

| Moi | Toi |
|---|---|
| code, tests, docs, plan chiffré | lancer les commandes sur Colab |
| correction des bugs que les runs révèlent | coller les sorties (`meta.json`, `--log-flops`, `--mem-report`) |
| choix techniques argumentés | arbitrer : professeur, licences, publication |

**Ce dont j'ai besoin en priorité** (dans cet ordre) :

1. `len(AutoTokenizer.from_pretrained("bigcode/starcoder2-3b"))` → doit valoir 49152 ;
2. la sortie de `scripts/train_plan.py --stage distill` sur ta machine ;
3. le premier `--log-flops` réel (MFU du T4) ;
4. `data/v2/meta.json` après construction.

Avec ces quatre éléments, je peux remplacer toutes les estimations par des
mesures et fixer les `max-steps` définitifs.

---

## 7. Risques et parades

| Risque | Probabilité | Parade |
|---|---|---|
| Tokenizer StarCoder2 incompatible | faible | vérification en phase 0, secours A |
| OOM : élève 1,2 Md + professeur 3 Md | moyenne | commencer par 0,6 Md ; `--seq-len 1024` ; checkpointing |
| `blackcode_1.2b` ne tient pas sous 1 Go | faible | mesuré après export ; repli sur 0,6 Md |
| Le français reste faible | moyenne | part FR à 8 %, filtre qualité manuel |
| Contamination des benchmarks | certaine | évaluer sur LiveCodeBench + tests cachés |
| Distillation plafonnée par StarCoder2-3B | certaine | changer de professeur (7B en 4 bits) une fois la chaîne validée |
