# Architecture BlackCode

## 1. Contraintes de conception

| Contrainte | Valeur | Conséquence |
|---|---|---|
| RAM mesurée | ≤ 1 Go | poids + KV + runtime C |
| Entraînement | 1× T4 16 Go | distillation/QAT, pas de pré-entraînement 1 Md |
| Usage | code + généraliste | tokenizer byte-level 32k, contexte 8k |
| Matériel cible | CPU x86/ARM | convolutions + GEMV ternaire, pas de gros GEMM |
| Objectif | ne pas réduire la taille | 1,19 Md de paramètres, stockés en 2 bits |

## 2. Vue d'ensemble

```
 tokens (int32)
      │
 ┌────▼─────────────────────────────┐
 │ Embedding 32768 × 2048, int8     │  67,1 M
 └────┬─────────────────────────────┘
      │  ×6 groupes de 4 blocs
 ┌────▼─────────────────────────────┐
 │ Bloc convolution (×3 par groupe) │
 │  RMSNorm → in_proj d→2d          │
 │  conv1d depthwise k=4            │
 │  × gate → out_proj d→d           │
 │  résidu                          │
 │  RMSNorm → SwiGLU d→5632→d       │
 │  résidu                          │
 └────┬─────────────────────────────┘
 ┌────▼─────────────────────────────┐
 │ Bloc attention (×1 par groupe)   │
 │  RMSNorm → GQA 16 têtes / 4 KV   │
 │  RoPE + QK-norm                  │
 │  out_proj d→d, résidu            │
 │  RMSNorm → SwiGLU, résidu        │
 └────┬─────────────────────────────┘
      │
 ┌────▼─────────────────────────────┐
 │ RMSNorm final                    │
 │ LM head lié à l'embedding        │
 └──────────────────────────────────┘
```

**Pourquoi 3 convolutions pour 1 attention ?** Le cache KV ne concerne que 6 couches sur 24. À 8k de contexte, cela fait 96 Mo au lieu de ~400 Mo pour 24 couches d'attention. Les convolutions ont un état de 3 vecteurs par couche (0,2 Mo au total) et sont aussi des opérations `O(T)` que les CPU savent faire. C'est le motif de LFM2 (10 conv + 6 attention) et, dans l'esprit, de Qwen3-Next (3:1 DeltaNet:attention).

## 3. Comptage exact des paramètres

| Composant | Paramètres | Part |
|---|---:|---:|
| Embeddings (liés) | 67,11 M | 5,7 % |
| Blocs d'attention (6) | 270,53 M | 22,8 % |
| Blocs de convolution (18) | 849,49 M | 71,5 % |
| Normes | 0,10 M | 0,01 % |
| **Total** | **1 187,24 M** | 100 % |
| dont hors embeddings | 1 120,13 M | 94,3 % |

Détail d'un bloc (d=2048, ffn=5632, k=4) :

- convolution : `2d²` (in_proj) + `d·k` (conv) + `d²` (out_proj) + `3d·ffn` (SwiGLU) = 47,19 M ;
- attention : `d²` (q) + `2d·(4×128)` (k,v) + `d²` (o) + `3d·ffn` = 45,09 M.

Le SwiGLU représente 73 % de chaque bloc. C'est le poste à quantifier en priorité ; les convolutions et les normes restent en fp16 car elles sont minuscules.

## 4. Budget mémoire

Mesures du planificateur (`blackcode.planner`), poids 2 bits, embeddings 8 bits, runtime 180 Mo :

| Contexte | Poids | Embeddings | KV | État conv | Runtime | **Total** | Marge / 1 Go |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 048 | 269,1 | 64,1 | 24,0 | 0,2 | 180 | **537,5** | 486,5 |
| 4 096 | 269,1 | 64,1 | 48,0 | 0,2 | 180 | **561,5** | 462,5 |
| 8 192 | 269,1 | 64,1 | 96,0 | 0,2 | 180 | **609,5** | 414,5 |
| 16 384 | 269,1 | 64,1 | 192,0 | 0,2 | 180 | **705,5** | 318,5 |

Le cache KV coûte **12 Ko/token** (6 couches × 2 × 4 têtes × 128 dims × 2 octets). En int8, il tombe à 6 Ko/token ; c'est une option d'export à ajouter si l'on veut 32k de contexte.

### 4.1 Le runtime est un poste à part entière

Les chiffres ci-dessus supposent un **runtime C minimal** (~100–180 Mo). Mesuré dans ce dépôt (`python -m blackcode.bench`) : un processus PyTorch occupe **499,7 Mo de RSS après import**, 524,0 Mo après chargement d'un modèle de 2,16 M, et 537,3 Mo au pic de génération. Autrement dit, **~500 Mo sont consommés avant même les poids du modèle**. Avec PyTorch, un modèle de 1,19 Md (430 Mo de poids + KV) frôlerait 1 Go sans marge ; avec un runtime C à 80 Mo, le total tombe à ~510 Mo. Deux conséquences :

1. le budget « 1 Go » doit être compris comme **poids + KV + runtime C**, pas comme un chiffre PyTorch ;
2. le moteur d'inférence est un livrable de première classe, pas un détail. Le format `.bcpkg` et le plan de runtime C sont décrits plus bas.

Avec un runtime C de 80 Mo, le total à 8k tombe à ~510 Mo, ce qui laisse de la place pour 16k de contexte ou une mémoire à clés-produit.

### 4.2 Comparaison à budget égal

| Solution | Paramètres | Poids | KV 8k | Runtime | Total estimé | Tient sous 1 Go ? |
|---|---:|---:|---:|---:|---:|:---:|
| Qwen2.5-Coder-1.5B, Q4_K_M | 1,54 Md | ~1,1 Go | ~0,1 Go | 0,1 Go | ~1,3 Go | non |
| Qwen3-0.6B, Q4 | 0,6 Md | ~0,4 Go | ~0,05 Go | 0,1 Go | ~0,55 Go | oui (mais 0,6 Md) |
| BitNet 2B4T, 2 bits | 2,4 Md | ~1,19 Go | ~0,15 Go | 0,1 Go | ~1,4 Go | non |
| **BlackCode-1B, 2 bits** | **1,19 Md** | **~0,33 Go** | **0,096 Go** | **0,08 Go** | **~0,51 Go** | **oui** |

BlackCode est la seule option à dépasser le milliard de paramètres sous 1 Go. C'est exactement l'objectif « ne pas réduire la taille ».

## 5. Choix justifiés, choix écartés

| Choix | Décision | Raison |
|---|---|---|
| Poids ternaires | oui | 8× vs fp16, kernels additifs, écosystème bitnet.cpp |
| Embeddings int8 | oui | 67 M de paramètres ; le ternaire dégraderait trop |
| Embedding lié | oui | −67 M de paramètres, perte <1 % (MobileLLM) |
| SwiGLU | oui | standard, bon rapport qualité/paramètres |
| GQA 16/4 | oui | KV ÷4, qualité proche de MHA |
| QK-norm | oui | stabilise l'entraînement à faible précision |
| RoPE | oui | extrapolation de contexte, pas de paramètres |
| Convolutions courtes | oui | état O(1), CPU-friendly, prouvé par LFM2 |
| DeltaNet/KDA | phase 2 | meilleur long contexte, noyau plus complexe |
| MoE | non | la mémoire totale augmente, incompatible 1 Go |
| Memory layers | phase 3 | capacité sans FLOPs, mais bande passante |
| Récursion (`loop`) | ablation | profondeur gratuite en mémoire, coûteuse en latence |
| Vocab 32k | oui | plateau d'efficacité, embeddings maîtrisés |

## 6. Plan d'ablations

Chaque ablation doit être entraînée sur **le même nombre de tokens** et évaluée sur la perte de validation + un petit banc de code (syntaxe, exact-match, pass@1 sur 50 problèmes unitaires).

| # | Variable | Référence | Variante | Métrique | Hypothèse |
|---|---|---|---|---|---|
| A1 | Précision | ternaire 2 bits | fp16 | val loss, RAM | écart <0,05 natif ; >0,2 si PTQ |
| A2 | Motif | 3:1 conv:attn | 1:1, tout-attention | val loss, KV, tok/s | 3:1 proche, KV ÷4 |
| A3 | Forme | 24×2048 | 12×2896 | val loss | profond ≥ large |
| A4 | Embeddings | liés | non liés | val loss, params | −5,7 % params, perte <1 % |
| A5 | Loop | 1 | 2, 3 | val loss, latence | +0,5 à 1 %, ×2 latence |
| A6 | Vocab | 32k | 16k, 64k | tokens/fichier, val loss | 32k optimal |
| A7 | Méthode | from scratch | distillation + QAT | tokens nécessaires | ×10–50 moins de tokens |
| A8 | Contexte | 2k | 8k, 16k | pass@1 long contexte | 8k suffit au code |
| A9 | Embedding bits | 8 | 16, 4 | val loss, RAM | 8 bits = bon compromis |
| A10 | KV | fp16 | int8 | latence, qualité | int8 acceptable |

Les ablations A1–A6 tiennent sur `t4_small` (51 M) en quelques heures chacune. A7 est la plus coûteuse et se fait sur l'étudiant de production.

## 7. Runtime d'inférence

Le format `.bcpkg` (voir `blackcode/export.py`) contient :

- un header JSON (config, table des tenseurs, offsets) ;
- les matrices ternaires packées 2 bits + une échelle fp16 par ligne ;
- les embeddings int8 + échelle par ligne ;
- les normes, convolutions et biais en fp16.

Le moteur cible (`runtime/`, phase 4) est un **C99 sans dépendance** :

1. `mmap` du fichier, pas de copie ;
2. embedding lookup int8 ;
3. RMSNorm en fp32 ;
4. GEMV ternaire : activations int8, poids 2 bits, accumulation int32, déquantification par ligne ;
5. convolution depthwise `k=4` ;
6. attention GQA avec cache KV ;
7. échantillonnage top-k.

Objectif : **< 80 Mo de RSS** hors poids, et parité numérique avec PyTorch à 1e-3 près. Tests : mêmes tokens → mêmes logits sur 100 pas, sur le modèle `nano` puis sur `t4_small`.

Pourquoi pas llama.cpp ? Son graphe ne connaît pas nos blocs de convolution courts ; l'ajouter demanderait un fork. Un moteur dédié de ~1 500 lignes est plus simple et plus contrôlable. `bitnet.cpp` reste une référence pour les kernels ternaires et une source d'inspiration pour les boucles GEMV.

## 8. Risques et parades

| Risque | Probabilité | Impact | Parade |
|---|---|---|---|
| Le ternaire à 1,2 Md est nettement sous le fp16 | moyenne | élevé | distillation longue, export 4 bits de secours, passer à 1,5 Md |
| Le runtime C dépasse 1 Go | faible | élevé | mesurer tôt, `mmap`, pas de copies, KV int8 |
| Les convolutions nuisent au rappel long | moyenne | moyen | ablation A2, ajouter 2 couches d'attention |
| Le T4 ne suffit pas pour la distillation | moyenne | élevé | professeur 1,5–3 B, logits précalculés, sessions multiples |
| Le tokenizer 32k est inefficace sur le code | faible | moyen | mesurer tokens/fichier, merge rules code |
| Surapprentissage du benchmark | élevée | moyen | LiveCodeBench, tests cachés, revue manuelle |

## 9. Ce qui rend cette architecture défendable

1. Chaque choix est adossé à un résultat publié et à un chiffre de budget.
2. Les trois leviers (précision, hybride, embeddings liés) sont **orthogonaux** : les gains se multiplient.
3. Le modèle reste **gros** (1,19 Md) : on ne compense pas la médiocrité par la petitesse.
4. Le budget est calculé **avant** l'entraînement, et vérifié après par l'export.
5. Le point faible — le runtime — est identifié et planifié comme un livrable, pas caché.
