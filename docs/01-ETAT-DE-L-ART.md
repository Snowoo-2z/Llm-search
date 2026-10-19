# État de l'art — réduire le coût sans réduire le modèle

> Revue ciblée : « comment obtenir un LLM généraliste et bon en code dans ≤ 1 Go de RAM, avec un T4 pour l'entraînement ? ».
> Les chiffres sont ceux des publications ou des mesures citées ; les affirmations commerciales sont signalées comme telles.

---

## 1. Décomposer le problème : où part la mémoire ?

Pour un modèle décodeur de `N` paramètres, `L` couches, `d` de largeur, `V` de vocabulaire, `T` de contexte :

| Poste | Formule | Ordre de grandeur (modèle 1,2 Md, 8k) |
|---|---|---|
| Poids non-embeddings | `N_non_emb × bits/8` | 280 Mo à 2 bits |
| Embeddings (liés) | `V × d × bits/8` | 67 Mo à 8 bits |
| Cache KV | `L_attn × 2 × n_kv × d_head × T × octets` | 96 Mo (6 couches, 4 KV, fp16) |
| État des convolutions | `L_conv × d × (k−1)` | 0,2 Mo |
| Activations + runtime | ~100–300 Mo | 180 Mo (hypothèse) |

Deux enseignements immédiats :

1. **Le cache KV est un poste de premier ordre** dès qu'on veut du contexte. Avec 24 couches d'attention pleine, 8k de contexte et 4 têtes KV de 128 dims, il coûterait ~400 Mo, soit 40 % du budget. C'est ce qui justifie l'hybride 3:1.
2. **Les embeddings ne sont pas négligeables** à cette échelle : 67 M de paramètres, soit 5,6 % du modèle. Les lier à la sortie et les stocker en int8 est un gain immédiat.

**Mesure importante** : le fichier GGUF de BitNet 2B4T fait ~1,19 Go pour 2,4 Md de paramètres, et une mesure communautaire donne ~1,3 Go de RSS en décodage CPU. Autrement dit, le « 0,4 Go » du papier ne décrit que les poids hors embeddings. Pour tenir **sous 1 Go mesuré**, il faut un modèle d'environ 1,1–1,3 Md de paramètres, pas 2,4 Md. C'est la principale correction apportée au discours habituel sur BitNet.

---

## 2. Axe A — La précision comme variable d'architecture

### 2.1 BitNet b1.58 : le ternaire natif

BitNet b1.58 quantifie les poids en `{-1, 0, +1}` (log₂3 ≈ 1,58 bit) avec une échelle absmean, et les activations en int8. Le point crucial : **le modèle est entraîné dès l'initialisation avec cette quantification**, pas quantifié après coup. Résultats du rapport 2B4T ([arXiv:2504.12285](https://arxiv.org/abs/2504.12285)) :

| Modèle | Params | Mémoire hors emb. | Latence CPU (ms/token) | Moyenne bench. |
|---|---:|---:|---:|---:|
| BitNet b1.58 2B4T | 2,4 Md | **0,4 Go** | **29** | 54,19 |
| Qwen2.5 1.5B | 1,5 Md | 2,6 Go | 65 | 55,23 |
| Llama 3.2 1B | 1,2 Md | 2,0 Go | 48 | 44,90 |
| SmolLM2 1.7B | 1,7 Md | 3,2 Go | 67 | — |

Avec `bitnet.cpp`, Microsoft annonce 6,17× de speedup x86 et −82 % d'énergie (chiffres constructeur). **Le gain de vitesse n'existe que dans les kernels dédiés** : dans `transformers`, le modèle tourne comme un petit modèle fp16, sans bénéfice.

Ce que ça nous apprend :
- à taille égale, le ternaire natif est compétitif avec le fp16 **à partir de ~3 Md** ; à 1,2 Md, l'écart est plus incertain ;
- le gain mémoire est réel et immédiat, indépendamment des kernels ;
- l'entraînement from scratch coûte cher (4 000 Md de tokens pour 2B4T), d'où l'intérêt de la distillation.

### 2.2 Lois d'échelle de précision

Kumar et al. ([arXiv:2411.04330](https://arxiv.org/abs/2411.04330), ICLR 2025) introduisent un **nombre de paramètres effectif** :

```
N_eff = N × (1 − exp(−P / γ))        puis        L = A·N_eff^(−α) + B·D^(−β) + E
```

- À **calcul fixe**, la précision optimale est ~7–8 bits, indépendante du budget ;
- à **taille de modèle fixée**, la précision optimale croît lentement avec le calcul ;
- les gains de précision saturent vers 6–7 bits ;
- **si la mémoire est la contrainte**, on peut échanger des bits contre des paramètres.

Dettmers & Zettlemoyer ([arXiv:2212.09720](https://arxiv.org/abs/2212.09720)) trouvent qu'à **nombre total de bits fixé**, 4 bits est presque toujours optimal pour la précision zero-shot, avec une taille de bloc de 64–128 et un type flottant. Mais ParetoQ ([arXiv:2502.02631](https://arxiv.org/abs/2502.02631)) montre que le ternaire et le 2 bits peuvent se trouver sur la frontière de Pareto, et que la quantification post-entraînement en dessous de 4 bits s'effondre sans entraînement adapté.

**Synthèse** : le 4 bits est le choix sûr si l'on quantifie après coup ; le 2 bits est le choix optimal si l'on entraîne nativement et que la mémoire est la contrainte dure. C'est le pari de BlackCode, et il doit être validé empiriquement — notre planificateur (`blackcode.planner --scan`) donne un optimum théorique à 2 bits, mais la loi est extrapolée en dessous de 4 bits.

### 2.3 Trois voies pour obtenir un modèle ternaire

| Voie | Coût | Qualité attendue | Pertinence T4 |
|---|---|---|---|
| **Pré-entraînement natif** (BitNet) | 100–4 000 Md tokens, dizaines de milliers de GPU-h | référence | irréaliste |
| **PTQ ternaire** (CAT-Q) | 512 échantillons de calibration, 1–60 h sur 8×A100 | bonne, mais sous le natif | trop lourd pour un T4, et qualité limitée |
| **QAT depuis un modèle pré-entraîné + distillation** | quelques GPU-h à quelques centaines | proche du professeur | **voie retenue** |

Deux résultats rendent cette voie crédible :
- **Ternary Mamba** ([arXiv:2606.18114](https://arxiv.org/abs/2606.18114)) : QAT depuis un checkpoint fp16 + distillation, **4 GPU-h sur H100**, 102 M tokens, score 48,1 vs 48,4 pour un modèle entraîné de zéro sur 105 Md tokens. La distillation compense massivement le manque de tokens.
- **CAT-Q** ([arXiv:2606.26650](https://arxiv.org/abs/2606.26650)) : quantification ternaire en 512 échantillons, 100 000× moins de tokens que BitNet, mais pour des modèles ≥1,7 Md et avec 8 A100.

Sur un T4 (16 Go, ~65 TFLOPS fp16 théoriques, pas de bf16 natif), la distillation d'un professeur 7B en 4 bits vers un étudiant 0,6–1,2 Md est faisable, mais le débit du professeur domine. Il faut **précalculer les logits/top-k du professeur** ou utiliser un professeur plus petit (1,5–3 B). Nous détaillons le budget dans `docs/04-PROTOCOLE-T4.md`.

---

## 3. Axe B — Architecture : plus de profondeur, moins d'état

### 3.1 Profond et étroit, embeddings liés

MobileLLM ([arXiv:2402.14905](https://arxiv.org/abs/2402.14905)) entraîne 19 modèles de 125 M et 350 M à largeur/profondeur variable. Résultats :

- à taille égale, **la profondeur bat la largeur** (+2,7 % / +4,3 % sur les benchmarks zero-shot face aux modèles 125 M/350 M de l'époque) ;
- le **partage entrée/sortie** économise 11,8 % des paramètres pour −0,2 à −0,6 % d'accuracy ;
- le **partage de blocs adjacents** (repeat-all-over) ajoute +0,7 à +0,8 % à taille constante ; la répétition ×2 est le meilleur compromis, au-delà les gains s'éteignent ;
- la configuration 350 M gagnante : 32 couches, d=960, 15 têtes, embeddings liés, partage par paires.

Notre config `blackcode_1b` reprend ces principes : 24 couches, d=2048, embeddings liés, option `loop` pour répéter la pile.

### 3.2 Hybrides : convolutions courtes et attention linéaire

Le consensus 2025–2026 est qu'il n'y a pas besoin d'attention pleine partout :

- **LFM2** ([arXiv:2511.23404](https://arxiv.org/abs/2511.23404)) : 16 blocs = 10 convolutions courtes doublement portées + 6 GQA, 32k de contexte, 10 000 Md de tokens. LFM2-1.2B atteint MMLU 55,2 / IFEval 74,9 / GSM8K 58,3, et revendique 2,3–2,8× plus vite en prefill que Qwen3-1.7B. Les convolutions sont particulièrement adaptées aux CPU embarqués.
- **Qwen3-Next** : motif 3:1 Gated DeltaNet → Gated Attention sur 48 couches (36 récurrentes, 12 attention).
- **Kimi Linear** ([arXiv:2510.26692](https://arxiv.org/abs/2510.26692)) : 3:1 KDA (DeltaNet à porte par canal) : MLA. À budget égal, il **bat l'attention pleine** sur le court et le long contexte, réduit le KV de 75 % et accélère le décodage jusqu'à 6× à 1M de tokens.

Pourquoi des convolutions plutôt que DeltaNet dans la v1 ? La convolution courte a un état `O(1)` par couche, un noyau `conv1d` déjà optimisé partout (CPU inclus), et pas de récurrence à implémenter. DeltaNet/KDA est une **amélioration de phase 2** : meilleure récupération à long terme, mais noyau plus complexe et moins portable. Le tableau d'ablation prévoit les deux.

### 3.3 Récursion et supervision profonde

Le Tiny Recursive Model ([arXiv:2510.04871](https://arxiv.org/abs/2510.04871)) : un réseau de **2 couches, 7 M de paramètres**, appliqué récursivement (24 itérations latentes × 16 étapes de supervision) atteint 45 % sur ARC-AGI-1, 87 % sur Sudoku-Extreme, 85 % sur Maze-Hard. Il bat des LLM 1 000× plus gros sur ces tâches. L'analyse indépendante de l'ARC Prize Foundation attribue l'essentiel du gain à la **supervision profonde** (19 % → 39 %) plutôt qu'à la récursion hiérarchique (35,7 % → 39,0 %).

Deux mises en garde : (a) ces tâches sont des puzzles avec beaucoup d'augmentation de données et un budget de calcul d'inférence important ; (b) le résultat ne se transpose pas directement à la génération de code. Mais le principe — **un petit réseau réappliqué plusieurs fois, supervisé à chaque étape** — est directement applicable à un réparateur de code.

---

## 4. Axe C — Ajouter de la capacité sans ajouter de FLOPs

### 4.1 Memory layers à clés-produit

Meta ([arXiv:2412.09764](https://arxiv.org/abs/2412.09764)) remplace certaines couches FFN par une **mémoire clé-valeur entraînable** :

- jusqu'à 128 Md de paramètres de mémoire pour un modèle de base de 8 Md ;
- un modèle de 1,3 Md avec 1 M d'entrées passe de 7,76 à 13,68 d'exact match sur NaturalQuestions ;
- avec 64 M d'entrées, il approche Llama2-7B, entraîné avec 10× plus de FLOPs ;
- gains aussi sur HumanEval et MBPP, pas seulement sur les faits.

Le revers : la mémoire est **limitée par la bande passante**, pas par les FLOPs. Le papier ne rapporte ni temps mural ni énergie. Sous 1 Go, une mémoire de 50 M d'entrées × 64 dims en 2 bits = 12,5 Mo, ce qui est très attractif ; mais chaque token doit lire quelques entrées, et le coût réel dépend du matériel. C'est une piste de phase 3, pas un prérequis.

### 4.2 MoE : à rejeter sous 1 Go

Un MoE réduit les FLOPs par token mais **augmente** la mémoire totale (tous les experts doivent être stockés) et ajoute un routeur. À 1 Go, le budget est déjà consommé par les poids denses. Les memory layers sont une meilleure forme de sparsité pour ce budget, car elles remplacent des FFN au lieu d'en ajouter.

### 4.3 Vocabulaire et tokenizer

La taille du vocabulaire fixe `V × d` paramètres d'embedding. Pour un petit modèle, un vocabulaire de 128k coûte plus cher que plusieurs couches. À l'inverse, un vocabulaire trop petit augmente le nombre de tokens par programme, donc le calcul et le KV. La littérature sur les tokenizers situe le plateau d'efficacité vers **32k** pour un usage général ; pour du code, 32k avec des merge rules adaptées est un bon compromis. Nous utilisons un BPE byte-level : aucun token inconnu, tout programme reste représentable.

### 4.4 Récupération et outils

La connaissance peut rester hors du modèle : documentation, index de dépôt, signatures d'API. Un modèle de 1,2 Md avec un bon retriever peut battre un modèle de 7 Md sans retrieval sur des questions factuelles. C'est aussi la manière la plus simple de rendre BlackCode « généraliste » sans grossir.

---

## 5. Axe D — Entraîner efficacement sur un T4

### 5.1 Distillation

La distillation classique (KL sur les logits du professeur) est **off-policy** : l'étudiant apprend sur des préfixes du dataset. La distillation on-policy (GKD, [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)) fait générer l'étudiant puis corrige avec le professeur ; elle surpasse l'off-policy dans les comparaisons publiées, au prix d'un appel professeur par rollout. Sur T4, le compromis est :

- **professeur 7B en 4 bits** pour 20–30 M tokens de code, logits top-k précalculés (une passe) ;
- **étudiant 0,6–1,2 Md** qui voit ces données 3–5 fois, avec QAT ternaire ;
- éventuellement une phase on-policy courte (5–10 M tokens) à la fin, si le budget le permet.

### 5.2 Données

Les modèles ≤ 1 Md sont **surentraînés** : MobileLLM utilise 1 000 Md de tokens pour 350 M de paramètres (~2 857 tokens/paramètre), LFM2 10 000 Md pour 1,2 Md. Ce n'est pas un hasard : le coût d'inférence domine, donc on préfère un petit modèle très entraîné. Sur T4, on ne peut pas reproduire cela. Il faut compenser par :

- **qualité > quantité** : déduplication agressive, filtrage par exécution (le code qui compile et passe ses tests), « textbooks » synthétiques ;
- **distillation** : le professeur fournit un signal dense qui remplace une partie des tokens ;
- **curriculum** : commencer par des fonctions courtes, puis des fichiers, puis des tâches de réparation.

### 5.3 Mémoire d'entraînement

Pour un étudiant de 0,6 Md : poids fp32 2,4 Go, moments Adam 4,8 Go, gradients 2,4 Go → ~9,6 Go, sans activations. Un T4 de 16 Go ne suffit pas en l'état. Options :

- **Adam 8-bit** (bitsandbytes) : divise les moments par 4 → ~6 Go ;
- **Adafactor** : moments factorisés, ~2,4 Go ;
- **gradient checkpointing** : réduit les activations, +30 % de calcul ;
- **BAdam / block coordinate descent** : un seul bloc a un état d'optimiseur à la fois ;
- **CPU offload** des poids inactifs.

La combinaison « QAT ternaire + Adam 8-bit + checkpointing + batch 1–2 × 1024 » tient dans 16 Go. C'est le protocole de la phase 2.

### 5.4 Le piège des logits

Pour un vocabulaire de 32k et une séquence de 4 096, les logits d'un batch de 8 représentent 8 × 4 096 × 32 000 × 2 octets ≈ 2 Go. C'est souvent le premier poste qui explose, avant les poids. BlackCode calcule la perte **par morceaux de 256 positions** (`forward_loss(logit_chunk=256)`), ce qui rend la mémoire indépendante de la longueur de séquence. C'est aussi ce que recommande le retour d'expérience du pré-entraînement d'un 1,1 Md sur un GPU 6 Go.

---

## 6. Axe E — Le calcul à l'inférence comme multiplicateur

### 6.1 Best-of-N avec tests unitaires

Pour le code, un test unitaire est un **vérificateur exact**. Si un modèle a une probabilité `p` de produire une solution correcte par échantillon, la probabilité qu'au moins un des `N` échantillons soit correct est `1 − (1 − p)^N`. Passer de `p = 0,15` à `N = 16` donne ~93 % de chances d'avoir une solution correcte *parmi les candidats* ; il reste à la sélectionner, ce que les tests font gratuitement. C'est le levier le plus rentable pour le code : **aucun paramètre supplémentaire, seulement du calcul d'inférence et un exécuteur**.

Les benchmarks publiés mesurent traditionnellement pass@1 et pass@k séparément ; notre protocole doit mesurer **pass@1 avec sélection par tests** (parfois appelé pass@1-oracle) et le coût en tokens/temps.

### 6.2 Réparation itérative

Les modèles de code échouent souvent de peu : une variable mal nommée, une condition inversée, un off-by-one. Réparer est une tâche plus locale que générer. Les travaux sur le feedback d'exécution multi-tours (Murphy, [arXiv:2511.07833](https://arxiv.org/abs/2511.07833)) montrent +4 à +6 points de pass@1 sur des modèles de 1,7–4 Md avec un protocole de réparation. Un **réparateur minuscule** (20–60 M) spécialisé dans le patch, conditionné par le traceback et les tests échoués, est une piste originale et peu coûteuse.

### 6.3 RLVR sur tests unitaires

Le RL avec récompense vérifiable (GRPO/GSPO) sur des tests unitaires fonctionne sur les petits modèles. L'étude 2026 sur Qwen3-0.6B ([arXiv:2605.30478](https://arxiv.org/abs/2605.30478)) rapporte **+13 à +14 points de pass@1 sur MBPP** avec LoRA, une récompense combinée tests + Ruff, et GRPO ou GSPO. C'est un des meilleurs retours sur investissement du projet : quelques heures de T4 pour un gain à deux chiffres.

### 6.4 Systèmes hétérogènes

NVIDIA ([arXiv:2506.02153](https://arxiv.org/abs/2506.02153)) estime que 40 % (Open Operator), 60 % (MetaGPT) et 70 % (Cradle) des appels d'un agent peuvent être servis par un SLM. Un système BlackCode réaliste est donc : un petit modèle local pour la majorité des étapes, un modèle plus gros (API ou local) pour la planification difficile, et des outils pour le reste. La « généralité » vient de l'orchestration, pas uniquement des poids.

---

## 7. Tableau de synthèse

| Levier | Gain mémoire | Gain qualité/coût | Maturité | Notre usage |
|---|---|---|---|---|
| Ternaire natif 2 bits | ×8 vs fp16 | parité à ≥3 Md, incertain à 1,2 Md | élevée | cœur |
| Hybride conv/attention 3:1 | KV ÷4 | parité ou mieux | élevée | cœur |
| Embeddings liés | −5 à −12 % | −0,2 à −0,6 % | élevée | cœur |
| Profond et étroit | 0 | +1 à +3 % | élevée | cœur |
| Partage de blocs | 0 (mémoire) | +0,7 % | moyenne | ablation |
| Memory layers | +10–25 Mo | +faits, +code | moyenne | phase 3 |
| Distillation + QAT | — | compense le manque de tokens | élevée | cœur |
| RLVR tests | 0 | +10–15 pts pass@1 | moyenne | phase 3 |
| Best-of-N + tests | 0 | ×2–5 en efficacité | élevée | phase 3 |
| Réparateur minuscule | +30 Mo | +5–10 pts | faible (à prouver) | recherche |
| Agent + outils | 0 | généralité | élevée | phase 4 |

---

## Références

1. Ma et al., *BitNet b1.58 2B4T Technical Report*, arXiv:2504.12285, 2025.
2. Wang et al., *BitNet: Scaling 1-bit Transformers for Large Language Models*, arXiv:2402.17764, 2023–2024.
3. Kumar et al., *Scaling Laws for Precision*, arXiv:2411.04330, ICLR 2025.
4. Dettmers & Zettlemoyer, *The case for 4-bit precision*, arXiv:2212.09720, ICML 2023.
5. Liu et al., *ParetoQ: Improving Scaling Laws in Extremely Low-bit LLM Quantization*, arXiv:2502.02631, 2025.
6. *CAT-Q: Cost-efficient and Accurate Ternary Quantization for LLMs*, arXiv:2606.26650, 2026.
7. *Ternary Mamba: Grouped Quantization-Aware Training of W1.58A16 State Space Models*, arXiv:2606.18114, 2026.
8. Liu et al., *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases*, arXiv:2402.14905, ICML 2024.
9. Liquid AI, *LFM2 Technical Report*, arXiv:2511.23404, 2025.
10. Moonshot AI, *Kimi Linear: An Expressive, Efficient Attention Architecture*, arXiv:2510.26692, 2025.
11. Meta FAIR, *Memory Layers at Scale*, arXiv:2412.09764, 2024.
12. Jolicoeur-Martineau, *Less is More: Recursive Reasoning with Tiny Networks*, arXiv:2510.04871, 2025.
13. Belcak et al., *Small Language Models are the Future of Agentic AI*, arXiv:2506.02153, 2025.
14. *Improving Small Language Models for Code Generation with RL from Verification Feedback*, arXiv:2605.30478, 2026.
15. *Murphy: Feedback-Aware GRPO with Retrospective Credit Assignment for Multi-Turn Code Generation*, arXiv:2511.07833, 2025.
16. Hui et al., *Qwen2.5-Coder Technical Report*, arXiv:2409.12186, 2024.
17. Hoffmann et al., *Training Compute-Optimal Large Language Models*, arXiv:2203.15556, 2022.
18. Agarwal et al., *On-Policy Distillation of Language Models (GKD)*, arXiv:2306.13649, 2023.
19. Microsoft, `bitnet.cpp` — https://github.com/microsoft/BitNet
20. Microsoft, `T-MAC` — https://github.com/microsoft/T-MAC
