# Réduire le temps d'entraînement sans réduire la taille

> Question : comment entraîner le même modèle (1,2 Md de paramètres, donc même mémoire d'inférence) beaucoup plus vite ?
> Réponse courte : le temps vaut **tokens nécessaires × temps par token ÷ parallélisme**. La littérature et les optimisations logicielles attaquent les trois facteurs ; la plupart des gens n'attaquent que le dernier, qui est le plus faible.

---

## 1. La décomposition qui compte

```
temps mural = (tokens nécessaires pour atteindre la qualité cible)
            × (secondes par token)
            ÷ (nombre de travailleurs parallèles)
```

| Facteur | Leviers | Gain réaliste | Coût |
|---|---|---|---|
| **Tokens nécessaires** | distillation, élagage de données, Muon, curriculum, muP | **×5 à ×50** | quelques jours de calcul |
| **Secondes par token** | compile, 8-bit, checkpoint sélectif, batch, fused, noyaux | **×1,3 à ×2** | quelques heures |
| **Parallélisme** | sessions multiples + DiLoCo | ×k (k ≤ 4 réaliste) | ingénierie |

Conséquence : **l'efficacité des données est dix fois plus rentable que l'optimisation du débit**. Mais c'est le débit qui est le plus simple à mettre en place aujourd'hui, donc on commence par là pendant que les données se préparent.

---

## 2. Les chiffres du projet (calculateur `blackcode.timing`)

### 2.1 Combien de temps pour un milliard de tokens ?

| Modèle | Contexte | FLOPs/token | Temps à 22,8 TFLOPS (T4, 35 % MFU) |
|---|---:|---:|---:|
| `t4_small` (51 M) | 1 024 | 0,33 GFLOP | **4,0 h** |
| `blackcode_1b` (1,19 Md) | 4 096 | 7,73 GFLOP | **3,9 jours** |
| `blackcode_1b` (1,19 Md) | 8 192 | 8,55 GFLOP | **4,3 jours** |

À 10 Md de tokens, le 1,2 Md demanderait **39 jours** de T4. C'est pour cela que le prétraining from scratch est exclu : ce n'est pas une question de réglage, c'est un facteur 10–100 manquant. La distillation n'est pas une astuce de confort, c'est la seule voie.

### 2.2 Détail intéressant : à petite échelle, la tête de sortie coûte cher

| Modèle | Corps | Attention | Tête (32k vocab) |
|---|---:|---:|---:|
| 51 M | 64 % | 6 % | **30 %** |
| 1,19 Md | 87 % | 8 % | 5 % |

Sur le véhicule de recherche `t4_small`, **30 % du calcul part dans la projection vocabulaire**. Deux conséquences pratiques :

- pour les ablations, utiliser un vocabulaire de 16k fait gagner ~15 % de temps sans changer les conclusions ;
- pour le modèle final, la tête ne coûte que 5 %, il n'y a rien à gagner à la réduire.

### 2.3 La mémoire d'entraînement décide de ce qui est possible

Mesure du calculateur pour 1,19 Md, fp32, 1,5 Go d'activations :

| Optimiseur | Poids | Gradients | États | **Total** | T4 16 Go |
|---|---:|---:|---:|---:|:---:|
| AdamW | 4 529 | 4 529 | 9 058 | **19 616 Mo** | **non** |
| AdEMAMix | 4 529 | 4 529 | 13 587 | **24 145 Mo** | **non** |
| Muon | 4 529 | 4 529 | 3 397 | **13 955 Mo** | oui (limite) |
| AdamW 8-bit | 4 529 | 4 529 | 2 264 | **12 822 Mo** | oui |
| Adafactor | 4 529 | 4 529 | 906 | **11 464 Mo** | oui |

Autrement dit : **avec AdamW, entraîner 1,2 Md sur un T4 est impossible**. Changer d'optimiseur n'est pas une optimisation de vitesse, c'est une condition de faisabilité. Adafactor laisse 4,5 Go de marge, Muon 2 Go.

---

## 3. Les leviers, classés par rapport gain / effort

### Niveau 1 — Changer le nombre de tokens (×5 à ×50)

#### 3.1 Distillation plutôt que from scratch — **×5 à ×50**

Le professeur fournit un signal dense (distribution complète) au lieu d'un one-hot. Ternary Mamba retrouve un modèle entraîné sur **105 Md de tokens avec 102 M tokens de distillation** ; c'est un facteur ~1 000 dans leur cas (grâce au checkpoint pré-entraîné), et un facteur 10–50 réaliste quand on distille un professeur code vers un étudiant de 0,6–1,2 Md.

C'est déjà le cœur de la phase 2 du projet (`docs/03-METHODES-NOUVELLES.md`, M1/TAD). Rappel : le coût du professeur doit être compté — précalculer ses top-k une seule fois.

#### 3.2 Élaguer les données — **×1,5 à ×3**

`blackcode.prune` mesure la perte d'un petit modèle sur chaque document et garde la bande « apprenable » : on jette les fichiers déjà appris (perte basse) et les outliers bruités (perte haute). Sorscher et al. (2022) et les travaux DCLM (2024) rapportent 1,5 à 3× moins de tokens à qualité égale.

```bash
python -m blackcode.prune \
  --ckpt runs/t4_small/best.pt --tokenizer data/code/tokenizer.json \
  --src /content/corpus --out /content/corpus_pruned \
  --lo 0.05 --hi 0.85 --copy-mode symlink

python -m blackcode.corpus prepare --src /content/corpus_pruned --out data/code_pruned
```

Sur le corpus de démonstration (80 fichiers, TheAlgorithms/Python), la bande [0,05 ; 0,85] conserve 80 % des fichiers pour 68 % des octets. Le gain réel doit être mesuré : **entraîner deux modèles identiques, l'un sur le corpus complet, l'autre sur le corpus élagué, à tokens égaux**, et comparer la perte de validation.

#### 3.3 Muon — **×1,3 à ×2**

Orthogonalisation du gradient par Newton–Schulz (`blackcode.optim.Muon`) : ~1,3 à 2× d'efficacité par token à taille fixée, validé jusqu'à 16 Md de paramètres (arXiv:2502.16982). Coût : 10–30 % de calcul en plus par pas.

```bash
python -m blackcode.train ... --optim muon --muon-lr 0.02 --lr 6e-4
```

Muon ne s'applique qu'aux matrices 2D cachées ; les embeddings et les normes restent sous AdamW (géré automatiquement : deux groupes, deux LR).

#### 3.4 WSD : ne jamais perdre un entraînement interrompu — **×1,0 à ×1,3**

Le schedule Warmup–Stable–Decay garde la LR constante pendant 80 % du budget, puis la fait décroître. Deux propriétés décisives sur Colab :

- une session coupée au milieu de la phase stable **n'a rien perdu** : on reprend, ou on prolonge `--max-steps`, la LR remonte naturellement ;
- on peut comparer plusieurs branches depuis le même checkpoint sans refaire toute la décroissance.

```bash
python -m blackcode.train ... --schedule wsd --stable-frac 0.8 --decay-shape sqrt
```

Avec un cosinus classique, prolonger l'entraînement après une coupure force à refaire la courbe de LR ou à accepter une LR trop basse — perte sèche de jours de calcul.

#### 3.5 muP et le transfert d'hyperparamètres — **×1,5 à ×3 sur le temps de réglage**

Régler le LR et l'init sur un modèle de 10–30 M avec les règles muP (Tensor Programs V), puis transférer à 1,2 Md sans refaire de balayage. Sans muP, chaque changement d'échelle coûte 5 à 20 runs complets. Non implémenté ici ; c'est le prochain gain « gratuit » après la phase 1.

### Niveau 2 — Accélérer chaque token (×1,3 à ×2)

| Optimisation | Gain typique | Commande | Remarque |
|---|---:|---|---|
| `torch.compile` | 10–25 % sur GPU | `--compile` | 0 % sur CPU (mesuré), à garder pour le T4 |
| AdamW 8-bit / Adafactor | permet un batch 2–4× plus grand | `--optim adamw8bit` / `adafactor` | nécessite `bitsandbytes` pour le 8-bit |
| Checkpointing **sélectif** au lieu de complet | 10–15 % (au lieu de −30 %) | `--checkpoint-policy selective` | 1 bloc sur 2, schéma √L |
| Gradients accumulés vs gros batch | 0 % de calcul, mais meilleur MFU | `--batch-size` ↑ `--grad-accum` ↓ | le batch n'aide que jusqu'au batch critique |
| `--fused` (AdamW CUDA) | 2–5 % | `--fused` | noyau fusionné |
| Contexte progressif | 0–10 % | `--seq-len-warmup-frac 0.2` | l'attention est peu coûteuse dans notre hybride (8 %) |
| Évals moins fréquentes | 5–10 % du temps mural | `--eval-interval 1000` | une éval de génération coûte des minutes |
| Checkpoints asynchrones | 10–60 s par sauvegarde | `--mirror-dir /content/drive/...` | écriture Drive en arrière-plan |
| Données sur disque local | variable | copier `data/` dans `/content` | Drive est lent en lecture aléatoire |
| `--log-flops` | 0 %, mais rend le reste mesurable | `--log-flops` | affiche les TFLOPS et le MFU réels |

Détail important : **le batch n'accélère pas indéfiniment**. Au-delà du batch critique (souvent 0,5–4 M tokens pour un petit modèle), on ajoute des pas sans gagner en qualité par token. Sur un seul T4, augmenter le batch sert surtout à améliorer le MFU et la stabilité, pas à diviser le temps.

### Niveau 3 — Paralléliser (×2 à ×4)

Une seule T4 ne peut pas être divisée. Mais plusieurs sessions Colab le peuvent, avec **DiLoCo** (Douillard et al., 2023) : chaque worker fait 500 pas en autonomie, puis on moyenne les poids. La communication est ~500× plus faible qu'un data-parallel classique, et la qualité est comparable.

- réaliste : **2 à 4 sessions**, synchronisées toutes les 500–1 000 pas via Drive ou un petit dépôt ;
- coût : ingénierie, gestion des coupures, moyennage des poids et des états d'optimiseur ;
- gain : ~k× sur le temps mural, moins 5–10 % de surcoût de synchronisation ;
- à ne faire qu'après avoir validé la recette sur une session (sinon on multiplie les bugs).

---

## 4. Ce qui ne marche pas sur ce matériel (et pourquoi)

| Idée | Pourquoi ça échoue |
|---|---|
| fp8 / bf16 | le T4 (sm75) n'a pas ces instructions natives ; l'émulation est plus lente |
| FlashAttention 2 | nécessite sm80+ ; PyTorch retombe sur le noyau « memory-efficient » |
| MoE pour réduire les FLOPs | augmente la mémoire totale, incompatible avec 1 Go d'inférence |
| Élaguer le modèle | interdit par l'énoncé : la taille doit rester la même |
| Sparsité non structurée | aucun gain sur des noyaux denses |
| AdamW fp32 classique sur 1,2 Md | 19,6 Go, ne rentre pas (voir §2.3) |
| Augmenter le batch au-delà du critique | pas de gain par token, juste plus de pas en parallèle |
| Multi-token prediction | ajoute du calcul par pas ; gains de tokens incertains à cette échelle, à tester |

---

## 5. Mesures faites dans ce dépôt (CPU, 2 cœurs)

Modèle `nano` (2,16 M), 150 pas, séquence 128, batch 8, corpus de 1,32 M tokens de Python.

| Configuration | Débit | Perte val à 150 pas | Commentaire |
|---|---:|---:|---|
| AdamW + cosinus (référence) | 6,1 k tok/s | 5,855 | — |
| Checkpointing complet | 4,9 k tok/s | 5,855 | −20 % de débit |
| Checkpointing sélectif | 5,2 k tok/s | 5,855 | −15 %, moitié de la mémoire économisée |
| Muon + WSD | 4,9 k tok/s | **5,209** | −20 % de débit, perte plus basse en 150 pas |
| `torch.compile` | 5,7–6,1 k tok/s | 6,422 (80 pas) | aucun gain sur CPU ; à retester sur T4 |
| Adafactor | 6,1 k tok/s | 6,88 (60 pas) | **LR ~30× plus élevée** ; l'implémentation le fait automatiquement |

Lecture honnête :

- la comparaison Muon vs AdamW **n'est pas à LR égalisée** (0,02 vs 1e-3) : elle montre que Muon apprend plus vite par pas, pas qu'il est meilleur à budget égal. Il faut une ablation propre sur `t4_small` sur 0,5 Md de tokens ;
- le checkpointing coûte moins cher sur CPU que sur GPU (le CPU est limité par le calcul, pas par la bande passante) ; sur T4, compter −25 à −35 % pour le mode complet, −10 à −15 % pour le sélectif ;
- `torch.compile` ne sert à rien sur CPU, mais reste le gain le plus simple sur T4 ;
- Adafactor normalise sa mise à jour par `sqrt(v)` : à LR égale, son pas est ~30× plus petit qu'AdamW. `--optim adafactor` multiplie donc la LR par 30 par défaut (surchargeable avec `--adafactor-lr`). Sans cela, le modèle semble ne pas apprendre.

Ces chiffres sont des **tests de bon fonctionnement**, pas des prédictions T4. Le seul chiffre qui compte est celui que vous mesurez avec `--log-flops`.

---

## 6. Plan d'action concret (dans cet ordre)

| # | Action | Temps | Gain attendu | Commande |
|---|---|---|---:|---|
| 1 | Mesurer le MFU réel | 1 h | 0 % (mais rend le reste mesurable) | `--log-flops` |
| 2 | Passer en WSD + checkpoints asynchrones + moins d'évals | 30 min | ×1,05–1,15 mural, robustesse | `--schedule wsd --mirror-dir ... --eval-interval 1000` |
| 3 | Adafactor ou 8-bit + checkpoint sélectif + compile | 2 h | ×1,2–1,4 de débit, et **rend 1,2 Md possible** | `--optim adafactor --checkpoint-policy selective --compile` |
| 4 | Élaguer le corpus avec le modèle de 51 M | 1 passe + 1 ablation | ×1,5–3 de tokens | `blackcode.prune` |
| 5 | Ablation Muon sur `t4_small`, 0,5 Md tokens | 2 sessions | ×1,3–2 si confirmé | `--optim muon` |
| 6 | Distillation depuis un professeur code | phase 2 | ×5–50 | voir `docs/04` |

Cumul réaliste : **×3 à ×6** sans réduire la taille, avant même la distillation. Avec elle, un entraînement de 1,2 Md qui coûtait 39 jours de T4 tombe dans une fenêtre de quelques jours.

---

## 7. Le piège à éviter

Optimiser le débit (compile, batch, noyaux) donne un facteur ~1,5. Changer le nombre de tokens nécessaires donne un facteur ~10. Mais le second demande de la réflexion sur les données et l'objectif, alors que le premier se résume à des flags.

**La bonne méthode** : mesurer le MFU (facteur 3), appliquer les flags (facteur 2), puis passer le plus clair du temps sur les données et la distillation (facteur 1). Et toujours comparer **à tokens égaux**, jamais à pas égaux — sinon Muon, un batch plus grand ou un schedule plus agressif semblent gagner pour de mauvaises raisons.

---

## 8. Note : la vitesse d'inférence n'est pas la vitesse d'entraînement

Les 9,2 tok/s du moteur C (`runtime/bcrun`) concernent **l'inférence** d'un 1,19 Md sur 2 cœurs. L'entraînement se fait sur GPU avec PyTorch et des milliers de tokens par pas en parallèle : `t4_small` tourne à ~70 000 tokens/s théoriques sur T4 (22,8 TFLOPS effectifs), soit un facteur ~7 500. Ne pas confondre les deux quand on lit un chiffre de débit.

Pour l'inférence, la limite est la bande passante mémoire : 267 Mo de poids ternaires lus à chaque token. À 9,2 tok/s, cela fait 2,5 Go/s — bien en dessous des 20–40 Go/s d'un CPU moderne. La marge est donc dans les noyaux SIMD et le nombre de cœurs, pas dans la taille du modèle.

## 9. Références

1. Jordan et al., *Muon: An optimizer for hidden layers in neural networks*, 2024 ; *Muon is Scalable for LLM Training*, arXiv:2502.16982, 2025.
2. Hu et al., *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies*, arXiv:2404.06395, 2024 (WSD).
3. Yang et al., *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer*, arXiv:2203.03466, 2022 (muP).
4. Sorscher et al., *Beyond neural scaling laws: beating power law scaling via data pruning*, NeurIPS 2022.
5. Li et al., *DataComp-LM (DCLM)*, 2024 ; Gunasekar et al., *Textbooks Are All You Need*, 2023 (phi-1).
6. Douillard et al., *DiLoCo: Distributed Low-Communication Training of Language Models*, arXiv:2311.08105, 2023.
7. Pagliardini et al., *AdEMAMix: A Better Adam Despite Fast Saturation of Gradient Second Moment*, arXiv:2409.03137, 2024.
8. *Scaling Laws for Precision*, arXiv:2411.04330, 2024.
9. *Ternary Mamba*, arXiv:2606.18114, 2026 (QAT + distillation, 4 GPU-h).
