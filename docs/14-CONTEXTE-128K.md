# Cible de contexte BlackCode : 128k (expérimentale)

> La cible de taille a été révisée : modèle dense français de **258,6 M paramètres**, entraîné de zéro. Voir [`configs/blackcode_fr_260m_128k.json`](../configs/blackcode_fr_260m_128k.json). Viser **131 072 tokens** en inférence et un minimum de **4 Go de RAM** reste un objectif de recherche, pas une capacité validée de bout en bout. Les calculs mémoire ci-dessous concernent l'ancienne architecture 1,22 Md et ne s'appliquent pas au modèle français.

## Pourquoi 128k

Les tâches de compréhension et de réparation à l'échelle dépôt sont évaluées
à plusieurs longueurs, notamment 32k, 64k et 128k dans LongCodeBench. L'étude
rapporte que de nombreux modèles testés culminent entre 64k et 128k, sans que
128k soit un optimum universel. Une fenêtre de 128k est donc un objectif
raisonnable pour traiter un dépôt moyen avec ses fichiers de contexte, ses
instructions et la sortie attendue. Les très grands monorepos nécessiteront
toujours une recherche/sélection de fichiers par l'agent : 128k ne garantit pas
que tout dépôt GitHub rentre dans la fenêtre.

- LongCodeBench, COLM 2025 : https://arxiv.org/abs/2505.07897
- YaRN, ICLR 2024 : https://arxiv.org/abs/2309.00071
- FlashAttention, NeurIPS 2022 : https://arxiv.org/abs/2205.14135
- Cache KV quantifié, documentation Transformers :
  https://huggingface.co/docs/transformers/kv_cache

## Budget RAM calculé pour la configuration 1,22B

Configuration actuelle : 24 couches, dont 6 couches d'attention GQA ; 4 têtes
KV par couche, dimension de tête 128, cache FP16. Le cache par token vaut :

```
2 (K et V) × 6 couches × 4 têtes KV × 128 dimensions × 2 octets
= 12 288 octets = 12 KiB / token
```

Le planificateur du dépôt a été exécuté avec le contexte 131 072 et un runtime
C de 80 Mo :

| Composant | Budget estimé |
|---|---:|
| Poids ternaires | 269,1 Mo |
| Embeddings int8 | 96,2 Mo |
| Cache KV FP16 à 128k | 1 536 Mo (1,5 GiB) |
| État convolutionnel | 0,2 Mo |
| Runtime budgété | 80 Mo |
| **Total estimé** | **1 981,5 Mo (~1,89 GiB)** |

Le calcul est batch 1. Il laisse une marge nominale sous 4 Go pour l'OS et les
outils, mais **le RSS à 128k n'a pas encore été mesuré**. La valeur n'est pas
une promesse de RAM physique minimale : il faudra mesurer le moteur d'inférence
C, tokenizer inclus si utilisé, et le pic de préremplissage.

Une KV int8 ramènerait le cache estimé à 768 Mo et le total du planificateur à
1 213,5 Mo. Le runtime C actuel n'implémente pas cette quantification : c'est
une piste d'optimisation, pas un résultat acquis.

## État réel et limites à résoudre

1. `configs/blackcode_1.2b.json` et le corpus v2 standard restent à 8k.
   `configs/blackcode_1.2b_128k.json` est la cible expérimentale. Le runtime C
   accepte désormais une liste d'IDs d'entrée de longueur dynamique ; son ancien
   plafond de 4 096 IDs a été retiré. L'extension reste non entraînée.
2. `blackcode.model` et le runtime C savent maintenant pré-calculer RoPE avec
   YaRN ; le facteur 8 et la longueur originale 16k sont configurés dans
   `configs/blackcode_1.2b_128k.json`. La qualité après extension n'est pas
   validée : augmenter seulement `max_seq_len` ne suffit pas sans entraînement
   sur des séquences longues.
3. StarCoder2-3B, professeur de distillation envisagé, a une fenêtre de 16 384
   tokens et une attention glissante de 4 096 : sa distillation directe ne
   fournit donc pas de supervision logit à 128k. Référence :
   https://huggingface.co/bigcode/starcoder2-3b
4. FlashAttention réduit la mémoire intermédiaire en évitant de matérialiser
   toute la matrice d'attention, mais le calcul de l'attention exacte reste
   quadratique avec la longueur. À 128k, six couches globales restent un risque
   majeur de temps et de VRAM sur 2×T4. Il faut vérifier les kernels réellement
   disponibles sur T4 ; ne pas supposer qu'un backend Flash est actif.
5. Les binaires actuels sont une suite de documents tokenisés. Pour apprendre
   à raisonner sur un dépôt entier, il faut ajouter des exemples de dépôt
   structurés (arbre des fichiers, chemins, séparateurs et tâches ancrées dans
   plusieurs fichiers), pas simplement fabriquer des séquences arbitraires de
   128k tokens.

## Plan de validation / optimisation

### A. Étendre la position et la supervision

- YaRN est maintenant implémenté dans PyTorch et le runtime C, avec fenêtre
  de base 16k et facteur 8 explicites. Faire passer les tests de parité, puis
  valider la qualité par entraînement/évaluation longue.
- Curriculum : distillation logit jusqu'à la limite réelle du professeur ;
  puis continuation du modèle élève sur de vrais exemples longs, en augmentant
  progressivement les longueurs (16k → 32k → 64k → 128k). Comparer aussi un
  professeur réellement long-contexte avant de le retenir.
- Évaluer récupération d'information et tâches de réparation à chaque palier,
  y compris au-delà de 16k, pour détecter la dégradation positionnelle.

YaRN a été conçu pour étendre RoPE par fine-tuning efficace ; il ne justifie
pas un simple changement de `max_seq_len` sans données/entraînement longs.

### B. Rendre le calcul T4 faisable

- Mesurer d'abord un pas à 16k, 32k, 64k et 128k sur chaque T4 (microbatch 1),
  avec le professeur désactivé puis activé. Enregistrer VRAM pic, RSS hôte,
  débit, pertes et kernel d'attention effectivement choisi.
- Activer checkpointing sélectif/complet et accumulation de gradient ; garder
  la taille du modèle fixe. Réduire temporairement la longueur pendant le
  warmup, puis augmenter-la : le code d'entraînement expose déjà
  `--seq-len-warmup-frac`.
- Comparer l'attention dense aux alternatives exactes/mémoire-efficaces et aux
  fenêtres locales/blocksparsity. Toute approximation doit être évaluée sur la
  qualité de réparation de dépôts.
- Définir un seuil de passage avant le run réel : pas d'OOM, débit mesuré et
  amélioration réelle aux tests de contexte long. Les benchmarks aléatoires
  seuls ne suffisent pas.

### C. Mesurer l'inférence à 128k

- Exporter un paquet représentatif de la configuration 1,22B.
- Exécuter le runtime C avec un prompt réellement prérempli à 131 072 tokens,
  batch 1 ; mesurer `ru_maxrss`, durée de préremplissage, tokens/s et RSS avec
  le tokenizer/processus appelant inclus.
- Tester FP16 KV et, après implémentation, KV int8. Le critère produit est
  **fonctionnement complet sur une machine 4 Go**, pas seulement un budget
  arithmétique.

## Commandes du calcul mémoire

```bash
python -m blackcode.planner \
  --config configs/blackcode_1.2b_128k.json \
  --context 131072 --ram 4096 --runtime-mb 80

# Projection avec cache KV int8 (non implémenté dans le runtime actuel)
python -m blackcode.planner \
  --config configs/blackcode_1.2b_128k.json \
  --context 131072 --ram 4096 --runtime-mb 80 --kv-bytes 1

python - <<'PY'
from blackcode.config import ModelConfig
from blackcode.memory import budget
cfg = ModelConfig.from_json("configs/blackcode_1.2b_128k.json")
for kv_bytes in (2, 1):
    b = budget(cfg, 131072, weight_bits=2, embed_bits=8,
               runtime_mb=80, kv_bytes=kv_bytes)
    print(f"KV {kv_bytes * 8}-bit:", b.as_dict())
PY
```
