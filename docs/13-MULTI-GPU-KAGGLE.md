# Exploiter les deux T4 gratuits de Kaggle

Kaggle a attribué à cette session **2 × Tesla T4, 15,36 Go de VRAM chacune**. La mémoire n'est pas un pool de 30 Go. BlackCode n'utilisait jusque-là que `cuda:0`; la distillation accepte maintenant `torchrun` et PyTorch DDP pour répartir le batch sur les deux cartes.

## Étape 1 — smoke test DDP obligatoire

Dans le notebook Kaggle, active l'accélérateur GPU et Internet, puis clone la branche :

```bash
!git clone --depth 1 --branch arena/aa605ee4-llm-search \\
  https://github.com/Snowoo-2z/Llm-search.git /kaggle/working/Llm-search
%cd /kaggle/working/Llm-search
!bash scripts/kaggle_ddp_smoke.sh
```

Le test utilise un professeur et un élève minuscules (pas de téléchargement de StarCoder2) ; il entraîne 3 pas, vérifie que le checkpoint indique `world_size=2`, puis vérifie sa copie de sauvegarde. La dernière ligne attendue est :

```text
SUCCÈS : DDP + sauvegarde vérifiés sur les 2 GPU
```

Si le test échoue, **ne pas démarrer l'entraînement réel** : envoyer les dernières lignes de sortie pour corriger le problème.

## Étape 2 — comparer réellement les performances

L'entraînement sur deux GPU réplique le professeur et l'élève sur chaque carte, calcule des exemples différents, puis synchronise les gradients. Ce n'est pas garanti deux fois plus rapide : le temps de calcul du professeur, les communications entre cartes et l'environnement Kaggle influencent le résultat.

On garde le même batch global que le plan mono-T4 :

- 1 T4 : `--batch-size 2 --grad-accum 8`
- 2 T4 : `--batch-size 1 --grad-accum 8` (batch global identique : 16 séquences par mise à jour)

Faire d'abord un test de **100 pas**, en réutilisant les mêmes données et la même configuration. Comparer le champ `tok_s` dans `distill_log.jsonl` et le temps écoulé. Le calcul du budget compte les deux GPU.

## Lancement 2 × T4 — BlackCode 0.6B

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m blackcode.distill \
  --teacher bigcode/starcoder2-3b --load-4bit \
  --config configs/blackcode_0.6b.json --data data/v2 \
  --out /kaggle/working/BlackCode/runs/distill_06b \
  --seq-len 2048 --batch-size 1 --grad-accum 8 \
  --max-steps 916 --topk 32 --alpha 0.5 --temperature 2.0 \
  --optim adafactor --lr 3e-4 --schedule wsd \
  --checkpoint-policy selective --save-interval 100 --eval-interval 200 --resume
```

À chaque étape de checkpoint, le **rang 0 seulement** écrit `ckpt.pt` et `best.pt`, évitant des écritures concurrentes. Le checkpoint contient le modèle, l'optimiseur et `world_size`; il peut reprendre sur un ou deux GPU. Pour Kaggle, garder les fichiers sous `/kaggle/working` et enregistrer une version du notebook avant de quitter. Une sortie Kaggle peut ensuite être ajoutée comme entrée d'un autre notebook.

### Lire les résultats

```bash
cat /kaggle/working/BlackCode/runs/distill_06b/distill_log.jsonl | tail -10
nvidia-smi
```

Comparer le `tok_s` du test mono-GPU à celui du test DDP, **pas seulement le temps restant affiché**. Ne conserver deux GPU pour le vrai run que si le débit global en tokens/s progresse et si la perte (CE/KL) baisse normalement.
