# Stockage gratuit et sauvegardes multi-sessions

## Décision recommandée

- **GitHub** : code et petits fichiers de configuration seulement. Ne pas pousser le corpus, les poids, les checkpoints ni utiliser Git LFS pour ça.
- **Google Drive gratuit** : copie persistante des données préparées et des checkpoints. Le quota gratuit standard est partagé avec les autres services Google du compte : vérifier l'espace réellement disponible avant de commencer.
- **Disque local Colab (`/content`)** : copie de travail rapide pendant l'entraînement. Il est éphémère : ne jamais y laisser l'unique copie.

Organisation proposée :

```text
Google Drive / MyDrive / BlackCode/
├── data/v2/                    # corpus préparé, à garder
└── runs/distill_06b/
    ├── ckpt.pt                 # dernier checkpoint complet, reprise
    └── best.pt                 # meilleur modèle validé
```

On commence avec `blackcode_0.6b`, pas 1.2B : le 0.6B a de meilleures chances de laisser de la place au corpus et aux deux checkpoints dans le quota gratuit. Après le premier checkpoint, mesurer la taille réelle :

```bash
du -sh /content/drive/MyDrive/BlackCode/*
du -h /content/drive/MyDrive/BlackCode/runs/distill_06b/*
```

Ne pas mettre le cache complet du professeur StarCoder2 sur Drive par défaut : il peut occuper plusieurs Go et se retélécharge depuis Hugging Face à une nouvelle session. Garder ce qui coûte le plus à reconstruire : le corpus tokenisé et le checkpoint élève.

## À faire dans chaque nouveau notebook Colab

### 1. Monter le même Drive

Exécuter dans une cellule Colab (une autorisation Google sera demandée) :

```python
from google.colab import drive
drive.mount('/content/drive')
```

Puis :

```bash
bash scripts/colab_setup.sh
mkdir -p /content/drive/MyDrive/BlackCode/data
mkdir -p /content/drive/MyDrive/BlackCode/runs/distill_06b
```

Pour changer de compte Colab sans perdre l'accès : depuis le compte propriétaire, partager le dossier `BlackCode` avec le second compte en **Éditeur**. Dans le Drive du second compte, ajouter un raccourci du dossier partagé à `Mon Drive`, puis monter ce Drive dans Colab. Le dossier reste la copie persistante commune ; le runtime GPU, lui, peut changer.

### 2. Réutiliser le corpus (ne pas le rebâtir à chaque session)

Après la première construction :

```bash
cp -a /content/Llm-search/data/v2 /content/drive/MyDrive/BlackCode/data/
```

Aux sessions suivantes, recopier du Drive vers le disque local rapide :

```bash
cp -a /content/drive/MyDrive/BlackCode/data/v2 /content/Llm-search/data/
```

Vérifier la copie avant d'entraîner :

```bash
cat /content/Llm-search/data/v2/meta.json
```

### 3. Première session et sauvegarde toutes les 100 étapes

Le code écrit d'abord sur le disque local puis copie **en arrière-plan** `ckpt.pt` et `best.pt` vers Drive. Les fichiers Drive sont remplacés atomiquement après copie ; un upload incomplet ne devient pas le checkpoint de reprise. Les sauvegardes ne sont pas mises dans GitHub.

```bash
mkdir -p /content/blackcode/runs/distill_06b
python -m blackcode.distill --teacher bigcode/starcoder2-3b --load-4bit \
  --config configs/blackcode_0.6b.json --data data/v2 \
  --out /content/blackcode/runs/distill_06b \
  --mirror-dir /content/drive/MyDrive/BlackCode/runs/distill_06b \
  --seq-len 2048 --batch-size 2 --grad-accum 8 \
  --max-steps 916 --topk 32 --alpha 0.5 --temperature 2.0 \
  --optim adafactor --lr 3e-4 --schedule wsd \
  --checkpoint-policy selective --save-interval 100 --eval-interval 200 --resume
```

À la vitesse prévisionnelle de 4,1 h, 100 étapes représentent environ 27 minutes. La vitesse réelle dépend de la T4 et du temps d'upload. Lire les lignes `[backup]` et vérifier régulièrement que la date/taille de `ckpt.pt` change dans le dossier Drive. Garder Colab ouvert jusqu'à ce qu'une ligne `[backup]` confirme la dernière sauvegarde souhaitée.

### 4. Reprendre après interruption ou depuis un autre compte Colab

Après avoir monté le Drive, synchronisé le dépôt et recopié le corpus :

```bash
mkdir -p /content/blackcode/runs/distill_06b
cp -a /content/drive/MyDrive/BlackCode/runs/distill_06b/ckpt.pt \
  /content/blackcode/runs/distill_06b/ckpt.pt
[ ! -f /content/drive/MyDrive/BlackCode/runs/distill_06b/best.pt ] || \
  cp -a /content/drive/MyDrive/BlackCode/runs/distill_06b/best.pt \
    /content/blackcode/runs/distill_06b/best.pt
```

Relancer exactement la même commande d'entraînement avec `--resume`. Elle reprend au pas enregistré dans `ckpt.pt`; `--mirror-dir` continue les sauvegardes vers le Drive commun.

## Vérification du vrai coût de stockage

À la première sauvegarde, exécuter :

```bash
du -h /content/blackcode/runs/distill_06b/ckpt.pt
du -sh /content/drive/MyDrive/BlackCode/data/v2
```

Si le quota Drive est insuffisant : ne supprime pas l'unique checkpoint. Commencer par enlever les sources brutes déjà récupérables et les caches inutiles ; garder le corpus tokenisé, le dernier checkpoint et, si possible, `best.pt`. Ne compter sur aucun stockage éphémère de Colab pour la reprise.
