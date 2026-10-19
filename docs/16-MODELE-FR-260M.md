# BlackCode-FR : pré-entraînement de zéro, 200–300 M paramètres

## Décision active

- **Taille retenue : 258 578 048 paramètres** (258,6 M), dans la plage demandée.
- **Initialisation aléatoire**, sans poids préentraînés, distillation ou conversations Discord.
- **Réponses et prose d'entraînement uniquement en français.** La syntaxe et les identifiants des extraits de code restent naturellement ceux des langages de programmation.
- **Modèle dense** pendant le pré-entraînement (`ternary: false`). La ternarisation éventuelle est une expérience distincte après entraînement et seulement si elle préserve assez la qualité.
- Objectifs ultérieurs : **128k tokens** et fonctionnement avec **au moins 4 Go de RAM**. Aucun des deux n'est validé à ce stade.

La config de base est [`configs/blackcode_fr_260m.json`](../configs/blackcode_fr_260m.json) : vocabulaire prévu de 24 576 tokens français, largeur 1 024, 20 blocs (15 convolutionnels, 5 attention), embeddings liés. La variante [`configs/blackcode_fr_260m_128k.json`](../configs/blackcode_fr_260m_128k.json) conserve exactement le nombre de paramètres et prépare une expérience YaRN à 131 072 tokens. À titre de calcul théorique, des poids FP16 prennent environ 0,52 Go et le cache KV FP16 de cette architecture à 128k environ 0,67 Go; cela laisse une marge sous 4 Go pour le runtime, mais ce n'est ni une mesure RSS ni une validation de la génération à 128k.

Ordre prévu : commencer l'apprentissage de zéro à 2k/8k, mesurer la qualité, puis tenter l'extension de contexte. Le fichier 128k est une **configuration expérimentale**; il ne signifie pas que le modèle sait déjà traiter 128k correctement.

## Données françaises, sans conversations Discord

Le script [`scripts/build_fr_corpus.sh`](../scripts/build_fr_corpus.sh) utilise maintenant le sous-ensemble **français `fra_Latn` de FineWeb2-HQ**, sélectionné par ses auteurs parmi les 10 % de documents ayant les meilleurs scores d'un filtre de qualité basé sur XLM-RoBERTa. La carte du dataset indique 32,2 M de documents français / 483 Go, et des évaluations sur des modèles 1B; ces chiffres sont ceux des auteurs, pas notre audit. Le script ajoute une vérification française Lingua, le filtrage de qualité BlackCode et la déduplication.

Avant le build complet, échantillonner la source et lire des extraits masqués :

```bash
python -m pip install datasets lingua-language-detector
python scripts/audit_fr_source.py --limit 2000 --samples-out data/audits/fineweb2_hq_fr.jsonl
```

Les extraits restent des données Web potentiellement personnelles : le masquage regex n'est pas une anonymisation garantie; garder ce fichier local, vérifier manuellement un échantillon et ne pas le publier. Le rapport donne taux de français, scores qualité/langue, doublons et motifs de rejet. Après revue :

```bash
MAX_MB=4000 MIN_QUALITY_SCORE=0.0 bash scripts/build_fr_corpus.sh
```

Le seuil `MIN_QUALITY_SCORE` reste volontairement à zéro au départ : choisir un seuil exige d'abord d'examiner la distribution du score et un échantillon, plutôt que de le deviner. Le lot de 4 000 Mo est une première expérience, pas un volume prouvé suffisant pour entraîner le modèle final.

FineWeb2-HQ annonce une licence de base de données ODC-By 1.0, mais indique aussi que l'usage est soumis aux conditions de Common Crawl. La carte précise que des données personnelles identifiables peuvent subsister malgré le traitement et propose un formulaire opt-out. Cela ne rend pas chaque page source libre de droits. Il faut conserver l'attribution/provenance, filtrer les données sensibles et vérifier les conditions des sources avant de redistribuer corpus ou poids. Le pipeline note la source comme `odc-by-1.0`; aucune conversation Discord n'est requise.

L'option `--french-only` rejette les documents de prose que Lingua ne classe pas en français avec suffisamment de confiance. Elle n'altère pas le code source; elle ne prouve donc pas que chaque chaîne/commentaire contenu dans du code est français. Pour la première étape généraliste, utiliser surtout de la prose française filtrée; ajouter ensuite un corpus de code documenté en français après un audit dédié.

## Ce que signifie « parler uniquement français »

Le vocabulaire d'entraînement conversationnel, les instructions, les réponses de l'assistant, les traces agentiques et les évaluations doivent être en français. La configuration ne peut pas garantir à elle seule qu'un modèle génératif ne produira jamais un mot d'une autre langue. On mesurera ce comportement sur un jeu de test français tenu à l'écart de l'entraînement, puis on renforcera le réglage avec des exemples français si nécessaire. La boucle d'agent de comparaison a aussi un système de consigne « français uniquement », mais ce n'est pas une preuve de conformité du futur modèle maison.

Les exemples agentiques créés par `blackcode.data` ont maintenant des consignes et commentaires de trajectoire en français. Le code et les noms d'outils gardent leur syntaxe. Les 13 cas de `benchmarks/agent_fr.jsonl` restent un petit jeu d'évaluation, **pas** un dataset suffisant de fine-tuning.

## Étapes d'entraînement

### 1. Vérifier la configuration

```bash
python - <<'PY'
from blackcode.config import ModelConfig
for path in (
    'configs/blackcode_fr_260m.json',
    'configs/blackcode_fr_260m_128k.json',
):
    cfg = ModelConfig.from_json(path)
    print(path, f"{cfg.num_params:,} paramètres", cfg.max_seq_len, 'tokens')
PY
```

Résultat attendu : `258,578,048 paramètres` pour les deux configs.

### 2. Petit pilote de l'initialisation aléatoire

À lancer seulement après avoir construit `data/fr_260m`. Le dépôt utilise actuellement le trainer simple, qui entraîne sur **un GPU**; les deux T4 ne sont pas automatiquement exploitées par `blackcode.train`. Ne pas interpréter les benchmarks DDP de distillation comme une mesure de ce trainer.

```bash
python -m blackcode.train \
  --config configs/blackcode_fr_260m.json \
  --data data/fr_260m \
  --out runs/fr_260m_pilot \
  --batch-size 1 --grad-accum 16 --seq-len 2048 \
  --max-steps 200 --optim adafactor --lr 3e-4 \
  --checkpoint-policy selective \
  --eval-interval 100 --eval-iters 4 --save-interval 100 \
  --resume \
  --mirror-dir /content/drive/MyDrive/blackcode/runs/fr_260m_pilot
```

Ce pilote consomme environ 6,55 M tokens d'apprentissage (hors évaluation) et sert uniquement à vérifier le chargement, la VRAM, la baisse de perte et la reprise après interruption. Il **ne produira pas** un assistant utile. Archiver le débit réel et la mémoire avant de fixer un budget final.

### 3. Pré-entraînement complet puis réglage agentique

Le budget final en tokens/étapes dépendra de la taille réellement construite du corpus et du débit mesuré au pilote; il n'est pas fixé ici. Un petit modèle dense entraîné de zéro nécessite beaucoup de texte français de qualité, un tokenizer appris sur ces mêmes données et une évaluation régulière. Une perte d'entraînement qui baisse ne prouve pas que le modèle répond bien.

Après le pré-entraînement, créer un **ensemble séparé** de consignes/réponses en français et de trajectoires d'outils françaises; ne pas recycler les évaluations dans le fine-tuning. Mesurer notamment :

- langue de réponse (français exclusif, mesuré par un détecteur + revue humaine) ;
- questions de culture générale et compréhension en français ;
- exactitude, refus quand le contexte manque et respect des consignes ;
- tâches d'outils : choix, arguments, ordre, résultat observé et compte rendu honnête ;
- perplexité française sur un split de validation dédupliqué ;
- mémoire, latence et qualité à 2k/8k, puis 32k/64k/128k séparément.

## Contraintes et risques

1. **Qualité :** 260 M de paramètres est une petite capacité. Même après entraînement, on ne promet pas le niveau d'un modèle préentraîné de plusieurs milliards; il faudra juger sur les tests, pas sur le nom du modèle.
2. **Données :** le lot FineWeb-2 est du Web français, pas un jeu de conversations. Il doit être audité pour répétitions, erreurs de langue, données personnelles et contenu inadapté; créer ensuite des instructions synthétiques françaises de qualité.
3. **Temps :** le coût réel reste inconnu avant le pilote; aucun calendrier de modèle final n'est promis. Sauvegarde et `--resume` sont nécessaires sur les sessions GPU interruptibles.
4. **128k :** la variante YaRN et la mémoire calculée ne valident ni la stabilité d'entraînement longue séquence, ni la qualité de rappel à 128k. Extension et test dédiés restent nécessaires.
5. **Mémoire :** les 4 Go sont un budget d'inférence cible minimum, pas une mesure de RSS. L'entraînement utilise beaucoup plus de mémoire et demande un GPU adapté.
6. **Outils :** les réponses conversationnelles peuvent être en français tandis que les noms d'API/outils et le code gardent des identifiants conventionnels. Les traces d'outils doivent rester entièrement commentées en français.

## Références données

- [FineWeb-2, carte du dataset et licence ODC-By](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
- [FineWeb-2, dépôt de traitement](https://github.com/huggingface/fineweb-2)
