# Prototype agentique de comparaison (trajectoire précédente)

**Statut : référence facultative uniquement.** La décision active est désormais d'entraîner BlackCode de zéro, uniquement en français, à environ 258,6 M paramètres; voir [`docs/16-MODELE-FR-260M.md`](16-MODELE-FR-260M.md). Qwen n'est pas la base de BlackCode et aucune distillation depuis Qwen n'est prévue par défaut. Ce prototype peut seulement servir de comparaison séparée. Il reste indépendant des conversations Discord, qui ne sont ni nécessaires ni incluses.

## Décision de base

Le prototype réutilise **`Qwen/Qwen3-4B-Instruct-2507`**, un modèle instruct préentraîné de 4 milliards de paramètres. Sa fiche officielle annonce le français parmi ses capacités multilingues, une fenêtre native de 262 144 tokens et des capacités d'appel d'outils. C'est une base open-weight à tester, pas un modèle entraîné par BlackCode ni une garantie de qualité. Licence et conditions de redistribution doivent être revérifiées sur la fiche du modèle avant de redistribuer des poids.

Le choix sépare deux objectifs :

1. **Assistant fonctionnel maintenant :** base préentraînée, runtime et boucle d'outils de ce document.
2. **BlackCode maison :** modèle de recherche distinct, à entraîner puis comparer et éventuellement distiller. L'initialisation aléatoire du modèle maison ne produit pas un assistant compétent.

Le prototype débute à **32 768 tokens** sur deux T4 pour limiter le risque mémoire. Cela ne valide pas 128k. La fiche du modèle annonce 262k natifs; le débit, la mémoire réelle et la qualité à 64k/128k doivent encore être mesurés sur le matériel cible. La cible RAM du projet reste **4 Go minimum**; le serveur Qwen 4B est conçu pour être testé sur GPU, et ne promet pas de fonctionner sur une machine 4 Go.

## Fonctionnalités présentes

- Boucle compatible avec les réponses d'outils OpenAI, jusqu'à 5 étapes et 8 appels par question.
- Calculatrice arithmétique isolée utilisant un AST limité, sans `eval`.
- Date/heure par fuseau IANA.
- Notes privées conservées dans `~/.blackcode/notes/notes.json` (dossier modifiable).
- Consultation en lecture seule du projet : liste des fichiers et lecture UTF-8 jusqu'à 80 Ko.
- Outils inconnus refusés. Aucun shell, accès Web, suppression ou écriture dans le dépôt n'est exposé.
- Un outil facultatif exécute `python -m unittest discover -s tests -v` avec un délai maximal de 120 s, et retourne le code de sortie/la sortie pour vérifier les tests. Il est désactivé par défaut : `--enable-tests` n'est à utiliser que dans un projet de confiance, car des tests peuvent exécuter du code; ce n'est pas un bac à sable.
- Jeu indépendant `benchmarks/agent_fr.jsonl` : questions générales, calcul, notes, inspection du projet, séquences multi-étapes et vérification d'une tentative de traversée de chemin.

Le jeu de 13 tâches est un **point de départ d'évaluation**, pas un corpus d'entraînement ni une preuve de qualité générale. Le score automatique mesure surtout le choix et l'ordre des outils; il faut aussi vérifier les réponses et enrichir les tests avant de conclure.

## Lancer sur un runtime Linux avec deux T4

À exécuter sur une instance GPU disposant de deux T4, par exemple le runtime Kaggle déjà testé. Vérifier d'abord `nvidia-smi`, la version CUDA et la version de PyTorch de l'image. Installer une version de vLLM compatible avec cette image selon ses instructions d'installation officielles : ne pas laisser une installation remplacer silencieusement le PyTorch/CUDA préinstallé. Le dépôt n'installe volontairement pas vLLM dans l'environnement de recherche CPU.

Depuis la racine du dépôt, démarrer le serveur en arrière-plan dans une cellule de notebook :

```python
import subprocess
log = open('/tmp/blackcode-vllm.log', 'w')
server = subprocess.Popen(
    ['bash', 'scripts/serve_qwen3_t4.sh'],
    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
)
print('PID vLLM :', server.pid, '| journal : /tmp/blackcode-vllm.log')
```

Le script démarre Qwen 4B en FP16, tensor-parallel sur 2 GPU, longueur maximale 32k, et l'analyseur Hermes pour les appels d'outils. Le téléchargement initial des poids prend plusieurs Go et dépend de l'accès réseau au dépôt Hugging Face.

Dans une autre cellule, vérifier que le serveur répond, puis démarrer le chat :

```bash
python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).status)"
python scripts/chat.py --project-root "$PWD"
# Alternative (à la place), uniquement pour un dépôt de confiance :
# python scripts/chat.py --project-root "$PWD" --enable-tests
```

Quelques questions à essayer :

```text
Calcule (26 + 14) * 3.
Quelle est la date et l'heure actuelle à Paris ?
Enregistre une note intitulée « Prochaine étape » avec le contenu « Évaluer les outils en français. »
Lis configs/blackcode_1.2b_128k.json et résume-le.
/reset
```

Les notes sont persistantes. Le contexte de conversation reste en mémoire jusqu'à `/reset` ou fermeture du processus.

## Évaluation et tests locaux

Les tests de runtime ne nécessitent ni modèle, ni GPU, ni réseau :

```bash
python -m unittest discover -s tests -p 'test_agent.py' -v
python scripts/eval_agent.py --limit 4
# Tout le jeu, y compris l'exécution de tests (dépôt de confiance seulement) :
python scripts/eval_agent.py --enable-tests
```

Pour évaluer le modèle sur toutes les tâches, le serveur vLLM doit être actif. Les notes d'évaluation sont placées dans un répertoire temporaire et supprimées à la fin. La métrique vérifie le choix/l'ordre des outils et quelques fragments simples, pas le jugement humain de la qualité. Une tâche échouée n'est pas automatiquement une erreur du modèle : l'API, le parser d'outils ou la mémoire GPU peuvent aussi être en cause.

Après un premier test stable, augmenter prudemment le contexte pour établir les mesures sur la T4 :

```bash
MAX_MODEL_LEN=65536 bash scripts/serve_qwen3_t4.sh
# Puis tester 131072 uniquement après avoir vérifié les logs et la mémoire GPU.
```

Le changement de longueur nécessite de redémarrer le serveur. Mesurer la mémoire et la latence au lieu de supposer que la longueur native de la fiche modèle tient dans la VRAM. Sur deux T4, 128k est une expérience à valider, pas une configuration garantie.

## Étapes suivantes

1. Exécuter les tests hors ligne, puis l'évaluation sur GPU et archiver le résultat, la version du serveur et la longueur de contexte.
2. Faire relire et compléter les scénarios agentiques en français; ajouter des réponses de référence et des vérifications ciblées.
3. Évaluer le modèle de base avant de le personnaliser; établir une partition d'évaluation qui ne sera pas utilisée pour l'entraînement.
4. Sélectionner des données publiques dont la licence et la provenance sont vérifiables. Les échanges Discord restent hors périmètre sans clarifications/permissions adéquates.
5. Comparer une spécialisation légère (LoRA) et/ou distillation avec BlackCode, sans prétendre qu'un jeu de 13 tâches suffit à entraîner un assistant général.

## Références

- [Fiche officielle Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [Documentation officielle Qwen — appels de fonctions](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
- [Documentation vLLM — tool calling](https://docs.vllm.ai/en/latest/features/tool_calling/)
