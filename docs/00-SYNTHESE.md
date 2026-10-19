# BlackCode — synthèse historique de la recherche

> **Cible active révisée :** modèle français uniquement, entraîné de zéro, 200–300 M paramètres; config exacte 258,6 M dans [`configs/blackcode_fr_260m.json`](../configs/blackcode_fr_260m.json). Le budget d'inférence est de 4 Go minimum et 128k de contexte; les objectifs mémoire/contexte ne sont pas encore validés. Le plan actif est [`docs/16-MODELE-FR-260M.md`](16-MODELE-FR-260M.md). Les analyses 1,2 Md/1 Go ci-dessous sont historiques.

**Objectif historique** : un LLM généraliste, fort en code, qui tourne dans **≤ 1 Go de RAM**, entraînable sur un **T4 Colab**, sans réduire sa taille.

**Thèse centrale** : on ne cherche pas à faire un petit modèle. On cherche à faire un **gros modèle qui coûte peu**. Un modèle de 1,19 Md de paramètres stocké en ternaire (2 bits) tient dans ~350 Mo de poids ; le même modèle en fp16 en demanderait 2,4 Go. À budget mémoire constant, on peut donc mettre **4 à 8 fois plus de paramètres**. C'est le levier principal.

---

## Les 10 constats qui structurent le projet

1. **La mémoire n'est pas proportionnelle au nombre de paramètres.** Un modèle ternaire natif de 2,4 Md (BitNet b1.58 2B4T) n'occupe que 0,4 Go hors embeddings et ~1,19 Go au total en GGUF, pour 29 ms/token sur CPU. Un modèle fp16 de même taille en demanderait ~4,8 Go. ([BitNet 2B4T](https://arxiv.org/abs/2504.12285))

2. **Mais le poids du fichier n'est pas la RAM du processus.** Les mesures communautaires de BitNet 2B4T donnent ~1,3 Go de RSS total, pas 0,4 Go : le runtime, le cache KV et les activations comptent. Pour tenir **sous 1 Go mesuré**, il faut viser ~1,1–1,2 Md de paramètres, pas 2,4 Md. C'est le choix de `blackcode_1b`.

3. **Sous contrainte de mémoire stricte, 2 bits peut battre 4 bits.** Les lois d'échelle de précision montrent qu'à *calcul* fixe, 7–8 bits sont optimales ; mais à *mémoire* fixe, descendre en précision et augmenter le nombre de paramètres peut gagner. Notre planificateur (`blackcode.planner`) prédit un optimum autour de **2 bits / 2,6 Md de paramètres** dans 1 Go, contre 336 M en 8 bits. Cette prédiction est une extrapolation à valider — c'est précisément l'objet de la recherche. ([Scaling Laws for Precision](https://arxiv.org/abs/2411.04330), [ParetoQ](https://arxiv.org/abs/2502.02631))

4. **Le ternaire natif n'est pas de la quantification post-entraînement.** BitNet est entraîné dès le départ avec des poids {-1, 0, +1} et des activations 8 bits. La quantification post-entraînement en dessous de 4 bits s'effondre généralement. Conséquence : il faut soit pré-entraîner ternaire (cher), soit **distiller un modèle fp16 vers un étudiant ternaire** (notre choix, cf. TAD).

5. **La profondeur et les embeddings liés sont gratuits ou presque.** MobileLLM montre qu'à taille égale, un modèle **profond et étroit** bat un modèle large et plat ; le partage entrée/sortie économise ~12 % des paramètres pour -0,2 à -0,6 % d'accuracy ; répéter les blocs (weight sharing) gagne encore +0,7 à +0,8 %. ([MobileLLM](https://arxiv.org/abs/2402.14905))

6. **75 % de l'attention peut disparaître sans perte.** Les architectures hybrides 3:1 (convolution courte ou attention linéaire, puis attention pleine) sont devenues l'état de l'art on-device : LFM2 (10 conv + 6 GQA), Qwen3-Next (Gated DeltaNet), Kimi Linear (KDA) réduit le cache KV de 75 % et accélère le décodage jusqu'à 6× à 1M de contexte, en *battant* l'attention pleine. ([LFM2](https://arxiv.org/abs/2511.23404), [Kimi Linear](https://arxiv.org/abs/2510.26692))

7. **La connaissance peut être stockée hors des FLOPs.** Les *memory layers* à clés-produit ajoutent jusqu'à 128 Md de paramètres sans augmenter les FLOPs, et un modèle de 1,3 Md approche Llama2-7B en QA factuelle ; les gains touchent aussi HumanEval et MBPP. Sous 1 Go, une mémoire de 20–50 M d'entrées en 2 bits coûte 10–25 Mo. ([Memory Layers at Scale](https://arxiv.org/abs/2412.09764))

8. **7 M de paramètres peuvent battre des LLM sur un raisonnement structuré.** Le Tiny Recursive Model (2 couches, récursion, supervision profonde) atteint 45 % sur ARC-AGI-1 et 87 % sur Sudoku-Extreme. L'analyse indépendante montre que c'est surtout la **supervision profonde** qui paie (19 % → 39 %), pas la récursion seule. C'est un levier direct pour un « répareur de code » minuscule. ([TRM](https://arxiv.org/abs/2510.04871))

9. **Pour le code, le calcul à l'inférence est presque gratuit.** Un compilateur et des tests unitaires sont un vérificateur parfait. Générer 16 candidats et garder celui qui passe les tests peut faire passer un modèle de 0,6 Md devant un modèle 10× plus gros en pass@1. Le RLVR sur tests unitaires a déjà montré +13 à +14 points de pass@1 sur Qwen3-0.6B avec LoRA. ([RLVR petit code](https://arxiv.org/abs/2605.30478))

10. **Un petit modèle ne doit pas être seul.** NVIDIA soutient que 60–80 % des appels d'un agent peuvent être servis par des SLM, dans un système hétérogène. BlackCode doit être un **agent** (routeur + ouvrier + outils), pas un chatbot isolé. ([SLM agents](https://arxiv.org/abs/2506.02153))

---

## Ce que nous construisons

| Décision | Pourquoi | Gain attendu |
|---|---|---|
| **Poids ternaires 2 bits** | 8× moins de mémoire que fp16, kernels CPU additifs | 1,12 Md de poids non-embeddings ≈ 280 Mo |
| **Hybride 3 conv / 1 attention** | cache KV constant sur 75 % des couches, rapide sur CPU | KV 8k : 96 Mo au lieu de ~400 Mo |
| **Embeddings liés + 32k vocab** | économie de 67 M de paramètres, tokens de code efficaces | −64 Mo, moins de tokens/programme |
| **Profond et étroit** (24 couches, d=2048) | MobileLLM : la profondeur compte plus que la largeur | +1–3 % à taille égale |
| **Partage de blocs optionnel** (`loop`) | plus de profondeur effective sans mémoire | ablation, +0 à +1 % |
| **Distillation + QAT (TAD)** | on ne peut pas pré-entraîner 1 Md sur T4 | qualité proche du fp16 avec 10–50× moins de tokens |
| **RLVR sur tests unitaires** | le code a un vérificateur gratuit | +10–15 pts pass@1 sur petit modèle |
| **Réseau de réparation minuscule (REN)** | réparer est plus facile que générer | +5–10 pts pour <50 Mo |
| **Harnais agentique** | compiler, tester, chercher, corriger | généralité sans paramètres |

**Budget cible mesuré** (`python -m blackcode.planner --config configs/blackcode_1b.json --context 8192`) :

```
paramètres        1187,24 M  (1120,13 M hors embeddings, 24 couches dont 6 attention)
poids ternaires    269,1 Mo
embeddings int8     64,1 Mo
cache KV 8k         96,0 Mo
état convolutions    0,2 Mo
runtime estimé     180,0 Mo
--------------------------------
total              609,5 Mo   → marge de 414,5 Mo sous 1 Go
```

Avec un runtime C de 80 Mo et **4 096 de contexte**, le total tombe à **461 Mo** (269 + 64 + 48 + 0,2 + 80) : on est sous 0,5 Go. C'est le scénario « 1 Go voire moins ».

**C'est mesuré, plus estimé.** Le moteur C de ce dépôt (`runtime/bcrun.c`) charge le paquet en mmap et donne, pour le modèle de 1,19 Md :

| Contexte | RSS pic |
|---:|---:|
| 4 096 | **385 Mo** |
| 8 192 | **431 Mo** |
| 16 384 | **536 Mo** |

Le runtime seul pèse 1,6 Mo. Avec PyTorch, à l'inverse, ~500 Mo sont consommés avant même les poids. La promesse « 1 Go » tient donc **avec le moteur C**, pas avec PyTorch. Détail : `docs/08-RAM-MESUREE.md`.

---

## Plan d'exécution (résumé)

- **Phase 0 — pipeline** : corpus → tokenizer → entraînement → génération → export `.bcpkg` → budget mémoire. *Fait dans ce dépôt, testé sur CPU.*
- **Phase 1 — véhicule de recherche** : `t4_small` (51 M) entraîné de zéro sur 1–4 Md tokens de code. Sert aux ablations (ternaire vs fp16, conv vs attention, profondeur, loop), pas à la production.
- **Phase 2 — BlackCode-1 (production)** : distillation d'un professeur code (Qwen2.5-Coder-7B/14B en 4 bits) vers un étudiant ternaire de 0,6–1,2 Md, puis QAT de récupération. C'est le seul chemin réaliste sur T4.
- **Phase 3 — capacités** : RLVR sur tests unitaires, REN, décodage par tests, mémoire à clés-produit.
- **Phase 4 — agent** : harnais outils + évaluation honnête (HumanEval+, MBPP+, LiveCodeBench, RAM, latence).

Le détail est dans `docs/05-FEUILLE-DE-ROUTE.md`. Le protocole T4, avec commandes et temps estimés, est dans `docs/04-PROTOCOLE-T4.md`. Les leviers de réduction du temps d'entraînement (distillation, élagage, Muon, WSD, débit) sont chiffrés dans `docs/07-TEMPS-D-ENTRAINEMENT.md`.

---

## Ce que nous ne promettons pas

- **Pas de « Claude dans 1 Go ».** Un modèle de 1,2 Md ternaire, même bien distillé, ne raisonnera pas comme un modèle de 100 Md. Il peut être excellent sur des tâches de code courtes et répétitives, et être *augmenté* par les tests, la récupération et les outils.
- **Pas de pré-entraînement 1 Md sur T4.** Le calcul manque d'un facteur ~100. La voie réaliste est distillation + ternarisation + RLVR, pas from scratch.
- **Pas de supériorité gratuite du ternaire.** À 1,2 Md, le ternaire natif n'a pas encore prouvé sa parité avec le fp16 (BitNet ne la montre qu'à partir de ~3 Md). La distillation et le QAT doivent compenser ; c'est l'hypothèse centrale à tester.
- **Pas de gains de vitesse CPU sans kernels dédiés.** Un modèle ternaire chargé dans PyTorch n'est pas plus rapide ; il faut `bitnet.cpp`/T-MAC ou nos propres kernels. Le gain mémoire, lui, est immédiat.
