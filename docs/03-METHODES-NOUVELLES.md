# Méthodes proposées (au-delà de « faire plus petit »)

> Objectif : gagner en capacité **sans** ajouter de RAM ni de calcul d'entraînement massif.
> Chaque méthode est formulée comme une hypothèse testable, avec un protocole, un coût et un critère d'arrêt.

---

## M1 — TAD : Ternary-Aware Distillation

**Idée.** On ne pré-entraîne pas un modèle ternaire de 1,2 Md from scratch. On part d'un **étudiant fp16** distillé d'un professeur code, puis on le **ternarise avec récupération par distillation**. Trois étages :

1. **Distillation fp16** : professeur Qwen2.5-Coder-7B/14B (4 bits) → étudiant 0,6–1,2 Md. Perte `KL(top-k)` + `CE`.
2. **Ternarisation progressive** : on remplace les matrices par des matrices ternaires avec STE, mais on **annneale la température de quantification** : au début, une version « douce » (poids quantifiés avec un pas plus fin, proche de 4 bits), puis on resserre vers {-1, 0, +1}. Le taux de quantification augmente linéairement sur les 20 % premiers pas.
3. **Récupération on-policy** : l'étudiant génère, le professeur note (GKD), sur 5–10 M tokens. Les exemples où l'étudiant ternaire diverge le plus du professeur sont **suréchantillonnés** (quantization-error-aware sampling) : le budget de distillation va là où le ternaire fait mal.

**Pourquoi ça peut marcher.** La distillation fournit un signal dense (distribution complète) au lieu d'un signal one-hot ; Ternary Mamba montre qu'un QAT + KD de 102 M tokens retrouve un modèle entraîné sur 105 Md tokens. L'annealing évite l'effondrement des premiers pas, et le suréchantillonnage cible le vrai goulot (les poids qui changent de signe).

**Protocole.**
- Étudiant `t4_base` (127 M) d'abord : mesurer les tokens nécessaires pour atteindre la perte du fp16, avec et sans annealing.
- Puis `blackcode_0.6b` (0,6 Md) et `blackcode_1b`.
- Métriques : perte de validation, pass@1 HumanEval+/MBPP+, bits effectifs, RAM exportée.

**Coût.** T4 : 3–10 jours selon la taille du professeur et le nombre de tokens. Le précalcul des top-k du professeur est la partie la plus longue ; il peut être fait une fois et réutilisé.

**Critère d'arrêt.** Si l'étudiant ternaire reste >0,3 de perte au-dessus du fp16 après 2× le budget de tokens, la voie « ternaire natif à 1,2 Md » est invalidée ; on bascule sur un export 4 bits (1,5 Go → non compatible 1 Go) ou on réduit à 0,8 Md.

---

## M2 — UTB : Unit-Test Beam (décodage par tests)

**Idée.** Pour le code, un test unitaire est un vérificateur exact et gratuit. On génère `N` candidats avec des températures/prompts variés, on exécute, et on sélectionne :

1. le plus court candidat qui passe **tous** les tests visibles ;
2. sinon, celui qui en passe le plus (score partiel) ;
3. si aucun ne passe, on passe la main à REN (M3) avec les traces d'échec.

**Nouveauté opérationnelle.** Introduire une métrique : le **MEM (Memory-Equivalent Multiplier)** — la taille de modèle qu'il faudrait, à `N=1`, pour atteindre le pass@1 obtenu avec `N` échantillons + tests. On mesure ainsi « combien de paramètres on remplace par du calcul d'inférence ». C'est plus informatif que le pass@k brut, qui suppose un oracle parfait.

**Budget adaptatif.** On ne génère pas 16 candidats systématiquement : un petit classifieur de confiance (ou l'entropie des logits) décide de `N ∈ {1, 2, 4, 8, 16}`. Les problèmes faciles coûtent 1 génération, les difficiles 16.

**Protocole.** HumanEval+/MBPP+ avec tests publics pour la sélection, tests cachés pour l'évaluation (évite la triche). Mesurer pass@1, MEM, tokens générés, secondes.

**Coût.** Aucun entraînement ; quelques heures de CPU/GPU pour les exécutions.

**Critère d'arrêt.** Si MEM < 2× à `N=16`, le levier n'est pas rentable sur ce modèle ; le code est trop difficile pour que la sélection par tests aide.

---

## M3 — REN : Repair-Edit Network (le réparateur minuscule)

**Idée.** Générer un programme entier est difficile ; **corriger un programme presque juste** est local. On entraîne un modèle spécialisé de 20–60 M de paramètres qui reçoit :

- le code buggé,
- les tests échoués et le traceback,
- éventuellement le diff des tests,

et produit un **patch** (liste d'éditions `(ligne, ancien, nouveau)` ou `search/replace`), pas une régénération complète.

**Architecture.** Décodeur 12–16 couches, d=512, ternaire, vocabulaire 16k, contexte 4k, avec un mécanisme de copie des identifiants. Entrée structurée :
```
<|code|> ... <|test|> assert f(1)==2 ... <|trace|> AssertionError line 4
<|patch|> - return x + 1
          + return x - 1
```

**Supervision profonde.** On applique le patch, on réexécute, on répète jusqu'à 6 fois. Chaque étape est supervisée (TRM-style), pas seulement la dernière. Le modèle apprend à converger, pas seulement à produire un patch.

**Données.** Mutations sémantiques de fonctions correctes (inverser un opérateur, décaler un indice, permuter deux variables, changer une constante), puis exécution des tests pour ne garder que les mutations qui échouent ; ajout de trajectoires réelles de réparation (commits de correction, traces d'agents).

**Pourquoi c'est nouveau.** La littérature multi-tours utilise le **même** modèle pour générer et réparer. Ici, on sépare les capacités : un générateur de 1,2 Md + un réparateur de 30 M, soit +2,5 % de mémoire. La capacité par octet est bien meilleure, et le réparateur peut être entraîné sur T4 en quelques heures (30 M × 6 × 50 M tokens ≈ 9e15 FLOPs ≈ 5 h).

**Métriques.** Taux de réparation (tests qui passent après patch), nombre moyen d'itérations, tokens économisés vs régénération, régressions (tests qui passaient et cassent).

**Critère d'arrêt.** Si le réparateur ne fait pas mieux qu'une régénération complète à budget de tokens égal, ou s'il casse >5 % des tests qui passaient, on l'abandonne.

---

## M4 — PKM-Code : mémoire à clés-produit pour les idiomes de code

**Idée.** Remplacer 2 à 4 couches FFN par une **mémoire clé-valeur** de type product-key, mais dont les valeurs stockent des **idiomes de code** (signatures d'API, motifs, snippets) plutôt que des faits. La mémoire est indexée par le contexte caché, pas par une requête textuelle : pas de coût de contexte, pas de RAG externe.

**Dimensionnement sous 1 Go.** 256 k entrées × 64 dims × 2 bits = 4 Mo ; 4 M entrées = 64 Mo. On peut donc ajouter 4 M d'« items » de connaissance pour 64 Mo, sans augmenter les FLOPs (seules quelques entrées sont lues par token).

**Pourquoi ça peut marcher.** Memory Layers at Scale montre des gains sur HumanEval et MBPP, et un modèle 1,3 Md approche Llama2-7B en factualité. Le code est un domaine où les idiomes sont répétitifs, donc particulièrement adapté.

**Risque.** La mémoire est limitée par la bande passante : si chaque token lit 8 entrées de 64 dims, cela reste petit, mais les accès aléatoires peuvent dominer la latence CPU. À tester avant d'investir.

**Protocole.** À partir de `t4_base`, remplacer les FFN des couches 4 et 8 par une mémoire de 1 M puis 4 M d'entrées ; entraîner 20 % de tokens supplémentaires ; mesurer pass@1 sur des APIs rares et la latence CPU.

**Critère d'arrêt.** Si le gain de pass@1 est <1 point pour +64 Mo, la mémoire n'est pas rentable sous 1 Go.

---

## M5 — LDD : Loop-Distilled Depth avec allocation adaptative

**Idée.** Réutiliser la même pile de blocs 2 ou 3 fois (`loop`), avec une **supervision profonde** à chaque passage, puis apprendre un **prédicteur de difficulté** qui choisit le nombre de passages à l'inférence : 1 pour une complétion triviale, 2–3 pour une fonction complexe.

**Pourquoi c'est intéressant.** La mémoire ne bouge pas (mêmes poids), seule la latence augmente. MobileLLM montre que répéter les blocs gagne +0,7 %, et TRM montre que la supervision profonde est le vrai moteur. La combinaison « loop + deep supervision + allocation adaptative » n'est pas encore standard pour le code.

**Protocole.** `t4_small` avec `loop ∈ {1, 2, 3}`, entraîné sur le même nombre de tokens. À l'évaluation, mesurer la qualité vs le nombre moyen de passages. Le prédicteur est un petit MLP sur l'état caché du dernier token (1 M de paramètres), entraîné à prédire si le modèle à 1 passage réussit.

**Critère d'arrêt.** Si le gain à loop=2 est <0,5 point, ou si la latence moyenne double pour <1 point, on garde loop=1.

---

## M6 — MAP : Memory-Aware Planner et la loi « bits vs paramètres » sous 1 Go

**Idée.** Le planificateur (`blackcode.planner`) implémente une loi de précision (`N_eff = N(1−e^{−P/γ})`) et cherche le couple (taille, bits) qui minimise la perte sous une contrainte de RAM. Il prédit qu'à 1 Go, **2 bits / ~2,6 Md de paramètres** bat 4 bits / 1,5 Md et 8 bits / 336 M.

**Ce qui est nouveau.** Cette prédiction n'est pas vérifiée expérimentalement à petite échelle. On propose une expérience contrôlée :

- entraîner des modèles de 30–150 M en fp16, 4 bits et 2 bits **à budget mémoire égal** (donc des nombres de paramètres différents) ;
- utiliser la même recette de distillation et le même nombre de tokens ;
- mesurer la perte et le pass@1 par Mo.

C'est l'expérience qui valide (ou invalide) le pari central du projet. Elle tient sur T4 en quelques jours et n'a, à notre connaissance, pas été publiée dans ce régime de mémoire.

**Critère d'arrêt.** Si, à mémoire égale, le 4 bits bat le 2 bits sur deux échelles, on réoriente BlackCode vers un export 4 bits et une cible de 700–800 M de paramètres.

---

## M7 — Routeur agentique optimisé pour le coût par tâche

**Idée.** Un routeur de 1–5 M de paramètres décide, pour chaque étape d'un agent : (a) le petit modèle local, (b) un modèle plus gros (API), (c) un outil (compilateur, tests, recherche), (d) demander une clarification. Il est entraîné sur des traces avec une récompense **coût par tâche résolue**, pas seulement l'accuracy.

**Pourquoi c'est utile.** NVIDIA estime que 40–70 % des appels d'un agent peuvent être servis par un SLM. Le bon objectif n'est pas « un modèle qui fait tout », mais « un système qui résout le plus de tâches par euro ». Le routeur est la pièce qui rend BlackCode « généraliste » sans gonfler ses poids.

**Protocole.** Collecter 2 000–5 000 traces sur des tâches de code (HumanEval, MBPP, petits bugs synthétiques), entraîner le routeur par imitation puis par bandit, comparer au tout-petit-modèle et au tout-gros-modèle.

**Critère d'arrêt.** Si le routeur ne fait pas mieux que la meilleure politique fixe, il est inutile.

---

## Ce qui est réellement nouveau, et ce qui ne l'est pas

| Élément | Statut |
|---|---|
| Ternaire natif, hybride conv/attention, embeddings liés | **recombinaison** de résultats publiés |
| Annealing de quantification + suréchantillonnage par erreur de quantification dans TAD | **nouveau** (à notre connaissance) |
| MEM comme métrique de remplacement paramètres → calcul | **nouveau** |
| REN comme modèle de réparation séparé et minuscule | **nouveau** (la littérature utilise le même modèle) |
| PKM appliqué aux idiomes de code sous 1 Go | **nouveau** (Memory Layers est généraliste) |
| Validation expérimentale de la loi bits/paramètres sous 1 Go | **nouveau** dans ce régime |
| Routeur optimisé coût par tâche | **recombinaison** (LLM routing) |

Le projet n'a pas besoin que les sept méthodes marchent. **M1 + M2 + M6** suffisent à tenir la promesse « 1 Go, fort en code » ; M3–M5 et M7 sont des multiplicateurs.
