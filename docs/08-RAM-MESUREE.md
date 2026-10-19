# Est-ce que ça tient dans 1 Go ? — mesure, pas estimation

> Question : une fois entraîné, le modèle tiendra-t-il dans 1 Go de RAM pour répondre ?
> Réponse courte : **oui — 431 Mo mesurés à 8 000 tokens de contexte**, avec le runtime C de ce dépôt. Non avec PyTorch.

---

## 1. Pourquoi la mesure est valable avant l'entraînement

La RAM consommée à l'inférence ne dépend **pas des valeurs des poids**, uniquement :

- de leur **nombre** ;
- de leur **encodage** (2 bits, 8 bits, fp16) ;
- du **cache KV** (nombre de couches d'attention × têtes KV × dimension × contexte) ;
- du **runtime**.

Un modèle avec des poids aléatoires a donc exactement la même empreinte qu'un modèle entraîné. C'est ce qui permet de répondre aujourd'hui, sans attendre la phase 2.

Protocole :

```bash
python scripts/make_random_bcpkg.py --config configs/blackcode_1b.json \
    --out /tmp/rss_blackcode_1b.bcpkg --context 8192
./runtime/bcrun --pkg /tmp/rss_blackcode_1b.bcpkg --ids 1 --max-new 3 \
    --context 8192 --temp 0 --mem-report
```

Le paquet fait **332,4 Mo** (2,35 bits/paramètre effectifs pour 1 187 M de paramètres), dont :

| Composant | Taille |
|---|---:|
| poids ternaires (2 bits + échelle) | 267,9 Mo |
| embeddings int8 (32k × 2048) | 64,1 Mo |
| normes, convolutions | 0,5 Mo |

---

## 2. Résultats mesurés

`ru_maxrss` du processus `bcrun`, en incluant les pages du fichier mmap réellement touchées :

| Contexte | Cache KV | **RSS pic** | Marge / 1 Go | Débit (2 threads) |
|---:|---:|---:|---:|---:|
| 512 | 6 Mo | **335,9 Mo** | 688 Mo | 2,6 tok/s |
| 1 024 | 12 Mo | **341,7 Mo** | 682 Mo | 2,4 tok/s |
| 2 048 | 24 Mo | **355,7 Mo** | 668 Mo | 2,6 tok/s |
| 4 096 | 48 Mo | **384,7 Mo** | 639 Mo | 2,5 tok/s |
| 8 192 | 96 Mo | **431,1 Mo** | 593 Mo | 2,7 tok/s |
| 16 384 | 192 Mo | **535,5 Mo** | 488 Mo | — |

Décomposition à 8k :

```
runtime de base (code + heap)          1,6 Mo
poids (mmap, 332,4 Mo)               332,4 Mo
cache KV (6 couches × 8k)             96,0 Mo
cache convolution + activations        1,2 Mo
---------------------------------------------
total mesuré                         431,1 Mo
```

Le runtime lui-même (hors poids et hors cache) pèse **1,6 Mo**. C'est le chiffre qui compte : PyTorch, lui, consomme ~500 Mo avant même de charger un modèle.

### 2.1 Ce que le chiffre inclut (et ce qu'il n'inclut pas)

**Inclus** : le binaire, la libc, le header JSON parsé (~1 Mo), les poids faulés depuis le fichier, le cache KV alloué et touché en entier, les buffers d'activation.

**Non inclus** : le système d'exploitation, le shell, un éventuel tokenizer en Python (compter 20–40 Mo si vous l'appelez depuis Python ; le runtime C lit des IDs).

**Nuance importante** : les 332 Mo de poids sont des **pages de fichier** (mmap en lecture seule). Le noyau peut les évincer sous pression mémoire et les recharger à la demande : le modèle fonctionnerait toujours avec moins de RAM physique, simplement plus lentement. La mémoire *anonyme* réellement indispensable (KV + buffers) est de **~100 Mo à 8k**. Autrement dit : 431 Mo est le chiffre « à chaud », et c'est le pire cas réaliste.

---

## 3. Comparaison honnête

| Solution | Paramètres | RSS attendu | Tient sous 1 Go ? |
|---|---:|---:|:---:|
| Qwen2.5-Coder-1.5B en Q4 via llama.cpp | 1,54 Md | ~1,2 Go | non |
| BitNet 2B4T via bitnet.cpp | 2,4 Md | ~1,3 Go | non |
| Qwen3-0.6B en Q4 | 0,6 Md | ~0,6 Go | oui, mais 0,6 Md |
| BlackCode-1.2B, PyTorch | 1,19 Md | > 1 Go (runtime seul ~0,5 Go) | non |
| **BlackCode-1.2B, runtime C** | **1,19 Md** | **431 Mo à 8k** | **oui, avec 593 Mo de marge** |

C'est exactement l'objectif : **ne pas réduire la taille**, et tenir quand même.

---

## 4. Et la vitesse ?

**2,5 à 2,7 tokens/s** sur 2 cœurs, avec un moteur C scalaire (pas de SIMD, pas de noyaux spécialisés). C'est lent, et il faut le dire clairement :

- le moteur actuel est écrit pour la **correction et la mesure mémoire**, pas pour la vitesse ;
- `bitnet.cpp` atteint 30–45 tok/s sur un 2,4 Md grâce à des noyaux AVX2/NEON optimisés et plusieurs cœurs ;
- les pistes évidentes pour ce moteur : SIMD sur la boucle ternaire, tuilage, pré-décodage des codes, threads.

Ordre de grandeur atteignable sur 8 cœurs modernes avec de bons noyaux : 15–30 tok/s. Sur un Raspberry Pi, 1–3 tok/s. La mémoire est acquise ; la vitesse est un travail d'ingénierie connu.

---

## 5. Validation : le moteur C reproduit-il PyTorch ?

Oui, à la quantification près. Sur le modèle `nano` entraîné (2,16 M), mêmes tokens, mêmes logits :

| Paquet | Erreur relative max | Argmax identique | Top-5 identique |
|---|---:|:---:|:---:|
| embeddings fp16 | **0,014 %** | oui (265) | oui |
| embeddings int8 | **0,127 %** | oui (265) | oui |

Le test est automatisé (`tests/test_runtime.py`) et rejoue la comparaison à chaque exécution de la suite.

---

## 6. Ce que ça change pour le projet

1. **La cible « 1 Go » n'est plus une hypothèse, c'est un résultat mesuré** : 431 Mo à 8k, 536 Mo à 16k.
2. **On peut voir plus grand** : avec 1 Go, on a la place pour ~2,5 Md de paramètres en 2 bits à 8k (700 Mo de poids + 200 Mo de KV). Le plafond devient la qualité du ternaire, pas la RAM.
3. **Le runtime est un livrable à part entière** — il l'est désormais : `runtime/bcrun.c`, 38 Ko de binaire, format `.bcpkg` documenté.
4. **La vitesse reste le chantier ouvert** : c'est le seul chiffre qui n'est pas au niveau (2,6 tok/s).

---

## 7. Reproductibilité

```bash
make -C runtime
python scripts/make_random_bcpkg.py --config configs/blackcode_1b.json \
    --out /tmp/rss.bcpkg --context 8192
./scripts/measure_rss.sh configs/blackcode_1b.json 8192
python -m unittest tests.test_runtime -v
```

Les paquets `.bcpkg` et les exécutables ne sont pas versionnés (voir `.gitignore`) : ils se régénèrent en ~25 s.
