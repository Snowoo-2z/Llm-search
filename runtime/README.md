# bcrun — moteur d'inférence C99

Moteur minimal pour les paquets `.bcpkg` : mmap, GEMV ternaire 2 bits,
embeddings int8, RMSNorm, RoPE, GQA avec cache KV fp16, convolutions courtes.

```bash
make                      # cc -O3 -march=native -fopenmp
./bcrun --pkg m.bcpkg --ids 1,2,3 --max-new 16 --context 131072 --mem-report
```

Options : `--pkg`, `--ids`, `--max-new`, `--context`, `--temp`, `--topk`,
`--dump-logits`, `--mem-report`, `--debug`, `--quiet`.

Mesure de référence (1,19 Md, 8k) : **431 Mo de RSS pic**, dont 1,6 Mo de
runtime. Voir `docs/08-RAM-MESUREE.md`.

Limites connues : mono-nœud, pas de SIMD explicite (le compilateur
auto-vectorise partiellement), pas de batching, pas de tokenizer. Vitesse
mesurée : ~2,6 tok/s sur 2 cœurs pour 1,19 Md — la mémoire est acquise, la
vitesse est le prochain chantier (SIMD, tuilage, threads).
