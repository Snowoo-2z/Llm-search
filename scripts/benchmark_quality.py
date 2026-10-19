#!/usr/bin/env python3
"""Benchmark vitesse filtrage qualité"""
import time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blackcode.data import quality_reasons, quality_score, is_french_prose, _french_heuristic_score

sample = ("La France est un pays d'Europe où le français est parlé. " * 20) + " " + "https://example.com " * 2
N=10000

t0=time.time()
for _ in range(N):
    quality_reasons(sample, "prose", "markdown")
print(f"quality_reasons: {N} docs en {time.time()-t0:.2f}s = {N/(time.time()-t0):.0f} docs/s")

t0=time.time()
for _ in range(N):
    quality_score(sample, "prose", "markdown")
print(f"quality_score: {N} docs en {time.time()-t0:.2f}s")

t0=time.time()
for _ in range(N):
    _french_heuristic_score(sample)
print(f"french_heuristic: {N} docs en {time.time()-t0:.2f}s")

# Test avec multiprocessing
from concurrent.futures import ProcessPoolExecutor
def filter_batch(texts):
    return [quality_reasons(t, "prose", "markdown") for t in texts]

texts=[sample]*N
t0=time.time()
with ProcessPoolExecutor(max_workers=4) as ex:
    chunks=[texts[i::4] for i in range(4)]
    list(ex.map(filter_batch, chunks))
print(f"parallel 4 workers: {N} docs en {time.time()-t0:.2f}s")
