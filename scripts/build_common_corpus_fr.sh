#!/usr/bin/env bash
# Construit un corpus à partir de PleIAs Common Corpus avec filtrage français + licences
# Nécessite un audit préalable via audit_fr_source.py
# Usage :
#   HF_DATASET=PleIAs/common_corpus MAX_MB=4000 bash scripts/build_common_corpus_fr.sh
set -euo pipefail

MAX_MB="${MAX_MB:-4000}"
OUT="${OUT:-data/common_fr}"
HF_DATASET="${HF_DATASET:-PleIAs/common_corpus}"
HF_FIELD="${HF_FIELD:-text}"
VOCAB="${VOCAB:-24576}"
SEQ_LEN="${SEQ_LEN:-2048}"

mkdir -p data/fr_seed

echo "[build-common] dataset=$HF_DATASET max_mb=$MAX_MB out=$OUT"

python -m blackcode.data build \
  --src data/fr_seed \
  --hf "$HF_DATASET" \
  --hf-split train \
  --hf-field "$HF_FIELD" \
  --hf-license-field license \
  --require-hf-license \
  --allowed-hf-license "MIT" \
  --allowed-hf-license "Apache 2.0" \
  --allowed-hf-license "CC0" \
  --allowed-hf-license "CC BY" \
  --allowed-hf-license "public domain" \
  --allowed-hf-license "Licence Ouverte" \
  --french-only \
  --max-mb "$MAX_MB" \
  --out "$OUT" \
  --vocab "$VOCAB" \
  --seq-len "$SEQ_LEN" \
  --fim-rate 0 \
  --agentic-frac 0 \
  --seed 1337

echo "[build-common] terminé. Vérifier $OUT/meta.json et manifest.jsonl"
echo "[build-common] ATTENTION : la liste allowed-hf-license ci-dessus est un exemple,"
echo "elle doit être validée juridiquement et affinée après audit (voir docs/17-CAHIER-SOURCES-FR.md)."
