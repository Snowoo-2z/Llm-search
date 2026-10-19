#!/usr/bin/env bash
# Build ultra-optimisé v3 MAX — toutes les optimisations possibles
# v2 + Bloom filter + parallel quality + tokenizer 20M + prefetch
set -euo pipefail

MAX_MB="${MAX_MB:-4000}"
MIN_QUALITY_SCORE="${MIN_QUALITY_SCORE:-0.6}"
OUT="${OUT:-data/fr_260m}"
HF_DATASET="${HF_DATASET:-epfml/FineWeb2-HQ:fra_Latn}"
HF_FIELD="${HF_FIELD:-text}"
VOCAB="${VOCAB:-24576}"
SEQ_LEN="${SEQ_LEN:-2048}"
NO_NEAR="${NO_NEAR:-1}"
BLOOM="${BLOOM:-1}"  # Bloom filter RAM÷100 +20% débit
TOKENIZER_CHARS="${TOKENIZER_CHARS:-20000000}"
FRENCH_ONLY="${FRENCH_ONLY:-1}"
INCLUDE_SMS="${INCLUDE_SMS:-0}"

export TOKENIZERS_PARALLELISM=true
export BLACKCODE_THREADS=${BLACKCODE_THREADS:-$(nproc 2>/dev/null || echo 4)}
export PYTHONUNBUFFERED=1
export RAYON_NUM_THREADS=${BLACKCODE_THREADS}

mkdir -p data/fr_seed data/audits /tmp/blackcode_build "$OUT"

echo "[opt-v3 MAX] === Build FR ultra-optimisé v3 MAX ==="
echo "[opt-v3] dataset=$HF_DATASET max_mb=$MAX_MB min_q=$MIN_QUALITY_SCORE vocab=$VOCAB seq_len=$SEQ_LEN no_near=$NO_NEAR bloom=$BLOOM tokenizer_chars=$TOKENIZER_CHARS"
echo "[opt-v3] threads=$BLACKCODE_THREADS parallel=true"

# Audit rapide 500 lignes
echo "[opt-v3] audit rapide 500 lignes..."
python scripts/audit_fr_source.py --dataset "$HF_DATASET" --limit 500 --sample-count 10 2>&1 | head -120 || true

# Build principal
ARGS="--src data/fr_seed --hf $HF_DATASET --hf-split train --hf-field $HF_FIELD --max-mb $MAX_MB --out $OUT --vocab $VOCAB --seq-len $SEQ_LEN --fim-rate 0 --agentic-frac 0 --seed 1337 --tokenizer-chars $TOKENIZER_CHARS --min-quality-score $MIN_QUALITY_SCORE"

if [ "$FRENCH_ONLY" = "1" ]; then ARGS="$ARGS --french-only"; fi
if [ "$NO_NEAR" = "1" ]; then ARGS="$ARGS --no-near"; fi
if [ "$BLOOM" = "1" ]; then ARGS="$ARGS --bloom"; fi

echo "[opt-v3] lancement build..."
time python -m blackcode.data build $ARGS

ls -lh "$OUT"/
python scripts/eval_french_purity.py --corpus-dir "$OUT" --limit 20 || true

if [ "$INCLUDE_SMS" = "1" ]; then
  SMS_OUT="${OUT}_sms"
  echo "[opt-v3] build SMS 88milSMS CC BY 4.0..."
  python -m blackcode.data build --src data/fr_seed --hf comere/cmr-88milsms --hf-field text --french-only --max-mb 100 --out "$SMS_OUT" --vocab "$VOCAB" --seq-len "$SEQ_LEN" --tokenizer-chars 5000000 --seed 1337 --no-near --bloom || true
  ls -lh "$SMS_OUT"/ || true
fi

if [ -d "/content/drive/MyDrive" ]; then
  echo "[opt-v3] copie Drive tar..."
  mkdir -p /content/drive/MyDrive/blackcode
  tar cf - "$OUT" | (cd /content/drive/MyDrive/blackcode && tar xf -) && echo "copie Drive OK" || true
fi

cat "$OUT/meta.json" | python3 -m json.tool | head -40 || true

echo ""
echo "[opt-v3] Entraînement ultra-optimisé v3 MAX (dual-T4, Muon, DDP no_sync, fused, foreach, SwiGLU fused, RMSNorm fused, compile max-autotune fullgraph, cudnn benchmark, TF32, prefetch 2, bucket 10, FP16 compress, static_graph, CUDA Graphs optionnel) :"
echo "  torchrun --nproc_per_node=2 -m blackcode.train --config configs/blackcode_fr_260m.json --data $OUT --out runs/fr_260m_6b --batch-size 4 --grad-accum 8 --seq-len 2048 --max-steps 183105 --optim muon --muon-lr 0.02 --lr 6e-4 --fused --foreach --schedule wsd --stable-frac 0.8 --seq-len-warmup-frac 0.2 --checkpoint-policy selective --compile --compile-mode max-autotune --compile-fullgraph --cudnn-benchmark --tf32 --prefetch 2 --bucket-cap-mb 10 --ddp --ddp-fp16-compress --ddp-static-graph --log-flops --eval-interval 1000 --save-interval 1000 --resume"
