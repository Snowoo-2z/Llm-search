#!/usr/bin/env bash
# Train ultra-optimisé v3 MAX — toutes les optimisations
set -euo pipefail

CONFIG="${CONFIG:-configs/blackcode_fr_260m.json}"
DATA="${DATA:-data/fr_260m}"
OUT="${OUT:-runs/fr_260m_6b}"
BATCH="${BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
SEQ_LEN="${SEQ_LEN:-2048}"
MAX_STEPS="${MAX_STEPS:-183105}"
OPTIM="${OPTIM:-muon}"
LR="${LR:-6e-4}"
MUON_LR="${MUON_LR:-0.02}"

# Détection dual-T4
if [ -n "${RANK:-}" ] || [ "${DDP:-0}" = "1" ]; then
  DDP_ARGS="--ddp --ddp-fp16-compress --ddp-static-graph --bucket-cap-mb 10"
  BATCH=4
  GRAD_ACCUM=8
  WORLD="dual-T4"
else
  DDP_ARGS=""
  BATCH=2
  GRAD_ACCUM=16
  WORLD="mono-T4"
fi

echo "[train-ultra-v3] config=$CONFIG data=$DATA out=$OUT batch=$BATCH accum=$GRAD_ACCUM seq_len=$SEQ_LEN max_steps=$MAX_STEPS optim=$OPTIM world=$WORLD"

# Optimisations globales
export TOKENIZERS_PARALLELISM=true
export BLACKCODE_THREADS=${BLACKCODE_THREADS:-$(nproc 2>/dev/null || echo 4)}
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1  # pour NCCL async
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Commande ultra
CMD="python -m blackcode.train --config $CONFIG --data $DATA --out $OUT --batch-size $BATCH --grad-accum $GRAD_ACCUM --seq-len $SEQ_LEN --max-steps $MAX_STEPS --optim $OPTIM --muon-lr $MUON_LR --lr $LR --fused --foreach --schedule wsd --stable-frac 0.8 --seq-len-warmup-frac 0.2 --checkpoint-policy selective --compile --compile-mode max-autotune --compile-fullgraph --cudnn-benchmark --tf32 --prefetch 2 $DDP_ARGS --log-flops --eval-interval 1000 --save-interval 1000 --mirror-dir ${MIRROR_DIR:-/content/drive/MyDrive/blackcode/runs} --resume"

echo "[train-ultra-v3] $CMD"

if [ -n "${RANK:-}" ]; then
  # déjà sous torchrun
  $CMD
else
  if [ "$WORLD" = "dual-T4" ]; then
    torchrun --nproc_per_node=2 -m blackcode.train --config $CONFIG --data $DATA --out $OUT --batch-size 4 --grad-accum 8 --seq-len $SEQ_LEN --max-steps $MAX_STEPS --optim $OPTIM --muon-lr $MUON_LR --lr $LR --fused --foreach --schedule wsd --stable-frac 0.8 --seq-len-warmup-frac 0.2 --checkpoint-policy selective --compile --compile-mode max-autotune --compile-fullgraph --cudnn-benchmark --tf32 --prefetch 2 --bucket-cap-mb 10 --ddp --ddp-fp16-compress --ddp-static-graph --log-flops --eval-interval 1000 --save-interval 1000 --resume
  else
    $CMD
  fi
fi
