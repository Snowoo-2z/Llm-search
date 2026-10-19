#!/usr/bin/env bash
# Lance t4_small sur T4 avec reprise, WSD et checkpoints asynchrones.
# Usage :
#   OUT=/content/drive/MyDrive/blackcode/runs/t4_small MAX_STEPS=3000 ./scripts/train_t4.sh
# Pour un modèle de 0,6-1,2 Md, ajouter --optim adafactor (mémoire).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
OUT="${OUT:-runs/t4_small}"
DATA="${DATA:-data/code}"
MAX_STEPS="${MAX_STEPS:-3000}"
OPTIM="${OPTIM:-adamw}"
MIRROR="${MIRROR:-}"

ARGS=(--config configs/t4_small.json --data "$DATA" --out "$OUT"
      --batch-size "${BATCH:-8}" --grad-accum "${ACCUM:-4}" --seq-len "${SEQ:-1024}"
      --max-steps "$MAX_STEPS" --lr 6e-4 --min-lr 6e-5 --warmup 200
      --optim "$OPTIM" --schedule wsd --stable-frac 0.8
      --checkpoint-policy selective --compile --log-flops
      --eval-interval 500 --save-interval 500)
if [ -n "$MIRROR" ]; then ARGS+=(--mirror-dir "$MIRROR"); fi
if [ -f "$OUT/ckpt.pt" ]; then ARGS+=(--resume); fi
"$PY" -m blackcode.train "${ARGS[@]}"
