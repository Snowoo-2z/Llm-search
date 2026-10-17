#!/usr/bin/env bash
# Lance t4_small sur T4 avec reprise automatique.
# Usage : OUT=/content/drive/MyDrive/blackcode/runs/t4_small MAX_STEPS=3000 ./scripts/train_t4.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
OUT="${OUT:-runs/t4_small}"
DATA="${DATA:-data/code}"
MAX_STEPS="${MAX_STEPS:-3000}"

ARGS=(--config configs/t4_small.json --data "$DATA" --out "$OUT"
      --batch-size "${BATCH:-8}" --grad-accum "${ACCUM:-4}" --seq-len "${SEQ:-1024}"
      --max-steps "$MAX_STEPS" --lr 6e-4 --min-lr 6e-5 --warmup 200
      --eval-interval 500 --save-interval 500 --grad-checkpoint)
if [ -f "$OUT/ckpt.pt" ]; then ARGS+=(--resume); fi
"$PY" -m blackcode.train "${ARGS[@]}"
