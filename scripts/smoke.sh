#!/usr/bin/env bash
# Valide le pipeline complet sur CPU en quelques minutes.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
CORPUS="${CORPUS:-/tmp/corpus_algorithms}"

if [ ! -d "$CORPUS" ]; then
  git clone --depth 1 https://github.com/TheAlgorithms/Python.git "$CORPUS"
fi

"$PY" -m blackcode.corpus prepare --src "$CORPUS" --out data/smoke \
  --vocab 8192 --suffixes .py
"$PY" -m blackcode.train --config configs/nano.json --data data/smoke \
  --out runs/smoke --batch-size 8 --seq-len 128 --max-steps 300 \
  --lr 1.5e-3 --eval-interval 100 --save-interval 100
"$PY" -m blackcode.generate --ckpt runs/smoke/best.pt \
  --tokenizer data/smoke/tokenizer.json --prompt "def add(a, b):" --max-new-tokens 60
"$PY" -m blackcode.export --ckpt runs/smoke/best.pt \
  --out artifacts/smoke.bcpkg --context 256
"$PY" -m unittest discover -s tests -v
echo "OK — pipeline validé."
