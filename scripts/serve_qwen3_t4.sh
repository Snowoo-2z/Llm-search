#!/usr/bin/env bash
# Lance le modèle de base préentraîné sur 2 GPU NVIDIA T4 (ex. Kaggle dual-T4).
# Installer une version de vLLM compatible avec l'image CUDA du fournisseur avant.
set -euo pipefail

MODEL="${BLACKCODE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
TP_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
PORT="${PORT:-8000}"

if ! command -v vllm >/dev/null 2>&1; then
  echo "vLLM n'est pas installé. Installez une version compatible avec l'image CUDA de ce runtime GPU." >&2
  exit 1
fi

exec vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --dtype half \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
