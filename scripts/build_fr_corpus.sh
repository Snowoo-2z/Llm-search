#!/usr/bin/env bash
# Construit un premier corpus de pré-entraînement français sans données Discord.
# À lancer sur un runtime disposant de l'accès Hugging Face et assez de RAM.
# Améliorations : support multi-source, allowlist de licences, manifest traçable.
set -euo pipefail

MAX_MB="${MAX_MB:-4000}"
MIN_QUALITY_SCORE="${MIN_QUALITY_SCORE:-0.0}"
OUT="${OUT:-data/fr_260m}"
HF_DATASET="${HF_DATASET:-epfml/FineWeb2-HQ:fra_Latn}"
HF_FIELD="${HF_FIELD:-text}"
VOCAB="${VOCAB:-24576}"
SEQ_LEN="${SEQ_LEN:-2048}"

mkdir -p data/fr_seed

python - <<'PY'
import importlib.util
missing = [name for name in ("datasets", "lingua") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Dépendances manquantes : " + ", ".join(missing) +
                     ". Installer avec : pip install datasets lingua-language-detector")
PY

echo "[build] dataset=$HF_DATASET max_mb=$MAX_MB min_quality=$MIN_QUALITY_SCORE out=$OUT vocab=$VOCAB seq_len=$SEQ_LEN"

python -m blackcode.data build \
  --src data/fr_seed \
  --hf "$HF_DATASET" \
  --hf-split train \
  --hf-field "$HF_FIELD" \
  --french-only \
  --min-quality-score "$MIN_QUALITY_SCORE" \
  --max-mb "$MAX_MB" \
  --out "$OUT" \
  --vocab "$VOCAB" \
  --seq-len "$SEQ_LEN" \
  --fim-rate 0 \
  --agentic-frac 0 \
  --seed 1337

echo "[build] terminé. Vérifier $OUT/meta.json et $OUT/manifest.jsonl"
echo "[build] stats rapides :"
python - <<PY
import json, pathlib
meta=pathlib.Path("$OUT/meta.json")
if meta.exists():
    print(meta.read_text()[:2000])
PY
