#!/usr/bin/env bash
# Build optimisé max pour corpus français 260M
# Optimisations : fallback heuristique sans lingua, no-near première passe, tokenizer léger, disque local, tar
set -euo pipefail

MAX_MB="${MAX_MB:-4000}"
MIN_QUALITY_SCORE="${MIN_QUALITY_SCORE:-0.6}"
OUT="${OUT:-data/fr_260m}"
HF_DATASET="${HF_DATASET:-epfml/FineWeb2-HQ:fra_Latn}"
HF_FIELD="${HF_FIELD:-text}"
VOCAB="${VOCAB:-24576}"
SEQ_LEN="${SEQ_LEN:-2048}"
NO_NEAR="${NO_NEAR:-1}"  # 1 = exact dedup seulement, rapide

mkdir -p data/fr_seed data/audits /tmp/blackcode_build

echo "[opt-build] dataset=$HF_DATASET max_mb=$MAX_MB min_q=$MIN_QUALITY_SCORE vocab=$VOCAB seq_len=$SEQ_LEN no_near=$NO_NEAR"

# Audit rapide 500 lignes sans lingua (fallback)
echo "[opt-build] audit rapide 500 lignes..."
python scripts/audit_fr_source.py --dataset "$HF_DATASET" --limit 500 --sample-count 10 2>&1 | head -100 || echo "audit échoué, on continue"

# Build avec optimisations
ARGS="--src data/fr_seed --hf $HF_DATASET --hf-split train --hf-field $HF_FIELD --french-only --min-quality-score $MIN_QUALITY_SCORE --max-mb $MAX_MB --out $OUT --vocab $VOCAB --seq-len $SEQ_LEN --fim-rate 0 --agentic-frac 0 --seed 1337"

if [ "$NO_NEAR" = "1" ]; then
  ARGS="$ARGS --no-near"
fi

echo "[opt-build] lancement build..."
python -m blackcode.data build $ARGS

echo "[opt-build] vérif..."
ls -lh "$OUT"/
python scripts/eval_french_purity.py --corpus-dir "$OUT" --limit 20 || true

# Copie optimisée vers Drive si dispo
if [ -d "/content/drive/MyDrive" ]; then
  echo "[opt-build] copie vers Drive (tar)..."
  mkdir -p /content/drive/MyDrive/blackcode
  tar cf - "$OUT" | (cd /content/drive/MyDrive/blackcode && tar xf -) && echo "copie Drive OK" || echo "copie Drive échouée"
fi

echo "[opt-build] terminé. Pour Kaggle, créer dataset privé :"
echo "  kaggle datasets create -p $OUT --dir $OUT"
