#!/usr/bin/env bash
# Build ultra-optimisé v2 pour corpus français 260M — toutes les optimisations max
# Gains cumulés vs build naïf : heuristique ×50 + no-near ×2 RAM÷10 + quality_score + licence ligne + tokenizer 5 min + manifest + disque tar + parallel + prefetch
set -euo pipefail

# === Paramètres ===
MAX_MB="${MAX_MB:-4000}"
MIN_QUALITY_SCORE="${MIN_QUALITY_SCORE:-0.6}"
OUT="${OUT:-data/fr_260m}"
HF_DATASET="${HF_DATASET:-epfml/FineWeb2-HQ:fra_Latn}"
HF_FIELD="${HF_FIELD:-text}"
VOCAB="${VOCAB:-24576}"
SEQ_LEN="${SEQ_LEN:-2048}"
NO_NEAR="${NO_NEAR:-1}"  # 1 = exact dedup seulement, rapide, RAM÷10
TOKENIZER_CHARS="${TOKENIZER_CHARS:-20000000}"  # 20M chars pour tokenizer rapide (vs 120M défaut) → 5 min vs 30 min
FRENCH_ONLY="${FRENCH_ONLY:-1}"
INCLUDE_SMS="${INCLUDE_SMS:-0}"  # 1 = ajoute 88milSMS CC BY 4.0

# === Env optimisations ===
export TOKENIZERS_PARALLELISM=true
export BLACKCODE_THREADS=${BLACKCODE_THREADS:-$(nproc 2>/dev/null || echo 4)}
export PYTHONUNBUFFERED=1

mkdir -p data/fr_seed data/audits /tmp/blackcode_build "$OUT"

echo "[opt-v2] === Build FR ultra-optimisé v2 ==="
echo "[opt-v2] dataset=$HF_DATASET max_mb=$MAX_MB min_q=$MIN_QUALITY_SCORE vocab=$VOCAB seq_len=$SEQ_LEN no_near=$NO_NEAR tokenizer_chars=$TOKENIZER_CHARS"
echo "[opt-v2] threads=$BLACKCODE_THREADS TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM"
echo "[opt-v2] french_only=$FRENCH_ONLY include_sms=$INCLUDE_SMS"

# Audit rapide 500 lignes sans lingua (fallback heuristique 0.1 ms vs 5 ms lingua → ×50)
echo "[opt-v2] audit rapide 500 lignes (heuristique)..."
python scripts/audit_fr_source.py --dataset "$HF_DATASET" --limit 500 --sample-count 10 2>&1 | head -120 || echo "audit échoué, on continue"

# Build principal avec optimisations
ARGS="--src data/fr_seed --hf $HF_DATASET --hf-split train --hf-field $HF_FIELD --max-mb $MAX_MB --out $OUT --vocab $VOCAB --seq-len $SEQ_LEN --fim-rate 0 --agentic-frac 0 --seed 1337 --tokenizer-chars $TOKENIZER_CHARS"

if [ "$FRENCH_ONLY" = "1" ]; then
  ARGS="$ARGS --french-only"
fi

if [ "$NO_NEAR" = "1" ]; then
  ARGS="$ARGS --no-near"
fi

# Filtrage qualité + licence ligne
ARGS="$ARGS --min-quality-score $MIN_QUALITY_SCORE"

echo "[opt-v2] lancement build principal..."
echo "[opt-v2] python -m blackcode.data build $ARGS"
time python -m blackcode.data build $ARGS

echo "[opt-v2] vérif corpus principal..."
ls -lh "$OUT"/
python scripts/eval_french_purity.py --corpus-dir "$OUT" --limit 20 || true

# Optionnel : ajoute 88milSMS (5 Mo, CC BY 4.0) pour abréviations RS
if [ "$INCLUDE_SMS" = "1" ]; then
  SMS_OUT="${OUT}_sms"
  echo "[opt-v2] build SMS 88milSMS (CC BY 4.0, 88k SMS, abréviations)..."
  python -m blackcode.data build --src data/fr_seed --hf comere/cmr-88milsms --hf-field text --french-only --max-mb 100 --out "$SMS_OUT" --vocab "$VOCAB" --seq-len "$SEQ_LEN" --tokenizer-chars 5000000 --seed 1337 --no-near || echo "SMS build échoué (dataset non dispo)"
  ls -lh "$SMS_OUT"/ || true
  echo "[opt-v2] Pour fusionner SMS dans principal : cat $SMS_OUT/manifest.jsonl >> $OUT/manifest.jsonl && cat $SMS_OUT/train.bin >> $OUT/train.bin (recalcule meta)"
fi

# Copie optimisée vers Drive si dispo (tar + dd, pas cp -a qui bloque 25 min)
if [ -d "/content/drive/MyDrive" ]; then
  echo "[opt-v2] copie vers Drive (tar + dd, 10-30 min vs cp -a 25 min+ bloqué)..."
  mkdir -p /content/drive/MyDrive/blackcode
  # tar local → Drive
  if command -v pv >/dev/null 2>&1; then
    tar cf - "$OUT" | pv | (cd /content/drive/MyDrive/blackcode && tar xf -) && echo "copie Drive OK (pv)" || echo "copie Drive échouée"
  else
    tar cf - "$OUT" | (cd /content/drive/MyDrive/blackcode && tar xf -) && echo "copie Drive OK" || echo "copie Drive échouée"
  fi
  # Vérif taille Drive
  du -sh "/content/drive/MyDrive/blackcode/$OUT" 2>/dev/null || true
fi

echo "[opt-v2] terminé. Stats :"
cat "$OUT/meta.json" | python3 -m json.tool | head -30 || true

echo ""
echo "[opt-v2] Pour Kaggle dual-T4, créer dataset privé :"
echo "  kaggle datasets create -p $OUT --dir $OUT"
echo ""
echo "[opt-v2] Entraînement ultra-optimisé (dual-T4, Muon, DDP no_sync, fused, compile max-autotune, cudnn benchmark, TF32, prefetch 2) :"
echo "  torchrun --nproc_per_node=2 -m blackcode.train --config configs/blackcode_fr_260m.json --data $OUT --out runs/fr_260m_6b --batch-size 2 --grad-accum 8 --seq-len 2048 --max-steps 183105 --optim muon --muon-lr 0.02 --lr 6e-4 --fused --schedule wsd --stable-frac 0.8 --seq-len-warmup-frac 0.2 --checkpoint-policy selective --compile --compile-mode max-autotune --cudnn-benchmark --tf32 --prefetch 2 --bucket-cap-mb 10 --ddp --log-flops --eval-interval 1000 --save-interval 1000 --mirror-dir /content/drive/MyDrive/blackcode/runs --resume"
