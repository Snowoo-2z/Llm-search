#!/usr/bin/env bash
# Valide toute la chaîne en ~10 minutes, sur Colab comme en local.
# À lancer avant de dépenser une session de 4 h.
#
#   bash scripts/colab_smoke.sh
#
# Étapes : corpus v2 (60 Mo) → entraînement 100 pas → distillation 20 pas.
# Si l'une échoue, inutile de lancer la vraie séance.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
RAW="${RAW:-data/raw}"
SMOKE="${SMOKE:-/tmp/blackcode_smoke}"

if [ ! -d "$RAW" ] && [ ! -d data/v2 ]; then
  echo "ERREUR : ni $RAW ni data/v2. Construire le corpus d'abord :"
  echo "  python scripts/fetch_corpus.py --out data/raw --pypi popular --github big --max-mb 1100"
  exit 1
fi

SRC="$RAW"
[ -d data/v2 ] && SRC="data/v2"
mkdir -p "$SMOKE"

echo "== 1. vérification du tokenizer du professeur =="
"$PY" - <<'PY'
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bigcode/starcoder2-3b")
    print(f"starcoder2-3b : {len(tok)} jetons")
    for name in ("<|endoftext|>", "<fim_prefix>", "<fim_middle>", "<fim_suffix>"):
        print(f"  {name:14s} → {tok.convert_tokens_to_ids(name)}")
    print("ATTENDU : 49152. Toute autre valeur invalide le plan de distillation.")
except Exception as exc:  # hors ligne, ou transformers absent
    print(f"vérification impossible ({exc}) — à refaire sur Colab")
PY

echo "== 2. corpus v2 sur 60 Mo =="
"$PY" -m blackcode.data build --src "$SRC" --out "$SMOKE/data" \
  --max-mb 60 --seq-len 1024 --fim-rate 0.5 \
  --agentic-frac 0.15 --agentic-exec --max-spawns 40

echo "== 3. entraînement court (100 pas) =="
"$PY" -m blackcode.train --config configs/t4_small.json --data "$SMOKE/data" \
  --out "$SMOKE/train" --batch-size 4 --grad-accum 2 --seq-len 512 \
  --max-steps 100 --optim muon --schedule wsd --compile --log-flops \
  --eval-interval 50 --save-interval 50

echo "== 4. distillation courte (20 pas, professeur = le modèle qu'on vient d'entraîner) =="
"$PY" -m blackcode.distill --teacher "local:$SMOKE/train/ckpt.pt" \
  --config configs/t4_small.json --data "$SMOKE/data" --out "$SMOKE/distill" \
  --seq-len 256 --batch-size 2 --grad-accum 1 --max-steps 20 \
  --optim adafactor --lr 3e-4 --log-interval 5 --eval-interval 0 --save-interval 0

echo
echo "Chaîne validée. Lancer le vrai plan :"
echo "  python scripts/train_plan.py --config configs/blackcode_0.6b.json --stage distill"
