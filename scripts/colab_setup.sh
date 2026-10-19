#!/usr/bin/env bash
# Prépare une session Colab (ou toute machine Linux) en une commande.
#
#   bash scripts/colab_setup.sh          # installe et vérifie
#   DRIVE=/content/drive/MyDrive/blackcode bash scripts/colab_setup.sh
#
# Ne réinstalle pas torch : Colab fournit déjà la bonne version CUDA.
set -euo pipefail

REPO="${REPO:-https://github.com/Snowoo-2z/Llm-search.git}"
BRANCH="${BRANCH:-arena/aa605ee4-llm-search}"
WORK="${WORK:-/content/Llm-search}"
DRIVE="${DRIVE:-/content/drive/MyDrive/blackcode}"
PY="${PYTHON:-python}"

echo "== 1. dépôt =="
if [ -d "$WORK/.git" ]; then
  git -C "$WORK" fetch --depth 1 origin "$BRANCH"
  git -C "$WORK" checkout -q "$BRANCH"
  git -C "$WORK" reset --hard -q "origin/$BRANCH"
else
  git clone --depth 1 -b "$BRANCH" "$REPO" "$WORK"
fi
cd "$WORK"
echo "commit : $(git rev-parse --short HEAD)"

echo "== 2. dépendances =="
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q numpy tokenizers datasets transformers accelerate
# bitsandbytes : requis seulement pour un professeur en 4 bits
"$PY" -m pip install -q bitsandbytes || echo "bitsandbytes indisponible (professeur fp16 seulement)"

echo "== 3. matériel =="
"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU {p.name} | {p.total_memory/1e9:.1f} Go | {p.multi_processor_count} SM")
else:
    print("Pas de GPU : entraînement CPU possible, mais en heures par dizaine de pas.")
PY

echo "== 4. Drive =="
if [ -d /content/drive ]; then
  mkdir -p "$DRIVE/runs"
  echo "checkpoints : $DRIVE/runs"
else
  echo "pas de Drive monté (normal hors Colab) — les checkpoints resteront locaux"
fi

echo "== 5. présence des données =="
for d in data/v2 data/code; do
  if [ -f "$d/meta.json" ]; then
    "$PY" - "$d" <<'PY'
import json, sys
m = json.load(open(sys.argv[1] + "/meta.json"))
print(f"{sys.argv[1]} : {m.get('train_tokens', 0)/1e6:.1f} M tokens train, "
      f"{m.get('documents', 0)} documents, vocab {m.get('vocab_size')}")
PY
  fi
done

echo
echo "Prêt. Étapes suivantes :"
echo "  python -m blackcode.data build --src data/raw --out data/v2 --tokenizer hf:bigcode/starcoder2-3b"
echo "  python scripts/train_plan.py --config configs/blackcode_0.6b.json --stage distill"
