#!/usr/bin/env bash
# Mesure le débit / la VRAM réels sur 2 T4 avec l'élève 0.6B et StarCoder2-3B.
# Le corpus aléatoire sert UNIQUEMENT à mesurer le calcul, pas à entraîner un modèle utile.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PYTHON:-python}"
SMOKE="${SMOKE:-/tmp/blackcode_t4_benchmark}"
LOG="${LOG:-/kaggle/working/blackcode_t4_benchmark.log}"
export HF_HOME="${HF_HOME:-/tmp/blackcode_hf_cache}"

"$PY" - <<'PY'
import importlib.util
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise SystemExit(f"Il faut 2 GPU visibles; trouvé {torch.cuda.device_count()}")
missing = [name for name in ("transformers", "accelerate", "bitsandbytes")
           if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Dépendances manquantes : " + ", ".join(missing) +
                     ". Installer avec : pip install -q transformers accelerate bitsandbytes")
print("PyTorch", torch.__version__, "CUDA", torch.version.cuda)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name}, {p.total_memory/1e9:.1f} Go")
PY

rm -rf "$SMOKE"
mkdir -p "$SMOKE/data" "$(dirname "$LOG")"
"$PY" - "$SMOKE" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
rng = np.random.default_rng(20261018)
rng.integers(0, 49152, size=2_000_000, dtype=np.uint16).tofile(root / "data/train.bin")
rng.integers(0, 49152, size=128_000, dtype=np.uint16).tofile(root / "data/val.bin")
(root / "data/meta.json").write_text(json.dumps({
    "vocab_size": 49152, "train_tokens": 2_000_000, "val_tokens": 128_000
}))
PY

# 20 pas x 32 768 tokens globaux ≈ 655k tokens. Pas d'évaluation ni de miroir.
# Le checkpoint temporaire et le corpus synthétique seront supprimés si le test passe.
set +e
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m blackcode.distill \
  --teacher bigcode/starcoder2-3b --load-4bit \
  --config configs/blackcode_0.6b.json --data "$SMOKE/data" \
  --out "$SMOKE/run" --seq-len 2048 --batch-size 1 --grad-accum 8 \
  --max-steps 20 --topk 32 --alpha 0.5 --temperature 2.0 \
  --optim adafactor --lr 3e-4 --schedule wsd \
  --checkpoint-policy selective --eval-interval 0 --save-interval 0 --log-interval 1 \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
if [ "$rc" -ne 0 ]; then
  echo "ÉCHEC du benchmark (code $rc). Les artefacts restent dans $SMOKE; voir $LOG."
  exit "$rc"
fi

"$PY" - "$LOG" <<'PY'
import json, sys
from pathlib import Path
records = []
for line in Path(sys.argv[1]).read_text().splitlines():
    try:
        item = json.loads(line)
    except json.JSONDecodeError:
        continue
    if "tok_s" in item:
        records.append(item)
if not records:
    raise SystemExit("Aucune mesure tok_s trouvée dans le journal")
last = records[-1]
print(f"DÉBIT À RETENIR : {last['tok_s']:.1f} tokens/s globaux à l'étape {last['step']}")
print(f"ESTIMATION POUR 30 M TOKENS : {30_000_000/last['tok_s']/3600:.2f} h "
      "(approximation, à confirmer sur corpus réel)")
PY
rm -rf "$SMOKE"
echo "Benchmark terminé. Journal conservé : $LOG"
