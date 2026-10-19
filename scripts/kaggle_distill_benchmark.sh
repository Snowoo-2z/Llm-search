#!/usr/bin/env bash
# Mesure le débit réel avec l'élève 0.6B et StarCoder2-3B sur 1 ou 2 T4.
# GPU_COUNT=1 fait le baseline mono-T4; GPU_COUNT=2 lance DDP.
# Le corpus aléatoire sert UNIQUEMENT à mesurer le calcul, pas à entraîner un modèle utile.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PYTHON:-python}"
SMOKE="${SMOKE:-/tmp/blackcode_t4_benchmark}"
LOG="${LOG:-/kaggle/working/blackcode_t4_benchmark.log}"
export HF_HOME="${HF_HOME:-/tmp/blackcode_hf_cache}"
GPU_COUNT="${GPU_COUNT:-2}"
if [[ "$GPU_COUNT" != 1 && "$GPU_COUNT" != 2 ]]; then
  echo "GPU_COUNT doit valoir 1 ou 2" >&2; exit 2
fi
if [[ "$GPU_COUNT" == 1 ]]; then
  # Cache le second T4 pour un vrai baseline mono-GPU.
  export CUDA_VISIBLE_DEVICES="${SINGLE_GPU:-0}"
fi
BATCH_SIZE=1
GRAD_ACCUM=$((16 / GPU_COUNT))
export GPU_COUNT

"$PY" - <<'PY'
import importlib.util, os
import torch
expected = int(os.environ["GPU_COUNT"])
if not torch.cuda.is_available() or torch.cuda.device_count() < expected:
    raise SystemExit(f"Il faut {expected} GPU visibles; trouvé {torch.cuda.device_count()}")
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

# Batch global identique : microbatch 1, accumulation 16 sur un GPU ou 8 sur deux.
# 20 pas x 32 768 tokens globaux ≈ 655k tokens, quel que soit le GPU_COUNT.
# Pas d'évaluation ni de miroir. Le checkpoint temporaire et le corpus synthétique
# seront supprimés si le test passe.
TRAIN_ARGS=(--teacher bigcode/starcoder2-3b --load-4bit
  --config configs/blackcode_0.6b.json --data "$SMOKE/data"
  --out "$SMOKE/run" --seq-len 2048 --batch-size "$BATCH_SIZE" --grad-accum "$GRAD_ACCUM"
  --max-steps 20 --topk 32 --alpha 0.5 --temperature 2.0
  --optim adafactor --lr 3e-4 --schedule wsd
  --checkpoint-policy selective --eval-interval 0 --save-interval 0 --log-interval 1)
if [[ "$GPU_COUNT" == 1 ]]; then
  RUNNER=("$PY" -m blackcode.distill)
else
  RUNNER=(torchrun --standalone --nnodes=1 --nproc_per_node=2 -m blackcode.distill)
fi
run_logged() {
  "${RUNNER[@]}" "${TRAIN_ARGS[@]}" 2>&1 | tee "$LOG"
  return "${PIPESTATUS[0]}"
}
set +e
run_logged
rc=$?
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
echo "Benchmark GPU_COUNT=$GPU_COUNT terminé. Journal conservé : $LOG"
