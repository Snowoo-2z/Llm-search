#!/usr/bin/env bash
# Smoke test PyTorch DDP sur les deux T4 Kaggle.
# À lancer depuis la racine du dépôt, après activation d'un notebook Kaggle GPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PYTHON:-python}"
SMOKE="${SMOKE:-/tmp/blackcode_ddp_smoke}"

"$PY" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise SystemExit(f"Il faut 2 GPU visibles; trouvé {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i),
          f"{torch.cuda.get_device_properties(i).total_memory/1e9:.1f} Go")
PY

rm -rf "$SMOKE"
mkdir -p "$SMOKE/data" "$SMOKE/run" "$SMOKE/backup"
"$PY" - "$SMOKE" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
import torch
from blackcode.config import ModelConfig
from blackcode.model import build_model

root = Path(sys.argv[1])
cfg = ModelConfig(name="ddp_smoke", vocab_size=256, max_seq_len=64,
                  d_model=64, n_layers=2, n_heads=2, n_kv_heads=1,
                  head_dim=32, ffn_hidden=128, conv_kernel=4,
                  attn_every=2, tie_embeddings=True, ternary=True)
(root / "config.json").write_text(json.dumps(cfg.to_dict()))
torch.manual_seed(7)
teacher = build_model(cfg, "cpu")
torch.save({"model": teacher.state_dict(), "config": cfg.to_dict()},
           root / "teacher.pt")
rng = np.random.default_rng(7)
rng.integers(0, 256, size=8192, dtype=np.uint16).tofile(root / "data/train.bin")
rng.integers(0, 256, size=512, dtype=np.uint16).tofile(root / "data/val.bin")
(root / "data/meta.json").write_text(json.dumps({
    "vocab_size": 256, "train_tokens": 8192, "val_tokens": 512
}))
PY

echo "== lancement DDP à deux GPU, 3 pas =="
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m blackcode.distill \
  --teacher "local:$SMOKE/teacher.pt" --config "$SMOKE/config.json" \
  --data "$SMOKE/data" --out "$SMOKE/run" --mirror-dir "$SMOKE/backup" \
  --seq-len 32 --batch-size 1 --grad-accum 1 --max-steps 3 \
  --optim adafactor --lr 1e-3 --topk 8 --alpha 0.5 \
  --checkpoint-policy none --eval-interval 0 --save-interval 0 --log-interval 1

"$PY" - "$SMOKE" <<'PY'
import sys
from pathlib import Path
import torch
root = Path(sys.argv[1])
for path in (root / "run/ckpt.pt", root / "backup/ckpt.pt"):
    if not path.exists():
        raise SystemExit(f"ÉCHEC : checkpoint manquant : {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    print(f"OK : {path}, step={state['step']}, world_size={state.get('world_size')}")
    assert state["step"] == 2 and state.get("world_size") == 2
print("SUCCÈS : DDP + sauvegarde vérifiés sur les 2 GPU")
PY
