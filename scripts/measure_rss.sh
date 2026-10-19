#!/usr/bin/env bash
# Mesure la RAM réelle du runtime C pour un modèle donné, à plusieurs contextes.
# Usage : ./scripts/measure_rss.sh [config] [contexte max]
# 128k  : ./scripts/measure_rss.sh configs/blackcode_1.2b_128k.json 131072
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
CFG="${1:-configs/blackcode_1b.json}"
MAXCTX="${2:-8192}"

make -C runtime >/dev/null
PKG="${TMPDIR:-/tmp}/rss_$(basename "$CFG" .json).bcpkg"
"$PY" scripts/make_random_bcpkg.py --config "$CFG" --out "$PKG" --context "$MAXCTX" | tail -3

echo
echo "| contexte | KV | RSS pic |"
echo "|---:|---:|---:|"
for ctx in 512 1024 2048 4096 8192 16384 32768 65536 131072; do
  [ "$ctx" -gt "$MAXCTX" ] && continue
  out=$(OMP_NUM_THREADS=2 ./runtime/bcrun --pkg "$PKG" --ids 1 --max-new 0 \
        --context "$ctx" --temp 0 --mem-report)
  kv=$(echo "$out" | awk -F': *' '/détail cache KV/{print $2}' | awk '{print $1}')
  peak=$(echo "$out" | awk -F': *' '/pic pendant/{print $2}' | awk '{print $1}')
  echo "| $ctx | ${kv} Mo | ${peak} Mo |"
done
echo
echo "paquet : $PKG ($(du -h "$PKG" | cut -f1))"
