#!/bin/bash
# Build the cold-expert ATS alias side file from your own copy of the pack.
#
#   bash scripts/build_cold_alias.sh
#
# Writes $ALIAS_DIR/experts.bin (9,228,943,360 bytes: embed.weight + 1,380
# coldest experts x w1/w2/w3, every tensor 64-byte aligned) and
# $ALIAS_DIR/manifest.json. Read-only on the pack; the side file costs disk, not
# GPU memory, and is mapped on demand.
#
# The measured member list is configs/cold-alias-selection.json (24 layers, 1,380
# experts). It is a routing-frequency selection, not calibration data: experts the
# router picks least often are the ones whose pages can be paged in from NVMe
# instead of held resident, which is what makes the rest of the pack fit.
#
# To derive your own membership from a route histogram instead:
#   VLLM_EXL3_ROUTE_STATS=1 ...          # serve once with stats on
#   python3 tools/ats_build_alias.py --model-dir "$MODEL_DIR" --out "$ALIAS_DIR" \
#       --select-npz route_stats.npz --budget-gib 8.59 --rows 7
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
. "$HERE/env.sh"

[ -d "$MODEL_DIR" ] || { echo "model dir not found: $MODEL_DIR (run download_and_verify.sh)" >&2; exit 1; }
[ -f "$MODEL_DIR/model.safetensors.index.json" ] || { echo "no index in $MODEL_DIR" >&2; exit 1; }

mkdir -p "$ALIAS_DIR"

python3 "$RECIPE_HOME/tools/ats_build_alias.py" \
  --model-dir "$MODEL_DIR" \
  --out "$ALIAS_DIR" \
  --experts-json "$RECIPE_HOME/configs/cold-alias-selection.json" \
  --embed \
  --align 64

echo
echo "alias dir: $ALIAS_DIR"
ls -la "$ALIAS_DIR"
echo "expected experts.bin size: 9228943360 bytes"
echo "point serve.sh at it with ALIAS_DIR=$ALIAS_DIR (the default)"