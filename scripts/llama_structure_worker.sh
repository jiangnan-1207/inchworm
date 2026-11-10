#!/bin/bash
set -euo pipefail
: "${REPO_ROOT:?Export this checkout path}"
: "${LLAMA_SOURCE_ROOT:?Point to the pinned official source}"
: "${LLAMA_ASSET_REPORT:?Point to the passed asset audit report}"
source "$REPO_ROOT/scripts/common.sh"
reference="${LLAMA_REFERENCE_IMAGE:-$DATA_ROOT/outputs/janus_test/apple_seed42.png}"
test -f "$reference"
"$LLAMA_PYTHON" -u tools/check_llamagen_structure.py --source-root "$LLAMA_SOURCE_ROOT" \
    --data-root "$DATA_ROOT" --asset-report "$LLAMA_ASSET_REPORT" --reference-image "$reference" \
    --out "$DATA_ROOT/outputs/llamagen_structure_${SLURM_JOB_ID:?}"
