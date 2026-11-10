#!/bin/bash
set -euo pipefail
: "${REPO_ROOT:?Export the new audit checkout path}"
: "${LLAMA_SOURCE_ROOT:?Point to the existing pinned official LlamaGen checkout}"
source "$REPO_ROOT/scripts/common.sh"
# CPU-only, offline, no model deserialization, installs or weight overwrites.
export CUDA_VISIBLE_DEVICES=""
"$LLAMA_PYTHON" -u tools/audit_llamagen_assets.py --source-root "$LLAMA_SOURCE_ROOT" \
    --data-root "$DATA_ROOT" --out "$DATA_ROOT/outputs/llamagen_asset_audit_${SLURM_JOB_ID:?}.json"
