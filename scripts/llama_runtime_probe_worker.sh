#!/bin/bash
set -euo pipefail
: "${REPO_ROOT:?Export this diagnostic checkout path}"
: "${LLAMA_SOURCE_ROOT:?Point to the pinned official LlamaGen source}"
: "${LLAMA_ASSET_REPORT:?Point to the successful asset audit JSON}"
source "$REPO_ROOT/scripts/common.sh"
"$LLAMA_PYTHON" -u tools/diagnose_llamagen_runtime.py --source-root "$LLAMA_SOURCE_ROOT" \
    --data-root "$DATA_ROOT" --asset-report "$LLAMA_ASSET_REPORT" \
    --out "$DATA_ROOT/outputs/llamagen_runtime_${SLURM_JOB_ID:?}"
