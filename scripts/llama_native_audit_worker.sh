#!/bin/bash
set -euo pipefail
: "${REPO_ROOT:?Export the new diagnostic checkout path}"
: "${LLAMA_SOURCE_ROOT:?Point to the existing pinned official LlamaGen checkout}"
source "$REPO_ROOT/scripts/common.sh"
"$LLAMA_PYTHON" -u tools/check_llamagen_native.py --source-root "$LLAMA_SOURCE_ROOT" \
  --data-root "$DATA_ROOT" --out "$DATA_ROOT/outputs/llamagen_native_audit_${SLURM_JOB_ID:?}"
