#!/bin/bash
set -euo pipefail
: "${REPO_ROOT:?Export the new checkout path}"
: "${EXISTING_ROOT:?Export the existing experiment root}"
source "$REPO_ROOT/scripts/common.sh"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export CUDA_VISIBLE_DEVICES=""
"$JANUS_PYTHON" -u "$REPO_ROOT/tools/prepare_requested_assets.py" \
    --existing-root "$EXISTING_ROOT" --data-root "$DATA_ROOT"
