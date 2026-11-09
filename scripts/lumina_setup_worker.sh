#!/bin/bash
set -euo pipefail
: "${REPO_ROOT:?Export the new Lumina checkout path}"
source "$REPO_ROOT/scripts/common.sh"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export CUDA_VISIBLE_DEVICES=""
SETUP_PYTHON="${LUMINA_SETUP_PYTHON:-${JANUS_PYTHON:-python3.10}}"
"$SETUP_PYTHON" -u tools/setup_lumina.py --data-root "$DATA_ROOT"
