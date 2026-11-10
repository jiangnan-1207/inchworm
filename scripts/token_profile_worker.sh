#!/bin/bash
set -euo pipefail
: "${REPO_ROOT:?Set by submit_token_profile.sh}"
source "$REPO_ROOT/scripts/common.sh"
case "$TOKEN_MODEL" in
    janus1b) python_bin="$JANUS_PYTHON" ;;
    llamagen) python_bin="$LLAMA_PYTHON" ;;
    *) exit 2 ;;
esac
"$python_bin" -u run_token_profile.py --model "$TOKEN_MODEL" \
    --assets-root "$ASSETS_ROOT" --data-root "$DATA_ROOT" --out "$DATA_ROOT/outputs/$RUN_TAG"
