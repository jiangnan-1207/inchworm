#!/bin/bash
set -euo pipefail
: "${REPO_ROOT:?Set by submission script}"
source "$REPO_ROOT/scripts/common.sh"
if [ "$EXPERIMENT_MODEL" = llamagen ]; then PYTHON="$LLAMA_PYTHON"; else PYTHON="$JANUS_PYTHON"; fi
"$PYTHON" -m unittest discover -s tests -p 'test_variant_*.py' -v
"$PYTHON" -u run_variant_experiments.py --root "$REPO_ROOT" --data-root "$DATA_ROOT" \
    --out "$DATA_ROOT/outputs/$RUN_TAG" --model "$EXPERIMENT_MODEL" --scope "$VARIANT_SCOPE"
