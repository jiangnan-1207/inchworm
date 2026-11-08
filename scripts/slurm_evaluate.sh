#!/bin/bash
set -euo pipefail
: "${REPO_ROOT:?Export REPO_ROOT from scripts/submit.sh}"
source "$REPO_ROOT/scripts/common.sh"
bash scripts/run.sh evaluate "$METHOD" "$RUN_TAG" "$RUN_SHARDS"
