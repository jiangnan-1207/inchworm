#!/bin/bash
set -euo pipefail
: "${REPO_ROOT:?Export REPO_ROOT from scripts/submit.sh}"
source "$REPO_ROOT/scripts/common.sh"
# One worker checks detector/CLIP CUDA compatibility while doing useful work.
# Other workers avoid repeatedly loading the independent evaluator environment.
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    bash scripts/run.sh preflight "$METHOD" "$RUN_TAG"
fi
bash scripts/run.sh generate "$METHOD" "$RUN_TAG" "$SLURM_ARRAY_TASK_ID" "$RUN_SHARDS"
