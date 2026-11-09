#!/bin/bash
set -euo pipefail
: "${REPO_ROOT:?Set by submit_four_tasks.sh}"
source "$REPO_ROOT/scripts/common.sh"
case "$FOUR_TASK_PHASE" in
generate)
    bash scripts/four_tasks.sh task4-generate "$EXPERIMENT_MODEL" "$RUN_TAG" "$SLURM_ARRAY_TASK_ID" "$FOUR_TASK_SHARDS" "$FOUR_TASK_VARIANT"
    ;;
score)
    bash scripts/four_tasks.sh task4-score "$EXPERIMENT_MODEL" "$RUN_TAG" "$FOUR_TASK_SHARDS" "$FOUR_TASK_VARIANT"
    ;;
direct)
    bash scripts/four_tasks.sh "$FOUR_TASK" "$EXPERIMENT_MODEL" "$RUN_TAG"
    ;;
*) exit 2 ;;
esac
