#!/bin/bash
# Run on an ordinary GPU server, or within an existing GPU allocation.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
ACTION="${1:?Usage: bash scripts/run.sh generate|evaluate|preflight METHOD RUN_TAG [SHARD_INDEX SHARDS]}"
METHOD="${2:?Specify janus_baseline, janus_inchworm, or llamagen_baseline}"
RUN_TAG="${3:?Specify a unique RUN_TAG}"
[[ "$RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ ]]
case "$METHOD" in janus_baseline|janus_inchworm|llamagen_baseline) ;; *) exit 2 ;; esac
RUN="$DATA_ROOT/outputs/$RUN_TAG"
case "$ACTION" in
generate)
    INDEX="${4:-0}"
    COUNT="${5:-1}"
    PYTHON="$JANUS_PYTHON"
    if [ "$METHOD" = llamagen_baseline ]; then PYTHON="$LLAMA_PYTHON"; fi
    "$PYTHON" -u generate_full_shard.py --root "$REPO_ROOT" --scratch "$DATA_ROOT" \
        --out "$RUN/shards/${METHOD}_${INDEX}" --scope full --method "$METHOD" \
        --shard-index "$INDEX" --shards "$COUNT"
    ;;
evaluate)
    COUNT="${4:-1}"
    "$GENEVAL_PYTHON" -u merge_full.py --root "$REPO_ROOT" --run "$RUN" --method "$METHOD" --shards "$COUNT"
    "$GENEVAL_PYTHON" -u evaluate_one_full.py "$METHOD" --root "$REPO_ROOT" --scratch "$DATA_ROOT" \
        --out "$RUN/merged" --scope full --stage evaluate
    ;;
preflight)
    "$GENEVAL_PYTHON" -u evaluate_benchmark.py --root "$REPO_ROOT" --scratch "$DATA_ROOT" \
        --out "$RUN" --scope full --stage preflight
    ;;
*) echo 'Unknown action' >&2; exit 2 ;;
esac
