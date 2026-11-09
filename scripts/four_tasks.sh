#!/bin/bash
# Ordinary GPU machine or an existing Slurm GPU allocation.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
TASK="${1:?Use task1/task2/task3/task4/task4-generate/task4-score}"
if [ "$TASK" = task2 ]; then
    RUN_TAG="${2:?Use the same RUN_TAG as the finished LlamaGen task1}"
    [[ "$RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ ]]
    python3 tools/report_four_tasks.py --run "$DATA_ROOT/outputs/$RUN_TAG"
    exit 0
fi
MODEL="${2:?Select janus1b/llamagen/janus7b}"
RUN_TAG="${3:?Provide a new RUN_TAG}"
[[ "$RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ ]]
case "$MODEL" in janus1b|llamagen|janus7b) ;; *) exit 2 ;; esac
PYTHON="$JANUS_PYTHON"
if [ "$MODEL" = llamagen ]; then PYTHON="$LLAMA_PYTHON"; fi
case "$TASK" in
task1)
    if [ "$MODEL" = janus7b ]; then echo 'Task1 profiles 1B and LlamaGen only' >&2; exit 2; fi
    "$PYTHON" -u run_requested_experiments.py --root "$REPO_ROOT" --data-root "$DATA_ROOT" \
        --out "$DATA_ROOT/outputs/$RUN_TAG" --model "$MODEL" --scope profile
    ;;
task3)
    "$PYTHON" -u run_requested_experiments.py --root "$REPO_ROOT" --data-root "$DATA_ROOT" \
        --out "$DATA_ROOT/outputs/$RUN_TAG" --model "$MODEL" --scope pairs
    ;;
task4|task4-generate|task4-score)
    if [ "$MODEL" = janus7b ]; then echo 'Task4 scores 1B and LlamaGen only' >&2; exit 2; fi
    if [ "$TASK" != task4-score ]; then
        "$PYTHON" -m unittest discover -s tests -p 'test_variant_*.py' -v
        "$PYTHON" -u generate_variant_full.py --root "$REPO_ROOT" --data-root "$DATA_ROOT" \
            --run "$DATA_ROOT/outputs/$RUN_TAG" --model "$MODEL" --shard-index "${4:-0}" --shards "${5:-1}"
    fi
    if [ "$TASK" != task4-generate ]; then
        if [ "$TASK" = task4 ]; then SHARD_COUNT=1; else SHARD_COUNT="${4:-1}"; fi
        "$GENEVAL_PYTHON" -u evaluate_variant_full.py --root "$REPO_ROOT" --data-root "$DATA_ROOT" \
            --run "$DATA_ROOT/outputs/$RUN_TAG" --model "$MODEL" --shards "$SHARD_COUNT"
    fi
    ;;
*) echo 'Unknown task' >&2; exit 2 ;;
esac
