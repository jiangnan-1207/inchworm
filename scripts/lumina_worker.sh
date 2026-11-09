#!/bin/bash
set -euo pipefail
: "${REPO_ROOT:?Set by submit_lumina.sh}"
source "$REPO_ROOT/scripts/common.sh"
PYTHON="$DATA_ROOT/envs/lumina-mgpt2/bin/python"
"$PYTHON" -m unittest discover -s tests -p 'test_lumina*.py' -v
"$PYTHON" -u run_lumina_pairs.py --root "$REPO_ROOT" --data-root "$DATA_ROOT" \
    --out "$DATA_ROOT/outputs/$RUN_TAG" --shard-index "$SLURM_ARRAY_TASK_ID" --shards "$LUMINA_SHARDS"
