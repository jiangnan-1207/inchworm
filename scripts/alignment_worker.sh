#!/bin/bash
set -euo pipefail
source "$REPO_ROOT/scripts/common.sh"
case "${ALIGNMENT_STAGE:?}" in
prepare)
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  "$JANUS_PYTHON" -u tools/prepare_alignment.py --data-root "$DATA_ROOT"
  "$JANUS_PYTHON" -u tools/setup_alignment_judges.py --data-root "$DATA_ROOT"
  ;;
score)
  "$DATA_ROOT/envs/alignment-$ALIGNMENT_BENCHMARK-v1/bin/python" -u score_alignment.py \
    --root "$REPO_ROOT" --data-root "$DATA_ROOT" --run "$DATA_ROOT/outputs/$RUN_TAG" \
    --model "$ALIGNMENT_MODEL" --benchmark "$ALIGNMENT_BENCHMARK"
  ;;
generate)
  PYTHON="$JANUS_PYTHON"
  if [ "$ALIGNMENT_MODEL" = llamagen ]; then PYTHON="$LLAMA_PYTHON"; fi
  "$PYTHON" -u run_alignment_generation.py --root "$REPO_ROOT" --data-root "$DATA_ROOT" \
    --run "$DATA_ROOT/outputs/$RUN_TAG" --model "$ALIGNMENT_MODEL" --shards "$ALIGNMENT_SHARDS" --index "$SLURM_ARRAY_TASK_ID"
  ;;
*) echo 'Unknown alignment stage' >&2; exit 2 ;;
esac
