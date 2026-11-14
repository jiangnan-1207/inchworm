#!/bin/bash
# One allocated GPU: complete pure Janus-Pro-1B generation, then full GenEval.
set -euo pipefail
: "${REPO_ROOT:?Export this code checkout path}"
: "${ASSETS_ROOT:?Point to the existing benchmark source checkout}"
: "${BASELINE_RUN_ROOT:?Choose a new writable shared run directory}"
source "$REPO_ROOT/scripts/common.sh"
mkdir -p "$BASELINE_RUN_ROOT/tmp" "$BASELINE_RUN_ROOT/cache"
export TMPDIR="$BASELINE_RUN_ROOT/tmp" TMP="$BASELINE_RUN_ROOT/tmp" TEMP="$BASELINE_RUN_ROOT/tmp"
export XDG_CACHE_HOME="$BASELINE_RUN_ROOT/cache" HF_HOME="$BASELINE_RUN_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$BASELINE_RUN_ROOT/cache/torch" CUDA_CACHE_PATH="$BASELINE_RUN_ROOT/cache/cuda"
export TORCHINDUCTOR_CACHE_DIR="$BASELINE_RUN_ROOT/cache/inductor" TRITON_CACHE_DIR="$BASELINE_RUN_ROOT/cache/triton"
printf 'JANUS1B_NATIVE_FULL_RUN=%s\n' "$BASELINE_RUN_ROOT"
"$JANUS_PYTHON" -u generate_variant_full.py --root "$ASSETS_ROOT" --data-root "$DATA_ROOT" \
    --run "$BASELINE_RUN_ROOT" --model janus1b --variant backbone --shards 1 --shard-index 0
# Separate processes release the generator's model before the evaluator loads.
"$GENEVAL_PYTHON" -u evaluate_variant_full.py --root "$ASSETS_ROOT" --data-root "$DATA_ROOT" \
    --run "$BASELINE_RUN_ROOT" --model janus1b --variant backbone --shards 1 \
    --evaluation-root "$BASELINE_RUN_ROOT/merged"
printf 'JANUS1B_NATIVE_FULL_GENEVAL_OK\n'
