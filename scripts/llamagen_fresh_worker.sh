#!/bin/bash
set -euo pipefail
: "${REPO_ROOT:?Set by launcher}"
: "${FRESH_LLAMA_ROOT:?Set by launcher}"
: "${ASSETS_ROOT:?Set by launcher}"
source "$REPO_ROOT/scripts/common.sh"
mkdir -p "$FRESH_LLAMA_ROOT/cache"
job_tmp=$(mktemp -d /tmp/lf.XXXXXXXX)
trap 'rm -rf -- "$job_tmp"' EXIT
export TMPDIR="$job_tmp" TMP="$job_tmp" TEMP="$job_tmp"
export HF_HOME="$FRESH_LLAMA_ROOT/cache/huggingface" XDG_CACHE_HOME="$FRESH_LLAMA_ROOT/cache"
export TORCH_HOME="$XDG_CACHE_HOME/torch" TRANSFORMERS_CACHE="$HF_HOME/transformers"
export CUDA_CACHE_PATH="$job_tmp/cuda" TORCHINDUCTOR_CACHE_DIR="$job_tmp/inductor" TRITON_CACHE_DIR="$job_tmp/triton"
case "$FRESH_PHASE" in
setup)
    "$JANUS_PYTHON" -u tools/setup_llamagen_fresh.py --fresh-root "$FRESH_LLAMA_ROOT" --data-root "$DATA_ROOT"
    ;;
demo|generate|score)
    python="$FRESH_LLAMA_ROOT/env/bin/python"
    if [ "$FRESH_PHASE" = score ]; then python="$GENEVAL_PYTHON"; fi
    "$python" -u tools/run_llamagen_fresh.py --phase "$FRESH_PHASE" --fresh-root "$FRESH_LLAMA_ROOT" \
        --assets-root "$ASSETS_ROOT" --data-root "$DATA_ROOT" --stage "${FRESH_STAGE:-512}" \
        --shards "${FRESH_SHARDS:-32}" --index "${SLURM_ARRAY_TASK_ID:-0}"
    ;;
*) exit 2 ;;
esac
