#!/bin/bash
# Score completed shards without writing images, links or reports to scratch.
set -euo pipefail
: "${REPO_ROOT:?Export this code checkout path}"
: "${ASSETS_ROOT:?Point to an existing checkout containing benchmark sources}"
: "${EXISTING_RUN:?Point to the completed shard run}"
: "${RESCORE_ROOT:?Choose a writable shared output directory}"
source "$REPO_ROOT/scripts/common.sh"
model="${RESCORE_MODEL:-janus1b}"
version="${RESCORE_VARIANT:-guard_only}"
shards="${RESCORE_SHARDS:-32}"
mkdir -p "$RESCORE_ROOT/tmp" "$RESCORE_ROOT/cache"
export TMPDIR="$RESCORE_ROOT/tmp" TMP="$RESCORE_ROOT/tmp" TEMP="$RESCORE_ROOT/tmp"
export XDG_CACHE_HOME="$RESCORE_ROOT/cache" HF_HOME="$RESCORE_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$RESCORE_ROOT/cache/torch" CUDA_CACHE_PATH="$RESCORE_ROOT/cache/cuda"
export TORCHINDUCTOR_CACHE_DIR="$RESCORE_ROOT/cache/inductor" TRITON_CACHE_DIR="$RESCORE_ROOT/cache/triton"
printf 'READ_SHARDS=%s\nWRITE_EVALUATION=%s\n' "$EXISTING_RUN" "$RESCORE_ROOT/merged"
"$GENEVAL_PYTHON" -u "$REPO_ROOT/evaluate_variant_full.py" --root "$ASSETS_ROOT" \
    --data-root "$DATA_ROOT" --run "$EXISTING_RUN" --model "$model" \
    --variant "$version" --shards "$shards" --evaluation-root "$RESCORE_ROOT/merged"
