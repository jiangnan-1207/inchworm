#!/bin/bash
# Read the existing weights, write all new outputs to a separate shared run.
set -euo pipefail
: "${REPO_ROOT:?Set by the recovery launcher}"
: "${ASSETS_ROOT:?Set by the recovery launcher}"
: "${ABLATION_RUN_ROOT:?Set by the recovery launcher}"
source "$REPO_ROOT/scripts/common.sh"
case "$RECOVERY_VARIANT" in guard_temperature|guard_gap) ;; *) exit 2 ;; esac
slot="${RECOVERY_VARIANT}_${RECOVERY_PHASE}_${SLURM_ARRAY_TASK_ID:-score}"
mkdir -p "$ABLATION_RUN_ROOT/tmp/$slot" "$ABLATION_RUN_ROOT/cache/$slot"
export TMPDIR="$ABLATION_RUN_ROOT/tmp/$slot" TMP="$ABLATION_RUN_ROOT/tmp/$slot" TEMP="$ABLATION_RUN_ROOT/tmp/$slot"
export XDG_CACHE_HOME="$ABLATION_RUN_ROOT/cache/$slot" HF_HOME="$ABLATION_RUN_ROOT/cache/$slot/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$XDG_CACHE_HOME/torch" CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/inductor" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
case "$RECOVERY_PHASE" in
generate)
    "$JANUS_PYTHON" -u generate_variant_full.py --root "$ASSETS_ROOT" --data-root "$DATA_ROOT" \
        --run "$ABLATION_RUN_ROOT" --model janus1b --variant "$RECOVERY_VARIANT" \
        --shards "$RECOVERY_SHARDS" --shard-index "$SLURM_ARRAY_TASK_ID"
    ;;
score)
    "$GENEVAL_PYTHON" -u evaluate_variant_full.py --root "$ASSETS_ROOT" --data-root "$DATA_ROOT" \
        --run "$ABLATION_RUN_ROOT" --model janus1b --variant "$RECOVERY_VARIANT" \
        --shards "$RECOVERY_SHARDS" --evaluation-root "$ABLATION_RUN_ROOT/merged"
    ;;
*) exit 2 ;;
esac
