#!/bin/bash
set -euo pipefail
: "${REPO_ROOT:?Export this checkout path}"
: "${LLAMA_SOURCE_ROOT:?Point to the pinned official source}"
: "${LLAMA_ASSET_REPORT:?Point to the passed asset audit report}"
source "$REPO_ROOT/scripts/common.sh"
reference="${LLAMA_REFERENCE_IMAGE:-$DATA_ROOT/outputs/janus_test/apple_seed42.png}"
test -f "$reference"
# A scratch inode quota can prevent even a tiny report from being created.
# Keep DATA_ROOT as the read-only asset location, but allow all diagnostic
# writes (including lazy library/kernel caches) on another shared filesystem.
if [ -n "${LLAMA_DIAGNOSTIC_ROOT:-}" ]; then
    write_root="$LLAMA_DIAGNOSTIC_ROOT"
    mkdir -p "$write_root/tmp" "$write_root/cache"
    export TMPDIR="$write_root/tmp" TMP="$write_root/tmp" TEMP="$write_root/tmp"
    export XDG_CACHE_HOME="$write_root/cache" HF_HOME="$write_root/cache/huggingface"
    export HF_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/transformers"
    export TORCH_HOME="$write_root/cache/torch" CUDA_CACHE_PATH="$write_root/cache/cuda"
    export TORCHINDUCTOR_CACHE_DIR="$write_root/cache/inductor" TRITON_CACHE_DIR="$write_root/cache/triton"
else
    write_root="$DATA_ROOT/outputs"
fi
printf 'DIAGNOSTIC_WRITE_ROOT=%s\nASSET_DATA_ROOT=%s\n' "$write_root" "$DATA_ROOT"
extra=()
if [ "${LLAMA_PARITY_ONLY:-0}" = 1 ]; then extra+=(--parity-only); fi
"$LLAMA_PYTHON" -u tools/check_llamagen_structure.py --source-root "$LLAMA_SOURCE_ROOT" \
    --data-root "$DATA_ROOT" --asset-report "$LLAMA_ASSET_REPORT" --reference-image "$reference" \
    --out "$write_root/llamagen_structure_${SLURM_JOB_ID:?}" "${extra[@]}"
