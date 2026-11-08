#!/bin/bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$REPO_ROOT/configs/local.sh}"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Copy configs/example.sh to configs/local.sh and configure DATA_ROOT first." >&2
    exit 1
fi
source "$CONFIG_FILE"
: "${DATA_ROOT:?Configure DATA_ROOT}"
if [ "$DATA_ROOT" = /absolute/path/to/inchworm-data ]; then
    echo 'Edit the DATA_ROOT placeholder in configs/local.sh first.' >&2
    exit 1
fi
mkdir -p "$DATA_ROOT/tmp" "$DATA_ROOT/logs"
export HF_HOME="$DATA_ROOT/hf_cache" TORCH_HOME="$DATA_ROOT/torch_cache"
export XDG_CACHE_HOME="$DATA_ROOT/.cache" TMPDIR="$DATA_ROOT/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${CPUS_PER_TASK:-4}}"
export TOKENIZERS_PARALLELISM=false
unset PYTHONPATH PYTHONHOME
cd "$REPO_ROOT"
