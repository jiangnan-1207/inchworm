#!/bin/bash
set -euo pipefail

if [ "${1:-}" = --worker ]; then
    source "${REPO_ROOT:?Set by launcher}/scripts/common.sh"
    job_tmp=$(mktemp -d /tmp/iw-retry.XXXXXXXX)
    trap 'rm -rf -- "$job_tmp"' EXIT
    export TMPDIR="$job_tmp" TEMP="$job_tmp" TMP="$job_tmp"
    export XDG_CACHE_HOME="$job_tmp" HF_HOME="$job_tmp/hf" TORCH_HOME="$job_tmp/torch"
    export HF_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/transformers"
    export CUDA_CACHE_PATH="$job_tmp/cuda"
    "$JANUS_PYTHON" -u run_window_retry.py --data-root "$DATA_ROOT" \
        --out "${WINDOW_RETRY_ROOT:?Set by launcher}/job_${SLURM_JOB_ID:?Use sbatch}" \
        --prompt "${WINDOW_RETRY_PROMPT:?Set by launcher}" --seed "${WINDOW_RETRY_SEED:?Set by launcher}"
    exit
fi

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: WINDOW_RETRY_ROOT=/path/to/results bash scripts/submit_window_retry.sh --submit 'prompt' [seed]" >&2
    exit 2
fi
mode="$1"
case "$mode" in --check-only|--submit) ;; *) echo "Use --check-only or --submit" >&2; exit 2 ;; esac
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
: "${WINDOW_RETRY_ROOT:?Set an absolute writable result path}"
WINDOW_RETRY_PROMPT="$2"
WINDOW_RETRY_SEED="${3:-42}"
test -x "$JANUS_PYTHON"
test -f "$DATA_ROOT/models/Janus-Pro-1B/config.json"
"$JANUS_PYTHON" -c 'import sys; assert sys.argv[1].strip() and 0 <= int(sys.argv[2]) < 2**63, "Invalid prompt or seed"' \
    "$WINDOW_RETRY_PROMPT" "$WINDOW_RETRY_SEED"
mkdir -p "$WINDOW_RETRY_ROOT/logs"
WINDOW_RETRY_ROOT="$(cd -- "$WINDOW_RETRY_ROOT" && pwd)"
"$JANUS_PYTHON" -c 'import sys,tempfile; tempfile.TemporaryFile(dir=sys.argv[1]).close()' "$WINDOW_RETRY_ROOT/logs"
export REPO_ROOT CONFIG_FILE WINDOW_RETRY_ROOT WINDOW_RETRY_PROMPT WINDOW_RETRY_SEED
resources=(--nodes=1 --ntasks=1 "--cpus-per-task=${CPUS_PER_TASK:-4}" "--mem=${MEMORY:-64G}"
    "--time=${WINDOW_RETRY_WALLTIME:-00:30:00}" "--gres=${GPU_REQUEST:-gpu:1}")
if [ -n "${ACCOUNT:-}" ]; then resources+=("--account=$ACCOUNT"); fi
if [ -n "${QOS:-}" ]; then resources+=("--qos=$QOS"); fi
if [ -n "${PARTITION:-}" ]; then resources+=("--partition=$PARTITION"); fi
if [ "$mode" = --check-only ]; then
    sbatch --test-only "${resources[@]}" --export=ALL scripts/submit_window_retry.sh --worker
    echo 'CHECK_ONLY_OK: no job submitted'
    exit 0
fi
job=$(sbatch --parsable "${resources[@]}" --export=ALL --job-name=janus_window_retry \
    --chdir="$REPO_ROOT" --output="$WINDOW_RETRY_ROOT/logs/retry_%j.out" \
    --error="$WINDOW_RETRY_ROOT/logs/retry_%j.err" scripts/submit_window_retry.sh --worker)
printf 'job=%s\nresults=%s/job_%s\n' "$job" "$WINDOW_RETRY_ROOT" "${job%%;*}"
