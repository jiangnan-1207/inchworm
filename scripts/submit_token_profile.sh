#!/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
MODE="${1:?Use --check-only or --submit}"
TOKEN_MODEL="${2:?Use janus1b or llamagen}"
RUN_TAG="${3:?Provide a new output tag}"
case "$MODE" in --check-only|--submit) ;; *) exit 2 ;; esac
case "$TOKEN_MODEL" in
    janus1b) test -x "$JANUS_PYTHON" ;;
    llamagen) test -x "$LLAMA_PYTHON" ;;
    *) echo 'Only janus1b and llamagen are supported.' >&2; exit 2 ;;
esac
[[ "$RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ ]]
: "${ASSETS_ROOT:?Set ASSETS_ROOT to the existing checkout containing benchmark sources}"
ASSETS_ROOT="$(cd -- "$ASSETS_ROOT" && pwd)"
test -f "$ASSETS_ROOT/benchmarks/geneval/prompts/evaluation_metadata.jsonl"
test -f "$ASSETS_ROOT/benchmarks/setup_records/model_revisions_v1.json"
if [ "$TOKEN_MODEL" = llamagen ]; then
    test -f "$ASSETS_ROOT/benchmarks/LlamaGen/autoregressive/models/gpt.py"
fi
export REPO_ROOT CONFIG_FILE RUN_TAG TOKEN_MODEL ASSETS_ROOT
resources=(--nodes=1 --ntasks=1 "--cpus-per-task=${CPUS_PER_TASK:-4}" "--mem=${MEMORY:-64G}"
    "--time=${TOKEN_WALLTIME:-04:00:00}" "--gres=${GPU_REQUEST:-gpu:1}")
if [ -n "${PARTITION:-}" ]; then resources+=("--partition=$PARTITION"); fi
if [ -n "${ACCOUNT:-}" ]; then resources+=("--account=$ACCOUNT"); fi
if [ -n "${QOS:-}" ]; then resources+=("--qos=$QOS"); fi
if [ "$MODE" = --check-only ]; then
    sbatch --test-only "${resources[@]}" --export=ALL scripts/token_profile_worker.sh
    echo 'CHECK_ONLY_OK: no job submitted'
    exit 0
fi
journal="$DATA_ROOT/logs/$RUN_TAG.$TOKEN_MODEL.token-submission.txt"
if [ -e "$journal" ]; then echo "Existing journal: $journal; inspect that job first." >&2; exit 1; fi
# noclobber prevents concurrent launchers from submitting the same run twice.
(set -o noclobber; printf 'RUN=%s MODEL=%s\n' "$RUN_TAG" "$TOKEN_MODEL" > "$journal")
job=$(sbatch --parsable "${resources[@]}" --export=ALL --job-name="token_$TOKEN_MODEL" \
    --output="$DATA_ROOT/logs/${RUN_TAG}_${TOKEN_MODEL}_%j.out" \
    --error="$DATA_ROOT/logs/${RUN_TAG}_${TOKEN_MODEL}_%j.err" scripts/token_profile_worker.sh)
printf '%s token_profile=%s\n' "$TOKEN_MODEL" "$job" | tee -a "$journal"
