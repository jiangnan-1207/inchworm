#!/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
MODE="${1:?Use --check-only or --submit}"
RUN_TAG="${2:?Provide a new run tag}"
VARIANT_SCOPE="${3:-pairs}"
[[ "$RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ ]]
case "$MODE" in --check-only|--submit) ;; *) exit 2 ;; esac
case "$VARIANT_SCOPE" in pairs|profile|both) ;; *) exit 2 ;; esac
MODELS=(janus1b llamagen janus7b)
if [ "$#" -gt 3 ]; then MODELS=("${@:4}"); fi
declare -A seen=()
for model in "${MODELS[@]}"; do
    case "$model" in janus1b|llamagen|janus7b) ;; *) exit 2 ;; esac
    if [ -n "${seen[$model]:-}" ]; then echo 'Duplicate model' >&2; exit 2; fi
    seen[$model]=1
done
export REPO_ROOT CONFIG_FILE RUN_TAG VARIANT_SCOPE
resources=("--gres=$GPU_REQUEST" "--cpus-per-task=$CPUS_PER_TASK" "--mem=$MEMORY" "--time=$WALLTIME" --nodes=1 --ntasks=1)
if [ -n "${ACCOUNT:-}" ]; then resources+=("--account=$ACCOUNT"); fi
if [ -n "${PARTITION:-}" ]; then resources+=("--partition=$PARTITION"); fi
if [ -n "${QOS:-}" ]; then resources+=("--qos=$QOS"); fi
journal="$DATA_ROOT/logs/$RUN_TAG.variant-submissions.txt"
if [ "$MODE" = --submit ]; then
    if [ -e "$journal" ]; then echo "Existing journal: $journal. Inspect existing jobs." >&2; exit 1; fi
    printf 'Run %s scope %s\n' "$RUN_TAG" "$VARIANT_SCOPE" > "$journal"
fi
for EXPERIMENT_MODEL in "${MODELS[@]}"; do
    export EXPERIMENT_MODEL
    if [ "$MODE" = --check-only ]; then
        sbatch --test-only "${resources[@]}" --export=ALL scripts/variant_worker.sh
    else
        id=$(sbatch --parsable "${resources[@]}" --export=ALL --job-name="variant_$EXPERIMENT_MODEL" \
            --output="$DATA_ROOT/logs/${RUN_TAG}_${EXPERIMENT_MODEL}_%j.out" \
            --error="$DATA_ROOT/logs/${RUN_TAG}_${EXPERIMENT_MODEL}_%j.err" scripts/variant_worker.sh)
        printf '%s %s\n' "$EXPERIMENT_MODEL" "$id" | tee -a "$journal"
    fi
done
