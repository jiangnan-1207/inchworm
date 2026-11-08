#!/bin/bash
# Submit only on a Slurm login node. It runs no neural model on that node.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
MODE="${1:---check-only}"
RUN_TAG="${2:?Usage: bash scripts/submit.sh --check-only|--submit RUN_TAG [METHOD ...]}"
shift 2
[[ "$RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ ]]
case "$MODE" in --check-only|--submit) ;; *) exit 2 ;; esac
METHODS=("$@")
if [ "${#METHODS[@]}" -eq 0 ]; then METHODS=(janus_baseline llamagen_baseline janus_inchworm); fi
for name in "${METHODS[@]}"; do
    case "$name" in janus_baseline|janus_inchworm|llamagen_baseline) ;; *) exit 2 ;; esac
done
[[ "$SHARDS" =~ ^[0-9]+$ && "$CONCURRENT" =~ ^[0-9]+$ ]]
(( SHARDS>=1 && SHARDS<=128 && CONCURRENT>=1 && CONCURRENT<=6 && CONCURRENT<=SHARDS ))
resource_args=("--gres=$GPU_REQUEST" "--cpus-per-task=$CPUS_PER_TASK" "--mem=$MEMORY" "--time=$WALLTIME" --nodes=1 --ntasks=1)
if [ -n "${ACCOUNT:-}" ]; then resource_args+=("--account=$ACCOUNT"); fi
if [ -n "${PARTITION:-}" ]; then resource_args+=("--partition=$PARTITION"); fi
if [ -n "${QOS:-}" ]; then resource_args+=("--qos=$QOS"); fi
export REPO_ROOT CONFIG_FILE RUN_TAG
RUN_SHARDS="$SHARDS"
export RUN_SHARDS
if [ "$MODE" = --check-only ]; then
    METHOD="${METHODS[0]}"; export METHOD
    sbatch --test-only "${resource_args[@]}" --array="0-$((SHARDS-1))%$CONCURRENT" --export=ALL scripts/slurm_worker.sh
    sbatch --test-only "${resource_args[@]}" --export=ALL scripts/slurm_evaluate.sh
    echo 'CHECK_ONLY_OK: no actual jobs submitted'
    exit 0
fi
JOURNAL="$DATA_ROOT/logs/${RUN_TAG}.submissions.txt"
if [ -e "$JOURNAL" ]; then
    echo "Existing submission journal: $JOURNAL. Inspect existing jobs instead of duplicating." >&2
    exit 1
fi
printf 'RUN_TAG=%s SHARDS=%s CONCURRENT=%s\n' "$RUN_TAG" "$SHARDS" "$CONCURRENT" > "$JOURNAL"
previous_generation=''
for METHOD in "${METHODS[@]}"; do
    export METHOD
    dependency=()
    if [ -n "$previous_generation" ]; then dependency=("--dependency=afterok:$previous_generation"); fi
    generation=$(sbatch --parsable "${resource_args[@]}" "${dependency[@]}" --export=ALL \
        --array="0-$((SHARDS-1))%$CONCURRENT" \
        --output="$DATA_ROOT/logs/full_%A_%a.out" --error="$DATA_ROOT/logs/full_%A_%a.err" scripts/slurm_worker.sh)
    generation="${generation%%;*}"
    printf '%s generation_array=%s\n' "$METHOD" "$generation" | tee -a "$JOURNAL"
    score=$(sbatch --parsable "${resource_args[@]}" --dependency="afterok:$generation" --export=ALL \
        --output="$DATA_ROOT/logs/score_%j.out" --error="$DATA_ROOT/logs/score_%j.err" scripts/slurm_evaluate.sh)
    score="${score%%;*}"
    previous_generation="$generation"
    printf '%s full_evaluation=%s\n' "$METHOD" "$score" | tee -a "$JOURNAL"
done
echo "Submission journal: $JOURNAL"
