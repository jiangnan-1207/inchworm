#!/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
MODE="${1:?Use --check-only or --submit}"
RUN_TAG="${2:?Provide a unique run tag}"
[[ "$RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ ]]
case "$MODE" in --check-only|--submit) ;; *) exit 2 ;; esac
export REPO_ROOT CONFIG_FILE RUN_TAG
resources=("--gres=$GPU_REQUEST" "--cpus-per-task=$CPUS_PER_TASK" "--mem=$MEMORY" "--time=$WALLTIME" --nodes=1 --ntasks=1)
if [ -n "${ACCOUNT:-}" ]; then resources+=("--account=$ACCOUNT"); fi
if [ -n "${PARTITION:-}" ]; then resources+=("--partition=$PARTITION"); fi
if [ -n "${QOS:-}" ]; then resources+=("--qos=$QOS"); fi
journal="$DATA_ROOT/logs/$RUN_TAG.requested-submissions.txt"
if [ "$MODE" = --submit ]; then
    if [ -e "$journal" ]; then echo "Existing journal: $journal. Inspect prior jobs." >&2; exit 1; fi
    printf 'Run %s\n' "$RUN_TAG" > "$journal"
fi
for EXPERIMENT_MODEL in janus1b llamagen janus7b; do
    export EXPERIMENT_MODEL
    if [ "$MODE" = --check-only ]; then
        sbatch --test-only "${resources[@]}" --export=ALL scripts/requested_worker.sh
    else
        id=$(sbatch --parsable "${resources[@]}" --export=ALL --job-name="profile_$EXPERIMENT_MODEL" \
            --output="$DATA_ROOT/logs/${RUN_TAG}_${EXPERIMENT_MODEL}_%j.out" \
            --error="$DATA_ROOT/logs/${RUN_TAG}_${EXPERIMENT_MODEL}_%j.err" scripts/requested_worker.sh)
        printf '%s %s\n' "$EXPERIMENT_MODEL" "$id" | tee -a "$journal"
    fi
done
echo 'Three independent jobs, one GPU each; each reuses its loaded model for the requested work.'
