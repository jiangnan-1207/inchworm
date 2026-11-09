#!/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
MODE="${1:?Use --check-only/--submit}"
FOUR_TASK="${2:?Use task1/task3/task4; task2 reads finished task1 results}"
RUN_TAG="${3:?Provide a unique RUN_TAG}"
shift 3
[[ "$RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ ]]
case "$MODE" in --check-only|--submit) ;; *) exit 2 ;; esac
case "$FOUR_TASK" in task1|task3|task4) ;; *) exit 2 ;; esac
MODELS=("$@")
if [ "${#MODELS[@]}" -eq 0 ]; then
    if [ "$FOUR_TASK" = task3 ]; then MODELS=(janus1b llamagen janus7b); else MODELS=(janus1b llamagen); fi
fi
declare -A seen=()
for model in "${MODELS[@]}"; do
    case "$model" in janus1b|llamagen) ;; janus7b) [ "$FOUR_TASK" = task3 ] || exit 2 ;; *) exit 2 ;; esac
    if [ -n "${seen[$model]:-}" ]; then echo 'Duplicate model' >&2; exit 2; fi
    seen[$model]=1
done
FOUR_TASK_SHARDS="${SHARDS:-32}"
WORKERS="${CONCURRENT:-1}"
[[ "$FOUR_TASK_SHARDS" =~ ^[0-9]+$ && "$WORKERS" =~ ^[0-9]+$ ]]
(( FOUR_TASK_SHARDS>=1 && FOUR_TASK_SHARDS<=553 && WORKERS>=1 && WORKERS<=FOUR_TASK_SHARDS ))
export REPO_ROOT CONFIG_FILE RUN_TAG FOUR_TASK FOUR_TASK_SHARDS
resources=("--gres=$GPU_REQUEST" "--cpus-per-task=$CPUS_PER_TASK" "--mem=$MEMORY" "--time=$WALLTIME" --nodes=1 --ntasks=1)
if [ -n "${ACCOUNT:-}" ]; then resources+=("--account=$ACCOUNT"); fi
if [ -n "${PARTITION:-}" ]; then resources+=("--partition=$PARTITION"); fi
if [ -n "${QOS:-}" ]; then resources+=("--qos=$QOS"); fi
journal="$DATA_ROOT/logs/$RUN_TAG.$FOUR_TASK.submissions.txt"
if [ "$MODE" = --submit ]; then
    if [ -e "$journal" ]; then echo "Existing journal: $journal; inspect jobs before resubmitting" >&2; exit 1; fi
    printf 'RUN=%s TASK=%s\n' "$RUN_TAG" "$FOUR_TASK" > "$journal"
fi
for EXPERIMENT_MODEL in "${MODELS[@]}"; do
    export EXPERIMENT_MODEL
    FOUR_TASK_PHASE=direct
    if [ "$FOUR_TASK" = task4 ]; then FOUR_TASK_PHASE=generate; fi
    export FOUR_TASK_PHASE
    array_args=()
    if [ "$FOUR_TASK" = task4 ]; then array_args=("--array=0-$((FOUR_TASK_SHARDS-1))%$WORKERS"); fi
    if [ "$MODE" = --check-only ]; then
        sbatch --test-only "${resources[@]}" "${array_args[@]}" --export=ALL scripts/four_tasks_worker.sh
        if [ "$FOUR_TASK" = task4 ]; then
            FOUR_TASK_PHASE=score; export FOUR_TASK_PHASE
            sbatch --test-only "${resources[@]}" --export=ALL scripts/four_tasks_worker.sh
        fi
    else
        generation=$(sbatch --parsable "${resources[@]}" "${array_args[@]}" --export=ALL \
            --job-name="$FOUR_TASK-$EXPERIMENT_MODEL" \
            --output="$DATA_ROOT/logs/${RUN_TAG}_${EXPERIMENT_MODEL}_%A_%a.out" \
            --error="$DATA_ROOT/logs/${RUN_TAG}_${EXPERIMENT_MODEL}_%A_%a.err" scripts/four_tasks_worker.sh)
        generation="${generation%%;*}"
        printf '%s generation=%s\n' "$EXPERIMENT_MODEL" "$generation" | tee -a "$journal"
        if [ "$FOUR_TASK" = task4 ]; then
            FOUR_TASK_PHASE=score; export FOUR_TASK_PHASE
            score=$(sbatch --parsable "${resources[@]}" --dependency="afterok:$generation" --export=ALL \
                --job-name="score-$EXPERIMENT_MODEL" \
                --output="$DATA_ROOT/logs/${RUN_TAG}_${EXPERIMENT_MODEL}_score_%j.out" \
                --error="$DATA_ROOT/logs/${RUN_TAG}_${EXPERIMENT_MODEL}_score_%j.err" scripts/four_tasks_worker.sh)
            printf '%s evaluation=%s\n' "$EXPERIMENT_MODEL" "$score" | tee -a "$journal"
        fi
    fi
done
