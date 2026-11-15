#!/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
mode="${1:?Use --check-only or --submit}"
case "$mode" in --check-only|--submit) ;; *) exit 2 ;; esac
: "${ASSETS_ROOT:?Point to the existing benchmark sources}"
: "${ABLATION_RUN_ROOT:?Choose a new shared writable directory outside scratch}"
test -x "$JANUS_PYTHON"
test -x "$GENEVAL_PYTHON"
test -f "$ASSETS_ROOT/benchmarks/geneval/prompts/evaluation_metadata.jsonl"
test -f "$ASSETS_ROOT/benchmarks/setup_records/model_revisions_v1.json"
test -f "$DATA_ROOT/models/Janus-Pro-1B/config.json"
RECOVERY_SHARDS="${RECOVERY_SHARDS:-32}"
workers="${RECOVERY_CONCURRENT:-2}"
[[ "$RECOVERY_SHARDS" =~ ^[0-9]+$ && "$workers" =~ ^[0-9]+$ ]]
(( RECOVERY_SHARDS>=1 && RECOVERY_SHARDS<=553 && workers>=1 && workers<=RECOVERY_SHARDS ))
mkdir -p "$ABLATION_RUN_ROOT/logs"
ABLATION_RUN_ROOT="$(cd -- "$ABLATION_RUN_ROOT" && pwd -P)"
asset_data="$(cd -- "$DATA_ROOT" && pwd -P)"
case "$ABLATION_RUN_ROOT/" in "$asset_data/"*) echo 'Recovery outputs must be outside DATA_ROOT' >&2; exit 2 ;; esac
"$JANUS_PYTHON" -c 'import sys,tempfile; f=tempfile.NamedTemporaryFile(dir=sys.argv[1]); f.write(b"ok"); f.flush(); f.close(); print("ABLATION_OUTPUT_WRITE_OK")' "$ABLATION_RUN_ROOT/logs"
export REPO_ROOT CONFIG_FILE ASSETS_ROOT ABLATION_RUN_ROOT RECOVERY_SHARDS
resources=(--nodes=1 --ntasks=1 "--cpus-per-task=${CPUS_PER_TASK:-4}" "--mem=${MEMORY:-64G}" "--gres=${GPU_REQUEST:-gpu:1}")
if [ -n "${ACCOUNT:-}" ]; then resources+=("--account=$ACCOUNT"); fi
if [ -n "${QOS:-}" ]; then resources+=("--qos=$QOS"); fi
if [ -n "${PARTITION:-}" ]; then resources+=("--partition=$PARTITION"); fi
journal="$ABLATION_RUN_ROOT/logs/submissions.txt"
if [ "$mode" = --submit ]; then
    if [ -e "$journal" ]; then echo "Existing journal: $journal; inspect jobs before resubmitting." >&2; exit 1; fi
    (set -o noclobber; printf 'RUN=%s\n' "$ABLATION_RUN_ROOT" > "$journal")
fi
for RECOVERY_VARIANT in guard_temperature guard_gap; do
    export RECOVERY_VARIANT
    RECOVERY_PHASE=generate; export RECOVERY_PHASE
    args=("${resources[@]}" "--time=${RECOVERY_WALLTIME:-02:00:00}" "--array=0-$((RECOVERY_SHARDS-1))%$workers" --export=ALL)
    if [ "$mode" = --check-only ]; then
        sbatch --test-only "${args[@]}" scripts/janus_ablation_recovery_worker.sh
    else
        generation=$(sbatch --parsable "${args[@]}" --chdir="$REPO_ROOT" \
            --job-name="recover-janus1b-$RECOVERY_VARIANT" \
            --output="$ABLATION_RUN_ROOT/logs/${RECOVERY_VARIANT}_%A_%a.out" \
            --error="$ABLATION_RUN_ROOT/logs/${RECOVERY_VARIANT}_%A_%a.err" scripts/janus_ablation_recovery_worker.sh)
        generation="${generation%%;*}"
        printf '%s generation=%s\n' "$RECOVERY_VARIANT" "$generation" | tee -a "$journal"
    fi
    RECOVERY_PHASE=score; export RECOVERY_PHASE
    if [ "$mode" = --check-only ]; then
        sbatch --test-only "${resources[@]}" --time=00:30:00 --export=ALL scripts/janus_ablation_recovery_worker.sh
    else
        score=$(sbatch --parsable "${resources[@]}" --time=00:30:00 --export=ALL \
            "--dependency=afterok:$generation" --chdir="$REPO_ROOT" \
            --job-name="recover-score-$RECOVERY_VARIANT" \
            --output="$ABLATION_RUN_ROOT/logs/${RECOVERY_VARIANT}_score_%j.out" \
            --error="$ABLATION_RUN_ROOT/logs/${RECOVERY_VARIANT}_score_%j.err" scripts/janus_ablation_recovery_worker.sh)
        printf '%s evaluation=%s\n' "$RECOVERY_VARIANT" "${score%%;*}" | tee -a "$journal"
    fi
done
if [ "$mode" = --check-only ]; then
    echo 'CHECK_ONLY_OK: no jobs submitted'
else
    printf 'Each version waits only for its own generation. At most %s GPUs across these two versions.\n' "$((2*workers))"
    printf 'Results: %s/merged/evaluation/\n' "$ABLATION_RUN_ROOT"
fi
