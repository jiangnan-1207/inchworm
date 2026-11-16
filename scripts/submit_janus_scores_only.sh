#!/bin/bash
# Recover scoring only; never call a generator or modify generation receipts.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
mode="${1:?Use --check-only or --submit}"
case "$mode" in --check-only|--submit) ;; *) exit 2 ;; esac
: "${ASSETS_ROOT:?Existing benchmark checkout}"
: "${BASELINE_RUN_ROOT:?Existing pure backbone run}"
: "${ABLATION_RUN_ROOT:?Existing ablation recovery run}"
: "${SCORE_LOG_ROOT:?New writable submission/log directory}"
: "${GAP_GENERATION_JOB:?Existing high-H/high-G generation array job ID}"
[[ "$GAP_GENERATION_JOB" =~ ^[0-9]+$ ]]
test -x "$GENEVAL_PYTHON"
test -f "$BASELINE_RUN_ROOT/shards/janus1b_backbone_0/run_spec.json"
test -f "$ABLATION_RUN_ROOT/shards/janus1b_guard_temperature_0/run_spec.json"
mkdir -p "$SCORE_LOG_ROOT"
"$GENEVAL_PYTHON" -c 'import sys,tempfile; f=tempfile.NamedTemporaryFile(dir=sys.argv[1]); f.write(b"ok"); f.flush(); f.close(); print("SCORE_LOG_WRITE_OK")' "$SCORE_LOG_ROOT"
export REPO_ROOT CONFIG_FILE ASSETS_ROOT
RESCORE_MODEL=janus1b; export RESCORE_MODEL
resources=(--nodes=1 --ntasks=1 "--cpus-per-task=${CPUS_PER_TASK:-4}" "--mem=${MEMORY:-64G}" "--gres=${GPU_REQUEST:-gpu:1}" "--time=${SCORE_WALLTIME:-02:00:00}")
if [ -n "${ACCOUNT:-}" ]; then resources+=("--account=$ACCOUNT"); fi
if [ -n "${QOS:-}" ]; then resources+=("--qos=$QOS"); fi
if [ -n "${PARTITION:-}" ]; then resources+=("--partition=$PARTITION"); fi
journal="$SCORE_LOG_ROOT/submissions.txt"
if [ "$mode" = --submit ]; then
    if [ -e "$journal" ]; then echo "Existing journal: $journal; inspect prior submissions." >&2; exit 1; fi
    (set -o noclobber; printf 'SCORE_ONLY\n' > "$journal")
fi
for RESCORE_VARIANT in backbone guard_temperature guard_gap; do
    if [ "$RESCORE_VARIANT" = backbone ]; then
        EXISTING_RUN="$BASELINE_RUN_ROOT"; RESCORE_SHARDS=1
    else
        EXISTING_RUN="$ABLATION_RUN_ROOT"; RESCORE_SHARDS=32
    fi
    RESCORE_ROOT="$EXISTING_RUN"
    export RESCORE_VARIANT EXISTING_RUN RESCORE_ROOT RESCORE_SHARDS
    dependency=()
    if [ "$RESCORE_VARIANT" = guard_gap ]; then dependency=("--dependency=afterok:$GAP_GENERATION_JOB"); fi
    if [ "$mode" = --check-only ]; then
        sbatch --test-only "${resources[@]}" --export=ALL scripts/rescore_existing_variant_worker.sh
    else
        job=$(sbatch --parsable "${resources[@]}" "${dependency[@]}" --export=ALL --chdir="$REPO_ROOT" \
            --job-name="score-fix-janus1b-$RESCORE_VARIANT" \
            --output="$SCORE_LOG_ROOT/${RESCORE_VARIANT}_%j.out" \
            --error="$SCORE_LOG_ROOT/${RESCORE_VARIANT}_%j.err" scripts/rescore_existing_variant_worker.sh)
        printf '%s score_only=%s\n' "$RESCORE_VARIANT" "${job%%;*}" | tee -a "$journal"
    fi
done
if [ "$mode" = --check-only ]; then echo 'CHECK_ONLY_OK: no jobs submitted'; fi
