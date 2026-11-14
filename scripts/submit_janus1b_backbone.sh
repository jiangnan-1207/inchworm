#!/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
mode="${1:?Use --check-only or --submit}"
case "$mode" in --check-only|--submit) ;; *) exit 2 ;; esac
: "${ASSETS_ROOT:?Point to the existing benchmark source checkout}"
: "${BASELINE_RUN_ROOT:?Choose a new writable shared run directory}"
test -x "$JANUS_PYTHON"
test -x "$GENEVAL_PYTHON"
test -f "$ASSETS_ROOT/benchmarks/geneval/prompts/evaluation_metadata.jsonl"
test -f "$ASSETS_ROOT/benchmarks/setup_records/model_revisions_v1.json"
test -f "$DATA_ROOT/models/Janus-Pro-1B/config.json"
mkdir -p "$BASELINE_RUN_ROOT/logs"
BASELINE_RUN_ROOT="$(cd -- "$BASELINE_RUN_ROOT" && pwd)"
"$JANUS_PYTHON" -c 'import sys,tempfile; f=tempfile.NamedTemporaryFile(dir=sys.argv[1]); f.write(b"ok"); f.flush(); f.close(); print("BASELINE_OUTPUT_WRITE_OK")' "$BASELINE_RUN_ROOT/logs"
export REPO_ROOT CONFIG_FILE ASSETS_ROOT BASELINE_RUN_ROOT
resources=(--nodes=1 --ntasks=1 "--cpus-per-task=${CPUS_PER_TASK:-4}" "--mem=${MEMORY:-64G}"
    "--time=${BACKBONE_WALLTIME:-08:00:00}" "--gres=${GPU_REQUEST:-gpu:1}")
if [ -n "${ACCOUNT:-}" ]; then resources+=("--account=$ACCOUNT"); fi
if [ -n "${QOS:-}" ]; then resources+=("--qos=$QOS"); fi
if [ -n "${PARTITION:-}" ]; then resources+=("--partition=$PARTITION"); fi
if [ "$mode" = --check-only ]; then
    sbatch --test-only "${resources[@]}" --export=ALL scripts/janus1b_backbone_full_worker.sh
    echo 'CHECK_ONLY_OK: no job submitted'
    exit 0
fi
journal="$BASELINE_RUN_ROOT/logs/submission.txt"
if [ -e "$journal" ]; then echo "Existing submission: $journal. Inspect the prior job before resubmitting." >&2; exit 1; fi
(set -o noclobber; printf 'RUN=%s\n' "$BASELINE_RUN_ROOT" > "$journal")
job=$(sbatch --parsable "${resources[@]}" --export=ALL --job-name=janus1b_native_full \
    --chdir="$REPO_ROOT" --output="$BASELINE_RUN_ROOT/logs/baseline_%j.out" \
    --error="$BASELINE_RUN_ROOT/logs/baseline_%j.err" scripts/janus1b_backbone_full_worker.sh)
printf 'generation_and_evaluation=%s\n' "$job" | tee -a "$journal"
