#!/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
mode="${1:?Use --check-only or --submit}"
task="${2:-deploy}"
case "$mode" in --check-only|--submit) ;; *) exit 2 ;; esac
case "$task" in deploy|full256|full512) ;; *) exit 2 ;; esac
: "${FRESH_LLAMA_ROOT:?Choose a new shared project directory for the independent deployment}"
: "${ASSETS_ROOT:?Existing benchmark sources}"
mkdir -p "$FRESH_LLAMA_ROOT/logs"
export REPO_ROOT CONFIG_FILE FRESH_LLAMA_ROOT ASSETS_ROOT
"$JANUS_PYTHON" -c 'import sys,tempfile; f=tempfile.NamedTemporaryFile(dir=sys.argv[1]); f.write(b"ok"); f.flush(); f.close(); print("FRESH_OUTPUT_WRITE_OK")' "$FRESH_LLAMA_ROOT/logs"
FRESH_SHARDS="${FRESH_SHARDS:-32}"; workers="${FRESH_CONCURRENT:-2}"
[[ "$FRESH_SHARDS" =~ ^[0-9]+$ && "$workers" =~ ^[0-9]+$ ]]
(( FRESH_SHARDS>=1 && FRESH_SHARDS<=553 && workers>=1 && workers<=FRESH_SHARDS ))
export FRESH_SHARDS
common=(--nodes=1 --ntasks=1)
if [ -n "${ACCOUNT:-}" ]; then common+=("--account=$ACCOUNT"); fi
if [ -n "${QOS:-}" ]; then common+=("--qos=$QOS"); fi
gpu=("${common[@]}" "--cpus-per-task=${CPUS_PER_TASK:-4}" "--mem=${MEMORY:-64G}" "--gres=${GPU_REQUEST:-gpu:1}")
if [ -n "${PARTITION:-}" ]; then gpu+=("--partition=$PARTITION"); fi
if [ "$task" = deploy ]; then
    if [ -e "$FRESH_LLAMA_ROOT/env" ]; then echo 'Fresh environment already exists; inspect that deployment first.' >&2; exit 1; fi
    first=setup; second=demo
    first_resources=("${common[@]}" --partition=cpu-small --cpus-per-task=2 --mem=8G --time=02:00:00)
    second_resources=("${gpu[@]}" --time=01:30:00)
else
    test -f "$FRESH_LLAMA_ROOT/setup_complete.json"
    test -f "$FRESH_LLAMA_ROOT/native_images/completed.json"
    FRESH_STAGE="${task#full}"; export FRESH_STAGE
    first=generate; second=score
    first_resources=("${gpu[@]}" --time=02:00:00 "--array=0-$((FRESH_SHARDS-1))%$workers")
    second_resources=("${gpu[@]}" --time=02:00:00)
fi
journal="$FRESH_LLAMA_ROOT/logs/$task.submissions.txt"
if [ "$mode" = --submit ]; then
    if [ -e "$journal" ]; then echo "Existing journal: $journal; inspect prior jobs." >&2; exit 1; fi
    (set -o noclobber; printf 'TASK=%s\n' "$task" > "$journal")
fi
FRESH_PHASE="$first"; export FRESH_PHASE
if [ "$mode" = --check-only ]; then
    sbatch --test-only "${first_resources[@]}" --export=ALL scripts/llamagen_fresh_worker.sh
else
    first_job=$(sbatch --parsable "${first_resources[@]}" --export=ALL --chdir="$REPO_ROOT" \
        --job-name="llama-fresh-$task-$first" --output="$FRESH_LLAMA_ROOT/logs/${task}_${first}_%A_%a.out" \
        --error="$FRESH_LLAMA_ROOT/logs/${task}_${first}_%A_%a.err" scripts/llamagen_fresh_worker.sh)
    first_job="${first_job%%;*}"
    printf '%s=%s\n' "$first" "$first_job" | tee -a "$journal"
fi
FRESH_PHASE="$second"; export FRESH_PHASE
if [ "$mode" = --check-only ]; then
    sbatch --test-only "${second_resources[@]}" --export=ALL scripts/llamagen_fresh_worker.sh
    echo 'CHECK_ONLY_OK: no jobs submitted'
else
    second_job=$(sbatch --parsable "${second_resources[@]}" --dependency="afterok:$first_job" --export=ALL --chdir="$REPO_ROOT" \
        --job-name="llama-fresh-$task-$second" --output="$FRESH_LLAMA_ROOT/logs/${task}_${second}_%j.out" \
        --error="$FRESH_LLAMA_ROOT/logs/${task}_${second}_%j.err" scripts/llamagen_fresh_worker.sh)
    printf '%s=%s\n' "$second" "${second_job%%;*}" | tee -a "$journal"
fi
