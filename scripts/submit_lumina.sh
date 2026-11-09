#!/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
MODE="${1:?Use --check-only or --submit}"
RUN_TAG="${2:?Provide a unique Lumina run tag}"
[[ "$RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ ]]
case "$MODE" in --check-only|--submit) ;; *) exit 2 ;; esac
LUMINA_SHARDS="${LUMINA_SHARDS:-6}"
LUMINA_CONCURRENT="${LUMINA_CONCURRENT:-2}"
[[ "$LUMINA_SHARDS" =~ ^[0-9]+$ && "$LUMINA_CONCURRENT" =~ ^[0-9]+$ ]]
(( LUMINA_SHARDS>=1 && LUMINA_SHARDS<=30 && LUMINA_CONCURRENT>=1 && LUMINA_CONCURRENT<=LUMINA_SHARDS ))
test -f "$REPO_ROOT/benchmarks/setup_records/lumina_assets.json"
test -x "$DATA_ROOT/envs/lumina-mgpt2/bin/python"
export REPO_ROOT CONFIG_FILE RUN_TAG LUMINA_SHARDS
resources=(--nodes=1 --ntasks=1 --cpus-per-task=4 --mem=96G "--time=${LUMINA_WALLTIME:-08:00:00}" \
    "--partition=${LUMINA_PARTITION:-gpu-h200}" "--gres=${LUMINA_GRES:-gpu:h200:1}")
if [ -n "${ACCOUNT:-}" ]; then resources+=("--account=$ACCOUNT"); fi
if [ -n "${QOS:-}" ]; then resources+=("--qos=$QOS"); fi
if [ "$MODE" = --check-only ]; then
    sbatch --test-only "${resources[@]}" --array="0-$((LUMINA_SHARDS-1))%$LUMINA_CONCURRENT" --export=ALL scripts/lumina_worker.sh
    exit 0
fi
journal="$DATA_ROOT/logs/$RUN_TAG.lumina-submissions.txt"
if [ -e "$journal" ]; then echo "Existing journal: $journal. Inspect the existing array." >&2; exit 1; fi
printf 'RUN=%s SHARDS=%s\n' "$RUN_TAG" "$LUMINA_SHARDS" > "$journal"
job=$(sbatch --parsable "${resources[@]}" --array="0-$((LUMINA_SHARDS-1))%$LUMINA_CONCURRENT" --export=ALL \
    --job-name=lumina_original_inchworm \
    --output="$DATA_ROOT/logs/${RUN_TAG}_%A_%a.out" --error="$DATA_ROOT/logs/${RUN_TAG}_%A_%a.err" scripts/lumina_worker.sh)
printf 'generation_array=%s\n' "$job" | tee -a "$journal"
