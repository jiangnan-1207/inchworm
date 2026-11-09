#!/bin/bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
MODE="${1:?Use --check-only/--submit}"
STAGE="${2:?Use prepare/generate}"
RUN_TAG="${3:?Unique run tag required}"
[[ "$RUN_TAG" =~ ^[a-zA-Z0-9_-]+$ ]]
case "$MODE" in --check-only|--submit) ;; *) exit 2 ;; esac
case "$STAGE" in prepare|generate) ;; *) exit 2 ;; esac
export REPO_ROOT CONFIG_FILE RUN_TAG
ALIGNMENT_STAGE="$STAGE"; export ALIGNMENT_STAGE
account=(); if [ -n "${ACCOUNT:-}" ]; then account+=("--account=$ACCOUNT"); fi
if [ -n "${QOS:-}" ]; then account+=("--qos=$QOS"); fi
if [ "$STAGE" = prepare ]; then
  resources=(--partition=cpu-small --cpus-per-task=4 --mem=32G --time=02:00:00)
else
  test -f "$REPO_ROOT/benchmarks/setup_records/alignment_manifest.json"
  test -f "$REPO_ROOT/benchmarks/setup_records/alignment_dpg_judge.json"
  test -f "$REPO_ROOT/benchmarks/setup_records/alignment_tiif_judge.json"
  test -x "$JANUS_PYTHON"; test -x "$LLAMA_PYTHON"
  ALIGNMENT_SHARDS="${ALIGNMENT_SHARDS:-16}"; ALIGNMENT_CONCURRENT="${ALIGNMENT_CONCURRENT:-2}"
  [[ "$ALIGNMENT_SHARDS" =~ ^[0-9]+$ && "$ALIGNMENT_CONCURRENT" =~ ^[0-9]+$ ]]
  (( ALIGNMENT_SHARDS>=1 && ALIGNMENT_SHARDS<=64 && ALIGNMENT_CONCURRENT>=1 && ALIGNMENT_CONCURRENT<=ALIGNMENT_SHARDS ))
  export ALIGNMENT_SHARDS
  resources=("--partition=${ALIGNMENT_PARTITION:-gpu-l40s}" "--gres=${ALIGNMENT_GRES:-gpu:l40s:1}" \
    --cpus-per-task=4 --mem=64G "--time=${ALIGNMENT_TIME:-08:00:00}" "--array=0-$((ALIGNMENT_SHARDS-1))%$ALIGNMENT_CONCURRENT")
fi
resources+=(--nodes=1 --ntasks=1 --export=ALL)
if [ "$STAGE" = prepare ]; then models=(datasets); else models=(janus1b llamagen); fi
journal="$DATA_ROOT/logs/$RUN_TAG.alignment-$STAGE.txt"
if [ "$MODE" = --submit ]; then
  if [ -e "$journal" ]; then echo "Existing submission journal: $journal; inspect it." >&2; exit 1; fi
  printf 'stage=%s\n' "$STAGE" > "$journal"
fi
for ALIGNMENT_MODEL in "${models[@]}"; do
  export ALIGNMENT_MODEL
  if [ "$MODE" = --check-only ]; then
    sbatch --test-only "${account[@]}" "${resources[@]}" scripts/alignment_worker.sh
  else
    job=$(sbatch --parsable "${account[@]}" "${resources[@]}" --job-name="align_${STAGE}_${ALIGNMENT_MODEL}" \
      --output="$DATA_ROOT/logs/${RUN_TAG}_${STAGE}_${ALIGNMENT_MODEL}_%A_%a.out" \
      --error="$DATA_ROOT/logs/${RUN_TAG}_${STAGE}_${ALIGNMENT_MODEL}_%A_%a.err" scripts/alignment_worker.sh)
    printf '%s %s\n' "$ALIGNMENT_MODEL" "$job" | tee -a "$journal"
    if [ "$STAGE" = generate ]; then
      generation_job="${job%%;*}"
      for ALIGNMENT_BENCHMARK in dpg tiif; do
        export ALIGNMENT_BENCHMARK
        ALIGNMENT_STAGE=score
        score_job=$(sbatch --parsable "${account[@]}" --nodes=1 --ntasks=1 --export=ALL \
          "--partition=${ALIGNMENT_PARTITION:-gpu-l40s}" "--gres=${ALIGNMENT_GRES:-gpu:l40s:1}" \
          --cpus-per-task=4 --mem=64G --time=08:00:00 "--dependency=afterok:$generation_job" \
          --job-name="align_score_${ALIGNMENT_MODEL}_${ALIGNMENT_BENCHMARK}" \
          --output="$DATA_ROOT/logs/${RUN_TAG}_score_${ALIGNMENT_MODEL}_${ALIGNMENT_BENCHMARK}_%j.out" \
          --error="$DATA_ROOT/logs/${RUN_TAG}_score_${ALIGNMENT_MODEL}_${ALIGNMENT_BENCHMARK}_%j.err" scripts/alignment_worker.sh)
        printf '%s %s score=%s\n' "$ALIGNMENT_MODEL" "$ALIGNMENT_BENCHMARK" "$score_job" | tee -a "$journal"
      done
      ALIGNMENT_STAGE=generate
    fi
  fi
done
