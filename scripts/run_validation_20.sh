#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENARIO_OR_CONFIG="${1:-normal}"
CUSTOM_OUTPUT_DIR="${2:-}"
SUMO_CONFIG=""
OUTPUT_DIR=""
NUM_RUNS="${NUM_RUNS:-20}"
PARALLEL_JOBS="${PARALLEL_JOBS:-10}"
EDGEDATA_PERIOD_SECONDS="${EDGEDATA_PERIOD_SECONDS:-300}"

usage() {
  cat <<EOF
Usage: $0 [normal|2040|SUMO_CONFIG] [OUTPUT_DIR]

Scenarios:
  normal, base      Use A1.sumocfg
  2040, forecast    Use A1_2040.sumocfg

Examples:
  $0 normal
  $0 2040
  $0 2040 "output sim/val_forecast_scenario"
  $0 A1.sumocfg "output sim/custom_validation"

Environment options:
  NUM_RUNS                 Number of SUMO validation runs to execute (default: 20)
  PARALLEL_JOBS            Number of validation runs to execute in parallel (default: 10)
  EDGEDATA_PERIOD_SECONDS  Aggregation period for generated edgeData files (default: 300)
  VALIDATION_SEEDS         Comma- or space-separated list of exactly NUM_RUNS seeds
EOF
}

case "$SCENARIO_OR_CONFIG" in
  -h|--help)
    usage
    exit 0
    ;;
  normal|base)
    SUMO_CONFIG="A1.sumocfg"
    OUTPUT_DIR="${CUSTOM_OUTPUT_DIR:-output sim/val_base_scenario}"
    ;;
  2040|forecast)
    SUMO_CONFIG="A1_2040.sumocfg"
    OUTPUT_DIR="${CUSTOM_OUTPUT_DIR:-output sim/val_forecast_scenario}"
    ;;
  *.sumocfg)
    SUMO_CONFIG="$SCENARIO_OR_CONFIG"
    OUTPUT_DIR="${CUSTOM_OUTPUT_DIR:-output sim/val3}"
    ;;
  *)
    echo "Unknown scenario or SUMO config: ${SCENARIO_OR_CONFIG}" >&2
    usage >&2
    exit 1
    ;;
esac

SEEDS_CSV="${OUTPUT_DIR}/validation_run_seeds.csv"

cd "$ROOT_DIR"

if [ ! -f "$SUMO_CONFIG" ]; then
  echo "SUMO config not found: ${SUMO_CONFIG}" >&2
  exit 1
fi

echo "Using SUMO config ${SUMO_CONFIG}; writing validation output to ${OUTPUT_DIR}"

mkdir -p "$OUTPUT_DIR"
printf "run,seed\n" > "$SEEDS_CSV"

move_output_if_exists() {
  local preferred_source="$1"
  local fallback_source="$2"
  local target_path="$3"

  if [ -f "$preferred_source" ]; then
    mv -f "$preferred_source" "$target_path"
    return
  fi

  if [ -f "$fallback_source" ]; then
    mv -f "$fallback_source" "$target_path"
    return
  fi

  echo "Warning: expected output not found for ${target_path}" >&2
}

run_single_validation() {
  local run_number="$1"
  local seed="$2"
  local run_output_dir
  local edgedata_additional_file
  run_output_dir="${OUTPUT_DIR}/run_$(printf '%02d' "$run_number")"
  edgedata_additional_file=".validation_edgedata_${run_number}.add.xml"

  mkdir -p "$run_output_dir"
  printf "%s\n" \
    "<additional>" \
    "  <edgeData id=\"mainline_edgedata_${run_number}\" file=\"edgedata${run_number}.xml\" freq=\"${EDGEDATA_PERIOD_SECONDS}\"/>" \
    "</additional>" > "$edgedata_additional_file"

  echo "Starting SUMO validation run ${run_number}/${NUM_RUNS} with seed ${seed}"

  if ! sumo -c "$SUMO_CONFIG" \
    --seed "$seed" \
    --duration-log.statistics \
    --no-warnings \
    --additional-files "detSIM.add.xml,detRM.add.xml,detDFR.add.xml,${edgedata_additional_file}" \
    --output-prefix "${run_output_dir}/" \
    --lanechange-output "lanechanges${run_number}.xml" \
    --tripinfo-output "tripinfo${run_number}.xml" \
    --device.emissions.probability 1; then
    rm -f "$edgedata_additional_file"
    return 1
  fi

  rm -f "$edgedata_additional_file"

  move_output_if_exists \
    "${run_output_dir}/LD_output.xml" \
    "LD_output.xml" \
    "$OUTPUT_DIR/LD_output${run_number}.xml"
  move_output_if_exists \
    "${run_output_dir}/sumo.log" \
    "sumo.log" \
    "$OUTPUT_DIR/sumo${run_number}.log"
  move_output_if_exists \
    "${run_output_dir}/lanechanges${run_number}.xml" \
    "lanechanges${run_number}.xml" \
    "$OUTPUT_DIR/lanechanges${run_number}.xml"
  move_output_if_exists \
    "${run_output_dir}/tripinfo${run_number}.xml" \
    "tripinfo${run_number}.xml" \
    "$OUTPUT_DIR/tripinfo${run_number}.xml"
  move_output_if_exists \
    "${run_output_dir}/edgedata${run_number}.xml" \
    "edgedata${run_number}.xml" \
    "$OUTPUT_DIR/edgedata${run_number}.xml"

  echo "Finished SUMO validation run ${run_number}/${NUM_RUNS}"
}

seeds=()
if [ -n "${VALIDATION_SEEDS:-}" ]; then
  while IFS= read -r seed; do
    [ -n "$seed" ] && seeds+=("$seed")
  done < <(printf "%s\n" "$VALIDATION_SEEDS" | tr ', ' '\n')

  if [ "${#seeds[@]}" -ne "$NUM_RUNS" ]; then
    echo "Expected ${NUM_RUNS} validation seeds, got ${#seeds[@]}." >&2
    exit 1
  fi

  for i in $(seq 1 "$NUM_RUNS"); do
    printf "%s,%s\n" "$i" "${seeds[$((i - 1))]}" >> "$SEEDS_CSV"
  done
else
  for i in $(seq 1 "$NUM_RUNS"); do
    seed=$(((RANDOM << 15) | RANDOM))
    seeds+=("$seed")
    printf "%s,%s\n" "$i" "$seed" >> "$SEEDS_CSV"
  done
fi

echo "Launching ${NUM_RUNS} SUMO validation runs with parallelism ${PARALLEL_JOBS}"

pids=()
for i in $(seq 1 "$NUM_RUNS"); do
  run_single_validation "$i" "${seeds[$((i - 1))]}" &
  pids+=("$!")

  while [ "$(jobs -pr | wc -l | tr -d '[:space:]')" -ge "$PARALLEL_JOBS" ]; do
    sleep 1
  done
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failures=$((failures + 1))
  fi
done

if [ "$failures" -ne 0 ]; then
  echo "Validation failed: ${failures} SUMO run(s) exited with an error." >&2
  exit 1
fi

echo "Finished ${NUM_RUNS} SUMO validation runs."
echo "Run post-processing with: bash scripts/postprocess_validation.sh \"${OUTPUT_DIR}\""
