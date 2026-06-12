#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CALIBRATION_DIR="${1:-output sim/alinea_calibration}"

usage() {
  cat <<EOF
Usage: $0 [CALIBRATION_DIR]

Post-process each K_P folder from scripts/calibrate_alinea_stochastic.py.

Examples:
  $0
  $0 "output sim/alinea_calibration"
EOF
}

case "$CALIBRATION_DIR" in
  -h|--help)
    usage
    exit 0
    ;;
esac

cd "$ROOT_DIR"

if [ ! -d "$CALIBRATION_DIR" ]; then
  echo "ALINEA calibration directory not found: ${CALIBRATION_DIR}" >&2
  exit 1
fi

RUNS_DIR="${CALIBRATION_DIR}/runs"
if [ ! -d "$RUNS_DIR" ]; then
  echo "ALINEA runs directory not found: ${RUNS_DIR}" >&2
  exit 1
fi

shopt -s nullglob
kp_dirs=("${RUNS_DIR}"/kp_*)
shopt -u nullglob

if [ "${#kp_dirs[@]}" -eq 0 ]; then
  echo "No K_P folders found in ${RUNS_DIR}" >&2
  exit 1
fi

echo "Post-processing ALINEA calibration results in ${CALIBRATION_DIR}"

for kp_dir in "${kp_dirs[@]}"; do
  if [ ! -d "$kp_dir" ]; then
    continue
  fi

  echo "Post-processing ${kp_dir}"

  python3 scripts/summarize_validation_runs.py \
    --input-dir "$kp_dir" \
    --output-dir "$kp_dir/validation_summary"

  python3 scripts/postprocessing/plot_input_sim_traffic_validation.py \
    --validation-dir "$kp_dir" \
    --output-dir "$kp_dir/start_edge_traffic_plots"

  python3 scripts/postprocessing/plot_metrics_csv_validation.py \
    --output-dir "$kp_dir/start_edge_traffic_plots/metrics_summary_plots" \
    "$kp_dir/start_edge_traffic_plots/plot_metrics.csv"

  if find "$kp_dir" -path "*/rep_*_seed_*/lanechanges.xml" -print -quit | grep -q .; then
    python3 scripts/postprocessing/plot_lanechanges_validation.py \
      --input-dir "$kp_dir" \
      --output-dir "$kp_dir/lanechange_plots"
  else
    echo "Skipping lane-change plots for ${kp_dir}: no lanechanges.xml files found"
  fi

  if find "$kp_dir" -path "*/rep_*_seed_*/edgedata.xml" -print -quit | grep -q .; then
    python3 scripts/postprocessing/plot_mainline_edgedata.py \
      --input-dir "$kp_dir" \
      --output-dir "$kp_dir/mainline_edgedata_plots"
  else
    echo "Skipping mainline edge-data plots for ${kp_dir}: no edgedata.xml files found"
  fi
done

echo "Finished post-processing ALINEA calibration results in ${CALIBRATION_DIR}"
