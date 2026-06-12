#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ARG="${1:-normal}"

usage() {
  cat <<EOF
Usage: $0 [normal|2040|RESULTS_DIR]

Arguments:
  normal, base      Post-process output sim/val_base_scenario
  2040, forecast    Post-process output sim/val_forecast_scenario
  RESULTS_DIR       Folder containing sumo*.log, LD_output*.xml, lanechanges*.xml, edgedata*.xml

Examples:
  $0 normal
  $0 2040
  $0 "output sim/val_base_scenario"
EOF
}

case "$RESULTS_ARG" in
  -h|--help)
    usage
    exit 0
    ;;
  normal|base)
    OUTPUT_DIR="output sim/val_base_scenario"
    ;;
  2040|forecast)
    OUTPUT_DIR="output sim/val_forecast_scenario"
    ;;
  *)
    OUTPUT_DIR="$RESULTS_ARG"
    ;;
esac

cd "$ROOT_DIR"

if [ ! -d "$OUTPUT_DIR" ]; then
  echo "Validation results directory not found: ${OUTPUT_DIR}" >&2
  exit 1
fi

echo "Post-processing validation results in ${OUTPUT_DIR}"

python3 scripts/summarize_validation_runs.py \
  --input-dir "$OUTPUT_DIR" \
  --output-dir "$OUTPUT_DIR/validation_summary"

python3 scripts/postprocessing/plot_input_sim_traffic_validation.py \
  --validation-dir "$OUTPUT_DIR" \
  --output-dir "$OUTPUT_DIR/start_edge_traffic_plots"

python3 scripts/postprocessing/plot_metrics_csv_validation.py \
  --output-dir "$OUTPUT_DIR/start_edge_traffic_plots/metrics_summary_plots" \
  "$OUTPUT_DIR/start_edge_traffic_plots/plot_metrics.csv"

python3 scripts/postprocessing/plot_lanechanges_validation.py \
  --input-dir "$OUTPUT_DIR" \
  --output-dir "$OUTPUT_DIR/lanechange_plots"

python3 scripts/postprocessing/plot_mainline_edgedata.py \
  --input-dir "$OUTPUT_DIR" \
  --output-dir "$OUTPUT_DIR/mainline_edgedata_plots"

echo "Finished post-processing validation results in ${OUTPUT_DIR}"
