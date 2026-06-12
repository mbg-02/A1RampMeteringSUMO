#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

PLOT_CACHE_DIR = Path(tempfile.gettempdir()) / "sumo_plot_cache"
PLOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PLOT_CACHE_DIR / "xdg"))

import plot_metrics_csv as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create metric summary plots for the 20-run validation traffic outputs."
        )
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Directory for generated metric summary plots. Defaults to a "
            "metrics_summary_plots folder next to the validation plot_metrics.csv."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=34,
        help="How many detectors to show per metric plot.",
    )
    parser.add_argument(
        "metrics_csv",
        nargs="?",
        help="Validation metrics CSV exported by plot_input_sim_traffic_validation.py.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics_csv = args.metrics_csv
    output_dir = args.output_dir
    if metrics_csv is None:
        if output_dir is None:
            metrics_csv = "output sim/val1/start_edge_traffic_plots/plot_metrics.csv"
        else:
            metrics_csv = str((Path(output_dir).parent / "plot_metrics.csv"))
    if output_dir is None:
        output_dir = str(Path(metrics_csv).parent / "metrics_summary_plots")
    sys.argv = [
        sys.argv[0],
        "--output-dir",
        output_dir,
        "--top-n",
        str(args.top_n),
        metrics_csv,
    ]
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
