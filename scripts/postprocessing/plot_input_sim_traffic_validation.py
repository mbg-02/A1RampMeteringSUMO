#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re
import tempfile

PLOT_CACHE_DIR = Path(tempfile.gettempdir()) / "sumo_plot_cache"
PLOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PLOT_CACHE_DIR / "xdg"))

import numpy as np

import plot_input_sim_traffic as base


SIMULATION_OUTPUT_PATTERN = re.compile(r"LD_output(\d+)\.xml$")
REPLICATION_DIR_PATTERN = re.compile(r"rep_(\d+)_seed_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create validation traffic plots using the median detector flow over 20 "
            "SUMO runs and the mean 5-minute speed over those runs."
        )
    )
    parser.add_argument(
        "--validation-dir",
        default="output sim/val1",
        help=(
            "Directory containing LD_output1.xml ... LD_output20.xml, or "
            "calibration-style rep_XX_seed_*/LD_output.xml files."
        ),
    )
    parser.add_argument(
        "--simulation-pattern",
        default="LD_output*.xml",
        help="Glob used to discover validation detector XML outputs.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Directory for generated validation plot files. Defaults to a "
            "start_edge_traffic_plots folder inside --validation-dir."
        ),
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["svg", "png", "pdf"],
        help="Output image format.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=base.INTERVAL_SECONDS,
        help="Aggregation bucket size in seconds.",
    )
    parser.add_argument(
        "--detector-flows-csv",
        default=base.DEFAULT_DETECTOR_FLOWS_CSV,
        help="CSV with detector input flows to overlay.",
    )
    parser.add_argument(
        "--speed-input-dir",
        "--speed-input-csv",
        dest="speed_input_dir",
        default=base.DEFAULT_SPEED_INPUT_DIR,
        help="Directory with raw VBV speed CSV files.",
    )
    parser.add_argument(
        "emitter_files",
        nargs="*",
        default=base.DEFAULT_EMITTER_FILES,
        help="Emitter XML files to read.",
    )
    return parser.parse_args()


def discover_simulation_outputs(
    validation_dir: Path,
    simulation_pattern: str,
) -> list[Path]:
    paths: list[tuple[int, Path]] = []
    for path in validation_dir.glob(simulation_pattern):
        match = SIMULATION_OUTPUT_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        paths.append((int(match.group(1)), path))

    if paths:
        paths.sort(key=lambda item: item[0])
        return [path for _, path in paths]

    for path in validation_dir.glob(f"rep_*_seed_*/{simulation_pattern}"):
        match = REPLICATION_DIR_PATTERN.fullmatch(path.parent.name)
        if match is None:
            continue
        paths.append((int(match.group(1)), path))

    paths.sort(key=lambda item: item[0])
    if not paths:
        raise FileNotFoundError(
            f"No validation detector XML files matching {simulation_pattern} found in "
            f"{validation_dir}"
        )
    return [path for _, path in paths]


def output_dir_for(validation_dir: Path, output_dir_arg: str | None) -> Path:
    if output_dir_arg:
        return base.resolve_output_dir(output_dir_arg)
    return validation_dir / "start_edge_traffic_plots"


def aggregate_median_counts(
    per_run_counts: list[dict[str, np.ndarray]],
    num_bins: int,
) -> dict[str, np.ndarray]:
    detector_ids = sorted(base.SIMULATION_DETECTOR_MAPPING)
    aggregated: dict[str, np.ndarray] = {}

    for detector_id in detector_ids:
        stacked = np.vstack(
            [
                run_counts.get(detector_id, np.zeros(num_bins, dtype=float))
                for run_counts in per_run_counts
            ]
        )
        aggregated[detector_id] = np.median(stacked, axis=0)

    return aggregated


def aggregate_mean_speeds(
    per_run_speeds: list[dict[str, np.ndarray]],
    num_bins: int,
) -> dict[str, np.ndarray]:
    detector_ids = sorted(
        {
            simulation_detector_id
            for lane_pairs in base.SPEED_DETECTOR_MAPPING.values()
            for _, simulation_detector_id in lane_pairs
        }
    )
    aggregated: dict[str, np.ndarray] = {}

    for detector_id in detector_ids:
        stacked = np.vstack(
            [
                run_speeds.get(detector_id, np.full(num_bins, np.nan, dtype=float))
                for run_speeds in per_run_speeds
            ]
        )
        all_nan_bins = np.all(np.isnan(stacked), axis=0)
        mean_values = np.full(num_bins, np.nan, dtype=float)
        if np.any(~all_nan_bins):
            mean_values[~all_nan_bins] = np.nanmean(
                stacked[:, ~all_nan_bins],
                axis=0,
            )
        aggregated[detector_id] = mean_values

    return aggregated


def main() -> int:
    args = parse_args()

    emitter_files = [base.resolve_input_path(file_name) for file_name in args.emitter_files]
    missing_emitter_files = [str(path) for path in emitter_files if not path.exists()]
    if missing_emitter_files:
        raise FileNotFoundError(
            f"Missing emitter file(s): {', '.join(missing_emitter_files)}"
        )

    validation_dir = base.resolve_input_path(args.validation_dir)
    simulation_output_paths = discover_simulation_outputs(
        validation_dir, args.simulation_pattern
    )

    output_dir = output_dir_for(validation_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    route_edges = base.load_route_edges(emitter_files)
    counts, num_bins = base.build_counts(
        emitter_files,
        args.interval_seconds,
        route_edges,
    )

    detector_flows_csv = base.resolve_input_path(args.detector_flows_csv)
    if not detector_flows_csv.exists():
        raise FileNotFoundError(f"Missing detector flows CSV: {detector_flows_csv}")
    detector_input_counts = base.build_detector_input_counts(
        detector_flows_csv,
        args.interval_seconds,
        num_bins,
    )

    speed_input_dir = base.resolve_input_path(args.speed_input_dir)
    if not speed_input_dir.exists():
        raise FileNotFoundError(f"Missing speed input directory: {speed_input_dir}")
    speed_input_profiles, missing_speed_lanes = base.build_speed_input_profiles(
        speed_input_dir,
        args.interval_seconds,
        num_bins,
    )

    per_run_counts = [
        base.build_simulation_counts(path, args.interval_seconds, num_bins)
        for path in simulation_output_paths
    ]
    per_run_speeds = [
        base.build_simulation_speed_series(path, args.interval_seconds, num_bins)
        for path in simulation_output_paths
    ]
    simulation_counts = aggregate_median_counts(per_run_counts, num_bins)
    simulation_speed_series = aggregate_mean_speeds(per_run_speeds, num_bins)

    metrics_rows: list[dict[str, str | float]] = []

    for detector_id, vehicle_counts in counts.items():
        output_path = output_dir / f"{detector_id}_stacked_traffic.{args.format}"
        (
            simulation_modeled_total,
            simulation_counts_for_plot,
            simulation_comparison_label,
        ) = base.simulation_comparison_series(
            detector_id,
            vehicle_counts,
            counts,
            simulation_counts,
            num_bins,
        )
        metrics = base.compute_metrics(
            vehicle_counts,
            detector_input_counts.get(detector_id),
            args.interval_seconds,
        )
        simulation_input_total, _ = base.input_comparison_series(
            detector_id,
            detector_input_counts,
            num_bins,
        )
        smoothed_simulation_counts = None
        if simulation_counts_for_plot is not None:
            smoothed_simulation_counts, _ = base.smooth_simulation_counts(
                simulation_counts_for_plot,
                args.interval_seconds,
            )
        simulation_metrics = None
        if smoothed_simulation_counts is not None:
            simulation_metrics = base.compute_total_metrics(
                smoothed_simulation_counts,
                simulation_modeled_total,
                args.interval_seconds,
            )
        simulation_input_metrics = None
        if (
            smoothed_simulation_counts is not None
            and simulation_input_total is not None
        ):
            simulation_input_metrics = base.compute_total_metrics(
                smoothed_simulation_counts,
                simulation_input_total,
                args.interval_seconds,
            )
        plot_edge = base.DETECTOR_TO_PLOT_EDGE[detector_id]
        metrics_rows.append(
            {
                "detector_id": detector_id,
                "edge": plot_edge,
                "sqv_interval_median": (
                    ""
                    if metrics is None or metrics["sqv_interval_median"] is None
                    else float(metrics["sqv_interval_median"])
                ),
                "hourly_sqv_median": (
                    ""
                    if metrics is None or metrics["hourly_sqv_median"] is None
                    else float(metrics["hourly_sqv_median"])
                ),
                "wape_5min": (
                    ""
                    if metrics is None or metrics["wape_5min"] is None
                    else float(metrics["wape_5min"])
                ),
                "hourly_wape_median": (
                    ""
                    if metrics is None or metrics["hourly_wape_median"] is None
                    else float(metrics["hourly_wape_median"])
                ),
                "daily_total_error": (
                    ""
                    if metrics is None or metrics["daily_total_error"] is None
                    else float(metrics["daily_total_error"])
                ),
                "notes": "" if metrics is None else str(metrics["notes"]),
                "simulation_sqv_interval_median": (
                    ""
                    if simulation_metrics is None
                    or simulation_metrics["sqv_interval_median"] is None
                    else float(simulation_metrics["sqv_interval_median"])
                ),
                "simulation_hourly_sqv_median": (
                    ""
                    if simulation_metrics is None
                    or simulation_metrics["hourly_sqv_median"] is None
                    else float(simulation_metrics["hourly_sqv_median"])
                ),
                "simulation_wape_5min": (
                    ""
                    if simulation_metrics is None
                    or simulation_metrics["wape_5min"] is None
                    else float(simulation_metrics["wape_5min"])
                ),
                "simulation_hourly_wape_median": (
                    ""
                    if simulation_metrics is None
                    or simulation_metrics["hourly_wape_median"] is None
                    else float(simulation_metrics["hourly_wape_median"])
                ),
                "simulation_daily_total_error": (
                    ""
                    if simulation_metrics is None
                    or simulation_metrics["daily_total_error"] is None
                    else float(simulation_metrics["daily_total_error"])
                ),
                "simulation_notes": (
                    ""
                    if simulation_metrics is None
                    else str(simulation_metrics["notes"])
                ),
                "simulation_input_sqv_interval_median": (
                    ""
                    if simulation_input_metrics is None
                    or simulation_input_metrics["sqv_interval_median"] is None
                    else float(simulation_input_metrics["sqv_interval_median"])
                ),
                "simulation_input_hourly_sqv_median": (
                    ""
                    if simulation_input_metrics is None
                    or simulation_input_metrics["hourly_sqv_median"] is None
                    else float(simulation_input_metrics["hourly_sqv_median"])
                ),
                "simulation_input_wape_5min": (
                    ""
                    if simulation_input_metrics is None
                    or simulation_input_metrics["wape_5min"] is None
                    else float(simulation_input_metrics["wape_5min"])
                ),
                "simulation_input_hourly_wape_median": (
                    ""
                    if simulation_input_metrics is None
                    or simulation_input_metrics["hourly_wape_median"] is None
                    else float(simulation_input_metrics["hourly_wape_median"])
                ),
                "simulation_input_daily_total_error": (
                    ""
                    if simulation_input_metrics is None
                    or simulation_input_metrics["daily_total_error"] is None
                    else float(simulation_input_metrics["daily_total_error"])
                ),
                "simulation_input_notes": (
                    ""
                    if simulation_input_metrics is None
                    else str(simulation_input_metrics["notes"])
                ),
            }
        )
        base.render_plot(
            detector_id,
            vehicle_counts,
            detector_input_counts.get(detector_id),
            simulation_counts_for_plot,
            simulation_modeled_total,
            simulation_input_total,
            simulation_comparison_label,
            output_path,
            args.interval_seconds,
        )

    speed_plot_count = 0
    speed_output_dir = output_dir / "speed_plots"
    speed_output_dir.mkdir(parents=True, exist_ok=True)
    for detector_group, lane_pairs in base.SPEED_DETECTOR_MAPPING.items():
        for input_lane_id, simulation_detector_id in lane_pairs:
            output_path = (
                speed_output_dir
                / f"{detector_group}_{input_lane_id}_speed.{args.format}"
            )
            created_plot, speed_mae = base.render_speed_plot(
                detector_group,
                input_lane_id,
                simulation_detector_id,
                speed_input_profiles.get(input_lane_id),
                simulation_speed_series.get(simulation_detector_id),
                output_path,
                args.interval_seconds,
            )
            if created_plot:
                speed_plot_count += 1
                metrics_rows.append(
                    {
                        "detector_id": f"{detector_group} / {input_lane_id}",
                        "edge": simulation_detector_id,
                        "sqv_interval_median": "",
                        "hourly_sqv_median": "",
                        "wape_5min": "",
                        "hourly_wape_median": "",
                        "daily_total_error": "",
                        "speed_mae_workday": (
                            "" if speed_mae is None else float(speed_mae)
                        ),
                        "notes": "Validation speed lane plot",
                        "simulation_sqv_interval_median": "",
                        "simulation_hourly_sqv_median": "",
                        "simulation_wape_5min": "",
                        "simulation_hourly_wape_median": "",
                        "simulation_daily_total_error": "",
                        "simulation_notes": "",
                        "simulation_input_sqv_interval_median": "",
                        "simulation_input_hourly_sqv_median": "",
                        "simulation_input_wape_5min": "",
                        "simulation_input_hourly_wape_median": "",
                        "simulation_input_daily_total_error": "",
                        "simulation_input_notes": "",
                    }
                )

    metrics_csv_path = output_dir / "plot_metrics.csv"
    with metrics_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "detector_id",
                "edge",
                "sqv_interval_median",
                "hourly_sqv_median",
                "wape_5min",
                "hourly_wape_median",
                "daily_total_error",
                "speed_mae_workday",
                "notes",
                "simulation_sqv_interval_median",
                "simulation_hourly_sqv_median",
                "simulation_wape_5min",
                "simulation_hourly_wape_median",
                "simulation_daily_total_error",
                "simulation_notes",
                "simulation_input_sqv_interval_median",
                "simulation_input_hourly_sqv_median",
                "simulation_input_wape_5min",
                "simulation_input_hourly_wape_median",
                "simulation_input_daily_total_error",
                "simulation_input_notes",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics_rows)

    print(
        f"Wrote {len(counts)} validation traffic plot(s) using "
        f"{len(simulation_output_paths)} simulation run(s) to {output_dir}"
    )
    print(f"Wrote {speed_plot_count} validation speed plot(s) to {speed_output_dir}")
    if missing_speed_lanes:
        print(
            "Skipped missing speed input lane(s): "
            + ", ".join(sorted(missing_speed_lanes))
        )
    print(f"Wrote validation metrics CSV to {metrics_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
