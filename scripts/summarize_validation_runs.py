#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import tempfile

PLOT_CACHE_DIR = Path(tempfile.gettempdir()) / "sumo_plot_cache"
PLOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PLOT_CACHE_DIR / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUN_LOG_PATTERN = re.compile(r"sumo(\d+)\.log$")
REPLICATION_DIR_PATTERN = re.compile(r"rep_(\d+)_seed_(\d+)$")
STATISTICS_HEADER_PATTERN = re.compile(r"^Statistics \(avg of \d+\):$")


@dataclass(frozen=True)
class MetricDefinition:
    key: str
    section: str
    field: str
    title: str
    color: str
    decimals: int


METRICS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        key="performance_duration_s",
        section="Performance",
        field="Duration",
        title="Performance: Duration [s]",
        color="#2563eb",
        decimals=2,
    ),
    MetricDefinition(
        key="performance_real_time_factor",
        section="Performance",
        field="Real time factor",
        title="Performance: Real time factor",
        color="#0891b2",
        decimals=4,
    ),
    MetricDefinition(
        key="performance_ups",
        section="Performance",
        field="UPS",
        title="Performance: UPS",
        color="#0f766e",
        decimals=0,
    ),
    MetricDefinition(
        key="vehicles_inserted",
        section="Vehicles",
        field="Inserted",
        title="Vehicles: Inserted",
        color="#16a34a",
        decimals=0,
    ),
    MetricDefinition(
        key="vehicles_running",
        section="Vehicles",
        field="Running",
        title="Vehicles: Running",
        color="#65a30d",
        decimals=0,
    ),
    MetricDefinition(
        key="vehicles_waiting",
        section="Vehicles",
        field="Waiting",
        title="Vehicles: Waiting",
        color="#ca8a04",
        decimals=0,
    ),
    MetricDefinition(
        key="vehicles_emergency_braking",
        section="Vehicles",
        field="Emergency Braking",
        title="Vehicles: Emergency braking",
        color="#dc2626",
        decimals=0,
    ),
    MetricDefinition(
        key="statistics_route_length",
        section="Statistics",
        field="RouteLength",
        title="Statistics: RouteLength",
        color="#7c3aed",
        decimals=2,
    ),
    MetricDefinition(
        key="statistics_speed",
        section="Statistics",
        field="Speed",
        title="Statistics: Speed",
        color="#9333ea",
        decimals=2,
    ),
    MetricDefinition(
        key="statistics_duration",
        section="Statistics",
        field="Duration",
        title="Statistics: Duration",
        color="#c026d3",
        decimals=2,
    ),
    MetricDefinition(
        key="statistics_waiting_time",
        section="Statistics",
        field="WaitingTime",
        title="Statistics: WaitingTime",
        color="#ea580c",
        decimals=2,
    ),
    MetricDefinition(
        key="statistics_time_loss",
        section="Statistics",
        field="TimeLoss",
        title="Statistics: TimeLoss",
        color="#f97316",
        decimals=2,
    ),
    MetricDefinition(
        key="statistics_depart_delay",
        section="Statistics",
        field="DepartDelay",
        title="Statistics: DepartDelay",
        color="#4b5563",
        decimals=2,
    ),
)

METRIC_LOOKUP = {
    (metric.section, metric.field): metric for metric in METRICS
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize 20 SUMO validation runs from sumo*.log files and create plots."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="output sim/val1",
        help="Directory containing sumo1.log ... sumo20.log.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Directory for summary CSV files and plots. Defaults to a "
            "validation_summary folder inside --input-dir."
        ),
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=("png", "pdf", "svg"),
        help="Image format for the validation summary plot.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def output_dir_for(input_dir: Path, output_dir_arg: str | None) -> Path:
    if output_dir_arg:
        return resolve_path(output_dir_arg)
    return input_dir / "validation_summary"


def discover_log_paths(input_dir: Path) -> list[tuple[int, Path]]:
    log_paths: list[tuple[int, Path]] = []
    for path in input_dir.glob("sumo*.log"):
        match = RUN_LOG_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        log_paths.append((int(match.group(1)), path))

    if log_paths:
        log_paths.sort(key=lambda item: item[0])
        return log_paths

    for path in input_dir.glob("rep_*_seed_*/sumo.log"):
        match = REPLICATION_DIR_PATTERN.fullmatch(path.parent.name)
        if match is None:
            continue
        log_paths.append((int(match.group(1)), path))

    log_paths.sort(key=lambda item: item[0])
    if not log_paths:
        raise FileNotFoundError(f"No validation sumo logs found in {input_dir}")
    return log_paths


def parse_numeric_value(raw_value: str) -> float:
    normalized = raw_value.strip()
    if normalized.endswith("s"):
        normalized = normalized[:-1]
    return float(normalized)


def parse_log(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    current_section: str | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped in {"Performance:", "Vehicles:"}:
            current_section = stripped[:-1]
            continue
        if STATISTICS_HEADER_PATTERN.match(stripped):
            current_section = "Statistics"
            continue
        if current_section is None or ":" not in stripped:
            continue

        field, raw_value = stripped.split(":", 1)
        metric = METRIC_LOOKUP.get((current_section, field.strip()))
        if metric is None:
            continue
        values[metric.key] = parse_numeric_value(raw_value)

    missing = [metric.key for metric in METRICS if metric.key not in values]
    if missing:
        raise ValueError(f"{path} is missing expected metric(s): {', '.join(missing)}")
    return values


def format_value(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def write_raw_runs_csv(
    output_path: Path,
    runs: list[tuple[int, dict[str, float]]],
    seeds_by_run: dict[int, int],
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["run", "seed", *[metric.key for metric in METRICS]],
        )
        writer.writeheader()
        for run_number, values in runs:
            row: dict[str, int | float | str] = {
                "run": run_number,
                "seed": seeds_by_run.get(run_number, ""),
            }
            for metric in METRICS:
                row[metric.key] = values[metric.key]
            writer.writerow(row)


def load_seed_mapping(input_dir: Path) -> dict[int, int]:
    seeds_path = input_dir / "validation_run_seeds.csv"
    if not seeds_path.exists():
        seeds_by_run: dict[int, int] = {}
        for path in input_dir.glob("rep_*_seed_*"):
            if not path.is_dir():
                continue
            match = REPLICATION_DIR_PATTERN.fullmatch(path.name)
            if match is None:
                continue
            seeds_by_run[int(match.group(1))] = int(match.group(2))
        return seeds_by_run

    seeds_by_run: dict[int, int] = {}
    with seeds_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            run_raw = row.get("run")
            seed_raw = row.get("seed")
            if run_raw is None or seed_raw is None:
                continue
            seeds_by_run[int(run_raw)] = int(seed_raw)
    return seeds_by_run


def write_summary_csv(
    output_path: Path,
    runs: list[tuple[int, dict[str, float]]],
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "metric_key",
                "metric_title",
                "median",
                "stddev",
                "mean",
                "min",
                "max",
            ],
        )
        writer.writeheader()
        for metric in METRICS:
            values = np.array([run_values[metric.key] for _, run_values in runs])
            stddev = sample_stddev(values)
            writer.writerow(
                {
                    "metric_key": metric.key,
                    "metric_title": metric.title,
                    "median": format_value(float(np.median(values)), metric.decimals),
                    "stddev": format_value(stddev, metric.decimals),
                    "mean": format_value(float(np.mean(values)), metric.decimals),
                    "min": format_value(float(np.min(values)), metric.decimals),
                    "max": format_value(float(np.max(values)), metric.decimals),
                }
            )


def sample_stddev(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    return float(np.std(values, ddof=1))


def plot_metrics(
    output_path: Path,
    runs: list[tuple[int, dict[str, float]]],
) -> None:
    run_numbers = np.array([run_number for run_number, _ in runs], dtype=int)
    ncols = 3
    nrows = math.ceil(len(METRICS) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.5 * nrows))
    axes_array = np.atleast_1d(axes).ravel()

    for ax, metric in zip(axes_array, METRICS):
        values = np.array([run_values[metric.key] for _, run_values in runs], dtype=float)
        median_value = float(np.median(values))
        stddev_value = sample_stddev(values)

        ax.plot(
            run_numbers,
            values,
            marker="o",
            linewidth=1.8,
            markersize=5,
            color=metric.color,
        )
        ax.axhline(
            median_value,
            color="#111827",
            linestyle="--",
            linewidth=1.5,
            label=f"Median {format_value(median_value, metric.decimals)}",
        )
        ax.set_title(metric.title, fontsize=11)
        ax.set_xlabel("Run")
        ax.set_ylabel("Value")
        ax.set_xticks(run_numbers)
        ax.grid(color="#d1d5db", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc="best", frameon=False, fontsize=8)
        ax.text(
            0.02,
            0.98,
            "\n".join(
                [
                    f"Median: {format_value(median_value, metric.decimals)}",
                    f"Std dev: {format_value(stddev_value, metric.decimals)}",
                ]
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
        )

    for ax in axes_array[len(METRICS) :]:
        ax.set_visible(False)

    fig.suptitle("SUMO Validation Runs: all 20 results and median", fontsize=16, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    output_dir = output_dir_for(input_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_paths = discover_log_paths(input_dir)
    runs = [(run_number, parse_log(path)) for run_number, path in log_paths]
    seeds_by_run = load_seed_mapping(input_dir)

    raw_csv_path = output_dir / "validation_run_metrics.csv"
    summary_csv_path = output_dir / "validation_run_summary.csv"
    plot_path = output_dir / f"validation_run_summary.{args.format}"

    write_raw_runs_csv(raw_csv_path, runs, seeds_by_run)
    write_summary_csv(summary_csv_path, runs)
    plot_metrics(plot_path, runs)

    print(f"Parsed {len(runs)} SUMO log file(s) from {input_dir}")
    print(f"Wrote raw run metrics to {raw_csv_path}")
    print(f"Wrote median/std-dev summary to {summary_csv_path}")
    print(f"Wrote validation summary plot to {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
