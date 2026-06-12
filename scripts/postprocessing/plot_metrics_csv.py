#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np


SPEED_LANE_LABELS = {
    "vd1": "(R)",
    "vd2": "(L)",
    "vd3": "(L)",
    "vd4": "(R)",
}

METRIC_KEYS = (
    "sqv_interval_median",
    "hourly_sqv_median",
    "wape_5min",
    "hourly_wape_median",
    "daily_total_error",
    "speed_mae_workday",
)
MetricKey = str


@dataclass(frozen=True)
class MetricConfig:
    key: MetricKey
    title: str
    filename: str
    color: str
    ascending: bool
    formatter: Callable[[float], str]
    xlabel: str


@dataclass
class RowData:
    detector_id: str
    edge: str
    notes: str
    metrics: dict[MetricKey, float]
    simulation_notes: str
    simulation_metrics: dict[MetricKey, float]
    simulation_input_notes: str
    simulation_input_metrics: dict[MetricKey, float]


METRICS: list[MetricConfig] = [
    MetricConfig(
        key="sqv_interval_median",
        title="5-Minute SQV Median",
        filename="sqv_interval_median.png",
        color="#0f766e",
        ascending=False,
        formatter=lambda value: f"{value:.2f}",
        xlabel="SQV",
    ),
    MetricConfig(
        key="hourly_sqv_median",
        title="Hourly SQV Median",
        filename="hourly_sqv_median.png",
        color="#0891b2",
        ascending=False,
        formatter=lambda value: f"{value:.2f}",
        xlabel="SQV",
    ),
    MetricConfig(
        key="wape_5min",
        title="5-Minute WAPE",
        filename="wape_5min.png",
        color="#d97706",
        ascending=False,
        formatter=lambda value: f"{value * 100:.1f}%",
        xlabel="WAPE",
    ),
    MetricConfig(
        key="hourly_wape_median",
        title="Median Hourly WAPE",
        filename="hourly_wape_median.png",
        color="#ea580c",
        ascending=False,
        formatter=lambda value: f"{value * 100:.1f}%",
        xlabel="Median Hourly WAPE",
    ),
    MetricConfig(
        key="daily_total_error",
        title="Daily Total Error",
        filename="daily_total_error.png",
        color="#7c3aed",
        ascending=False,
        formatter=lambda value: f"{value * 100:.1f}%",
        xlabel="Relative Error",
    ),
    MetricConfig(
        key="speed_mae_workday",
        title="Speed MAE vs Workday Average",
        filename="speed_mae_workday.png",
        color="#111827",
        ascending=False,
        formatter=lambda value: f"{value:.2f} km/h",
        xlabel="MAE [km/h]",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create plots from plot_metrics.csv exported by plot_input_sim_traffic.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="start_edge_traffic_plots/metrics_summary_plots",
        help="Directory for generated metric summary plots.",
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
        default="start_edge_traffic_plots/plot_metrics.csv",
        help="Input metrics CSV exported by plot_input_sim_traffic.py.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def csv_value(row: dict[str, str | None], key: str) -> str:
    value = row.get(key)
    return "" if value is None else value


def load_rows(path: Path) -> list[RowData]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows: list[RowData] = []
        for row in reader:
            metrics: dict[MetricKey, float] = {}
            simulation_metrics: dict[MetricKey, float] = {}
            simulation_input_metrics: dict[MetricKey, float] = {}
            for metric in METRICS:
                raw_value = csv_value(row, metric.key)
                metrics[metric.key] = float(raw_value) if raw_value else np.nan
                simulation_key = f"simulation_{metric.key}"
                raw_simulation_value = csv_value(row, simulation_key)
                simulation_metrics[metric.key] = (
                    float(raw_simulation_value) if raw_simulation_value else np.nan
                )
                simulation_input_key = f"simulation_input_{metric.key}"
                raw_simulation_input_value = csv_value(row, simulation_input_key)
                simulation_input_metrics[metric.key] = (
                    float(raw_simulation_input_value)
                    if raw_simulation_input_value
                    else np.nan
                )
            rows.append(
                RowData(
                    detector_id=csv_value(row, "detector_id"),
                    edge=csv_value(row, "edge"),
                    notes=csv_value(row, "notes"),
                    metrics=metrics,
                    simulation_notes=csv_value(row, "simulation_notes"),
                    simulation_metrics=simulation_metrics,
                    simulation_input_notes=csv_value(row, "simulation_input_notes"),
                    simulation_input_metrics=simulation_input_metrics,
                )
            )
    return rows


def detector_label(row: RowData) -> str:
    label = row.detector_id
    if " / " in label:
        detector_group, speed_lane = label.split(" / ", maxsplit=1)
        vd_match = re.search(r"(vd\d+)", speed_lane)
        label = (
            f"{detector_group}_{SPEED_LANE_LABELS.get(vd_match.group(1), vd_match.group(1))}"
            if vd_match is not None
            else detector_group
        )
    if label.endswith("_SP"):
        label = label[:-3]
    return label.replace("_", " ")


def metric_sort_value(metric_key: MetricKey, value: float) -> float:
    if np.isnan(value):
        return -np.inf
    if metric_key == "daily_total_error":
        return abs(value)
    return value


def metrics_for_source(row: RowData, source: str) -> dict[MetricKey, float]:
    if source == "simulation":
        return row.simulation_metrics
    if source == "simulation_input":
        return row.simulation_input_metrics
    return row.metrics


def metric_title(metric: MetricConfig, source_label: str) -> str:
    if source_label == "Simulation vs Input" and metric.key == "sqv_interval_median":
        return "Input Flows vs. Simulation: 5min. SQV"
    return f"{source_label}: {metric.title}"


def metric_filename(metric: MetricConfig, source: str) -> str:
    if source == "simulation":
        return f"simulation_{metric.filename}"
    if source == "simulation_input":
        return f"simulation_input_{metric.filename}"
    return metric.filename


def dashboard_filename(source: str) -> str:
    if source == "simulation":
        return "simulation_metrics_dashboard.png"
    if source == "simulation_input":
        return "simulation_input_metrics_dashboard.png"
    return "metrics_dashboard.png"


def plot_metric(
    ax: Axes,
    rows: list[RowData],
    metric: MetricConfig,
    top_n: int,
    source: str,
    source_label: str,
) -> None:
    metric_key = metric.key
    valid_rows = [
        row for row in rows if not np.isnan(metrics_for_source(row, source)[metric_key])
    ]
    valid_rows.sort(
        key=lambda row: metric_sort_value(
            metric_key,
            metrics_for_source(row, source)[metric_key],
        ),
        reverse=True,
    )
    selected = valid_rows[:top_n]

    labels = [detector_label(row) for row in selected]
    values = np.array(
        [metrics_for_source(row, source)[metric_key] for row in selected],
        dtype=float,
    )
    y = np.arange(len(selected))

    colors = [metric.color] * len(selected)
    if metric_key == "daily_total_error":
        colors = ["#dc2626" if value < 0 else "#059669" for value in values]

    is_speed_metric = metric_key == "speed_mae_workday"
    label_fontsize = 11 if is_speed_metric else 10
    value_fontsize = 11 if is_speed_metric else 10
    axis_fontsize = 13
    tick_fontsize = 11

    ax.barh(y, values, color=colors, alpha=0.9)
    if is_speed_metric:
        ax.margins(y=0.18)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=label_fontsize)
    ax.invert_yaxis()
    if not is_speed_metric:
        ax.set_title(metric_title(metric, source_label), fontsize=16, pad=12)
    ax.set_xlabel(metric.xlabel, fontsize=axis_fontsize)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.grid(axis="x", color="#d1d5db", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if metric_key in {"sqv_interval_median", "hourly_sqv_median"}:
        ax.set_xlim(0.8, 1.0)
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    elif metric_key in {"wape_5min", "hourly_wape_median"}:
        ax.set_xlim(0, 0.5)
    elif metric_key == "daily_total_error":
        ax.set_xlim(-0.5, 0.5)
        ax.axvline(0, color="#111827", linewidth=1.0)
    else:
        ax.set_xlim(0, max(0.05, float(np.max(values)) * 1.15) if len(values) else 0.05)

    all_values = np.array(
        [metrics_for_source(row, source)[metric_key] for row in valid_rows],
        dtype=float,
    )
    median_value = float(np.median(all_values)) if len(all_values) else float("nan")
    if not np.isnan(median_value):
        ax.axvline(
            median_value,
            color="#111827",
            linewidth=1.4,
            linestyle=(0, (3, 3)),
            alpha=0.9,
        )
        ax.text(
            median_value,
            0.02,
            f"Median: {metric.formatter(median_value)}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=value_fontsize,
            color="#111827",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#d1d5db",
                "alpha": 0.9,
            },
        )

    formatter = metric.formatter
    for idx, value in enumerate(values):
        text = formatter(value)
        if metric_key == "daily_total_error":
            offset = 0.01 * max(1.0, ax.get_xlim()[1] - ax.get_xlim()[0])
            x = value + offset if value >= 0 else value - offset
            ha = "left" if value >= 0 else "right"
        else:
            offset = 0.01 * max(1.0, ax.get_xlim()[1])
            x = value + offset
            ha = "left"
        ax.text(x, idx, text, va="center", ha=ha, fontsize=value_fontsize, color="#111827")


def save_single_metric_plots(
    rows: list[RowData],
    output_dir: Path,
    top_n: int,
    source: str,
    source_label: str,
) -> None:
    for metric in METRICS:
        if metric.key == "speed_mae_workday":
            valid_count = sum(
                not np.isnan(metrics_for_source(row, source)[metric.key])
                for row in rows
            )
            fig_height = max(5.2, min(top_n, valid_count) * 0.5)
        else:
            fig_height = max(8, top_n * 0.34)
        fig, ax = plt.subplots(figsize=(6, fig_height))
        plot_metric(ax, rows, metric, top_n, source, source_label)
        fig.tight_layout()
        fig.savefig(output_dir / metric_filename(metric, source), bbox_inches="tight")
        plt.close(fig)


def save_dashboard(
    rows: list[RowData],
    output_dir: Path,
    top_n: int,
    source: str,
    source_label: str,
) -> None:
    num_metrics = len(METRICS)
    num_cols = 2
    num_rows = math.ceil(num_metrics / num_cols)
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(12, max(14, num_rows * max(5.5, top_n * 0.21))),
        constrained_layout=True,
    )
    axes_list = list(np.atleast_1d(axes).flat)
    for ax, metric in zip(axes_list, METRICS):
        plot_metric(ax, rows, metric, top_n, source, source_label)
    for ax in axes_list[num_metrics:]:
        ax.remove()
    fig.savefig(output_dir / dashboard_filename(source), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    metrics_csv = resolve_path(args.metrics_csv)
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {metrics_csv}")

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(metrics_csv)
    if not rows:
        raise ValueError(f"No rows found in {metrics_csv}")

    top_n = max(1, min(args.top_n, len(rows)))
    metric_sources = [
        ("input", "Input"),
        ("simulation", "Simulation"),
        ("simulation_input", "Simulation vs Input"),
    ]
    for source, source_label in metric_sources:
        save_single_metric_plots(rows, output_dir, top_n, source, source_label)
        save_dashboard(rows, output_dir, top_n, source, source_label)

    print(f"Wrote metric plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
