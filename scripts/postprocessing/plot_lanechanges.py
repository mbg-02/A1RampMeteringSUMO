#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
from typing import DefaultDict
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

RgbaColor = tuple[float, float, float, float]
PlotColor = str | RgbaColor

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
DEFAULT_INPUT_XML = "output sim/17/lanechanges.xml"
DEFAULT_INTERVAL_SECONDS = 300
DEFAULT_DAY_SECONDS = 24 * 60 * 60
MPS_TO_KMPH = 3.6

REASON_COLORS = {
    "strategic": "#2563eb",
    "cooperative": "#d97706",
    "speedGain": "#16a34a",
    "keepRight": "#7c3aed",
    "sublane": "#dc2626",
    "traci": "#0f766e",
    "strategic|urgent": "#1d4ed8",
    "cooperative|urgent": "#b45309",
    "urgent": "#b91c1c",
    "not urgent": "#475569",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create lane-change analysis plots from SUMO lanechanges.xml output.",
    )
    parser.add_argument(
        "input_xml",
        nargs="?",
        default=DEFAULT_INPUT_XML,
        help="Path to the SUMO lanechange-output XML file.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for generated plots. Defaults to a lanechange_plots folder next to the input XML.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=DEFAULT_INTERVAL_SECONDS,
        help="Aggregation bin size in seconds for the time-series plot.",
    )
    parser.add_argument(
        "--day-seconds",
        type=int,
        default=DEFAULT_DAY_SECONDS,
        help="Minimum x-axis span in seconds for the time-series plot.",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=("png", "pdf", "svg"),
        help="Image format for generated plots.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def parse_optional_float(value: str | None) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def primary_reason(reason: str) -> str:
    return reason.split("|", 1)[0]


def urgency_bucket(reason: str) -> str:
    return "urgent" if "urgent" in reason.split("|") else "not urgent"


def pretty_reason_label(reason: str) -> str:
    if "|" in reason:
        return reason.replace("|", " + ")
    return reason


def human_time_label(seconds: float) -> str:
    total_minutes = int(round(seconds / 60))
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}"


def histogram_bins(values: np.ndarray, minimum: int = 20, maximum: int = 80) -> int:
    if values.size < minimum:
        return minimum
    candidate = int(np.sqrt(values.size))
    return max(minimum, min(maximum, candidate))


def leader_hexbin_gridsize(num_points: int) -> int:
    if num_points <= 0:
        return 20
    return max(20, min(60, int(math.sqrt(num_points) / 2)))


def collect_lanechange_data(
    input_xml: Path,
    interval_seconds: int,
    day_seconds: int,
) -> tuple[
    dict[str, Counter[int]],
    dict[str, Counter[int]],
    dict[str, Counter[int]],
    int,
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    int,
]:
    primary_reason_counts: DefaultDict[str, Counter[int]] = defaultdict(Counter)
    full_reason_counts: DefaultDict[str, Counter[int]] = defaultdict(Counter)
    urgency_counts: DefaultDict[str, Counter[int]] = defaultdict(Counter)
    max_bin_index = max(0, math.ceil(day_seconds / interval_seconds) - 1)
    change_count = 0
    speeds_kmh: list[float] = []
    positions_m: list[float] = []
    leader_gaps_m: list[float] = []
    leader_speeds_kmh: list[float] = []
    paired_gap_m: list[float] = []
    paired_speed_kmh: list[float] = []

    for _, element in ET.iterparse(input_xml, events=("end",)):
        if element.tag != "change":
            continue

        change_count += 1
        attributes = element.attrib

        time_seconds = parse_optional_float(attributes.get("time"))
        if time_seconds is not None:
            bin_index = int(time_seconds // interval_seconds)
            max_bin_index = max(max_bin_index, bin_index)
        else:
            bin_index = 0

        reason = attributes.get("reason", "unknown")
        primary_reason_counts[primary_reason(reason)][bin_index] += 1
        full_reason_counts[reason][bin_index] += 1
        urgency_counts[urgency_bucket(reason)][bin_index] += 1

        speed_mps = parse_optional_float(attributes.get("speed"))
        if speed_mps is not None:
            speeds_kmh.append(speed_mps * MPS_TO_KMPH)

        position = parse_optional_float(attributes.get("pos"))
        if position is not None:
            positions_m.append(position)

        leader_gap = parse_optional_float(attributes.get("leaderGap"))
        leader_speed_mps = parse_optional_float(attributes.get("leaderSpeed"))
        if leader_gap is not None:
            leader_gaps_m.append(leader_gap)
        if leader_speed_mps is not None:
            leader_speeds_kmh.append(leader_speed_mps * MPS_TO_KMPH)
        if leader_gap is not None and leader_speed_mps is not None:
            paired_gap_m.append(leader_gap)
            paired_speed_kmh.append(leader_speed_mps * MPS_TO_KMPH)

        element.clear()

    return (
        dict(primary_reason_counts),
        dict(full_reason_counts),
        dict(urgency_counts),
        max_bin_index + 1,
        speeds_kmh,
        positions_m,
        leader_gaps_m,
        leader_speeds_kmh,
        paired_gap_m,
        paired_speed_kmh,
        change_count,
    )


def build_reason_matrix(
    reason_counts: dict[str, Counter[int]],
    num_bins: int,
) -> tuple[list[str], np.ndarray]:
    sorted_reasons = sorted(
        reason_counts,
        key=lambda reason: sum(reason_counts[reason].values()),
        reverse=True,
    )
    matrix = np.zeros((len(sorted_reasons), num_bins), dtype=int)
    for row_index, reason in enumerate(sorted_reasons):
        for bin_index, count in reason_counts[reason].items():
            if 0 <= bin_index < num_bins:
                matrix[row_index, bin_index] = count
    return sorted_reasons, matrix


def reason_color(reason: str, index: int) -> PlotColor:
    if reason in REASON_COLORS:
        return REASON_COLORS[reason]
    cmap = plt.get_cmap("tab20")
    return cmap(index % cmap.N)


def plot_reason_counts(
    reasons: list[str],
    matrix: np.ndarray,
    interval_seconds: int,
    output_path: Path,
    title: str,
    legend_title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(matrix.shape[1])
    bottom = np.zeros(matrix.shape[1], dtype=int)

    for index, reason in enumerate(reasons):
        counts = matrix[index]
        ax.bar(
            x,
            counts,
            bottom=bottom,
            width=1.0,
            color=reason_color(reason, index),
            edgecolor="none",
            label=f"{pretty_reason_label(reason)} ({int(counts.sum()):,})",
        )
        bottom += counts

    tick_step = max(1, int((2 * 60 * 60) / interval_seconds))
    tick_indices = np.arange(0, matrix.shape[1], tick_step)
    tick_labels = [human_time_label(index * interval_seconds) for index in tick_indices]

    ax.set_title(title)
    ax.set_xlabel("Time of day")
    ax.set_ylabel(f"Lane changes per {interval_seconds // 60:.0f} min")
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.grid(axis="y", color="#d1d5db", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(title=legend_title, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def add_histogram_stats(ax: Axes, values: np.ndarray, color: str) -> None:
    mean_value = float(np.mean(values))
    median_value = float(np.median(values))
    ax.axvline(
        mean_value,
        color=color,
        linestyle="--",
        linewidth=1.5,
        label=f"Mean {mean_value:.1f}",
    )
    ax.axvline(
        median_value,
        color="#111827",
        linestyle=":",
        linewidth=1.5,
        label=f"Median {median_value:.1f}",
    )
    ax.legend(loc="upper right")


def plot_histogram(
    values: np.ndarray,
    title: str,
    xlabel: str,
    output_path: Path,
    color: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        values, bins=histogram_bins(values), color=color, edgecolor="white", alpha=0.9
    )
    add_histogram_stats(ax, values, color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Lane changes")
    ax.grid(axis="y", color="#d1d5db", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_leader_context(
    leader_gaps_m: np.ndarray,
    leader_speeds_kmh: np.ndarray,
    paired_gaps_m: np.ndarray,
    paired_speeds_kmh: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].hist(
        leader_gaps_m,
        bins=histogram_bins(leader_gaps_m),
        color="#0891b2",
        edgecolor="white",
        alpha=0.9,
    )
    add_histogram_stats(axes[0], leader_gaps_m, "#0891b2")
    axes[0].set_title("Leader Gap at Lane Change")
    axes[0].set_xlabel("Leader gap [m]")
    axes[0].set_ylabel("Lane changes")

    axes[1].hist(
        leader_speeds_kmh,
        bins=histogram_bins(leader_speeds_kmh),
        color="#7c3aed",
        edgecolor="white",
        alpha=0.9,
    )
    add_histogram_stats(axes[1], leader_speeds_kmh, "#7c3aed")
    axes[1].set_title("Leader Speed at Lane Change")
    axes[1].set_xlabel("Leader speed [km/h]")
    axes[1].set_ylabel("Lane changes")

    if paired_gaps_m.size > 0 and paired_speeds_kmh.size > 0:
        hb = axes[2].hexbin(
            paired_gaps_m,
            paired_speeds_kmh,
            gridsize=leader_hexbin_gridsize(paired_gaps_m.size),
            cmap="viridis",
            mincnt=1,
        )
        fig.colorbar(hb, ax=axes[2], label="Lane changes")
    else:
        axes[2].text(
            0.5,
            0.5,
            "No paired leader gap/speed values available",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
    axes[2].set_title("Leader Gap vs Leader Speed")
    axes[2].set_xlabel("Leader gap [m]")
    axes[2].set_ylabel("Leader speed [km/h]")

    for ax in axes:
        ax.grid(color="#d1d5db", linewidth=0.8, alpha=0.5)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def ensure_non_empty(values: list[float], label: str) -> np.ndarray:
    if not values:
        raise ValueError(f"No valid {label} values found in the input file.")
    return np.asarray(values, dtype=float)


def output_dir_for(input_xml: Path, output_dir_arg: str | None) -> Path:
    if output_dir_arg:
        return resolve_path(output_dir_arg)
    return input_xml.parent / "lanechange_plots"


def stats_summary(values: np.ndarray, unit: str) -> str:
    mean_value = float(np.mean(values))
    median_value = float(np.median(values))
    p90_value = float(np.percentile(values, 90))
    return (
        f"count={values.size:,}, mean={mean_value:.1f}{unit}, "
        f"median={median_value:.1f}{unit}, p90={p90_value:.1f}{unit}"
    )


def main() -> int:
    args = parse_args()
    input_xml = resolve_path(args.input_xml)
    output_dir = output_dir_for(input_xml, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        primary_reason_counts,
        reason_counts,
        urgency_counts,
        num_bins,
        speeds_kmh,
        positions_m,
        leader_gaps_m,
        leader_speeds_kmh,
        paired_gap_m,
        paired_speed_kmh,
        change_count,
    ) = collect_lanechange_data(
        input_xml=input_xml,
        interval_seconds=args.interval_seconds,
        day_seconds=args.day_seconds,
    )

    primary_reasons, primary_matrix = build_reason_matrix(
        primary_reason_counts, num_bins
    )
    reasons, matrix = build_reason_matrix(reason_counts, num_bins)
    urgency_reasons, urgency_matrix = build_reason_matrix(urgency_counts, num_bins)

    speed_array = ensure_non_empty(speeds_kmh, "speed")
    position_array = ensure_non_empty(positions_m, "position")
    leader_gap_array = ensure_non_empty(leader_gaps_m, "leader gap")
    leader_speed_array = ensure_non_empty(leader_speeds_kmh, "leader speed")

    paired_gap_array = np.asarray(paired_gap_m, dtype=float)
    paired_speed_array = np.asarray(paired_speed_kmh, dtype=float)

    extension = args.format
    plot_reason_counts(
        reasons,
        matrix,
        args.interval_seconds,
        output_dir / f"lanechanges_by_reason_5min.{extension}",
        title="Lane Changes Over the Day by Full Reason",
        legend_title="Full reason",
    )
    plot_reason_counts(
        primary_reasons,
        primary_matrix,
        args.interval_seconds,
        output_dir / f"lanechanges_by_primary_reason_5min.{extension}",
        title="Lane Changes Over the Day by Primary Reason",
        legend_title="Primary reason",
    )
    plot_reason_counts(
        urgency_reasons,
        urgency_matrix,
        args.interval_seconds,
        output_dir / f"lanechanges_by_urgency_5min.{extension}",
        title="Lane Changes Over the Day by Urgency",
        legend_title="Urgency",
    )
    plot_histogram(
        speed_array,
        title="Lane Change Speeds",
        xlabel="Speed [km/h]",
        output_path=output_dir / f"lanechange_speed_hist.{extension}",
        color="#2563eb",
    )
    plot_histogram(
        position_array,
        title="Lane Change Positions",
        xlabel="Position on lane [m]",
        output_path=output_dir / f"lanechange_position_hist.{extension}",
        color="#d97706",
    )
    plot_leader_context(
        leader_gap_array,
        leader_speed_array,
        paired_gap_array,
        paired_speed_array,
        output_dir / f"lanechange_leader_context.{extension}",
    )

    print(f"Parsed {change_count:,} lane changes from {input_xml}")
    for reason in reasons:
        print(
            f"  {pretty_reason_label(reason)}: {sum(reason_counts[reason].values()):,}"
        )
    for urgency in urgency_reasons:
        print(f"  {urgency}: {sum(urgency_counts[urgency].values()):,}")
    print(f"  lane-change speed: {stats_summary(speed_array, ' km/h')}")
    print(f"  lane-change position: {stats_summary(position_array, ' m')}")
    print(f"  leader gap: {stats_summary(leader_gap_array, ' m')}")
    print(f"  leader speed: {stats_summary(leader_speed_array, ' km/h')}")
    print(f"  paired leader gap/speed points: {paired_gap_array.size:,}")
    print(f"Saved plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
