#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_lanechanges as base


LANECHANGE_PATTERN = re.compile(r"lanechanges(\d+)\.xml$")
REPLICATION_DIR_PATTERN = re.compile(r"rep_(\d+)_seed_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create validation lane-change plots from lanechanges1.xml ..."
            " lanechanges20.xml."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="output sim/val1",
        help=(
            "Directory containing lanechanges1.xml ... lanechanges20.xml, or "
            "calibration-style rep_XX_seed_*/lanechanges.xml files."
        ),
    )
    parser.add_argument(
        "--pattern",
        default="lanechanges*.xml",
        help="Glob used to discover lane-change XML files.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Directory for generated plots. Defaults to a lanechange_plots folder "
            "inside the input directory."
        ),
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=base.DEFAULT_INTERVAL_SECONDS,
        help="Aggregation bin size in seconds for the time-series plots.",
    )
    parser.add_argument(
        "--day-seconds",
        type=int,
        default=base.DEFAULT_DAY_SECONDS,
        help="Minimum x-axis span in seconds for the time-series plots.",
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
    return Path.cwd() / path


def discover_lanechange_paths(input_dir: Path, pattern: str) -> list[Path]:
    matches: list[tuple[int, Path]] = []
    for path in input_dir.glob(pattern):
        match = LANECHANGE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        matches.append((int(match.group(1)), path))
    matches.sort(key=lambda item: item[0])
    if matches:
        return [path for _, path in matches]

    for path in input_dir.glob(f"rep_*_seed_*/{pattern}"):
        match = REPLICATION_DIR_PATTERN.fullmatch(path.parent.name)
        if match is None:
            continue
        matches.append((int(match.group(1)), path))

    matches.sort(key=lambda item: item[0])
    if not matches:
        raise FileNotFoundError(f"No lanechange XML files found in {input_dir}")
    return [path for _, path in matches]


def output_dir_for(input_dir: Path, output_dir_arg: str | None) -> Path:
    if output_dir_arg:
        return resolve_path(output_dir_arg)
    return input_dir / "lanechange_plots"


def aggregate_reason_median(
    per_run_reason_counts: list[dict[str, Counter[int]]],
    num_bins: int,
) -> tuple[list[str], np.ndarray]:
    reasons = sorted(
        {
            reason
            for reason_counts in per_run_reason_counts
            for reason in reason_counts
        },
        key=lambda reason: sum(
            sum(reason_counts.get(reason, Counter()).values())
            for reason_counts in per_run_reason_counts
        ),
        reverse=True,
    )

    if not reasons:
        return [], np.zeros((0, num_bins), dtype=float)

    per_run_matrices: list[np.ndarray] = []
    for reason_counts in per_run_reason_counts:
        matrix = np.zeros((len(reasons), num_bins), dtype=float)
        for row_index, reason in enumerate(reasons):
            for bin_index, count in reason_counts.get(reason, Counter()).items():
                if 0 <= bin_index < num_bins:
                    matrix[row_index, bin_index] = count
        per_run_matrices.append(matrix)

    stacked = np.stack(per_run_matrices, axis=0)
    return reasons, np.median(stacked, axis=0)


def plot_reason_counts_median(
    reasons: list[str],
    matrix: np.ndarray,
    interval_seconds: int,
    output_path: Path,
    title: str,
    legend_title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(matrix.shape[1])
    bottom = np.zeros(matrix.shape[1], dtype=float)

    for index, reason in enumerate(reasons):
        counts = matrix[index]
        ax.bar(
            x,
            counts,
            bottom=bottom,
            width=1.0,
            color=base.reason_color(reason, index),
            edgecolor="none",
            label=f"{base.pretty_reason_label(reason)} ({counts.sum():.1f})",
        )
        bottom += counts

    tick_step = max(1, int((2 * 60 * 60) / interval_seconds))
    tick_indices = np.arange(0, matrix.shape[1], tick_step)
    tick_labels = [base.human_time_label(index * interval_seconds) for index in tick_indices]

    ax.set_title(title)
    ax.set_xlabel("Time of day")
    ax.set_ylabel(f"Median lane changes per {interval_seconds // 60:.0f} min")
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.grid(axis="y", color="#d1d5db", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(title=legend_title, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    input_paths = discover_lanechange_paths(input_dir, args.pattern)
    output_dir = output_dir_for(input_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_run_primary_reason_counts: list[dict[str, Counter[int]]] = []
    per_run_reason_counts: list[dict[str, Counter[int]]] = []
    per_run_urgency_counts: list[dict[str, Counter[int]]] = []
    max_num_bins = max(1, math.ceil(args.day_seconds / args.interval_seconds))
    total_change_count = 0
    all_speeds_kmh: list[float] = []
    all_positions_m: list[float] = []
    all_leader_gaps_m: list[float] = []
    all_leader_speeds_kmh: list[float] = []
    all_paired_gap_m: list[float] = []
    all_paired_speed_kmh: list[float] = []

    for input_path in input_paths:
        try:
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
            ) = base.collect_lanechange_data(
                input_xml=input_path,
                interval_seconds=args.interval_seconds,
                day_seconds=args.day_seconds,
            )
        except ET.ParseError as error:
            print(f"Warning: skipping malformed lane-change XML {input_path}: {error}")
            continue

        per_run_primary_reason_counts.append(primary_reason_counts)
        per_run_reason_counts.append(reason_counts)
        per_run_urgency_counts.append(urgency_counts)
        max_num_bins = max(max_num_bins, num_bins)
        total_change_count += change_count
        all_speeds_kmh.extend(speeds_kmh)
        all_positions_m.extend(positions_m)
        all_leader_gaps_m.extend(leader_gaps_m)
        all_leader_speeds_kmh.extend(leader_speeds_kmh)
        all_paired_gap_m.extend(paired_gap_m)
        all_paired_speed_kmh.extend(paired_speed_kmh)

    primary_reasons, primary_matrix = aggregate_reason_median(
        per_run_primary_reason_counts,
        max_num_bins,
    )
    reasons, matrix = aggregate_reason_median(
        per_run_reason_counts,
        max_num_bins,
    )
    urgency_reasons, urgency_matrix = aggregate_reason_median(
        per_run_urgency_counts,
        max_num_bins,
    )

    speed_array = base.ensure_non_empty(all_speeds_kmh, "speed")
    position_array = base.ensure_non_empty(all_positions_m, "position")
    leader_gap_array = base.ensure_non_empty(all_leader_gaps_m, "leader gap")
    leader_speed_array = base.ensure_non_empty(all_leader_speeds_kmh, "leader speed")
    paired_gap_array = np.asarray(all_paired_gap_m, dtype=float)
    paired_speed_array = np.asarray(all_paired_speed_kmh, dtype=float)

    extension = args.format
    plot_reason_counts_median(
        reasons,
        matrix,
        args.interval_seconds,
        output_dir / f"lanechanges_by_reason_5min.{extension}",
        title="Median Lane Changes Over the Day by Full Reason",
        legend_title="Full reason",
    )
    plot_reason_counts_median(
        primary_reasons,
        primary_matrix,
        args.interval_seconds,
        output_dir / f"lanechanges_by_primary_reason_5min.{extension}",
        title="Median Lane Changes Over the Day by Primary Reason",
        legend_title="Primary reason",
    )
    plot_reason_counts_median(
        urgency_reasons,
        urgency_matrix,
        args.interval_seconds,
        output_dir / f"lanechanges_by_urgency_5min.{extension}",
        title="Median Lane Changes Over the Day by Urgency",
        legend_title="Urgency",
    )
    base.plot_histogram(
        speed_array,
        title="Lane Change Speeds Across All Validation Runs",
        xlabel="Speed [km/h]",
        output_path=output_dir / f"lanechange_speed_hist.{extension}",
        color="#2563eb",
    )
    base.plot_histogram(
        position_array,
        title="Lane Change Positions Across All Validation Runs",
        xlabel="Position on lane [m]",
        output_path=output_dir / f"lanechange_position_hist.{extension}",
        color="#d97706",
    )
    base.plot_leader_context(
        leader_gap_array,
        leader_speed_array,
        paired_gap_array,
        paired_speed_array,
        output_dir / f"lanechange_leader_context.{extension}",
    )

    print(
        f"Parsed {len(input_paths)} lanechange XML file(s) with "
        f"{total_change_count:,} total lane changes"
    )
    print(f"  lane-change speed: {base.stats_summary(speed_array, ' km/h')}")
    print(f"  lane-change position: {base.stats_summary(position_array, ' m')}")
    print(f"  leader gap: {base.stats_summary(leader_gap_array, ' m')}")
    print(f"  leader speed: {base.stats_summary(leader_speed_array, ' km/h')}")
    print(f"  paired leader gap/speed points: {paired_gap_array.size:,}")
    print(f"Saved validation lane-change plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
