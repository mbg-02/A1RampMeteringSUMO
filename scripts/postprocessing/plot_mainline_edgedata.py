#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Sequence, Set as AbstractSet
from collections import defaultdict
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import xml.etree.ElementTree as ET

PLOT_CACHE_DIR = Path(tempfile.gettempdir()) / "sumo_plot_cache"
PLOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PLOT_CACHE_DIR / "xdg"))

import matplotlib

matplotlib.use("Agg")
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
EDGEDATA_PATTERN = re.compile(r"edgedata(\d+)\.xml$")
REPLICATION_DIR_PATTERN = re.compile(r"rep_(\d+)_seed_(\d+)$")

ZURICH_EDGES = (
    "E1Z E2Z E3Z E4Z E5Z E6Z E7Z E8Z E9Z E10Z E11Z E12Z E13Z E14Z E15Z "
    "E16Z E17Z E18Z E19Z E20Z E21Z E22Z E23Z E24Z E25Z E26Z E27Z E28Z "
    "E29Z E30Z E31Z E32Z E33Z E34Z E35Z E36Z E37Z E38Z E39Z E40Z E41Z "
    "E42Z E43Z E44Z E45Z E46Z E47Z E48Z E49Z E50Z E51Z E52Z E53Z E54Z "
    "E55Z E56Z E57Z E58Z E59Z E60Z E61Z"
).split()
BERNE_EDGES = (
    "E1B E2B E3B E4B E5B E6B E7B E8B E9B E10B E11B E12B E13B E14B E15B "
    "E16B E17B E18B E19B E20B E21B E22B E23B E24B E25B E26B E27B E28B "
    "E29B E30B E31B E32B E33B E34B E35B E36B E37B E38B E39B E40B E41B "
    "E42B E43B E44B E45B E46B E47B E48B E49B E50B E51B E52B E53B E54B "
    "E55B E56B E57B E58B E59B E60B E61B E62B"
).split()

DIRECTIONS = (
    ("berne", "Direction Berne", BERNE_EDGES, True),
    ("zurich", "Direction Zurich", ZURICH_EDGES, False),
)
RAMP_MERGE_EDGES = {
    "berne": ("E6B", "E17B", "E29B", "E44B", "E62B"),
    "zurich": ("E5Z", "E23Z", "E38Z", "E48Z", "E56Z"),
}
RAMP_MERGE_LABELS = {
    "berne": ("52", "51", "50", "49", "48"),
    "zurich": ("48", "49", "50", "51", "52"),
}
METRICS = (
    ("density", "density", 1.0, "Density [veh/km]", "mainline_density"),
    ("speed", "speed", 3.6, "Speed [km/h]", "mainline_speed"),
    ("flow", "flow", 1.0, "Flow [veh/h]", "mainline_flow"),
)
METRIC_COLORMAPS = {
    "density": "viridis",
    "speed": "RdYlGn",
    "flow": "Blues",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create mainline edge-data heatmaps for density, speed and flow. "
            "Values are averaged across all edgedata*.xml files in the input folder."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="output sim/val_base_scenario",
        help=(
            "Directory containing edgedata1.xml ... edgedata20.xml, or "
            "calibration-style rep_XX_seed_*/edgedata.xml files."
        ),
    )
    parser.add_argument(
        "--pattern",
        default="edgedata*.xml",
        help="Glob used to discover edge-data XML files.",
    )
    parser.add_argument(
        "--net-file",
        default="network.net.xml",
        help="SUMO net file used to read mainline edge lengths.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Directory for generated plots. Defaults to mainline_edgedata_plots "
            "inside --input-dir."
        ),
    )
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--density-limits",
        nargs=2,
        type=float,
        metavar=("VMIN", "VMAX"),
        help="Fixed color scale limits for density plots.",
    )
    parser.add_argument(
        "--speed-limits",
        nargs=2,
        type=float,
        metavar=("VMIN", "VMAX"),
        help="Fixed color scale limits for speed plots.",
    )
    parser.add_argument(
        "--flow-limits",
        nargs=2,
        type=float,
        metavar=("VMIN", "VMAX"),
        help="Fixed color scale limits for flow plots.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path

    candidates = (Path.cwd() / path, PROJECT_DIR / path, SCRIPT_DIR / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return PROJECT_DIR / path


def output_dir_for(input_dir: Path, output_dir_arg: str | None) -> Path:
    if output_dir_arg:
        return resolve_path(output_dir_arg)
    return input_dir / "mainline_edgedata_plots"


def edgedata_sort_key(path: Path) -> tuple[float, str]:
    match = EDGEDATA_PATTERN.search(path.name)
    if match:
        return int(match.group(1)), path.name
    return math.inf, path.name


def discover_edgedata_paths(input_dir: Path, pattern: str) -> list[Path]:
    paths = sorted(input_dir.glob(pattern), key=edgedata_sort_key)
    if paths:
        return paths

    nested_paths: list[tuple[int, Path]] = []
    for path in input_dir.glob(f"rep_*_seed_*/{pattern}"):
        match = REPLICATION_DIR_PATTERN.fullmatch(path.parent.name)
        if match is None:
            continue
        nested_paths.append((int(match.group(1)), path))
    if nested_paths:
        nested_paths.sort(key=lambda item: item[0])
        return [path for _, path in nested_paths]

    raise FileNotFoundError(f"No edge-data XML files matching {pattern} in {input_dir}")


def load_edge_lengths(net_file: Path, edge_ids: AbstractSet[str]) -> dict[str, float]:
    lengths: dict[str, float] = {}
    root = ET.parse(net_file).getroot()
    for edge in root.findall("edge"):
        edge_id = edge.attrib.get("id")
        if edge_id is None or edge_id not in edge_ids:
            continue
        lanes = edge.findall("lane")
        if not lanes:
            continue
        length = lanes[0].attrib.get("length")
        if length is None:
            continue
        lengths[edge_id] = float(length)

    missing = sorted(edge_id for edge_id in edge_ids if edge_id not in lengths)
    if missing:
        raise ValueError(
            f"{net_file} is missing length data for edge(s): {', '.join(missing)}"
        )
    return lengths


def cumulative_boundaries_km(
    edge_ids: Sequence[str], lengths_m: dict[str, float]
) -> np.ndarray:
    boundaries = [0.0]
    for edge_id in edge_ids:
        boundaries.append(boundaries[-1] + lengths_m[edge_id] / 1000.0)
    return np.asarray(boundaries, dtype=float)


def edge_start_positions_km(
    edge_ids: Sequence[str],
    lengths_m: dict[str, float],
    target_edge_ids: tuple[str, ...],
) -> list[float]:
    target_edges = set(target_edge_ids)
    positions: list[float] = []
    position_km = 0.0
    for edge_id in edge_ids:
        if edge_id in target_edges:
            positions.append(position_km)
        position_km += lengths_m[edge_id] / 1000.0
    return positions


def parse_float(value: str | None) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_edgedata(
    paths: list[Path],
    edge_ids: AbstractSet[str],
) -> tuple[list[tuple[float, float]], dict[str, dict[float, dict[str, list[float]]]]]:
    interval_ends: dict[float, float] = {}
    values: dict[str, dict[float, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for path in paths:
        try:
            for _, interval in ET.iterparse(path, events=("end",)):
                if interval.tag != "interval":
                    continue

                begin = parse_float(interval.attrib.get("begin"))
                end = parse_float(interval.attrib.get("end"))
                if begin is None or end is None:
                    interval.clear()
                    continue

                interval_ends[begin] = max(end, interval_ends.get(begin, end))
                for edge in interval.findall("edge"):
                    edge_id = edge.attrib.get("id")
                    if edge_id is None or edge_id not in edge_ids:
                        continue
                    for metric_key, attr_name, scale, _, _ in METRICS:
                        raw_value = parse_float(edge.attrib.get(attr_name))
                        if raw_value is not None:
                            values[edge_id][begin][metric_key].append(raw_value * scale)
                interval.clear()
        except ET.ParseError as error:
            print(
                f"Warning: skipping malformed edge-data XML {path}: {error}",
                file=sys.stderr,
            )

    sorted_intervals = [
        (begin, interval_ends[begin]) for begin in sorted(interval_ends)
    ]
    if not sorted_intervals:
        raise ValueError("No edge-data intervals found.")
    return sorted_intervals, values


def time_boundaries_hours(intervals: list[tuple[float, float]]) -> np.ndarray:
    boundaries = [intervals[0][0] / 3600.0]
    boundaries.extend(end / 3600.0 for _, end in intervals)
    return np.asarray(boundaries, dtype=float)


def metric_matrix(
    edge_ids: Sequence[str],
    intervals: list[tuple[float, float]],
    values: dict[str, dict[float, dict[str, list[float]]]],
    metric_key: str,
) -> np.ndarray:
    matrix = np.full((len(edge_ids), len(intervals)), np.nan, dtype=float)
    for edge_index, edge_id in enumerate(edge_ids):
        for interval_index, (begin, _) in enumerate(intervals):
            samples = values.get(edge_id, {}).get(begin, {}).get(metric_key, [])
            if samples:
                matrix[edge_index, interval_index] = float(np.mean(samples))
    return matrix


def plot_metric(
    output_path: Path,
    metric_key: str,
    label: str,
    intervals: list[tuple[float, float]],
    values: dict[str, dict[float, dict[str, list[float]]]],
    lengths_m: dict[str, float],
    image_format: str,
    dpi: int,
    color_limit: tuple[float, float] | None,
) -> None:
    x_bounds = time_boundaries_hours(intervals)
    with plt.rc_context(
        {
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.titlesize": 18,
        }
    ):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 9),
            sharex=True,
            constrained_layout=True,
        )
        _plot_metric_on_axes(
            fig,
            axes,
            x_bounds,
            metric_key,
            label,
            intervals,
            values,
            lengths_m,
            color_limit,
        )
        fig.savefig(output_path, format=image_format, dpi=dpi)
        plt.close(fig)


def _plot_metric_on_axes(
    fig: Figure,
    axes: np.ndarray,
    x_bounds: np.ndarray,
    metric_key: str,
    label: str,
    intervals: list[tuple[float, float]],
    values: dict[str, dict[float, dict[str, list[float]]]],
    lengths_m: dict[str, float],
    color_limit: tuple[float, float] | None,
) -> None:
    meshes = []

    for ax, (direction_id, title, edge_ids, flip_y_axis) in zip(axes, DIRECTIONS):
        y_bounds = cumulative_boundaries_km(edge_ids, lengths_m)
        matrix = metric_matrix(edge_ids, intervals, values, metric_key)
        mesh = ax.pcolormesh(
            x_bounds,
            y_bounds,
            matrix,
            shading="auto",
            cmap=METRIC_COLORMAPS[metric_key],
        )
        meshes.append(mesh)
        ax.set_title(title)
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("Mainline position [km]")
        draw_ramp_merge_markers(
            ax,
            edge_start_positions_km(
                edge_ids,
                lengths_m,
                RAMP_MERGE_EDGES[direction_id],
            ),
            RAMP_MERGE_LABELS[direction_id],
        )
        if flip_y_axis:
            ax.invert_yaxis()

    if color_limit is not None:
        vmin, vmax = color_limit
        for mesh in meshes:
            mesh.set_clim(vmin, vmax)
    else:
        finite_values = np.concatenate(
            [
                (
                    mesh.get_array().compressed()
                    if hasattr(mesh.get_array(), "compressed")
                    else np.asarray(mesh.get_array())[
                        np.isfinite(mesh.get_array())
                    ]
                )
                for mesh in meshes
            ]
        )
        if finite_values.size:
            vmin = float(np.nanpercentile(finite_values, 2))
            vmax = float(np.nanpercentile(finite_values, 98))
            if vmin < vmax:
                for mesh in meshes:
                    mesh.set_clim(vmin, vmax)

    fig.colorbar(meshes[-1], ax=axes, label=label, shrink=0.95)


def draw_ramp_merge_markers(
    ax: Axes,
    positions_km: list[float],
    labels: tuple[str, ...],
) -> None:
    for position_km, label in zip(positions_km, labels):
        ax.plot(
            [1.01, 1.045],
            [position_km, position_km],
            color="#dc2626",
            linewidth=3.0,
            solid_capstyle="round",
            transform=ax.get_yaxis_transform(),
            clip_on=False,
        )
        ax.text(
            1.055,
            position_km,
            label,
            color="#dc2626",
            fontsize=11,
            fontweight="bold",
            va="center",
            ha="left",
            transform=ax.get_yaxis_transform(),
            clip_on=False,
        )


def main() -> int:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    net_file = resolve_path(args.net_file)
    output_dir = output_dir_for(input_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    edgedata_paths = discover_edgedata_paths(input_dir, args.pattern)
    edge_ids = set(ZURICH_EDGES).union(BERNE_EDGES)
    lengths_m = load_edge_lengths(net_file, edge_ids)
    intervals, values = read_edgedata(edgedata_paths, edge_ids)
    color_limits = {
        "density": tuple(args.density_limits) if args.density_limits else None,
        "speed": tuple(args.speed_limits) if args.speed_limits else None,
        "flow": tuple(args.flow_limits) if args.flow_limits else None,
    }

    for metric_key, _, _, label, filename in METRICS:
        output_path = output_dir / f"{filename}.{args.format}"
        plot_metric(
            output_path=output_path,
            metric_key=metric_key,
            label=label,
            intervals=intervals,
            values=values,
            lengths_m=lengths_m,
            image_format=args.format,
            dpi=args.dpi,
            color_limit=color_limits[metric_key],
        )

    if len(intervals) == 1:
        print(
            "Warning: edge-data files contain only one interval; plots show one "
            "all-period time column. Future validation runs should use 5-minute "
            "edgeData output."
        )
    print(
        f"Wrote mainline edge-data plots from {len(edgedata_paths)} file(s) "
        f"and {len(intervals)} interval(s) to {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
