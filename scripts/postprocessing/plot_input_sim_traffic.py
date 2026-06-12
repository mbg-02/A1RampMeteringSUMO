#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
from pathlib import Path
import sys
from typing import TypeAlias
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
DEFAULT_EMITTER_FILES = [
    "FLOWbus.emitters.xml",
    "FLOWcar.emitters.xml",
    "FLOWdeliv.emitters.xml",
    "FLOWmotc.emitters.xml",
    "FLOWtruck.emitters.xml",
]
DEFAULT_DETECTOR_FLOWS_CSV = "flows_detFWR.csv"
DEFAULT_SIMULATION_OUTPUT_XML = "output sim/17/LD_output.xml"
DEFAULT_SPEED_INPUT_DIR = "data/VBV"
VEHICLE_TYPES = ["car", "truck", "deliv", "motc", "bus"]
VEHICLE_LABELS = {
    "bus": "Bus",
    "car": "Car",
    "deliv": "Delivery",
    "motc": "Motorcycle",
    "truck": "Truck",
}
VEHICLE_COLORS = {
    "bus": "#0f766e",
    "car": "#2563eb",
    "deliv": "#d97706",
    "motc": "#dc2626",
    "truck": "#4f46e5",
}
CSV_FLOW_COLUMNS = {
    "bus": "qBus",
    "car": "qCar",
    "deliv": "qDeliv",
    "motc": "qMotc",
    "truck": "qTruck",
}
DETECTOR_TO_PLOT_EDGE = {
    "072_B_SP": "E33B",
    "072_Z_SP": "E30Z",
    "229_B_SP": "E55B",
    "229_Z_SP": "E8Z",
    "280_B_SP": "E24B",
    "280_Z_SP": "E39Z",
    "298_B_SP": "E12B",
    "298_Z_SP": "E50Z",
    "335_B_SP": "E62B",
    "335_Z_SP": "E1Z",
    "382_Z_SP": "E61Z",
    "382_B_SP": "E1B",
    "48_B_off_SP": "E59Bout3",
    "48_B_on_SP": "E62Bin2",
    "48_Z_off_SP": "E1Zout3",
    "48_Z_on_SP": "E5Zin2",
    "49_B_off_SP": "E41Bout2",
    "49_B_on_SP": "E44Bin3",
    "49_Z_off_SP": "E20Zout2",
    "49_Z_on_SP": "E23Zin2",
    "50_B_off_SP": "E23Bout3",
    "50_B_on_1_SP": "E29Bin4",
    "50_B_on_2_SP": "E29Bin6",
    "50_Z_off_SP": "E33Zout3",
    "50_Z_on_1_SP": "E38Zin3",
    "50_Z_on_2_SP": "E38Zin5",
    "51_B_off_SP": "E14Bout",
    "51_B_on_SP": "E17Bin",
    "51_Z_off_SP": "E45Zout",
    "51_Z_on_SP": "E48Zin",
    "52_B_off_SP": "E3Bout3",
    "52_B_on_SP": "E6Bin4",
    "52_Z_off_SP": "E52Zout3",
    "52_Z_on_SP": "E56Zin3",
}
SIMULATION_DETECTOR_MAPPING = {
    "072_B_SP": ("072_sim1", "072_sim2"),
    "072_Z_SP": ("072_sim3", "072_sim4"),
    "229_B_SP": ("229_sim1", "229_sim2"),
    "229_Z_SP": ("229_sim3", "229_sim4"),
    "280_B_SP": ("280_sim2", "280_sim3"),
    "280_Z_SP": ("280_sim4", "280_sim5"),
    "298_B_SP": ("298_sim1", "298_sim2"),
    "298_Z_SP": ("298_sim3", "298_sim4"),
    "335_B_SP": ("335_sim1", "335_sim2", "335_sim3"),
    "335_Z_SP": ("335_sim4", "335_sim5", "335_sim6"),
    "382_B_SP": ("382_sim1", "382_sim2"),
    "382_Z_SP": ("382_sim3", "382_sim4"),
    "48_B_on_SP": ("48_B_on1", "48_B_on2"),
    "48_B_off_SP": ("48_B_off1", "48_B_off2"),
    "48_Z_off_SP": ("48_Z_off1", "48_Z_off2"),
    "48_Z_on_SP": ("48_Z_on1", "48_Z_on2"),
    "49_B_on_SP": ("49_B_on",),
    "49_B_off_SP": ("49_B_off1", "49_B_off2"),
    "49_Z_off_SP": ("49_Z_off1", "49_Z_off2"),
    "49_Z_on_SP": ("49_Z_on",),
    "50_B_off_SP": ("50_B_off1", "50_B_off2"),
    "50_B_on_1_SP": ("50_B_on_1",),
    "50_B_on_2_SP": ("50_B_on_2",),
    "50_Z_off_SP": ("50_Z_off1", "50_Z_off2"),
    "50_Z_on_1_SP": ("50_Z_on_1",),
    "50_Z_on_2_SP": ("50_Z_on_2",),
    "51_B_off_SP": ("51_B_off",),
    "51_B_on_SP": ("51_B_on",),
    "51_Z_off_SP": ("51_Z_off",),
    "51_Z_on_SP": ("51_Z_on",),
    "52_B_off_SP": ("52_B_off1", "52_B_off2"),
    "52_B_on_SP": ("52_B_on1", "52_B_on2"),
    "52_Z_off_SP": ("52_Z_off1", "52_Z_off2"),
    "52_Z_on_SP": ("52_Z_on",),
}
SPEED_DETECTOR_MAPPING = {
    "072_B": (
        ("VBV_072_vd4", "072_sim1"),
        ("VBV_072_vd3", "072_sim2"),
    ),
    "072_Z": (
        ("VBV_072_vd2", "072_sim3"),
        ("VBV_072_vd1", "072_sim4"),
    ),
    "229_B": (
        ("VBV_229_vd4", "229_sim1"),
        ("VBV_229_vd3", "229_sim2"),
    ),
    "229_Z": (
        ("VBV_229_vd2", "229_sim3"),
        ("VBV_229_vd1", "229_sim4"),
    ),
    "298_B": (
        ("VBV_298_vd4", "298_sim1"),
        ("VBV_298_vd3", "298_sim2"),
    ),
    "298_Z": (
        ("VBV_298_vd2", "298_sim3"),
        ("VBV_298_vd1", "298_sim4"),
    ),
}
COMBINED_SIMULATION_COMPARISONS = {
    "51_B_on_SP": ("51_on", ("51_B_on_SP", "51_Z_on_SP")),
    "51_Z_on_SP": ("51_on", ("51_B_on_SP", "51_Z_on_SP")),
    "51_B_off_SP": ("51_off", ("51_B_off_SP", "51_Z_off_SP")),
    "51_Z_off_SP": ("51_off", ("51_B_off_SP", "51_Z_off_SP")),
}
INPUT_LINE_STYLE = (0, (6, 3))
SIMULATION_LINE_STYLE = (0, (1, 2.5))
INTERVAL_SECONDS = 300
SIMULATION_SMOOTHING_SECONDS = 60 * 60
SECONDS_PER_DAY = 24 * 60 * 60
SQV_HOURLY_SCALE = 1000.0
MPS_TO_KMPH = 3.6
VBV_SPEED_INPUT_FILES = [
    "VBV_072_2025-10-06_2025-10-12_all.csv",
    "VBV_229_2025-10-06_2025-10-12_all.csv",
    "VBV_298_2025-10-06_2025-10-12_all.csv",
]
VBV_REQUIRED_COLUMNS = {"src_time_src", "vd", "vd_speed_val", "vd_class_val"}
VBV_MAX_SPEED_BY_CLASS = {
    "101": 120.0,
    "102": 200.0,
    "103": 200.0,
    "104": 200.0,
    "105": 200.0,
    "106": 120.0,
    "107": 100.0,
    "108": 100.0,
    "109": 100.0,
    "110": 100.0,
}
CsvDialectLike: TypeAlias = csv.Dialect | type[csv.Dialect]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create stacked 24h start-edge traffic plots from SUMO emitter files.",
    )
    parser.add_argument(
        "--output-dir",
        default="start_edge_traffic_plots",
        help="Directory for generated plot files.",
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
        default=INTERVAL_SECONDS,
        help="Aggregation bucket size in seconds.",
    )
    parser.add_argument(
        "--detector-flows-csv",
        default=DEFAULT_DETECTOR_FLOWS_CSV,
        help="CSV with detector input flows to overlay for mapped start edges.",
    )
    parser.add_argument(
        "--simulation-output",
        default=DEFAULT_SIMULATION_OUTPUT_XML,
        help="SUMO detector output XML to overlay as aggregated simulation totals.",
    )
    parser.add_argument(
        "--speed-input-dir",
        "--speed-input-csv",
        dest="speed_input_dir",
        default=DEFAULT_SPEED_INPUT_DIR,
        help="Directory with raw VBV speed CSV files to compare against simulation detectors.",
    )
    parser.add_argument(
        "emitter_files",
        nargs="*",
        default=DEFAULT_EMITTER_FILES,
        help="Emitter XML files to read.",
    )
    return parser.parse_args()


def resolve_input_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        SCRIPT_DIR / path,
        PROJECT_DIR / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fall back to the project directory so missing-file messages stay useful.
    return PROJECT_DIR / path


def resolve_output_dir(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def route_file_for_emitter(emitter_file: Path) -> Path:
    suffix = ".emitters.xml"
    if emitter_file.name.endswith(suffix):
        return emitter_file.with_name(emitter_file.name.replace(suffix, ".routes.xml"))
    return emitter_file.with_name(f"{emitter_file.stem}.routes.xml")


def load_route_edges(emitter_files: list[Path]) -> dict[str, tuple[str, ...]]:
    route_edges: dict[str, tuple[str, ...]] = {}
    for emitter_file in emitter_files:
        route_file = route_file_for_emitter(emitter_file)
        if not route_file.exists():
            raise FileNotFoundError(
                f"Missing route file for {emitter_file.name}: {route_file}"
            )

        root = ET.parse(route_file).getroot()
        for route in root.findall("route"):
            route_id = route.attrib["id"]
            edges = tuple(route.attrib.get("edges", "").split())
            if not edges:
                raise ValueError(f"Route {route_id} in {route_file} has no edges")
            route_edges[route_id] = edges
    return route_edges


def build_counts(
    emitter_files: list[Path],
    interval_seconds: int,
    route_edges: dict[str, tuple[str, ...]],
) -> tuple[dict[str, dict[str, np.ndarray]], int]:
    num_bins = SECONDS_PER_DAY // interval_seconds
    counts: dict[str, dict[str, np.ndarray]] = defaultdict(
        lambda: {
            vehicle_type: np.zeros(num_bins, dtype=float)
            for vehicle_type in VEHICLE_TYPES
        }
    )
    edge_to_detectors: dict[str, list[str]] = defaultdict(list)
    for detector_id, plot_edge in DETECTOR_TO_PLOT_EDGE.items():
        edge_to_detectors[plot_edge].append(detector_id)

    for emitter_file in emitter_files:
        root = ET.parse(emitter_file).getroot()
        for flow in root.findall("flow"):
            vehicle_type = flow.attrib["type"]
            if vehicle_type not in VEHICLE_TYPES:
                continue

            begin_seconds = float(flow.attrib["begin"])
            bucket = int(round(begin_seconds / interval_seconds))
            if not 0 <= bucket < num_bins:
                continue

            route_id = flow.attrib.get("route", flow.attrib["id"])
            edges = route_edges.get(route_id)
            if edges is None:
                raise KeyError(f"Missing route definition for {route_id}")

            traversed_detectors = {
                detector_id
                for edge in edges
                for detector_id in edge_to_detectors.get(edge, [])
            }
            for detector_id in traversed_detectors:
                counts[detector_id][vehicle_type][bucket] += float(
                    flow.attrib["number"]
                )

    return dict(sorted(counts.items())), num_bins


def build_detector_input_counts(
    detector_flows_csv: Path,
    interval_seconds: int,
    num_bins: int,
) -> dict[str, dict[str, np.ndarray]]:
    counts: dict[str, dict[str, np.ndarray]] = defaultdict(
        lambda: {
            vehicle_type: np.zeros(num_bins, dtype=float)
            for vehicle_type in VEHICLE_TYPES
        }
    )

    with detector_flows_csv.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        required_columns = {"Detector", "Time", *CSV_FLOW_COLUMNS.values()}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing_list = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"{detector_flows_csv} is missing required columns: {missing_list}"
            )

        for row in reader:
            detector_id = (row.get("Detector") or "").strip()
            if detector_id not in DETECTOR_TO_PLOT_EDGE:
                continue

            time_minutes = float(row["Time"])
            bucket = int(round((time_minutes * 60.0) / interval_seconds))
            if not 0 <= bucket < num_bins:
                continue

            for vehicle_type in VEHICLE_TYPES:
                counts[detector_id][vehicle_type][bucket] += float(
                    row[CSV_FLOW_COLUMNS[vehicle_type]]
                )

    return dict(sorted(counts.items()))


def build_simulation_counts(
    simulation_output_xml: Path,
    interval_seconds: int,
    num_bins: int,
) -> dict[str, np.ndarray]:
    detector_lookup = {
        source_detector_id: detector_id
        for detector_id, source_detector_ids in SIMULATION_DETECTOR_MAPPING.items()
        for source_detector_id in source_detector_ids
    }
    counts: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(num_bins, dtype=float))

    root = ET.parse(simulation_output_xml).getroot()
    for interval in root.findall("interval"):
        source_detector_id = interval.attrib["id"]
        detector_id = detector_lookup.get(source_detector_id)
        if detector_id is None:
            continue

        begin_seconds = float(interval.attrib["begin"])
        bucket = int(begin_seconds // interval_seconds)
        if not 0 <= bucket < num_bins:
            continue

        counts[detector_id][bucket] += float(interval.attrib["nVehContrib"])

    return dict(sorted(counts.items()))


def detect_csv_dialect(csv_path: Path) -> CsvDialectLike:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
        sample = source.read(65536)

    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel()
        dialect.delimiter = ";"
        return dialect


def make_vbv_source_label(path: Path) -> str:
    stem = path.stem
    if stem.startswith("VBV_"):
        return "_".join(stem.split("_")[:2])
    return stem


def parse_vbv_speed(raw_speed: str) -> float | None:
    value = (raw_speed or "").strip()
    if not value:
        return None
    try:
        speed = float(value.replace(",", "."))
    except ValueError:
        return None
    if speed <= 0:
        return None
    return speed


def is_valid_vbv_speed(class_code: str, speed: float) -> bool:
    max_speed = VBV_MAX_SPEED_BY_CLASS.get(class_code)
    if max_speed is None:
        return True
    return speed <= max_speed


def parse_vbv_timestamp(raw_timestamp: str) -> datetime:
    return datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00")).replace(
        tzinfo=None
    )


def speed_bucket_index(timestamp: datetime, interval_seconds: int) -> int:
    seconds_of_day = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
    return seconds_of_day // interval_seconds


def empty_float_buckets(num_bins: int) -> list[float]:
    return [0.0] * num_bins


def empty_int_buckets(num_bins: int) -> list[int]:
    return [0] * num_bins


def finalize_speed_profiles(
    speed_sums: dict[str, dict[str, list[float]]],
    speed_counts: dict[str, dict[str, list[int]]],
) -> dict[str, dict[str, list[float | None]]]:
    profiles: dict[str, dict[str, list[float | None]]] = {}
    for lane_id in sorted(speed_sums):
        lane_profiles: dict[str, list[float | None]] = {}
        for date_label in sorted(speed_sums[lane_id]):
            values: list[float | None] = []
            for speed_sum, speed_count in zip(
                speed_sums[lane_id][date_label],
                speed_counts[lane_id][date_label],
            ):
                if speed_count == 0:
                    values.append(None)
                else:
                    values.append(speed_sum / speed_count)
            lane_profiles[date_label] = values
        profiles[lane_id] = lane_profiles
    return profiles


def average_speed_profiles(
    values_by_date: dict[str, list[float | None]],
    target_weekdays: set[int],
) -> list[float | None]:
    first_values = next(iter(values_by_date.values()))
    combined_sums = [0.0] * len(first_values)
    combined_counts = [0] * len(first_values)

    for date_label, values in values_by_date.items():
        current_date = datetime.fromisoformat(date_label).date()
        if current_date.weekday() not in target_weekdays:
            continue
        for bucket_index, value in enumerate(values):
            if value is None:
                continue
            combined_sums[bucket_index] += value
            combined_counts[bucket_index] += 1

    averages: list[float | None] = []
    for speed_sum, speed_count in zip(combined_sums, combined_counts):
        if speed_count == 0:
            averages.append(None)
        else:
            averages.append(speed_sum / speed_count)
    return averages


def speed_profile_to_array(values: list[float | None]) -> np.ndarray:
    return np.array(
        [np.nan if value is None else float(value) for value in values],
        dtype=float,
    )


def compute_speed_mae(
    simulated: np.ndarray | None,
    observed: np.ndarray | None,
) -> float | None:
    if simulated is None or observed is None:
        return None

    valid = ~np.isnan(simulated) & ~np.isnan(observed)
    if not np.any(valid):
        return None

    return float(np.mean(np.abs(simulated[valid] - observed[valid])))


def build_speed_input_profiles(
    speed_input_dir: Path,
    interval_seconds: int,
    num_bins: int,
) -> tuple[dict[str, dict[str, list[float | None]]], list[str]]:
    lane_ids = sorted(
        input_lane_id
        for lane_pairs in SPEED_DETECTOR_MAPPING.values()
        for input_lane_id, _ in lane_pairs
    )
    speed_sums: dict[str, dict[str, list[float]]] = defaultdict(dict)
    speed_counts: dict[str, dict[str, list[int]]] = defaultdict(dict)

    input_paths = [speed_input_dir / file_name for file_name in VBV_SPEED_INPUT_FILES]
    missing_files = [str(path) for path in input_paths if not path.exists()]
    if missing_files:
        print(
            "Warning: missing VBV speed input file(s): "
            + ", ".join(missing_files),
            file=sys.stderr,
        )

    for input_path in input_paths:
        if not input_path.exists():
            continue
        source_label = make_vbv_source_label(input_path)
        dialect = detect_csv_dialect(input_path)
        with input_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source, dialect=dialect)
            normalized_fieldnames = {
                ((name or "").strip().lower()) for name in (reader.fieldnames or [])
            }
            missing_columns = VBV_REQUIRED_COLUMNS.difference(normalized_fieldnames)
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                print(
                    f"Warning: skipping {input_path}; missing required columns: {missing}",
                    file=sys.stderr,
                )
                continue

            for row in reader:
                normalized_row = {
                    ((key or "").strip().lower()): value for key, value in row.items()
                }
                lane_number = (normalized_row.get("vd") or "").strip()
                class_code = (normalized_row.get("vd_class_val") or "").strip()
                raw_timestamp = (normalized_row.get("src_time_src") or "").strip()
                speed = parse_vbv_speed(normalized_row.get("vd_speed_val") or "")
                if (
                    not lane_number
                    or not class_code
                    or not raw_timestamp
                    or speed is None
                ):
                    continue
                if not is_valid_vbv_speed(class_code, speed):
                    continue

                lane_id = f"{source_label}_vd{lane_number}"
                if lane_id not in lane_ids:
                    continue

                try:
                    timestamp = parse_vbv_timestamp(raw_timestamp)
                except ValueError:
                    continue

                bucket = speed_bucket_index(timestamp, interval_seconds)
                if not 0 <= bucket < num_bins:
                    continue

                date_label = timestamp.date().isoformat()
                lane_sums = speed_sums[lane_id].setdefault(
                    date_label, empty_float_buckets(num_bins)
                )
                lane_counts = speed_counts[lane_id].setdefault(
                    date_label, empty_int_buckets(num_bins)
                )
                lane_sums[bucket] += speed
                lane_counts[bucket] += 1

    profiles = finalize_speed_profiles(speed_sums, speed_counts)
    missing_lanes = [lane_id for lane_id in lane_ids if lane_id not in profiles]
    return profiles, missing_lanes


def build_simulation_speed_series(
    simulation_output_xml: Path,
    interval_seconds: int,
    num_bins: int,
) -> dict[str, np.ndarray]:
    detector_ids = {
        simulation_detector_id
        for lane_pairs in SPEED_DETECTOR_MAPPING.values()
        for _, simulation_detector_id in lane_pairs
    }
    weighted_speed_totals: dict[str, np.ndarray] = {
        detector_id: np.zeros(num_bins, dtype=float) for detector_id in detector_ids
    }
    vehicle_counts: dict[str, np.ndarray] = {
        detector_id: np.zeros(num_bins, dtype=float) for detector_id in detector_ids
    }

    root = ET.parse(simulation_output_xml).getroot()
    for interval in root.findall("interval"):
        detector_id = interval.attrib["id"]
        if detector_id not in detector_ids:
            continue

        begin_seconds = float(interval.attrib["begin"])
        bucket = int(begin_seconds // interval_seconds)
        if not 0 <= bucket < num_bins:
            continue

        n_vehicles = float(interval.attrib.get("nVehContrib", "0"))
        speed_mps = float(interval.attrib.get("speed", "-1"))
        if n_vehicles <= 0 or speed_mps < 0:
            continue

        weighted_speed_totals[detector_id][bucket] += (
            n_vehicles * speed_mps * MPS_TO_KMPH
        )
        vehicle_counts[detector_id][bucket] += n_vehicles

    simulation_speeds: dict[str, np.ndarray] = {}
    for detector_id in sorted(detector_ids):
        speed_series = np.full(num_bins, np.nan, dtype=float)
        positive_counts = vehicle_counts[detector_id] > 0
        speed_series[positive_counts] = (
            weighted_speed_totals[detector_id][positive_counts]
            / vehicle_counts[detector_id][positive_counts]
        )
        simulation_speeds[detector_id] = speed_series

    return simulation_speeds


def format_hour_tick(hour: int) -> str:
    return f"{hour:02d}:00"


def sum_vehicle_counts(vehicle_counts: dict[str, np.ndarray]) -> np.ndarray:
    return np.sum(
        np.vstack([vehicle_counts[vehicle_type] for vehicle_type in VEHICLE_TYPES]),
        axis=0,
    )


def sum_detector_totals(
    detector_ids: tuple[str, ...],
    counts: dict[str, dict[str, np.ndarray]],
    num_bins: int,
) -> np.ndarray:
    total = np.zeros(num_bins, dtype=float)
    for detector_id in detector_ids:
        vehicle_counts = counts.get(detector_id)
        if vehicle_counts is not None:
            total += sum_vehicle_counts(vehicle_counts)
    return total


def sum_available_detector_totals(
    detector_ids: tuple[str, ...],
    counts: dict[str, dict[str, np.ndarray]],
    num_bins: int,
) -> np.ndarray | None:
    total = np.zeros(num_bins, dtype=float)
    found_counts = False
    for detector_id in detector_ids:
        vehicle_counts = counts.get(detector_id)
        if vehicle_counts is not None:
            total += sum_vehicle_counts(vehicle_counts)
            found_counts = True
    return total if found_counts else None


def sum_detector_series(
    detector_ids: tuple[str, ...],
    detector_series: dict[str, np.ndarray],
    num_bins: int,
) -> np.ndarray | None:
    total = np.zeros(num_bins, dtype=float)
    found_series = False
    for detector_id in detector_ids:
        series = detector_series.get(detector_id)
        if series is not None:
            total += series
            found_series = True
    return total if found_series else None


def simulation_comparison_series(
    detector_id: str,
    vehicle_counts: dict[str, np.ndarray],
    all_counts: dict[str, dict[str, np.ndarray]],
    simulation_counts: dict[str, np.ndarray],
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray | None, str | None]:
    combined_comparison = COMBINED_SIMULATION_COMPARISONS.get(detector_id)
    if combined_comparison is None:
        return (
            sum_vehicle_counts(vehicle_counts),
            simulation_counts.get(detector_id),
            None,
        )

    label, detector_ids = combined_comparison
    return (
        sum_detector_totals(detector_ids, all_counts, num_bins),
        sum_detector_series(detector_ids, simulation_counts, num_bins),
        label,
    )


def input_comparison_series(
    detector_id: str,
    all_input_counts: dict[str, dict[str, np.ndarray]],
    num_bins: int,
) -> tuple[np.ndarray | None, str | None]:
    combined_comparison = COMBINED_SIMULATION_COMPARISONS.get(detector_id)
    if combined_comparison is None:
        detector_input_counts = all_input_counts.get(detector_id)
        if detector_input_counts is None:
            return None, None
        return sum_vehicle_counts(detector_input_counts), None

    label, detector_ids = combined_comparison
    return (
        sum_available_detector_totals(detector_ids, all_input_counts, num_bins),
        label,
    )


def aggregate_hourly(values: np.ndarray, interval_seconds: int) -> np.ndarray:
    hourly = np.zeros(24, dtype=float)
    hour_indices = (np.arange(values.shape[0]) * interval_seconds // 3600).astype(int)
    np.add.at(hourly, hour_indices, values)
    return hourly


def rolling_average(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1:
        return values

    left_pad = window_size // 2
    right_pad = window_size - 1 - left_pad
    padded = np.pad(values, (left_pad, right_pad), mode="edge")
    kernel = np.ones(window_size, dtype=float) / window_size
    return np.convolve(padded, kernel, mode="valid")


def simulation_smoothing_window(interval_seconds: int) -> int:
    return max(1, round(SIMULATION_SMOOTHING_SECONDS / interval_seconds))


def smooth_simulation_counts(
    simulation_counts: np.ndarray,
    interval_seconds: int,
) -> tuple[np.ndarray, int]:
    window_size = simulation_smoothing_window(interval_seconds)
    smoothing_minutes = window_size * interval_seconds // 60
    return rolling_average(simulation_counts, window_size), smoothing_minutes


def compute_sqv(
    modeled: np.ndarray, observed: np.ndarray, scale_factor: float
) -> np.ndarray:
    sqv = np.full(modeled.shape, np.nan, dtype=float)
    both_zero = (modeled == 0) & (observed == 0)
    sqv[both_zero] = np.nan

    observed_positive = observed > 0
    abs_deviation = np.abs(modeled[observed_positive] - observed[observed_positive])
    sqv[observed_positive] = 1.0 / (
        1.0 + np.sqrt((abs_deviation**2) / (scale_factor * observed[observed_positive]))
    )

    modeled_without_observed = (observed <= 0) & (modeled > 0)
    sqv[modeled_without_observed] = 0.0
    return sqv


def compute_total_metrics(
    modeled_total: np.ndarray,
    observed_total: np.ndarray | None,
    interval_seconds: int,
) -> dict[str, float | str | None] | None:
    if observed_total is None:
        return None

    observed_sum = float(np.sum(observed_total))
    if observed_sum <= 0:
        return {
            "sqv_interval_median": None,
            "hourly_sqv_median": None,
            "wape_5min": None,
            "hourly_wape_median": None,
            "daily_total_error": None,
            "notes": "No observed flow",
        }

    hourly_modeled = aggregate_hourly(modeled_total, interval_seconds)
    hourly_observed = aggregate_hourly(observed_total, interval_seconds)

    hourly_sqv = compute_sqv(hourly_modeled, hourly_observed, SQV_HOURLY_SCALE)
    valid_sqv = ~np.isnan(hourly_sqv)
    if np.any(valid_sqv):
        hourly_sqv_median = float(np.median(hourly_sqv[valid_sqv]))
    else:
        hourly_sqv_median = float("nan")

    hourly_wape = np.full(hourly_modeled.shape, np.nan, dtype=float)
    observed_hourly_positive = hourly_observed > 0
    hourly_wape[observed_hourly_positive] = (
        np.abs(
            hourly_modeled[observed_hourly_positive]
            - hourly_observed[observed_hourly_positive]
        )
        / hourly_observed[observed_hourly_positive]
    )
    valid_hourly_wape = ~np.isnan(hourly_wape)
    if np.any(valid_hourly_wape):
        hourly_wape_median = float(np.median(hourly_wape[valid_hourly_wape]))
    else:
        hourly_wape_median = float("nan")

    interval_sqv_scale = SQV_HOURLY_SCALE * (interval_seconds / 3600.0)
    interval_sqv = compute_sqv(modeled_total, observed_total, interval_sqv_scale)
    valid_interval_sqv = ~np.isnan(interval_sqv)
    if np.any(valid_interval_sqv):
        interval_sqv_median = float(np.median(interval_sqv[valid_interval_sqv]))
    else:
        interval_sqv_median = float("nan")

    wape = float(np.sum(np.abs(modeled_total - observed_total)) / observed_sum)
    daily_total_error = float((np.sum(modeled_total) - observed_sum) / observed_sum)

    return {
        "sqv_interval_median": (
            None if np.isnan(interval_sqv_median) else interval_sqv_median
        ),
        "hourly_sqv_median": None if np.isnan(hourly_sqv_median) else hourly_sqv_median,
        "wape_5min": wape,
        "hourly_wape_median": (
            None if np.isnan(hourly_wape_median) else hourly_wape_median
        ),
        "daily_total_error": daily_total_error,
        "notes": "",
    }


def compute_metrics(
    vehicle_counts: dict[str, np.ndarray],
    detector_input_counts: dict[str, np.ndarray] | None,
    interval_seconds: int,
) -> dict[str, float | str | None] | None:
    if detector_input_counts is None:
        return None

    return compute_total_metrics(
        sum_vehicle_counts(vehicle_counts),
        sum_vehicle_counts(detector_input_counts),
        interval_seconds,
    )


def format_metrics_text(
    metrics: dict[str, float | str | None] | None,
    interval_seconds: int,
    comparison_label: str,
    include_header: bool = True,
) -> str | None:
    if metrics is None:
        return None

    def fmt_decimal(value: float) -> str:
        return "n/a" if np.isnan(value) else f"{value:.2f}"

    def fmt_percent(value: float) -> str:
        return "n/a" if np.isnan(value) else f"{value * 100:.1f}%"

    def normalize_float(value: float | str | None) -> float:
        if value is None:
            return float("nan")
        if isinstance(value, str):
            return float("nan")
        return float(value)

    notes = metrics.get("notes")
    if notes:
        if include_header:
            return f"{comparison_label}\n{notes}"
        return str(notes)

    lines = [
        f"{interval_seconds // 60}-min SQV median: {fmt_decimal(normalize_float(metrics['sqv_interval_median']))}",
        f"Hourly SQV median: {fmt_decimal(normalize_float(metrics['hourly_sqv_median']))}",
        f"5-min WAPE: {fmt_percent(normalize_float(metrics['wape_5min']))}",
        f"Hourly WAPE median: {fmt_percent(normalize_float(metrics['hourly_wape_median']))}",
        f"Daily total error: {fmt_percent(normalize_float(metrics['daily_total_error']))}",
    ]
    if include_header:
        lines = [comparison_label, "all classes combined", *lines]
    return "\n".join(lines)


def render_plot(
    detector_id: str,
    vehicle_counts: dict[str, np.ndarray],
    detector_input_counts: dict[str, np.ndarray] | None,
    simulation_counts: np.ndarray | None,
    simulation_modeled_total: np.ndarray,
    simulation_input_total: np.ndarray | None,
    simulation_comparison_label: str | None,
    output_path: Path,
    interval_seconds: int,
) -> None:
    num_bins = next(iter(vehicle_counts.values())).shape[0]
    hours = np.arange(num_bins) * interval_seconds / 3600.0
    stacked_values = np.vstack(
        [vehicle_counts[vehicle_type] for vehicle_type in VEHICLE_TYPES]
    )
    colors = [VEHICLE_COLORS[vehicle_type] for vehicle_type in VEHICLE_TYPES]
    labels = [VEHICLE_LABELS[vehicle_type] for vehicle_type in VEHICLE_TYPES]
    cumulative = np.cumsum(stacked_values, axis=0)

    fig, (input_ax, simulation_ax, input_simulation_ax) = plt.subplots(
        1,
        3,
        figsize=(28, 7),
        sharey=True,
    )

    input_ax.stackplot(hours, stacked_values, labels=labels, colors=colors, alpha=0.88)
    for idx, vehicle_type in enumerate(VEHICLE_TYPES):
        input_ax.plot(
            hours,
            cumulative[idx],
            color=VEHICLE_COLORS[vehicle_type],
            linewidth=1.3,
        )

    if detector_input_counts is not None:
        detector_stacked_values = np.vstack(
            [detector_input_counts[vehicle_type] for vehicle_type in VEHICLE_TYPES]
        )
        detector_cumulative = np.cumsum(detector_stacked_values, axis=0)
        for idx, vehicle_type in enumerate(VEHICLE_TYPES):
            input_ax.plot(
                hours,
                detector_cumulative[idx],
                color=VEHICLE_COLORS[vehicle_type],
                linewidth=2.0,
                linestyle=INPUT_LINE_STYLE,
                alpha=0.95,
                label=f"Input {VEHICLE_LABELS[vehicle_type]}",
            )

    input_metrics = compute_metrics(
        vehicle_counts, detector_input_counts, interval_seconds
    )
    input_metrics_text = format_metrics_text(
        input_metrics,
        interval_seconds,
        "Metrics vs input",
        include_header=False,
    )
    if input_metrics_text is not None:
        input_ax.text(
            0.995,
            0.995,
            input_metrics_text,
            transform=input_ax.transAxes,
            ha="right",
            va="top",
            fontsize=11,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#d1d5db",
                "alpha": 0.92,
            },
        )

    simulation_ax.plot(
        hours,
        simulation_modeled_total,
        color="#111827",
        linewidth=2.1,
        label=(
            "Total of Flowrouter Flows"
            if simulation_comparison_label is None
            else f"{simulation_comparison_label} Total of Flowrouter Flows"
        ),
    )
    smoothed_simulation_counts = None
    smoothing_minutes = None
    if simulation_counts is not None:
        smoothed_simulation_counts, smoothing_minutes = smooth_simulation_counts(
            simulation_counts,
            interval_seconds,
        )
        simulation_ax.plot(
            hours,
            smoothed_simulation_counts,
            color="#be123c",
            linewidth=2.5,
            linestyle=SIMULATION_LINE_STYLE,
            alpha=0.98,
            label=f"Simulation Total ({smoothing_minutes}-min rolling avg)",
        )

    simulation_metrics = None
    if smoothed_simulation_counts is not None:
        simulation_metrics = compute_total_metrics(
            smoothed_simulation_counts,
            simulation_modeled_total,
            interval_seconds,
        )
    simulation_metrics_text = format_metrics_text(
        simulation_metrics,
        interval_seconds,
        "Metrics vs simulation",
        include_header=False,
    )
    if simulation_metrics_text is not None:
        simulation_ax.text(
            0.995,
            0.995,
            simulation_metrics_text,
            transform=simulation_ax.transAxes,
            ha="right",
            va="top",
            fontsize=11,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#d1d5db",
                "alpha": 0.92,
            },
        )

    if simulation_input_total is not None:
        input_simulation_ax.plot(
            hours,
            simulation_input_total,
            color="#111827",
            linewidth=2.1,
            label=(
                "Input Total"
                if simulation_comparison_label is None
                else f"{simulation_comparison_label} Input Total"
            ),
        )
    if smoothed_simulation_counts is not None:
        input_simulation_ax.plot(
            hours,
            smoothed_simulation_counts,
            color="#be123c",
            linewidth=2.5,
            linestyle=SIMULATION_LINE_STYLE,
            alpha=0.98,
            label=f"Simulation Total ({smoothing_minutes}-min rolling avg)",
        )

    input_simulation_metrics = None
    if smoothed_simulation_counts is not None and simulation_input_total is not None:
        input_simulation_metrics = compute_total_metrics(
            smoothed_simulation_counts,
            simulation_input_total,
            interval_seconds,
        )
    input_simulation_metrics_text = format_metrics_text(
        input_simulation_metrics,
        interval_seconds,
        "Metrics vs input total",
    )
    if input_simulation_metrics_text is not None:
        input_simulation_ax.text(
            0.995,
            0.995,
            input_simulation_metrics_text,
            transform=input_simulation_ax.transAxes,
            ha="right",
            va="top",
            fontsize=11,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#d1d5db",
                "alpha": 0.92,
            },
        )

    input_ax.set_title(
        "Aggregated Input Flows vs. Flowrouter Flow",
        fontsize=17,
        pad=12,
    )
    simulation_ax.set_title(
        "Flowrouter Flows vs. Simulation",
        fontsize=17,
        pad=12,
    )
    input_simulation_ax.set_title(
        "Aggregated Input Flows vs. Simulation",
        fontsize=17,
        pad=12,
    )
    input_ax.set_ylabel(
        f"Vehicles per {interval_seconds // 60}-minute interval",
        fontsize=13,
    )
    for ax in (input_ax, simulation_ax, input_simulation_ax):
        ax.set_xlabel("Time of day", fontsize=13)
        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 2))
        ax.set_xticklabels([format_hour_tick(hour) for hour in range(0, 25, 2)])
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(axis="y", color="#d1d5db", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    input_ax.legend(loc="upper left", ncol=1, frameon=False, fontsize=12)
    simulation_ax.legend(loc="upper left", frameon=False, fontsize=12)
    input_simulation_ax.legend(loc="upper left", frameon=False, fontsize=12)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_speed_plot(
    detector_group: str,
    input_lane_id: str,
    simulation_detector_id: str,
    input_speed_profiles: dict[str, list[float | None]] | None,
    simulation_speed_series: np.ndarray | None,
    output_path: Path,
    interval_seconds: int,
) -> tuple[bool, float | None]:
    if input_speed_profiles is None and simulation_speed_series is None:
        return False, None

    input_available = input_speed_profiles is not None and any(
        any(value is not None for value in values)
        for values in input_speed_profiles.values()
    )
    simulation_available = simulation_speed_series is not None and np.any(
        ~np.isnan(simulation_speed_series)
    )
    if not input_available and not simulation_available:
        return False, None

    num_bins = SECONDS_PER_DAY // interval_seconds
    x_values = list(range(num_bins))
    tick_step = max(1, 3600 // interval_seconds)
    tick_positions = list(range(0, num_bins, tick_step))
    tick_labels = [
        format_hour_tick((position * interval_seconds) // 3600)
        for position in tick_positions
    ]
    fig, ax = plt.subplots(figsize=(12, 7))

    plotted_values: list[float] = []
    workday_average_array: np.ndarray | None = None
    if input_available and input_speed_profiles is not None:
        for date_label, values in sorted(input_speed_profiles.items()):
            values_array = speed_profile_to_array(values)
            ax.plot(
                x_values,
                values_array,
                linewidth=1.0,
                alpha=0.45,
                label=date_label,
            )
            plotted_values.extend([value for value in values if value is not None])

        workday_average = average_speed_profiles(input_speed_profiles, {0, 1, 2, 3, 4})
        weekend_average = average_speed_profiles(input_speed_profiles, {5, 6})
        workday_average_array = speed_profile_to_array(workday_average)
        weekend_average_array = speed_profile_to_array(weekend_average)
        ax.plot(
            x_values,
            workday_average_array,
            linewidth=2.2,
            linestyle="--",
            color="#1f77b4",
            label="Workday average",
        )
        ax.plot(
            x_values,
            weekend_average_array,
            linewidth=2.2,
            linestyle="--",
            color="#d62728",
            label="Weekend average",
        )
        plotted_values.extend([value for value in workday_average if value is not None])
        plotted_values.extend([value for value in weekend_average if value is not None])

    if simulation_available and simulation_speed_series is not None:
        ax.plot(
            x_values,
            simulation_speed_series,
            color="#111827",
            linewidth=2.4,
            linestyle=SIMULATION_LINE_STYLE,
            label=f"Simulation ({simulation_detector_id})",
        )
        plotted_values.extend(
            [float(value) for value in simulation_speed_series if not np.isnan(value)]
        )

    min_speed = min(plotted_values) if plotted_values else 50.0
    max_speed = max(plotted_values) if plotted_values else 150.0
    y_min = min(50.0, np.floor(min_speed / 10.0) * 10.0)
    y_max = max(150.0, np.ceil(max_speed / 10.0) * 10.0)

    ax.set_ylabel("Speed [km/h]", fontsize=14)
    ax.set_xlabel("Time of day", fontsize=14)
    ax.set_xlim(0, num_bins - 1)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.tick_params(axis="both", labelsize=12)
    ax.set_ylim(y_min, y_max)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, ncols=2)

    speed_mae = compute_speed_mae(simulation_speed_series, workday_average_array)
    if speed_mae is not None:
        ax.text(
            0.995,
            0.995,
            f"MAE vs workday avg: {speed_mae:.2f} km/h",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=12,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#d1d5db",
                "alpha": 0.92,
            },
        )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True, speed_mae


def main() -> int:
    args = parse_args()

    if args.interval_seconds <= 0 or SECONDS_PER_DAY % args.interval_seconds != 0:
        raise ValueError("--interval-seconds must be a positive divisor of 86400.")

    emitter_files = [resolve_input_path(file_name) for file_name in args.emitter_files]
    missing_files = [str(path) for path in emitter_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing emitter file(s): {', '.join(missing_files)}")

    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    route_edges = load_route_edges(emitter_files)
    counts, num_bins = build_counts(emitter_files, args.interval_seconds, route_edges)
    detector_flows_csv = resolve_input_path(args.detector_flows_csv)
    if not detector_flows_csv.exists():
        raise FileNotFoundError(f"Missing detector flows CSV: {detector_flows_csv}")
    detector_input_counts = build_detector_input_counts(
        detector_flows_csv,
        args.interval_seconds,
        num_bins,
    )
    speed_input_dir = resolve_input_path(args.speed_input_dir)
    if not speed_input_dir.exists():
        raise FileNotFoundError(f"Missing speed input directory: {speed_input_dir}")
    speed_input_profiles, missing_speed_lanes = build_speed_input_profiles(
        speed_input_dir,
        args.interval_seconds,
        num_bins,
    )
    simulation_output_xml = resolve_input_path(args.simulation_output)
    if not simulation_output_xml.exists():
        raise FileNotFoundError(
            f"Missing simulation output XML: {simulation_output_xml}"
        )
    simulation_counts = build_simulation_counts(
        simulation_output_xml,
        args.interval_seconds,
        num_bins,
    )
    simulation_speed_series = build_simulation_speed_series(
        simulation_output_xml,
        args.interval_seconds,
        num_bins,
    )
    metrics_rows: list[dict[str, str | float]] = []

    for detector_id, vehicle_counts in counts.items():
        output_path = output_dir / f"{detector_id}_stacked_traffic.{args.format}"
        (
            simulation_modeled_total,
            simulation_counts_for_plot,
            simulation_comparison_label,
        ) = simulation_comparison_series(
            detector_id,
            vehicle_counts,
            counts,
            simulation_counts,
            num_bins,
        )
        metrics = compute_metrics(
            vehicle_counts,
            detector_input_counts.get(detector_id),
            args.interval_seconds,
        )
        simulation_input_total, _ = input_comparison_series(
            detector_id,
            detector_input_counts,
            num_bins,
        )
        smoothed_simulation_counts = None
        if simulation_counts_for_plot is not None:
            smoothed_simulation_counts, _ = smooth_simulation_counts(
                simulation_counts_for_plot,
                args.interval_seconds,
            )
        simulation_metrics = None
        if smoothed_simulation_counts is not None:
            simulation_metrics = compute_total_metrics(
                smoothed_simulation_counts,
                simulation_modeled_total,
                args.interval_seconds,
            )
        simulation_input_metrics = None
        if (
            smoothed_simulation_counts is not None
            and simulation_input_total is not None
        ):
            simulation_input_metrics = compute_total_metrics(
                smoothed_simulation_counts,
                simulation_input_total,
                args.interval_seconds,
            )
        plot_edge = DETECTOR_TO_PLOT_EDGE[detector_id]
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
        render_plot(
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
    for detector_group, lane_pairs in SPEED_DETECTOR_MAPPING.items():
        for input_lane_id, simulation_detector_id in lane_pairs:
            output_path = (
                speed_output_dir
                / f"{detector_group}_{input_lane_id}_speed.{args.format}"
            )
            created_plot, speed_mae = render_speed_plot(
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
                        "notes": "Speed lane plot",
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

    print(f"Wrote {len(counts)} plot(s) to {output_dir}")
    print(f"Wrote {speed_plot_count} speed plot(s) to {speed_output_dir}")
    if missing_speed_lanes:
        print(
            "Skipped missing speed input lane(s): "
            + ", ".join(sorted(missing_speed_lanes))
        )
    print(f"Wrote metrics CSV to {metrics_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
