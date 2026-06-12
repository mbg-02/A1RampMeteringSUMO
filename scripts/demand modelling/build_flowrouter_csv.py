#!/usr/bin/env python3

"""Build a flowrouter-compatible CSV from typed detector definitions.

The script creates a full-day template at a configurable interval and can
overlay detector counts from the two-row CSV exports in ``data/``.  The current
export format stores one header row with detector/ramp labels and one header
row with vehicle classes.  Vehicle classes are read by name, so the column
order may differ between files.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

VEHICLE_CLASSES = ("bus", "motc", "car", "deliv", "truck")
VEHICLE_CLASS_LABELS = {
    "BUS": "bus",
    "MOTC": "motc",
    "MOTO": "motc",
    "MOTORCYCLE": "motc",
    "CAR": "car",
    "PKW": "car",
    "DELIV": "deliv",
    "DELIVERY": "deliv",
    "TRUCK": "truck",
    "LKW": "truck",
}
FLOW_COLUMNS = {
    "bus": "qBus",
    "motc": "qMotc",
    "car": "qCar",
    "deliv": "qDeliv",
    "truck": "qTruck",
}


@dataclass(frozen=True)
class FlowRow:
    detector: str
    time_min: float
    flows: dict[str, float] = field(
        default_factory=lambda: {vehicle_class: 0.0 for vehicle_class in VEHICLE_CLASSES}
    )

    @property
    def q_all(self) -> float:
        return sum(self.flows.get(vehicle_class, 0.0) for vehicle_class in VEHICLE_CLASSES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detectors",
        default="detDFRgen.detectors.xml",
        help="Typed detector file produced for dfrouter/flowrouter.",
    )
    parser.add_argument(
        "--output",
        default="flows_detFWR_2040.csv",
        help="Destination CSV path.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help=(
            "Directory to scan for two-row classed CSV exports when no explicit "
            "import files are given."
        ),
    )
    parser.add_argument(
        "--no-auto-import",
        action="store_true",
        help="Only write the blank detector/time template unless import files are given.",
    )
    parser.add_argument(
        "--import-detector-csv",
        action="append",
        default=[],
        help="Optional mainline detector export CSV to aggregate into the output template.",
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=5,
        help="Output aggregation interval in minutes.",
    )
    parser.add_argument(
        "--import-ramp-csv",
        action="append",
        default=[],
        help="Optional ramp export CSV to aggregate into the output template.",
    )
    parser.add_argument(
        "--import-hourly-csv",
        action="append",
        default=[],
        help="Optional hourly detector CSV to spread across 5-minute bins.",
    )
    return parser.parse_args()


def read_detector_ids(detector_file: Path) -> list[str]:
    root = ET.parse(detector_file).getroot()
    detector_ids = []
    for detector in root.findall("detectorDefinition"):
        detector_ids.append(detector.attrib["id"])
    if not detector_ids:
        raise RuntimeError(f"No detectorDefinition entries found in {detector_file}")
    return sorted(detector_ids)


def blank_rows(detector_ids: list[str], interval_minutes: int) -> dict[tuple[str, int], FlowRow]:
    rows: dict[tuple[str, int], FlowRow] = {}
    for detector_id in detector_ids:
        for time_min in range(0, 24 * 60, interval_minutes):
            rows[(detector_id, time_min)] = FlowRow(detector=detector_id, time_min=time_min)
    return rows


def normalize_ramp_header(label: str) -> str:
    match = re.match(r"\s*(\d+)\s+([BZ])\s+(on|off)(?:\s+(\d+))?\s*(?:\([^)]*\))?\s*$", label)
    if not match:
        raise ValueError(f"Unsupported ramp header label: {label!r}")
    number, direction, mode, suffix = match.groups()
    if number == "50" and direction == "B" and mode == "off":
        return "50_B_off"
    result = f"{number}_{direction}_{mode}"
    if suffix:
        result += f"_{suffix}"
    return result


def normalize_detector_header(label: str) -> str:
    match = re.match(r"\s*(\d+)\s+([BZ])(?:\s+\d+\s*min)?\s*(?:\([^)]*\))?\s*$", label)
    if not match:
        raise ValueError(f"Unsupported detector header label: {label!r}")
    number, direction = match.groups()
    return f"{number}_{direction}"


def normalize_measurement_header(label: str) -> str:
    try:
        return normalize_ramp_header(label)
    except ValueError:
        return normalize_detector_header(label)


def normalize_hourly_header(label: str) -> str:
    match = re.match(r"\s*(\d+)\s+([BZ])\s*$", label)
    if not match:
        raise ValueError(f"Unsupported hourly header label: {label!r}")
    number, direction = match.groups()
    return f"{number}_{direction}"


def normalize_vehicle_class(label: str) -> str | None:
    key = re.sub(r"[^A-Z0-9]", "", label.upper())
    return VEHICLE_CLASS_LABELS.get(key)


def parse_time_minutes(value: str, fallback: int) -> float:
    value = value.strip()
    if not value:
        return float(fallback)
    if ":" not in value:
        return float(value)

    parts = [float(part) for part in value.split(":")]
    if len(parts) == 2:
        hours, minutes = parts
        return hours * 60 + minutes
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 60 + minutes + seconds / 60
    raise ValueError(f"Unsupported time value: {value!r}")


def fmt(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def find_measurement_columns(
    top_header: list[str],
    class_header: list[str],
    normalizer,
) -> list[tuple[str, dict[str, int]]]:
    starts: list[tuple[str, int]] = []
    for idx, cell in enumerate(top_header):
        label = cell.strip()
        if not label or label.lower() == "time":
            continue
        starts.append((normalizer(label), idx))

    detector_columns: list[tuple[str, dict[str, int]]] = []
    for start_index, (detector_id, start_col) in enumerate(starts):
        end_col = starts[start_index + 1][1] if start_index + 1 < len(starts) else len(top_header)
        class_columns: dict[str, int] = {}
        for col in range(start_col, end_col):
            if col >= len(class_header):
                continue
            vehicle_class = normalize_vehicle_class(class_header[col])
            if vehicle_class is not None:
                class_columns[vehicle_class] = col
        missing = [vehicle_class for vehicle_class in VEHICLE_CLASSES if vehicle_class not in class_columns]
        if missing:
            raise ValueError(
                f"{detector_id} is missing vehicle class columns: {', '.join(missing)}"
            )
        detector_columns.append((detector_id, class_columns))

    if not detector_columns:
        raise ValueError("No detector/ramp columns found.")
    return detector_columns


def is_classed_export(path: Path) -> bool:
    try:
        with path.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.reader(fh)
            next(reader)
            class_header = next(reader)
    except (OSError, StopIteration, UnicodeDecodeError):
        return False
    classes = {normalize_vehicle_class(cell) for cell in class_header}
    return set(VEHICLE_CLASSES).issubset(classes)


def discover_classed_exports(data_dir: Path, output_file: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    output_file = output_file.resolve()
    result = []
    for path in sorted(data_dir.glob("*.csv")):
        if path.resolve() == output_file:
            continue
        if is_classed_export(path):
            result.append(path)
    return result


def import_classed_csv(
    path: Path,
    interval_minutes: int,
    normalizer=normalize_measurement_header,
) -> dict[tuple[str, int], FlowRow]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        top_header = next(reader)
        class_header = next(reader)

        detector_columns = find_measurement_columns(top_header, class_header, normalizer)

        aggregated: dict[tuple[str, int], dict[str, float]] = defaultdict(
            lambda: {vehicle_class: 0.0 for vehicle_class in VEHICLE_CLASSES}
        )

        for row_index, row in enumerate(reader):
            if not row or not row[0].strip():
                continue
            time_min = int(parse_time_minutes(row[0], row_index * interval_minutes))
            for detector_id, class_columns in detector_columns:
                bucket = aggregated[(detector_id, time_min)]
                for vehicle_class, col in class_columns.items():
                    value = row[col].strip() if col < len(row) else ""
                    if value:
                        bucket[vehicle_class] += float(value)

    result: dict[tuple[str, int], FlowRow] = {}
    for (detector_id, time_min), values in aggregated.items():
        result[(detector_id, time_min)] = FlowRow(detector=detector_id, time_min=time_min, flows=dict(values))
    return result


def import_detector_csv(path: Path, interval_minutes: int) -> dict[tuple[str, int], FlowRow]:
    return import_classed_csv(path, interval_minutes, normalize_detector_header)


def import_ramp_csv(path: Path, interval_minutes: int) -> dict[tuple[str, int], FlowRow]:
    return import_classed_csv(path, interval_minutes, normalize_ramp_header)


def import_hourly_csv(path: Path, interval_minutes: int) -> dict[tuple[str, int], FlowRow]:
    if 60 % interval_minutes != 0:
        raise ValueError("Hourly import requires an interval that divides 60 minutes evenly.")

    bins_per_hour = 60 // interval_minutes

    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        top_header = next(reader)
        class_header = next(reader)

        detector_columns = find_measurement_columns(top_header, class_header, normalize_hourly_header)

        result: dict[tuple[str, int], FlowRow] = {}
        for hour_index, row in enumerate(reader):
            if not row or not any(cell.strip() for cell in row):
                continue
            for detector_id, class_columns in detector_columns:
                distributed: dict[str, list[int]] = {}
                for vehicle_class, col in class_columns.items():
                    hourly_value = int(float(row[col])) if col < len(row) and row[col].strip() else 0
                    base = hourly_value // bins_per_hour
                    remainder = hourly_value % bins_per_hour
                    distributed[vehicle_class] = [
                        base + (1 if idx < remainder else 0) for idx in range(bins_per_hour)
                    ]
                for bin_index, offset in enumerate(range(0, 60, interval_minutes)):
                    time_min = hour_index * 60 + offset
                    result[(detector_id, time_min)] = FlowRow(
                        detector=detector_id,
                        time_min=time_min,
                        flows={
                            vehicle_class: distributed[vehicle_class][bin_index]
                            for vehicle_class in VEHICLE_CLASSES
                        },
                    )
    return result


def merge_rows(base: dict[tuple[str, int], FlowRow], overlay: dict[tuple[str, int], FlowRow]) -> None:
    for key, row in overlay.items():
        target_key = key
        if target_key not in base and not key[0].endswith("_SP"):
            target_key = (f"{key[0]}_SP", key[1])
        if target_key not in base:
            sys.stderr.write(f"Warning: imported detector/time not found in template: {key}\n")
            continue
        base[target_key] = FlowRow(
            detector=target_key[0],
            time_min=row.time_min,
            flows=dict(row.flows),
        )


def write_rows(rows: dict[tuple[str, int], FlowRow], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(
            ["Detector", "Time"]
            + [FLOW_COLUMNS[vehicle_class] for vehicle_class in VEHICLE_CLASSES]
            + ["qAll"]
        )
        for _, row in sorted(rows.items(), key=lambda item: (item[0][0], item[0][1])):
            writer.writerow(
                [
                    row.detector,
                    fmt(row.time_min),
                    *[
                        fmt(row.flows.get(vehicle_class, 0.0))
                        for vehicle_class in VEHICLE_CLASSES
                    ],
                    fmt(row.q_all),
                ]
            )


def main() -> int:
    args = parse_args()
    detector_file = Path(args.detectors)
    output_file = Path(args.output)

    rows = blank_rows(read_detector_ids(detector_file), args.interval_minutes)

    if (
        not args.no_auto_import
        and not args.import_detector_csv
        and not args.import_hourly_csv
        and not args.import_ramp_csv
    ):
        for path in discover_classed_exports(Path(args.data_dir), output_file):
            merge_rows(rows, import_classed_csv(path, args.interval_minutes))

    for detector_csv in args.import_detector_csv:
        merge_rows(rows, import_detector_csv(Path(detector_csv), args.interval_minutes))

    for hourly_csv in args.import_hourly_csv:
        merge_rows(rows, import_hourly_csv(Path(hourly_csv), args.interval_minutes))

    for ramp_csv in args.import_ramp_csv:
        merge_rows(rows, import_ramp_csv(Path(ramp_csv), args.interval_minutes))

    write_rows(rows, output_file)
    print(f"Wrote {len(rows)} rows to {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
