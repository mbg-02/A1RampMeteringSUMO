#!/usr/bin/env python3

"""Extract flow numbers and insertion rates from emitter XML files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import xml.etree.ElementTree as ET


DEFAULT_EMITTER_FILES = [
    "FLOWbus.emitters.xml",
    "FLOWbus2040.emitters.xml",
    "FLOWcar.emitters.xml",
    "FLOWcar2040.emitters.xml",
    "FLOWdeliv.emitters.xml",
    "FLOWdeliv2040.emitters.xml",
    "FLOWmotc.emitters.xml",
    "FLOWmotc2040.emitters.xml",
    "FLOWtruck.emitters.xml",
    "FLOWtruck2040.emitters.xml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="flow_emitters_numbers_5classes.csv",
        help="Destination CSV path.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=300.0,
        help="Flow interval in seconds used to compute X = number / interval.",
    )
    parser.add_argument(
        "emitter_files",
        nargs="*",
        default=DEFAULT_EMITTER_FILES,
        help="Emitter XML files to read. Defaults to the five class files.",
    )
    return parser.parse_args()


def fmt(value: float) -> str:
    return f"{value:.15g}"


def iter_flow_rows(path: Path, interval_seconds: float):
    root = ET.parse(path).getroot()
    for flow in root.findall("flow"):
        number = float(flow.attrib["number"])
        yield [
            path.name,
            flow.attrib["id"],
            fmt(number),
            fmt(number / interval_seconds),
        ]


def main() -> int:
    args = parse_args()
    output = Path(args.output)

    if args.interval_seconds <= 0:
        raise ValueError("--interval-seconds must be positive.")

    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source_file", "flow_id", "number", "X"])
        for emitter_file in args.emitter_files:
            writer.writerows(iter_flow_rows(Path(emitter_file), args.interval_seconds))

    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
