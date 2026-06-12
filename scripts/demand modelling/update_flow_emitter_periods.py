#!/usr/bin/env python3

"""Replace flow end attributes with period="exp(X)" from a CSV mapping."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import re


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

FLOW_ID_RE = re.compile(r'\bid="([^"]+)"')
END_ATTR_RE = re.compile(r'\s+end="[^"]*"')
PERIOD_ATTR_RE = re.compile(r'\s+period="[^"]*"')
PERIOD_EXP_RE = re.compile(r'\s+period="exp\(([^"]+)\)"')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mapping-csv",
        default="flow_emitters_numbers_5classes.csv",
        help="CSV with source_file, flow_id and X columns.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report changes without writing files.",
    )
    parser.add_argument(
        "emitter_files",
        nargs="*",
        default=DEFAULT_EMITTER_FILES,
        help="Emitter XML files to update. Defaults to the five class files.",
    )
    return parser.parse_args()


def load_mapping(path: Path) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required_columns = {"source_file", "flow_id", "X"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing_list = ", ".join(sorted(missing_columns))
            raise ValueError(f"{path} is missing required columns: {missing_list}")

        for row_number, row in enumerate(reader, start=2):
            source_file = (row.get("source_file") or "").strip()
            flow_id = (row.get("flow_id") or "").strip()
            x_value = (row.get("X") or "").strip()
            if not source_file or not flow_id or not x_value:
                raise ValueError(f"Incomplete mapping row {row_number} in {path}")

            file_mapping = mapping.setdefault(source_file, {})
            previous = file_mapping.get(flow_id)
            if previous is not None and previous != x_value:
                raise ValueError(
                    f"Conflicting X values for {source_file} / {flow_id}: "
                    f"{previous} vs {x_value}"
                )
            file_mapping[flow_id] = x_value
    return mapping


def update_emitter_file(
    path: Path,
    file_mapping: dict[str, str],
    dry_run: bool,
) -> tuple[int, int]:
    original_text = path.read_text(encoding="utf-8")
    lines = original_text.splitlines(keepends=True)

    updated_lines: list[str] = []
    seen_flow_ids: set[str] = set()
    flow_count = 0
    change_count = 0

    for line_number, line in enumerate(lines, start=1):
        if "<flow " not in line:
            updated_lines.append(line)
            continue

        flow_count += 1
        id_match = FLOW_ID_RE.search(line)
        if id_match is None:
            raise ValueError(f"Missing flow id in {path} line {line_number}")

        flow_id = id_match.group(1)
        x_value = file_mapping.get(flow_id)
        if x_value is None:
            raise KeyError(f"No CSV mapping for {path.name} / {flow_id}")

        replacement = f' period="exp({x_value})"'
        updated_line = line
        if END_ATTR_RE.search(updated_line):
            updated_line = END_ATTR_RE.sub(replacement, updated_line, count=1)
        elif period_match := PERIOD_EXP_RE.search(updated_line):
            current_x_value = period_match.group(1)
            if not periods_match(current_x_value, x_value):
                updated_line = PERIOD_ATTR_RE.sub(replacement, updated_line, count=1)
        elif PERIOD_ATTR_RE.search(updated_line):
            updated_line = PERIOD_ATTR_RE.sub(replacement, updated_line, count=1)
        else:
            raise ValueError(
                f"Flow {flow_id} in {path} line {line_number} has neither end nor period"
            )

        if updated_line != line:
            change_count += 1
        updated_lines.append(updated_line)
        seen_flow_ids.add(flow_id)

    unused_flow_ids = sorted(set(file_mapping) - seen_flow_ids)
    if unused_flow_ids:
        preview = ", ".join(unused_flow_ids[:5])
        suffix = "" if len(unused_flow_ids) <= 5 else ", ..."
        raise ValueError(
            f"{path.name} has {len(unused_flow_ids)} mapping rows not found in XML: "
            f"{preview}{suffix}"
        )

    updated_text = "".join(updated_lines)
    if not dry_run and updated_text != original_text:
        path.write_text(updated_text, encoding="utf-8")

    return flow_count, change_count


def periods_match(current: str, expected: str) -> bool:
    try:
        current_value = float(current)
        expected_value = float(expected)
    except ValueError:
        return current == expected
    return math.isclose(current_value, expected_value, rel_tol=1e-9, abs_tol=1e-12)


def main() -> int:
    args = parse_args()
    mapping = load_mapping(Path(args.mapping_csv))

    total_flows = 0
    total_changes = 0
    for emitter_file in args.emitter_files:
        path = Path(emitter_file)
        file_mapping = mapping.get(path.name)
        if file_mapping is None:
            raise KeyError(f"No CSV mappings found for {path.name}")

        flow_count, change_count = update_emitter_file(path, file_mapping, args.dry_run)
        total_flows += flow_count
        total_changes += change_count
        status = "Would update" if args.dry_run else "Updated"
        print(f"{status} {path}: {change_count} flow(s) across {flow_count} entries")

    print(f"Processed {len(args.emitter_files)} file(s), {total_changes} updated flow(s) total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
