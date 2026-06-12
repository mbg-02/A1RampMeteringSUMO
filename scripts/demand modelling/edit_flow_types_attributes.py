#!/usr/bin/env python3

"""Insert vehicle type attributes into emitter flow definitions."""

from __future__ import annotations

from pathlib import Path
import re


FILE_TYPES = {
    "FLOWbus.emitters.xml": "bus",
    "FLOWbus2040.emitters.xml": "bus",
    "FLOWcar.emitters.xml": "car",
    "FLOWcar2040.emitters.xml": "car",
    "FLOWdeliv.emitters.xml": "deliv",
    "FLOWdeliv2040.emitters.xml": "deliv",
    "FLOWmotc.emitters.xml": "motc",
    "FLOWmotc2040.emitters.xml": "motc",
    "FLOWtruck.emitters.xml": "truck",
    "FLOWtruck2040.emitters.xml": "truck",
}
FLOW_ATTRIBUTES = {
    "departLane": "best",
    "departPos": "base",
    "departSpeed": "max",
    "insertionChecks": "collision",
}

ID_ATTR_RE = re.compile(r'(<flow\s+id="[^"]+")')
TYPE_ATTR_RE = re.compile(r'\s+type="[^"]+"')


def set_attribute(line: str, name: str, value: str) -> str:
    attr_re = re.compile(rf'\s+{re.escape(name)}="[^"]*"')
    replacement = f' {name}="{value}"'
    if attr_re.search(line):
        return attr_re.sub(replacement, line, count=1)
    updated_line, count = ID_ATTR_RE.subn(rf'\1{replacement}', line, count=1)
    if count == 0:
        raise ValueError(f"Could not insert attribute {name!r}")
    return updated_line


def update_file(path: Path, vehicle_type: str) -> int:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    updated_lines: list[str] = []
    changes = 0

    for line_number, line in enumerate(lines, start=1):
        if "<flow " not in line:
            updated_lines.append(line)
            continue

        if TYPE_ATTR_RE.search(line):
            updated_line = TYPE_ATTR_RE.sub(f' type="{vehicle_type}"', line, count=1)
        else:
            updated_line, count = ID_ATTR_RE.subn(
                rf'\1 type="{vehicle_type}"',
                line,
                count=1,
            )
            if count == 0:
                raise ValueError(f"Could not find flow id in {path} line {line_number}")

        for name, value in FLOW_ATTRIBUTES.items():
            updated_line = set_attribute(updated_line, name, value)

        if updated_line != line:
            changes += 1
        updated_lines.append(updated_line)

    updated_text = "".join(updated_lines)
    if updated_text != text:
        path.write_text(updated_text, encoding="utf-8")
    return changes


def main() -> int:
    total_changes = 0
    for file_name, vehicle_type in FILE_TYPES.items():
        path = Path(file_name)
        changes = update_file(path, vehicle_type)
        total_changes += changes
        print(f"Updated {path}: {changes} flow(s)")
    print(f"Updated {len(FILE_TYPES)} file(s), {total_changes} flow(s) total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
