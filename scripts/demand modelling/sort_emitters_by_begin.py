#!/usr/bin/env python3
"""Sort SUMO emitter XML files by each element's begin time.

The script keeps the original XML lines intact and only reorders complete
entries that contain a ``begin="..."`` attribute. Equal begin times keep their
original relative order.
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


DEFAULT_EMITTER_FILES = (
    "FLOWcar.emitters.xml",
    "FLOWbus.emitters.xml",
    "FLOWtruck.emitters.xml",
    "FLOWdeliv.emitters.xml",
    "FLOWmotc.emitters.xml",
)

BEGIN_RE = re.compile(r'\bbegin="([^"]+)"')
SELF_CLOSING_RE = re.compile(r"/>\s*$")


@dataclass(frozen=True)
class SortableBlock:
    begin: float
    original_index: int
    lines: list[str]


def parse_time_seconds(value: str) -> float:
    """Parse SUMO numeric times and HH:MM:SS-style times."""
    value = value.strip()
    if ":" not in value:
        return float(value)

    parts = [float(part) for part in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Unsupported time value: {value!r}")


def split_xml(lines: list[str]) -> tuple[list[str], list[SortableBlock], list[str]]:
    """Split an <additional> XML file into header, sortable blocks, and footer."""
    header: list[str] = []
    footer: list[str] = []
    blocks: list[SortableBlock] = []
    in_body = False
    current_block: list[str] = []

    for line in lines:
        stripped = line.strip()

        if not in_body:
            header.append(line)
            if stripped.startswith("<additional") and stripped.endswith(">"):
                in_body = True
            continue

        if stripped.startswith("</additional"):
            if current_block:
                blocks.append(block_from_lines(current_block, len(blocks)))
                current_block = []
            footer.append(line)
            in_body = False
            continue

        if not stripped:
            continue

        current_block.append(line)
        if SELF_CLOSING_RE.search(line):
            blocks.append(block_from_lines(current_block, len(blocks)))
            current_block = []

    if current_block:
        blocks.append(block_from_lines(current_block, len(blocks)))

    if not footer and in_body:
        raise ValueError("Missing closing </additional> tag")

    return header, blocks, footer


def block_from_lines(lines: list[str], original_index: int) -> SortableBlock:
    block_text = "".join(lines)
    match = BEGIN_RE.search(block_text)
    if not match:
        raise ValueError(f"Found an element without begin attribute: {block_text.strip()}")
    return SortableBlock(
        begin=parse_time_seconds(match.group(1)),
        original_index=original_index,
        lines=list(lines),
    )


def sort_emitter_file(path: Path, dry_run: bool, backup: bool) -> tuple[int, bool]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    header, blocks, footer = split_xml(lines)
    sorted_blocks = sorted(blocks, key=lambda block: (block.begin, block.original_index))
    changed = [block.original_index for block in sorted_blocks] != [
        block.original_index for block in blocks
    ]

    if changed and not dry_run:
        if backup:
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        sorted_lines = header + [
            line for block in sorted_blocks for line in block.lines
        ] + footer
        path.write_text("".join(sorted_lines), encoding="utf-8")

    return len(blocks), changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sort SUMO emitter files by the begin attribute."
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Emitter XML files to sort. Defaults to the five root FLOW*.emitters.xml files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report whether files would change.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write .bak files before modifying emitters.",
    )
    args = parser.parse_args()

    files = args.files or [Path(name) for name in DEFAULT_EMITTER_FILES]
    for path in files:
        if not path.exists():
            raise FileNotFoundError(path)
        count, changed = sort_emitter_file(
            path=path,
            dry_run=args.dry_run,
            backup=not args.no_backup,
        )
        action = "would sort" if args.dry_run and changed else "sorted"
        if not changed:
            action = "already sorted"
        print(f"{path}: {action} {count} entries")


if __name__ == "__main__":
    main()
