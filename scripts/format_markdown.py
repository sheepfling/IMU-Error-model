#!/usr/bin/env python3
"""Normalize the repository's Markdown whitespace and block spacing."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILES = (Path("README.md"), Path("CHANGELOG.md"), Path("examples/imu_profiles/README.md"))
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
HEADING = re.compile(r"^ {0,3}#{1,6}(?:\s|$)")
LIST_ITEM = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+")

def _read_text(path: Path) -> str:
    """Read Markdown without normalizing platform-specific newlines."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        return stream.read()
####


def _write_text(path: Path, text: str) -> None:
    """Write Markdown without translating platform-specific newlines."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)
####


def _is_fence(line: str) -> bool:
    """Return whether a line opens or closes a fenced code block."""
    return FENCE.match(line) is not None
####


def _is_heading(line: str) -> bool:
    """Return whether a line is an ATX heading outside a code fence."""
    return HEADING.match(line) is not None
####


def _is_list_item(line: str) -> bool:
    """Return whether a line starts a Markdown list item."""
    return LIST_ITEM.match(line) is not None
####


def format_markdown_text(text: str) -> str:
    """Normalize Markdown whitespace while leaving fenced code content intact."""
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    formatted: list[str] = []
    in_fence = False
    for original in lines:
        if _is_fence(original):
            formatted.append(original if in_fence else original.rstrip())
            in_fence = not in_fence
            continue
        ####
        if in_fence:
            formatted.append(original)
            continue
        ####
        line = original.rstrip()
        if not line.strip():
            if formatted and formatted[-1] != "":
                formatted.append("")
            ####
            continue
        ####
        if _is_heading(line):
            if formatted and formatted[-1] != "":
                formatted.append("")
            ####
        elif _is_list_item(line):
            if formatted and formatted[-1] != "" and not _is_list_item(formatted[-1]):
                formatted.append("")
            ####
        elif formatted and _is_heading(formatted[-1]):
            formatted.append("")
        ####
        formatted.append(line)
    ####
    while formatted and formatted[-1] == "":
        formatted.pop()
    ####
    return newline.join(formatted) + newline
####


def markdown_files(paths: list[Path]) -> list[Path]:
    """Expand explicit Markdown files and directories."""
    files: set[Path] = set()
    for path in paths:
        candidate = ROOT / path if not path.is_absolute() else path
        if candidate.is_file() and candidate.suffix.lower() == ".md":
            files.add(candidate)
        elif candidate.is_dir():
            files.update(candidate.rglob("*.md"))
        ####
    return sorted(files)
####


def format_markdown_files(paths: list[Path]) -> list[Path]:
    """Rewrite files that need Markdown formatting and return changed files."""
    changed: list[Path] = []
    for path in markdown_files(paths):
        original = _read_text(path)
        formatted = format_markdown_text(original)
        if formatted == original:
            continue
        ####
        _write_text(path, formatted)
        changed.append(path)
    ####
    return changed
####


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true", help="rewrite Markdown files in normalized form")
    parser.add_argument("paths", nargs="*", type=Path, default=list(DEFAULT_FILES))
    args = parser.parse_args()
    files = markdown_files(args.paths)
    changed = format_markdown_files(args.paths) if args.fix else []
    for path in changed:
        print(f"Formatted Markdown: {path.relative_to(ROOT)}")
    ####
    if not args.fix:
        pending = [path for path in files if format_markdown_text(_read_text(path)) != _read_text(path)]
        if pending:
            print("Markdown formatting needed; run scripts/format_markdown.py --fix")
            return 1
        ####
    ####
    return 0
####


if __name__ == "__main__":
    raise SystemExit(main())
####
