#!/usr/bin/env python3
"""Check Markdown links, ownership, and ignored-source cleanliness."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL_ALLOWLIST = {Path("README.md"), Path("CHANGELOG.md")}
FORBIDDEN = re.compile(r"(?:INBOX/|research/sources/|manual|\.pdf|source document|controlled source)", re.IGNORECASE)
LINK = re.compile(r"(?<!!)(?:\[[^\]]*\])\(([^)]+)\)")
SOURCE_SUFFIXES = {".c", ".cfg", ".cpp", ".h", ".json", ".py", ".tex", ".toml", ".yaml", ".yml"}


def markdown_files() -> list[Path]:
    return sorted(
        path.relative_to(ROOT)
        for path in ROOT.rglob("*.md")
        if ".git" not in path.parts and ".pytest_cache" not in path.parts
    )


def link_target(raw: str) -> str:
    target = raw.strip().strip("<>").split(" ", 1)[0]
    return target


def source_files() -> list[Path]:
    return sorted(
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SOURCE_SUFFIXES
        and ".git" not in path.parts
        and ".pytest_cache" not in path.parts
        and path.relative_to(ROOT) != Path("scripts/check_markdown.py")
        and path.relative_to(ROOT) != Path(".gitignore")
        and ".github" not in path.parts
    )


def check() -> list[str]:
    files = markdown_files()
    errors: list[str] = []
    linked: set[Path] = set()

    for relative in files:
        path = ROOT / relative
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if FORBIDDEN.search(line):
                errors.append(f"{relative}:{line_number}: forbidden source-document reference")
            for match in LINK.finditer(line):
                target = link_target(match.group(1))
                parsed = urlsplit(target)
                if parsed.scheme or target.startswith("//"):
                    errors.append(f"{relative}:{line_number}: link is not relative: {target}")
                    continue
                target_path = target.split("#", 1)[0]
                if not target_path:
                    continue
                resolved = (path.parent / target_path).resolve()
                try:
                    resolved.relative_to(ROOT)
                except ValueError:
                    errors.append(f"{relative}:{line_number}: link escapes repository: {target}")
                    continue
                if not resolved.exists():
                    errors.append(f"{relative}:{line_number}: link target does not exist: {target}")
                    continue
                ignored = resolved.relative_to(ROOT).as_posix()
                if ignored.startswith(("INBOX/", "research/sources/", "research/artifacts/")):
                    errors.append(f"{relative}:{line_number}: link target is gitignored: {target}")
                    continue
                resolved_relative = resolved.relative_to(ROOT)
                if resolved_relative.suffix.lower() == ".md":
                    linked.add(resolved_relative)

    for relative in source_files():
        path = ROOT / relative
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if FORBIDDEN.search(line):
                errors.append(f"{relative}:{line_number}: forbidden ignored-source or document reference")

    for relative in files:
        if relative not in TOP_LEVEL_ALLOWLIST and relative not in linked:
            errors.append(f"{relative}: orphan Markdown file")
    return errors


def main() -> int:
    errors = check()
    if errors:
        print("Markdown cleanliness check failed:", file=sys.stderr)
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Markdown cleanliness check passed ({len(markdown_files())} files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
