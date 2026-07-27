#!/usr/bin/env python3
"""Apply Ruff fixes and repository-specific Python formatting conventions."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOTS = ("src", "tests", "scripts", "examples")

def _run(*arguments: str) -> None:
    """Run one formatting command from the repository root."""
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)
####


def main() -> int:
    """Run Ruff's fixes, Ruff's formatter, and the scope-marker fixer."""
    _run("-m", "ruff", "check", "--fix", *PYTHON_ROOTS)
    _run("-m", "ruff", "format", *PYTHON_ROOTS)
    _run("scripts/check_scope_markers.py", "--fix", *PYTHON_ROOTS)
    _run("scripts/format_markdown.py", "--fix")
    return 0
####


if __name__ == "__main__":
    raise SystemExit(main())
####
