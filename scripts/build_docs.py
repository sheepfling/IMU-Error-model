#!/usr/bin/env python3
"""Regenerate showcase figures and build both repository PDF documents."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

def run(*arguments: str) -> None:
    subprocess.run([sys.executable, "scripts/ci.py", *arguments], cwd=ROOT, check=True)
####


def main() -> int:
    """Refresh generated figures, then compile both LaTeX documents."""
    run("showcase")
    run("docs", "--build")
    return 0
####


if __name__ == "__main__":
    raise SystemExit(main())
####
