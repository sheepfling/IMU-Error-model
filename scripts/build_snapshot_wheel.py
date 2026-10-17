"""Build a named local-development wheel from a Git-free reviewed snapshot.

The source keeps its upstream setuptools-scm release mechanism. Only the
throwaway build directory receives an explicit local version. Requires recent
setuptools with its built-in bdist_wheel command; does not install dependencies.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from shutil import copy2, copytree
import subprocess
import sys
from tempfile import TemporaryDirectory


SNAPSHOT_VERSION = "0.1.3+calibration.t8"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("dist"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix="imu-snapshot-build-") as directory:
        stage = Path(directory)
        copytree(root/"src",stage/"src",ignore=lambda path,names:[n for n in names if n == "__pycache__" or n.endswith(".egg-info")])
        for name in ("README.md","LICENSE"):
            copy2(root/name,stage/name)
        ####
        text = (root/"pyproject.toml").read_text()
        if 'dynamic = ["version"]' not in text:
            raise ValueError("snapshot builder expects the upstream dynamic version field")
        ####
        text = text.replace('dynamic = ["version"]',f'version = "{SNAPSHOT_VERSION}"')
        text = text.replace('"setuptools-scm[simple]>=8"','"setuptools>=80"')
        (stage/"pyproject.toml").write_text(text)
        subprocess.run(
            [sys.executable,"-c",f"from setuptools.build_meta import build_wheel; build_wheel({str(output)!r})"],
            cwd=stage,check=True,
        )
    ####
    return 0
####


if __name__ == "__main__":
    raise SystemExit(main())
####
