#!/usr/bin/env python3
"""Thin, cross-platform task runner used locally and by GitHub Actions."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    command = [sys.executable, *args]
    print("+", " ".join(command), flush=True)
    environment = os.environ.copy()
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")
    subprocess.run(command, cwd=ROOT, check=True, env=environment)


def test() -> None:
    run("-m", "pytest", "-q")


def verify() -> None:
    """Run source-backed profile and Allan-model regression checks."""
    run("-m", "pytest", "-q", "tests/test_profile_verification.py", "tests/test_allan_parity.py")


def allan(duration: float, points: int, adis_duration: float) -> None:
    """Regenerate the standard Allan-deviation analysis artifacts."""
    run(
        "examples/allan_variance.py",
        "--duration",
        str(duration),
        "--points",
        str(points),
        "--profile",
        "profiles/test/short-correlation.json",
        "--profile",
        "profiles/test/flicker-band.json",
    )
    run(
        "examples/adis_allan_overlay.py",
        "--duration",
        str(adis_duration),
        "--points",
        str(points),
        "--output-dir",
        "artifacts/adis-allan-overlay",
    )


def showcase(allan_duration: float, reconstruction_duration: float, temperature_points: int) -> None:
    """Regenerate polished storefront-oriented showcase plots."""
    run(
        "examples/showcase.py",
        "--allan-duration",
        str(allan_duration),
        "--reconstruction-duration",
        str(reconstruction_duration),
        "--temperature-points",
        str(temperature_points),
    )


def analysis(
    duration: float,
    points: int,
    adis_duration: float,
    reconstruction_duration: float,
    temperature_points: int,
) -> None:
    """Regenerate the complete analysis and showcase artifact bundle."""
    allan(duration, points, adis_duration)
    showcase(duration, reconstruction_duration, temperature_points)


def coverage() -> None:
    run("-m", "coverage", "run", "-m", "pytest", "-q")
    run("-m", "coverage", "report", "--fail-under=85")
    run("-m", "coverage", "xml", "-o", "coverage.xml")


def lint() -> None:
    run("-m", "ruff", "check", "src", "tests", "scripts", "examples")


def build() -> None:
    # The dev environment already contains the declared build frontend/backend.
    # Avoiding a second isolated pip environment makes local builds work offline.
    run("-m", "build", "--no-isolation", "--sdist", "--wheel")


def docs(build_pdf: bool) -> None:
    source = ROOT / "docs" / "imu_error_model.tex"
    if not source.exists():
        raise FileNotFoundError(source)
    if not build_pdf:
        print(f"Documentation source present: {source}")
        return
    latex = shutil.which("latexmk") or shutil.which("pdflatex")
    if latex is None:
        raise RuntimeError("LaTeX builder not found; install latexmk/pdflatex or run docs without --build")
    output = ROOT / "docs" / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    if Path(latex).name == "latexmk":
        command = [latex, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-outdir=" + str(output), str(source)]
    else:
        command = [latex, "-interaction=nonstopmode", "-halt-on-error", "-output-directory=" + str(output), str(source)]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="task", required=True)
    for name in ("test", "verify", "coverage", "lint", "build", "allan", "showcase", "analysis", "all"):
        subparsers.add_parser(name)
    allan_parser = subparsers.choices["allan"]
    allan_parser.add_argument("--duration", type=float, default=120.0, help="all-profile record duration in seconds")
    allan_parser.add_argument("--points", type=int, default=24, help="number of logarithmic Allan points")
    allan_parser.add_argument("--adis-duration", type=float, default=2048.0, help="ADIS overlay record duration in seconds")
    showcase_parser = subparsers.choices["showcase"]
    showcase_parser.add_argument("--allan-duration", type=float, default=240.0)
    showcase_parser.add_argument("--reconstruction-duration", type=float, default=60.0)
    showcase_parser.add_argument("--temperature-points", type=int, default=120)
    analysis_parser = subparsers.choices["analysis"]
    analysis_parser.add_argument("--duration", type=float, default=240.0, help="Allan and showcase record duration in seconds")
    analysis_parser.add_argument("--points", type=int, default=24, help="number of logarithmic Allan points")
    analysis_parser.add_argument("--adis-duration", type=float, default=2048.0, help="ADIS overlay record duration in seconds")
    analysis_parser.add_argument("--reconstruction-duration", type=float, default=60.0)
    analysis_parser.add_argument("--temperature-points", type=int, default=120)
    docs_parser = subparsers.add_parser("docs")
    docs_parser.add_argument("--build", action="store_true", help="compile the LaTeX PDF")
    args = parser.parse_args()
    if args.task == "test":
        test()
    elif args.task == "verify":
        verify()
    elif args.task == "allan":
        if args.duration <= 0 or args.adis_duration <= 0 or args.points < 2:
            raise ValueError("Allan durations must be positive and points must be at least two")
        allan(args.duration, args.points, args.adis_duration)
    elif args.task == "showcase":
        if args.allan_duration <= 0 or args.reconstruction_duration <= 0 or args.temperature_points < 2:
            raise ValueError("showcase durations must be positive and temperature-points must be at least two")
        showcase(args.allan_duration, args.reconstruction_duration, args.temperature_points)
    elif args.task == "analysis":
        if (
            args.duration <= 0
            or args.adis_duration <= 0
            or args.points < 2
            or args.reconstruction_duration <= 0
            or args.temperature_points < 2
        ):
            raise ValueError(
                "analysis durations must be positive, points must be at least two, and temperature-points must be at least two"
            )
        analysis(
            args.duration,
            args.points,
            args.adis_duration,
            args.reconstruction_duration,
            args.temperature_points,
        )
    elif args.task == "coverage":
        coverage()
    elif args.task == "lint":
        lint()
    elif args.task == "build":
        build()
    elif args.task == "docs":
        docs(args.build)
    else:
        lint()
        verify()
        coverage()
        build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
