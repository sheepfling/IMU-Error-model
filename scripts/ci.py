#!/usr/bin/env python3
"""Thin, cross-platform task runner used locally and by GitHub Actions."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]

def run(*args: str) -> None:
    command = [sys.executable, *args]
    print("+", " ".join(command), flush=True)
    environment = os.environ.copy()
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")
    subprocess.run(command, cwd=ROOT, check=True, env=environment)
####

def run_isolated(
        *args: str,
        pythonpath: Path | None = None,
        extra_environment: dict[str, str] | None = None,
) -> None:
    """Run a command without the repository source tree on its import path."""
    command = [sys.executable, *args]
    print("+", " ".join(command), flush=True)
    environment = os.environ.copy()
    if pythonpath is None:
        environment.pop("PYTHONPATH", None)
    else:
        environment["PYTHONPATH"] = str(pythonpath)
    ####
    if extra_environment is not None:
        environment.update(extra_environment)
    ####
    subprocess.run(command, cwd=ROOT, check=True, env=environment)
####

def test() -> None:
    run("-m", "pytest", "-q")
####

def verify() -> None:
    """Run source-backed profile and Allan-model regression checks."""
    run("-m", "pytest", "-q", "tests/test_profile_verification.py", "tests/test_allan_parity.py")
####

def allan(duration: float, points: int) -> None:
    """Regenerate the standard Allan-deviation analysis artifacts."""
    run(
        "examples/allan_variance.py",
        "--duration",
        str(duration),
        "--points",
        str(points),
        "--profile",
        "tests/profiles/test/short-correlation.json",
        "--profile",
        "tests/profiles/test/flicker-band.json",
    )
####

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
####

def analysis(
        duration: float,
        points: int,
        reconstruction_duration: float,
        temperature_points: int,
) -> None:
    """Regenerate the complete analysis and showcase artifact bundle."""
    allan(duration, points)
    showcase(duration, reconstruction_duration, temperature_points)
####

def coverage() -> None:
    run("-m", "coverage", "run", "-m", "pytest", "-q")
    run("-m", "coverage", "report", "--fail-under=85")
    run("-m", "coverage", "xml", "-o", "coverage.xml")
####

def lint() -> None:
    run("-m", "ruff", "check", "src", "tests", "scripts", "examples")
    run("scripts/check_scope_markers.py", "src", "scripts", "tests", "examples")
####

def format_sources() -> None:
    run("scripts/format.py")
####

def markdown() -> None:
    run("scripts/check_markdown.py")
####

def typecheck() -> None:
    run("-m", "pyright", "src", "examples")
    run("-m", "pyright", "-p", "pyright-testsconfig.json")
####

def build() -> None:
    # The dev environment already contains the declared build frontend/backend.
    # Avoiding a second isolated pip environment makes local builds work offline.
    run("-m", "build", "--no-isolation", "--sdist", "--wheel")
####

def package_smoke() -> None:
    """Build a wheel and test it from an isolated install target."""
    with TemporaryDirectory(prefix="imu-error-model-package-") as temporary:
        temporary_root = Path(temporary)
        wheel_directory = temporary_root / "wheel"
        wheel_directory.mkdir()
        run("-m", "build", "--no-isolation", "--wheel", "--outdir", str(wheel_directory))
        wheels = sorted(wheel_directory.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected exactly one wheel, found {len(wheels)}")
        ####
        wheel = wheels[0]
        package_root = ROOT / "src"
        expected_package_files = sorted(
            path.relative_to(package_root).as_posix()
            for path in (package_root / "imu_error_model").rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
        with ZipFile(wheel) as archive:
            wheel_files = set(archive.namelist())
        expected_files = set(expected_package_files)
        missing = sorted(path for path in expected_files if path not in wheel_files)
        if missing:
            raise RuntimeError(f"wheel is missing package data: {', '.join(missing)}")
        ####
        unexpected = sorted(
            path for path in wheel_files
            if path.startswith("imu_error_model/") and path not in expected_files
        )
        if unexpected:
            raise RuntimeError(f"wheel contains unexpected package files: {', '.join(unexpected)}")
        ####
        install_directory = temporary_root / "install"
        install_directory.mkdir()
        run_isolated(
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(install_directory),
            str(wheel),
        )
        run_isolated(
            "-m",
            "pytest",
            "-q",
            "-o",
            "pythonpath=",
            str(ROOT / "tests"),
            pythonpath=install_directory,
            extra_environment={"IMU_ERROR_MODEL_EXPECTED_PACKAGE_ROOT": str(install_directory)},
        )
####


def docs(build_pdf: bool) -> None:
    sources = [
        ROOT / "docs" / "imu_error_model.tex",
        ROOT / "docs" / "imu_profile_showcase.tex",
    ]
    missing = [source for source in sources if not source.exists()]
    if missing:
        raise FileNotFoundError(missing[0])
    ####
    if not build_pdf:
        for source in sources:
            print(f"Documentation source present: {source}")
        ####
        return
    ####
    latex = shutil.which("latexmk") or shutil.which("pdflatex")
    if latex is None:
        raise RuntimeError("LaTeX builder not found; install latexmk/pdflatex or run docs without --build")
    ####
    output = ROOT / "docs" / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    for source in sources:
        # Compile from the source directory so sibling inputs such as the
        # shared style fragment and bibliography resolve for direct users too.
        source_directory = source.parent
        if Path(latex).name == "latexmk":
            command = [latex, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-outdir=" + str(output),
                       source.name]
        else:
            command = [latex, "-interaction=nonstopmode", "-halt-on-error", "-output-directory=" + str(output),
                       source.name]
        ####
        print("+", " ".join(command), flush=True)
        subprocess.run(command, cwd=source_directory, check=True)
    ####
####

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="task", required=True)
    for name in ("test", "verify", "coverage", "lint", "format", "markdown", "typecheck", "build", "package", "allan", "showcase",
                 "analysis", "all"):
        subparsers.add_parser(name)
    ####
    allan_parser = subparsers.choices["allan"]
    allan_parser.add_argument("--duration", type=float, default=120.0, help="all-profile record duration in seconds")
    allan_parser.add_argument("--points", type=int, default=24, help="number of logarithmic Allan points")
    showcase_parser = subparsers.choices["showcase"]
    showcase_parser.add_argument("--allan-duration", type=float, default=240.0)
    showcase_parser.add_argument("--reconstruction-duration", type=float, default=60.0)
    showcase_parser.add_argument("--temperature-points", type=int, default=120)
    analysis_parser = subparsers.choices["analysis"]
    analysis_parser.add_argument("--duration", type=float, default=240.0,
                                 help="Allan and showcase record duration in seconds")
    analysis_parser.add_argument("--points", type=int, default=24, help="number of logarithmic Allan points")
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
        if args.duration <= 0 or args.points < 2:
            raise ValueError("Allan durations must be positive and points must be at least two")
        ####
        allan(args.duration, args.points)
    elif args.task == "showcase":
        if args.allan_duration <= 0 or args.reconstruction_duration <= 0 or args.temperature_points < 2:
            raise ValueError("showcase durations must be positive and temperature-points must be at least two")
        ####
        showcase(args.allan_duration, args.reconstruction_duration, args.temperature_points)
    elif args.task == "analysis":
        if (
                args.duration <= 0
                or args.points < 2
                or args.reconstruction_duration <= 0
                or args.temperature_points < 2
        ):
            raise ValueError(
                "analysis durations must be positive, points must be at least two, and temperature-points must be at least two"
            )
        ####
        analysis(
            args.duration,
            args.points,
            args.reconstruction_duration,
            args.temperature_points,
        )
    elif args.task == "coverage":
        coverage()
    elif args.task == "lint":
        lint()
    elif args.task == "format":
        format_sources()
    elif args.task == "markdown":
        markdown()
    elif args.task == "typecheck":
        typecheck()
    elif args.task == "build":
        build()
    elif args.task == "package":
        package_smoke()
    elif args.task == "docs":
        docs(args.build)
    else:
        lint()
        markdown()
        typecheck()
        verify()
        coverage()
        build()
        package_smoke()
    ####
    return 0
####

if __name__ == "__main__":
    raise SystemExit(main())
####
