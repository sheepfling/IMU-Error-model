#!/usr/bin/env python3
"""Compare the ADIS16470 profile with approximate datasheet Allan curves."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from allan_variance import allan_deviation, cluster_sizes_for_scale
from imu_error_model import ImuModel, load_yaml_profile


G_TO_MPS2 = 9.80665
REFERENCE = Path("research/data/adis16470_allan_reference.csv")
PROFILE = Path("examples/imu_profiles/hardware-estimates/ADIS16470.yaml")


def simulate_rates(duration: float, sample_period: float, seed: int, flicker_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Generate ADIS rates at a practical comparison interval."""
    profile = load_yaml_profile(PROFILE)
    config = replace(
        profile.config,
        accelerometer=replace(
            profile.config.accelerometer,
            flicker_bias_std=profile.config.accelerometer.flicker_bias_std * flicker_scale,
        ),
        gyroscope=replace(
            profile.config.gyroscope,
            flicker_bias_std=profile.config.gyroscope.flicker_bias_std * flicker_scale,
        ),
    )
    model = ImuModel(config, rng=np.random.default_rng(seed))
    count = int(duration / sample_period)
    acceleration = np.empty(count, dtype=float)
    gyroscope = np.empty(count, dtype=float)
    for index in range(count + 1):
        output = model.measure(index * sample_period, np.zeros(3), np.eye(3))
        if output.dt:
            sample_index = index - 1
            acceleration[sample_index] = float(output.delta_v[0] / output.dt / profile.config.accelerometer.output_scale)
            gyroscope[sample_index] = float(output.delta_theta[0] / output.dt / profile.config.gyroscope.output_scale)
    return acceleration, gyroscope


def load_reference(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    curves: dict[str, list[tuple[float, float]]] = {"accelerometer": [], "gyroscope": []}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            curves[row["sensor"]].append((float(row["tau_s"]), float(row["deviation"])))
    return {
        sensor: (np.asarray([pair[0] for pair in points]), np.asarray([pair[1] for pair in points]))
        for sensor, points in curves.items()
    }


def write_overlay(rows: list[dict[str, object]], reference: dict[str, tuple[np.ndarray, np.ndarray]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "adis16470_allan_overlay.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sensor", "series", "tau_s", "deviation", "unit"])
        writer.writeheader()
        writer.writerows(rows)

    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plot generation requires the 'analysis' extra") from exc

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    labels = {
        "gyroscope": ("Gyroscope", "deg/hr"),
        "accelerometer": ("Accelerometer", "ug"),
    }
    for axis, sensor in zip(axes, ("gyroscope", "accelerometer")):
        title, unit = labels[sensor]
        generated = [row for row in rows if row["sensor"] == sensor and row["series"] == "generated"]
        axis.loglog(
            [row["tau_s"] for row in generated],
            [row["deviation"] for row in generated],
            "o-",
            label="Generated profile",
        )
        tau, deviation = reference[sensor]
        axis.loglog(tau, deviation, "k--", linewidth=1.5, label="Datasheet digitization")
        axis.set_title(title)
        axis.set_xlabel("Cluster period tau (s)")
        axis.set_ylabel(f"Allan deviation ({unit})")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize="small")
    figure.suptitle("ADIS16470 Allan-deviation comparison")
    figure.tight_layout()
    figure.savefig(output_dir / "adis16470_allan_overlay.png", dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=512.0)
    parser.add_argument("--sample-period-s", type=float, default=0.01)
    parser.add_argument("--points", type=int, default=28)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--flicker-scale", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/adis-allan-overlay"))
    args = parser.parse_args()
    if args.duration <= 0 or args.sample_period_s <= 0 or args.flicker_scale < 0:
        raise ValueError("duration and sample period must be positive; flicker scale cannot be negative")

    acceleration, gyroscope = simulate_rates(args.duration, args.sample_period_s, args.seed, args.flicker_scale)
    clusters = cluster_sizes_for_scale(args.sample_period_s, args.duration, 0.01, None, args.points)
    rows: list[dict[str, object]] = []
    for sensor, values, conversion, unit in (
        ("accelerometer", acceleration, 1e6 / G_TO_MPS2, "ug"),
        ("gyroscope", gyroscope, 180.0 / np.pi * 3600.0, "deg/hr"),
    ):
        tau, deviation = allan_deviation(values, args.sample_period_s, clusters)
        rows.extend(
            {
                "sensor": sensor,
                "series": "generated",
                "tau_s": float(tau_value),
                "deviation": float(deviation_value * conversion),
                "unit": unit,
            }
            for tau_value, deviation_value in zip(tau, deviation)
        )
    write_overlay(rows, load_reference(REFERENCE), args.output_dir)
    print(f"Wrote ADIS overlay to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
