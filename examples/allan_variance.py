#!/usr/bin/env python3
"""Generate Allan-deviation data and charts for the example IMU estimates."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from imu_error_model import ImuModel, load_profile, load_yaml_profile


def allan_deviation(samples: np.ndarray, sample_period: float, cluster_sizes: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute non-overlapping Allan deviation for a scalar rate sequence."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 1 or samples.size < 4:
        raise ValueError("samples must be a one-dimensional sequence with at least four values")
    if sample_period <= 0:
        raise ValueError("sample_period must be positive")
    if cluster_sizes is None:
        maximum = samples.size // 4
        cluster_sizes = np.unique(np.geomspace(1, maximum, num=24).astype(int))
    taus: list[float] = []
    deviations: list[float] = []
    for cluster_size in cluster_sizes:
        if cluster_size < 1 or 2 * cluster_size > samples.size:
            continue
        count = samples.size // cluster_size
        means = samples[: count * cluster_size].reshape(count, cluster_size).mean(axis=1)
        if means.size < 2:
            continue
        taus.append(cluster_size * sample_period)
        deviations.append(float(np.sqrt(.5 * np.mean(np.diff(means) ** 2))))
    return np.asarray(taus), np.asarray(deviations)


def cluster_sizes_for_scale(sample_period: float, duration: float, minimum_tau: float | None, maximum_tau: float | None, points: int) -> np.ndarray:
    if points < 2:
        raise ValueError("points must be at least two")
    minimum = max(sample_period, minimum_tau or sample_period)
    maximum = min(duration / 4.0, maximum_tau or duration / 4.0)
    if maximum < minimum:
        raise ValueError("maximum Allan period must be at least the minimum period")
    return np.unique(np.geomspace(minimum / sample_period, maximum / sample_period, num=points).astype(int).clip(1))


def collect_profile_rates(path: Path, duration: float, seed: int) -> tuple[dict[str, str], np.ndarray, np.ndarray, float, dict[str, dict[str, float | None]]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = load_profile(path)
        dt = float(payload.get("sample_period_s", .01))
        identity = {
            "model_name": str(payload.get("model_name", path.stem)),
            "family": str(payload.get("family", "test")),
            "grade": str(payload.get("grade", "test")),
        }
    else:
        profile = load_yaml_profile(path)
        config = profile.config
        dt = profile.sample_period_s
        identity = {
            "model_name": profile.model_name,
            "family": str(profile.metadata.get("family", "unknown")),
            "grade": str(profile.metadata.get("grade", "unknown")),
        }
    count = int(duration / dt)
    model = ImuModel(config, rng=np.random.default_rng(seed))
    accel_scale = config.accelerometer.output_scale
    gyro_scale = config.gyroscope.output_scale
    accelerometer: list[float] = []
    gyroscope: list[float] = []
    for index in range(count + 1):
        output = model.measure(index * dt, np.zeros(3), np.eye(3))
        if output.dt:
            accelerometer.append(float((output.delta_v / output.dt / accel_scale)[0]))
            gyroscope.append(float((output.delta_theta / output.dt / gyro_scale)[0]))
    parameters = {
        "accelerometer": {
            "white_noise_density": config.accelerometer.white_noise_density,
            "bias_correlation_time": config.accelerometer.bias_correlation_time,
        },
        "gyroscope": {
            "white_noise_density": config.gyroscope.white_noise_density,
            "bias_correlation_time": config.gyroscope.bias_correlation_time,
        },
    }
    return identity, np.asarray(accelerometer), np.asarray(gyroscope), dt, parameters


def write_charts(rows: list[dict[str, object]], output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plot generation requires the 'analysis' extra: pip install -e '.[analysis]'") from exc
    for sensor, label, unit in (
        ("accelerometer", "Accelerometer Allan deviation", "m/s²"),
        ("gyroscope", "Gyroscope Allan deviation", "rad/s"),
    ):
        figure, axis = plt.subplots(figsize=(12, 7))
        series: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            if row["sensor"] == sensor:
                series.setdefault(str(row["model_name"]), []).append(row)
        for model_name, model_rows in series.items():
            axis.loglog(
                [row["tau_s"] for row in model_rows],
                [row["deviation"] for row in model_rows],
                marker="o",
                markersize=2.5,
                linewidth=1.0,
                label=model_name,
            )
        axis.set_title(label)
        axis.set_xlabel("Cluster period τ (s)")
        axis.set_ylabel(f"Allan deviation ({unit})")
        axis.grid(True, which="both", alpha=.3)
        axis.legend(fontsize="small", bbox_to_anchor=(1.02, 1), loc="upper left")
        figure.subplots_adjust(right=.72)
        figure.savefig(output_dir / f"allan_{sensor}.png", dpi=160)
        plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=120.0, help="noise record duration in seconds")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/allan"))
    parser.add_argument("--tau-min-s", type=float, default=None, help="minimum Allan cluster period")
    parser.add_argument("--tau-max-s", type=float, default=None, help="maximum Allan cluster period; record length limits this")
    parser.add_argument("--points", type=int, default=24, help="number of logarithmic cluster periods")
    parser.add_argument("--profile", type=Path, action="append", default=[], help="additional JSON/YAML profile to include")
    parser.add_argument("--no-plot", action="store_true", help="write CSV data without requiring matplotlib")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    profile_paths = sorted(Path("examples/imu_profiles/hardware-estimates").glob("*.yaml")) + args.profile
    for index, path in enumerate(profile_paths):
        identity, acceleration, gyro, dt, parameters = collect_profile_rates(path, args.duration, index)
        clusters = cluster_sizes_for_scale(dt, args.duration, args.tau_min_s, args.tau_max_s, args.points)
        for sensor, values in (("accelerometer", acceleration), ("gyroscope", gyro)):
            if values.size < 4:
                continue
            tau, deviation = allan_deviation(values, dt, clusters)
            sensor_parameters = parameters[sensor]
            for tau_value, deviation_value in zip(tau, deviation):
                rows.append({
                    **identity,
                    "sensor": sensor,
                    "tau_s": tau_value,
                    "deviation": deviation_value,
                    "white_noise_density": sensor_parameters["white_noise_density"],
                    "bias_correlation_time_s": sensor_parameters["bias_correlation_time"],
                })
    csv_path = args.output_dir / "allan_deviation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model_name", "family", "grade", "sensor", "tau_s", "deviation", "white_noise_density", "bias_correlation_time_s"])
        writer.writeheader()
        writer.writerows(rows)
    if not args.no_plot:
        write_charts(rows, args.output_dir)
    print(f"Wrote {len(rows)} Allan-deviation points to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
