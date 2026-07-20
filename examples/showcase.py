#!/usr/bin/env python3
"""Generate polished showcase plots for the IMU error-model package."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from imu_error_model import AxisConfig, ImuConfig, ImuModel, load_profile, load_yaml_profile
from imu_error_model.kinematics import rotation_vector_from_matrix


ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "examples" / "imu_profiles" / "hardware-estimates"
PROFILE_COLORS = {
    "HG9900": "#1864ab",
    "HG5700CA01": "#2b8a3e",
    "HG1700AG58": "#e67700",
    "iPhoneLike": "#c2255c",
}


def load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Showcase plots require: python -m pip install -e '.[analysis]'") from exc
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )
    return plt


def save_figure(plt, figure, output_dir: Path, stem: str) -> None:
    figure.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(figure)


def allan_deviation(samples: np.ndarray, sample_period: float, cluster_sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(samples, dtype=float)
    taus: list[float] = []
    deviations: list[float] = []
    for cluster_size in cluster_sizes:
        if cluster_size < 1 or 2 * cluster_size > values.size:
            continue
        count = values.size // cluster_size
        means = values[: count * cluster_size].reshape(count, cluster_size).mean(axis=1)
        if means.size < 2:
            continue
        taus.append(cluster_size * sample_period)
        deviations.append(float(np.sqrt(0.5 * np.mean(np.diff(means) ** 2))))
    return np.asarray(taus), np.asarray(deviations)


def cluster_sizes(sample_period: float, duration: float, points: int = 28) -> np.ndarray:
    minimum = sample_period
    maximum = duration / 4.0
    return np.unique(np.geomspace(minimum / sample_period, maximum / sample_period, points).astype(int).clip(1))


def collect_profile_rates(path: Path, duration: float, seed: int) -> tuple[str, float, np.ndarray, np.ndarray]:
    profile = load_yaml_profile(path)
    dt = profile.sample_period_s
    model = ImuModel(profile.config, rng=np.random.default_rng(seed))
    count = int(duration / dt)
    accelerometer: list[float] = []
    gyroscope: list[float] = []
    for index in range(count + 1):
        output = model.measure(index * dt, np.zeros(3), np.eye(3))
        if output.dt:
            accelerometer.append(float(output.acceleration[0] / profile.config.accelerometer.output_scale))
            gyroscope.append(float(output.angular_rate[0] / profile.config.gyroscope.output_scale))
    return profile.model_name, dt, np.asarray(accelerometer), np.asarray(gyroscope)


def collect_config_rate(config: ImuConfig, dt: float, duration: float, seed: int = 42) -> np.ndarray:
    model = ImuModel(config, rng=np.random.default_rng(seed))
    samples: list[float] = []
    for index in range(int(duration / dt) + 1):
        output = model.measure(index * dt, np.zeros(3), np.eye(3))
        if output.dt:
            samples.append(float(output.acceleration[0]))
    return np.asarray(samples)


def rotation_matrix(rotation_vector: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(rotation_vector)
    if angle == 0:
        return np.eye(3)
    axis = rotation_vector / angle
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def truth_trajectory(duration: float, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times = np.arange(0.0, duration + dt / 2.0, dt)
    acceleration = np.array([0.2, -0.1, 0.05])
    velocity = times[:, None] * acceleration
    position = 0.5 * times[:, None] ** 2 * acceleration
    angular_rate = np.array([0.0, 0.0, 0.15])
    orientations = np.array([rotation_matrix(angular_rate * time) for time in times])
    return times, position, velocity, orientations


def plot_profile_ladder(output_dir: Path, duration: float) -> str:
    plt = load_matplotlib()
    selected = ["hg9900.yaml", "hg5700ca01.yaml", "hg1700ag58.yaml", "iphone_like.yaml"]
    records = []
    for index, filename in enumerate(selected):
        name, dt, accel, gyro = collect_profile_rates(PROFILE_DIR / filename, duration, 100 + index)
        tau = cluster_sizes(dt, duration)
        records.append((name, allan_deviation(accel, dt, tau), allan_deviation(gyro, dt, tau)))

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), constrained_layout=True)
    for axis, sensor, title, unit in (
        (axes[0], 1, "Accelerometer error", "m/s²"),
        (axes[1], 2, "Gyroscope error", "rad/s"),
    ):
        for name, accel_curve, gyro_curve in records:
            tau, deviation = (accel_curve, gyro_curve)[sensor - 1]
            axis.loglog(tau, deviation, marker="o", markersize=3, linewidth=2, label=name, color=PROFILE_COLORS[name])
        axis.set_title(title)
        axis.set_xlabel("Averaging time τ (s)")
        axis.set_ylabel(f"Allan deviation ({unit})")
        axis.legend(loc="best", frameon=False)
        axis.text(0.02, 0.02, "Representative notional profiles", transform=axis.transAxes, fontsize=8, alpha=0.7)
    figure.suptitle("One measurement-error interface across sensor classes", fontsize=17, fontweight="bold")
    save_figure(plt, figure, output_dir, "hero-allan-ladder")
    return "hero-allan-ladder"


def plot_error_anatomy(output_dir: Path, duration: float) -> str:
    plt = load_matplotlib()
    base = load_profile(ROOT / "profiles" / "test" / "short-correlation.json")
    white = AxisConfig(white_noise_density=0.04, apply_clipping=False)
    markov = AxisConfig(bias_std=0.02, bias_correlation_time=0.5, apply_clipping=False)
    flicker = AxisConfig(
        flicker_bias_std=0.02,
        flicker_min_correlation_time=0.1,
        flicker_max_correlation_time=10.0,
        flicker_components=12,
        apply_clipping=False,
    )
    combined = replace(base.accelerometer, apply_clipping=False)
    variants = {
        "White noise": white,
        "Gauss–Markov bias": markov,
        "Finite-band flicker": flicker,
        "Combined model": combined,
    }
    figure, axis = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)
    colors = ["#1971c2", "#e67700", "#9c36b5", "#212529"]
    for (label, axis_config), color in zip(variants.items(), colors):
        config = replace(base, accelerometer=axis_config, gyroscope=AxisConfig())
        samples = collect_config_rate(config, 0.01, duration, seed=42)
        tau = cluster_sizes(0.01, duration)
        curve_tau, deviation = allan_deviation(samples, 0.01, tau)
        axis.loglog(curve_tau, deviation, marker="o", markersize=3, linewidth=2, label=label, color=color)
    axis.set_title("Composable error terms leave distinct Allan signatures", fontsize=17, fontweight="bold")
    axis.set_xlabel("Averaging time τ (s)")
    axis.set_ylabel("Accelerometer Allan deviation (m/s²)")
    axis.legend(frameon=False)
    axis.text(0.02, 0.02, "Synthetic fixture; parameter meanings are intentionally inspectable", transform=axis.transAxes, fontsize=8, alpha=0.7)
    save_figure(plt, figure, output_dir, "error-anatomy-allan")
    return "error-anatomy-allan"


def reconstruction_series(path: Path, duration: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    profile = load_yaml_profile(path)
    dt = profile.sample_period_s
    times, truth_position, truth_velocity, truth_orientation = truth_trajectory(duration, dt)
    model = ImuModel(profile.config, rng=np.random.default_rng(seed))
    estimated_position = np.zeros(3)
    estimated_velocity = np.zeros(3)
    estimated_orientation = truth_orientation[0].copy()
    previous_time = times[0]
    position_errors: list[float] = []
    attitude_errors: list[float] = []
    for time, velocity, orientation, true_position, true_orientation in zip(
        times, truth_velocity, truth_orientation, truth_position, truth_orientation
    ):
        output = model.measure(float(time), velocity, orientation)
        if output.dt:
            step_dt = time - previous_time
            delta_v_body = output.delta_v / profile.config.accelerometer.output_scale
            delta_theta_body = output.delta_theta / profile.config.gyroscope.output_scale
            delta_v_world = estimated_orientation @ delta_v_body
            estimated_position += estimated_velocity * step_dt + 0.5 * delta_v_world * step_dt
            estimated_velocity += delta_v_world
            estimated_orientation = estimated_orientation @ rotation_matrix(delta_theta_body)
            previous_time = time
        attitude_error = rotation_vector_from_matrix(true_orientation.T @ estimated_orientation)
        position_errors.append(float(np.linalg.norm(estimated_position - true_position)))
        attitude_errors.append(float(np.rad2deg(np.linalg.norm(attitude_error))))
    return times, np.asarray(position_errors), np.asarray(attitude_errors)


def plot_reconstruction(output_dir: Path, duration: float) -> str:
    plt = load_matplotlib()
    selected = ["hg9900.yaml", "hg1700ag58.yaml", "iphone_like.yaml"]
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    for index, filename in enumerate(selected):
        name = load_yaml_profile(PROFILE_DIR / filename).model_name
        times, position_error, attitude_error = reconstruction_series(PROFILE_DIR / filename, duration, 300 + index)
        color = PROFILE_COLORS[name]
        axes[0].semilogy(times, np.maximum(position_error, 1e-9), linewidth=2, label=name, color=color)
        axes[1].semilogy(times, np.maximum(attitude_error, 1e-9), linewidth=2, label=name, color=color)
    axes[0].set_ylabel("Position error (m)")
    axes[1].set_ylabel("Attitude error (deg)")
    axes[1].set_xlabel("Elapsed time (s)")
    axes[0].set_title("The same truth motion, reconstructed through different error models", fontsize=17, fontweight="bold")
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    for axis in axes:
        axis.text(0.99, 0.05, "Dead reckoning; no aiding", transform=axis.transAxes, ha="right", fontsize=8, alpha=0.7)
    save_figure(plt, figure, output_dir, "reconstruction-drift")
    return "reconstruction-drift"


def plot_measurement_time_series(output_dir: Path, duration: float = 8.0) -> str:
    plt = load_matplotlib()
    profile = load_yaml_profile(PROFILE_DIR / "hg1700ag58.yaml")
    dt = profile.sample_period_s
    times = np.arange(0.0, duration + dt / 2.0, dt)
    truth_acceleration = 0.4 + 0.1 * np.sin(2.0 * np.pi * times / 3.0)
    truth_gyro = 0.15 + 0.03 * np.sin(2.0 * np.pi * times / 2.5)
    truth_angle = np.zeros(times.size)
    truth_angle[1:] = np.cumsum((truth_gyro[:-1] + truth_gyro[1:]) * 0.5 * dt)
    world_acceleration = np.column_stack(
        (truth_acceleration * np.cos(truth_angle), truth_acceleration * np.sin(truth_angle), np.zeros(times.size))
    )
    truth_velocity = np.zeros((times.size, 3))
    truth_velocity[1:] = np.cumsum((world_acceleration[:-1] + world_acceleration[1:]) * 0.5 * dt, axis=0)
    model = ImuModel(profile.config, rng=np.random.default_rng(701))
    measured_acceleration: list[float] = []
    measured_gyro: list[float] = []
    for time, velocity, angle in zip(times, truth_velocity, truth_angle):
        output = model.measure(float(time), velocity, rotation_matrix(np.array([0.0, 0.0, angle])))
        if output.dt:
            measured_acceleration.append(float(output.acceleration[0] / profile.config.accelerometer.output_scale))
            measured_gyro.append(float(output.angular_rate[2] / profile.config.gyroscope.output_scale))
    measured_accel = np.asarray(measured_acceleration)
    measured_rate = np.asarray(measured_gyro)
    plot_times = times[1:]
    figure, axes = plt.subplots(2, 2, figsize=(12, 6.8), sharex="col")
    figure.subplots_adjust(bottom=0.16, top=0.82, wspace=0.28, hspace=0.3)
    axes[0, 0].plot(times, truth_acceleration, color="#212529", linewidth=2, linestyle="--", label="Truth")
    axes[0, 0].plot(plot_times, measured_accel, color="#e67700", linewidth=1.2, label="Measured")
    axes[0, 1].plot(plot_times, (measured_accel - truth_acceleration[1:]) * 1e6 / 9.80665, color="#e67700", linewidth=1.3)
    axes[1, 0].plot(times, truth_gyro, color="#212529", linewidth=2, linestyle="--", label="Truth")
    axes[1, 0].plot(plot_times, measured_rate, color="#c2255c", linewidth=1.2, label="Measured")
    axes[1, 1].plot(plot_times, np.rad2deg(measured_rate - truth_gyro[1:]) * 3600.0, color="#c2255c", linewidth=1.3)
    axes[0, 0].set_title("Accelerometer x")
    axes[0, 1].set_title("Accelerometer error")
    axes[1, 0].set_title("Gyroscope z")
    axes[1, 1].set_title("Gyroscope error")
    axes[0, 0].set_ylabel("Rate (m/s²)")
    axes[0, 1].set_ylabel("Error (µg)")
    axes[1, 0].set_ylabel("Rate (rad/s)")
    axes[1, 1].set_ylabel("Error (deg/hr)")
    axes[1, 0].set_xlabel("Elapsed time (s)")
    axes[1, 1].set_xlabel("Elapsed time (s)")
    axes[0, 0].legend(frameon=False, loc="upper right")
    axes[1, 0].legend(frameon=False, loc="upper right")
    figure.suptitle("Ideal motion becomes imperfect sampled measurements", fontsize=17, fontweight="bold")
    figure.text(0.5, 0.01, "HG1700AG58 notional example; deterministic truth with seeded stochastic errors", ha="center", fontsize=8, alpha=0.7)
    save_figure(plt, figure, output_dir, "measurement-time-series")
    return "measurement-time-series"


def plot_temperature(output_dir: Path, points: int) -> str:
    plt = load_matplotlib()
    source = load_yaml_profile(PROFILE_DIR / "sbg_pulse_40.yaml")
    accel = replace(
        source.config.accelerometer,
        white_noise_density=0.0,
        turn_on_bias_std=0.0,
        bias_std=0.0,
        flicker_bias_std=0.0,
        scale_factor=0.0,
        nonlinear_factor=0.0,
        measurement_range=None,
        output_scale=1.0,
    )
    gyro = replace(
        source.config.gyroscope,
        white_noise_density=0.0,
        turn_on_bias_std=0.0,
        bias_std=0.0,
        flicker_bias_std=0.0,
        scale_factor=0.0,
        nonlinear_factor=0.0,
        measurement_range=None,
        output_scale=1.0,
    )
    model = ImuModel(ImuConfig(accelerometer=accel, gyroscope=gyro), rng=np.random.default_rng(11))
    model.measure(0.0, np.zeros(3), np.eye(3), temperature=25.0)
    temperatures = np.linspace(-40.0, 71.0, points)
    accel_error: list[float] = []
    gyro_error: list[float] = []
    dt = 0.1
    for index, temperature in enumerate(temperatures, start=1):
        time = index * dt
        angular_rate = 0.1
        # World velocity generated by a constant 1 m/s² body-x acceleration
        # while the body rotates about z at angular_rate.
        velocity = np.array(
            [np.sin(angular_rate * time) / angular_rate, (1.0 - np.cos(angular_rate * time)) / angular_rate, 0.0]
        )
        orientation = rotation_matrix(np.array([0.0, 0.0, angular_rate * time]))
        output = model.measure(time, velocity, orientation, temperature=temperature)
        accel_error.append(float((output.acceleration[0] - 1.0) * 1000.0 / 9.80665))
        gyro_error.append(float(np.rad2deg(output.angular_rate[2] - 0.1) * 3600.0))
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    figure.subplots_adjust(bottom=0.18, top=0.82, wspace=0.28)
    axes[0].plot(temperatures, accel_error, linewidth=2.5, color="#1864ab")
    axes[1].plot(temperatures, gyro_error, linewidth=2.5, color="#c2255c")
    axes[0].axhline(0.0, color="#868e96", linewidth=1)
    axes[1].axhline(0.0, color="#868e96", linewidth=1)
    axes[0].set_title("Accelerometer")
    axes[1].set_title("Gyroscope")
    axes[0].set_ylabel("Measurement error (mg)")
    axes[1].set_ylabel("Measurement error (deg/hr)")
    for axis in axes:
        axis.set_xlabel("Temperature (°C)")
        axis.grid(True, alpha=0.22)
    figure.suptitle("Thermal error terms become visible as temperature moves from the reference", fontsize=16, fontweight="bold")
    figure.text(0.5, 0.005, "Thermal-only rendering of the notional SBG PULSE 40 example; reference temperature = 25 °C", ha="center", fontsize=8, alpha=0.7)
    save_figure(plt, figure, output_dir, "thermal-trends")
    return "thermal-trends"


def write_manifest(output_dir: Path, stems: list[str]) -> None:
    descriptions = {
        "hero-allan-ladder": "Representative sensor-class Allan-deviation comparison",
        "error-anatomy-allan": "Allan signatures of composable error terms",
        "reconstruction-drift": "Dead-reckoning error growth from identical truth motion",
        "measurement-time-series": "Truth rates compared with imperfect sampled rates",
        "thermal-trends": "Thermal-only accelerometer and gyroscope error trends",
    }
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["asset", "description"])
        writer.writeheader()
        writer.writerows({"asset": stem, "description": descriptions[stem]} for stem in stems)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/showcase"))
    parser.add_argument("--allan-duration", type=float, default=240.0)
    parser.add_argument("--reconstruction-duration", type=float, default=60.0)
    parser.add_argument("--temperature-points", type=int, default=120)
    args = parser.parse_args()
    if args.allan_duration <= 0 or args.reconstruction_duration <= 0 or args.temperature_points < 2:
        raise ValueError("durations must be positive and temperature-points must be at least two")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".matplotlib"))
    stems = [
        plot_profile_ladder(args.output_dir, args.allan_duration),
        plot_error_anatomy(args.output_dir, min(args.allan_duration, 120.0)),
        plot_reconstruction(args.output_dir, args.reconstruction_duration),
        plot_measurement_time_series(args.output_dir),
        plot_temperature(args.output_dir, args.temperature_points),
    ]
    write_manifest(args.output_dir, stems)
    for stem in stems:
        print(f"Wrote {stem}.png and {stem}.svg")
    print(f"Wrote showcase manifest to {args.output_dir / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
