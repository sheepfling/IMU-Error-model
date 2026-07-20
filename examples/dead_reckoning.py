#!/usr/bin/env python3
"""Compare dead-reckoning drift across the example IMU estimate profiles."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from imu_error_model import ImuModel, load_yaml_profile
from imu_error_model.kinematics import rotation_vector_from_matrix


def rotation_matrix(rotation_vector: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(rotation_vector)
    if angle == 0:
        return np.eye(3)
    axis = rotation_vector / angle
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def truth_trajectory(duration: float, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times = np.arange(0.0, duration + dt / 2.0, dt)
    acceleration = np.array([0.2, -0.1, 0.05])
    velocity = times[:, None] * acceleration
    position = 0.5 * times[:, None] ** 2 * acceleration
    angular_rate = np.array([0.0, 0.0, 0.15])
    orientations = np.array([rotation_matrix(angular_rate * time) for time in times])
    return times, position, velocity, orientations


def run_case(profile_path: Path, duration: float = 10.0, seed: int = 0) -> dict[str, object]:
    profile = load_yaml_profile(profile_path)
    dt = profile.sample_period_s
    times, truth_position, truth_velocity, truth_orientation = truth_trajectory(duration, dt)
    model = ImuModel(profile.config, rng=np.random.default_rng(seed))
    estimated_position = np.zeros(3)
    estimated_velocity = np.zeros(3)
    estimated_orientation = truth_orientation[0].copy()
    previous_time = times[0]

    # Profiles may describe raw output counts. Decode those counts before
    # using the increments for physical-state reconstruction.
    accel_scale = profile.config.accelerometer.output_scale
    gyro_scale = profile.config.gyroscope.output_scale
    for time, velocity, orientation in zip(times, truth_velocity, truth_orientation):
        output = model.measure(float(time), velocity, orientation)
        if output.dt == 0:
            continue
        step_dt = time - previous_time
        delta_v_body = output.delta_v / accel_scale
        delta_theta_body = output.delta_theta / gyro_scale
        estimated_position += 0.5 * (estimated_velocity + estimated_velocity + estimated_orientation @ delta_v_body) * step_dt
        estimated_velocity += estimated_orientation @ delta_v_body
        estimated_orientation = estimated_orientation @ rotation_matrix(delta_theta_body)
        previous_time = time

    attitude_error = rotation_vector_from_matrix(truth_orientation[-1].T @ estimated_orientation)
    return {
        "model_name": profile.model_name,
        "family": profile.metadata.get("family", "unknown"),
        "vendor": profile.metadata.get("vendor", "unknown"),
        "grade": profile.metadata.get("grade", "unknown"),
        "sample_period_s": dt,
        "duration_s": duration,
        "final_position_error_m": float(np.linalg.norm(estimated_position - truth_position[-1])),
        "final_velocity_error_mps": float(np.linalg.norm(estimated_velocity - truth_velocity[-1])),
        "final_attitude_error_deg": float(np.rad2deg(np.linalg.norm(attitude_error))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/dead_reckoning_summary.csv"))
    args = parser.parse_args()
    profile_paths = sorted(Path("examples/imu_profiles/hardware-estimates").glob("*.yaml"))
    rows = [run_case(path, duration=args.duration, seed=index) for index, path in enumerate(profile_paths)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['model_name']:14s} position={row['final_position_error_m']:.6g} m  "
              f"velocity={row['final_velocity_error_mps']:.6g} m/s  "
              f"attitude={row['final_attitude_error_deg']:.6g} deg")
    print(f"Wrote {len(rows)} comparisons to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
