"""Run the second calibration-playground validation tranche."""

from __future__ import annotations

import json
from math import pi
from pathlib import Path

from numpy import abs as np_abs, median, random

from imu_error_model import AxisConfig, ImuConfig, ImuModel

from .gyrocompass import simulate_static_gyrocompass, wrap_angle_rad
from .known_reference import ferraris_style_reference_program, fit_known_reference_calibration, simulate_reference_program
from .monte_carlo import run_monte_carlo_recovery
from .observability import analyze_known_reference_observability
from .thermal import expected_thermal_axis, fit_accelerometer_thermal_model, thermal_reference_program


def main(output_path: str | Path = "artifacts/calibration/statistical_campaign.json") -> Path:
    """Run deterministic, statistical, observability, thermal, and gyrocompass checks."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    config = ImuConfig(
        accelerometer=AxisConfig(
            white_noise_density=0.002,
            fixed_bias=(0.08, -0.05, 0.03),
            sensitivity_matrix=((1.012, 0.004, -0.002), (-0.003, 0.993, 0.005), (0.001, -0.004, 1.007)),
        ),
        gyroscope=AxisConfig(
            white_noise_density=2e-5,
            fixed_bias=(0.001, -0.002, 0.0015),
            sensitivity_matrix=((1.008, 0.003, -0.002), (-0.004, 0.996, 0.002), (0.001, -0.003, 1.005)),
        ),
        gyroscope_specific_force_sensitivity=(
            (1e-5, -2e-5, 0.5e-5),
            (-1e-5, 1.5e-5, 2e-5),
            (0.25e-5, -0.5e-5, 1e-5),
        ),
    )
    monte_carlo = run_monte_carlo_recovery(
        config,
        trial_count=200,
        samples_per_segment=50,
        sample_period_s=0.02,
        seed=583,
    )

    observability_dataset = simulate_reference_program(ImuModel(), ferraris_style_reference_program())
    accel_observability, gyro_observability = analyze_known_reference_observability(observability_dataset)

    thermal_accelerometer = AxisConfig(
        fixed_bias=(0.08, -0.05, 0.03),
        sensitivity_matrix=((1.012, 0.004, -0.002), (-0.003, 0.993, 0.005), (0.001, -0.004, 1.007)),
        thermal_bias_coefficient=(4e-4, -2e-4, 3e-4),
        thermal_scale_factor_coefficient=(20e-6, -12e-6, 16e-6),
    )
    thermal_config = ImuConfig(accelerometer=thermal_accelerometer)
    thermal_dataset = simulate_reference_program(ImuModel(thermal_config), thermal_reference_program())
    thermal_estimate = fit_accelerometer_thermal_model(thermal_dataset, thermal_accelerometer)
    thermal_truth = expected_thermal_axis(thermal_accelerometer)
    thermal_max_error = max(
        float(np_abs(thermal_estimate.base_matrix - thermal_truth.base_matrix).max()),
        float(np_abs(thermal_estimate.thermal_matrix - thermal_truth.thermal_matrix).max()),
        float(np_abs(thermal_estimate.base_bias - thermal_truth.base_bias).max()),
        float(np_abs(thermal_estimate.thermal_bias - thermal_truth.thermal_bias).max()),
    )

    compass_config = ImuConfig(
        accelerometer=AxisConfig(
            white_noise_density=0.0005,
            fixed_bias=(0.08, -0.05, 0.03),
            sensitivity_matrix=((1.012, 0.004, -0.002), (-0.003, 0.993, 0.005), (0.001, -0.004, 1.007)),
        ),
        gyroscope=AxisConfig(
            white_noise_density=2e-6,
            fixed_bias=(8e-6, -5e-6, 4e-6),
            sensitivity_matrix=((1.008, 0.003, -0.002), (-0.004, 0.996, 0.002), (0.001, -0.003, 1.005)),
        ),
        gyroscope_specific_force_sensitivity=(
            (2e-7, -1e-7, 0.5e-7),
            (-0.5e-7, 1e-7, 1.5e-7),
            (0.25e-7, -0.5e-7, 0.75e-7),
        ),
    )
    calibration_program = tuple(
        type(segment)(
            label=segment.label,
            duration_s=4.0,
            specific_force_body=segment.specific_force_body,
            angular_rate_body=segment.angular_rate_body,
            temperature_celsius=segment.temperature_celsius,
            sample_count=200,
        )
        for segment in ferraris_style_reference_program(duration_s=4.0)
    )
    calibration = fit_known_reference_calibration(
        simulate_reference_program(
            ImuModel(compass_config, rng=random.default_rng(44)),
            calibration_program,
        )
    )
    truth_heading_rad = -28.0 * pi / 180.0
    compass = simulate_static_gyrocompass(
        ImuModel(compass_config, rng=random.default_rng(45)),
        calibration,
        latitude_rad=34.7304 * pi / 180.0,
        heading_rad=truth_heading_rad,
        sample_period_s=0.02,
        sample_count=12000,
    )
    heading_error_deg = abs(wrap_angle_rad(compass.heading_rad - truth_heading_rad)) * 180.0 / pi

    payload = {
        "monte_carlo": {
            "trial_count": monte_carlo.trial_count,
            "parameter_count": monte_carlo.parameter_count,
            "mean_coverage_95": monte_carlo.mean_coverage_95,
            "max_abs_mean_error": float(np_abs(monte_carlo.mean_error).max()),
            "median_predicted_to_empirical_std_ratio": float(
                median(monte_carlo.predicted_standard_error / monte_carlo.empirical_standard_deviation)
            ),
        },
        "observability": {
            "accelerometer_rank": accel_observability.rank,
            "accelerometer_parameter_count": accel_observability.parameter_count,
            "accelerometer_condition_number": accel_observability.condition_number,
            "gyroscope_rank": gyro_observability.rank,
            "gyroscope_parameter_count": gyro_observability.parameter_count,
            "gyroscope_condition_number": gyro_observability.condition_number,
        },
        "thermal": {"max_parameter_error": thermal_max_error},
        "gyrocompass": {
            "latitude_deg": 34.7304,
            "truth_heading_deg": -28.0,
            "estimated_heading_deg": compass.heading_deg,
            "heading_error_deg": heading_error_deg,
            "sample_count": compass.sample_count,
            "earth_rate_residual_radps": compass.earth_rate_residual_radps,
        },
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return destination
####


if __name__ == "__main__":
    main()
####
