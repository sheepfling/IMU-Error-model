"""Generate compact tranche-9 high-rate preintegration evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imu_error_model import AxisConfig, ImuConfig, ImuModel

from .known_reference import calibration_parameter_error, ferraris_style_reference_program, fit_known_reference_calibration, simulate_reference_program
from .preintegration import ConingTruth, ConvergencePoint, MinorImuIncrement, ScullingTruth, apply_calibration_to_minor_increments, convergence_study, first_rate_meeting_error, preintegrate_continuous_truth, preintegrate_piecewise_constant, sample_minor_increments


def _serialize(points: tuple[ConvergencePoint, ...]) -> list[dict[str, float | int]]:
    return [
        {
            "minor_count": point.minor_count,
            "sample_rate_hz": point.sample_rate_hz,
            "exact_piecewise_attitude_error_rad": point.exact_piecewise_attitude_error_rad,
            "corrected_attitude_error_rad": point.corrected_attitude_error_rad,
            "naive_attitude_error_rad": point.naive_attitude_error_rad,
            "exact_piecewise_velocity_error_mps": point.exact_piecewise_velocity_error_mps,
            "corrected_velocity_error_mps": point.corrected_velocity_error_mps,
            "naive_velocity_error_mps": point.naive_velocity_error_mps,
        }
        for point in points
    ]
####


def main() -> None:
    output = Path("artifacts/tranche9")
    output.mkdir(parents=True, exist_ok=True)
    counts = (4, 8, 16, 32, 64, 128)

    coning = ConingTruth(angular_rate_amplitude_radps=1.0, frequency_hz=5.0)
    coning_reference = preintegrate_continuous_truth(
        coning, start_time_s=0.0, end_time_s=coning.period_s, reference_substeps=65536
    )
    coning_points = convergence_study(
        coning,
        start_time_s=0.0,
        end_time_s=coning.period_s,
        minor_counts=counts,
        reference_substeps=65536,
        quadrature_substeps=64,
    )

    sculling = ScullingTruth(angle_amplitude_rad=0.1, force_amplitude_mps2=2.0, frequency_hz=5.0)
    sculling_reference = preintegrate_continuous_truth(
        sculling, start_time_s=0.0, end_time_s=sculling.period_s, reference_substeps=65536
    )
    sculling_points = convergence_study(
        sculling,
        start_time_s=0.0,
        end_time_s=sculling.period_s,
        minor_counts=counts,
        reference_substeps=65536,
        quadrature_substeps=64,
    )


    calibration_config = ImuConfig(
        accelerometer=AxisConfig(
            fixed_bias=(0.03, -0.02, 0.01),
            sensitivity_matrix=((1.01, 0.004, -0.002), (-0.003, 0.995, 0.005), (0.002, -0.004, 1.006)),
        ),
        gyroscope=AxisConfig(
            fixed_bias=(0.002, -0.001, 0.0005),
            sensitivity_matrix=((0.998, 0.003, -0.002), (-0.004, 1.005, 0.001), (0.002, -0.003, 1.002)),
        ),
        gyroscope_specific_force_sensitivity=(
            (1.0e-4, -2.0e-5, 1.0e-5),
            (3.0e-5, -8.0e-5, 2.0e-5),
            (-1.0e-5, 2.0e-5, 5.0e-5),
        ),
    )
    calibration_dataset = simulate_reference_program(
        ImuModel(calibration_config, rng=np.random.default_rng(7)),
        ferraris_style_reference_program(duration_s=0.5, angular_rate_radps=1.2),
    )
    calibration = fit_known_reference_calibration(calibration_dataset)
    ideal_minors = sample_minor_increments(
        sculling, start_time_s=0.0, end_time_s=sculling.period_s, minor_count=32, quadrature_substeps=64
    )
    sensor = ImuModel(calibration_config, rng=np.random.default_rng(11))
    measured_minors: list[MinorImuIncrement] = []
    time_s = 0.0
    for ideal in ideal_minors:
        measured = sensor.measure_ideal_rates(
            start_time=time_s,
            end_time=time_s + ideal.dt_s,
            specific_force_body=ideal.delta_v_body / ideal.dt_s,
            angular_rate_body=ideal.delta_theta_body / ideal.dt_s,
        )
        measured_minors.append(MinorImuIncrement(measured.delta_theta, measured.delta_v, ideal.dt_s))
        time_s += ideal.dt_s
    ####
    corrected_major = preintegrate_piecewise_constant(
        apply_calibration_to_minor_increments(tuple(measured_minors), calibration)
    )
    ideal_major = preintegrate_piecewise_constant(ideal_minors)

    payload = {
        "coning": {
            "frequency_hz": coning.frequency_hz,
            "angular_rate_amplitude_radps": coning.angular_rate_amplitude_radps,
            "truth_delta_theta_rad": coning_reference.delta_theta_start.tolist(),
            "truth_rotation_magnitude_deg": float(np.rad2deg(np.linalg.norm(coning_reference.delta_theta_start))),
            "points": _serialize(coning_points),
            "first_exact_piecewise_rate_below_0p001_deg_hz": first_rate_meeting_error(
                coning_points, attitude_error_rad=float(np.deg2rad(0.001)), method="exact_piecewise"
            ),
            "first_corrected_rate_below_0p01_deg_hz": first_rate_meeting_error(
                coning_points, attitude_error_rad=float(np.deg2rad(0.01)), method="corrected"
            ),
            "first_corrected_rate_below_0p001_deg_hz": first_rate_meeting_error(
                coning_points, attitude_error_rad=float(np.deg2rad(0.001)), method="corrected"
            ),
        },
        "calibrated_minor_route": {
            "calibration_max_parameter_error": float(np.max(np.abs(calibration_parameter_error(calibration, calibration_config)))),
            "preintegrated_delta_theta_error_rad": float(np.linalg.norm(corrected_major.delta_theta_start - ideal_major.delta_theta_start)),
            "preintegrated_delta_v_error_mps": float(np.linalg.norm(corrected_major.delta_v_start - ideal_major.delta_v_start)),
            "minor_sample_count": len(ideal_minors),
        },
        "sculling": {
            "frequency_hz": sculling.frequency_hz,
            "angle_amplitude_rad": sculling.angle_amplitude_rad,
            "force_amplitude_mps2": sculling.force_amplitude_mps2,
            "truth_delta_v_mps": sculling_reference.delta_v_start.tolist(),
            "points": _serialize(sculling_points),
            "first_exact_piecewise_rate_below_0p1_mmps_hz": first_rate_meeting_error(
                sculling_points, velocity_error_mps=1e-4, method="exact_piecewise"
            ),
            "first_corrected_rate_below_0p1_mmps_hz": first_rate_meeting_error(
                sculling_points, velocity_error_mps=1e-4, method="corrected"
            ),
        },
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
####


if __name__ == "__main__":
    main()
####
