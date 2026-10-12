from __future__ import annotations

from math import pi

import numpy as np
import pytest

from imu_error_model.continuous_acquisition import ContinuousTruthSample
from playground.calibration.preintegration import (
    ConingTruth,
    MinorImuIncrement,
    ScullingTruth,
    convergence_study,
    first_rate_meeting_error,
    naive_major_interval,
    preintegrate_continuous_truth,
    preintegrate_piecewise_constant,
    recursive_coning_sculling,
    sample_minor_increments,
    so3_exp,
    so3_log,
)


def test_so3_exp_log_round_trip() -> None:
    rotation_vector = np.asarray((0.2, -0.3, 0.4), dtype=float)
    recovered = so3_log(so3_exp(rotation_vector))
    assert np.allclose(recovered, rotation_vector, atol=1e-12, rtol=1e-12)
####


def test_piecewise_constant_full_turn_rotating_force_integrates_to_zero() -> None:
    sample_count = 32
    dt_s = 1.0 / sample_count
    samples = tuple(
        MinorImuIncrement(
            delta_theta_body=np.asarray((0.0, 0.0, 2.0 * pi * dt_s)),
            delta_v_body=np.asarray((1.0 * dt_s, 0.0, 0.0)),
            dt_s=dt_s,
        )
        for _ in range(sample_count)
    )
    result = preintegrate_piecewise_constant(samples)
    naive = naive_major_interval(samples)
    assert np.linalg.norm(result.delta_theta_start) < 1e-10
    assert np.linalg.norm(result.delta_v_start) < 1e-12
    assert np.allclose(naive.delta_v_start, (1.0, 0.0, 0.0), atol=1e-12)
####


def test_coning_cycle_has_nonzero_geometric_rotation_despite_zero_mean_rate() -> None:
    truth = ConingTruth(angular_rate_amplitude_radps=1.0, frequency_hz=5.0)
    reference = preintegrate_continuous_truth(
        truth,
        start_time_s=0.0,
        end_time_s=truth.period_s,
        reference_substeps=32768,
    )
    samples = sample_minor_increments(
        truth,
        start_time_s=0.0,
        end_time_s=truth.period_s,
        minor_count=64,
        quadrature_substeps=32,
    )
    naive = naive_major_interval(samples)
    piecewise = preintegrate_piecewise_constant(samples)
    assert np.linalg.norm(naive.delta_theta_start) < 1e-12
    assert np.linalg.norm(reference.delta_theta_start) > 3e-3
    assert np.linalg.norm(piecewise.delta_theta_start - reference.delta_theta_start) < 6e-6
####


def test_classical_coning_correction_reduces_low_rate_error() -> None:
    truth = ConingTruth(angular_rate_amplitude_radps=1.0, frequency_hz=5.0)
    reference = preintegrate_continuous_truth(
        truth,
        start_time_s=0.0,
        end_time_s=truth.period_s,
        reference_substeps=32768,
    )
    samples = sample_minor_increments(
        truth,
        start_time_s=0.0,
        end_time_s=truth.period_s,
        minor_count=8,
        quadrature_substeps=64,
    )
    naive = naive_major_interval(samples)
    corrected = recursive_coning_sculling(samples)
    assert np.linalg.norm(corrected.delta_theta_start - reference.delta_theta_start) < 0.1 * np.linalg.norm(
        naive.delta_theta_start - reference.delta_theta_start
    )
####


def test_sculling_cycle_has_nonzero_velocity_despite_zero_raw_sums() -> None:
    truth = ScullingTruth(angle_amplitude_rad=0.1, force_amplitude_mps2=2.0, frequency_hz=5.0)
    reference = preintegrate_continuous_truth(
        truth,
        start_time_s=0.0,
        end_time_s=truth.period_s,
        reference_substeps=32768,
    )
    samples = sample_minor_increments(
        truth,
        start_time_s=0.0,
        end_time_s=truth.period_s,
        minor_count=16,
        quadrature_substeps=64,
    )
    naive = naive_major_interval(samples)
    corrected = recursive_coning_sculling(samples)
    assert np.linalg.norm(naive.delta_v_start) < 1e-12
    assert reference.delta_v_start[2] == pytest.approx(0.019975, abs=5e-6)
    assert abs(corrected.delta_v_start[2] - reference.delta_v_start[2]) < 3e-5
####


def test_coning_piecewise_reference_converges_with_minor_rate() -> None:
    truth = ConingTruth(angular_rate_amplitude_radps=1.0, frequency_hz=5.0)
    points = convergence_study(
        truth,
        start_time_s=0.0,
        end_time_s=truth.period_s,
        minor_counts=(4, 8, 16, 32, 64),
        reference_substeps=32768,
        quadrature_substeps=64,
    )
    errors = [point.exact_piecewise_attitude_error_rad for point in points]
    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
    assert errors[-1] < 6e-6
    assert points[-1].naive_attitude_error_rad > 500.0 * errors[-1]
####


def test_sculling_piecewise_reference_converges_with_minor_rate() -> None:
    truth = ScullingTruth(angle_amplitude_rad=0.1, force_amplitude_mps2=2.0, frequency_hz=5.0)
    points = convergence_study(
        truth,
        start_time_s=0.0,
        end_time_s=truth.period_s,
        minor_counts=(4, 8, 16, 32, 64),
        reference_substeps=32768,
        quadrature_substeps=64,
    )
    errors = [point.exact_piecewise_velocity_error_mps for point in points]
    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
    assert errors[-1] < 4e-5
    assert points[-1].naive_velocity_error_mps > 500.0 * errors[-1]
####


def test_first_rate_meeting_error_returns_first_tested_rate() -> None:
    truth = ConingTruth(angular_rate_amplitude_radps=1.0, frequency_hz=5.0)
    points = convergence_study(
        truth,
        start_time_s=0.0,
        end_time_s=truth.period_s,
        minor_counts=(4, 8, 16, 32, 64),
        reference_substeps=16384,
        quadrature_substeps=32,
    )
    sample_rate = first_rate_meeting_error(points, attitude_error_rad=np.deg2rad(0.001), method="exact_piecewise")
    assert sample_rate == pytest.approx(320.0)
####


def test_continuous_truth_input_validation() -> None:
    def truth(_time_s: float) -> ContinuousTruthSample:
        return ContinuousTruthSample(
            specific_force_body=(0.0, 0.0, 0.0),
            angular_rate_body=(0.0, 0.0, 0.0),
        )
    ####

    with pytest.raises(ValueError, match="minor_count"):
        sample_minor_increments(truth, start_time_s=0.0, end_time_s=1.0, minor_count=0)
    ####
    with pytest.raises(ValueError, match="error target"):
        first_rate_meeting_error(())
    ####
####


def test_calibration_is_applied_before_sculling_preintegration() -> None:
    from imu_error_model import AxisConfig, ImuConfig, ImuModel
    from playground.calibration.gyrocompass import apply_calibration
    from playground.calibration.known_reference import (
        calibration_parameter_error,
        ferraris_style_reference_program,
        fit_known_reference_calibration,
        simulate_reference_program,
    )
    from playground.calibration.preintegration import apply_calibration_to_minor_increments

    config = ImuConfig(
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
    calibration_model = ImuModel(config, rng=np.random.default_rng(7))
    calibration_dataset = simulate_reference_program(
        calibration_model,
        ferraris_style_reference_program(duration_s=0.5, angular_rate_radps=1.2),
    )
    calibration = fit_known_reference_calibration(calibration_dataset)
    assert np.max(np.abs(calibration_parameter_error(calibration, config))) < 1e-12

    truth = ScullingTruth(angle_amplitude_rad=0.1, force_amplitude_mps2=2.0, frequency_hz=5.0)
    ideal_minors = sample_minor_increments(
        truth,
        start_time_s=0.0,
        end_time_s=truth.period_s,
        minor_count=32,
        quadrature_substeps=64,
    )
    sensor = ImuModel(config, rng=np.random.default_rng(11))
    measured_minors = []
    time_s = 0.0
    for ideal in ideal_minors:
        output = sensor.measure_ideal_rates(
            start_time=time_s,
            end_time=time_s + ideal.dt_s,
            specific_force_body=ideal.delta_v_body / ideal.dt_s,
            angular_rate_body=ideal.delta_theta_body / ideal.dt_s,
        )
        measured_minors.append(
            MinorImuIncrement(
                delta_theta_body=output.delta_theta,
                delta_v_body=output.delta_v,
                dt_s=ideal.dt_s,
            )
        )
        time_s += ideal.dt_s
    ####

    corrected = apply_calibration_to_minor_increments(tuple(measured_minors), calibration)
    corrected_major = preintegrate_piecewise_constant(corrected)
    ideal_major = preintegrate_piecewise_constant(ideal_minors)
    assert np.allclose(corrected_major.delta_theta_start, ideal_major.delta_theta_start, atol=1e-12, rtol=1e-12)
    assert np.allclose(corrected_major.delta_v_start, ideal_major.delta_v_start, atol=1e-12, rtol=1e-12)

    measured_accel_mean = np.sum([sample.delta_v_body for sample in measured_minors], axis=0) / truth.period_s
    measured_gyro_mean = np.sum([sample.delta_theta_body for sample in measured_minors], axis=0) / truth.period_s
    corrected_force_mean, corrected_rate_mean = apply_calibration(measured_accel_mean, measured_gyro_mean, calibration)
    wrong_major = preintegrate_piecewise_constant(
        (
            MinorImuIncrement(
                delta_theta_body=corrected_rate_mean * truth.period_s,
                delta_v_body=corrected_force_mean * truth.period_s,
                dt_s=truth.period_s,
            ),
        )
    )
    assert np.linalg.norm(wrong_major.delta_v_start) < 1e-12
    assert ideal_major.delta_v_start[2] > 0.019
####
