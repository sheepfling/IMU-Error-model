from dataclasses import replace

from numpy import asarray, diag, flatnonzero, random, zeros
from numpy.linalg import eigvalsh

from imu_error_model import ImuModel
from playground.calibration.bootstrap import moving_block_bootstrap_calibration
from playground.calibration.known_reference import (
    CalibrationDataset,
    ferraris_style_reference_program,
    fit_known_reference_calibration,
    simulate_reference_program,
)


def _repeated_program(sample_count: int):
    base = ferraris_style_reference_program(duration_s=sample_count * 0.1)
    return tuple(replace(segment, sample_count=sample_count) for segment in base)
####


def _colored_dataset(sample_count: int = 120) -> CalibrationDataset:
    ideal = simulate_reference_program(ImuModel(), _repeated_program(sample_count))
    rng = random.default_rng(22)
    gyro_noise = zeros(ideal.measured_angular_rate.shape)
    labels = asarray(ideal.labels, dtype=object)
    for label in tuple(dict.fromkeys(ideal.labels)):
        indices = flatnonzero(labels == label)
        innovations = rng.normal(0.0, 1.2e-6, (indices.size, 3))
        state = zeros(3)
        for local_index, row_index in enumerate(indices):
            state = 0.97 * state + innovations[local_index]
            gyro_noise[row_index] = state
        ####
    ####
    return CalibrationDataset(
        labels=ideal.labels,
        start_times=ideal.start_times,
        end_times=ideal.end_times,
        temperatures_celsius=ideal.temperatures_celsius,
        specific_force_body=ideal.specific_force_body,
        angular_rate_body=ideal.angular_rate_body,
        measured_acceleration=ideal.measured_acceleration,
        measured_angular_rate=ideal.measured_angular_rate + gyro_noise,
    )
####


def test_moving_block_bootstrap_returns_psd_calibration_covariance() -> None:
    result = moving_block_bootstrap_calibration(
        _colored_dataset(sample_count=80),
        replicate_count=60,
        block_length=16,
        seed=91,
    )

    assert result.estimate.accelerometer_coefficient_covariance.shape == (12, 12)
    assert result.estimate.gyroscope_coefficient_covariance.shape == (21, 21)
    assert eigvalsh(result.estimate.accelerometer_coefficient_covariance).min() > -1e-18
    assert eigvalsh(result.estimate.gyroscope_coefficient_covariance).min() > -1e-18
    assert "moving_block_bootstrap" in result.estimate.covariance_assumption
####


def test_block_bootstrap_exposes_colored_gyro_bias_understatement() -> None:
    dataset = _colored_dataset(sample_count=160)
    classical = fit_known_reference_calibration(dataset)
    bootstrap = moving_block_bootstrap_calibration(
        dataset,
        replicate_count=100,
        block_length=40,
        seed=92,
    )
    bias_indices = asarray((6, 13, 20), dtype=int)
    classical_bias_variance = float(diag(classical.gyroscope_coefficient_covariance)[bias_indices].mean())
    bootstrap_bias_variance = float(
        diag(bootstrap.estimate.gyroscope_coefficient_covariance)[bias_indices].mean()
    )

    assert bootstrap_bias_variance > 3.0 * classical_bias_variance
    assert bootstrap.gyroscope_median_variance_inflation > 1.5
####
