from numpy import allclose, diag, random
from numpy.linalg import eigvalsh

from imu_error_model import AxisConfig, ImuConfig, ImuModel
from playground.calibration.known_reference import (
    calibration_coefficient_vectors,
    calibration_from_coefficient_vectors,
    ferraris_style_reference_program,
    fit_known_reference_calibration,
    simulate_reference_program,
)


def _noisy_calibration(samples_per_segment: int = 40):
    config = ImuConfig(
        accelerometer=AxisConfig(white_noise_density=0.002),
        gyroscope=AxisConfig(white_noise_density=2e-5),
    )
    program = tuple(
        type(segment)(
            label=segment.label,
            duration_s=samples_per_segment * 0.02,
            specific_force_body=segment.specific_force_body,
            angular_rate_body=segment.angular_rate_body,
            temperature_celsius=segment.temperature_celsius,
            sample_count=samples_per_segment,
        )
        for segment in ferraris_style_reference_program(duration_s=samples_per_segment * 0.02)
    )
    return fit_known_reference_calibration(
        simulate_reference_program(
            ImuModel(config=config, rng=random.default_rng(7)),
            program,
        )
    )
####


def test_known_reference_fit_retains_full_coefficient_covariance() -> None:
    estimate = _noisy_calibration()

    assert estimate.accelerometer_coefficient_covariance.shape == (12, 12)
    assert estimate.gyroscope_coefficient_covariance.shape == (21, 21)
    assert allclose(
        estimate.accelerometer_coefficient_covariance,
        estimate.accelerometer_coefficient_covariance.T,
    )
    assert allclose(
        estimate.gyroscope_coefficient_covariance,
        estimate.gyroscope_coefficient_covariance.T,
    )
    assert eigvalsh(estimate.accelerometer_coefficient_covariance).min() >= -1e-12
    assert eigvalsh(estimate.gyroscope_coefficient_covariance).min() >= -1e-12
    assert diag(estimate.accelerometer_coefficient_covariance).max() > 0.0
    assert diag(estimate.gyroscope_coefficient_covariance).max() > 0.0
####


def test_calibration_coefficient_vector_round_trip_is_lossless() -> None:
    estimate = _noisy_calibration()
    accel_vector, gyro_vector = calibration_coefficient_vectors(estimate)

    rebuilt = calibration_from_coefficient_vectors(estimate, accel_vector, gyro_vector)

    assert allclose(rebuilt.accelerometer_sensitivity_matrix, estimate.accelerometer_sensitivity_matrix)
    assert allclose(rebuilt.accelerometer_bias, estimate.accelerometer_bias)
    assert allclose(rebuilt.gyroscope_sensitivity_matrix, estimate.gyroscope_sensitivity_matrix)
    assert allclose(
        rebuilt.gyroscope_specific_force_sensitivity,
        estimate.gyroscope_specific_force_sensitivity,
    )
    assert allclose(rebuilt.gyroscope_bias, estimate.gyroscope_bias)
####
