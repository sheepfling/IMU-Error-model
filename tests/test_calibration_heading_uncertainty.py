from math import pi

from numpy import asarray, random, zeros

from imu_error_model import AxisConfig, ImuConfig, ImuModel
from playground.calibration.data_confidence import run_heading_uncertainty_coverage
from playground.calibration.gyrocompass import simulate_static_measurements
from playground.calibration.heading_uncertainty import long_run_mean_covariance
from playground.calibration.known_reference import (
    ferraris_style_reference_program,
    fit_known_reference_calibration,
    simulate_reference_program,
)
from playground.calibration.quality import quality_aware_static_gyrocompass


def _ideal_calibration():
    return fit_known_reference_calibration(
        simulate_reference_program(ImuModel(), ferraris_style_reference_program())
    )
####


def test_zero_lag_mean_covariance_matches_biased_sample_formula() -> None:
    samples = asarray(
        (
            (1.0, 2.0),
            (3.0, 4.0),
            (5.0, 8.0),
            (7.0, 10.0),
        ),
        dtype=float,
    )
    centered = samples - samples.mean(axis=0)
    expected = centered.T @ centered / samples.shape[0] ** 2

    actual = long_run_mean_covariance(samples, max_lag=0)

    assert abs(actual - expected).max() < 1e-12
####



def test_serial_correlation_increases_long_run_mean_covariance() -> None:
    rng = random.default_rng(7)
    innovations = rng.normal(0.0, 1.0, 4000)
    samples = zeros((innovations.size, 1))
    for index in range(1, innovations.size):
        samples[index, 0] = 0.9 * samples[index - 1, 0] + innovations[index]
    ####

    iid = long_run_mean_covariance(samples, max_lag=0)
    correlated = long_run_mean_covariance(samples, max_lag=100)

    assert correlated[0, 0] > 8.0 * iid[0, 0]
####

def test_noiseless_static_record_has_negligible_propagated_uncertainty() -> None:
    latitude_rad = 35.0 * pi / 180.0
    batch = simulate_static_measurements(
        ImuModel(),
        latitude_rad=latitude_rad,
        heading_rad=20.0 * pi / 180.0,
        sample_period_s=0.05,
        sample_count=200,
    )

    result = quality_aware_static_gyrocompass(
        batch,
        _ideal_calibration(),
        latitude_rad=latitude_rad,
        covariance_max_lag=0,
    )

    assert result.accepted
    assert result.uncertainty is not None
    assert result.uncertainty.standard_deviation_deg < 2e-10
####


def test_joint_heading_covariance_has_reasonable_monte_carlo_coverage() -> None:
    config = ImuConfig(
        accelerometer=AxisConfig(
            white_noise_density=0.001,
            fixed_bias=(0.02, -0.01, 0.01),
        ),
        gyroscope=AxisConfig(
            white_noise_density=2e-6,
            fixed_bias=(5e-6, -3e-6, 2e-6),
        ),
    )

    summary = run_heading_uncertainty_coverage(
        config=config,
        calibration_samples_per_segment=100,
        compass_duration_s=30.0,
        latitude_rad=35.0 * pi / 180.0,
        heading_rad=20.0 * pi / 180.0,
        sample_period_s=0.1,
        trial_count=40,
        covariance_max_lag=0,
        seed=123,
    )

    ratio = summary.mean_predicted_standard_deviation_deg / summary.empirical_standard_deviation_deg
    assert summary.accepted_fraction == 1.0
    assert 0.80 <= summary.coverage_95 <= 1.0
    assert 0.65 <= ratio <= 1.45
    assert 0.0 < summary.mean_calibration_variance_fraction < 1.0
    assert 0.0 < summary.mean_measurement_variance_fraction < 1.0
####


def test_joint_calibration_cross_covariance_contributes_to_heading_variance() -> None:
    from dataclasses import replace

    from numpy import outer
    from numpy.linalg import norm

    latitude_rad = 35.0 * pi / 180.0
    calibration = _ideal_calibration()
    batch = simulate_static_measurements(
        ImuModel(),
        latitude_rad=latitude_rad,
        heading_rad=20.0 * pi / 180.0,
        sample_period_s=0.05,
        sample_count=200,
    )
    from playground.calibration.earth_reference import LocalEarthReference, WGS84_EARTH_ROTATION_RADPS
    from playground.calibration.heading_uncertainty import propagate_heading_uncertainty

    earth = LocalEarthReference(latitude_rad=latitude_rad)
    baseline = propagate_heading_uncertainty(
        batch.acceleration.mean(axis=0),
        batch.angular_rate.mean(axis=0),
        zeros((6, 6)),
        calibration,
        latitude_rad=latitude_rad,
        earth_rotation_radps=WGS84_EARTH_ROTATION_RADPS,
        gravity_mps2=float(earth.gravity_ned_mps2[2]),
    )
    accel_direction = baseline.accelerometer_calibration_jacobian / norm(
        baseline.accelerometer_calibration_jacobian
    )
    gyro_direction = baseline.gyroscope_calibration_jacobian / norm(
        baseline.gyroscope_calibration_jacobian
    )
    scale = 1e-10
    correlated = replace(
        calibration,
        accelerometer_coefficient_covariance=scale * outer(accel_direction, accel_direction),
        gyroscope_coefficient_covariance=scale * outer(gyro_direction, gyro_direction),
        accelerometer_gyroscope_coefficient_cross_covariance=(
            0.8 * scale * outer(accel_direction, gyro_direction)
        ),
    )
    propagated = propagate_heading_uncertainty(
        batch.acceleration.mean(axis=0),
        batch.angular_rate.mean(axis=0),
        zeros((6, 6)),
        correlated,
        latitude_rad=latitude_rad,
        earth_rotation_radps=WGS84_EARTH_ROTATION_RADPS,
        gravity_mps2=float(earth.gravity_ned_mps2[2]),
    )

    assert propagated.calibration_cross_covariance_contribution_rad2 > 0.0
    assert abs(
        propagated.total_variance_rad2
        - propagated.calibration_variance_rad2
    ) < 1e-18
####
