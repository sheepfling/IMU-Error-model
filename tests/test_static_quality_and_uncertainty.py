from math import pi

from numpy import random

from imu_error_model import AxisConfig, ImuConfig, ImuModel
from playground.calibration.gyrocompass import simulate_static_measurements
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


def _calibration_for(config: ImuConfig, samples_per_segment: int, seed: int):
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
            ImuModel(config=config, rng=random.default_rng(seed)),
            program,
        )
    )
####


def test_predicted_heading_confidence_improves_with_more_compass_data() -> None:
    config = ImuConfig(
        accelerometer=AxisConfig(white_noise_density=2e-4),
        gyroscope=AxisConfig(white_noise_density=2e-6),
    )
    short_batch = simulate_static_measurements(
        ImuModel(config=config, rng=random.default_rng(9)),
        latitude_rad=35.0 * pi / 180.0,
        heading_rad=20.0 * pi / 180.0,
        sample_period_s=0.02,
        sample_count=1000,
    )
    long_batch = simulate_static_measurements(
        ImuModel(config=config, rng=random.default_rng(10)),
        latitude_rad=35.0 * pi / 180.0,
        heading_rad=20.0 * pi / 180.0,
        sample_period_s=0.02,
        sample_count=4000,
    )

    short = quality_aware_static_gyrocompass(
        short_batch,
        _ideal_calibration(),
        latitude_rad=35.0 * pi / 180.0,
        covariance_max_lag=0,
    )
    long = quality_aware_static_gyrocompass(
        long_batch,
        _ideal_calibration(),
        latitude_rad=35.0 * pi / 180.0,
        covariance_max_lag=0,
    )

    assert short.accepted and short.uncertainty is not None
    assert long.accepted and long.uncertainty is not None
    assert long.uncertainty.standard_deviation_deg < 0.65 * short.uncertainty.standard_deviation_deg
    assert short.measurement_effective_sample_size is not None
    assert long.measurement_effective_sample_size is not None
    assert float(long.measurement_effective_sample_size.min()) > float(short.measurement_effective_sample_size.min())
####


def test_more_calibration_data_reduces_calibration_heading_variance() -> None:
    config = ImuConfig(
        accelerometer=AxisConfig(white_noise_density=5e-4),
        gyroscope=AxisConfig(white_noise_density=2e-6),
    )
    weak = _calibration_for(config, samples_per_segment=20, seed=11)
    strong = _calibration_for(config, samples_per_segment=300, seed=12)
    batch = simulate_static_measurements(
        ImuModel(config=config, rng=random.default_rng(13)),
        latitude_rad=35.0 * pi / 180.0,
        heading_rad=-18.0 * pi / 180.0,
        sample_period_s=0.02,
        sample_count=2000,
    )

    weak_estimate = quality_aware_static_gyrocompass(
        batch,
        weak,
        latitude_rad=35.0 * pi / 180.0,
        covariance_max_lag=0,
    )
    strong_estimate = quality_aware_static_gyrocompass(
        batch,
        strong,
        latitude_rad=35.0 * pi / 180.0,
        covariance_max_lag=0,
    )

    assert weak_estimate.accepted and weak_estimate.uncertainty is not None
    assert strong_estimate.accepted and strong_estimate.uncertainty is not None
    weak_variance = (
        weak_estimate.uncertainty.accelerometer_calibration_variance_rad2
        + weak_estimate.uncertainty.gyroscope_calibration_variance_rad2
    )
    strong_variance = (
        strong_estimate.uncertainty.accelerometer_calibration_variance_rad2
        + strong_estimate.uncertainty.gyroscope_calibration_variance_rad2
    )
    assert strong_variance < 0.2 * weak_variance
####
