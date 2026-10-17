from math import pi

from numpy import random

from imu_error_model import AxisConfig, ImuConfig, ImuModel
from playground.calibration.gyrocompass import simulate_static_gyrocompass, wrap_angle_rad
from playground.calibration.known_reference import (
    ferraris_style_reference_program,
    fit_known_reference_calibration,
    simulate_reference_program,
)


def _config(*, noisy: bool) -> ImuConfig:
    return ImuConfig(
        accelerometer=AxisConfig(
            white_noise_density=0.0005 if noisy else 0.0,
            fixed_bias=(0.08, -0.05, 0.03),
            sensitivity_matrix=(
                (1.012, 0.004, -0.002),
                (-0.003, 0.993, 0.005),
                (0.001, -0.004, 1.007),
            ),
        ),
        gyroscope=AxisConfig(
            white_noise_density=2e-6 if noisy else 0.0,
            fixed_bias=(8e-6, -5e-6, 4e-6),
            sensitivity_matrix=(
                (1.008, 0.003, -0.002),
                (-0.004, 0.996, 0.002),
                (0.001, -0.003, 1.005),
            ),
        ),
        gyroscope_specific_force_sensitivity=(
            (2e-7, -1e-7, 0.5e-7),
            (-0.5e-7, 1e-7, 1.5e-7),
            (0.25e-7, -0.5e-7, 0.75e-7),
        ),
    )
####


def test_noiseless_calibrated_gyrocompass_recovers_heading() -> None:
    config = _config(noisy=False)
    calibration = fit_known_reference_calibration(
        simulate_reference_program(ImuModel(config=config), ferraris_style_reference_program())
    )
    truth_heading = 37.0 * pi / 180.0

    result = simulate_static_gyrocompass(
        ImuModel(config=config),
        calibration,
        latitude_rad=34.7304 * pi / 180.0,
        heading_rad=truth_heading,
        sample_count=20,
    )

    assert abs(wrap_angle_rad(result.heading_rad - truth_heading)) < 1e-10
    assert result.earth_rate_residual_radps < 1e-12
####


def test_noisy_calibrated_gyrocompass_averages_toward_heading() -> None:
    config = _config(noisy=True)
    calibration_program = tuple(
        type(segment)(
            label=segment.label,
            duration_s=2.0,
            specific_force_body=segment.specific_force_body,
            angular_rate_body=segment.angular_rate_body,
            temperature_celsius=segment.temperature_celsius,
            sample_count=200,
        )
        for segment in ferraris_style_reference_program(duration_s=2.0)
    )
    calibration = fit_known_reference_calibration(
        simulate_reference_program(
            ImuModel(config=config, rng=random.default_rng(44)),
            calibration_program,
        )
    )
    truth_heading = -28.0 * pi / 180.0

    result = simulate_static_gyrocompass(
        ImuModel(config=config, rng=random.default_rng(45)),
        calibration,
        latitude_rad=34.7304 * pi / 180.0,
        heading_rad=truth_heading,
        sample_period_s=0.02,
        sample_count=6000,
    )

    heading_error_deg = abs(wrap_angle_rad(result.heading_rad - truth_heading)) * 180.0 / pi
    assert heading_error_deg < 1.5
####
