from math import pi

from numpy import random

from imu_error_model import AxisConfig, ImuConfig, ImuModel
from playground.calibration.carousel import (
    average_sensor_from_base_rotation,
    simulate_carousel_gyrocompass,
)
from playground.calibration.gyrocompass import simulate_static_gyrocompass, wrap_angle_rad
from playground.calibration.known_reference import (
    ferraris_style_reference_program,
    fit_known_reference_calibration,
    simulate_reference_program,
)


def _calibration(config: ImuConfig):
    return fit_known_reference_calibration(
        simulate_reference_program(ImuModel(config), ferraris_style_reference_program())
    )
####


def test_carousel_interval_average_matches_small_step_midpoint_limit() -> None:
    start = 0.3
    end = start + 1e-7
    average = average_sensor_from_base_rotation(start, end)
    midpoint = average_sensor_from_base_rotation(start + 0.5e-7, start + 0.5e-7)

    assert abs(average - midpoint).max() < 1e-8
####


def test_noiseless_carousel_recovers_heading_with_earth_rotation_and_table_rate() -> None:
    config = ImuConfig(
        accelerometer=AxisConfig(
            fixed_bias=(0.03, -0.02, 0.01),
            sensitivity_matrix=((1.01, 0.003, -0.002), (-0.002, 0.995, 0.004), (0.001, -0.003, 1.006)),
        ),
        gyroscope=AxisConfig(
            fixed_bias=(2e-6, -3e-6, 1e-6),
            sensitivity_matrix=((1.008, 0.003, -0.002), (-0.004, 0.996, 0.002), (0.001, -0.003, 1.005)),
        ),
    )
    truth_heading = -31.0 * pi / 180.0

    result = simulate_carousel_gyrocompass(
        ImuModel(config),
        _calibration(config),
        latitude_rad=34.7304 * pi / 180.0,
        heading_rad=truth_heading,
        rotation_rate_radps=pi / 4.0,
        duration_s=32.0,
        sample_period_s=0.02,
    )

    assert abs(wrap_angle_rad(result.result.heading_rad - truth_heading)) < 1e-10
    assert abs(result.revolution_count - 4.0) < 1e-12
####


def test_continuous_carousel_rejects_sensor_fixed_horizontal_turn_on_bias() -> None:
    deterministic = ImuConfig(
        accelerometer=AxisConfig(fixed_bias=(0.02, -0.01, 0.01)),
        gyroscope=AxisConfig(fixed_bias=(1e-6, -2e-6, 1e-6)),
    )
    calibration = _calibration(deterministic)
    compass_config = deterministic.model_copy(
        update={
            "gyroscope": deterministic.gyroscope.model_copy(update={"turn_on_bias_std": 2e-5})
        }
    )
    truth_heading = 20.0 * pi / 180.0
    duration_s = 64.0
    sample_period_s = 0.02
    sample_count = round(duration_s / sample_period_s)

    static = simulate_static_gyrocompass(
        ImuModel(compass_config, rng=random.default_rng(42)),
        calibration,
        latitude_rad=34.7304 * pi / 180.0,
        heading_rad=truth_heading,
        sample_period_s=sample_period_s,
        sample_count=sample_count,
    )
    carousel = simulate_carousel_gyrocompass(
        ImuModel(compass_config, rng=random.default_rng(42)),
        calibration,
        latitude_rad=34.7304 * pi / 180.0,
        heading_rad=truth_heading,
        rotation_rate_radps=pi / 4.0,
        duration_s=duration_s,
        sample_period_s=sample_period_s,
    )
    static_error = abs(wrap_angle_rad(static.heading_rad - truth_heading))
    carousel_error = abs(wrap_angle_rad(carousel.result.heading_rad - truth_heading))

    assert static_error > 0.5 * pi / 180.0
    assert carousel_error < 1e-10
####
