from math import pi

from numpy import random

from imu_error_model import AxisConfig, ImuConfig, ImuModel
from playground.calibration.gyrocompass import wrap_angle_rad
from playground.calibration.known_reference import (
    ferraris_style_reference_program,
    fit_known_reference_calibration,
    simulate_reference_program,
)
from playground.calibration.maytag import maytag_positions, simulate_maytag_gyrocompass


def test_maytag_position_sets_are_explicit() -> None:
    assert maytag_positions(1) == (0.0,)
    assert maytag_positions(2) == (0.0, pi)
    assert maytag_positions(4) == (0.0, pi / 2.0, pi, 3.0 * pi / 2.0)
####


def test_two_position_maytag_rejects_fixed_turn_on_heading_bias() -> None:
    deterministic = ImuConfig(
        accelerometer=AxisConfig(fixed_bias=(0.02, -0.01, 0.01)),
        gyroscope=AxisConfig(fixed_bias=(1e-6, -2e-6, 1e-6)),
    )
    calibration = fit_known_reference_calibration(
        simulate_reference_program(ImuModel(deterministic), ferraris_style_reference_program())
    )
    compass_config = deterministic.model_copy(
        update={
            "gyroscope": deterministic.gyroscope.model_copy(update={"turn_on_bias_std": 2e-5})
        }
    )
    truth_heading = 20.0 * pi / 180.0
    kwargs = dict(
        calibration=calibration,
        latitude_rad=34.7304 * pi / 180.0,
        heading_rad=truth_heading,
        samples_per_position=100,
        sample_period_s=0.02,
    )
    one = simulate_maytag_gyrocompass(
        ImuModel(compass_config, rng=random.default_rng(42)),
        position_count=1,
        **kwargs,
    )
    two = simulate_maytag_gyrocompass(
        ImuModel(compass_config, rng=random.default_rng(42)),
        position_count=2,
        **kwargs,
    )
    one_error = abs(wrap_angle_rad(one.heading_rad - truth_heading))
    two_error = abs(wrap_angle_rad(two.heading_rad - truth_heading))

    assert one_error > 0.5 * pi / 180.0
    assert two_error < 1e-10
####


def test_maytag_confidence_sweep_uses_total_duration_and_cancels_fixed_bias() -> None:
    deterministic = ImuConfig(
        gyroscope=AxisConfig(turn_on_bias_std=2e-5),
    )
    calibration = fit_known_reference_calibration(
        simulate_reference_program(ImuModel(), ferraris_style_reference_program())
    )
    from playground.calibration.maytag import run_maytag_heading_confidence_sweep

    sweep = run_maytag_heading_confidence_sweep(
        model_factory=lambda rng: ImuModel(deterministic, rng=rng),
        calibration=calibration,
        latitude_rad=34.7304 * pi / 180.0,
        heading_rad=20.0 * pi / 180.0,
        position_count=2,
        durations_s=(20.0, 40.0),
        sample_period_s=0.1,
        trial_count=12,
        seed=77,
    )

    assert tuple(point.duration_s for point in sweep.points) == (20.0, 40.0)
    assert max(point.p95_absolute_error_deg for point in sweep.points) < 1e-8
####
