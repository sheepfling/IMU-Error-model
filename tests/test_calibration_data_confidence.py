from math import pi

from imu_error_model import AxisConfig, ImuConfig, ImuModel
from playground.calibration.data_confidence import (
    run_heading_confidence_sweep,
    white_noise_heading_sigma_deg,
)
from playground.calibration.known_reference import (
    ferraris_style_reference_program,
    fit_known_reference_calibration,
    simulate_reference_program,
)


def _ideal_calibration():
    return fit_known_reference_calibration(
        simulate_reference_program(ImuModel(), ferraris_style_reference_program())
    )
####


def test_white_noise_heading_confidence_improves_with_averaging_time() -> None:
    noise_density = 2e-6
    config = ImuConfig(gyroscope=AxisConfig(white_noise_density=noise_density))
    latitude_rad = 35.0 * pi / 180.0
    sweep = run_heading_confidence_sweep(
        model_factory=lambda rng: ImuModel(config, rng=rng),
        calibration=_ideal_calibration(),
        latitude_rad=latitude_rad,
        heading_rad=20.0 * pi / 180.0,
        durations_s=(10.0, 40.0),
        sample_period_s=0.05,
        trial_count=40,
        seed=4,
    )
    short, long = sweep.points
    predicted_long = white_noise_heading_sigma_deg(
        noise_density,
        averaging_duration_s=long.duration_s,
        latitude_rad=latitude_rad,
    )

    assert long.standard_deviation_deg < short.standard_deviation_deg
    assert 0.5 < long.standard_deviation_deg / predicted_long < 1.5
####


def test_colored_bias_breaks_white_noise_only_data_scaling() -> None:
    latitude_rad = 35.0 * pi / 180.0
    config = ImuConfig(
        gyroscope=AxisConfig(
            white_noise_density=2e-6,
            bias_std=5e-6,
            bias_correlation_time=100.0,
        )
    )
    sweep = run_heading_confidence_sweep(
        model_factory=lambda rng: ImuModel(config, rng=rng),
        calibration=_ideal_calibration(),
        latitude_rad=latitude_rad,
        heading_rad=20.0 * pi / 180.0,
        durations_s=(10.0, 40.0),
        sample_period_s=0.05,
        trial_count=30,
        seed=5,
    )

    assert sweep.points[1].standard_deviation_deg > 2.0
####
