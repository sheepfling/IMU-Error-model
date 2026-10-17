from math import pi

from numpy import asarray

from imu_error_model import ImuModel
from playground.calibration.gyrocompass import simulate_static_measurements
from playground.calibration.known_reference import (
    ferraris_style_reference_program,
    fit_known_reference_calibration,
    simulate_reference_program,
)
from playground.calibration.motion import HarmonicMotionExcitation
from playground.calibration.quality import (
    StationarityFlag,
    StationarityThresholds,
    quality_aware_static_gyrocompass,
)


def _ideal_calibration():
    return fit_known_reference_calibration(
        simulate_reference_program(ImuModel(), ferraris_style_reference_program())
    )
####


def test_clean_static_record_passes_quality_gate() -> None:
    latitude_rad = 35.0 * pi / 180.0
    batch = simulate_static_measurements(
        ImuModel(),
        latitude_rad=latitude_rad,
        heading_rad=-15.0 * pi / 180.0,
        sample_period_s=0.02,
        sample_count=200,
    )

    result = quality_aware_static_gyrocompass(
        batch,
        _ideal_calibration(),
        latitude_rad=latitude_rad,
        covariance_max_lag=0,
    )

    assert result.accepted
    assert result.stationarity.flags == ()
    assert result.result is not None
####


def test_controlled_motion_contamination_is_rejected() -> None:
    latitude_rad = 35.0 * pi / 180.0
    motion = HarmonicMotionExcitation(
        specific_force_amplitude_body=asarray((0.0, 0.0, 0.3)),
        angular_rate_amplitude_body=asarray((0.0, 0.01, 0.0)),
        frequency_hz=0.5,
    )
    batch = simulate_static_measurements(
        ImuModel(),
        latitude_rad=latitude_rad,
        heading_rad=10.0 * pi / 180.0,
        sample_period_s=0.02,
        sample_count=500,
        motion_excitation=motion,
    )

    result = quality_aware_static_gyrocompass(
        batch,
        _ideal_calibration(),
        latitude_rad=latitude_rad,
        covariance_max_lag=0,
    )

    assert not result.accepted
    assert result.result is None
    assert StationarityFlag.ACCELERATION_VARIATION in result.stationarity.flags
    assert StationarityFlag.ANGULAR_RATE_VARIATION in result.stationarity.flags
####


def test_short_record_reports_both_sample_and_duration_failures() -> None:
    latitude_rad = 35.0 * pi / 180.0
    batch = simulate_static_measurements(
        ImuModel(),
        latitude_rad=latitude_rad,
        heading_rad=0.0,
        sample_period_s=0.02,
        sample_count=20,
    )

    result = quality_aware_static_gyrocompass(
        batch,
        _ideal_calibration(),
        latitude_rad=latitude_rad,
        thresholds=StationarityThresholds(minimum_sample_count=100, minimum_duration_s=2.0),
        covariance_max_lag=0,
    )

    assert StationarityFlag.INSUFFICIENT_SAMPLES in result.stationarity.flags
    assert StationarityFlag.INSUFFICIENT_DURATION in result.stationarity.flags
####


def test_polar_reference_geometry_is_rejected_before_triad_failure() -> None:
    batch = simulate_static_measurements(
        ImuModel(),
        latitude_rad=pi / 2.0,
        heading_rad=0.0,
        sample_period_s=0.02,
        sample_count=200,
    )

    result = quality_aware_static_gyrocompass(
        batch,
        _ideal_calibration(),
        latitude_rad=pi / 2.0,
        covariance_max_lag=0,
    )

    assert not result.accepted
    assert StationarityFlag.EARTH_RATE_GEOMETRY_ILL_CONDITIONED in result.stationarity.flags
    assert result.stationarity.horizontal_earth_rate_radps is not None
    assert result.stationarity.horizontal_earth_rate_radps < 1e-12
####
