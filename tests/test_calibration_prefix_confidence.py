from math import pi

from numpy import random
import pytest

from imu_error_model import AxisConfig, ImuConfig, ImuModel
from playground.calibration.data_confidence import evaluate_prefix_confidence
from playground.calibration.gyrocompass import simulate_static_measurements
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


def test_prefix_trace_reports_first_quality_pass_and_confidence_target() -> None:
    latitude_rad = 35.0 * pi / 180.0
    config = ImuConfig(
        accelerometer=AxisConfig(white_noise_density=2e-4),
        gyroscope=AxisConfig(white_noise_density=1e-6),
    )
    batch = simulate_static_measurements(
        ImuModel(config=config, rng=random.default_rng(101)),
        latitude_rad=latitude_rad,
        heading_rad=12.0 * pi / 180.0,
        sample_period_s=0.1,
        sample_count=800,
    )

    trace = evaluate_prefix_confidence(
        batch,
        _ideal_calibration(),
        latitude_rad=latitude_rad,
        durations_s=(1.0, 5.0, 20.0, 80.0),
        target_half_width_deg=0.5,
        covariance_max_lag=0,
    )

    assert not trace.points[0].accepted
    assert "insufficient_samples" in trace.points[0].stationarity_flags
    assert trace.first_accepted_duration_s == 20.0
    assert trace.duration_to_target_s in {20.0, 80.0}
    accepted_widths = tuple(
        point.predicted_95_half_width_deg
        for point in trace.points
        if point.predicted_95_half_width_deg is not None
    )
    assert accepted_widths[-1] < accepted_widths[0]
    assert trace.points[-1].minimum_effective_sample_size is not None
    assert trace.points[2].minimum_effective_sample_size is not None
    assert trace.points[-1].minimum_effective_sample_size > trace.points[2].minimum_effective_sample_size
####


def test_prefix_trace_rejects_unavailable_or_duplicate_prefixes() -> None:
    batch = simulate_static_measurements(
        ImuModel(),
        latitude_rad=0.0,
        heading_rad=0.0,
        sample_period_s=0.1,
        sample_count=20,
    )

    with pytest.raises(ValueError, match="exceeds"):
        evaluate_prefix_confidence(
            batch,
            _ideal_calibration(),
            latitude_rad=0.0,
            durations_s=(1.0, 3.0),
            target_half_width_deg=1.0,
        )
    ####
    with pytest.raises(ValueError, match="duplicate sample counts"):
        evaluate_prefix_confidence(
            batch,
            _ideal_calibration(),
            latitude_rad=0.0,
            durations_s=(0.11, 0.12),
            target_half_width_deg=1.0,
        )
    ####
####
