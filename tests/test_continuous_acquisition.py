"""Independent law checks for physical sampling and continuous-time acquisition."""

from __future__ import annotations

from math import atan, pi, sin, sqrt

import numpy as np
import pytest

from imu_error_model import (
    AcquisitionConfig, AnalogBandwidthConfig, ContinuousAcquiredImuModel,
    ContinuousAcquisitionConfig, ContinuousTruthSample, FractionalSkewConfig,
    PhysicalSamplingConfig,
)


def _zero_truth(physical_time_s: float) -> ContinuousTruthSample:
    del physical_time_s
    return ContinuousTruthSample(specific_force_body=(0.0, 0.0, 0.0), angular_rate_body=(0.0, 0.0, 0.0))
####


def test_physical_clock_error_changes_actual_sample_support() -> None:
    config = ContinuousAcquisitionConfig(physical=PhysicalSamplingConfig(frequency_error_ppm=1000.0, truth_substeps=1))
    model = ContinuousAcquiredImuModel(acquisition_config=AcquisitionConfig(sample_period_s=0.01), continuous_config=config)
    samples = [model.measure_next(_zero_truth) for _ in range(1000)]
    duration = samples[-1].physical_end_time_s - samples[0].physical_start_time_s
    assert duration == pytest.approx(10.0 / 1.001, abs=1e-12)
    assert duration != pytest.approx(10.0)
####


def test_physical_boundary_jitter_is_deterministic_and_changes_truth_sampling() -> None:
    cfg = ContinuousAcquisitionConfig(
        physical=PhysicalSamplingConfig(boundary_jitter_bound_s=0.001, jitter_seed=8, truth_substeps=4)
    )
    a = ContinuousAcquiredImuModel(continuous_config=cfg)
    b = ContinuousAcquiredImuModel(continuous_config=cfg)

    def ramp(physical_time_s: float) -> ContinuousTruthSample:
        return ContinuousTruthSample(specific_force_body=(physical_time_s, 0.0, 0.0), angular_rate_body=(0.0, 0.0, 0.0))
    ####

    left = [a.measure_next(ramp) for _ in range(20)]
    right = [b.measure_next(ramp) for _ in range(20)]
    assert [x.physical_end_time_s for x in left] == [x.physical_end_time_s for x in right]
    np.testing.assert_array_equal([x.acceleration for x in left], [x.acceleration for x in right])
    assert any(abs((x.physical_end_time_s - x.physical_start_time_s) - 0.01) > 1e-8 for x in left)
####


def test_finite_aperture_quadrature_matches_sinusoid_interval_average() -> None:
    frequency_hz = 7.0
    dt = 0.01
    model = ContinuousAcquiredImuModel(
        acquisition_config=AcquisitionConfig(sample_period_s=dt),
        continuous_config=ContinuousAcquisitionConfig(physical=PhysicalSamplingConfig(truth_substeps=512)),
    )

    def truth(physical_time_s: float) -> ContinuousTruthSample:
        return ContinuousTruthSample(
            specific_force_body=(0.0, 0.0, 0.0),
            angular_rate_body=(sin(2.0*pi*frequency_hz*physical_time_s), 0.0, 0.0),
        )
    ####

    sample = model.measure_next(truth)
    expected = (1.0 - np.cos(2.0*pi*frequency_hz*dt)) / (2.0*pi*frequency_hz*dt)
    assert sample.angular_rate[0] == pytest.approx(expected, abs=2e-7)
####


def test_analog_lowpass_matches_frequency_response_after_settling() -> None:
    frequency_hz = 2.0
    tau_s = 0.08
    dt = 0.002
    model = ContinuousAcquiredImuModel(
        acquisition_config=AcquisitionConfig(sample_period_s=dt),
        continuous_config=ContinuousAcquisitionConfig(
            physical=PhysicalSamplingConfig(truth_substeps=8),
            analog=AnalogBandwidthConfig(gyroscope_time_constant_s=tau_s),
        ),
    )

    def truth(physical_time_s: float) -> ContinuousTruthSample:
        return ContinuousTruthSample(
            specific_force_body=(0.0, 0.0, 0.0),
            angular_rate_body=(sin(2.0*pi*frequency_hz*physical_time_s), 0.0, 0.0),
        )
    ####

    values = np.asarray([model.measure_next(truth).angular_rate[0] for _ in range(5000)])
    times = (np.arange(values.size) + 0.5) * dt
    keep = times > 4.0
    basis = np.column_stack((np.sin(2*pi*frequency_hz*times[keep]), np.cos(2*pi*frequency_hz*times[keep])))
    coeff, *_ = np.linalg.lstsq(basis, values[keep], rcond=None)
    amplitude = float(np.linalg.norm(coeff))
    expected = 1.0 / sqrt(1.0 + (2*pi*frequency_hz*tau_s)**2)
    assert amplitude == pytest.approx(expected, rel=2e-3)
    phase = float(atan(coeff[1] / coeff[0]))
    assert phase == pytest.approx(-atan(2*pi*frequency_hz*tau_s), abs=3e-3)
####


def test_fractional_skew_delays_post_transform_ramp() -> None:
    skew_s = 0.0025
    dt = 0.01
    model = ContinuousAcquiredImuModel(
        acquisition_config=AcquisitionConfig(sample_period_s=dt),
        continuous_config=ContinuousAcquisitionConfig(
            physical=PhysicalSamplingConfig(truth_substeps=64),
            fractional_skew=FractionalSkewConfig(seconds=(0.0,0.0,0.0,skew_s,0.0,0.0)),
        ),
    )

    def ramp(physical_time_s: float) -> ContinuousTruthSample:
        return ContinuousTruthSample(
            specific_force_body=(0.0, 0.0, 0.0), angular_rate_body=(physical_time_s, 0.0, 0.0),
        )
    ####

    first = model.measure_next(ramp)
    second = model.measure_next(ramp)
    assert first.valid_channels[3] is False
    expected_current = 0.015
    assert second.angular_rate[0] == pytest.approx(expected_current - skew_s, abs=2e-5)
    assert second.valid_channels[3] is True
####


def test_invalid_cross_config_skew_is_rejected() -> None:
    config = ContinuousAcquisitionConfig(fractional_skew=FractionalSkewConfig(seconds=(0.02,0,0,0,0,0)))
    with pytest.raises(ValueError, match="smaller than one physical sample period"):
        ContinuousAcquiredImuModel(acquisition_config=AcquisitionConfig(sample_period_s=0.01), continuous_config=config)
    ####
####
