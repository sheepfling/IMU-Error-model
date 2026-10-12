"""Identification and alias tests for continuous acquisition support."""

from __future__ import annotations

import pytest

from playground.calibration.continuous_support import (
    alias_amplitude, fit_analog_gyro_time_constant, physical_clock_angle_error_ppm,
)


def test_multi_frequency_phase_identifies_analog_time_constant() -> None:
    result = fit_analog_gyro_time_constant(
        frequencies_hz=(1.0, 2.0, 4.0, 7.0), sample_period_s=0.002,
        duration_s=4.0, analog_tau_s=0.025,
    )
    assert result.time_constant_s == pytest.approx(0.025, rel=0.012)
    assert max(abs(value - 0.025) for value in result.per_frequency_s) < 5e-4
####


def test_analog_bandwidth_reduces_above_nyquist_alias() -> None:
    raw = alias_amplitude(input_frequency_hz=80.0, sample_period_s=0.01, duration_s=4.0, analog_tau_s=0.0)
    filtered = alias_amplitude(input_frequency_hz=80.0, sample_period_s=0.01, duration_s=4.0, analog_tau_s=0.01)
    assert raw.frequency_hz == pytest.approx(20.0)
    assert filtered.amplitude < 0.25 * raw.amplitude
####


def test_physical_clock_drift_is_not_a_timestamp_only_effect() -> None:
    error = physical_clock_angle_error_ppm(frequency_error_ppm=750.0, sample_count=1000, nominal_period_s=0.01)
    assert error == pytest.approx(750.0, abs=1e-6)
####
