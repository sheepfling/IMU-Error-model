"""Calibration experiments for physical sampling and continuous-time support."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, pi, tan

import numpy as np

from imu_error_model import (
    AcquisitionConfig, AnalogBandwidthConfig, ContinuousAcquiredImuModel,
    ContinuousAcquisitionConfig, ContinuousTruthSample, PhysicalSamplingConfig,
)


@dataclass(frozen=True, slots=True)
class SineResponseEstimate:
    frequency_hz: float
    amplitude: float
    phase_rad: float
####


@dataclass(frozen=True, slots=True)
class AnalogTauEstimate:
    time_constant_s: float
    per_frequency_s: tuple[float, ...]
    responses: tuple[SineResponseEstimate, ...]
####


def _fit_sine(values: np.ndarray, centers_s: np.ndarray, frequency_hz: float) -> SineResponseEstimate:
    phase = 2.0 * pi * frequency_hz * centers_s
    design = np.column_stack((np.sin(phase), np.cos(phase), np.ones_like(phase)))
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    sine, cosine = float(coefficients[0]), float(coefficients[1])
    return SineResponseEstimate(
        frequency_hz=frequency_hz,
        amplitude=float(np.hypot(sine, cosine)),
        phase_rad=float(atan2(cosine, sine)),
    )
####


def simulate_gyro_sine_response(
        *, frequency_hz: float, sample_period_s: float, duration_s: float,
        analog_tau_s: float, discard_s: float = 1.0,
) -> SineResponseEstimate:
    """Fit the sampled X-gyro response to a unit continuous sinusoid."""
    model = ContinuousAcquiredImuModel(
        acquisition_config=AcquisitionConfig(sample_period_s=sample_period_s),
        continuous_config=ContinuousAcquisitionConfig(
            physical=PhysicalSamplingConfig(truth_substeps=32),
            analog=AnalogBandwidthConfig(gyroscope_time_constant_s=analog_tau_s),
        ),
    )

    def truth(physical_time_s: float) -> ContinuousTruthSample:
        return ContinuousTruthSample(
            specific_force_body=(0.0, 0.0, 0.0),
            angular_rate_body=(float(np.sin(2.0*pi*frequency_hz*physical_time_s)), 0.0, 0.0),
        )
    ####

    count = int(round(duration_s / sample_period_s))
    samples = [model.measure_next(truth) for _ in range(count)]
    centers = np.asarray(
        [0.5*(sample.physical_start_time_s + sample.physical_end_time_s) for sample in samples], dtype=float,
    )
    values = np.asarray([sample.angular_rate[0] for sample in samples], dtype=float)
    keep = centers >= discard_s
    if int(np.count_nonzero(keep)) < 8:
        raise ValueError("insufficient settled samples for sine-response fit")
    ####
    return _fit_sine(values[keep], centers[keep], frequency_hz)
####


def fit_analog_gyro_time_constant(
        *, frequencies_hz: tuple[float, ...], sample_period_s: float,
        duration_s: float, analog_tau_s: float,
) -> AnalogTauEstimate:
    """Estimate one first-order gyro time constant from independently excited phases."""
    if len(frequencies_hz) < 2:
        raise ValueError("at least two excitation frequencies are required")
    ####
    responses = tuple(
        simulate_gyro_sine_response(
            frequency_hz=frequency, sample_period_s=sample_period_s,
            duration_s=duration_s, analog_tau_s=analog_tau_s,
        )
        for frequency in frequencies_hz
    )
    estimates = []
    for response in responses:
        omega = 2.0*pi*response.frequency_hz
        value = -tan(response.phase_rad) / omega
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("phase response does not identify a positive first-order time constant")
        ####
        estimates.append(float(value))
    ####
    return AnalogTauEstimate(
        time_constant_s=float(np.median(estimates)),
        per_frequency_s=tuple(estimates),
        responses=responses,
    )
####


def alias_amplitude(
        *, input_frequency_hz: float, sample_period_s: float, duration_s: float,
        analog_tau_s: float,
) -> SineResponseEstimate:
    """Measure an above-Nyquist input at its folded alias frequency."""
    sample_rate = 1.0 / sample_period_s
    folded = abs(((input_frequency_hz + 0.5*sample_rate) % sample_rate) - 0.5*sample_rate)
    if folded == 0.0:
        raise ValueError("input frequency folds to DC; choose a nondegenerate alias case")
    ####
    model = ContinuousAcquiredImuModel(
        acquisition_config=AcquisitionConfig(sample_period_s=sample_period_s),
        continuous_config=ContinuousAcquisitionConfig(
            physical=PhysicalSamplingConfig(truth_substeps=128),
            analog=AnalogBandwidthConfig(gyroscope_time_constant_s=analog_tau_s),
        ),
    )

    def truth(physical_time_s: float) -> ContinuousTruthSample:
        return ContinuousTruthSample(
            specific_force_body=(0.0, 0.0, 0.0),
            angular_rate_body=(float(np.sin(2.0*pi*input_frequency_hz*physical_time_s)), 0.0, 0.0),
        )
    ####

    count = int(round(duration_s / sample_period_s))
    samples = [model.measure_next(truth) for _ in range(count)]
    centers = np.asarray(
        [0.5*(sample.physical_start_time_s + sample.physical_end_time_s) for sample in samples], dtype=float,
    )
    values = np.asarray([sample.angular_rate[0] for sample in samples], dtype=float)
    keep = centers >= min(1.0, 0.25*duration_s)
    return _fit_sine(values[keep], centers[keep], folded)
####


def physical_clock_angle_error_ppm(*, frequency_error_ppm: float, sample_count: int, nominal_period_s: float) -> float:
    """Return nominal-count integration error versus true elapsed angle for constant unit rate."""
    model = ContinuousAcquiredImuModel(
        acquisition_config=AcquisitionConfig(sample_period_s=nominal_period_s),
        continuous_config=ContinuousAcquisitionConfig(
            physical=PhysicalSamplingConfig(frequency_error_ppm=frequency_error_ppm, truth_substeps=1),
        ),
    )

    def truth(physical_time_s: float) -> ContinuousTruthSample:
        del physical_time_s
        return ContinuousTruthSample(
            specific_force_body=(0.0, 0.0, 0.0), angular_rate_body=(1.0, 0.0, 0.0),
        )
    ####

    samples = [model.measure_next(truth) for _ in range(sample_count)]
    true_elapsed = samples[-1].physical_end_time_s - samples[0].physical_start_time_s
    nominal_elapsed = sample_count * nominal_period_s
    return float((nominal_elapsed / true_elapsed - 1.0) * 1e6)
####
