"""Causal digital readout and exact covariance under IID prefilter rate noise."""

from __future__ import annotations

from collections import deque
from math import isfinite, pi
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .acquisition_config import AcquisitionConfig, ClockConfig

FloatArray: TypeAlias = NDArray[np.float64]
Vector6: TypeAlias = tuple[float, float, float, float, float, float]


def vector6(value: FloatArray) -> Vector6:
    """Copy one finite six-channel vector into an immutable tuple."""
    array = np.asarray(value, dtype=float)
    if array.shape != (6,) or not np.isfinite(array).all():
        raise ValueError("rate data must have finite shape (6,)")
    ####
    return (float(array[0]), float(array[1]), float(array[2]), float(array[3]), float(array[4]), float(array[5]))
####


def effective_fir_kernels(config: AcquisitionConfig) -> FloatArray:
    """Return padded (6, L) effective taps, including channel readout delays."""
    taps = [config.accelerometer_filter.taps] * 3 + [config.gyroscope_filter.taps] * 3
    delays = config.accelerometer_delay_samples + config.gyroscope_delay_samples
    size = max(len(t) + d for t, d in zip(taps, delays))
    kernels = np.zeros((6, size), dtype=float)
    for axis, (coefficients, delay) in enumerate(zip(taps, delays)):
        kernels[axis, delay:delay + len(coefficients)] = coefficients
    ####
    return kernels
####


def fir_frequency_response(config: AcquisitionConfig, frequency_hz: float) -> NDArray[np.complex128]:
    """Signed-frequency digital response, excluding input aperture and clock tags."""
    if not isfinite(frequency_hz) or abs(frequency_hz) >= 0.5 / config.sample_period_s:
        raise ValueError("frequency must lie strictly inside the physical Nyquist band")
    ####
    kernels = effective_fir_kernels(config)
    phase = np.exp(-2j * pi * frequency_hz * config.sample_period_s * np.arange(kernels.shape[1]))
    return np.asarray(kernels @ phase, dtype=np.complex128)
####


def filtered_white_mean_covariance(
        input_rate_covariance: FloatArray, config: AcquisitionConfig, sample_count: int,
) -> FloatArray:
    """Exact mean covariance after FIR warm-up, for independent prefilter samples.

    No stationary bias, calibration uncertainty, or clock error is silently
    included. The input is rate-sample covariance, NOT a noise-density matrix.
    """
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError("sample_count must be a positive integer")
    ####
    covariance = np.asarray(input_rate_covariance, dtype=float)
    if covariance.shape != (6, 6) or not np.isfinite(covariance).all():
        raise ValueError("input covariance must be finite with shape (6, 6)")
    ####
    scale = max(float(np.max(np.abs(covariance))), np.finfo(float).tiny)
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12*scale):
        raise ValueError("input covariance must be symmetric")
    ####
    if np.linalg.eigvalsh(covariance).min() < -1e-12*scale:
        raise ValueError("input covariance must be positive semidefinite")
    ####
    kernels = effective_fir_kernels(config)
    average = np.ones(sample_count, dtype=float) / sample_count
    weights = np.vstack([np.convolve(row, average) for row in kernels])
    return np.asarray(covariance * (weights @ weights.T), dtype=float)
####


def clock_tag(config: ClockConfig, physical_time_s: float, boundary_index: int) -> float:
    """Deterministic counter-indexed bounded timestamp jitter, independent of sensor RNG."""
    if not isfinite(physical_time_s):
        raise ValueError("physical_time_s must be finite")
    ####
    if isinstance(boundary_index, bool) or not isinstance(boundary_index, int) or boundary_index < 0:
        raise ValueError("boundary_index must be a nonnegative integer")
    ####
    jitter = 0.0
    if config.timestamp_jitter_bound_s:
        rng = np.random.default_rng(np.random.SeedSequence([config.jitter_seed, boundary_index]))
        jitter = float(rng.uniform(-config.timestamp_jitter_bound_s, config.timestamp_jitter_bound_s))
    ####
    elapsed = physical_time_s - config.epoch_s
    result = config.epoch_s + config.offset_s + elapsed * (1.0 + config.frequency_error_ppm*1e-6) + jitter
    if not isfinite(result):
        raise ValueError("reported timestamp overflow")
    ####
    return result
####


class FirReadout:
    """Stateful uniform-cadence instrument-channel FIR, with explicit invalid warm-up."""

    def __init__(self, config: AcquisitionConfig) -> None:
        self.config = config
        self.kernels = effective_fir_kernels(config)
        self.kernels.setflags(write=False)
        self._history: deque[Vector6] = deque(maxlen=self.kernels.shape[1])
        self.sample_count = 0
        self._required = (
            tuple(len(config.accelerometer_filter.taps) + x for x in config.accelerometer_delay_samples)
            + tuple(len(config.gyroscope_filter.taps) + x for x in config.gyroscope_delay_samples)
        )
    ####

    def reset(self) -> None:
        self._history.clear()
        self.sample_count = 0
    ####

    def step(self, rates: FloatArray) -> tuple[Vector6, tuple[bool, ...]]:
        values = vector6(rates)
        history = (values, *tuple(self._history))[:self.kernels.shape[1]]
        array = np.asarray(history, dtype=float)
        result = np.sum(self.kernels[:, :len(history)] * array.T, axis=1)
        output = vector6(result)
        self._history.appendleft(values)
        self.sample_count += 1
        return output, tuple(self.sample_count >= required for required in self._required)
    ####

    def history(self) -> tuple[Vector6, ...]:
        return tuple(self._history)
    ####

    def restore(self, history: tuple[Vector6, ...], sample_count: int) -> None:
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 0:
            raise ValueError("sample_count must be a nonnegative integer")
        ####
        if len(history) != min(sample_count, self.kernels.shape[1]):
            raise ValueError("history length does not match sample count and filter support")
        ####
        checked = tuple(vector6(np.asarray(row, dtype=float)) for row in history)
        self._history = deque(checked, maxlen=self.kernels.shape[1])
        self.sample_count = sample_count
    ####
####
