"""Continuous-time sampling support around :class:`ImuModel`.

This opt-in layer owns physical sample timing, finite-aperture truth integration,
first-order analog bandwidth, and post-transform fractional channel skew. It is
kept separate from ``AcquiredImuModel`` so existing uniform-cadence contracts do
not silently change semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite
from typing import Annotated, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .acquisition_config import AcquisitionConfig
from .config import ImuConfig
from .model import ImuModel
from .readout import FirReadout, FloatArray, Vector6, clock_tag


class ContinuousTruthSample(BaseModel):
    """Ideal body-frame truth evaluated at one physical instant."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    specific_force_body: tuple[float, float, float]
    angular_rate_body: tuple[float, float, float]
    ambient_temperature_celsius: Annotated[float, Field(allow_inf_nan=False)] = 25.0

    @model_validator(mode="after")
    def finite_vectors(self) -> "ContinuousTruthSample":
        values = np.asarray(self.specific_force_body + self.angular_rate_body, dtype=float)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError("continuous truth vectors must be finite three-vectors")
        ####
        return self
    ####
####


@runtime_checkable
class ContinuousTruthProtocol(Protocol):
    """Callable ideal truth source used only by the simulation-side sampler."""

    def __call__(self, physical_time_s: float) -> ContinuousTruthSample: ...
####


class PhysicalSamplingConfig(BaseModel):
    """Physical oscillator and finite-aperture settings.

    ``frequency_error_ppm`` changes the actual sample cadence. ``boundary_jitter``
    moves physical sample boundaries and therefore changes which truth is sampled;
    it is distinct from the reported-clock jitter in :class:`ClockConfig`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    start_time_s: Annotated[float, Field(allow_inf_nan=False)] = 0.0
    frequency_error_ppm: Annotated[float, Field(gt=-1e6, allow_inf_nan=False)] = 0.0
    boundary_jitter_bound_s: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] = 0.0
    jitter_seed: Annotated[int, Field(ge=0, le=2**63 - 1, strict=True)] = 0
    aperture_fraction: Annotated[float, Field(gt=0.0, le=1.0, allow_inf_nan=False)] = 1.0
    truth_substeps: Annotated[int, Field(ge=1, le=4096, strict=True)] = 8

    @model_validator(mode="after")
    def valid_jitter_envelope(self) -> "PhysicalSamplingConfig":
        return self
    ####
####


class AnalogBandwidthConfig(BaseModel):
    """Independent first-order analog low-pass time constants for the two triads."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    accelerometer_time_constant_s: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] = 0.0
    gyroscope_time_constant_s: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] = 0.0
####


class FractionalSkewConfig(BaseModel):
    """Post-transform causal fractional channel delay.

    This intentionally approximates inter-axis sample-aperture skew after the
    intrinsic sensor transform. It preserves the semantics of full 3x3 intrinsic
    matrices; true asynchronous pre-transform axis apertures require a richer
    analog channel model and are not claimed here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    seconds: tuple[
        Annotated[float, Field(ge=0.0, allow_inf_nan=False)],
        Annotated[float, Field(ge=0.0, allow_inf_nan=False)],
        Annotated[float, Field(ge=0.0, allow_inf_nan=False)],
        Annotated[float, Field(ge=0.0, allow_inf_nan=False)],
        Annotated[float, Field(ge=0.0, allow_inf_nan=False)],
        Annotated[float, Field(ge=0.0, allow_inf_nan=False)],
    ] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
####


class ContinuousAcquisitionConfig(BaseModel):
    """Physical-sampling extension layered ahead of the digital acquisition model."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    physical: PhysicalSamplingConfig = PhysicalSamplingConfig()
    analog: AnalogBandwidthConfig = AnalogBandwidthConfig()
    fractional_skew: FractionalSkewConfig = FractionalSkewConfig()

    def validate_against(self, acquisition: AcquisitionConfig) -> None:
        """Validate cross-config cadence and skew relationships."""
        physical_period = acquisition.sample_period_s / (1.0 + self.physical.frequency_error_ppm * 1e-6)
        if physical_period <= 2.0 * self.physical.boundary_jitter_bound_s:
            raise ValueError("physical period must exceed twice the physical boundary jitter bound")
        ####
        if max(self.fractional_skew.seconds) >= physical_period:
            raise ValueError("fractional channel skew must be smaller than one physical sample period")
        ####
    ####
####


@dataclass(frozen=True, slots=True)
class ContinuousAcquiredImuSample:
    """Sensor-visible sample plus physical support metadata for simulation evaluation."""

    rates_si: Vector6
    start_time_s: float
    end_time_s: float
    sample_index: int
    valid_channels: tuple[bool, bool, bool, bool, bool, bool]
    physical_start_time_s: float
    physical_end_time_s: float

    @property
    def acceleration(self) -> FloatArray:
        return np.asarray(self.rates_si[:3], dtype=float)
    ####

    @property
    def angular_rate(self) -> FloatArray:
        return np.asarray(self.rates_si[3:], dtype=float)
    ####

    @property
    def is_valid(self) -> bool:
        return all(self.valid_channels)
    ####
####


def _first_order_interval(
        state: FloatArray | None, input_value: FloatArray, dt_s: float, tau_s: float,
) -> tuple[FloatArray, FloatArray]:
    """Return exact interval-mean and endpoint for a zero-order-held first-order LPF."""
    u = np.asarray(input_value, dtype=float)
    if state is None or tau_s == 0.0:
        return u.copy(), u.copy()
    ####
    a = exp(-dt_s / tau_s)
    end = u + (state - u) * a
    mean = u + (state - u) * (tau_s / dt_s) * (1.0 - a)
    return np.asarray(mean, dtype=float), np.asarray(end, dtype=float)
####


def _physical_boundary(
        config: PhysicalSamplingConfig, nominal_period_s: float, boundary_index: int,
) -> float:
    """Deterministic physical boundary including oscillator error and bounded jitter."""
    if isinstance(boundary_index, bool) or not isinstance(boundary_index, int) or boundary_index < 0:
        raise ValueError("boundary_index must be a nonnegative integer")
    ####
    period = nominal_period_s / (1.0 + config.frequency_error_ppm * 1e-6)
    jitter = 0.0
    if config.boundary_jitter_bound_s:
        rng = np.random.default_rng(np.random.SeedSequence([config.jitter_seed, boundary_index]))
        jitter = float(rng.uniform(-config.boundary_jitter_bound_s, config.boundary_jitter_bound_s))
    ####
    result = config.start_time_s + boundary_index * period + jitter
    if not isfinite(result):
        raise ValueError("physical sampling boundary overflow")
    ####
    return result
####


class ContinuousAcquiredImuModel:
    """Reference continuous-time acquisition wrapper.

    Processing order: physical scheduler -> finite-aperture truth quadrature ->
    first-order analog triad bandwidth -> ``ImuModel`` -> causal fractional
    channel skew -> existing digital FIR/integer delay -> reported clock tags.
    """

    def __init__(
            self,
            sensor_config: ImuConfig | None = None,
            acquisition_config: AcquisitionConfig | None = None,
            continuous_config: ContinuousAcquisitionConfig | None = None,
            *,
            sensor_rng: np.random.Generator | None = None,
    ) -> None:
        self.sensor_config = sensor_config or ImuConfig()
        self.acquisition_config = acquisition_config or AcquisitionConfig()
        self.continuous_config = continuous_config or ContinuousAcquisitionConfig()
        self.continuous_config.validate_against(self.acquisition_config)
        self._sensor = ImuModel(self.sensor_config, rng=sensor_rng)
        self._readout = FirReadout(self.acquisition_config)
        self._analog_accel_state: FloatArray | None = None
        self._analog_gyro_state: FloatArray | None = None
        self._previous_raw: FloatArray | None = None
        self._sample_count = 0
        self._previous_reported_time: float | None = None
    ####

    def reset(self) -> None:
        self._sensor.reset()
        self._readout.reset()
        self._analog_accel_state = None
        self._analog_gyro_state = None
        self._previous_raw = None
        self._sample_count = 0
        self._previous_reported_time = None
    ####

    def _integrated_truth(self, truth: ContinuousTruthProtocol, start_s: float, end_s: float) -> tuple[FloatArray, FloatArray, float]:
        physical = self.continuous_config.physical
        interval = end_s - start_s
        aperture_width = interval * physical.aperture_fraction
        center = 0.5 * (start_s + end_s)
        aperture_start = center - 0.5 * aperture_width
        aperture_end = center + 0.5 * aperture_width
        step = interval / physical.truth_substeps
        accel_sum = np.zeros(3, dtype=float)
        gyro_sum = np.zeros(3, dtype=float)
        temperature_sum = 0.0
        accumulated_width = 0.0
        for index in range(physical.truth_substeps):
            sub_start = start_s + index * step
            sub_end = sub_start + step
            time_s = 0.5 * (sub_start + sub_end)
            sample = truth(time_s)
            accel_input = np.asarray(sample.specific_force_body, dtype=float)
            gyro_input = np.asarray(sample.angular_rate_body, dtype=float)
            accel_mean, self._analog_accel_state = _first_order_interval(
                self._analog_accel_state, accel_input, step,
                self.continuous_config.analog.accelerometer_time_constant_s,
            )
            gyro_mean, self._analog_gyro_state = _first_order_interval(
                self._analog_gyro_state, gyro_input, step,
                self.continuous_config.analog.gyroscope_time_constant_s,
            )
            overlap = max(0.0, min(sub_end, aperture_end) - max(sub_start, aperture_start))
            if overlap > 0.0:
                accel_sum += accel_mean * overlap
                gyro_sum += gyro_mean * overlap
                temperature_sum += sample.ambient_temperature_celsius * overlap
                accumulated_width += overlap
            ####
        ####
        if accumulated_width <= 0.0:
            raise RuntimeError("truth quadrature did not cover the configured aperture")
        ####
        return accel_sum / accumulated_width, gyro_sum / accumulated_width, temperature_sum / accumulated_width
    ####

    def _apply_fractional_skew(self, current: FloatArray, dt_s: float) -> tuple[FloatArray, tuple[bool, ...]]:
        skew = np.asarray(self.continuous_config.fractional_skew.seconds, dtype=float)
        if self._previous_raw is None:
            valid = tuple(bool(value == 0.0) for value in skew)
            self._previous_raw = current.copy()
            return current.copy(), valid
        ####
        fraction = skew / dt_s
        output = current * (1.0 - fraction) + self._previous_raw * fraction
        self._previous_raw = current.copy()
        return np.asarray(output, dtype=float), (True, True, True, True, True, True)
    ####

    def measure_next(self, truth: ContinuousTruthProtocol) -> ContinuousAcquiredImuSample:
        """Acquire the next physical sample from a callable continuous truth source."""
        start = _physical_boundary(self.continuous_config.physical, self.acquisition_config.sample_period_s, self._sample_count)
        end = _physical_boundary(self.continuous_config.physical, self.acquisition_config.sample_period_s, self._sample_count + 1)
        if end <= start:
            raise ValueError("physical sampling jitter produced a non-increasing interval")
        ####
        force, rate, temperature = self._integrated_truth(truth, start, end)
        sensor = self._sensor.measure_ideal_rates(
            start_time=start,
            end_time=end,
            specific_force_body=force,
            angular_rate_body=rate,
            temperature_celsius=temperature,
        )
        raw = np.concatenate((sensor.acceleration, sensor.angular_rate))
        skewed, skew_valid = self._apply_fractional_skew(raw, end - start)
        filtered, filter_valid = self._readout.step(skewed)
        valid = tuple(bool(a and b) for a, b in zip(skew_valid, filter_valid))
        start_tag = self._previous_reported_time
        if start_tag is None:
            start_tag = clock_tag(self.acquisition_config.clock, start, self._sample_count)
        ####
        end_tag = clock_tag(self.acquisition_config.clock, end, self._sample_count + 1)
        if end_tag <= start_tag:
            raise ValueError("reported timestamps are not representably increasing")
        ####
        result = ContinuousAcquiredImuSample(
            rates_si=(
                float(filtered[0]), float(filtered[1]), float(filtered[2]),
                float(filtered[3]), float(filtered[4]), float(filtered[5]),
            ),
            start_time_s=start_tag,
            end_time_s=end_tag,
            sample_index=self._sample_count,
            valid_channels=(valid[0], valid[1], valid[2], valid[3], valid[4], valid[5]),
            physical_start_time_s=start,
            physical_end_time_s=end,
        )
        self._previous_reported_time = end_tag
        self._sample_count += 1
        return result
    ####
####
