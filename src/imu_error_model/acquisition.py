"""Opt-in physical-temperature / sampled-readout boundary around ImuModel.

This is NOT an ImuModelProtocol implementation: filtered channel histories and
erroneous clock tags must not be mislabeled as instantaneous body-frame truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from typing import Annotated, Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .acquisition_config import AcquisitionConfig
from .checkpoint import ImuModelCheckpoint
from .config import ImuConfig
from .model import ImuModel
from .readout import FirReadout, FloatArray, Vector6, clock_tag
from .thermal import LinearThermalModel
from .thermal_dynamics import thermal_interval


@dataclass(frozen=True, slots=True)
class AcquiredImuSample:
    """Sensor-visible rates, clock tags, and per-channel filter-history validity.

    Rates are SI values in instrument-channel coordinates. Integrating them with
    a chosen clock is a downstream decision. No die temperature or physical-time
    truth is leaked into the sensor-visible result.
    """

    rates_si: Vector6
    start_time_s: float
    end_time_s: float
    temperature_celsius: float
    sample_index: int
    valid_channels: tuple[bool, bool, bool, bool, bool, bool]

    def __post_init__(self) -> None:
        if len(self.rates_si) != 6 or not all(isfinite(x) for x in self.rates_si):
            raise ValueError("sample rates must be six finite values")
        ####
        if len(self.valid_channels) != 6 or any(type(x) is not bool for x in self.valid_channels):
            raise ValueError("sample validity must contain six booleans")
        ####
        if not all(isfinite(x) for x in (self.start_time_s, self.end_time_s, self.temperature_celsius)):
            raise ValueError("sample times and temperature must be finite")
        ####
        if self.end_time_s <= self.start_time_s:
            raise ValueError("sample clock interval must be positive")
        ####
        if type(self.sample_index) is not int or self.sample_index < 0:
            raise ValueError("sample_index must be a nonnegative integer")
        ####
        object.__setattr__(self, "rates_si", tuple(self.rates_si))
        object.__setattr__(self, "valid_channels", tuple(self.valid_channels))
    ####

    @property
    def is_valid(self) -> bool:
        return all(self.valid_channels)
    ####

    @property
    def reported_dt_s(self) -> float:
        return self.end_time_s - self.start_time_s
    ####

    @property
    def acceleration(self) -> FloatArray:
        return np.asarray(self.rates_si[:3], dtype=float)
    ####

    @property
    def angular_rate(self) -> FloatArray:
        return np.asarray(self.rates_si[3:], dtype=float)
    ####
####


class AcquisitionCheckpoint(BaseModel):
    """Simulation-side replay state; never pass this to a calibration estimator."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[1] = 1
    acquisition_config: AcquisitionConfig
    sensor: ImuModelCheckpoint
    sample_count: Annotated[int, Field(ge=0, strict=True)]
    first_physical_time_s: float | None
    previous_physical_time_s: float | None
    previous_reported_time_s: float | None
    die_temperature_celsius: Annotated[float, Field(allow_inf_nan=False)]
    filter_history_newest_first: tuple[Vector6, ...]
####


@runtime_checkable
class AcquiredImuModelProtocol(Protocol):
    """Consumer contract for the optional acquisition-aware sensor boundary."""

    def reset(self) -> None: ...

    def measure_ideal_rates(
            self, *, start_time: float, end_time: float, specific_force_body: FloatArray,
            angular_rate_body: FloatArray, ambient_temperature_celsius: float,
    ) -> AcquiredImuSample: ...
####


class AcquiredImuModel:
    """Reference acquisition backend with independently configured error stages.

    Pipeline: held ambient -> die mean -> ImuModel (including noise/electronics)
    -> digital FIR/readout delay -> erroneous reported clock tags.

    It currently requires uniform physical periods and built-in linear thermal
    coefficients. Die-mean evaluation is exact for linear deterministic thermal
    terms and constant ideal rates, but not exact for temperature-varying noise
    variance, clipping or nonlinear thermal interactions. Refine the time step
    for those combinations; this wrapper is not an analog anti-aliasing model.
    """

    def __init__(
            self, sensor_config: ImuConfig | None = None,
            acquisition_config: AcquisitionConfig | None = None,
            *, sensor_rng: np.random.Generator | None = None,
    ) -> None:
        self.sensor_config = sensor_config or ImuConfig()
        self.acquisition_config = acquisition_config or AcquisitionConfig()
        self._sensor = ImuModel(self.sensor_config, rng=sensor_rng)
        self._readout = FirReadout(self.acquisition_config)
        self._accel_thermal = LinearThermalModel(self.sensor_config.accelerometer)
        self._gyro_thermal = LinearThermalModel(self.sensor_config.gyroscope)
        self._die_temperature = self.acquisition_config.thermal.initial_die_temperature_celsius
        self._first_physical_time: float | None = None
        self._previous_physical_time: float | None = None
        self._previous_reported_time: float | None = None
    ####

    def reset(self) -> None:
        """Resample sensor run-level errors; clear history and reset the clock counter."""
        self._sensor.reset()
        self._readout.reset()
        self._die_temperature = self.acquisition_config.thermal.initial_die_temperature_celsius
        self._first_physical_time = None
        self._previous_physical_time = None
        self._previous_reported_time = None
    ####

    def measure_ideal_rates(
            self, *, start_time: float, end_time: float, specific_force_body: FloatArray,
            angular_rate_body: FloatArray, ambient_temperature_celsius: float,
    ) -> AcquiredImuSample:
        """Sample one physical interval and emit rates with possibly erroneous tags."""
        if not all(isfinite(x) for x in (start_time, end_time, ambient_temperature_celsius)):
            raise ValueError("physical times and ambient temperature must be finite")
        ####
        dt = end_time - start_time
        period = self.acquisition_config.sample_period_s
        if dt <= 0.0 or not isclose(dt, period, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("acquisition requires the configured uniform physical period")
        ####
        if self._previous_physical_time is not None and not isclose(
                start_time, self._previous_physical_time, rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError("physical intervals must be contiguous; reset for a new run")
        ####
        force = np.asarray(specific_force_body, dtype=float)
        rate = np.asarray(angular_rate_body, dtype=float)
        if force.shape != (3,) or rate.shape != (3,) or not np.isfinite(force).all() or not np.isfinite(rate).all():
            raise ValueError("ideal rates must be finite three-vectors")
        ####
        sample_index = self._readout.sample_count
        start_tag = self._previous_reported_time
        if start_tag is None:
            start_tag = clock_tag(self.acquisition_config.clock, start_time, sample_index)
        ####
        end_tag = clock_tag(self.acquisition_config.clock, end_time, sample_index + 1)
        if end_tag <= start_tag:
            raise ValueError("reported timestamps are not representably increasing; use local epochs")
        ####
        temperatures = thermal_interval(
            start_celsius=self._die_temperature, ambient_celsius=ambient_temperature_celsius,
            dt_s=dt, time_constant_s=self.acquisition_config.thermal.time_constant_s,
        )
        # Validate predictable thermal failures before the sensor consumes RNG state.
        self._accel_thermal.evaluate(temperatures.mean_celsius)
        self._gyro_thermal.evaluate(temperatures.mean_celsius)
        output = self._sensor.measure_ideal_rates(
            start_time=start_time, end_time=end_time, specific_force_body=force,
            angular_rate_body=rate, temperature_celsius=temperatures.mean_celsius,
        )
        filtered, valid = self._readout.step(np.concatenate((output.acceleration, output.angular_rate)))
        self._die_temperature = temperatures.end_celsius
        if self._first_physical_time is None:
            self._first_physical_time = start_time
        ####
        self._previous_physical_time = end_time
        self._previous_reported_time = end_tag
        return AcquiredImuSample(
            rates_si=filtered, start_time_s=start_tag, end_time_s=end_tag,
            temperature_celsius=ambient_temperature_celsius, sample_index=sample_index,
            valid_channels=(valid[0], valid[1], valid[2], valid[3], valid[4], valid[5]),
        )
    ####

    def snapshot(self) -> AcquisitionCheckpoint:
        """Return full replay state, including latent temperature and filter history."""
        return AcquisitionCheckpoint(
            acquisition_config=self.acquisition_config, sensor=self._sensor.snapshot(),
            sample_count=self._readout.sample_count,
            first_physical_time_s=self._first_physical_time,
            previous_physical_time_s=self._previous_physical_time,
            previous_reported_time_s=self._previous_reported_time,
            die_temperature_celsius=self._die_temperature,
            filter_history_newest_first=self._readout.history(),
        )
    ####

    @classmethod
    def from_checkpoint(cls, checkpoint: AcquisitionCheckpoint) -> AcquiredImuModel:
        """Create a resumable instance; validation never mutates the input checkpoint."""
        result = cls(checkpoint.sensor.config, checkpoint.acquisition_config)
        result.restore(checkpoint)
        return result
    ####

    def restore(self, checkpoint: AcquisitionCheckpoint) -> None:
        """Validate and atomically replace owned state; unknown versions fail closed."""
        checkpoint = AcquisitionCheckpoint.model_validate_json(checkpoint.model_dump_json())
        if checkpoint.acquisition_config != self.acquisition_config or checkpoint.sensor.config != self.sensor_config:
            raise ValueError("checkpoint configurations do not match")
        ####
        times = (
            checkpoint.first_physical_time_s, checkpoint.previous_physical_time_s,
            checkpoint.previous_reported_time_s,
        )
        if checkpoint.sample_count == 0:
            if any(x is not None for x in times) or checkpoint.sensor.previous_timestamp is not None:
                raise ValueError("empty checkpoint must have no time boundary")
            ####
            if checkpoint.die_temperature_celsius != self.acquisition_config.thermal.initial_die_temperature_celsius:
                raise ValueError("empty checkpoint must have the configured initial temperature")
            ####
        else:
            if any(x is None or not isfinite(x) for x in times):
                raise ValueError("active checkpoint requires finite time boundaries")
            ####
            first, previous, reported = times
            assert first is not None and previous is not None and reported is not None
            if not isclose(
                    previous - first, checkpoint.sample_count*self.acquisition_config.sample_period_s,
                    rel_tol=1e-9, abs_tol=1e-12,
            ):
                raise ValueError("checkpoint duration disagrees with uniform sample count")
            ####
            if checkpoint.sensor.previous_timestamp != previous:
                raise ValueError("sensor and acquisition time boundaries disagree")
            ####
            expected_reported = clock_tag(self.acquisition_config.clock, previous, checkpoint.sample_count)
            if reported != expected_reported:
                raise ValueError("checkpoint clock boundary disagrees with clock configuration")
            ####
        ####
        if checkpoint.sensor.previous_velocity is not None or checkpoint.sensor.previous_orientation is not None:
            raise ValueError("acquisition checkpoint must use the direct-input sensor path")
        ####
        sensor = ImuModel.from_checkpoint(checkpoint.sensor)
        readout = FirReadout(self.acquisition_config)
        readout.restore(checkpoint.filter_history_newest_first, checkpoint.sample_count)
        self._sensor = sensor
        self._readout = readout
        self._die_temperature = checkpoint.die_temperature_celsius
        self._first_physical_time = checkpoint.first_physical_time_s
        self._previous_physical_time = checkpoint.previous_physical_time_s
        self._previous_reported_time = checkpoint.previous_reported_time_s
    ####
####
