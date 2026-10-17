"""Opt-in sampled readout/thermal/clock configuration, separate from sensor errors."""

from __future__ import annotations

from typing import Annotated, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Finite = Annotated[float, Field(allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
Delay = Annotated[int, Field(ge=0, le=4096, strict=True)]


class FirFilterConfig(BaseModel):
    """Causal unity-DC-gain FIR; taps are ordered current to oldest sample."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    taps: Annotated[tuple[Finite, ...], Field(min_length=1, max_length=4096)] = (1.0,)

    @field_validator("taps")
    @classmethod
    def unity_dc_gain(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not np.isclose(sum(value), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("FIR taps must sum to one; high-pass readouts are not supported")
        ####
        return value
    ####
####


class ClockConfig(BaseModel):
    """Reported clock error, NOT physical sampling jitter or oscillator scheduling.

    tag(t, k) = epoch + offset + (t - epoch)*(1 + ppm*1e-6) + jitter(k).
    Adjacent intervals share the same jittered boundary, without sorting/clamping.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    epoch_s: Finite = 0.0
    offset_s: Finite = 0.0
    frequency_error_ppm: Annotated[float, Field(gt=-1e6, allow_inf_nan=False)] = 0.0
    timestamp_jitter_bound_s: Nonnegative = 0.0
    jitter_seed: Annotated[int, Field(ge=0, le=2**63 - 1, strict=True)] = 0
####


class ThermalLagConfig(BaseModel):
    """One die-temperature node driven by zero-order-held ambient temperature."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    time_constant_s: Nonnegative = 0.0
    initial_die_temperature_celsius: Finite = 25.0
####


class AcquisitionConfig(BaseModel):
    """Uniform physical sampling followed by thermal, digital and timestamp effects.

    Axis delays are integer-sample readout delays; they do not shift the physical
    sampling aperture. Delays and taps act in fixed instrument-channel coordinates.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[1] = 1
    sample_period_s: Annotated[float, Field(gt=0.0, allow_inf_nan=False)] = 0.01
    accelerometer_filter: FirFilterConfig = FirFilterConfig()
    gyroscope_filter: FirFilterConfig = FirFilterConfig()
    accelerometer_delay_samples: tuple[Delay, Delay, Delay] = (0, 0, 0)
    gyroscope_delay_samples: tuple[Delay, Delay, Delay] = (0, 0, 0)
    thermal: ThermalLagConfig = ThermalLagConfig()
    clock: ClockConfig = ClockConfig()

    @model_validator(mode="after")
    def monotone_clock_envelope(self) -> Self:
        span = self.sample_period_s * (1.0 + self.clock.frequency_error_ppm * 1e-6)
        if not np.isfinite(span) or span <= 2.0 * self.clock.timestamp_jitter_bound_s:
            raise ValueError("reported period must exceed twice the timestamp jitter bound")
        ####
        return self
    ####
####
