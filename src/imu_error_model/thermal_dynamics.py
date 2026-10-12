"""Exact one-node thermal evolution under a held ambient input."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, expm1, isfinite, log


@dataclass(frozen=True, slots=True)
class ThermalInterval:
    """Latent physical die temperatures, not sensor-visible thermometer readings."""

    start_celsius: float
    end_celsius: float
    mean_celsius: float
####


def thermal_interval(
        *, start_celsius: float, ambient_celsius: float, dt_s: float, time_constant_s: float,
) -> ThermalInterval:
    """Solve dT/dt=(ambient-T)/tau and its interval average without cancellation."""
    if not all(isfinite(x) for x in (start_celsius, ambient_celsius, dt_s, time_constant_s)):
        raise ValueError("thermal inputs must be finite")
    ####
    if dt_s <= 0.0 or time_constant_s < 0.0:
        raise ValueError("dt_s must be positive and time_constant_s nonnegative")
    ####
    if time_constant_s == 0.0:
        return ThermalInterval(ambient_celsius, ambient_celsius, ambient_celsius)
    ####
    x = dt_s / time_constant_s
    end_fraction = -expm1(-x)
    if x < 1e-4:
        mean_fraction = x / 2.0 - x*x / 6.0 + x**3 / 24.0 - x**4 / 120.0
    else:
        mean_fraction = 1.0 - end_fraction / x
    ####
    difference = ambient_celsius - start_celsius
    end = start_celsius + difference * end_fraction
    mean = start_celsius + difference * mean_fraction
    if not isfinite(end) or not isfinite(mean):
        raise ValueError("thermal interval overflow")
    ####
    return ThermalInterval(start_celsius, end, mean)
####


def thermal_settling_time_s(
        *, initial_difference_celsius: float, tolerance_celsius: float, time_constant_s: float,
) -> float:
    """Time for a constant ambient step to reach a specified absolute die error."""
    if not all(isfinite(x) for x in (initial_difference_celsius, tolerance_celsius, time_constant_s)):
        raise ValueError("settling inputs must be finite")
    ####
    if tolerance_celsius <= 0.0 or time_constant_s < 0.0:
        raise ValueError("tolerance must be positive and time constant nonnegative")
    ####
    difference = abs(initial_difference_celsius)
    if difference <= tolerance_celsius or time_constant_s == 0.0:
        return 0.0
    ####
    return time_constant_s * (log(difference) - log(tolerance_celsius))
####


def remaining_thermal_error_celsius(
        *, initial_difference_celsius: float, elapsed_s: float, time_constant_s: float,
) -> float:
    """Absolute settling envelope for an unchanged ambient temperature."""
    if not all(isfinite(x) for x in (initial_difference_celsius, elapsed_s, time_constant_s)):
        raise ValueError("thermal envelope inputs must be finite")
    ####
    if elapsed_s < 0.0 or time_constant_s < 0.0:
        raise ValueError("elapsed and time constant must be nonnegative")
    ####
    if time_constant_s == 0.0:
        return 0.0
    ####
    return abs(initial_difference_celsius) * exp(-elapsed_s / time_constant_s)
####
