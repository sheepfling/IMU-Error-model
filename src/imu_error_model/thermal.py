from dataclasses import dataclass
from typing import Protocol

from numpy import any, asarray, ndarray

from .config import AxisConfig
from .runtime import compile_axis_config


@dataclass(frozen=True)
class ThermalState:
    """Temperature-dependent error terms for one configured sensor channel."""

    bias_offset: float | ndarray = 0.0
    noise_multiplier: float | ndarray = 1.0
    scale_factor_offset: float | ndarray = 0.0
####


class ThermalModel(Protocol):
    """Interface for a temperature model bound to one channel configuration."""

    def evaluate(self, temperature_celsius: float) -> ThermalState: ...
####

class LinearThermalModel:
    """Evaluate first-order temperature coefficients for one channel."""

    def __init__(self, config: AxisConfig) -> None:
        self._config = config
        compiled = compile_axis_config(config)
        self._bias_coefficient = compiled.thermal_bias_coefficient
        self._noise_coefficient = compiled.thermal_noise_coefficient
        self._scale_factor_coefficient = compiled.thermal_scale_factor_coefficient
    ####

    def evaluate(self, temperature_celsius: float) -> ThermalState:
        """Evaluate channel errors at a temperature expressed in degrees Celsius."""
        delta = temperature_celsius - self._config.reference_temperature_celsius
        multiplier = 1.0 + self._noise_coefficient * delta
        if any(asarray(multiplier) < 0):
            raise ValueError("thermal noise multiplier must be non-negative")
        ####
        return ThermalState(
            bias_offset=self._bias_coefficient * delta,
            noise_multiplier=multiplier,
            scale_factor_offset=self._scale_factor_coefficient * delta,
        )
    ####
####
