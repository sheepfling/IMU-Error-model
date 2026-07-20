from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import AxisConfig


@dataclass(frozen=True)
class ThermalState:
    bias_offset: float | np.ndarray = 0.0
    noise_multiplier: float | np.ndarray = 1.0
    scale_factor_offset: float | np.ndarray = 0.0


class ThermalModel(Protocol):
    def evaluate(self, config: AxisConfig, temperature: float) -> ThermalState: ...


class LinearThermalModel:
    """Default first-order temperature model using coefficients in AxisConfig."""

    def evaluate(self, config: AxisConfig, temperature: float) -> ThermalState:
        delta = temperature - config.reference_temperature
        multiplier = 1.0 + np.asarray(config.thermal_noise_coefficient) * delta
        if np.any(np.asarray(multiplier) < 0):
            raise ValueError("thermal noise multiplier must be non-negative")
        return ThermalState(
            bias_offset=np.asarray(config.thermal_bias_coefficient) * delta,
            noise_multiplier=multiplier,
            scale_factor_offset=np.asarray(config.thermal_scale_factor_coefficient) * delta,
        )
