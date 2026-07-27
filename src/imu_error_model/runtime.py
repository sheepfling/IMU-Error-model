"""Compiled, immutable NumPy parameters for the sampling hot path."""

from dataclasses import dataclass
from functools import lru_cache

from numpy import asarray, full, ndarray

from .config import AxisConfig, AxisValue


def _vector3(value: AxisValue) -> ndarray:
    values = asarray(value, dtype=float)
    result = full(3, float(values)) if values.ndim == 0 else values.copy()
    result.setflags(write=False)
    return result
####


def _readonly_matrix(value: tuple[tuple[float, float, float], ...] | ndarray | None) -> ndarray | None:
    if value is None:
        return None
    ####
    result = asarray(value, dtype=float).copy()
    result.setflags(write=False)
    return result
####


@dataclass(frozen=True, slots=True)
class CompiledAxisConfig:
    """Immutable NumPy representation of configuration-derived axis values."""

    scale_factor: ndarray
    turn_on_bias_std: ndarray
    bias_std: ndarray
    thermal_bias_coefficient: ndarray
    thermal_noise_coefficient: ndarray
    thermal_scale_factor_coefficient: ndarray
    noise_covariance: ndarray | None
    quantization_step: ndarray | None
    misalignment_covariance: ndarray | None
####


@lru_cache(maxsize=128)
def _compile_axis_config_cached(config_json: str) -> CompiledAxisConfig:
    config = AxisConfig.model_validate_json(config_json)
    noise_covariance = _readonly_matrix(config.noise_covariance)
    quantization_step = None if config.quantization_step is None else _vector3(config.quantization_step)
    return CompiledAxisConfig(
        scale_factor=_vector3(config.scale_factor),
        turn_on_bias_std=_vector3(config.turn_on_bias_std),
        bias_std=_vector3(config.bias_std),
        thermal_bias_coefficient=_vector3(config.thermal_bias_coefficient),
        thermal_noise_coefficient=_vector3(config.thermal_noise_coefficient),
        thermal_scale_factor_coefficient=_vector3(config.thermal_scale_factor_coefficient),
        noise_covariance=noise_covariance,
        quantization_step=quantization_step,
        misalignment_covariance=_readonly_matrix(config.misalignment_covariance),
    )
####


def compile_axis_config(config: AxisConfig) -> CompiledAxisConfig:
    """Compile immutable arrays once for an otherwise immutable configuration."""
    return _compile_axis_config_cached(config.model_dump_json())
####
