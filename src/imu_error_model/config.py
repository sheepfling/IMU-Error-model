from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class AxisConfig:
    """Noise and deterministic error parameters for one 3-axis channel."""

    white_noise_density: float = 0.0
    turn_on_bias_std: float = 0.0
    bias_std: float = 0.0
    bias_correlation_time: float | None = None
    flicker_bias_std: float = 0.0
    flicker_min_correlation_time: float | None = None
    flicker_max_correlation_time: float | None = None
    flicker_components: int = 8
    scale_factor: float = 0.0
    nonlinear_factor: float = 0.0
    misalignment_std: float = 0.0
    measurement_range: float | None = None
    thermal_bias_coefficient: float | tuple[float, float, float] = 0.0
    thermal_noise_coefficient: float | tuple[float, float, float] = 0.0
    thermal_scale_factor_coefficient: float | tuple[float, float, float] = 0.0
    reference_temperature: float = 25.0
    noise_covariance: tuple[tuple[float, float, float], ...] | None = None
    quantization_step: float | None = None
    apply_clipping: bool = True
    output_scale: float = 1.0

    def __post_init__(self) -> None:
        for name in ("white_noise_density", "turn_on_bias_std", "bias_std", "flicker_bias_std", "misalignment_std"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("thermal_bias_coefficient", "thermal_noise_coefficient", "thermal_scale_factor_coefficient"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape not in ((), (3,)) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite scalar or 3-vector")
        if self.bias_correlation_time is not None and self.bias_correlation_time <= 0:
            raise ValueError("bias_correlation_time must be positive or None")
        if self.flicker_bias_std > 0:
            if self.flicker_min_correlation_time is None or self.flicker_max_correlation_time is None:
                raise ValueError("flicker correlation-time bounds are required when flicker_bias_std is non-zero")
            if self.flicker_min_correlation_time <= 0 or self.flicker_max_correlation_time <= self.flicker_min_correlation_time:
                raise ValueError("flicker correlation-time bounds must be positive and increasing")
            if self.flicker_components < 2:
                raise ValueError("flicker_components must be at least two")
        if self.measurement_range is not None and self.measurement_range <= 0:
            raise ValueError("measurement_range must be positive or None")
        if self.quantization_step is not None and self.quantization_step <= 0:
            raise ValueError("quantization_step must be positive or None")
        if self.output_scale <= 0 or not np.isfinite(self.output_scale):
            raise ValueError("output_scale must be positive and finite")
        if self.noise_covariance is not None:
            covariance = np.asarray(self.noise_covariance, dtype=float)
            if covariance.shape != (3, 3) or not np.allclose(covariance, covariance.T):
                raise ValueError("noise_covariance must be a symmetric 3x3 matrix")
            if np.min(np.linalg.eigvalsh(covariance)) < -1e-12:
                raise ValueError("noise_covariance must be positive semidefinite")


@dataclass(frozen=True)
class ImuConfig:
    accelerometer: AxisConfig = field(default_factory=AxisConfig)
    gyroscope: AxisConfig = field(default_factory=AxisConfig)
    output_scale_accelerometer: float = 1.0
    output_scale_gyroscope: float = 1.0

    def __post_init__(self) -> None:
        if self.output_scale_accelerometer <= 0 or self.output_scale_gyroscope <= 0:
            raise ValueError("output scales must be positive")
