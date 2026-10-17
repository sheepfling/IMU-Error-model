"""Temperature-sweep calibration and model-mismatch experiments."""

from __future__ import annotations

from dataclasses import dataclass

from numpy import asarray, column_stack, diag, ndarray, ones, zeros
from numpy.linalg import lstsq, solve

from imu_error_model import AxisConfig, ImuConfig

from .known_reference import (
    CalibrationDataset,
    ReferenceSegment,
    expected_intrinsic_vector,
)


@dataclass(frozen=True, slots=True)
class ThermalAxisEstimate:
    """Affine temperature-dependent forward model y=(M0+dT*MT)x+b0+dT*bT."""

    base_matrix: ndarray
    thermal_matrix: ndarray
    base_bias: ndarray
    thermal_bias: ndarray
    reference_temperature_celsius: float
####


def fit_thermal_axis(
        ideal_input: ndarray,
        measured_output: ndarray,
        temperatures_celsius: ndarray,
        *,
        reference_temperature_celsius: float,
) -> ThermalAxisEstimate:
    """Fit a directly identifiable linear temperature-dependent sensor map."""
    ideal = asarray(ideal_input, dtype=float)
    measured = asarray(measured_output, dtype=float)
    temperatures = asarray(temperatures_celsius, dtype=float).reshape(-1)
    if ideal.shape != measured.shape or ideal.ndim != 2 or ideal.shape[1] != 3 or temperatures.shape != (ideal.shape[0],):
        raise ValueError("thermal fit arrays have incompatible shapes")
    ####
    delta_temperature = temperatures - reference_temperature_celsius
    design = column_stack(
        (
            ideal,
            ideal * delta_temperature[:, None],
            ones(ideal.shape[0]),
            delta_temperature,
        )
    )
    coefficients, _, rank, _ = lstsq(design, measured, rcond=None)
    if rank < 8:
        raise ValueError("thermal reference program is rank deficient")
    ####
    return ThermalAxisEstimate(
        base_matrix=coefficients[:3, :].T,
        thermal_matrix=coefficients[3:6, :].T,
        base_bias=coefficients[6, :],
        thermal_bias=coefficients[7, :],
        reference_temperature_celsius=reference_temperature_celsius,
    )
####


def thermal_reference_program(
        *,
        temperatures_celsius: tuple[float, ...] = (-20.0, 0.0, 25.0, 50.0, 70.0),
        magnitude: float = 9.80665,
        sample_count: int = 20,
        sample_period_s: float = 0.01,
) -> tuple[ReferenceSegment, ...]:
    """Generate full-rank static ±axis references at several temperatures."""
    axes = asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    segments: list[ReferenceSegment] = []
    for temperature in temperatures_celsius:
        for axis_index, axis_name in enumerate(("x", "y", "z")):
            for sign, sign_name in ((1.0, "p"), (-1.0, "a")):
                segments.append(
                    ReferenceSegment(
                        label=f"t{temperature:g}_{axis_name}_{sign_name}",
                        duration_s=sample_count * sample_period_s,
                        specific_force_body=sign * magnitude * axes[axis_index],
                        angular_rate_body=zeros(3),
                        temperature_celsius=temperature,
                        sample_count=sample_count,
                    )
                )
            ####
        ####
    ####
    return tuple(segments)
####


def fit_accelerometer_thermal_model(dataset: CalibrationDataset, config: AxisConfig) -> ThermalAxisEstimate:
    """Fit accelerometer thermal behavior from a synthetic reference sweep."""
    return fit_thermal_axis(
        dataset.specific_force_body,
        dataset.measured_acceleration,
        dataset.temperatures_celsius,
        reference_temperature_celsius=config.reference_temperature_celsius,
    )
####


def expected_thermal_axis(config: AxisConfig) -> ThermalAxisEstimate:
    """Return the exact linear thermal coefficients implied by ``AxisConfig``."""
    scale = asarray(config.scale_factor, dtype=float)
    scale_vector = ones(3) * float(scale) if scale.ndim == 0 else scale
    thermal_scale = asarray(config.thermal_scale_factor_coefficient, dtype=float)
    thermal_scale_vector = ones(3) * float(thermal_scale) if thermal_scale.ndim == 0 else thermal_scale
    sensitivity = asarray(config.sensitivity_matrix, dtype=float) if config.sensitivity_matrix is not None else diag(ones(3))
    base_matrix = sensitivity @ diag(1.0 + scale_vector)
    thermal_sensitivity = (
        zeros((3, 3))
        if config.thermal_sensitivity_matrix_coefficient is None
        else asarray(config.thermal_sensitivity_matrix_coefficient, dtype=float)
    )
    thermal_matrix = sensitivity @ diag(thermal_scale_vector) + thermal_sensitivity
    return ThermalAxisEstimate(
        base_matrix=base_matrix,
        thermal_matrix=thermal_matrix,
        base_bias=expected_intrinsic_vector(config.fixed_bias),
        thermal_bias=expected_intrinsic_vector(config.thermal_bias_coefficient),
        reference_temperature_celsius=config.reference_temperature_celsius,
    )
####


def correct_thermal_axis(
        measured: ndarray,
        temperature_celsius: float,
        estimate: ThermalAxisEstimate,
) -> ndarray:
    """Invert a fitted thermal axis model at one temperature."""
    delta_temperature = temperature_celsius - estimate.reference_temperature_celsius
    matrix = estimate.base_matrix + delta_temperature * estimate.thermal_matrix
    bias = estimate.base_bias + delta_temperature * estimate.thermal_bias
    return solve(matrix, asarray(measured, dtype=float) - bias)
####


def fit_gyroscope_specific_force_thermal_model(
        dataset: CalibrationDataset,
        config: ImuConfig,
) -> ThermalAxisEstimate:
    """Fit static gyro g-sensitivity, gyro bias, and their linear thermal drift."""
    return fit_thermal_axis(
        dataset.specific_force_body,
        dataset.measured_angular_rate,
        dataset.temperatures_celsius,
        reference_temperature_celsius=config.gyroscope.reference_temperature_celsius,
    )
####


def expected_gyroscope_specific_force_thermal(config: ImuConfig) -> ThermalAxisEstimate:
    """Return exact static gyro force-sensitivity thermal coefficients from configuration."""
    base_matrix = (
        zeros((3, 3))
        if config.gyroscope_specific_force_sensitivity is None
        else asarray(config.gyroscope_specific_force_sensitivity, dtype=float)
    )
    thermal_matrix = (
        zeros((3, 3))
        if config.gyroscope_specific_force_sensitivity_temperature_coefficient is None
        else asarray(config.gyroscope_specific_force_sensitivity_temperature_coefficient, dtype=float)
    )
    return ThermalAxisEstimate(
        base_matrix=base_matrix,
        thermal_matrix=thermal_matrix,
        base_bias=expected_intrinsic_vector(config.gyroscope.fixed_bias),
        thermal_bias=expected_intrinsic_vector(config.gyroscope.thermal_bias_coefficient),
        reference_temperature_celsius=config.gyroscope.reference_temperature_celsius,
    )
####
