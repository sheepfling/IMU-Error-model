"""First-order heading covariance from finite data and calibration uncertainty."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite, pi, sqrt
from typing import Literal

from numpy import asarray, concatenate, isfinite as array_isfinite, ndarray, zeros

from .gyrocompass import apply_calibration, static_gyrocompass, wrap_angle_rad
from .known_reference import (
    CalibrationEstimate,
    calibration_coefficient_vectors,
    calibration_from_coefficient_vectors,
)
from .time_series import estimate_mean_covariance


NORMAL_95_QUANTILE = 1.959963984540054


def _readonly_array(value: ndarray, shape: tuple[int, ...], field_name: str) -> ndarray:
    array = asarray(value, dtype=float).copy()
    if array.shape != shape or not bool(array_isfinite(array).all()):
        raise ValueError(f"{field_name} must be finite with shape {shape}")
    ####
    array.setflags(write=False)
    return array
####

def estimate_mean_covariance_from_covariance(value: ndarray) -> ndarray:
    """Validate and symmetrize an externally supplied positive-semidefinite covariance."""
    from numpy.linalg import eigh

    matrix = asarray(value, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.size == 0 or not bool(array_isfinite(matrix).all()):
        raise ValueError("covariance must be a nonempty finite square matrix")
    ####
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = eigh(symmetric)
    tolerance = 2.3e-14 * max(float(abs(eigenvalues).max()), 1e-24)
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError("covariance must be positive semidefinite; negative variance is not repairable")
    ####
    projected = (eigenvectors * eigenvalues.clip(min=0.0)) @ eigenvectors.T
    return 0.5 * (projected + projected.T)
####



def long_run_mean_covariance(samples: ndarray, *, max_lag: int | None = None) -> ndarray:
    """Return the canonical Bartlett/HAC covariance estimate of a sample mean."""
    values = asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or not bool(array_isfinite(values).all()):
        raise ValueError("samples must be finite with shape (n, p) and n >= 2")
    ####
    active_lag = (
        min(values.shape[0] - 1, max(0, round(values.shape[0] ** (1.0 / 3.0))))
        if max_lag is None
        else max_lag
    )
    return estimate_mean_covariance(values, max_lag=active_lag).covariance
####


@dataclass(frozen=True, slots=True)
class HeadingUncertaintyEstimate:
    """First-order variance decomposition for one calibrated heading estimate."""

    measurement_variance_rad2: float
    accelerometer_calibration_variance_rad2: float
    gyroscope_calibration_variance_rad2: float
    calibration_cross_covariance_contribution_rad2: float
    total_variance_rad2: float
    measurement_jacobian: ndarray
    accelerometer_calibration_jacobian: ndarray
    gyroscope_calibration_jacobian: ndarray

    def __post_init__(self) -> None:
        for field_name in (
                "measurement_variance_rad2",
                "accelerometer_calibration_variance_rad2",
                "gyroscope_calibration_variance_rad2",
                "total_variance_rad2",
        ):
            value = getattr(self, field_name)
            if not isfinite(value) or value < -1e-18:
                raise ValueError(f"{field_name} must be finite and non-negative")
            ####
            object.__setattr__(self, field_name, max(0.0, float(value)))
        ####
        if not isfinite(self.calibration_cross_covariance_contribution_rad2):
            raise ValueError("calibration_cross_covariance_contribution_rad2 must be finite")
        ####
        object.__setattr__(
            self,
            "measurement_jacobian",
            _readonly_array(self.measurement_jacobian, (6,), "measurement_jacobian"),
        )
        object.__setattr__(
            self,
            "accelerometer_calibration_jacobian",
            _readonly_array(
                self.accelerometer_calibration_jacobian,
                (12,),
                "accelerometer_calibration_jacobian",
            ),
        )
        object.__setattr__(
            self,
            "gyroscope_calibration_jacobian",
            _readonly_array(
                self.gyroscope_calibration_jacobian,
                (21,),
                "gyroscope_calibration_jacobian",
            ),
        )
    ####

    @property
    def standard_deviation_rad(self) -> float:
        return sqrt(self.total_variance_rad2)
    ####

    @property
    def standard_deviation_deg(self) -> float:
        return self.standard_deviation_rad * 180.0 / pi
    ####

    @property
    def confidence_95_half_width_deg(self) -> float:
        return NORMAL_95_QUANTILE * self.standard_deviation_deg
    ####

    @property
    def calibration_variance_rad2(self) -> float:
        """Return the joint calibration contribution including cross covariance."""
        return (
            self.accelerometer_calibration_variance_rad2
            + self.gyroscope_calibration_variance_rad2
            + self.calibration_cross_covariance_contribution_rad2
        )
    ####

    @property
    def measurement_fraction(self) -> float:
        return 0.0 if self.total_variance_rad2 <= 0.0 else self.measurement_variance_rad2 / self.total_variance_rad2
    ####

    @property
    def calibration_fraction(self) -> float:
        return 0.0 if self.total_variance_rad2 <= 0.0 else self.calibration_variance_rad2 / self.total_variance_rad2
    ####
####


def _wrapped_central_difference(
        values: ndarray,
        steps: ndarray,
        evaluator: Callable[[ndarray], float],
) -> ndarray:
    """Return a wrapped central-difference Jacobian for an angular output."""
    jacobian = zeros(values.size)
    for index in range(values.size):
        step = float(steps[index])
        plus = values.copy()
        minus = values.copy()
        plus[index] += step
        minus[index] -= step
        jacobian[index] = wrap_angle_rad(float(evaluator(plus)) - float(evaluator(minus))) / (2.0 * step)
    ####
    return jacobian
####


def _relative_steps(values: ndarray, floors: ndarray, relative: float = 1e-6) -> ndarray:
    return asarray([max(abs(float(value)) * relative, float(floor)) for value, floor in zip(values, floors)], dtype=float)
####


def propagate_heading_uncertainty(
        measured_acceleration_mean: ndarray,
        measured_angular_rate_mean: ndarray,
        measurement_mean_covariance: ndarray,
        calibration: CalibrationEstimate,
        *,
        latitude_rad: float,
        earth_rotation_radps: float,
        gravity_mps2: float,
        accelerometer_convention: Literal["apparent_gravity", "specific_force"] = "apparent_gravity",
) -> HeadingUncertaintyEstimate:
    """Propagate independent measurement-mean and calibration-fit covariance.

    Calibration and north-finding records are assumed independent. Joint
    accelerometer/gyroscope calibration covariance is retained, including its
    signed cross contribution. Colored calibration residuals require a block or
    otherwise serial-correlation-aware covariance estimator.
    """
    raw = concatenate(
        (
            asarray(measured_acceleration_mean, dtype=float),
            asarray(measured_angular_rate_mean, dtype=float),
        )
    )
    if raw.shape != (6,) or not bool(array_isfinite(raw).all()):
        raise ValueError("measured means must be finite 3-vectors")
    ####
    measurement_covariance = asarray(measurement_mean_covariance, dtype=float)
    if (
            measurement_covariance.shape != (6, 6)
            or not bool(array_isfinite(measurement_covariance).all())
            or not bool((abs(measurement_covariance - measurement_covariance.T) <= 1e-10).all())
    ):
        raise ValueError("measurement_mean_covariance must be a finite symmetric 6x6 matrix")
    ####
    measurement_covariance = estimate_mean_covariance_from_covariance(measurement_covariance)
    accel_vector, gyro_vector = calibration_coefficient_vectors(calibration)

    def heading_from(raw_values: ndarray, accel_values: ndarray, gyro_values: ndarray) -> float:
        active_calibration = calibration_from_coefficient_vectors(
            calibration,
            accel_values,
            gyro_values,
        )
        calibrated_accel, calibrated_gyro = apply_calibration(
            raw_values[:3],
            raw_values[3:],
            active_calibration,
        )
        return static_gyrocompass(
            calibrated_accel,
            calibrated_gyro,
            latitude_rad=latitude_rad,
            earth_rotation_radps=earth_rotation_radps,
            gravity_mps2=gravity_mps2,
            accelerometer_convention=accelerometer_convention,
        ).heading_rad
    ####

    raw_floors = asarray((1e-6, 1e-6, 1e-6, 1e-10, 1e-10, 1e-10), dtype=float)
    accel_floors = asarray(tuple(1e-6 if index % 4 < 3 else 1e-7 for index in range(12)), dtype=float)
    gyro_floors = asarray(
        tuple(
            1e-6 if index % 7 < 3 else 1e-10
            for index in range(21)
        ),
        dtype=float,
    )
    measurement_jacobian = _wrapped_central_difference(
        raw,
        _relative_steps(raw, raw_floors),
        lambda candidate: heading_from(candidate, accel_vector, gyro_vector),
    )
    accel_jacobian = _wrapped_central_difference(
        accel_vector,
        _relative_steps(accel_vector, accel_floors),
        lambda candidate: heading_from(raw, candidate, gyro_vector),
    )
    gyro_jacobian = _wrapped_central_difference(
        gyro_vector,
        _relative_steps(gyro_vector, gyro_floors),
        lambda candidate: heading_from(raw, accel_vector, candidate),
    )

    measurement_variance = float(measurement_jacobian @ measurement_covariance @ measurement_jacobian)
    accel_variance = float(
        accel_jacobian
        @ calibration.accelerometer_coefficient_covariance
        @ accel_jacobian
    )
    gyro_variance = float(
        gyro_jacobian
        @ calibration.gyroscope_coefficient_covariance
        @ gyro_jacobian
    )
    cross_covariance = calibration.accelerometer_gyroscope_coefficient_cross_covariance
    cross_contribution = (
        0.0
        if cross_covariance is None
        else float(2.0 * accel_jacobian @ cross_covariance @ gyro_jacobian)
    )
    total_variance = measurement_variance + accel_variance + gyro_variance + cross_contribution
    return HeadingUncertaintyEstimate(
        measurement_variance_rad2=measurement_variance,
        accelerometer_calibration_variance_rad2=accel_variance,
        gyroscope_calibration_variance_rad2=gyro_variance,
        calibration_cross_covariance_contribution_rad2=cross_contribution,
        total_variance_rad2=total_variance,
        measurement_jacobian=measurement_jacobian,
        accelerometer_calibration_jacobian=accel_jacobian,
        gyroscope_calibration_jacobian=gyro_jacobian,
    )
####
