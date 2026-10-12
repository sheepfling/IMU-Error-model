"""Known-reference calibration playground for deterministic IMU intrinsics.

The first vertical slice intentionally uses explicit ideal body-frame rates.
It proves that the forward model can generate a full-rank calibration corpus
and that an independent least-squares recovery path can identify the injected
fixed matrices, biases, and gyro specific-force sensitivity.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

from numpy import (
    asarray,
    column_stack,
    concatenate,
    diag,
    eye,
    isfinite,
    isnan,
    kron,
    ndarray,
    ones,
    sqrt,
    zeros,
)
from numpy.linalg import eigvalsh, lstsq, pinv

from imu_error_model import AxisConfig, ImuConfig, ImuModel

from .uncertainty import linear_fit_diagnostics


STANDARD_GRAVITY_MPS2 = 9.80665


def _readonly_vector3(value: ndarray, field_name: str) -> ndarray:
    vector = asarray(value, dtype=float).copy()
    if vector.shape != (3,) or not bool(isfinite(vector).all()):
        raise ValueError(f"{field_name} must be a finite vector with shape (3,)")
    ####
    vector.setflags(write=False)
    return vector
####


def _readonly_matrix(value: ndarray, shape: tuple[int, int], field_name: str) -> ndarray:
    matrix = asarray(value, dtype=float).copy()
    if matrix.shape != shape or not bool(isfinite(matrix).all()):
        raise ValueError(f"{field_name} must be finite with shape {shape}")
    ####
    matrix.setflags(write=False)
    return matrix
####


def _readonly_sample_vector(
        value: ndarray, sample_count: int, field_name: str, *, allow_nan: bool = False
) -> ndarray:
    vector = asarray(value, dtype=float).reshape(-1).copy()
    valid = isfinite(vector) | isnan(vector) if allow_nan else isfinite(vector)
    if vector.shape != (sample_count,) or not bool(valid.all()):
        qualifier = "finite or NaN" if allow_nan else "finite"
        raise ValueError(f"{field_name} must be {qualifier} with shape ({sample_count},)")
    ####
    vector.setflags(write=False)
    return vector
####


@dataclass(frozen=True, slots=True)
class ReferenceSegment:
    """One constant-rate calibration interval with known ideal inputs."""

    label: str
    duration_s: float
    specific_force_body: ndarray
    angular_rate_body: ndarray
    temperature_celsius: float | None = None
    sample_count: int = 1

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("label must not be empty")
        ####
        if not isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("duration_s must be finite and positive")
        ####
        if self.temperature_celsius is not None and not isfinite(self.temperature_celsius):
            raise ValueError("temperature_celsius must be finite when supplied")
        ####
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int) or self.sample_count <= 0:
            raise ValueError("sample_count must be a positive integer")
        ####
        object.__setattr__(
            self,
            "specific_force_body",
            _readonly_vector3(self.specific_force_body, "specific_force_body"),
        )
        object.__setattr__(
            self,
            "angular_rate_body",
            _readonly_vector3(self.angular_rate_body, "angular_rate_body"),
        )
    ####
####


@dataclass(frozen=True, slots=True)
class CalibrationDataset:
    """Synthetic raw measurements paired with known calibration references."""

    labels: tuple[str, ...]
    start_times: ndarray
    end_times: ndarray
    temperatures_celsius: ndarray
    specific_force_body: ndarray
    angular_rate_body: ndarray
    measured_acceleration: ndarray
    measured_angular_rate: ndarray

    def __post_init__(self) -> None:
        sample_count = len(self.labels)
        object.__setattr__(
            self,
            "start_times",
            _readonly_sample_vector(self.start_times, sample_count, "start_times"),
        )
        object.__setattr__(
            self,
            "end_times",
            _readonly_sample_vector(self.end_times, sample_count, "end_times"),
        )
        object.__setattr__(
            self,
            "temperatures_celsius",
            _readonly_sample_vector(
                self.temperatures_celsius, sample_count, "temperatures_celsius", allow_nan=True
            ),
        )
        object.__setattr__(
            self,
            "specific_force_body",
            _readonly_matrix(self.specific_force_body, (sample_count, 3), "specific_force_body"),
        )
        object.__setattr__(
            self,
            "angular_rate_body",
            _readonly_matrix(self.angular_rate_body, (sample_count, 3), "angular_rate_body"),
        )
        object.__setattr__(
            self,
            "measured_acceleration",
            _readonly_matrix(self.measured_acceleration, (sample_count, 3), "measured_acceleration"),
        )
        object.__setattr__(
            self,
            "measured_angular_rate",
            _readonly_matrix(self.measured_angular_rate, (sample_count, 3), "measured_angular_rate"),
        )
    ####

    def to_csv(self, path: str | Path) -> None:
        """Write a lossless, calibration-tool-friendly CSV corpus."""
        destination = Path(path)
        fieldnames = (
            "label",
            "start_time_s",
            "end_time_s",
            "temperature_celsius",
            "ideal_specific_force_x_mps2",
            "ideal_specific_force_y_mps2",
            "ideal_specific_force_z_mps2",
            "ideal_angular_rate_x_radps",
            "ideal_angular_rate_y_radps",
            "ideal_angular_rate_z_radps",
            "measured_acceleration_x_mps2",
            "measured_acceleration_y_mps2",
            "measured_acceleration_z_mps2",
            "measured_angular_rate_x_radps",
            "measured_angular_rate_y_radps",
            "measured_angular_rate_z_radps",
        )
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for index, label in enumerate(self.labels):
                writer.writerow(
                    {
                        "label": label,
                        "start_time_s": self.start_times[index],
                        "end_time_s": self.end_times[index],
                        "temperature_celsius": (
                            ""
                            if isnan(self.temperatures_celsius[index])
                            else self.temperatures_celsius[index]
                        ),
                        "ideal_specific_force_x_mps2": self.specific_force_body[index, 0],
                        "ideal_specific_force_y_mps2": self.specific_force_body[index, 1],
                        "ideal_specific_force_z_mps2": self.specific_force_body[index, 2],
                        "ideal_angular_rate_x_radps": self.angular_rate_body[index, 0],
                        "ideal_angular_rate_y_radps": self.angular_rate_body[index, 1],
                        "ideal_angular_rate_z_radps": self.angular_rate_body[index, 2],
                        "measured_acceleration_x_mps2": self.measured_acceleration[index, 0],
                        "measured_acceleration_y_mps2": self.measured_acceleration[index, 1],
                        "measured_acceleration_z_mps2": self.measured_acceleration[index, 2],
                        "measured_angular_rate_x_radps": self.measured_angular_rate[index, 0],
                        "measured_angular_rate_y_radps": self.measured_angular_rate[index, 1],
                        "measured_angular_rate_z_radps": self.measured_angular_rate[index, 2],
                    }
                )
            ####
        ####
    ####
####


@dataclass(frozen=True, slots=True)
class CalibrationEstimate:
    """Recovered deterministic intrinsics and classical fit covariance.

    Coefficient covariance uses column-major vectorization of the underlying
    least-squares coefficient matrices. The accelerometer coefficient matrix
    has shape ``(4, 3)`` and the gyroscope coefficient matrix has shape
    ``(7, 3)``.
    """

    accelerometer_sensitivity_matrix: ndarray
    accelerometer_bias: ndarray
    gyroscope_sensitivity_matrix: ndarray
    gyroscope_specific_force_sensitivity: ndarray
    gyroscope_bias: ndarray
    accelerometer_design_rank: int
    gyroscope_design_rank: int
    accelerometer_condition_number: float
    gyroscope_condition_number: float
    accelerometer_residual_rms: float
    gyroscope_residual_rms: float
    accelerometer_residual_covariance: ndarray
    gyroscope_residual_covariance: ndarray
    accelerometer_coefficient_covariance: ndarray
    gyroscope_coefficient_covariance: ndarray
    accelerometer_gyroscope_coefficient_cross_covariance: ndarray | None = None
    covariance_assumption: str = "classical_joint_iid_residual_rows"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "accelerometer_sensitivity_matrix",
            _readonly_matrix(
                self.accelerometer_sensitivity_matrix,
                (3, 3),
                "accelerometer_sensitivity_matrix",
            ),
        )
        object.__setattr__(
            self,
            "accelerometer_bias",
            _readonly_vector3(self.accelerometer_bias, "accelerometer_bias"),
        )
        object.__setattr__(
            self,
            "gyroscope_sensitivity_matrix",
            _readonly_matrix(self.gyroscope_sensitivity_matrix, (3, 3), "gyroscope_sensitivity_matrix"),
        )
        object.__setattr__(
            self,
            "gyroscope_specific_force_sensitivity",
            _readonly_matrix(
                self.gyroscope_specific_force_sensitivity,
                (3, 3),
                "gyroscope_specific_force_sensitivity",
            ),
        )
        object.__setattr__(self, "gyroscope_bias", _readonly_vector3(self.gyroscope_bias, "gyroscope_bias"))
        object.__setattr__(
            self,
            "accelerometer_residual_covariance",
            _readonly_matrix(self.accelerometer_residual_covariance, (3, 3), "accelerometer_residual_covariance"),
        )
        object.__setattr__(
            self,
            "gyroscope_residual_covariance",
            _readonly_matrix(self.gyroscope_residual_covariance, (3, 3), "gyroscope_residual_covariance"),
        )
        object.__setattr__(
            self,
            "accelerometer_coefficient_covariance",
            _readonly_matrix(
                self.accelerometer_coefficient_covariance,
                (12, 12),
                "accelerometer_coefficient_covariance",
            ),
        )
        object.__setattr__(
            self,
            "gyroscope_coefficient_covariance",
            _readonly_matrix(
                self.gyroscope_coefficient_covariance,
                (21, 21),
                "gyroscope_coefficient_covariance",
            ),
        )
        cross_covariance = (
            zeros((12, 21))
            if self.accelerometer_gyroscope_coefficient_cross_covariance is None
            else self.accelerometer_gyroscope_coefficient_cross_covariance
        )
        object.__setattr__(
            self,
            "accelerometer_gyroscope_coefficient_cross_covariance",
            _readonly_matrix(
                cross_covariance,
                (12, 21),
                "accelerometer_gyroscope_coefficient_cross_covariance",
            ),
        )
        if eigvalsh(self.joint_coefficient_covariance).min() < -1e-12:
            raise ValueError("joint calibration coefficient covariance must be positive semidefinite")
        ####
        if not self.covariance_assumption:
            raise ValueError("covariance_assumption must not be empty")
        ####
    ####

    @property
    def accelerometer_coefficients(self) -> ndarray:
        """Return the ``(4, 3)`` affine coefficient matrix used by the fit."""
        coefficients = zeros((4, 3))
        coefficients[:3, :] = self.accelerometer_sensitivity_matrix.T
        coefficients[3, :] = self.accelerometer_bias
        return coefficients
    ####

    @property
    def gyroscope_coefficients(self) -> ndarray:
        """Return the ``(7, 3)`` affine coefficient matrix used by the fit."""
        coefficients = zeros((7, 3))
        coefficients[:3, :] = self.gyroscope_sensitivity_matrix.T
        coefficients[3:6, :] = self.gyroscope_specific_force_sensitivity.T
        coefficients[6, :] = self.gyroscope_bias
        return coefficients
    ####

    @property
    def joint_coefficient_covariance(self) -> ndarray:
        """Return the coherent ``(33, 33)`` covariance of both affine fits."""
        cross_covariance = self.accelerometer_gyroscope_coefficient_cross_covariance
        assert cross_covariance is not None
        covariance = zeros((33, 33))
        covariance[:12, :12] = self.accelerometer_coefficient_covariance
        covariance[12:, 12:] = self.gyroscope_coefficient_covariance
        covariance[:12, 12:] = cross_covariance
        covariance[12:, :12] = cross_covariance.T
        covariance.setflags(write=False)
        return covariance
    ####
####


def calibration_coefficient_vectors(estimate: CalibrationEstimate) -> tuple[ndarray, ndarray]:
    """Return column-major coefficient vectors consistent with stored covariance."""
    return (
        estimate.accelerometer_coefficients.reshape(-1, order="F"),
        estimate.gyroscope_coefficients.reshape(-1, order="F"),
    )
####


def calibration_from_coefficient_vectors(
        template: CalibrationEstimate,
        accelerometer_vector: ndarray,
        gyroscope_vector: ndarray,
) -> CalibrationEstimate:
    """Return a calibration estimate with perturbed affine coefficients."""
    accel = asarray(accelerometer_vector, dtype=float)
    gyro = asarray(gyroscope_vector, dtype=float)
    if accel.shape != (12,) or gyro.shape != (21,) or not bool(isfinite(accel).all() and isfinite(gyro).all()):
        raise ValueError("calibration coefficient vectors must be finite with shapes (12,) and (21,)")
    ####
    accel_coefficients = accel.reshape((4, 3), order="F")
    gyro_coefficients = gyro.reshape((7, 3), order="F")
    return replace(
        template,
        accelerometer_sensitivity_matrix=accel_coefficients[:3, :].T,
        accelerometer_bias=accel_coefficients[3, :],
        gyroscope_sensitivity_matrix=gyro_coefficients[:3, :].T,
        gyroscope_specific_force_sensitivity=gyro_coefficients[3:6, :].T,
        gyroscope_bias=gyro_coefficients[6, :],
    )
####


def ferraris_style_reference_program(
        *,
        gravity_mps2: float = STANDARD_GRAVITY_MPS2,
        angular_rate_radps: float = 1.0,
        duration_s: float = 1.0,
        temperature_celsius: float | None = None,
) -> tuple[ReferenceSegment, ...]:
    """Return six static poses plus symmetric known rotations about each axis.

    The six ``±g`` poses match the static portion of a Ferraris-style bench
    procedure. Symmetric ``±ω`` rotations add redundancy and make the gyro
    sensitivity matrix independently observable in this verification harness.
    """
    if not isfinite(gravity_mps2) or gravity_mps2 <= 0.0:
        raise ValueError("gravity_mps2 must be finite and positive")
    ####
    if not isfinite(angular_rate_radps) or angular_rate_radps <= 0.0:
        raise ValueError("angular_rate_radps must be finite and positive")
    ####
    if not isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    ####
    axes = eye(3)
    segments: list[ReferenceSegment] = [
        ReferenceSegment(
            label="zero_reference",
            duration_s=duration_s,
            specific_force_body=zeros(3),
            angular_rate_body=zeros(3),
            temperature_celsius=temperature_celsius,
        )
    ]
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        for sign, sign_name in ((1.0, "positive"), (-1.0, "negative")):
            segments.append(
                ReferenceSegment(
                    label=f"static_{sign_name}_{axis_name}",
                    duration_s=duration_s,
                    specific_force_body=sign * gravity_mps2 * axes[axis_index],
                    angular_rate_body=zeros(3),
                    temperature_celsius=temperature_celsius,
                )
            )
        ####
    ####
    turntable_specific_force = gravity_mps2 * axes[2]
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        for sign, sign_name in ((1.0, "positive"), (-1.0, "negative")):
            segments.append(
                ReferenceSegment(
                    label=f"rotation_{sign_name}_{axis_name}",
                    duration_s=duration_s,
                    specific_force_body=turntable_specific_force,
                    angular_rate_body=sign * angular_rate_radps * axes[axis_index],
                    temperature_celsius=temperature_celsius,
                )
            )
        ####
    ####
    return tuple(segments)
####


def simulate_reference_program(
        model: ImuModel,
        program: tuple[ReferenceSegment, ...],
        *,
        start_time: float = 0.0,
        reset_model: bool = True,
) -> CalibrationDataset:
    """Run a reference program through the stateful error model."""
    if not program:
        raise ValueError("program must contain at least one segment")
    ####
    if not isfinite(start_time):
        raise ValueError("start_time must be finite")
    ####
    if reset_model:
        model.reset()
    ####
    labels: list[str] = []
    starts: list[float] = []
    ends: list[float] = []
    temperatures: list[float] = []
    specific_forces: list[ndarray] = []
    angular_rates: list[ndarray] = []
    measured_accelerations: list[ndarray] = []
    measured_angular_rates: list[ndarray] = []
    current_time = start_time
    for segment in program:
        sample_duration = segment.duration_s / segment.sample_count
        for _ in range(segment.sample_count):
            end_time = current_time + sample_duration
            output = model.measure_ideal_rates(
                start_time=current_time,
                end_time=end_time,
                specific_force_body=segment.specific_force_body,
                angular_rate_body=segment.angular_rate_body,
                temperature_celsius=segment.temperature_celsius,
            )
            labels.append(segment.label)
            starts.append(current_time)
            ends.append(end_time)
            temperatures.append(
                float("nan") if segment.temperature_celsius is None else segment.temperature_celsius
            )
            specific_forces.append(segment.specific_force_body)
            angular_rates.append(segment.angular_rate_body)
            measured_accelerations.append(output.acceleration)
            measured_angular_rates.append(output.angular_rate)
            current_time = end_time
        ####
    ####
    sample_count = len(labels)
    return CalibrationDataset(
        labels=tuple(labels),
        start_times=asarray(starts, dtype=float).reshape(sample_count, 1),
        end_times=asarray(ends, dtype=float).reshape(sample_count, 1),
        temperatures_celsius=asarray(temperatures, dtype=float).reshape(sample_count, 1),
        specific_force_body=asarray(specific_forces, dtype=float),
        angular_rate_body=asarray(angular_rates, dtype=float),
        measured_acceleration=asarray(measured_accelerations, dtype=float),
        measured_angular_rate=asarray(measured_angular_rates, dtype=float),
    )
####


def _joint_classical_coefficient_covariance(
        accel_design: ndarray,
        gyro_design: ndarray,
        accel_residuals: ndarray,
        gyro_residuals: ndarray,
) -> ndarray:
    """Return coherent joint OLS covariance under iid six-axis residual rows."""
    sample_count = accel_design.shape[0]
    degrees_of_freedom = sample_count - max(accel_design.shape[1], gyro_design.shape[1])
    if degrees_of_freedom <= 0:
        return zeros((33, 33))
    ####
    residuals = concatenate((accel_residuals, gyro_residuals), axis=1)
    residual_covariance = residuals.T @ residuals / degrees_of_freedom
    accel_operator = pinv(accel_design)
    gyro_operator = pinv(gyro_design)
    accel_geometry = accel_operator @ accel_operator.T
    gyro_geometry = gyro_operator @ gyro_operator.T
    cross_geometry = accel_operator @ gyro_operator.T
    covariance = zeros((33, 33))
    covariance[:12, :12] = kron(residual_covariance[:3, :3], accel_geometry)
    covariance[12:, 12:] = kron(residual_covariance[3:, 3:], gyro_geometry)
    covariance[:12, 12:] = kron(residual_covariance[:3, 3:], cross_geometry)
    covariance[12:, :12] = covariance[:12, 12:].T
    return 0.5 * (covariance + covariance.T)
####


def fit_known_reference_calibration(dataset: CalibrationDataset) -> CalibrationEstimate:
    """Recover the affine deterministic model and classical fit covariance."""
    sample_count = len(dataset.labels)
    accel_design = column_stack((dataset.specific_force_body, ones(sample_count)))
    accel_coefficients, _, accel_rank, _ = lstsq(
        accel_design,
        dataset.measured_acceleration,
        rcond=None,
    )
    if accel_rank < 4:
        raise ValueError("accelerometer reference program is rank deficient")
    ####
    gyro_design = column_stack(
        (dataset.angular_rate_body, dataset.specific_force_body, ones(sample_count))
    )
    gyro_coefficients, _, gyro_rank, _ = lstsq(
        gyro_design,
        dataset.measured_angular_rate,
        rcond=None,
    )
    if gyro_rank < 7:
        raise ValueError("gyroscope reference program is rank deficient")
    ####
    accel_diagnostics = linear_fit_diagnostics(
        accel_design,
        dataset.measured_acceleration,
        accel_coefficients,
    )
    gyro_diagnostics = linear_fit_diagnostics(
        gyro_design,
        dataset.measured_angular_rate,
        gyro_coefficients,
    )
    joint_covariance = _joint_classical_coefficient_covariance(
        accel_design,
        gyro_design,
        dataset.measured_acceleration - accel_design @ accel_coefficients,
        dataset.measured_angular_rate - gyro_design @ gyro_coefficients,
    )
    return CalibrationEstimate(
        accelerometer_sensitivity_matrix=accel_coefficients[:3, :].T,
        accelerometer_bias=accel_coefficients[3, :],
        gyroscope_sensitivity_matrix=gyro_coefficients[:3, :].T,
        gyroscope_specific_force_sensitivity=gyro_coefficients[3:6, :].T,
        gyroscope_bias=gyro_coefficients[6, :],
        accelerometer_design_rank=int(accel_rank),
        gyroscope_design_rank=int(gyro_rank),
        accelerometer_condition_number=accel_diagnostics.condition_number,
        gyroscope_condition_number=gyro_diagnostics.condition_number,
        accelerometer_residual_rms=float(sqrt((accel_diagnostics.residual_rms ** 2).mean())),
        gyroscope_residual_rms=float(sqrt((gyro_diagnostics.residual_rms ** 2).mean())),
        accelerometer_residual_covariance=accel_diagnostics.residual_covariance,
        gyroscope_residual_covariance=gyro_diagnostics.residual_covariance,
        accelerometer_coefficient_covariance=joint_covariance[:12, :12],
        gyroscope_coefficient_covariance=joint_covariance[12:, 12:],
        accelerometer_gyroscope_coefficient_cross_covariance=joint_covariance[:12, 12:],
    )
####


def effective_sensitivity_matrix(config: AxisConfig) -> ndarray:
    """Return the full fixed linear map represented by an axis config."""
    scale = asarray(config.scale_factor, dtype=float)
    scale_vector = ones(3) * float(scale) if scale.ndim == 0 else scale
    diagonal = diag(1.0 + scale_vector)
    return diagonal if config.sensitivity_matrix is None else asarray(config.sensitivity_matrix) @ diagonal
####


def expected_intrinsic_vector(value: float | tuple[float, float, float]) -> ndarray:
    """Broadcast a configured scalar or return its three-axis vector."""
    vector = asarray(value, dtype=float)
    return ones(3) * float(vector) if vector.ndim == 0 else vector
####


def calibration_parameter_error(estimate: CalibrationEstimate, config: ImuConfig) -> ndarray:
    """Return one flattened truth-minus-estimate vector for regression reporting."""
    expected_gyro_force_sensitivity = (
        zeros((3, 3))
        if config.gyroscope_specific_force_sensitivity is None
        else asarray(config.gyroscope_specific_force_sensitivity, dtype=float)
    )
    return concatenate(
        (
            (estimate.accelerometer_sensitivity_matrix - effective_sensitivity_matrix(config.accelerometer)).ravel(),
            (estimate.accelerometer_bias - expected_intrinsic_vector(config.accelerometer.fixed_bias)).ravel(),
            (estimate.gyroscope_sensitivity_matrix - effective_sensitivity_matrix(config.gyroscope)).ravel(),
            (estimate.gyroscope_specific_force_sensitivity - expected_gyro_force_sensitivity).ravel(),
            (estimate.gyroscope_bias - expected_intrinsic_vector(config.gyroscope.fixed_bias)).ravel(),
        )
    )
####
