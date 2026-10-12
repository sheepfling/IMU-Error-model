"""High-rate inertial preintegration reference tools.

This module lives in the calibration playground rather than the runtime package.
It provides an independent SO(3) truth oracle and lower-fidelity aggregation
routes for testing coning/sculling sensitivity, sample-rate sufficiency, and
calibrated minor-increment processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, cos, isfinite, pi, sin
from typing import TYPE_CHECKING, Callable

import numpy as np
from numpy.typing import NDArray

from imu_error_model.continuous_acquisition import ContinuousTruthSample

if TYPE_CHECKING:
    from .known_reference import CalibrationEstimate
####


FloatArray = NDArray[np.float64]
ContinuousTruthCallable = Callable[[float], ContinuousTruthSample]


def _vector3(value: FloatArray | tuple[float, float, float], field_name: str) -> FloatArray:
    vector = np.asarray(value, dtype=float).copy()
    if vector.shape != (3,) or not bool(np.isfinite(vector).all()):
        raise ValueError(f"{field_name} must be a finite vector with shape (3,)")
    ####
    return vector
####


def _skew(vector: FloatArray) -> FloatArray:
    x, y, z = vector
    return np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=float)
####


def so3_exp(rotation_vector: FloatArray) -> FloatArray:
    """Return ``Exp([phi]x)`` with a stable small-angle series."""
    phi = _vector3(rotation_vector, "rotation_vector")
    angle = float(np.linalg.norm(phi))
    skew = _skew(phi)
    if angle < 1e-8:
        return np.eye(3) + skew + 0.5 * (skew @ skew)
    ####
    sine = sin(angle) / angle
    one_minus_cosine = (1.0 - cos(angle)) / (angle * angle)
    return np.eye(3) + sine * skew + one_minus_cosine * (skew @ skew)
####


def so3_log(rotation: FloatArray) -> FloatArray:
    """Return the principal rotation vector for a proper 3x3 rotation matrix."""
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not bool(np.isfinite(matrix).all()):
        raise ValueError("rotation must be a finite 3x3 matrix")
    ####
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = acos(cosine)
    vee = np.asarray(
        (
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ),
        dtype=float,
    )
    if angle < 1e-8:
        return 0.5 * vee
    ####
    if pi - angle < 1e-6:
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + np.eye(3)))
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        axis = axis / np.linalg.norm(axis)
        if np.dot(axis, vee) < 0.0:
            axis = -axis
        ####
        return axis * angle
    ####
    return (angle / (2.0 * sin(angle))) * vee
####


def _rotation_integral(angular_rate_body: FloatArray, dt_s: float) -> FloatArray:
    """Return ``integral_0^dt Exp([omega]x t) dt`` for constant body rate."""
    omega = _vector3(angular_rate_body, "angular_rate_body")
    if not isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    ####
    magnitude = float(np.linalg.norm(omega))
    skew = _skew(omega)
    if magnitude < 1e-10:
        return dt_s * np.eye(3) + 0.5 * dt_s**2 * skew + (dt_s**3 / 6.0) * (skew @ skew)
    ####
    angle = magnitude * dt_s
    return (
        dt_s * np.eye(3)
        + ((1.0 - cos(angle)) / magnitude**2) * skew
        + ((angle - sin(angle)) / magnitude**3) * (skew @ skew)
    )
####


@dataclass(frozen=True, slots=True)
class MinorImuIncrement:
    """Raw IMU increments over one high-rate subinterval.

    Both increments are expressed in the sensor/body axes associated with the
    subinterval. ``delta_v_body`` is the raw accelerometer component integral,
    not yet rotated into the major-interval start frame.
    """

    delta_theta_body: FloatArray
    delta_v_body: FloatArray
    dt_s: float

    def __post_init__(self) -> None:
        if not isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        ####
        object.__setattr__(self, "delta_theta_body", _vector3(self.delta_theta_body, "delta_theta_body"))
        object.__setattr__(self, "delta_v_body", _vector3(self.delta_v_body, "delta_v_body"))
    ####
####


@dataclass(frozen=True, slots=True)
class PreintegratedIncrement:
    """Major-interval inertial increment expressed in the start body frame."""

    delta_theta_start: FloatArray
    delta_v_start: FloatArray
    rotation_start_from_end: FloatArray
    duration_s: float
    minor_count: int
    method: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "delta_theta_start", _vector3(self.delta_theta_start, "delta_theta_start"))
        object.__setattr__(self, "delta_v_start", _vector3(self.delta_v_start, "delta_v_start"))
        rotation = np.asarray(self.rotation_start_from_end, dtype=float).copy()
        if rotation.shape != (3, 3) or not bool(np.isfinite(rotation).all()):
            raise ValueError("rotation_start_from_end must be a finite 3x3 matrix")
        ####
        object.__setattr__(self, "rotation_start_from_end", rotation)
        if not isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("duration_s must be finite and positive")
        ####
        if isinstance(self.minor_count, bool) or not isinstance(self.minor_count, int) or self.minor_count <= 0:
            raise ValueError("minor_count must be a positive integer")
        ####
        if not self.method:
            raise ValueError("method must not be empty")
        ####
    ####
####


def preintegrate_piecewise_constant(samples: tuple[MinorImuIncrement, ...]) -> PreintegratedIncrement:
    """Exactly preintegrate piecewise-constant body rate and specific force.

    Each minor interval is integrated analytically on SO(3). This is the
    reference aggregation for already discretized minor increments.
    """
    if not samples:
        raise ValueError("at least one minor increment is required")
    ####
    rotation = np.eye(3)
    delta_v = np.zeros(3)
    duration = 0.0
    for sample in samples:
        angular_rate = sample.delta_theta_body / sample.dt_s
        specific_force = sample.delta_v_body / sample.dt_s
        delta_v += rotation @ _rotation_integral(angular_rate, sample.dt_s) @ specific_force
        rotation = rotation @ so3_exp(sample.delta_theta_body)
        duration += sample.dt_s
    ####
    return PreintegratedIncrement(
        delta_theta_start=so3_log(rotation),
        delta_v_start=delta_v,
        rotation_start_from_end=rotation,
        duration_s=duration,
        minor_count=len(samples),
        method="piecewise_constant_so3_exact",
    )
####


def naive_major_interval(samples: tuple[MinorImuIncrement, ...]) -> PreintegratedIncrement:
    """Aggregate raw component increments without coning or sculling correction."""
    if not samples:
        raise ValueError("at least one minor increment is required")
    ####
    delta_theta = np.sum(np.stack([sample.delta_theta_body for sample in samples]), axis=0)
    delta_v = np.sum(np.stack([sample.delta_v_body for sample in samples]), axis=0)
    duration = float(sum(sample.dt_s for sample in samples))
    return PreintegratedIncrement(
        delta_theta_start=delta_theta,
        delta_v_start=delta_v,
        rotation_start_from_end=so3_exp(delta_theta),
        duration_s=duration,
        minor_count=len(samples),
        method="naive_component_sum",
    )
####


def recursive_coning_sculling(samples: tuple[MinorImuIncrement, ...]) -> PreintegratedIncrement:
    """Apply classical small-angle recursive coning/sculling corrections.

    The recursion follows the common Savage-style high-rate correction form:
    accumulated raw angle/velocity plus coning and sculling cross terms, with
    the first-order velocity-rotation term added at the major interval. It is
    intentionally retained as an approximation and is scored against the
    independent SO(3) truth oracle.
    """
    if not samples:
        raise ValueError("at least one minor increment is required")
    ####
    theta = np.zeros(3)
    velocity = np.zeros(3)
    coning = np.zeros(3)
    sculling = np.zeros(3)
    previous_theta = np.zeros(3)
    previous_velocity = np.zeros(3)
    duration = 0.0
    for sample in samples:
        delta_theta = sample.delta_theta_body
        delta_velocity = sample.delta_v_body
        coning += 0.5 * np.cross(theta + previous_theta / 6.0, delta_theta)
        sculling += 0.5 * (
            np.cross(theta + previous_theta / 6.0, delta_velocity)
            + np.cross(velocity + previous_velocity / 6.0, delta_theta)
        )
        theta += delta_theta
        velocity += delta_velocity
        previous_theta = delta_theta
        previous_velocity = delta_velocity
        duration += sample.dt_s
    ####
    corrected_theta = theta + coning
    corrected_velocity = velocity + sculling + 0.5 * np.cross(theta, velocity)
    return PreintegratedIncrement(
        delta_theta_start=corrected_theta,
        delta_v_start=corrected_velocity,
        rotation_start_from_end=so3_exp(corrected_theta),
        duration_s=duration,
        minor_count=len(samples),
        method="recursive_small_angle_coning_sculling",
    )
####


def sample_minor_increments(
        truth: ContinuousTruthCallable,
        *,
        start_time_s: float,
        end_time_s: float,
        minor_count: int,
        quadrature_substeps: int = 32,
) -> tuple[MinorImuIncrement, ...]:
    """Component-integrate ideal gyro/accelerometer outputs on uniform minors."""
    if not isfinite(start_time_s) or not isfinite(end_time_s) or end_time_s <= start_time_s:
        raise ValueError("truth interval must be finite and increasing")
    ####
    if isinstance(minor_count, bool) or not isinstance(minor_count, int) or minor_count <= 0:
        raise ValueError("minor_count must be a positive integer")
    ####
    if isinstance(quadrature_substeps, bool) or not isinstance(quadrature_substeps, int) or quadrature_substeps <= 0:
        raise ValueError("quadrature_substeps must be a positive integer")
    ####
    minor_dt = (end_time_s - start_time_s) / minor_count
    results: list[MinorImuIncrement] = []
    for minor_index in range(minor_count):
        minor_start = start_time_s + minor_index * minor_dt
        quadrature_dt = minor_dt / quadrature_substeps
        delta_theta = np.zeros(3)
        delta_v = np.zeros(3)
        for sub_index in range(quadrature_substeps):
            time_s = minor_start + (sub_index + 0.5) * quadrature_dt
            sample = truth(time_s)
            delta_theta += np.asarray(sample.angular_rate_body, dtype=float) * quadrature_dt
            delta_v += np.asarray(sample.specific_force_body, dtype=float) * quadrature_dt
        ####
        results.append(MinorImuIncrement(delta_theta, delta_v, minor_dt))
    ####
    return tuple(results)
####


def preintegrate_continuous_truth(
        truth: ContinuousTruthCallable,
        *,
        start_time_s: float,
        end_time_s: float,
        reference_substeps: int = 32768,
) -> PreintegratedIncrement:
    """High-rate independent truth oracle for a continuous body-frame signal."""
    samples = sample_minor_increments(
        truth,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        minor_count=reference_substeps,
        quadrature_substeps=1,
    )
    result = preintegrate_piecewise_constant(samples)
    return PreintegratedIncrement(
        delta_theta_start=result.delta_theta_start,
        delta_v_start=result.delta_v_start,
        rotation_start_from_end=result.rotation_start_from_end,
        duration_s=result.duration_s,
        minor_count=result.minor_count,
        method="continuous_midpoint_so3_oracle",
    )
####


@dataclass(frozen=True, slots=True)
class ConingTruth:
    """Circular two-axis body-rate excitation with zero component-wise mean."""

    angular_rate_amplitude_radps: float = 1.0
    frequency_hz: float = 5.0
    temperature_celsius: float = 25.0

    def __post_init__(self) -> None:
        if not isfinite(self.angular_rate_amplitude_radps) or self.angular_rate_amplitude_radps <= 0.0:
            raise ValueError("angular_rate_amplitude_radps must be finite and positive")
        ####
        if not isfinite(self.frequency_hz) or self.frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be finite and positive")
        ####
    ####

    @property
    def period_s(self) -> float:
        return 1.0 / self.frequency_hz
    ####

    def __call__(self, physical_time_s: float) -> ContinuousTruthSample:
        phase = 2.0 * pi * self.frequency_hz * physical_time_s
        amplitude = self.angular_rate_amplitude_radps
        return ContinuousTruthSample(
            specific_force_body=(0.0, 0.0, 0.0),
            angular_rate_body=(amplitude * cos(phase), amplitude * sin(phase), 0.0),
            ambient_temperature_celsius=self.temperature_celsius,
        )
    ####
####


@dataclass(frozen=True, slots=True)
class ScullingTruth:
    """Canonical orthogonal angular-rate/specific-force sculling excitation."""

    angle_amplitude_rad: float = 0.1
    force_amplitude_mps2: float = 2.0
    frequency_hz: float = 5.0
    temperature_celsius: float = 25.0

    def __post_init__(self) -> None:
        if not isfinite(self.angle_amplitude_rad) or self.angle_amplitude_rad <= 0.0:
            raise ValueError("angle_amplitude_rad must be finite and positive")
        ####
        if not isfinite(self.force_amplitude_mps2) or self.force_amplitude_mps2 <= 0.0:
            raise ValueError("force_amplitude_mps2 must be finite and positive")
        ####
        if not isfinite(self.frequency_hz) or self.frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be finite and positive")
        ####
    ####

    @property
    def period_s(self) -> float:
        return 1.0 / self.frequency_hz
    ####

    def __call__(self, physical_time_s: float) -> ContinuousTruthSample:
        omega = 2.0 * pi * self.frequency_hz
        phase = omega * physical_time_s
        return ContinuousTruthSample(
            specific_force_body=(0.0, self.force_amplitude_mps2 * sin(phase), 0.0),
            angular_rate_body=(self.angle_amplitude_rad * omega * cos(phase), 0.0, 0.0),
            ambient_temperature_celsius=self.temperature_celsius,
        )
    ####
####


@dataclass(frozen=True, slots=True)
class ConvergencePoint:
    """One sample-density comparison against a fixed high-rate truth oracle."""

    minor_count: int
    sample_rate_hz: float
    exact_piecewise_attitude_error_rad: float
    corrected_attitude_error_rad: float
    naive_attitude_error_rad: float
    exact_piecewise_velocity_error_mps: float
    corrected_velocity_error_mps: float
    naive_velocity_error_mps: float
####


def convergence_study(
        truth: ContinuousTruthCallable,
        *,
        start_time_s: float,
        end_time_s: float,
        minor_counts: tuple[int, ...],
        reference_substeps: int = 32768,
        quadrature_substeps: int = 32,
) -> tuple[ConvergencePoint, ...]:
    """Compare aggregation routes across minor-sample densities."""
    if not minor_counts or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in minor_counts):
        raise ValueError("minor_counts must contain positive integers")
    ####
    reference = preintegrate_continuous_truth(
        truth,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        reference_substeps=reference_substeps,
    )
    duration = end_time_s - start_time_s
    results: list[ConvergencePoint] = []
    for minor_count in minor_counts:
        samples = sample_minor_increments(
            truth,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
            minor_count=minor_count,
            quadrature_substeps=quadrature_substeps,
        )
        exact_piecewise = preintegrate_piecewise_constant(samples)
        corrected = recursive_coning_sculling(samples)
        naive = naive_major_interval(samples)
        results.append(
            ConvergencePoint(
                minor_count=minor_count,
                sample_rate_hz=minor_count / duration,
                exact_piecewise_attitude_error_rad=float(
                    np.linalg.norm(exact_piecewise.delta_theta_start - reference.delta_theta_start)
                ),
                corrected_attitude_error_rad=float(
                    np.linalg.norm(corrected.delta_theta_start - reference.delta_theta_start)
                ),
                naive_attitude_error_rad=float(np.linalg.norm(naive.delta_theta_start - reference.delta_theta_start)),
                exact_piecewise_velocity_error_mps=float(
                    np.linalg.norm(exact_piecewise.delta_v_start - reference.delta_v_start)
                ),
                corrected_velocity_error_mps=float(
                    np.linalg.norm(corrected.delta_v_start - reference.delta_v_start)
                ),
                naive_velocity_error_mps=float(np.linalg.norm(naive.delta_v_start - reference.delta_v_start)),
            )
        )
    ####
    return tuple(results)
####


def first_rate_meeting_error(
        points: tuple[ConvergencePoint, ...],
        *,
        attitude_error_rad: float | None = None,
        velocity_error_mps: float | None = None,
        method: str = "exact_piecewise",
) -> float | None:
    """Return the first tested sample rate meeting all supplied error targets."""
    if attitude_error_rad is None and velocity_error_mps is None:
        raise ValueError("at least one error target is required")
    ####
    if attitude_error_rad is not None and (not isfinite(attitude_error_rad) or attitude_error_rad < 0.0):
        raise ValueError("attitude_error_rad must be finite and non-negative")
    ####
    if velocity_error_mps is not None and (not isfinite(velocity_error_mps) or velocity_error_mps < 0.0):
        raise ValueError("velocity_error_mps must be finite and non-negative")
    ####
    valid_methods = {"exact_piecewise", "corrected", "naive"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {sorted(valid_methods)}")
    ####
    for point in sorted(points, key=lambda item: item.sample_rate_hz):
        attitude_error = getattr(point, f"{method}_attitude_error_rad")
        velocity_error = getattr(point, f"{method}_velocity_error_mps")
        if attitude_error_rad is not None and attitude_error > attitude_error_rad:
            continue
        ####
        if velocity_error_mps is not None and velocity_error > velocity_error_mps:
            continue
        ####
        return point.sample_rate_hz
    ####
    return None
####


def apply_calibration_to_minor_increments(
        samples: tuple[MinorImuIncrement, ...],
        calibration: "CalibrationEstimate",
) -> tuple[MinorImuIncrement, ...]:
    """Apply deterministic calibration to raw minor increments before preintegration."""
    if not samples:
        raise ValueError("at least one minor increment is required")
    ####
    from .gyrocompass import apply_calibration

    corrected: list[MinorImuIncrement] = []
    for sample in samples:
        measured_acceleration = sample.delta_v_body / sample.dt_s
        measured_angular_rate = sample.delta_theta_body / sample.dt_s
        specific_force, angular_rate = apply_calibration(
            measured_acceleration,
            measured_angular_rate,
            calibration,
        )
        corrected.append(
            MinorImuIncrement(
                delta_theta_body=np.asarray(angular_rate, dtype=float) * sample.dt_s,
                delta_v_body=np.asarray(specific_force, dtype=float) * sample.dt_s,
                dt_s=sample.dt_s,
            )
        )
    ####
    return tuple(corrected)
####
