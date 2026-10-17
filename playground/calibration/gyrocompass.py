"""Static gyrocompass playground using calibrated IMU measurements."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import atan2, cos, pi, sin
from typing import Literal

from numpy import asarray, cross, isfinite, ndarray, zeros
from numpy.linalg import norm, solve

from imu_error_model import ImuModel

from .earth_reference import LocalEarthReference, WGS84_EARTH_ROTATION_RADPS
from .known_reference import CalibrationEstimate


EARTH_ROTATION_RADPS = WGS84_EARTH_ROTATION_RADPS
STANDARD_GRAVITY_MPS2 = 9.80665
MotionExcitation = Callable[[float], tuple[ndarray, ndarray]]


def _rotation_z(angle_rad: float) -> ndarray:
    cosine = cos(angle_rad)
    sine = sin(angle_rad)
    return asarray(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)), dtype=float)
####


def _readonly_samples(value: ndarray, field_name: str) -> ndarray:
    samples = asarray(value, dtype=float).copy()
    if samples.ndim != 2 or samples.shape[1] != 3 or not bool(isfinite(samples).all()):
        raise ValueError(f"{field_name} must be finite with shape (n, 3)")
    ####
    samples.setflags(write=False)
    return samples
####


def _solve_rows(matrix: ndarray, values: ndarray) -> ndarray:
    """Solve a sensor matrix for one vector or a row-major sample matrix."""
    return solve(matrix, values) if values.ndim == 1 else solve(matrix, values.T).T
####


def _apply_rows(matrix: ndarray, values: ndarray) -> ndarray:
    """Apply a sensor matrix to one vector or a row-major sample matrix."""
    return matrix @ values if values.ndim == 1 else values @ matrix.T
####


def wrap_angle_rad(angle_rad: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle_rad + pi) % (2.0 * pi) - pi
####


@dataclass(frozen=True, slots=True)
class ImuMeasurementBatch:
    """Uniformly sampled raw accelerometer and gyroscope rate measurements."""

    acceleration: ndarray
    angular_rate: ndarray
    sample_period_s: float
    start_time: float = 0.0

    def __post_init__(self) -> None:
        acceleration = _readonly_samples(self.acceleration, "acceleration")
        angular_rate = _readonly_samples(self.angular_rate, "angular_rate")
        if acceleration.shape != angular_rate.shape:
            raise ValueError("acceleration and angular_rate must have the same shape")
        ####
        if acceleration.shape[0] == 0:
            raise ValueError("measurement batch must contain at least one sample")
        ####
        if not isfinite(self.sample_period_s) or self.sample_period_s <= 0.0:
            raise ValueError("sample_period_s must be finite and positive")
        ####
        if not isfinite(self.start_time):
            raise ValueError("start_time must be finite")
        ####
        object.__setattr__(self, "acceleration", acceleration)
        object.__setattr__(self, "angular_rate", angular_rate)
    ####

    @property
    def sample_count(self) -> int:
        return int(self.acceleration.shape[0])
    ####

    @property
    def duration_s(self) -> float:
        return self.sample_count * self.sample_period_s
    ####
####


@dataclass(frozen=True, slots=True)
class GyrocompassResult:
    """Static north-finding result and signal-quality diagnostics."""

    heading_rad: float
    attitude_nav_from_body: ndarray
    earth_rate_horizontal_radps: float
    earth_rate_residual_radps: float
    sample_count: int

    @property
    def heading_deg(self) -> float:
        return self.heading_rad * 180.0 / pi
    ####
####


def apply_calibration(
        measured_acceleration: ndarray,
        measured_angular_rate: ndarray,
        calibration: CalibrationEstimate,
) -> tuple[ndarray, ndarray]:
    """Invert the fitted deterministic model for one sample or ``(n, 3)`` rows."""
    accel = asarray(measured_acceleration, dtype=float)
    gyro = asarray(measured_angular_rate, dtype=float)
    if accel.shape != gyro.shape or accel.ndim not in {1, 2}:
        raise ValueError("measured acceleration and angular rate must share shape (3,) or (n, 3)")
    ####
    if accel.shape[-1] != 3 or not bool(isfinite(accel).all() and isfinite(gyro).all()):
        raise ValueError("measured acceleration and angular rate must be finite three-axis values")
    ####
    specific_force = _solve_rows(
        calibration.accelerometer_sensitivity_matrix,
        accel - calibration.accelerometer_bias,
    )
    force_sensitive_rate = _apply_rows(
        calibration.gyroscope_specific_force_sensitivity,
        specific_force,
    )
    angular_rate = _solve_rows(
        calibration.gyroscope_sensitivity_matrix,
        gyro - calibration.gyroscope_bias - force_sensitive_rate,
    )
    return specific_force, angular_rate
####


def _triad_attitude(
        primary_body: ndarray,
        secondary_body: ndarray,
        primary_nav: ndarray,
        secondary_nav: ndarray,
) -> ndarray:
    """Return the proper rotation mapping body vectors into navigation axes."""
    b1 = primary_body / norm(primary_body)
    b2_cross = cross(b1, secondary_body)
    if norm(b2_cross) <= 1e-15:
        raise ValueError("body reference vectors are collinear")
    ####
    b2 = b2_cross / norm(b2_cross)
    b3 = cross(b1, b2)

    n1 = primary_nav / norm(primary_nav)
    n2_cross = cross(n1, secondary_nav)
    if norm(n2_cross) <= 1e-15:
        raise ValueError("navigation reference vectors are collinear")
    ####
    n2 = n2_cross / norm(n2_cross)
    n3 = cross(n1, n2)
    body_basis = asarray((b1, b2, b3), dtype=float).T
    nav_basis = asarray((n1, n2, n3), dtype=float).T
    return nav_basis @ body_basis.T
####


def static_gyrocompass(
        acceleration_body: ndarray,
        angular_rate_body: ndarray,
        *,
        latitude_rad: float,
        earth_rotation_radps: float = EARTH_ROTATION_RADPS,
        gravity_mps2: float = STANDARD_GRAVITY_MPS2,
        accelerometer_convention: Literal["apparent_gravity", "specific_force"] = "apparent_gravity",
) -> GyrocompassResult:
    """Estimate attitude from apparent gravity and Earth rotation using TRIAD.

    Navigation axes use NED-like references: +x north, +y east, +z down.
    Legacy inputs default to apparent gravity (+down at rest). Physical sensor
    data must declare specific_force (-down at rest); its sign is changed only
    after intrinsic calibration. Motion admission is the caller's responsibility.
    """
    if not isfinite(latitude_rad) or abs(latitude_rad) >= pi / 2.0:
        raise ValueError("latitude_rad must be finite and strictly between the poles")
    ####
    if not isfinite(earth_rotation_radps) or earth_rotation_radps <= 0.0:
        raise ValueError("earth_rotation_radps must be finite and positive")
    ####
    if not isfinite(gravity_mps2) or gravity_mps2 <= 0.0:
        raise ValueError("gravity_mps2 must be finite and positive")
    ####
    accel = asarray(acceleration_body, dtype=float)
    gyro = asarray(angular_rate_body, dtype=float)
    if accel.shape != (3,) or gyro.shape != (3,) or not bool(isfinite(accel).all() and isfinite(gyro).all()):
        raise ValueError("acceleration_body and angular_rate_body must be finite 3-vectors")
    ####
    gravity_nav = asarray((0.0, 0.0, gravity_mps2), dtype=float)
    earth_rate_nav = earth_rotation_radps * asarray((cos(latitude_rad), 0.0, -sin(latitude_rad)), dtype=float)
    horizontal = abs(earth_rotation_radps * cos(latitude_rad))
    if horizontal <= 1e-12:
        raise ValueError("horizontal Earth-rate component is too small for gyrocompassing")
    ####
    if accelerometer_convention not in {"apparent_gravity", "specific_force"}:
        raise ValueError("unknown accelerometer convention")
    ####
    apparent_gravity = -accel if accelerometer_convention == "specific_force" else accel
    attitude = _triad_attitude(apparent_gravity, gyro, gravity_nav, earth_rate_nav)
    heading = wrap_angle_rad(atan2(float(attitude[1, 0]), float(attitude[0, 0])))
    expected_gyro = attitude.T @ earth_rate_nav
    residual = float(norm(gyro - expected_gyro))
    return GyrocompassResult(
        heading_rad=heading,
        attitude_nav_from_body=attitude,
        earth_rate_horizontal_radps=horizontal,
        earth_rate_residual_radps=residual,
        sample_count=1,
    )
####


def simulate_static_measurements(
        model: ImuModel,
        *,
        latitude_rad: float,
        heading_rad: float,
        sample_period_s: float = 0.01,
        sample_count: int = 6000,
        start_time: float = 0.0,
        height_m: float = 0.0,
        earth_rotation_radps: float = EARTH_ROTATION_RADPS,
        motion_excitation: MotionExcitation | None = None,
) -> ImuMeasurementBatch:
    """Generate raw stationary measurements, optionally with known motion contamination."""
    if not isfinite(sample_period_s) or sample_period_s <= 0.0:
        raise ValueError("sample_period_s must be finite and positive")
    ####
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    ####
    if not isfinite(start_time):
        raise ValueError("start_time must be finite")
    ####
    nav_from_body = _rotation_z(heading_rad)
    earth = LocalEarthReference(
        latitude_rad=latitude_rad,
        height_m=height_m,
        earth_rotation_radps=earth_rotation_radps,
    )
    gravity_body = nav_from_body.T @ earth.gravity_ned_mps2
    earth_rate_body = nav_from_body.T @ earth.earth_rotation_ned_radps
    model.reset()
    acceleration = zeros((sample_count, 3))
    angular_rate = zeros((sample_count, 3))
    current = start_time
    for sample_index in range(sample_count):
        midpoint = current + 0.5 * sample_period_s
        force_perturbation = zeros(3)
        rate_perturbation = zeros(3)
        if motion_excitation is not None:
            force_perturbation, rate_perturbation = motion_excitation(midpoint)
            force_perturbation = asarray(force_perturbation, dtype=float)
            rate_perturbation = asarray(rate_perturbation, dtype=float)
            if (
                    force_perturbation.shape != (3,)
                    or rate_perturbation.shape != (3,)
                    or not bool(isfinite(force_perturbation).all() and isfinite(rate_perturbation).all())
            ):
                raise ValueError("motion excitation must return finite 3-vectors")
            ####
        ####
        output = model.measure_ideal_rates(
            start_time=current,
            end_time=current + sample_period_s,
            specific_force_body=gravity_body + force_perturbation,
            angular_rate_body=earth_rate_body + rate_perturbation,
        )
        acceleration[sample_index] = output.acceleration
        angular_rate[sample_index] = output.angular_rate
        current += sample_period_s
    ####
    return ImuMeasurementBatch(
        acceleration=acceleration,
        angular_rate=angular_rate,
        sample_period_s=sample_period_s,
        start_time=start_time,
    )
####


def simulate_static_gyrocompass(
        model: ImuModel,
        calibration: CalibrationEstimate,
        *,
        latitude_rad: float,
        heading_rad: float,
        sample_period_s: float = 0.01,
        sample_count: int = 6000,
        start_time: float = 0.0,
        height_m: float = 0.0,
        earth_rotation_radps: float = EARTH_ROTATION_RADPS,
) -> GyrocompassResult:
    """Generate, calibrate, average, and solve one stationary north-finding run."""
    batch = simulate_static_measurements(
        model,
        latitude_rad=latitude_rad,
        heading_rad=heading_rad,
        sample_period_s=sample_period_s,
        sample_count=sample_count,
        start_time=start_time,
        height_m=height_m,
        earth_rotation_radps=earth_rotation_radps,
    )
    calibrated_accel, calibrated_gyro = apply_calibration(
        batch.acceleration,
        batch.angular_rate,
        calibration,
    )
    earth = LocalEarthReference(
        latitude_rad=latitude_rad,
        height_m=height_m,
        earth_rotation_radps=earth_rotation_radps,
    )
    result = static_gyrocompass(
        calibrated_accel.mean(axis=0),
        calibrated_gyro.mean(axis=0),
        latitude_rad=latitude_rad,
        earth_rotation_radps=earth_rotation_radps,
        gravity_mps2=float(earth.gravity_ned_mps2[2]),
    )
    return GyrocompassResult(
        heading_rad=result.heading_rad,
        attitude_nav_from_body=result.attitude_nav_from_body,
        earth_rate_horizontal_radps=result.earth_rate_horizontal_radps,
        earth_rate_residual_radps=result.earth_rate_residual_radps,
        sample_count=batch.sample_count,
    )
####
