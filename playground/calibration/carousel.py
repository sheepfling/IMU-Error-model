"""Continuous down-axis carouseling experiments for gyrocompass modulation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite, pi

from numpy import asarray, isfinite as array_isfinite, ndarray, percentile, random, sqrt, zeros

from imu_error_model import ImuModel

from .data_confidence import HeadingConfidencePoint, HeadingConfidenceSweep
from .earth_reference import LocalEarthReference, WGS84_EARTH_ROTATION_RADPS
from .gyrocompass import GyrocompassResult, apply_calibration, static_gyrocompass, wrap_angle_rad
from .known_reference import CalibrationEstimate
from .rotating_frame import average_sensor_from_base_rotation, demodulate_interval_average, rotation_z


def _readonly_vector3(value: ndarray, field_name: str) -> ndarray:
    vector = asarray(value, dtype=float).copy()
    if vector.shape != (3,) or not bool(array_isfinite(vector).all()):
        raise ValueError(f"{field_name} must be a finite 3-vector")
    ####
    vector.setflags(write=False)
    return vector
####


@dataclass(frozen=True, slots=True)
class CarouselGyrocompassResult:
    """Demodulated heading result and table-experiment metadata."""

    result: GyrocompassResult
    rotation_rate_radps: float
    duration_s: float
    revolution_count: float
    mean_acceleration_base: ndarray
    mean_angular_rate_base: ndarray

    def __post_init__(self) -> None:
        if not isfinite(self.rotation_rate_radps) or self.rotation_rate_radps == 0.0:
            raise ValueError("rotation_rate_radps must be finite and non-zero")
        ####
        if not isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("duration_s must be finite and positive")
        ####
        if not isfinite(self.revolution_count):
            raise ValueError("revolution_count must be finite")
        ####
        object.__setattr__(
            self,
            "mean_acceleration_base",
            _readonly_vector3(self.mean_acceleration_base, "mean_acceleration_base"),
        )
        object.__setattr__(
            self,
            "mean_angular_rate_base",
            _readonly_vector3(self.mean_angular_rate_base, "mean_angular_rate_base"),
        )
    ####
####


def simulate_carousel_gyrocompass(
        model: ImuModel,
        calibration: CalibrationEstimate,
        *,
        latitude_rad: float,
        heading_rad: float,
        rotation_rate_radps: float,
        duration_s: float,
        sample_period_s: float = 0.02,
        height_m: float = 0.0,
        earth_rotation_radps: float = WGS84_EARTH_ROTATION_RADPS,
        initial_table_angle_rad: float = 0.0,
) -> CarouselGyrocompassResult:
    """Simulate and demodulate one continuously rotating, level IMU.

    The gyro input contains both the known table rate and Earth rotation. The
    table rate is removed after deterministic calibration. Each measured sample
    is then mapped back to the nonrotating base frame using the exact
    interval-average aperture map. A sensor-fixed horizontal bias therefore
    rotates in the base frame and cancels over complete revolutions.
    """
    if not isfinite(rotation_rate_radps) or rotation_rate_radps == 0.0:
        raise ValueError("rotation_rate_radps must be finite and non-zero")
    ####
    if not isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    ####
    if not isfinite(sample_period_s) or sample_period_s <= 0.0:
        raise ValueError("sample_period_s must be finite and positive")
    ####
    if not isfinite(initial_table_angle_rad):
        raise ValueError("initial_table_angle_rad must be finite")
    ####
    sample_count = round(duration_s / sample_period_s)
    if sample_count <= 0:
        raise ValueError("duration_s must span at least one sample")
    ####
    actual_duration = sample_count * sample_period_s
    earth = LocalEarthReference(
        latitude_rad=latitude_rad,
        height_m=height_m,
        earth_rotation_radps=earth_rotation_radps,
    )
    nav_from_base = rotation_z(heading_rad)
    gravity_base = nav_from_base.T @ earth.gravity_ned_mps2
    earth_rate_base = nav_from_base.T @ earth.earth_rotation_ned_radps
    table_rate_sensor = asarray((0.0, 0.0, rotation_rate_radps), dtype=float)
    acceleration_sum = zeros(3)
    angular_rate_sum = zeros(3)
    model.reset()
    current = 0.0

    for _ in range(sample_count):
        start_angle = initial_table_angle_rad + rotation_rate_radps * current
        end_angle = start_angle + rotation_rate_radps * sample_period_s
        sensor_from_base_average = average_sensor_from_base_rotation(start_angle, end_angle)
        output = model.measure_ideal_rates(
            start_time=current,
            end_time=current + sample_period_s,
            specific_force_body=sensor_from_base_average @ gravity_base,
            angular_rate_body=sensor_from_base_average @ earth_rate_base + table_rate_sensor,
        )
        calibrated_accel, calibrated_gyro = apply_calibration(
            output.acceleration,
            output.angular_rate,
            calibration,
        )
        acceleration_sum += demodulate_interval_average(
            sensor_from_base_average,
            calibrated_accel,
        )
        angular_rate_sum += demodulate_interval_average(
            sensor_from_base_average,
            calibrated_gyro - table_rate_sensor,
        )
        current += sample_period_s
    ####

    mean_acceleration = acceleration_sum / sample_count
    mean_angular_rate = angular_rate_sum / sample_count
    result = static_gyrocompass(
        mean_acceleration,
        mean_angular_rate,
        latitude_rad=latitude_rad,
        earth_rotation_radps=earth_rotation_radps,
        gravity_mps2=float(earth.gravity_ned_mps2[2]),
    )
    return CarouselGyrocompassResult(
        result=GyrocompassResult(
            heading_rad=result.heading_rad,
            attitude_nav_from_body=result.attitude_nav_from_body,
            earth_rate_horizontal_radps=result.earth_rate_horizontal_radps,
            earth_rate_residual_radps=result.earth_rate_residual_radps,
            sample_count=sample_count,
        ),
        rotation_rate_radps=rotation_rate_radps,
        duration_s=actual_duration,
        revolution_count=rotation_rate_radps * actual_duration / (2.0 * pi),
        mean_acceleration_base=mean_acceleration,
        mean_angular_rate_base=mean_angular_rate,
    )
####


def run_carousel_heading_confidence_sweep(
        *,
        model_factory: Callable[[random.Generator], ImuModel],
        calibration: CalibrationEstimate,
        latitude_rad: float,
        heading_rad: float,
        rotation_rate_radps: float,
        durations_s: tuple[float, ...],
        sample_period_s: float = 0.02,
        trial_count: int = 100,
        height_m: float = 0.0,
        earth_rotation_radps: float = WGS84_EARTH_ROTATION_RADPS,
        seed: int = 0,
) -> HeadingConfidenceSweep:
    """Estimate empirical carousel heading confidence versus total data duration."""
    if not durations_s or any(not isfinite(value) or value <= 0.0 for value in durations_s):
        raise ValueError("durations_s must contain finite positive durations")
    ####
    if tuple(sorted(durations_s)) != durations_s:
        raise ValueError("durations_s must be sorted in ascending order")
    ####
    if isinstance(trial_count, bool) or not isinstance(trial_count, int) or trial_count <= 1:
        raise ValueError("trial_count must be an integer greater than one")
    ####
    seed_sequence = random.SeedSequence(seed)
    duration_seeds = seed_sequence.spawn(len(durations_s))
    points: list[HeadingConfidencePoint] = []
    for duration_s, duration_seed in zip(durations_s, duration_seeds):
        errors = zeros(trial_count)
        for trial_index, trial_seed in enumerate(duration_seed.spawn(trial_count)):
            result = simulate_carousel_gyrocompass(
                model_factory(random.default_rng(trial_seed)),
                calibration,
                latitude_rad=latitude_rad,
                heading_rad=heading_rad,
                rotation_rate_radps=rotation_rate_radps,
                duration_s=duration_s,
                sample_period_s=sample_period_s,
                height_m=height_m,
                earth_rotation_radps=earth_rotation_radps,
            )
            errors[trial_index] = wrap_angle_rad(result.result.heading_rad - heading_rad) * 180.0 / pi
        ####
        points.append(
            HeadingConfidencePoint(
                duration_s=duration_s,
                trial_count=trial_count,
                mean_error_deg=float(errors.mean()),
                standard_deviation_deg=float(errors.std(ddof=1)),
                rms_error_deg=float(sqrt((errors ** 2).mean())),
                p95_absolute_error_deg=float(percentile(abs(errors), 95.0)),
            )
        )
    ####
    return HeadingConfidenceSweep(
        latitude_rad=latitude_rad,
        sample_period_s=sample_period_s,
        points=tuple(points),
    )
####
