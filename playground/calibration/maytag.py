"""Two- and four-position static gyrocompass modulation experiments."""

from __future__ import annotations

from collections.abc import Callable
from math import cos, isfinite, pi, sin

from numpy import asarray, ndarray, percentile, random, sqrt, zeros

from imu_error_model import ImuModel

from .data_confidence import HeadingConfidencePoint, HeadingConfidenceSweep
from .earth_reference import LocalEarthReference, WGS84_EARTH_ROTATION_RADPS
from .gyrocompass import GyrocompassResult, apply_calibration, static_gyrocompass, wrap_angle_rad
from .known_reference import CalibrationEstimate


def _rotation_z(angle_rad: float) -> ndarray:
    cosine = cos(angle_rad)
    sine = sin(angle_rad)
    return asarray(
        (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=float,
    )
####


def maytag_positions(position_count: int) -> tuple[float, ...]:
    """Return equally spaced down-axis modulation angles for 1, 2, or 4 positions."""
    if position_count not in {1, 2, 4}:
        raise ValueError("position_count must be one of 1, 2, or 4")
    ####
    return tuple(2.0 * pi * index / position_count for index in range(position_count))
####


def simulate_maytag_gyrocompass(
        model: ImuModel,
        calibration: CalibrationEstimate,
        *,
        latitude_rad: float,
        heading_rad: float,
        position_count: int = 2,
        samples_per_position: int = 3000,
        sample_period_s: float = 0.02,
        height_m: float = 0.0,
        earth_rotation_radps: float = WGS84_EARTH_ROTATION_RADPS,
) -> GyrocompassResult:
    """North-find after rotating one physical IMU through static Maytag positions.

    The model is reset only once, so run-level turn-on bias is fixed in sensor
    coordinates across all positions and is therefore modulated by the physical
    rotations rather than resampled.
    """
    positions = maytag_positions(position_count)
    if isinstance(samples_per_position, bool) or not isinstance(samples_per_position, int) or samples_per_position <= 0:
        raise ValueError("samples_per_position must be a positive integer")
    ####
    if not isfinite(sample_period_s) or sample_period_s <= 0.0:
        raise ValueError("sample_period_s must be finite and positive")
    ####
    earth = LocalEarthReference(
        latitude_rad=latitude_rad,
        height_m=height_m,
        earth_rotation_radps=earth_rotation_radps,
    )
    nav_from_base = _rotation_z(heading_rad)
    base_accel_sum = zeros(3)
    base_gyro_sum = zeros(3)
    model.reset()
    current = 0.0

    for position_angle in positions:
        base_from_sensor = _rotation_z(position_angle)
        nav_from_sensor = nav_from_base @ base_from_sensor
        gravity_sensor = nav_from_sensor.T @ earth.gravity_ned_mps2
        earth_rate_sensor = nav_from_sensor.T @ earth.earth_rotation_ned_radps
        sensor_accel_sum = zeros(3)
        sensor_gyro_sum = zeros(3)
        for _ in range(samples_per_position):
            output = model.measure_ideal_rates(
                start_time=current,
                end_time=current + sample_period_s,
                specific_force_body=gravity_sensor,
                angular_rate_body=earth_rate_sensor,
            )
            calibrated_accel, calibrated_gyro = apply_calibration(
                output.acceleration,
                output.angular_rate,
                calibration,
            )
            sensor_accel_sum += calibrated_accel
            sensor_gyro_sum += calibrated_gyro
            current += sample_period_s
        ####
        base_accel_sum += base_from_sensor @ (sensor_accel_sum / samples_per_position)
        base_gyro_sum += base_from_sensor @ (sensor_gyro_sum / samples_per_position)
    ####

    base_accel = base_accel_sum / position_count
    base_gyro = base_gyro_sum / position_count
    result = static_gyrocompass(
        base_accel,
        base_gyro,
        latitude_rad=latitude_rad,
        earth_rotation_radps=earth_rotation_radps,
        gravity_mps2=float(earth.gravity_ned_mps2[2]),
    )
    return GyrocompassResult(
        heading_rad=result.heading_rad,
        attitude_nav_from_body=result.attitude_nav_from_body,
        earth_rate_horizontal_radps=result.earth_rate_horizontal_radps,
        earth_rate_residual_radps=result.earth_rate_residual_radps,
        sample_count=position_count * samples_per_position,
    )
####



def run_maytag_heading_confidence_sweep(
        *,
        model_factory: Callable[[random.Generator], ImuModel],
        calibration: CalibrationEstimate,
        latitude_rad: float,
        heading_rad: float,
        position_count: int,
        durations_s: tuple[float, ...],
        sample_period_s: float = 0.02,
        trial_count: int = 100,
        height_m: float = 0.0,
        earth_rotation_radps: float = WGS84_EARTH_ROTATION_RADPS,
        seed: int = 0,
) -> HeadingConfidenceSweep:
    """Estimate empirical Maytag heading confidence versus total test duration."""
    maytag_positions(position_count)
    if not durations_s or any(not isfinite(value) or value <= 0.0 for value in durations_s):
        raise ValueError("durations_s must contain finite positive durations")
    ####
    if tuple(sorted(durations_s)) != durations_s:
        raise ValueError("durations_s must be sorted in ascending order")
    ####
    if not isfinite(sample_period_s) or sample_period_s <= 0.0:
        raise ValueError("sample_period_s must be finite and positive")
    ####
    if isinstance(trial_count, bool) or not isinstance(trial_count, int) or trial_count <= 1:
        raise ValueError("trial_count must be an integer greater than one")
    ####
    samples_per_position = tuple(
        round(duration / (position_count * sample_period_s))
        for duration in durations_s
    )
    if any(value <= 0 for value in samples_per_position):
        raise ValueError("each duration must provide at least one sample per Maytag position")
    ####
    if len(set(samples_per_position)) != len(samples_per_position):
        raise ValueError("durations_s collapse to duplicate samples-per-position values")
    ####
    points: list[HeadingConfidencePoint] = []
    for samples, duration_seed in zip(samples_per_position, random.SeedSequence(seed).spawn(len(durations_s)), strict=True):
        errors = zeros(trial_count)
        for trial_index, trial_seed in enumerate(duration_seed.spawn(trial_count)):
            result = simulate_maytag_gyrocompass(
                model_factory(random.default_rng(trial_seed)),
                calibration,
                latitude_rad=latitude_rad,
                heading_rad=heading_rad,
                position_count=position_count,
                samples_per_position=samples,
                sample_period_s=sample_period_s,
                height_m=height_m,
                earth_rotation_radps=earth_rotation_radps,
            )
            errors[trial_index] = wrap_angle_rad(result.heading_rad - heading_rad) * 180.0 / pi
        ####
        actual_duration = position_count * samples * sample_period_s
        points.append(
            HeadingConfidencePoint(
                duration_s=actual_duration,
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
