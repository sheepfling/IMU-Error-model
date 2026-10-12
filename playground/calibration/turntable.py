"""WGS-84-aware Ferraris and rate-table excitation programs."""

from __future__ import annotations

from math import isfinite, pi

from numpy import asarray, eye, ndarray, zeros

from imu_error_model import ImuModel

from .earth_reference import LocalEarthReference, WGS84_EARTH_ROTATION_RADPS
from .known_reference import CalibrationDataset
from .rotating_frame import average_sensor_from_base_rotation


def _ferraris_pose_nav_from_body(label: str) -> ndarray:
    """Return proper NED-from-body orientation for one Ferraris static pose."""
    north = asarray((1.0, 0.0, 0.0), dtype=float)
    east = asarray((0.0, 1.0, 0.0), dtype=float)
    down = asarray((0.0, 0.0, 1.0), dtype=float)
    poses = {
        "x_p": asarray((down, east, -north), dtype=float).T,
        "x_a": asarray((-down, east, north), dtype=float).T,
        "y_p": asarray((north, down, -east), dtype=float).T,
        "y_a": asarray((north, -down, east), dtype=float).T,
        "z_p": eye(3),
        "z_a": asarray((north, -east, -down), dtype=float).T,
    }
    try:
        return poses[label].copy()
    except KeyError as error:
        raise ValueError(f"unknown Ferraris pose: {label!r}") from error
    ####
####


def simulate_wgs84_ferraris_turntable(
        model: ImuModel,
        *,
        latitude_rad: float,
        height_m: float = 0.0,
        sampling_rate_hz: float = 100.0,
        static_duration_s: float = 2.0,
        table_rate_radps: float = pi / 2.0,
        expected_angle_rad: float = -2.0 * pi,
        temperature_celsius: float | None = None,
        include_earth_rotation: bool = True,
        earth_rotation_radps: float = WGS84_EARTH_ROTATION_RADPS,
) -> CalibrationDataset:
    """Generate a nine-region Ferraris corpus with dynamic Earth-rate projection.

    The table axis is local down.  For each rotation region the corresponding
    positive body axis is aligned with down, matching the standard Ferraris
    geometry.  Earth rotation is represented in NED and integrated through the
    exact interval-average rotating-frame transform for each sample aperture.
    """
    for field_name, value in (
        ("sampling_rate_hz", sampling_rate_hz),
        ("static_duration_s", static_duration_s),
        ("table_rate_radps", table_rate_radps),
    ):
        if not isfinite(value) or value <= 0.0:
            raise ValueError(f"{field_name} must be finite and positive")
        ####
    ####
    if not isfinite(expected_angle_rad) or expected_angle_rad == 0.0:
        raise ValueError("expected_angle_rad must be finite and non-zero")
    ####
    if temperature_celsius is not None and not isfinite(temperature_celsius):
        raise ValueError("temperature_celsius must be finite when supplied")
    ####

    earth = LocalEarthReference(
        latitude_rad=latitude_rad,
        height_m=height_m,
        earth_rotation_radps=earth_rotation_radps,
    )
    gravity_nav = earth.gravity_ned_mps2
    earth_rate_nav = earth.earth_rotation_ned_radps if include_earth_rotation else zeros(3)
    sample_period_s = 1.0 / sampling_rate_hz
    static_sample_count = max(1, round(static_duration_s * sampling_rate_hz))
    rotation_duration_s = abs(expected_angle_rad) / table_rate_radps
    rotation_sample_count = max(1, round(rotation_duration_s * sampling_rate_hz))
    signed_table_rate = expected_angle_rad / (rotation_sample_count * sample_period_s)

    labels: list[str] = []
    starts: list[float] = []
    ends: list[float] = []
    temperatures: list[float] = []
    ideal_accel: list[ndarray] = []
    ideal_gyro: list[ndarray] = []
    measured_accel: list[ndarray] = []
    measured_gyro: list[ndarray] = []
    current = 0.0
    model.reset()

    def append_sample(
            label: str,
            acceleration_body: ndarray,
            angular_rate_body: ndarray,
    ) -> None:
        nonlocal current
        output = model.measure_ideal_rates(
            start_time=current,
            end_time=current + sample_period_s,
            specific_force_body=acceleration_body,
            angular_rate_body=angular_rate_body,
            temperature_celsius=temperature_celsius,
        )
        labels.append(label)
        starts.append(current)
        current += sample_period_s
        ends.append(current)
        temperatures.append(float("nan") if temperature_celsius is None else temperature_celsius)
        ideal_accel.append(acceleration_body)
        ideal_gyro.append(angular_rate_body)
        measured_accel.append(output.acceleration)
        measured_gyro.append(output.angular_rate)
    ####

    for pose_label in ("x_p", "x_a", "y_p", "y_a", "z_p", "z_a"):
        nav_from_body = _ferraris_pose_nav_from_body(pose_label)
        acceleration_body = nav_from_body.T @ gravity_nav
        angular_rate_body = nav_from_body.T @ earth_rate_nav
        for _ in range(static_sample_count):
            append_sample(pose_label, acceleration_body, angular_rate_body)
        ####
    ####

    body_axes = eye(3)
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        base_orientation = _ferraris_pose_nav_from_body(f"{axis_name}_p")
        gravity_base = base_orientation.T @ gravity_nav
        earth_rate_base = base_orientation.T @ earth_rate_nav
        for sample_index in range(rotation_sample_count):
            start_angle = signed_table_rate * sample_index * sample_period_s
            end_angle = start_angle + signed_table_rate * sample_period_s
            # The table rotates about NED down, not the body's third axis.
            sensor_from_base_average = (
                base_orientation.T
                @ average_sensor_from_base_rotation(start_angle, end_angle)
                @ base_orientation
            )
            append_sample(
                f"{axis_name}_rot",
                sensor_from_base_average @ gravity_base,
                sensor_from_base_average @ earth_rate_base
                + signed_table_rate * body_axes[axis_index],
            )
        ####
    ####

    return CalibrationDataset(
        labels=tuple(labels),
        start_times=asarray(starts, dtype=float),
        end_times=asarray(ends, dtype=float),
        temperatures_celsius=asarray(temperatures, dtype=float),
        specific_force_body=asarray(ideal_accel, dtype=float),
        angular_rate_body=asarray(ideal_gyro, dtype=float),
        measured_acceleration=asarray(measured_accel, dtype=float),
        measured_angular_rate=asarray(measured_gyro, dtype=float),
    )
####
