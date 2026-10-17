"""Exact finite-aperture helpers for constant-rate down-axis rotation."""

from __future__ import annotations

from math import cos, isfinite, sin

from numpy import asarray, ndarray


def rotation_z(angle_rad: float) -> ndarray:
    """Return a right-handed active rotation about the positive third axis."""
    if not isfinite(angle_rad):
        raise ValueError("angle_rad must be finite")
    ####
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


def average_sensor_from_base_rotation(start_angle_rad: float, end_angle_rad: float) -> ndarray:
    """Return the exact interval average of ``R_sensor_from_base(alpha)``.

    A sensor rotating about the base-frame third axis observes a base-fixed
    vector through ``R_z(alpha).T``.  This function integrates that transform
    exactly over one sample aperture rather than replacing it with a midpoint
    orientation.  A full revolution within one sample is singular because its
    horizontal projection averages to zero.
    """
    if not isfinite(start_angle_rad) or not isfinite(end_angle_rad):
        raise ValueError("rotation angles must be finite")
    ####
    delta = end_angle_rad - start_angle_rad
    if abs(delta) <= 1e-12:
        return rotation_z(0.5 * (start_angle_rad + end_angle_rad)).T
    ####
    mean_cosine = (sin(end_angle_rad) - sin(start_angle_rad)) / delta
    mean_sine = (cos(start_angle_rad) - cos(end_angle_rad)) / delta
    transform = asarray(
        (
            (mean_cosine, mean_sine, 0.0),
            (-mean_sine, mean_cosine, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=float,
    )
    if abs(mean_cosine) + abs(mean_sine) <= 1e-12:
        raise ValueError("one sample spans a singular full-revolution aperture")
    ####
    return transform
####


def demodulate_interval_average(sensor_from_base_average: ndarray, sensor_vector: ndarray) -> ndarray:
    """Invert the structured aperture map without a generic matrix solve."""
    transform = asarray(sensor_from_base_average, dtype=float)
    vector = asarray(sensor_vector, dtype=float)
    if transform.shape != (3, 3) or vector.shape != (3,):
        raise ValueError("rotation transform and vector must have shapes (3, 3) and (3,)")
    ####
    cosine_component = float(transform[0, 0])
    sine_component = float(transform[0, 1])
    horizontal_gain_squared = cosine_component ** 2 + sine_component ** 2
    if horizontal_gain_squared <= 1e-24:
        raise ValueError("rotation aperture map is singular")
    ####
    return asarray(
        (
            (cosine_component * vector[0] - sine_component * vector[1]) / horizontal_gain_squared,
            (sine_component * vector[0] + cosine_component * vector[1]) / horizontal_gain_squared,
            vector[2],
        ),
        dtype=float,
    )
####
