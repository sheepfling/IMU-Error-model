"""WGS-84 local Earth references for calibration and gyrocompass experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, pi, sin, sqrt

from numpy import asarray, ndarray


WGS84_SEMI_MAJOR_AXIS_M = 6_378_137.0
WGS84_INVERSE_FLATTENING = 298.257223563
WGS84_FLATTENING = 1.0 / WGS84_INVERSE_FLATTENING
WGS84_SEMI_MINOR_AXIS_M = WGS84_SEMI_MAJOR_AXIS_M * (1.0 - WGS84_FLATTENING)
WGS84_EARTH_ROTATION_RADPS = 7.292115e-5
WGS84_GEOCENTRIC_GRAVITATIONAL_CONSTANT_M3PS2 = 3.986004418e14
WGS84_EQUATORIAL_NORMAL_GRAVITY_MPS2 = 9.7803253359
WGS84_SOMIGLIANA_K = 0.00193185265241
WGS84_ECCENTRICITY_SQUARED = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING)


def _validate_latitude(latitude_rad: float) -> None:
    if not isfinite(latitude_rad) or abs(latitude_rad) > pi / 2.0:
        raise ValueError("latitude_rad must be finite and within [-pi/2, pi/2]")
    ####
####


def earth_rotation_ned(
        latitude_rad: float,
        earth_rotation_radps: float = WGS84_EARTH_ROTATION_RADPS,
) -> ndarray:
    """Return a configurable Earth rotation vector in local NED axes."""
    _validate_latitude(latitude_rad)
    if not isfinite(earth_rotation_radps) or earth_rotation_radps <= 0.0:
        raise ValueError("earth_rotation_radps must be finite and positive")
    ####
    return earth_rotation_radps * asarray(
        (cos(latitude_rad), 0.0, -sin(latitude_rad)),
        dtype=float,
    )
####


def normal_gravity_mps2(latitude_rad: float, height_m: float = 0.0) -> float:
    """Return WGS-84 ellipsoidal normal gravity with a second-order height correction."""
    _validate_latitude(latitude_rad)
    if not isfinite(height_m):
        raise ValueError("height_m must be finite")
    ####
    sine_squared = sin(latitude_rad) ** 2
    surface = WGS84_EQUATORIAL_NORMAL_GRAVITY_MPS2 * (
        1.0 + WGS84_SOMIGLIANA_K * sine_squared
    ) / sqrt(1.0 - WGS84_ECCENTRICITY_SQUARED * sine_squared)
    m = (
        WGS84_EARTH_ROTATION_RADPS ** 2
        * WGS84_SEMI_MAJOR_AXIS_M ** 2
        * WGS84_SEMI_MINOR_AXIS_M
        / WGS84_GEOCENTRIC_GRAVITATIONAL_CONSTANT_M3PS2
    )
    first_order = (
        2.0
        / WGS84_SEMI_MAJOR_AXIS_M
        * (1.0 + WGS84_FLATTENING + m - 2.0 * WGS84_FLATTENING * sine_squared)
        * height_m
    )
    second_order = 3.0 * height_m ** 2 / WGS84_SEMI_MAJOR_AXIS_M ** 2
    return surface * (1.0 - first_order + second_order)
####


def apparent_gravity_ned(latitude_rad: float, height_m: float = 0.0) -> ndarray:
    """Return the local down-positive normal-gravity reference used by the playground."""
    return asarray((0.0, 0.0, normal_gravity_mps2(latitude_rad, height_m)), dtype=float)
####


@dataclass(frozen=True, slots=True)
class LocalEarthReference:
    """Local WGS-84 gravity and Earth-rate vectors for a stationary Earth-fixed site."""

    latitude_rad: float
    height_m: float = 0.0
    earth_rotation_radps: float = WGS84_EARTH_ROTATION_RADPS

    def __post_init__(self) -> None:
        _validate_latitude(self.latitude_rad)
        if not isfinite(self.height_m):
            raise ValueError("height_m must be finite")
        ####
        if not isfinite(self.earth_rotation_radps) or self.earth_rotation_radps <= 0.0:
            raise ValueError("earth_rotation_radps must be finite and positive")
        ####
    ####

    @property
    def earth_rotation_ned_radps(self) -> ndarray:
        return earth_rotation_ned(self.latitude_rad, self.earth_rotation_radps)
    ####

    @property
    def gravity_ned_mps2(self) -> ndarray:
        return apparent_gravity_ned(self.latitude_rad, self.height_m)
    ####

    @property
    def horizontal_earth_rate_radps(self) -> float:
        return abs(self.earth_rotation_radps * cos(self.latitude_rad))
    ####
####
