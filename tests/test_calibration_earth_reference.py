from math import cos, pi

from numpy import isclose
from numpy.linalg import norm

from playground.calibration.earth_reference import (
    WGS84_EARTH_ROTATION_RADPS,
    LocalEarthReference,
    earth_rotation_ned,
    normal_gravity_mps2,
)


def test_wgs84_earth_rate_preserves_magnitude_and_local_components() -> None:
    equator = earth_rotation_ned(0.0)
    midlatitude = earth_rotation_ned(45.0 * pi / 180.0)
    pole = earth_rotation_ned(pi / 2.0)

    assert isclose(norm(equator), WGS84_EARTH_ROTATION_RADPS, rtol=0.0, atol=1e-15)
    assert isclose(norm(midlatitude), WGS84_EARTH_ROTATION_RADPS, rtol=0.0, atol=1e-15)
    assert isclose(norm(pole), WGS84_EARTH_ROTATION_RADPS, rtol=0.0, atol=1e-15)
    assert isclose(equator[0], WGS84_EARTH_ROTATION_RADPS)
    assert isclose(pole[2], -WGS84_EARTH_ROTATION_RADPS)
####


def test_normal_gravity_varies_with_latitude_and_height() -> None:
    equator = normal_gravity_mps2(0.0)
    pole = normal_gravity_mps2(pi / 2.0)
    high = normal_gravity_mps2(35.0 * pi / 180.0, 2000.0)
    sea_level = normal_gravity_mps2(35.0 * pi / 180.0, 0.0)

    assert pole > equator
    assert high < sea_level
    assert 9.77 < equator < 9.79
    assert 9.82 < pole < 9.84
####


def test_local_reference_reports_horizontal_earth_rate() -> None:
    reference = LocalEarthReference(latitude_rad=60.0 * pi / 180.0, height_m=1000.0)

    assert isclose(
        reference.horizontal_earth_rate_radps,
        0.5 * WGS84_EARTH_ROTATION_RADPS,
        rtol=1e-14,
    )
####


def test_local_reference_accepts_a_traceable_earth_rate_override() -> None:
    custom_rate = 1.0002 * WGS84_EARTH_ROTATION_RADPS
    latitude_rad = 35.0 * pi / 180.0
    reference = LocalEarthReference(
        latitude_rad=latitude_rad,
        earth_rotation_radps=custom_rate,
    )

    assert isclose(norm(reference.earth_rotation_ned_radps), custom_rate, rtol=0.0, atol=1e-15)
    assert isclose(
        reference.horizontal_earth_rate_radps,
        custom_rate * cos(latitude_rad),
        rtol=1e-14,
    )
####
