from math import pi

from numpy import abs as np_abs, eye

from imu_error_model import ImuModel
from playground.calibration.earth_reference import WGS84_EARTH_ROTATION_RADPS, normal_gravity_mps2
from playground.calibration.imucal_adapter import fit_ferraris_equation_reference
from playground.calibration.turntable import simulate_wgs84_ferraris_turntable


def _fitted_gyro_error(table_rate_degps: float, *, include_earth_rotation: bool) -> float:
    latitude_rad = 34.7304 * pi / 180.0
    sampling_rate_hz = 20.0
    dataset = simulate_wgs84_ferraris_turntable(
        ImuModel(),
        latitude_rad=latitude_rad,
        sampling_rate_hz=sampling_rate_hz,
        static_duration_s=0.2,
        table_rate_radps=table_rate_degps * pi / 180.0,
        expected_angle_rad=-2.0 * pi,
        include_earth_rotation=include_earth_rotation,
    )
    estimate = fit_ferraris_equation_reference(
        dataset,
        sampling_rate_hz=sampling_rate_hz,
        gravity_mps2=normal_gravity_mps2(latitude_rad),
        expected_angle_degrees=-360.0,
    )
    return float(np_abs(estimate.gyroscope_sensitivity_matrix - eye(3)).max())
####


def test_earth_free_dynamic_turntable_recovers_identity() -> None:
    assert _fitted_gyro_error(1.0, include_earth_rotation=False) < 1e-12
####


def test_slow_turntable_exposes_earth_rate_model_mismatch() -> None:
    fast_error = _fitted_gyro_error(90.0, include_earth_rotation=True)
    slow_error = _fitted_gyro_error(1.0, include_earth_rotation=True)

    assert fast_error < 1e-4
    assert slow_error > 1e-3
    assert slow_error > 50.0 * fast_error
####


def test_turntable_uses_exact_finite_aperture_earth_rate_projection() -> None:
    from numpy import asarray, allclose

    from playground.calibration.earth_reference import LocalEarthReference
    from playground.calibration.rotating_frame import average_sensor_from_base_rotation, rotation_z

    latitude_rad = 34.7304 * pi / 180.0
    sampling_rate_hz = 1.0
    table_rate_radps = pi / 2.0
    dataset = simulate_wgs84_ferraris_turntable(
        ImuModel(),
        latitude_rad=latitude_rad,
        sampling_rate_hz=sampling_rate_hz,
        static_duration_s=1.0,
        table_rate_radps=table_rate_radps,
        expected_angle_rad=-2.0 * pi,
        include_earth_rotation=True,
    )
    first_rotation_index = dataset.labels.index("x_rot")
    measured_ideal_rate = dataset.angular_rate_body[first_rotation_index]
    earth_rate_nav = LocalEarthReference(latitude_rad).earth_rotation_ned_radps
    north = asarray((1.0, 0.0, 0.0), dtype=float)
    east = asarray((0.0, 1.0, 0.0), dtype=float)
    down = asarray((0.0, 0.0, 1.0), dtype=float)
    nav_from_base = asarray((down, east, -north), dtype=float).T
    signed_rate = -table_rate_radps
    exact = (
        nav_from_base.T
        @ average_sensor_from_base_rotation(0.0, signed_rate / sampling_rate_hz)
        @ earth_rate_nav
        + asarray((signed_rate, 0.0, 0.0), dtype=float)
    )
    midpoint = (
        nav_from_base.T
        @ rotation_z(0.5 * signed_rate / sampling_rate_hz).T
        @ earth_rate_nav
        + asarray((signed_rate, 0.0, 0.0), dtype=float)
    )

    assert allclose(measured_ideal_rate, exact, rtol=0.0, atol=1e-15)
    assert float(np_abs(exact - midpoint).max()) > 1e-6 * WGS84_EARTH_ROTATION_RADPS
####
