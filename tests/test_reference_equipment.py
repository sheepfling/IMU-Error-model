from math import pi

import numpy as np
import pytest

from playground.calibration.earth_reference import LocalEarthReference
from playground.calibration.reference_equipment import AngleLaw, RateTableReference, TableGeometry, aperture_reference, rotation_vector, skew
from playground.calibration.rotating_frame import average_sensor_from_base_rotation
from playground.calibration.turntable import _ferraris_pose_nav_from_body, simulate_wgs84_ferraris_turntable
from imu_error_model import ImuModel


def test_stationary_specific_force_has_physical_sign_and_earth_rate() -> None:
    earth = LocalEarthReference(0.6)
    p = RateTableReference(earth, AngleLaw(), TableGeometry())
    s = p.sample(np.array([0., 1.]))
    np.testing.assert_allclose(s.specific_force_sensor, np.tile(-earth.gravity_ned_mps2, (2, 1)), atol=1e-15)
    np.testing.assert_allclose(s.inertial_rate_sensor, np.tile(earth.earth_rotation_ned_radps, (2, 1)), atol=1e-15)
    assert not s.specific_force_sensor.flags.writeable
####


@pytest.mark.parametrize("axis", range(3))
def test_rodrigues_matches_quarter_turns(axis: int) -> None:
    vector = np.eye(3)[axis] * pi / 2
    r = rotation_vector(tuple(vector))
    e = np.eye(3)
    np.testing.assert_allclose(r @ e[axis], e[axis], atol=1e-15)
    np.testing.assert_allclose(r @ e[(axis + 1) % 3], e[(axis + 2) % 3], atol=1e-15)
    np.testing.assert_allclose(r.T @ r, e, atol=1e-15)
    assert np.linalg.det(r) == pytest.approx(1.)
####


def test_wobble_orientation_derivatives_against_finite_difference() -> None:
    p = RateTableReference(LocalEarthReference(0.6), AngleLaw(rate_radps=0.8, sine_rad=0.2, cosine_rad=0.1), TableGeometry(tilt_rotvec_ned_rad=(0.03, -0.02, 0.01), wobble_x_rad=0.03, wobble_y_rad=0.02, wobble_order=2, platform_from_sensor_rotvec_rad=(0.2, 0.3, -0.1)))
    t = np.array([0.2, 0.7, 1.1])
    h = 1e-4
    s, before, after = p.sample(t), p.sample(t-h), p.sample(t+h)
    np.testing.assert_allclose((after.orientation_ned_from_sensor-before.orientation_ned_from_sensor)/(2*h), s.orientation_dot, atol=2e-8)
    np.testing.assert_allclose((after.orientation_ned_from_sensor-2*s.orientation_ned_from_sensor+before.orientation_ned_from_sensor)/h**2, s.orientation_ddot, atol=2e-7)
    for i in range(len(t)):
        r = s.orientation_ned_from_sensor[i]
        np.testing.assert_allclose(r.T @ s.orientation_dot[i], skew(s.relative_rate_sensor[i]), atol=2e-15)
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-15)
    ####
####


@pytest.mark.parametrize("sine", (0.0, 0.3))
def test_lever_force_includes_tangential_and_centripetal_acceleration(sine: float) -> None:
    earth = LocalEarthReference(0.6)
    law = AngleLaw(rate_radps=2.0, sine_rad=sine)
    r = np.array([0.1, 0.03, 0.07])
    p = RateTableReference(earth, law, TableGeometry(lever_arm_sensor_m=tuple(r)), False)
    times = np.array([0.2, 0.8, 1.3])
    sample = p.sample(times)
    _, w, alpha = law.evaluate(times)
    expected = np.array([np.cross([0,0,a], r) + np.cross([0,0,v], np.cross([0,0,v], r)) - earth.gravity_ned_mps2 for v, a in zip(w, alpha, strict=True)])
    np.testing.assert_allclose(sample.specific_force_sensor, expected, atol=3e-15)
####


def test_coriolis_lever_term_is_retained() -> None:
    earth = LocalEarthReference(0.6)
    geometry = TableGeometry(lever_arm_sensor_m=(0.1, 0.0, 0.0))
    on = RateTableReference(earth, AngleLaw(rate_radps=2.0), geometry, True).sample(np.array([0.0]))
    off = RateTableReference(earth, AngleLaw(rate_radps=2.0), geometry, False).sample(np.array([0.0]))
    expected = 2 * np.cross(earth.earth_rotation_ned_radps, [0, .2, 0])
    np.testing.assert_allclose(on.specific_force_sensor[0] - off.specific_force_sensor[0], expected, atol=1e-15)
####


def test_aperture_matches_independent_constant_axis_identity() -> None:
    earth = LocalEarthReference(0.6)
    base = rotation_vector((0.2, -0.3, 0.1))
    mount = rotation_vector((pi/2, 0., 0.))
    law = AngleLaw(offset_rad=0.3, rate_radps=1.7)
    geometry = TableGeometry(base_rotvec_ned_rad=(0.2, -0.3, 0.1), platform_from_sensor_rotvec_rad=(pi/2, 0., 0.))
    p = RateTableReference(earth, law, geometry)
    starts, ends = np.array([0., .8]), np.array([.8, 1.6])
    averaged = aperture_reference(p, starts, ends, order=12)
    for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
        a = average_sensor_from_base_rotation(.3 + 1.7*start, .3 + 1.7*end)
        expected_w = mount.T @ (a @ base.T @ earth.earth_rotation_ned_radps + [0, 0, 1.7])
        expected_f = -mount.T @ a @ base.T @ earth.gravity_ned_mps2
        np.testing.assert_allclose(averaged.force[i], expected_f, atol=1e-13)
        np.testing.assert_allclose(averaged.rate[i], expected_w, atol=1e-15)
    ####
####


def test_general_aperture_converges_with_quadrature_order() -> None:
    p = RateTableReference(LocalEarthReference(.6), AngleLaw(rate_radps=2., sine_rad=.5, frequency_hz=.6), TableGeometry(wobble_x_rad=.04, wobble_y_rad=.03, wobble_order=3, lever_arm_sensor_m=(.08, -.03, .02)))
    starts, ends = np.array([0.,.4]), np.array([.4,.8])
    values = [aperture_reference(p, starts, ends, order=n) for n in (2,4,8,16,32)]
    errors = [np.max(np.abs(np.column_stack((v.force, v.rate))-np.column_stack((values[-1].force, values[-1].rate)))) for v in values[:-1]]
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[3] < errors[2]
    assert errors[3] < 1e-10
####


@pytest.mark.parametrize("label,axis", (("x_rot",0),("y_rot",1),("z_rot",2)))
def test_legacy_turntable_earth_projection_uses_correct_physical_axis(label: str, axis: int) -> None:
    earth = LocalEarthReference(.6)
    dataset = simulate_wgs84_ferraris_turntable(ImuModel(), latitude_rad=.6, sampling_rate_hz=2., static_duration_s=.5, table_rate_radps=pi/2, expected_angle_rad=-pi)
    index = dataset.labels.index(label)
    r0 = _ferraris_pose_nav_from_body(label[:1]+"_p")
    # Independent midpoint integration in the correct NED rotation plane.
    t = (np.arange(20000)+.5)*.5/20000
    phi = -pi/2*t
    north, east, down = earth.earth_rotation_ned_radps
    earth_rotated = np.column_stack((np.cos(phi)*north+np.sin(phi)*east,-np.sin(phi)*north+np.cos(phi)*east,np.full(len(t),down)))
    expected = np.mean(earth_rotated @ r0,axis=0) - pi/2*np.eye(3)[axis]
    np.testing.assert_allclose(dataset.angular_rate_body[index], expected, atol=1e-14)
    np.testing.assert_allclose(dataset.specific_force_body[index], earth.gravity_ned_mps2[2]*np.eye(3)[axis], atol=1e-14)
####


@pytest.mark.parametrize("kwargs", ({"wobble_order":0},{"wobble_order":True},{"wobble_x_rad":float("nan")},{"tilt_rotvec_ned_rad":(0.,0.)},{"lever_arm_sensor_m":(0.,0.,float("inf"))}))
def test_geometry_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TableGeometry(**kwargs)
    ####
####


@pytest.mark.parametrize("starts,ends", (([],[]),([1.],[1.]),([0., .1],[.2,.3]),([0.],[float("inf")]),([0.],[.1,.2])))
def test_aperture_rejects_bad_intervals(starts: list[float], ends: list[float]) -> None:
    with pytest.raises(ValueError):
        aperture_reference(RateTableReference(LocalEarthReference(.6), AngleLaw(), TableGeometry()), np.array(starts),np.array(ends))
    ####
####


@pytest.mark.parametrize("order", (1,65,True))
def test_aperture_rejects_invalid_order(order: int) -> None:
    with pytest.raises(ValueError):
        aperture_reference(RateTableReference(LocalEarthReference(.6),AngleLaw(),TableGeometry()),np.array([0.]),np.array([1.]),order=order)
    ####
####


def test_static_alignment_accepts_physical_specific_force_explicitly() -> None:
    from playground.calibration.gyrocompass import static_gyrocompass
    earth=LocalEarthReference(.6)
    p=RateTableReference(earth,AngleLaw(),TableGeometry(base_rotvec_ned_rad=(.03,-.05,.4)))
    s=p.sample(np.array([0.]))
    result=static_gyrocompass(s.specific_force_sensor[0],s.inertial_rate_sensor[0],latitude_rad=.6,gravity_mps2=float(earth.gravity_ned_mps2[2]),accelerometer_convention="specific_force")
    np.testing.assert_allclose(result.attitude_nav_from_body,s.orientation_ned_from_sensor[0],atol=1e-14)
####


def test_heading_covariance_rejects_negative_variance_and_supports_physical_sign() -> None:
    from playground.calibration.heading_uncertainty import propagate_heading_uncertainty
    from playground.calibration.known_reference import fit_known_reference_calibration, simulate_reference_program, ferraris_style_reference_program
    cal=fit_known_reference_calibration(simulate_reference_program(ImuModel(),ferraris_style_reference_program()))
    earth=LocalEarthReference(.6)
    args=dict(latitude_rad=.6,earth_rotation_radps=earth.earth_rotation_radps,gravity_mps2=float(earth.gravity_ned_mps2[2]),accelerometer_convention="specific_force")
    valid=propagate_heading_uncertainty(-earth.gravity_ned_mps2,earth.earth_rotation_ned_radps,np.diag([1e-8]*3+[1e-14]*3),cal,**args)
    assert 0<valid.confidence_95_half_width_deg<1
    with pytest.raises(ValueError,match="positive semidefinite"):
        propagate_heading_uncertainty(-earth.gravity_ned_mps2,earth.earth_rotation_ned_radps,np.diag([-1e-8]+[1e-8]*5),cal,**args)
    ####
####
