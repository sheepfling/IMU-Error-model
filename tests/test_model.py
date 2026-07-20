import numpy as np
import pytest

from imu_error_model import AxisConfig, ImuConfig, ImuModel


def test_zero_error_model_is_identity() -> None:
    model = ImuModel(rng=np.random.default_rng(1))
    model.measure(0.0, np.zeros(3), np.eye(3))
    out = model.measure(.01, np.array([1.0, 2.0, 3.0]), np.eye(3))
    np.testing.assert_allclose(out.delta_v, [1, 2, 3])
    np.testing.assert_allclose(out.delta_theta, [0, 0, 0])


def test_seeded_white_noise_is_reproducible() -> None:
    cfg = ImuConfig(accelerometer=AxisConfig(white_noise_density=.2))
    a_model, b_model = ImuModel(cfg, np.random.default_rng(7)), ImuModel(cfg, np.random.default_rng(7))
    a_model.measure(0, np.zeros(3), np.eye(3)); b_model.measure(0, np.zeros(3), np.eye(3))
    a = a_model.measure(.01, np.zeros(3), np.eye(3))
    b = b_model.measure(.01, np.zeros(3), np.eye(3))
    np.testing.assert_array_equal(a.delta_v, b.delta_v)


def test_turn_on_bias_is_fixed_while_in_run_bias_evolves() -> None:
    config = ImuConfig(
        accelerometer=AxisConfig(turn_on_bias_std=0.5, bias_std=0.2, bias_correlation_time=1.0),
    )
    with_turn_on = ImuModel(config, np.random.default_rng(9))
    without_turn_on = ImuModel(
        ImuConfig(
            accelerometer=AxisConfig(bias_std=0.2, bias_correlation_time=1.0),
        ),
        np.random.default_rng(9),
    )
    expected_turn_on_bias = with_turn_on._accel_turn_on_bias.copy()
    for timestamp in (0.0, 0.1, 0.2, 0.3):
        with_turn_on.measure(timestamp, np.zeros(3), np.eye(3))
        without_turn_on.measure(timestamp, np.zeros(3), np.eye(3))
    for timestamp in (0.4, 0.5, 0.6):
        output_with = with_turn_on.measure(timestamp, np.zeros(3), np.eye(3))
        output_without = without_turn_on.measure(timestamp, np.zeros(3), np.eye(3))
        np.testing.assert_allclose(output_with.acceleration - output_without.acceleration, expected_turn_on_bias)


def test_clipping() -> None:
    cfg = ImuConfig(accelerometer=AxisConfig(measurement_range=1.0))
    model = ImuModel(cfg); model.measure(0, np.zeros(3), np.eye(3))
    out = model.measure(.1, np.array([.2, -.2, .05]), np.eye(3))
    np.testing.assert_allclose(out.acceleration, [1, -1, .5])


def test_invalid_dt() -> None:
    with pytest.raises(ValueError):
        model = ImuModel(); model.measure(0, np.zeros(3), np.eye(3)); model.measure(0, np.zeros(3), np.eye(3))


def test_rotation_delta_is_reported_in_start_body_frame() -> None:
    model = ImuModel(rng=np.random.default_rng(3))
    model.measure(0.0, np.zeros(3), np.eye(3))
    angle = np.pi / 2
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    out = model.measure(1.0, np.zeros(3), rotation)
    np.testing.assert_allclose(out.delta_theta, [0, 0, angle], atol=1e-7)


def test_velocity_delta_is_transformed_from_world_to_start_body() -> None:
    model = ImuModel(rng=np.random.default_rng(4))
    angle = np.pi / 2
    start = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    model.measure(0.0, np.zeros(3), start)
    out = model.measure(1.0, np.array([1.0, 0.0, 0.0]), start)
    np.testing.assert_allclose(out.delta_v, [0, -1, 0], atol=1e-7)


def test_rotation_delta_handles_half_turn() -> None:
    model = ImuModel(rng=np.random.default_rng(6))
    model.measure(0.0, np.zeros(3), np.eye(3))
    rotation = np.diag([1.0, -1.0, -1.0])
    out = model.measure(1.0, np.zeros(3), rotation)
    np.testing.assert_allclose(np.linalg.norm(out.delta_theta), np.pi, atol=1e-6)
