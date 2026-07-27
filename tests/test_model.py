import pytest
from numpy import allclose, array, cos, diag, eye, ones, pi, sin, testing, zeros
from numpy.linalg import norm
from numpy.random import default_rng

from imu_error_model import AxisConfig, ImuConfig, ImuModel
from imu_error_model.runtime import compile_axis_config


def test_zero_error_model_is_identity() -> None:
    model = ImuModel(rng=default_rng(1))
    baseline = model.measure(0.0, zeros(3), eye(3))
    testing.assert_allclose((baseline.start_time, baseline.end_time, baseline.dt), (0.0, 0.0, 0.0))
    out = model.measure(.01, array([1.0, 2.0, 3.0]), eye(3))
    testing.assert_allclose((out.start_time, out.end_time, out.dt), (0.0, 0.01, 0.01))
    testing.assert_allclose(out.delta_v, [1, 2, 3])
    testing.assert_allclose(out.delta_theta, [0, 0, 0])
####

def test_seeded_white_noise_is_reproducible() -> None:
    cfg = ImuConfig(accelerometer=AxisConfig(white_noise_density=.2))
    a_model, b_model = ImuModel(cfg, default_rng(7)), ImuModel(cfg, default_rng(7))
    a_model.measure(0, zeros(3), eye(3))
    b_model.measure(0, zeros(3), eye(3))
    a = a_model.measure(.01, zeros(3), eye(3))
    b = b_model.measure(.01, zeros(3), eye(3))
    testing.assert_array_equal(a.delta_v, b.delta_v)
####

def test_turn_on_bias_is_fixed_while_in_run_bias_evolves() -> None:
    config = ImuConfig(
        accelerometer=AxisConfig(turn_on_bias_std=0.5, bias_std=0.2, bias_correlation_time=1.0),
    )
    with_turn_on = ImuModel(config, default_rng(9))
    without_turn_on = ImuModel(
        ImuConfig(
            accelerometer=AxisConfig(bias_std=0.2, bias_correlation_time=1.0),
        ),
        default_rng(9),
    )
    expected_turn_on_bias = with_turn_on._accel_turn_on_bias.copy()
    for timestamp in (0.0, 0.1, 0.2, 0.3):
        with_turn_on.measure(timestamp, zeros(3), eye(3))
        without_turn_on.measure(timestamp, zeros(3), eye(3))
    ####
    for timestamp in (0.4, 0.5, 0.6):
        output_with = with_turn_on.measure(timestamp, zeros(3), eye(3))
        output_without = without_turn_on.measure(timestamp, zeros(3), eye(3))
        testing.assert_allclose(output_with.acceleration - output_without.acceleration, expected_turn_on_bias)
    ####
####

def test_random_walk_bias_starts_at_zero_and_grows_from_process_noise() -> None:
    config = ImuConfig(accelerometer=AxisConfig(bias_std=0.2))
    model = ImuModel(config, default_rng(17))
    testing.assert_array_equal(model._accel_bias, zeros(3))
    model.measure(0.0, zeros(3), eye(3))
    model.measure(0.5, zeros(3), eye(3))
    assert not allclose(model._accel_bias, zeros(3))
####

def test_clipping() -> None:
    cfg = ImuConfig(accelerometer=AxisConfig(measurement_range=1.0))
    model = ImuModel(cfg)
    model.measure(0, zeros(3), eye(3))
    out = model.measure(.1, array([.2, -.2, .05]), eye(3))
    testing.assert_allclose(out.acceleration, [1, -1, .5])
####

def test_axis_specific_scale_and_quantization() -> None:
    config = ImuConfig(
        accelerometer=AxisConfig(
            scale_factor=(0.1, 0.2, 0.3),
            quantization_step=(0.5, 1.0, 2.0),
        )
    )
    model = ImuModel(config, rng=default_rng(10))
    model.measure(0.0, zeros(3), eye(3))
    output = model.measure(1.0, ones(3), eye(3))
    testing.assert_allclose(output.delta_v, [1.0, 1.0, 2.0])
####

def test_compiled_axis_parameters_are_cached_and_read_only() -> None:
    config = ImuConfig(
        accelerometer=AxisConfig(
            scale_factor=(0.1, 0.2, 0.3),
            quantization_step=(0.5, 1.0, 2.0),
        )
    )
    first = ImuModel(config, rng=default_rng(1))
    second = ImuModel(config, rng=default_rng(2))

    assert first._accel_runtime is second._accel_runtime
    assert first._accel_runtime is compile_axis_config(config.accelerometer)
    assert not first._accel_runtime.scale_factor.flags.writeable
    quantization_step = first._accel_runtime.quantization_step
    assert quantization_step is not None
    assert not quantization_step.flags.writeable
    with pytest.raises(ValueError):
        first._accel_runtime.scale_factor[0] = 9.0
    ####
####

def test_anisotropic_misalignment_covariance_is_supported() -> None:
    covariance = ((1e-6, 0.0, 0.0), (0.0, 4e-6, 0.0), (0.0, 0.0, 9e-6))
    model = ImuModel(
        ImuConfig(accelerometer=AxisConfig(misalignment_covariance=covariance)),
        rng=default_rng(11),
    )
    testing.assert_allclose(
        model._accel_misalignment.T @ model._accel_misalignment,
        eye(3),
        atol=1e-12,
    )
####

def test_invalid_dt() -> None:
    with pytest.raises(ValueError):
        model = ImuModel()
        model.measure(0, zeros(3), eye(3))
        model.measure(0, zeros(3), eye(3))
    ####
####

def test_rotation_delta_is_reported_in_start_body_frame() -> None:
    model = ImuModel(rng=default_rng(3))
    model.measure(0.0, zeros(3), eye(3))
    angle = pi / 2
    rotation = array([[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]])
    out = model.measure(1.0, zeros(3), rotation)
    testing.assert_allclose(out.delta_theta, [0, 0, angle], atol=1e-7)
####

def test_velocity_delta_is_transformed_from_world_to_start_body() -> None:
    model = ImuModel(rng=default_rng(4))
    angle = pi / 2
    start = array([[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]])
    model.measure(0.0, zeros(3), start)
    out = model.measure(1.0, array([1.0, 0.0, 0.0]), start)
    testing.assert_allclose(out.delta_v, [0, -1, 0], atol=1e-7)
####

def test_run_level_misalignment_rotates_accelerometer_output() -> None:
    config = ImuConfig(accelerometer=AxisConfig(misalignment_std=0.1))
    model = ImuModel(config, rng=default_rng(12))
    model.measure(0.0, zeros(3), eye(3))
    out = model.measure(1.0, array([1.0, 2.0, 3.0]), eye(3))

    testing.assert_allclose(out.acceleration, model._accel_misalignment @ array([1.0, 2.0, 3.0]))
####

def test_rotation_delta_handles_half_turn() -> None:
    model = ImuModel(rng=default_rng(6))
    model.measure(0.0, zeros(3), eye(3))
    rotation = diag([1.0, -1.0, -1.0])
    out = model.measure(1.0, zeros(3), rotation)
    testing.assert_allclose(norm(out.delta_theta), pi, atol=1e-6)
####
