from typing import Any

from numpy import array, diag, eye, inf, isclose, nan, ones, testing, zeros
from numpy.random import default_rng
import pytest

from imu_error_model import AxisConfig, ImuConfig, ImuModel
from imu_error_model.kinematics import rotation_vector_from_matrix, validate_orientation
from imu_error_model.processes import (
    FlickerBiasProcess,
    flicker_transition_parameters,
    gauss_markov,
    gauss_markov_parameters,
    white_noise,
)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"white_noise_density": -1},
        {"turn_on_bias_std": -1},
        {"bias_std": -1},
        {"misalignment_std": -1},
        {"bias_correlation_time": 0},
        {"measurement_range": 0},
        {"quantization_step": 0},
        {"quantization_step": (0.1, 0.0, 0.1)},
        {"bias_std": (-0.1, 0.0, 0.1)},
        {"scale_factor": nan},
        {"scale_factor": (0.0, inf, 0.0)},
        {"nonlinear_factor": inf},
        {"reference_temperature_celsius": nan},
        {"noise_covariance": ((nan, 0, 0), (0, 1, 0), (0, 0, 1))},
        {"misalignment_covariance": ((1, 2, 0), (2, 1, 0), (0, 0, 1))},
    ],
)
def test_axis_configuration_rejects_invalid_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        AxisConfig(**kwargs)
####

def test_configuration_rejects_bad_covariance_and_scale() -> None:
    with pytest.raises(ValueError):
        AxisConfig(noise_covariance=((1, 2, 0), (0, 1, 0), (0, 0, 1)))
    with pytest.raises(ValueError):
        ImuConfig(output_scale_gyroscope=0)
    with pytest.raises(ValueError):
        ImuConfig(output_scale_accelerometer=inf)
    with pytest.raises(ValueError):
        AxisConfig(
            misalignment_std=0.1,
            misalignment_covariance=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
####

def test_processes_handle_zero_dt_and_random_walk() -> None:
    rng = default_rng(8)
    testing.assert_array_equal(white_noise(1, 0, rng), zeros(3))
    initial = ones(3)
    testing.assert_array_equal(gauss_markov(initial, 1, 10, 0, rng), initial)
    result = gauss_markov(initial, 1, None, 1, rng)
    assert result.shape == (3,)
####

def test_process_transition_variances_remain_resolved_for_short_intervals() -> None:
    _, variance = gauss_markov_parameters(0.3, 10.0, 1e-12)
    testing.assert_allclose(variance, 0.3 ** 2 * 2e-12 / 10.0, rtol=1e-12)

    _, flicker_variance = flicker_transition_parameters(
        array([10.0]), array([0.3]), 1e-12
    )
    testing.assert_allclose(flicker_variance, [0.3 ** 2 * 2e-12 / 10.0], rtol=1e-12)
####

def test_flicker_transitions_are_cached_per_sample_interval() -> None:
    process = FlickerBiasProcess(
        AxisConfig(
            flicker_bias_std=0.1,
            flicker_min_correlation_time=1.0,
            flicker_max_correlation_time=10.0,
        )
    )
    rng = default_rng(13)

    process.step(0.01, rng)
    first_phi = process._last_phi
    first_innovation_std = process._last_innovation_std
    process.step(0.01, rng)
    assert process._last_phi is first_phi
    assert process._last_innovation_std is first_innovation_std

    process.step(0.02, rng)
    assert process._last_phi is not first_phi
####

def test_kinematics_validates_and_handles_identity() -> None:
    testing.assert_array_equal(rotation_vector_from_matrix(eye(3)), zeros(3))
    with pytest.raises(ValueError):
        validate_orientation(eye(2))
    with pytest.raises(ValueError):
        validate_orientation(diag([1.0, 1.0, -1.0]))
####

def test_model_rejects_bad_truth_inputs_and_can_reset() -> None:
    model = ImuModel()
    with pytest.raises(ValueError):
        model.measure(0, array([nan, 0, 0]), eye(3))
    with pytest.raises(ValueError):
        model.measure(0, zeros(3), eye(3), temperature_celsius=inf)
    model.measure(0, zeros(3), eye(3))
    model.reset()
    assert isclose(model.measure(0, zeros(3), eye(3)).dt, 0.0)
####
