from typing import Any, cast

import pytest
from numpy import array, block, cov, diag, exp, eye, full, hstack, inf, nan, testing, zeros
from numpy.random import default_rng

from imu_error_model import AxisConfig, ImuConfig, characterize_imu_error_process


def test_characterization_is_deterministic_and_does_not_mutate_config() -> None:
    config = ImuConfig(
        accelerometer=AxisConfig(
            white_noise_density=0.5,
            turn_on_bias_std=0.2,
            bias_std=0.3,
            bias_correlation_time=2.0,
        )
    )
    before = config.model_dump()

    first = characterize_imu_error_process(config, dt=0.5, process_age_at_interval_start_s=3.0)
    second = characterize_imu_error_process(config, dt=0.5, process_age_at_interval_start_s=3.0)

    assert config.model_dump() == before
    testing.assert_array_equal(
        first.conditional_increment_covariance,
        second.conditional_increment_covariance,
    )
    testing.assert_array_equal(
        first.accelerometer.persistent.prior_covariance,
        second.accelerometer.persistent.prior_covariance,
    )
####

def test_characterization_reports_white_noise_in_rate_and_increment_forms() -> None:
    config = ImuConfig(
        accelerometer=AxisConfig(white_noise_density=0.5, output_scale=1.5),
        output_scale_accelerometer=2.0,
    )
    estimate = characterize_imu_error_process(config, dt=0.2)

    # Q_c = 0.5^2 I; the total output scale is 1.5 * 2.0.
    expected_rate = eye(3) * 0.25 / 0.2
    expected_increment = eye(3) * 0.2 ** 2 * 3.0 ** 2 * (0.25 / 0.2)
    testing.assert_allclose(
        estimate.accelerometer.continuous_white_noise_covariance,
        eye(3) * 0.25,
    )
    testing.assert_allclose(estimate.accelerometer.interval_rate_covariance, expected_rate)
    testing.assert_allclose(estimate.accelerometer.direct_increment_covariance, expected_increment)
    testing.assert_allclose(
        estimate.accelerometer.conditional_increment_covariance,
        expected_increment,
    )
    testing.assert_allclose(
        estimate.conditional_increment_covariance[:3, :3],
        expected_increment,
    )
    testing.assert_allclose(estimate.conditional_increment_covariance[3:, 3:], zeros((3, 3)))
####

def test_characterization_exactly_discretizes_gauss_markov_and_random_walk_priors() -> None:
    gauss_markov_config = ImuConfig(
        accelerometer=AxisConfig(
            turn_on_bias_std=0.2,
            bias_std=0.3,
            bias_correlation_time=2.0,
        )
    )
    estimate = characterize_imu_error_process(
        gauss_markov_config,
        dt=0.5,
        process_age_at_interval_start_s=3.0,
    ).accelerometer.persistent
    phi = exp(-0.5 / 2.0)
    q = 0.3 ** 2 * (1.0 - phi ** 2)
    testing.assert_allclose(estimate.bias_transition, eye(3) * phi)
    testing.assert_allclose(estimate.bias_process_covariance, eye(3) * q)
    testing.assert_allclose(estimate.initial_bias_covariance, eye(3) * (0.2 ** 2 + 0.3 ** 2))
    testing.assert_allclose(estimate.prior_bias_covariance, eye(3) * (0.2 ** 2 + 0.3 ** 2))

    random_walk_config = ImuConfig(
        gyroscope=AxisConfig(bias_std=0.3),
    )
    random_walk = characterize_imu_error_process(
        random_walk_config,
        dt=0.5,
        process_age_at_interval_start_s=3.0,
    ).gyroscope.persistent
    testing.assert_array_equal(random_walk.bias_transition, eye(3))
    testing.assert_allclose(random_walk.bias_process_covariance, eye(3) * 0.3 ** 2 * 0.5)
    # Random-walk density and initial covariance are distinct quantities. The
    # current schema starts the random-walk state at zero.
    testing.assert_allclose(random_walk.initial_bias_covariance, zeros((3, 3)))
    testing.assert_allclose(random_walk.prior_bias_covariance, eye(3) * 0.3 ** 2 * 3.5)
####

def test_characterization_preserves_axis_specific_bias_statistics() -> None:
    config = ImuConfig(
        accelerometer=AxisConfig(
            turn_on_bias_std=(0.1, 0.2, 0.3),
            bias_std=(0.01, 0.02, 0.03),
        )
    )
    persistent = characterize_imu_error_process(config, dt=0.5).accelerometer.persistent
    testing.assert_allclose(
        persistent.initial_bias_covariance,
        diag([0.1 ** 2, 0.2 ** 2, 0.3 ** 2]),
    )
    testing.assert_allclose(
        persistent.bias_process_covariance,
        diag([0.01 ** 2, 0.02 ** 2, 0.03 ** 2]) * 0.5,
    )
####

def test_characterization_exposes_flicker_blocks_and_marginal_covariance() -> None:
    config = ImuConfig(
        accelerometer=AxisConfig(
            flicker_bias_std=0.1,
            flicker_min_correlation_time=1.0,
            flicker_max_correlation_time=100.0,
            flicker_components=3,
        )
    )
    estimate = characterize_imu_error_process(config, dt=0.25, process_age_at_interval_start_s=2.0)
    persistent = estimate.accelerometer.persistent

    assert persistent.state_block_labels == (
        "flicker_0",
        "flicker_1",
        "flicker_2",
    )
    assert persistent.transition.shape == (9, 9)
    assert persistent.measurement_matrix.shape == (3, 9)
    testing.assert_allclose(
        estimate.accelerometer.marginal_increment_covariance,
        estimate.accelerometer.direct_increment_covariance
        + 0.25 ** 2 * persistent.measurement_matrix @ persistent.end_covariance @ persistent.measurement_matrix.T,
    )
    assert any("Marginal" in message for message in estimate.metadata)
####

def test_characterization_reports_end_prior_and_increment_innovation_cross_covariance() -> None:
    config = ImuConfig(
        accelerometer=AxisConfig(
            bias_std=0.4,
            turn_on_bias_std=0.1,
            bias_correlation_time=None,
        )
    )
    estimate = characterize_imu_error_process(
        config,
        dt=0.5,
        process_age_at_interval_start_s=2.0,
    ).accelerometer
    persistent = estimate.persistent
    b = 0.5 * persistent.measurement_matrix

    testing.assert_allclose(
        persistent.prior_bias_covariance,
        eye(3) * (0.1 ** 2 + 0.4 ** 2 * 2.5),
    )
    testing.assert_allclose(
        estimate.increment_state_process_cross_covariance,
        b @ persistent.process_covariance,
    )
    testing.assert_allclose(
        estimate.conditional_increment_covariance,
        estimate.direct_increment_covariance + b @ persistent.process_covariance @ b.T,
    )
####

def test_zero_persistent_configuration_has_zero_dimensional_state() -> None:
    estimate = characterize_imu_error_process(ImuConfig(), dt=0.1)
    persistent = estimate.accelerometer.persistent
    assert persistent.state_block_labels == ()
    assert persistent.state_block_slices == ()
    assert persistent.transition.shape == (0, 0)
    assert persistent.process_covariance.shape == (0, 0)
    assert persistent.measurement_matrix.shape == (3, 0)
    assert estimate.increment_state_process_cross_covariance.shape == (6, 0)
####

def test_one_interval_joint_covariance_matches_monte_carlo() -> None:
    config = ImuConfig(
        accelerometer=AxisConfig(bias_std=0.2),
    )
    estimate = characterize_imu_error_process(config, dt=0.25).accelerometer
    persistent = estimate.persistent
    state_to_increment = 0.25 * persistent.measurement_matrix
    joint = block([
        [estimate.conditional_increment_covariance, estimate.increment_state_process_cross_covariance],
        [estimate.increment_state_process_cross_covariance.T, persistent.process_covariance],
    ])

    rng = default_rng(23)
    samples = 100_000
    innovations = rng.multivariate_normal(
        zeros(persistent.process_covariance.shape[0]),
        persistent.process_covariance,
        size=samples,
    )
    fresh = rng.multivariate_normal(
        zeros(3),
        estimate.direct_increment_covariance,
        size=samples,
    )
    increments = fresh + innovations @ state_to_increment.T
    empirical = cov(hstack((increments, innovations)), rowvar=False, bias=True)
    testing.assert_allclose(empirical, joint, rtol=0.03, atol=1e-4)
####

def test_characterization_marks_thermal_and_quantization_assumptions() -> None:
    config = ImuConfig(
        accelerometer=AxisConfig(
            white_noise_density=0.5,
            thermal_bias_coefficient=0.2,
            thermal_noise_coefficient=0.1,
            quantization_step=0.2,
        )
    )
    estimate = characterize_imu_error_process(
        config,
        dt=0.2,
        temperature_celsius=35.0,
        include_quantization_approximation=True,
    ).accelerometer

    testing.assert_allclose(estimate.deterministic_rate_offset, full(3, 2.0))
    testing.assert_allclose(estimate.deterministic_increment_mean, full(3, 0.4))
    testing.assert_allclose(
        estimate.interval_rate_covariance,
        eye(3) * (2.0 ** 2 * 0.5 ** 2 / 0.2),
    )
    testing.assert_allclose(
        estimate.conditional_increment_covariance - estimate.direct_increment_covariance,
        eye(3) * 0.2 ** 2 * (0.2 ** 2 / 12.0),
    )
    assert any("high-resolution" in message for message in estimate.metadata)
    assert any("custom ThermalModel" in message for message in estimate.metadata)
####

def test_characterization_uses_axis_specific_quantization_approximation() -> None:
    config = ImuConfig(
        accelerometer=AxisConfig(quantization_step=(0.1, 0.2, 0.3)),
    )
    estimate = characterize_imu_error_process(
        config,
        dt=0.5,
        include_quantization_approximation=True,
    ).accelerometer
    expected = diag(array([0.1, 0.2, 0.3]) ** 2 / 12.0) * 0.5 ** 2
    testing.assert_allclose(
        estimate.conditional_increment_covariance - estimate.direct_increment_covariance,
        expected,
    )

    without_temperature = characterize_imu_error_process(config, dt=0.2).accelerometer
    testing.assert_array_equal(without_temperature.deterministic_rate_offset, zeros(3))
    assert any("temperature is unavailable" in message for message in without_temperature.metadata)
####

@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": 0.0},
        {"dt": -1.0},
        {"dt": inf},
        {"process_age_at_interval_start_s": -1.0},
        {"temperature_celsius": nan},
    ],
)
def test_characterization_rejects_invalid_independent_inputs(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        inputs = {"dt": 1.0, **kwargs}
        characterize_imu_error_process(ImuConfig(), **cast(Any, inputs))
    ####
####
