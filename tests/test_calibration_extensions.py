import pytest
from numpy import array, eye, testing, zeros
from numpy.random import default_rng

from imu_error_model import AxisConfig, ImuConfig, ImuModel


def test_fixed_intrinsics_and_gyro_specific_force_sensitivity() -> None:
    accelerometer_matrix = (
        (1.02, 0.01, -0.02),
        (-0.01, 0.98, 0.03),
        (0.02, -0.01, 1.01),
    )
    gyroscope_matrix = (
        (0.99, 0.02, 0.01),
        (-0.01, 1.03, -0.02),
        (0.01, 0.00, 1.01),
    )
    gyro_specific_force_sensitivity = (
        (1.0e-3, -2.0e-3, 0.5e-3),
        (0.2e-3, 0.8e-3, -1.0e-3),
        (-0.4e-3, 0.3e-3, 0.7e-3),
    )
    accelerometer_bias = (0.1, -0.2, 0.3)
    gyroscope_bias = (0.01, -0.02, 0.03)
    specific_force = array([2.0, -3.0, 4.0])
    angular_rate = array([0.4, -0.5, 0.6])
    model = ImuModel(
        ImuConfig(
            accelerometer=AxisConfig(
                fixed_bias=accelerometer_bias,
                sensitivity_matrix=accelerometer_matrix,
            ),
            gyroscope=AxisConfig(
                fixed_bias=gyroscope_bias,
                sensitivity_matrix=gyroscope_matrix,
            ),
            gyroscope_specific_force_sensitivity=gyro_specific_force_sensitivity,
        ),
        rng=default_rng(1),
    )

    output = model.measure_ideal_rates(
        start_time=0.0,
        end_time=0.25,
        specific_force_body=specific_force,
        angular_rate_body=angular_rate,
    )

    expected_acceleration = array(accelerometer_matrix) @ specific_force + array(accelerometer_bias)
    expected_angular_rate = (
        array(gyroscope_matrix) @ angular_rate
        + array(gyroscope_bias)
        + array(gyro_specific_force_sensitivity) @ specific_force
    )
    testing.assert_allclose(output.acceleration, expected_acceleration)
    testing.assert_allclose(output.angular_rate, expected_angular_rate)
####


def test_ideal_increment_and_rate_entry_points_are_equivalent() -> None:
    config = ImuConfig(
        accelerometer=AxisConfig(fixed_bias=(0.1, 0.2, 0.3)),
        gyroscope=AxisConfig(fixed_bias=(0.01, 0.02, 0.03)),
    )
    rate_model = ImuModel(config, rng=default_rng(3))
    increment_model = ImuModel(config, rng=default_rng(3))
    specific_force = array([1.0, 2.0, 3.0])
    angular_rate = array([0.4, 0.5, 0.6])
    dt = 0.2

    rate_output = rate_model.measure_ideal_rates(
        start_time=0.0,
        end_time=dt,
        specific_force_body=specific_force,
        angular_rate_body=angular_rate,
    )
    increment_output = increment_model.measure_ideal_increments(
        start_time=0.0,
        end_time=dt,
        delta_v_body=specific_force * dt,
        delta_theta_body=angular_rate * dt,
    )

    testing.assert_allclose(increment_output.delta_v, rate_output.delta_v)
    testing.assert_allclose(increment_output.delta_theta, rate_output.delta_theta)
####


def test_ideal_intervals_must_be_contiguous_until_reset() -> None:
    model = ImuModel()
    model.measure_ideal_rates(
        start_time=0.0,
        end_time=0.1,
        specific_force_body=zeros(3),
        angular_rate_body=zeros(3),
    )
    with pytest.raises(ValueError, match="previous measurement boundary"):
        model.measure_ideal_rates(
            start_time=0.2,
            end_time=0.3,
            specific_force_body=zeros(3),
            angular_rate_body=zeros(3),
        )
    ####

    model.reset()
    model.measure_ideal_rates(
        start_time=0.2,
        end_time=0.3,
        specific_force_body=zeros(3),
        angular_rate_body=zeros(3),
    )
####


def test_truth_measurement_rebaselines_after_direct_ideal_input() -> None:
    model = ImuModel()
    model.measure_ideal_rates(
        start_time=0.0,
        end_time=1.0,
        specific_force_body=zeros(3),
        angular_rate_body=zeros(3),
    )

    baseline = model.measure(1.0, zeros(3), eye(3))
    testing.assert_allclose(baseline.delta_v, zeros(3))
    testing.assert_allclose(baseline.delta_theta, zeros(3))

    output = model.measure(2.0, array([1.0, 0.0, 0.0]), eye(3))
    testing.assert_allclose(output.delta_v, [1.0, 0.0, 0.0])
####


@pytest.mark.parametrize(
    "field_name",
    ("sensitivity_matrix", "gyroscope_specific_force_sensitivity"),
)
def test_intrinsic_matrices_reject_invalid_shape(field_name: str) -> None:
    invalid = ((1.0, 0.0), (0.0, 1.0))
    with pytest.raises(ValueError, match="finite 3x3 matrix"):
        if field_name == "sensitivity_matrix":
            AxisConfig(sensitivity_matrix=invalid)  # type: ignore[arg-type]
        else:
            ImuConfig(gyroscope_specific_force_sensitivity=invalid)  # type: ignore[arg-type]
        ####
    ####
####
