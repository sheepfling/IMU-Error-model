from numpy import abs as np_abs

from imu_error_model import AxisConfig, ImuConfig, ImuModel
from playground.calibration.known_reference import simulate_reference_program
from playground.calibration.thermal import (
    expected_gyroscope_specific_force_thermal,
    expected_thermal_axis,
    fit_accelerometer_thermal_model,
    fit_gyroscope_specific_force_thermal_model,
    thermal_reference_program,
)


def test_full_cross_axis_thermal_matrix_is_recoverable() -> None:
    accelerometer = AxisConfig(
        fixed_bias=(0.02, -0.01, 0.03),
        sensitivity_matrix=((1.01, 0.002, -0.001), (-0.003, 0.99, 0.004), (0.002, -0.001, 1.005)),
        thermal_scale_factor_coefficient=(10e-6, -20e-6, 15e-6),
        thermal_sensitivity_matrix_coefficient=(
            (2e-6, -3e-6, 1e-6),
            (4e-6, -1e-6, 2e-6),
            (-2e-6, 3e-6, 1e-6),
        ),
        thermal_bias_coefficient=(2e-4, -3e-4, 1e-4),
    )
    dataset = simulate_reference_program(
        ImuModel(ImuConfig(accelerometer=accelerometer)),
        thermal_reference_program(sample_count=4),
    )
    estimate = fit_accelerometer_thermal_model(dataset, accelerometer)
    truth = expected_thermal_axis(accelerometer)

    assert float(np_abs(estimate.thermal_matrix - truth.thermal_matrix).max()) < 1e-12
    assert float(np_abs(estimate.base_matrix - truth.base_matrix).max()) < 1e-12
####


def test_temperature_dependent_gyro_g_sensitivity_is_recoverable() -> None:
    config = ImuConfig(
        gyroscope=AxisConfig(
            fixed_bias=(1e-5, -2e-5, 1.5e-5),
            thermal_bias_coefficient=(2e-7, -3e-7, 1e-7),
        ),
        gyroscope_specific_force_sensitivity=(
            (2e-7, -1e-7, 0.5e-7),
            (-0.5e-7, 1e-7, 1.5e-7),
            (0.25e-7, -0.5e-7, 0.75e-7),
        ),
        gyroscope_specific_force_sensitivity_temperature_coefficient=(
            (2e-9, -1e-9, 0.5e-9),
            (-0.5e-9, 1e-9, 1.5e-9),
            (0.25e-9, -0.5e-9, 0.75e-9),
        ),
    )
    dataset = simulate_reference_program(
        ImuModel(config),
        thermal_reference_program(sample_count=4),
    )
    estimate = fit_gyroscope_specific_force_thermal_model(dataset, config)
    truth = expected_gyroscope_specific_force_thermal(config)

    assert float(np_abs(estimate.base_matrix - truth.base_matrix).max()) < 1e-12
    assert float(np_abs(estimate.thermal_matrix - truth.thermal_matrix).max()) < 1e-12
    assert float(np_abs(estimate.base_bias - truth.base_bias).max()) < 1e-12
    assert float(np_abs(estimate.thermal_bias - truth.thermal_bias).max()) < 1e-12
####
