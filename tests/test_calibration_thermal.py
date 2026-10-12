from numpy import array, zeros

from imu_error_model import AxisConfig, ImuConfig, ImuModel
from playground.calibration.known_reference import (
    ferraris_style_reference_program,
    fit_known_reference_calibration,
    simulate_reference_program,
)
from playground.calibration.thermal import (
    correct_thermal_axis,
    expected_thermal_axis,
    fit_accelerometer_thermal_model,
    thermal_reference_program,
)


def test_thermal_sweep_recovers_exact_accelerometer_model() -> None:
    accel = AxisConfig(
        fixed_bias=(0.08, -0.05, 0.03),
        sensitivity_matrix=(
            (1.012, 0.004, -0.002),
            (-0.003, 0.993, 0.005),
            (0.001, -0.004, 1.007),
        ),
        thermal_bias_coefficient=(2e-4, -1e-4, 3e-4),
        thermal_scale_factor_coefficient=(8e-6, -4e-6, 6e-6),
    )
    config = ImuConfig(accelerometer=accel)
    dataset = simulate_reference_program(ImuModel(config=config), thermal_reference_program())

    estimate = fit_accelerometer_thermal_model(dataset, accel)
    expected = expected_thermal_axis(accel)

    assert abs(estimate.base_matrix - expected.base_matrix).max() < 1e-12
    assert abs(estimate.thermal_matrix - expected.thermal_matrix).max() < 1e-12
    assert abs(estimate.base_bias - expected.base_bias).max() < 1e-12
    assert abs(estimate.thermal_bias - expected.thermal_bias).max() < 1e-12
####


def test_thermal_model_removes_hot_validation_residual_that_room_calibration_leaves() -> None:
    accel = AxisConfig(
        fixed_bias=(0.08, -0.05, 0.03),
        sensitivity_matrix=(
            (1.012, 0.004, -0.002),
            (-0.003, 0.993, 0.005),
            (0.001, -0.004, 1.007),
        ),
        thermal_bias_coefficient=(4e-4, -2e-4, 3e-4),
        thermal_scale_factor_coefficient=(20e-6, -12e-6, 16e-6),
    )
    config = ImuConfig(accelerometer=accel)
    room_program = tuple(
        type(segment)(
            label=segment.label,
            duration_s=segment.duration_s,
            specific_force_body=segment.specific_force_body,
            angular_rate_body=segment.angular_rate_body,
            temperature_celsius=25.0,
            sample_count=segment.sample_count,
        )
        for segment in ferraris_style_reference_program(temperature_celsius=25.0)
    )
    room_estimate = fit_known_reference_calibration(
        simulate_reference_program(ImuModel(config=config), room_program)
    )
    thermal_estimate = fit_accelerometer_thermal_model(
        simulate_reference_program(ImuModel(config=config), thermal_reference_program()),
        accel,
    )

    truth = array((3.0, -4.0, 8.0))
    output = ImuModel(config=config).measure_ideal_rates(
        start_time=0.0,
        end_time=0.01,
        specific_force_body=truth,
        angular_rate_body=zeros(3),
        temperature_celsius=60.0,
    )
    room_corrected = __import__("numpy").linalg.solve(
        room_estimate.accelerometer_sensitivity_matrix,
        output.acceleration - room_estimate.accelerometer_bias,
    )
    thermal_corrected = correct_thermal_axis(output.acceleration, 60.0, thermal_estimate)

    assert abs(room_corrected - truth).max() > 1e-3
    assert abs(thermal_corrected - truth).max() < 1e-11
####
