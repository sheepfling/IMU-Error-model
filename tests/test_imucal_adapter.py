from types import SimpleNamespace

import pytest
from numpy import abs as absolute, array, diag, eye, pi, testing

from imu_error_model import AxisConfig, ImuConfig, ImuModel
from playground.calibration.imucal_adapter import (
    compute_imucal_ferraris,
    ferraris_imucal_program,
    fit_ferraris_equation_reference,
    imucal_parameter_error,
    imucal_to_forward_parameters,
    to_imucal_dataframe,
)
from playground.calibration.known_reference import simulate_reference_program


def _imucal_compatible_config() -> ImuConfig:
    return ImuConfig(
        accelerometer=AxisConfig(
            fixed_bias=(0.08, -0.05, 0.03),
            sensitivity_matrix=(
                (1.012, 0.004, -0.003),
                (-0.002, 0.994, 0.006),
                (0.003, -0.005, 1.008),
            ),
        ),
        gyroscope=AxisConfig(
            fixed_bias=(0.001, -0.002, 0.0015),
            sensitivity_matrix=(
                (0.997, 0.003, -0.002),
                (-0.001, 1.006, 0.004),
                (0.002, -0.003, 1.003),
            ),
        ),
        gyroscope_specific_force_sensitivity=(
            (1.0e-4, -2.0e-4, 0.5e-4),
            (0.2e-4, 0.8e-4, -1.0e-4),
            (-0.4e-4, 0.3e-4, 0.7e-4),
        ),
    )
####


def test_imucal_program_has_nine_regions_and_exact_integrated_angles() -> None:
    sampling_rate_hz = 20.0
    program = ferraris_imucal_program(
        sampling_rate_hz=sampling_rate_hz,
        static_duration_s=0.5,
        nominal_rotation_rate_radps=pi,
        expected_angle_degrees=-360.0,
    )
    assert tuple(segment.label for segment in program) == (
        "x_p",
        "x_a",
        "y_p",
        "y_a",
        "z_p",
        "z_a",
        "x_rot",
        "y_rot",
        "z_rot",
    )
    assert all(segment.sample_count == 10 for segment in program[:6])

    dataset = simulate_reference_program(ImuModel(), program)
    frame = to_imucal_dataframe(dataset)
    assert frame.index.name == "part"
    assert tuple(frame.columns) == ("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z")
    for axis_name in ("x", "y", "z"):
        rotation = frame.loc[f"{axis_name}_rot", f"gyr_{axis_name}"]
        integrated_degrees = float(rotation.sum() / sampling_rate_hz)
        assert integrated_degrees == pytest.approx(-360.0)
    ####
####


def test_imucal_parameter_conversion_reconstructs_forward_matrices() -> None:
    calibration = SimpleNamespace(
        K_a=diag([1.1, 0.9, 1.2]),
        R_a=array(
            [
                [1.0, 0.01, -0.02],
                [-0.01, 1.0, 0.03],
                [0.02, -0.03, 1.0],
            ]
        ),
        b_a=array([0.1, -0.2, 0.3]),
        K_g=diag([0.95, 1.05, 1.01]),
        R_g=eye(3),
        K_ga=array(
            [
                [0.1, -0.2, 0.05],
                [0.02, 0.08, -0.1],
                [-0.04, 0.03, 0.07],
            ]
        ),
        b_g=array([0.5, -0.25, 0.1]),
    )
    parameters = imucal_to_forward_parameters(calibration)

    testing.assert_allclose(parameters.accelerometer_sensitivity_matrix, calibration.K_a @ calibration.R_a)
    testing.assert_allclose(parameters.gyroscope_sensitivity_matrix, calibration.K_g @ calibration.R_g)
    testing.assert_allclose(parameters.gyroscope_specific_force_sensitivity, calibration.K_ga * pi / 180.0)
    testing.assert_allclose(parameters.gyroscope_bias_radps, calibration.b_g * pi / 180.0)
####


def test_ferraris_equation_reference_recovers_noiseless_synthetic_parameters() -> None:
    sampling_rate_hz = 100.0
    config = _imucal_compatible_config()
    dataset = simulate_reference_program(
        ImuModel(config),
        ferraris_imucal_program(
            sampling_rate_hz=sampling_rate_hz,
            static_duration_s=0.2,
            nominal_rotation_rate_radps=2.0 * pi,
            expected_angle_degrees=-360.0,
        ),
    )
    parameters = fit_ferraris_equation_reference(
        dataset,
        sampling_rate_hz=sampling_rate_hz,
    )
    error = imucal_parameter_error(parameters, config)
    assert absolute(error).max() < 1e-10
####


def test_actual_imucal_recovers_noiseless_synthetic_parameters_when_installed() -> None:
    pytest.importorskip("imucal")
    sampling_rate_hz = 100.0
    config = _imucal_compatible_config()
    dataset = simulate_reference_program(
        ImuModel(config),
        ferraris_imucal_program(
            sampling_rate_hz=sampling_rate_hz,
            static_duration_s=0.2,
            nominal_rotation_rate_radps=2.0 * pi,
            expected_angle_degrees=-360.0,
        ),
    )
    result = compute_imucal_ferraris(dataset, sampling_rate_hz=sampling_rate_hz)
    parameters = imucal_to_forward_parameters(result)
    error = imucal_parameter_error(parameters, config)
    assert absolute(error).max() < 1e-10
####
