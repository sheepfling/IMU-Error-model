from pathlib import Path

import pytest
from numpy import abs as absolute, array, isfinite, testing, zeros

from imu_error_model import AxisConfig, ImuConfig, ImuModel
from playground.calibration.known_reference import (
    CalibrationDataset,
    calibration_parameter_error,
    ferraris_style_reference_program,
    fit_known_reference_calibration,
    simulate_reference_program,
)


def _calibration_config() -> ImuConfig:
    return ImuConfig(
        accelerometer=AxisConfig(
            fixed_bias=(0.08, -0.05, 0.03),
            scale_factor=(0.002, -0.003, 0.001),
            sensitivity_matrix=(
                (1.012, 0.004, -0.003),
                (-0.002, 0.994, 0.006),
                (0.003, -0.005, 1.008),
            ),
        ),
        gyroscope=AxisConfig(
            fixed_bias=(0.001, -0.002, 0.0015),
            scale_factor=(-0.001, 0.002, 0.001),
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


def test_known_reference_playground_recovers_injected_intrinsics(tmp_path: Path) -> None:
    config = _calibration_config()
    dataset = simulate_reference_program(
        ImuModel(config),
        ferraris_style_reference_program(duration_s=0.25),
    )
    estimate = fit_known_reference_calibration(dataset)
    error = calibration_parameter_error(estimate, config)

    assert estimate.accelerometer_design_rank == 4
    assert estimate.gyroscope_design_rank == 7
    assert isfinite(estimate.accelerometer_condition_number)
    assert isfinite(estimate.gyroscope_condition_number)
    assert absolute(error).max() < 1e-12
    assert estimate.accelerometer_residual_rms < 1e-12
    assert estimate.gyroscope_residual_rms < 1e-12

    csv_path = tmp_path / "known_reference.csv"
    dataset.to_csv(csv_path)
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(dataset.labels) + 1
    assert "measured_angular_rate_z_radps" in lines[0]
####


def test_reference_program_contains_six_static_poses_and_six_rotations() -> None:
    labels = tuple(segment.label for segment in ferraris_style_reference_program())
    assert len(labels) == 13
    assert sum(label.startswith("static_") for label in labels) == 6
    assert sum(label.startswith("rotation_") for label in labels) == 6
####


def test_rank_deficient_dataset_is_rejected() -> None:
    dataset = CalibrationDataset(
        labels=("only",),
        start_times=array([0.0]),
        end_times=array([1.0]),
        temperatures_celsius=array([float("nan")]),
        specific_force_body=zeros((1, 3)),
        angular_rate_body=zeros((1, 3)),
        measured_acceleration=zeros((1, 3)),
        measured_angular_rate=zeros((1, 3)),
    )
    with pytest.raises(ValueError, match="accelerometer reference program is rank deficient"):
        fit_known_reference_calibration(dataset)
    ####
####


def test_calibration_dataset_arrays_are_read_only() -> None:
    dataset = simulate_reference_program(
        ImuModel(),
        ferraris_style_reference_program(),
    )
    assert not dataset.specific_force_body.flags.writeable
    with pytest.raises(ValueError):
        dataset.specific_force_body[0, 0] = 1.0
    ####
    testing.assert_allclose(dataset.measured_acceleration, dataset.specific_force_body)
####
