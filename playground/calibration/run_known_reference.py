"""Run the first noiseless calibration-recovery vertical slice."""

from pathlib import Path

from numpy import abs as absolute

from imu_error_model import AxisConfig, ImuConfig, ImuModel
from playground.calibration.known_reference import (
    calibration_parameter_error,
    ferraris_style_reference_program,
    fit_known_reference_calibration,
    simulate_reference_program,
)


def main() -> None:
    config = ImuConfig(
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
    dataset = simulate_reference_program(
        ImuModel(config),
        ferraris_style_reference_program(),
    )
    estimate = fit_known_reference_calibration(dataset)
    error = calibration_parameter_error(estimate, config)
    output_path = Path("artifacts/calibration/known_reference_noiseless.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path)

    print("Known-reference calibration recovery")
    print(f"  samples: {len(dataset.labels)}")
    print(f"  accelerometer design rank: {estimate.accelerometer_design_rank}")
    print(f"  gyroscope design rank: {estimate.gyroscope_design_rank}")
    print(f"  maximum absolute parameter error: {absolute(error).max():.3e}")
    print(f"  accelerometer residual RMS: {estimate.accelerometer_residual_rms:.3e}")
    print(f"  gyroscope residual RMS: {estimate.gyroscope_residual_rms:.3e}")
    print(f"  wrote: {output_path}")
####


if __name__ == "__main__":
    main()
####
