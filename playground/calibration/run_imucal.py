"""Generate and validate a noiseless Ferraris corpus with optional imucal."""

from __future__ import annotations

import json
from math import pi
from pathlib import Path

from numpy import abs as absolute

from imu_error_model import AxisConfig, ImuConfig, ImuModel

from .imucal_adapter import (
    compute_imucal_ferraris,
    ferraris_imucal_program,
    imucal_parameter_error,
    imucal_to_forward_parameters,
    write_imucal_csv,
)
from .known_reference import simulate_reference_program


SAMPLING_RATE_HZ = 100.0
OUTPUT_DIRECTORY = Path("artifacts/calibration")


def _reference_config() -> ImuConfig:
    """Return an imucal-compatible deterministic sensor with known truth."""
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


def main() -> int:
    """Write the corpus, run imucal, and record truth-recovery diagnostics."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = _reference_config()
    program = ferraris_imucal_program(
        sampling_rate_hz=SAMPLING_RATE_HZ,
        static_duration_s=0.5,
        nominal_rotation_rate_radps=2.0 * pi,
        expected_angle_degrees=-360.0,
    )
    dataset = simulate_reference_program(ImuModel(config), program)
    corpus_path = OUTPUT_DIRECTORY / "imucal_ferraris_noiseless.csv"
    write_imucal_csv(dataset, corpus_path)

    try:
        calibration = compute_imucal_ferraris(
            dataset,
            sampling_rate_hz=SAMPLING_RATE_HZ,
        )
    except RuntimeError as error:
        print(f"wrote imucal corpus: {corpus_path}")
        print(error)
        return 2
    ####

    calibration_path = OUTPUT_DIRECTORY / "imucal_ferraris_calibration.json"
    calibration.to_json_file(calibration_path)
    parameters = imucal_to_forward_parameters(calibration)
    error = imucal_parameter_error(parameters, config)
    maximum_absolute_error = float(absolute(error).max())
    validation_path = OUTPUT_DIRECTORY / "imucal_ferraris_validation.json"
    validation_path.write_text(
        json.dumps(
            {
                "sampling_rate_hz": SAMPLING_RATE_HZ,
                "sample_count": len(dataset.labels),
                "maximum_absolute_parameter_error": maximum_absolute_error,
                "acceptance_tolerance": 1.0e-10,
                "passed": maximum_absolute_error < 1.0e-10,
                "corpus": str(corpus_path),
                "calibration": str(calibration_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"samples: {len(dataset.labels)}")
    print(f"maximum parameter error: {maximum_absolute_error:.3e}")
    print(f"corpus: {corpus_path}")
    print(f"calibration: {calibration_path}")
    print(f"validation: {validation_path}")
    return 0 if maximum_absolute_error < 1.0e-10 else 1
####


if __name__ == "__main__":
    raise SystemExit(main())
####
