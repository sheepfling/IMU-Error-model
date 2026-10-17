"""Run the offline Ferraris-equation parity experiment."""

from __future__ import annotations

import json
from math import pi

from numpy import abs as absolute

from imu_error_model import ImuModel

from .imucal_adapter import (
    ferraris_imucal_program,
    fit_ferraris_equation_reference,
    imucal_parameter_error,
    write_imucal_csv,
)
from .known_reference import simulate_reference_program
from .run_imucal import OUTPUT_DIRECTORY, SAMPLING_RATE_HZ, _reference_config


IMUCAL_REFERENCE_TAG = "v2.6.0"
IMUCAL_REFERENCE_COMMIT = "51bbeae34efbe90401a143d719600532e59afd60"


def main() -> int:
    """Generate a nine-region corpus and evaluate source-aligned equations."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = _reference_config()
    dataset = simulate_reference_program(
        ImuModel(config),
        ferraris_imucal_program(
            sampling_rate_hz=SAMPLING_RATE_HZ,
            static_duration_s=0.5,
            nominal_rotation_rate_radps=2.0 * pi,
            expected_angle_degrees=-360.0,
        ),
    )
    corpus_path = OUTPUT_DIRECTORY / "imucal_ferraris_noiseless.csv"
    write_imucal_csv(dataset, corpus_path)
    parameters = fit_ferraris_equation_reference(
        dataset,
        sampling_rate_hz=SAMPLING_RATE_HZ,
    )
    error = imucal_parameter_error(parameters, config)
    maximum_absolute_error = float(absolute(error).max())
    validation_path = OUTPUT_DIRECTORY / "imucal_ferraris_equation_validation.json"
    region_counts = {
        label: sum(sample_label == label for sample_label in dataset.labels)
        for label in dict.fromkeys(dataset.labels)
    }
    validation_path.write_text(
        json.dumps(
            {
                "reference_package": "imucal",
                "reference_tag": IMUCAL_REFERENCE_TAG,
                "reference_commit": IMUCAL_REFERENCE_COMMIT,
                "validation_mode": "source-aligned Ferraris equations; external package not imported",
                "sampling_rate_hz": SAMPLING_RATE_HZ,
                "sample_count": len(dataset.labels),
                "region_counts": region_counts,
                "maximum_absolute_parameter_error": maximum_absolute_error,
                "acceptance_tolerance": 1.0e-10,
                "passed": maximum_absolute_error < 1.0e-10,
                "corpus": str(corpus_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"samples: {len(dataset.labels)}")
    print(f"maximum parameter error: {maximum_absolute_error:.3e}")
    print(f"corpus: {corpus_path}")
    print(f"validation: {validation_path}")
    return 0 if maximum_absolute_error < 1.0e-10 else 1
####


if __name__ == "__main__":
    raise SystemExit(main())
####
