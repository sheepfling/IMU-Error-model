"""Run Earth-fidelity and data-confidence calibration campaign."""

from __future__ import annotations

import csv
import json
from math import pi
from pathlib import Path

from numpy import abs as np_abs, eye, random

from imu_error_model import AxisConfig, ImuConfig, ImuModel

from .data_confidence import (
    run_heading_confidence_sweep,
    run_joint_data_budget_sweep,
    white_noise_duration_for_p95_heading_error,
    white_noise_heading_sigma_deg,
)
from .earth_reference import LocalEarthReference, normal_gravity_mps2
from .gyrocompass import wrap_angle_rad
from .imucal_adapter import fit_ferraris_equation_reference
from .known_reference import (
    CalibrationEstimate,
    ferraris_style_reference_program,
    fit_known_reference_calibration,
    simulate_reference_program,
)
from .maytag import simulate_maytag_gyrocompass
from .turntable import simulate_wgs84_ferraris_turntable


def _ideal_calibration() -> CalibrationEstimate:
    return fit_known_reference_calibration(
        simulate_reference_program(ImuModel(), ferraris_style_reference_program())
    )
####


def _turntable_error(rate_degps: float, latitude_rad: float) -> float:
    sampling_rate_hz = 5.0
    dataset = simulate_wgs84_ferraris_turntable(
        ImuModel(),
        latitude_rad=latitude_rad,
        sampling_rate_hz=sampling_rate_hz,
        static_duration_s=0.2,
        table_rate_radps=rate_degps * pi / 180.0,
        expected_angle_rad=-2.0 * pi,
        include_earth_rotation=True,
    )
    estimate = fit_ferraris_equation_reference(
        dataset,
        sampling_rate_hz=sampling_rate_hz,
        gravity_mps2=normal_gravity_mps2(latitude_rad),
        expected_angle_degrees=-360.0,
    )
    return float(np_abs(estimate.gyroscope_sensitivity_matrix - eye(3)).max())
####


def main(output_directory: str | Path = "artifacts/calibration/tranche3") -> Path:
    """Write JSON and CSV evidence for Earth fidelity and data requirements."""
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    latitude_rad = 34.7304 * pi / 180.0
    heading_rad = -28.0 * pi / 180.0
    ideal_calibration = _ideal_calibration()

    noise_density = 2e-6
    white_config = ImuConfig(gyroscope=AxisConfig(white_noise_density=noise_density))
    durations = (10.0, 30.0, 60.0, 120.0)
    white_sweep = run_heading_confidence_sweep(
        model_factory=lambda rng: ImuModel(white_config, rng=rng),
        calibration=ideal_calibration,
        latitude_rad=latitude_rad,
        heading_rad=heading_rad,
        durations_s=durations,
        sample_period_s=0.1,
        trial_count=50,
        seed=583,
    )

    colored_config = ImuConfig(
        gyroscope=AxisConfig(
            white_noise_density=noise_density,
            bias_std=5e-6,
            bias_correlation_time=100.0,
        )
    )
    colored_sweep = run_heading_confidence_sweep(
        model_factory=lambda rng: ImuModel(colored_config, rng=rng),
        calibration=ideal_calibration,
        latitude_rad=latitude_rad,
        heading_rad=heading_rad,
        durations_s=durations,
        sample_period_s=0.1,
        trial_count=30,
        seed=584,
    )

    latitude_rows: list[dict[str, float]] = []
    for latitude_deg in (0.0, 35.0, 60.0, 75.0):
        lat_rad = latitude_deg * pi / 180.0
        sweep = run_heading_confidence_sweep(
            model_factory=lambda rng: ImuModel(white_config, rng=rng),
            calibration=ideal_calibration,
            latitude_rad=lat_rad,
            heading_rad=heading_rad,
            durations_s=(60.0,),
            sample_period_s=0.1,
            trial_count=30,
            seed=700 + int(latitude_deg),
        )
        latitude_rows.append(
            {
                "latitude_deg": latitude_deg,
                "horizontal_earth_rate_radps": LocalEarthReference(lat_rad).horizontal_earth_rate_radps,
                "p95_absolute_heading_error_deg": sweep.points[0].p95_absolute_error_deg,
                "standard_deviation_deg": sweep.points[0].standard_deviation_deg,
            }
        )
    ####

    joint_config = ImuConfig(
        accelerometer=AxisConfig(white_noise_density=0.001, fixed_bias=(0.02, -0.01, 0.01)),
        gyroscope=AxisConfig(white_noise_density=2e-6, fixed_bias=(5e-6, -3e-6, 2e-6)),
    )
    budget = run_joint_data_budget_sweep(
        config=joint_config,
        calibration_samples_per_segment=(20, 100, 500),
        compass_durations_s=(30.0, 120.0),
        latitude_rad=latitude_rad,
        heading_rad=heading_rad,
        sample_period_s=0.1,
        trial_count=15,
        seed=900,
    )

    deterministic = ImuConfig(
        accelerometer=AxisConfig(fixed_bias=(0.02, -0.01, 0.01)),
        gyroscope=AxisConfig(fixed_bias=(1e-6, -2e-6, 1e-6)),
    )
    deterministic_calibration = fit_known_reference_calibration(
        simulate_reference_program(ImuModel(deterministic), ferraris_style_reference_program())
    )
    maytag_config = deterministic.model_copy(
        update={
            "gyroscope": deterministic.gyroscope.model_copy(update={"turn_on_bias_std": 2e-5})
        }
    )
    maytag_rows: list[dict[str, float | int]] = []
    for position_count in (1, 2, 4):
        result = simulate_maytag_gyrocompass(
            ImuModel(maytag_config, rng=random.default_rng(42)),
            deterministic_calibration,
            latitude_rad=latitude_rad,
            heading_rad=heading_rad,
            position_count=position_count,
            samples_per_position=100,
            sample_period_s=0.02,
        )
        maytag_rows.append(
            {
                "position_count": position_count,
                "heading_error_deg": abs(wrap_angle_rad(result.heading_rad - heading_rad)) * 180.0 / pi,
                "earth_rate_residual_radps": result.earth_rate_residual_radps,
            }
        )
    ####

    turntable_rows = [
        {
            "table_rate_degps": rate_degps,
            "max_apparent_gyro_matrix_error": _turntable_error(rate_degps, latitude_rad),
        }
        for rate_degps in (90.0, 10.0, 1.0, 0.1)
    ]

    white_rows = []
    for point in white_sweep.points:
        white_rows.append(
            {
                "duration_s": point.duration_s,
                "empirical_std_deg": point.standard_deviation_deg,
                "empirical_p95_abs_deg": point.p95_absolute_error_deg,
                "predicted_white_std_deg": white_noise_heading_sigma_deg(
                    noise_density,
                    averaging_duration_s=point.duration_s,
                    latitude_rad=latitude_rad,
                ),
            }
        )
    ####
    colored_rows = [
        {
            "duration_s": point.duration_s,
            "empirical_std_deg": point.standard_deviation_deg,
            "empirical_p95_abs_deg": point.p95_absolute_error_deg,
        }
        for point in colored_sweep.points
    ]

    payload = {
        "site": {
            "latitude_deg": latitude_rad * 180.0 / pi,
            "normal_gravity_mps2": normal_gravity_mps2(latitude_rad),
            "horizontal_earth_rate_radps": LocalEarthReference(latitude_rad).horizontal_earth_rate_radps,
        },
        "white_noise_heading": white_rows,
        "colored_bias_heading": colored_rows,
        "latitude_sweep_60s": latitude_rows,
        "joint_data_budget": [cell.__dict__ if hasattr(cell, "__dict__") else {
            "calibration_samples_per_segment": cell.calibration_samples_per_segment,
            "compass_duration_s": cell.compass_duration_s,
            "trial_count": cell.trial_count,
            "p95_absolute_heading_error_deg": cell.p95_absolute_heading_error_deg,
            "rms_heading_error_deg": cell.rms_heading_error_deg,
        } for cell in budget],
        "maytag": maytag_rows,
        "slow_turntable_earth_rotation": turntable_rows,
        "analytic_white_noise_p95_duration_s": {
            str(target_deg): white_noise_duration_for_p95_heading_error(
                noise_density,
                target_absolute_error_deg=target_deg,
                latitude_rad=latitude_rad,
            )
            for target_deg in (1.0, 0.5, 0.25, 0.1)
        },
    }
    json_path = destination / "tranche3_campaign.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for filename, rows in (
        ("white_noise_confidence.csv", white_rows),
        ("colored_bias_confidence.csv", colored_rows),
        ("latitude_confidence.csv", latitude_rows),
        ("turntable_earth_rotation.csv", turntable_rows),
    ):
        path = destination / filename
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        ####
    ####
    print(json.dumps(payload, indent=2))
    return json_path
####


if __name__ == "__main__":
    main()
####
