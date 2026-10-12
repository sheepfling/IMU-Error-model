"""Generate tranche-four confidence, quality-gate, and modulation evidence."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from math import pi
from pathlib import Path

from numpy import asarray, random

from imu_error_model import AxisConfig, ImuConfig, ImuModel

from .bootstrap import moving_block_bootstrap_calibration
from .carousel import run_carousel_heading_confidence_sweep
from .data_confidence import (
    HeadingUncertaintyCoverageSummary,
    evaluate_prefix_confidence,
    run_heading_confidence_sweep,
    run_heading_uncertainty_budget_sweep,
    white_noise_duration_for_p95_heading_error,
)
from .gyrocompass import simulate_static_measurements
from .known_reference import (
    CalibrationEstimate,
    ReferenceSegment,
    ferraris_style_reference_program,
    fit_known_reference_calibration,
    simulate_reference_program,
)
from .maytag import run_maytag_heading_confidence_sweep
from .motion import HarmonicMotionExcitation
from .planning import build_joint_data_plan
from .quality import StationarityThresholds, quality_aware_static_gyrocompass


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("rows must not be empty")
    ####
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ####
####


def _ideal_calibration() -> CalibrationEstimate:
    return fit_known_reference_calibration(
        simulate_reference_program(ImuModel(), ferraris_style_reference_program())
    )
####


def _program_with_samples(
        sample_count: int,
        sample_period_s: float,
) -> tuple[ReferenceSegment, ...]:
    base = ferraris_style_reference_program(duration_s=sample_count * sample_period_s)
    return tuple(replace(segment, sample_count=sample_count) for segment in base)
####


def _confidence_coverage(
        *,
        config: ImuConfig,
        latitude_rad: float,
        heading_rad: float,
        sample_period_s: float,
) -> tuple[tuple[HeadingUncertaintyCoverageSummary, ...], list[dict[str, object]]]:
    summaries = run_heading_uncertainty_budget_sweep(
        config=config,
        calibration_samples_per_segment=(20, 100, 500),
        compass_durations_s=(30.0, 120.0),
        latitude_rad=latitude_rad,
        heading_rad=heading_rad,
        sample_period_s=sample_period_s,
        trial_count=12,
        covariance_max_lag=0,
        seed=1000,
    )
    rows = [
        {
            "calibration_samples_per_segment": summary.calibration_samples_per_segment,
            "compass_duration_s": summary.compass_duration_s,
            "trial_count": summary.trial_count,
            "accepted_fraction": summary.accepted_fraction,
            "empirical_standard_deviation_deg": summary.empirical_standard_deviation_deg,
            "empirical_p95_absolute_error_deg": summary.empirical_p95_absolute_error_deg,
            "mean_predicted_standard_deviation_deg": summary.mean_predicted_standard_deviation_deg,
            "mean_predicted_95_half_width_deg": summary.mean_predicted_95_half_width_deg,
            "coverage_95": summary.coverage_95,
            "measurement_variance_fraction": summary.mean_measurement_variance_fraction,
            "calibration_variance_fraction": summary.mean_calibration_variance_fraction,
        }
        for summary in summaries
    ]
    return summaries, rows
####


def _modulation_rows(
        *,
        config: ImuConfig,
        calibration: CalibrationEstimate,
        latitude_rad: float,
        heading_rad: float,
        sample_period_s: float,
) -> list[dict[str, object]]:
    def model_factory(rng: random.Generator) -> ImuModel:
        return ImuModel(config, rng=rng)
    ####

    sweeps = (
        (
            "static",
            run_heading_confidence_sweep(
                model_factory=model_factory,
                calibration=calibration,
                latitude_rad=latitude_rad,
                heading_rad=heading_rad,
                durations_s=(30.0, 60.0),
                sample_period_s=sample_period_s,
                trial_count=16,
                seed=2084,
            ),
        ),
        (
            "maytag_2",
            run_maytag_heading_confidence_sweep(
                model_factory=model_factory,
                calibration=calibration,
                latitude_rad=latitude_rad,
                heading_rad=heading_rad,
                position_count=2,
                durations_s=(30.0, 60.0),
                sample_period_s=sample_period_s,
                trial_count=16,
                seed=2084,
            ),
        ),
        (
            "maytag_4",
            run_maytag_heading_confidence_sweep(
                model_factory=model_factory,
                calibration=calibration,
                latitude_rad=latitude_rad,
                heading_rad=heading_rad,
                position_count=4,
                durations_s=(30.0, 60.0),
                sample_period_s=sample_period_s,
                trial_count=16,
                seed=2084,
            ),
        ),
        (
            "carousel",
            run_carousel_heading_confidence_sweep(
                model_factory=model_factory,
                calibration=calibration,
                latitude_rad=latitude_rad,
                heading_rad=heading_rad,
                rotation_rate_radps=2.0 * pi / 60.0,
                durations_s=(30.0, 60.0),
                sample_period_s=sample_period_s,
                trial_count=16,
                seed=2084,
            ),
        ),
    )
    rows: list[dict[str, object]] = []
    for method, sweep in sweeps:
        for point in sweep.points:
            rows.append(
                {
                    "method": method,
                    "duration_s": point.duration_s,
                    "trial_count": point.trial_count,
                    "mean_error_deg": point.mean_error_deg,
                    "standard_deviation_deg": point.standard_deviation_deg,
                    "rms_error_deg": point.rms_error_deg,
                    "p95_absolute_error_deg": point.p95_absolute_error_deg,
                }
            )
        ####
    ####
    return rows
####


def _carousel_rate_rows(
        *,
        config: ImuConfig,
        calibration: CalibrationEstimate,
        latitude_rad: float,
        heading_rad: float,
        sample_period_s: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for table_rate_degps in (1.0, 3.0, 6.0, 12.0):
        table_rate_radps = table_rate_degps * pi / 180.0
        point = run_carousel_heading_confidence_sweep(
            model_factory=lambda rng: ImuModel(config, rng=rng),
            calibration=calibration,
            latitude_rad=latitude_rad,
            heading_rad=heading_rad,
            rotation_rate_radps=table_rate_radps,
            durations_s=(60.0,),
            sample_period_s=sample_period_s,
            trial_count=12,
            seed=3000 + int(table_rate_degps),
        ).points[0]
        rows.append(
            {
                "table_rate_degps": table_rate_degps,
                "duration_s": point.duration_s,
                "revolution_count": table_rate_radps * point.duration_s / (2.0 * pi),
                "standard_deviation_deg": point.standard_deviation_deg,
                "p95_absolute_error_deg": point.p95_absolute_error_deg,
            }
        )
    ####
    return rows
####


def _quality_rows(
        *,
        calibration: CalibrationEstimate,
        latitude_rad: float,
        heading_rad: float,
) -> list[dict[str, object]]:
    config = ImuConfig(
        accelerometer=AxisConfig(white_noise_density=0.001),
        gyroscope=AxisConfig(white_noise_density=2e-6),
    )
    clean_batch = simulate_static_measurements(
        ImuModel(config, rng=random.default_rng(40)),
        latitude_rad=latitude_rad,
        heading_rad=heading_rad,
        sample_period_s=0.02,
        sample_count=1000,
    )
    motion = HarmonicMotionExcitation(
        specific_force_amplitude_body=asarray((0.0, 0.0, 0.3)),
        angular_rate_amplitude_body=asarray((0.0, 0.01, 0.0)),
        frequency_hz=0.5,
    )
    contaminated_batch = simulate_static_measurements(
        ImuModel(config, rng=random.default_rng(41)),
        latitude_rad=latitude_rad,
        heading_rad=heading_rad,
        sample_period_s=0.02,
        sample_count=1000,
        motion_excitation=motion,
    )
    rows: list[dict[str, object]] = []
    for label, batch in (("clean", clean_batch), ("harmonic_motion", contaminated_batch)):
        result = quality_aware_static_gyrocompass(
            batch,
            calibration,
            latitude_rad=latitude_rad,
            covariance_max_lag=0,
        )
        rows.append(
            {
                "case": label,
                "accepted": result.accepted,
                "flags": ";".join(flag.value for flag in result.stationarity.flags),
                "acceleration_norm_std_mps2": result.stationarity.acceleration_norm_std_mps2,
                "maximum_angular_rate_axis_std_radps": result.stationarity.maximum_angular_rate_axis_std_radps,
                "acceleration_split_mean_delta_mps2": result.stationarity.acceleration_split_mean_delta_mps2,
                "angular_rate_split_mean_delta_radps": result.stationarity.angular_rate_split_mean_delta_radps,
            }
        )
    ####
    return rows
####


def _serial_correlation_rows(
        *,
        calibration: CalibrationEstimate,
        latitude_rad: float,
        heading_rad: float,
        sample_period_s: float,
) -> list[dict[str, object]]:
    config = ImuConfig(
        gyroscope=AxisConfig(
            white_noise_density=2e-6,
            bias_std=5e-6,
            bias_correlation_time=100.0,
        )
    )
    batch = simulate_static_measurements(
        ImuModel(config, rng=random.default_rng(8)),
        latitude_rad=latitude_rad,
        heading_rad=heading_rad,
        sample_period_s=sample_period_s,
        sample_count=1200,
    )
    permissive_drift = StationarityThresholds(
        maximum_angular_rate_split_mean_delta_radps=5e-5
    )
    rows: list[dict[str, object]] = []
    for max_lag in (0, 10, 100, 500):
        result = quality_aware_static_gyrocompass(
            batch,
            calibration,
            latitude_rad=latitude_rad,
            thresholds=permissive_drift,
            covariance_max_lag=max_lag,
        )
        if result.uncertainty is None or result.measurement_effective_sample_size is None:
            raise RuntimeError("colored confidence record unexpectedly failed the stationarity gate")
        ####
        rows.append(
            {
                "covariance_method": result.covariance_method,
                "covariance_max_lag_samples": max_lag,
                "lag_horizon_s": max_lag * sample_period_s,
                "minimum_effective_sample_size": float(result.measurement_effective_sample_size.min()),
                "predicted_standard_deviation_deg": result.uncertainty.standard_deviation_deg,
                "predicted_95_half_width_deg": result.uncertainty.confidence_95_half_width_deg,
            }
        )
    ####
    return rows
####


def _prefix_rows(
        *,
        config: ImuConfig,
        latitude_rad: float,
        heading_rad: float,
        sample_period_s: float,
) -> tuple[list[dict[str, object]], float | None]:
    calibration = fit_known_reference_calibration(
        simulate_reference_program(
            ImuModel(config, rng=random.default_rng(500)),
            _program_with_samples(250, sample_period_s),
        )
    )
    batch = simulate_static_measurements(
        ImuModel(config, rng=random.default_rng(501)),
        latitude_rad=latitude_rad,
        heading_rad=heading_rad,
        sample_period_s=sample_period_s,
        sample_count=1200,
    )
    trace = evaluate_prefix_confidence(
        batch,
        calibration,
        latitude_rad=latitude_rad,
        durations_s=(2.0, 5.0, 10.0, 30.0, 60.0, 120.0),
        target_half_width_deg=0.5,
        covariance_max_lag=0,
    )
    rows = [
        {
            "duration_s": point.duration_s,
            "sample_count": point.sample_count,
            "accepted": point.accepted,
            "stationarity_flags": ";".join(point.stationarity_flags),
            "heading_deg": point.heading_deg,
            "predicted_standard_deviation_deg": point.predicted_standard_deviation_deg,
            "predicted_95_half_width_deg": point.predicted_95_half_width_deg,
            "measurement_variance_fraction": point.measurement_variance_fraction,
            "calibration_variance_fraction": point.calibration_variance_fraction,
            "minimum_effective_sample_size": point.minimum_effective_sample_size,
            "covariance_method": point.covariance_method,
            "covariance_max_lag": point.covariance_max_lag,
            "meets_target": point.meets_target(trace.target_half_width_deg),
        }
        for point in trace.points
    ]
    return rows, trace.duration_to_target_s
####


def _bootstrap_rows(
        *,
        latitude_rad: float,
        heading_rad: float,
        sample_period_s: float,
) -> list[dict[str, object]]:
    calibration_config = ImuConfig(
        accelerometer=AxisConfig(white_noise_density=5e-4),
        gyroscope=AxisConfig(
            white_noise_density=1e-6,
            bias_std=7e-6,
            bias_correlation_time=20.0,
        ),
    )
    dataset = simulate_reference_program(
        ImuModel(calibration_config, rng=random.default_rng(700)),
        _program_with_samples(200, sample_period_s),
    )
    classical = fit_known_reference_calibration(dataset)
    bootstrap = moving_block_bootstrap_calibration(
        dataset,
        replicate_count=80,
        block_length=30,
        seed=701,
    )
    compass_batch = simulate_static_measurements(
        ImuModel(),
        latitude_rad=latitude_rad,
        heading_rad=heading_rad,
        sample_period_s=sample_period_s,
        sample_count=1200,
    )
    rows: list[dict[str, object]] = []
    for method, calibration in (("classical_iid", classical), ("moving_block_bootstrap", bootstrap.estimate)):
        result = quality_aware_static_gyrocompass(
            compass_batch,
            calibration,
            latitude_rad=latitude_rad,
            covariance_max_lag=0,
        )
        if result.uncertainty is None:
            raise RuntimeError("bootstrap comparison unexpectedly failed the stationarity gate")
        ####
        rows.append(
            {
                "method": method,
                "covariance_assumption": calibration.covariance_assumption,
                "predicted_standard_deviation_deg": result.uncertainty.standard_deviation_deg,
                "predicted_95_half_width_deg": result.uncertainty.confidence_95_half_width_deg,
                "measurement_variance_fraction": result.uncertainty.measurement_fraction,
                "calibration_variance_fraction": result.uncertainty.calibration_fraction,
                "accelerometer_median_variance_inflation": (
                    1.0 if method == "classical_iid" else bootstrap.accelerometer_median_variance_inflation
                ),
                "gyroscope_median_variance_inflation": (
                    1.0 if method == "classical_iid" else bootstrap.gyroscope_median_variance_inflation
                ),
            }
        )
    ####
    return rows
####


def _white_noise_planning_rows(*, latitude_rad: float, gyro_noise_density: float) -> list[dict[str, object]]:
    return [
        {
            "target_p95_heading_error_deg": target,
            "gyro_white_noise_density_radps_sqrt_s": gyro_noise_density,
            "required_averaging_duration_s": white_noise_duration_for_p95_heading_error(
                gyro_noise_density,
                target_absolute_error_deg=target,
                latitude_rad=latitude_rad,
            ),
        }
        for target in (1.0, 0.5, 0.25, 0.1)
    ]
####


def main(destination: str | Path = "artifacts/calibration/tranche4") -> Path:
    """Run the complete tranche-four evidence campaign and write machine-readable results."""
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    latitude_rad = 35.0 * pi / 180.0
    heading_rad = 20.0 * pi / 180.0
    sample_period_s = 0.2
    ideal_calibration = _ideal_calibration()
    confidence_config = ImuConfig(
        accelerometer=AxisConfig(
            white_noise_density=0.001,
            fixed_bias=(0.02, -0.01, 0.01),
        ),
        gyroscope=AxisConfig(
            white_noise_density=2e-6,
            fixed_bias=(5e-6, -3e-6, 2e-6),
        ),
    )
    mixed_bias_config = ImuConfig(
        gyroscope=AxisConfig(
            white_noise_density=2e-6,
            turn_on_bias_std=2e-5,
            bias_std=5e-6,
            bias_correlation_time=100.0,
        )
    )

    uncertainty_summaries, uncertainty_rows = _confidence_coverage(
        config=confidence_config,
        latitude_rad=latitude_rad,
        heading_rad=heading_rad,
        sample_period_s=sample_period_s,
    )
    joint_plan = build_joint_data_plan(
        uncertainty_summaries,
        sample_period_s=sample_period_s,
        calibration_segment_count=len(ferraris_style_reference_program()),
        target_95_half_width_deg=0.5,
    )
    joint_plan_rows = [asdict(point) | {"limiting_source": point.limiting_source} for point in joint_plan.points]
    modulation_rows = _modulation_rows(
        config=mixed_bias_config,
        calibration=ideal_calibration,
        latitude_rad=latitude_rad,
        heading_rad=heading_rad,
        sample_period_s=sample_period_s,
    )
    rate_rows = _carousel_rate_rows(
        config=mixed_bias_config,
        calibration=ideal_calibration,
        latitude_rad=latitude_rad,
        heading_rad=heading_rad,
        sample_period_s=sample_period_s,
    )
    quality_rows = _quality_rows(
        calibration=ideal_calibration,
        latitude_rad=latitude_rad,
        heading_rad=heading_rad,
    )
    correlation_rows = _serial_correlation_rows(
        calibration=ideal_calibration,
        latitude_rad=latitude_rad,
        heading_rad=heading_rad,
        sample_period_s=sample_period_s,
    )
    prefix_rows, duration_to_target_s = _prefix_rows(
        config=confidence_config,
        latitude_rad=latitude_rad,
        heading_rad=heading_rad,
        sample_period_s=sample_period_s,
    )
    bootstrap_rows = _bootstrap_rows(
        latitude_rad=latitude_rad,
        heading_rad=heading_rad,
        sample_period_s=sample_period_s,
    )
    white_noise_rows = _white_noise_planning_rows(
        latitude_rad=latitude_rad,
        gyro_noise_density=2e-6,
    )

    files = {
        "joint_uncertainty_coverage.csv": uncertainty_rows,
        "joint_data_plan.csv": joint_plan_rows,
        "static_maytag_carousel.csv": modulation_rows,
        "carousel_rate_sweep.csv": rate_rows,
        "stationarity_gate.csv": quality_rows,
        "serial_correlation_covariance.csv": correlation_rows,
        "prefix_confidence.csv": prefix_rows,
        "calibration_bootstrap.csv": bootstrap_rows,
        "white_noise_planning.csv": white_noise_rows,
    }
    for name, rows in files.items():
        _write_csv(output / name, rows)
    ####
    payload = {
        "joint_uncertainty_coverage": uncertainty_rows,
        "joint_data_plan": {
            "target_95_half_width_deg": joint_plan.target_95_half_width_deg,
            "calibration_segment_count": joint_plan.calibration_segment_count,
            "sample_period_s": joint_plan.sample_period_s,
            "per_transition_overhead_s": joint_plan.per_transition_overhead_s,
            "recommendation": (
                None
                if joint_plan.recommendation is None
                else asdict(joint_plan.recommendation)
                | {"limiting_source": joint_plan.recommendation.limiting_source}
            ),
            "pareto_front": [
                asdict(point) | {"limiting_source": point.limiting_source}
                for point in joint_plan.pareto_front
            ],
            "points": joint_plan_rows,
        },
        "static_maytag_carousel": modulation_rows,
        "carousel_rate_sweep": rate_rows,
        "stationarity_gate": quality_rows,
        "serial_correlation_covariance": correlation_rows,
        "prefix_confidence": prefix_rows,
        "prefix_duration_to_0_5_deg_95_s": duration_to_target_s,
        "calibration_bootstrap": bootstrap_rows,
        "white_noise_planning": white_noise_rows,
        "assumptions": {
            "classical_calibration_covariance": "multivariate least squares with iid residual rows",
            "bootstrap_calibration_covariance": "label-wise circular moving blocks; no cross-label state reconstruction",
            "measurement_covariance": "Bartlett/Newey-West long-run covariance with reported maximum lag",
            "carousel": "level table, known angle/rate, exact interval-average frame map",
            "prefix_target": "first tested accepted prefix; no untested-duration interpolation",
            "joint_data_plan_time": "sampled acquisition only; transition/setup overhead set to zero",
        },
    }
    json_path = output / "tranche4_campaign.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return json_path
####


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "destination",
        nargs="?",
        default="artifacts/calibration/tranche4",
        help="directory for CSV and JSON evidence",
    )
    arguments = parser.parse_args()
    main(arguments.destination)
####
