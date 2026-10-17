"""Monte Carlo calibration-recovery campaigns."""

from __future__ import annotations

from dataclasses import dataclass

from numpy import asarray, column_stack, concatenate, ndarray, ones, random
from numpy.linalg import lstsq

from imu_error_model import ImuConfig, ImuModel

from .known_reference import (
    CalibrationDataset,
    calibration_parameter_error,
    ferraris_style_reference_program,
    fit_known_reference_calibration,
    simulate_reference_program,
)
from .uncertainty import linear_fit_diagnostics


@dataclass(frozen=True, slots=True)
class MonteCarloRecoverySummary:
    """Empirical calibration accuracy and nominal-interval coverage."""

    trial_count: int
    parameter_count: int
    mean_error: ndarray
    empirical_standard_deviation: ndarray
    rmse: ndarray
    predicted_standard_error: ndarray
    coverage_95: ndarray

    @property
    def mean_coverage_95(self) -> float:
        """Return mean marginal 95-percent coverage across fitted parameters."""
        return float(self.coverage_95.mean())
    ####
####


def _flatten_standard_errors(dataset: CalibrationDataset) -> ndarray:
    """Return standard errors ordered like ``calibration_parameter_error``."""
    sample_count = len(dataset.labels)
    accel_design = column_stack((dataset.specific_force_body, ones(sample_count)))
    accel_coefficients, _, _, _ = lstsq(accel_design, dataset.measured_acceleration, rcond=None)
    accel_diag = linear_fit_diagnostics(accel_design, dataset.measured_acceleration, accel_coefficients)

    gyro_design = column_stack((dataset.angular_rate_body, dataset.specific_force_body, ones(sample_count)))
    gyro_coefficients, _, _, _ = lstsq(gyro_design, dataset.measured_angular_rate, rcond=None)
    gyro_diag = linear_fit_diagnostics(gyro_design, dataset.measured_angular_rate, gyro_coefficients)

    # Coefficients are [input_feature, output_axis].  Match the transpose-and-ravel
    # ordering used by CalibrationEstimate matrices.
    return concatenate(
        (
            accel_diag.coefficient_standard_error[:3, :].T.ravel(),
            accel_diag.coefficient_standard_error[3, :].ravel(),
            gyro_diag.coefficient_standard_error[:3, :].T.ravel(),
            gyro_diag.coefficient_standard_error[3:6, :].T.ravel(),
            gyro_diag.coefficient_standard_error[6, :].ravel(),
        )
    )
####


def run_monte_carlo_recovery(
        config: ImuConfig,
        *,
        trial_count: int = 200,
        samples_per_segment: int = 50,
        sample_period_s: float = 0.01,
        seed: int = 583,
) -> MonteCarloRecoverySummary:
    """Exercise the known-reference fit repeatedly under configured noise."""
    if trial_count < 2:
        raise ValueError("trial_count must be at least two")
    ####
    if samples_per_segment < 2:
        raise ValueError("samples_per_segment must be at least two")
    ####
    if sample_period_s <= 0.0:
        raise ValueError("sample_period_s must be positive")
    ####
    program = tuple(
        type(segment)(
            label=segment.label,
            duration_s=sample_period_s * samples_per_segment,
            specific_force_body=segment.specific_force_body,
            angular_rate_body=segment.angular_rate_body,
            temperature_celsius=segment.temperature_celsius,
            sample_count=samples_per_segment,
        )
        for segment in ferraris_style_reference_program(duration_s=sample_period_s * samples_per_segment)
    )
    seed_sequence = random.SeedSequence(seed)
    trial_seeds = seed_sequence.spawn(trial_count)
    errors: list[ndarray] = []
    standard_errors: list[ndarray] = []
    for trial_seed in trial_seeds:
        model = ImuModel(config=config, rng=random.default_rng(trial_seed))
        dataset = simulate_reference_program(model, program)
        estimate = fit_known_reference_calibration(dataset)
        errors.append(calibration_parameter_error(estimate, config))
        standard_errors.append(_flatten_standard_errors(dataset))
    ####
    error_matrix = asarray(errors, dtype=float)
    standard_error_matrix = asarray(standard_errors, dtype=float)
    mean_error = error_matrix.mean(axis=0)
    empirical_standard_deviation = error_matrix.std(axis=0, ddof=1)
    rmse = (error_matrix ** 2).mean(axis=0) ** 0.5
    predicted_standard_error = standard_error_matrix.mean(axis=0)
    coverage_95 = (abs(error_matrix) <= 1.96 * standard_error_matrix).mean(axis=0)
    for array in (mean_error, empirical_standard_deviation, rmse, predicted_standard_error, coverage_95):
        array.setflags(write=False)
    ####
    return MonteCarloRecoverySummary(
        trial_count=trial_count,
        parameter_count=error_matrix.shape[1],
        mean_error=mean_error,
        empirical_standard_deviation=empirical_standard_deviation,
        rmse=rmse,
        predicted_standard_error=predicted_standard_error,
        coverage_95=coverage_95,
    )
####
