"""Empirical data-duration and confidence studies for calibrated gyrocompassing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import pi
from typing import TYPE_CHECKING

from numpy import asarray, isfinite, ndarray, percentile, random, sqrt, zeros

from imu_error_model import ImuConfig, ImuModel

from .earth_reference import LocalEarthReference
from .gyrocompass import (
    GyrocompassResult,
    apply_calibration,
    simulate_static_measurements,
    static_gyrocompass,
    wrap_angle_rad,
)
from .known_reference import CalibrationEstimate


if TYPE_CHECKING:
    from .gyrocompass import ImuMeasurementBatch
    from .quality import StationarityThresholds
####


@dataclass(frozen=True, slots=True)
class HeadingConfidencePoint:
    """Empirical heading-error statistics for one averaging duration."""

    duration_s: float
    trial_count: int
    mean_error_deg: float
    standard_deviation_deg: float
    rms_error_deg: float
    p95_absolute_error_deg: float
####


@dataclass(frozen=True, slots=True)
class HeadingConfidenceSweep:
    """Heading confidence versus averaging time for one latitude and sensor model."""

    latitude_rad: float
    sample_period_s: float
    points: tuple[HeadingConfidencePoint, ...]

    def duration_for_p95_error(self, target_error_deg: float) -> float | None:
        """Return the first tested duration whose empirical p95 error meets the target."""
        if not isfinite(target_error_deg) or target_error_deg <= 0.0:
            raise ValueError("target_error_deg must be finite and positive")
        ####
        for point in self.points:
            if point.p95_absolute_error_deg <= target_error_deg:
                return point.duration_s
            ####
        ####
        return None
    ####
####


def _rotation_z(angle_rad: float) -> ndarray:
    from math import cos, sin

    return asarray(
        (
            (cos(angle_rad), -sin(angle_rad), 0.0),
            (sin(angle_rad), cos(angle_rad), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=float,
    )
####


def run_heading_confidence_sweep(
        *,
        model_factory: Callable[[random.Generator], ImuModel],
        calibration: CalibrationEstimate,
        latitude_rad: float,
        heading_rad: float,
        durations_s: tuple[float, ...],
        sample_period_s: float = 0.02,
        trial_count: int = 100,
        height_m: float = 0.0,
        seed: int = 0,
) -> HeadingConfidenceSweep:
    """Estimate heading confidence by reusing prefixes of one long realization per trial."""
    if not durations_s or any((not isfinite(value) or value <= 0.0) for value in durations_s):
        raise ValueError("durations_s must contain finite positive durations")
    ####
    if tuple(sorted(durations_s)) != durations_s:
        raise ValueError("durations_s must be sorted in ascending order")
    ####
    if not isfinite(sample_period_s) or sample_period_s <= 0.0:
        raise ValueError("sample_period_s must be finite and positive")
    ####
    if isinstance(trial_count, bool) or not isinstance(trial_count, int) or trial_count <= 1:
        raise ValueError("trial_count must be an integer greater than one")
    ####
    earth = LocalEarthReference(latitude_rad=latitude_rad, height_m=height_m)
    nav_from_body = _rotation_z(heading_rad)
    gravity_body = nav_from_body.T @ earth.gravity_ned_mps2
    earth_rate_body = nav_from_body.T @ earth.earth_rotation_ned_radps
    sample_counts = tuple(max(1, round(duration / sample_period_s)) for duration in durations_s)
    max_samples = sample_counts[-1]
    errors = zeros((trial_count, len(durations_s)))
    seed_sequence = random.SeedSequence(seed)
    child_seeds = seed_sequence.spawn(trial_count)

    for trial_index, child_seed in enumerate(child_seeds):
        model: ImuModel = model_factory(random.default_rng(child_seed))
        model.reset()
        accel_sum = zeros(3)
        gyro_sum = zeros(3)
        duration_index = 0
        current = 0.0
        for sample_index in range(1, max_samples + 1):
            output = model.measure_ideal_rates(
                start_time=current,
                end_time=current + sample_period_s,
                specific_force_body=gravity_body,
                angular_rate_body=earth_rate_body,
            )
            calibrated_accel, calibrated_gyro = apply_calibration(
                output.acceleration,
                output.angular_rate,
                calibration,
            )
            accel_sum += calibrated_accel
            gyro_sum += calibrated_gyro
            current += sample_period_s
            if sample_index == sample_counts[duration_index]:
                result = static_gyrocompass(
                    accel_sum / sample_index,
                    gyro_sum / sample_index,
                    latitude_rad=latitude_rad,
                    earth_rotation_radps=7.292115e-5,
                    gravity_mps2=float(earth.gravity_ned_mps2[2]),
                )
                errors[trial_index, duration_index] = wrap_angle_rad(result.heading_rad - heading_rad) * 180.0 / pi
                duration_index += 1
                if duration_index == len(sample_counts):
                    break
                ####
            ####
        ####
    ####

    points: list[HeadingConfidencePoint] = []
    for index, duration_s in enumerate(durations_s):
        values = errors[:, index]
        points.append(
            HeadingConfidencePoint(
                duration_s=duration_s,
                trial_count=trial_count,
                mean_error_deg=float(values.mean()),
                standard_deviation_deg=float(values.std(ddof=1)),
                rms_error_deg=float(sqrt((values ** 2).mean())),
                p95_absolute_error_deg=float(percentile(abs(values), 95.0)),
            )
        )
    ####
    return HeadingConfidenceSweep(
        latitude_rad=latitude_rad,
        sample_period_s=sample_period_s,
        points=tuple(points),
    )
####


def white_noise_heading_sigma_deg(
        gyro_white_noise_density_radps_sqrt_s: float,
        *,
        averaging_duration_s: float,
        latitude_rad: float,
) -> float:
    """First-order static-heading sigma for white gyro rate noise only."""
    from math import cos

    from .earth_reference import WGS84_EARTH_ROTATION_RADPS

    if not isfinite(gyro_white_noise_density_radps_sqrt_s) or gyro_white_noise_density_radps_sqrt_s < 0.0:
        raise ValueError("gyro_white_noise_density_radps_sqrt_s must be finite and non-negative")
    ####
    if not isfinite(averaging_duration_s) or averaging_duration_s <= 0.0:
        raise ValueError("averaging_duration_s must be finite and positive")
    ####
    horizontal_rate = abs(WGS84_EARTH_ROTATION_RADPS * cos(latitude_rad))
    if horizontal_rate <= 1e-12:
        raise ValueError("horizontal Earth-rate component is too small for this approximation")
    ####
    sigma_rad = gyro_white_noise_density_radps_sqrt_s / sqrt(averaging_duration_s) / horizontal_rate
    return sigma_rad * 180.0 / pi
####


@dataclass(frozen=True, slots=True)
class DataBudgetCell:
    """Joint calibration/gyrocompass data budget result."""

    calibration_samples_per_segment: int
    compass_duration_s: float
    trial_count: int
    p95_absolute_heading_error_deg: float
    rms_heading_error_deg: float
####


def run_joint_data_budget_sweep(
        *,
        config: ImuConfig,
        calibration_samples_per_segment: tuple[int, ...],
        compass_durations_s: tuple[float, ...],
        latitude_rad: float,
        heading_rad: float,
        sample_period_s: float = 0.02,
        trial_count: int = 50,
        seed: int = 0,
) -> tuple[DataBudgetCell, ...]:
    """Quantify how calibration data and north-finding data jointly set heading confidence."""
    from .known_reference import ferraris_style_reference_program, fit_known_reference_calibration, simulate_reference_program

    if not isinstance(config, ImuConfig):
        raise TypeError("config must be an ImuConfig")
    ####
    if not calibration_samples_per_segment or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 2
        for value in calibration_samples_per_segment
    ):
        raise ValueError("calibration_samples_per_segment must contain integers >= 2")
    ####
    if not compass_durations_s or any(not isfinite(value) or value <= 0.0 for value in compass_durations_s):
        raise ValueError("compass_durations_s must contain finite positive durations")
    ####
    if trial_count < 2:
        raise ValueError("trial_count must be at least two")
    ####

    errors = zeros((len(calibration_samples_per_segment), len(compass_durations_s), trial_count))
    seed_sequence = random.SeedSequence(seed)
    trial_seeds = seed_sequence.spawn(trial_count)
    for trial_index, trial_seed in enumerate(trial_seeds):
        calibration_seed, compass_seed_root = trial_seed.spawn(2)
        for calibration_index, sample_count in enumerate(calibration_samples_per_segment):
            base_program = ferraris_style_reference_program(duration_s=sample_count * sample_period_s)
            program = tuple(
                type(segment)(
                    label=segment.label,
                    duration_s=sample_count * sample_period_s,
                    specific_force_body=segment.specific_force_body,
                    angular_rate_body=segment.angular_rate_body,
                    temperature_celsius=segment.temperature_celsius,
                    sample_count=sample_count,
                )
                for segment in base_program
            )
            calibration_rng = random.default_rng(calibration_seed.spawn(1)[0])
            calibration = fit_known_reference_calibration(
                simulate_reference_program(ImuModel(config=config, rng=calibration_rng), program)
            )
            compass_seeds = compass_seed_root.spawn(len(compass_durations_s))
            for duration_index, (duration_s, compass_seed) in enumerate(zip(compass_durations_s, compass_seeds, strict=True)):
                sample_count_compass = max(1, round(duration_s / sample_period_s))
                result = _simulate_one_compass(
                    model=ImuModel(config=config, rng=random.default_rng(compass_seed)),
                    calibration=calibration,
                    latitude_rad=latitude_rad,
                    heading_rad=heading_rad,
                    sample_period_s=sample_period_s,
                    sample_count=sample_count_compass,
                )
                errors[calibration_index, duration_index, trial_index] = (
                    wrap_angle_rad(result.heading_rad - heading_rad) * 180.0 / pi
                )
            ####
        ####
    ####

    cells: list[DataBudgetCell] = []
    for calibration_index, calibration_samples in enumerate(calibration_samples_per_segment):
        for duration_index, duration_s in enumerate(compass_durations_s):
            values = errors[calibration_index, duration_index, :]
            cells.append(
                DataBudgetCell(
                    calibration_samples_per_segment=calibration_samples,
                    compass_duration_s=duration_s,
                    trial_count=trial_count,
                    p95_absolute_heading_error_deg=float(percentile(abs(values), 95.0)),
                    rms_heading_error_deg=float(sqrt((values ** 2).mean())),
                )
            )
        ####
    ####
    return tuple(cells)
####


def _simulate_one_compass(
        *,
        model: ImuModel,
        calibration: CalibrationEstimate,
        latitude_rad: float,
        heading_rad: float,
        sample_period_s: float,
        sample_count: int,
) -> GyrocompassResult:
    from .gyrocompass import simulate_static_gyrocompass

    return simulate_static_gyrocompass(
        model,
        calibration,
        latitude_rad=latitude_rad,
        heading_rad=heading_rad,
        sample_period_s=sample_period_s,
        sample_count=sample_count,
    )
####


@dataclass(frozen=True, slots=True)
class HeadingUncertaintyCoverageSummary:
    """Empirical coverage of propagated calibration and measurement covariance."""

    calibration_samples_per_segment: int
    compass_duration_s: float
    trial_count: int
    accepted_trial_count: int
    empirical_standard_deviation_deg: float
    empirical_p95_absolute_error_deg: float
    mean_predicted_standard_deviation_deg: float
    mean_predicted_95_half_width_deg: float
    coverage_95: float
    mean_measurement_variance_fraction: float
    mean_calibration_variance_fraction: float

    @property
    def accepted_fraction(self) -> float:
        return self.accepted_trial_count / self.trial_count
    ####
####


def run_heading_uncertainty_budget_sweep(
        *,
        config: ImuConfig,
        calibration_samples_per_segment: tuple[int, ...],
        compass_durations_s: tuple[float, ...],
        latitude_rad: float,
        heading_rad: float,
        sample_period_s: float = 0.02,
        trial_count: int = 100,
        covariance_max_lag: int | None = 0,
        seed: int = 0,
) -> tuple[HeadingUncertaintyCoverageSummary, ...]:
    """Evaluate a paired calibration/compass grid while reusing long records."""
    from .gyrocompass import ImuMeasurementBatch
    from .known_reference import ferraris_style_reference_program, fit_known_reference_calibration, simulate_reference_program
    from .quality import quality_aware_static_gyrocompass

    if not calibration_samples_per_segment or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 2
        for value in calibration_samples_per_segment
    ):
        raise ValueError("calibration_samples_per_segment must contain integers >= 2")
    ####
    if not compass_durations_s or any(not isfinite(value) or value <= 0.0 for value in compass_durations_s):
        raise ValueError("compass_durations_s must contain finite positive durations")
    ####
    if tuple(sorted(compass_durations_s)) != compass_durations_s:
        raise ValueError("compass_durations_s must be sorted in ascending order")
    ####
    if not isfinite(sample_period_s) or sample_period_s <= 0.0:
        raise ValueError("sample_period_s must be finite and positive")
    ####
    if isinstance(trial_count, bool) or not isinstance(trial_count, int) or trial_count < 2:
        raise ValueError("trial_count must be an integer >= 2")
    ####
    sample_counts = tuple(max(2, round(duration / sample_period_s)) for duration in compass_durations_s)
    cell_errors: dict[tuple[int, int], list[float]] = {}
    cell_sigmas: dict[tuple[int, int], list[float]] = {}
    cell_half_widths: dict[tuple[int, int], list[float]] = {}
    cell_measurement_fractions: dict[tuple[int, int], list[float]] = {}
    cell_calibration_fractions: dict[tuple[int, int], list[float]] = {}
    cell_covered: dict[tuple[int, int], list[bool]] = {}
    for calibration_index in range(len(calibration_samples_per_segment)):
        for duration_index in range(len(compass_durations_s)):
            key = (calibration_index, duration_index)
            cell_errors[key] = []
            cell_sigmas[key] = []
            cell_half_widths[key] = []
            cell_measurement_fractions[key] = []
            cell_calibration_fractions[key] = []
            cell_covered[key] = []
        ####
    ####

    for trial_seed in random.SeedSequence(seed).spawn(trial_count):
        child_seeds = trial_seed.spawn(len(calibration_samples_per_segment) + 1)
        compass_batch_full = simulate_static_measurements(
            ImuModel(config=config, rng=random.default_rng(child_seeds[-1])),
            latitude_rad=latitude_rad,
            heading_rad=heading_rad,
            sample_period_s=sample_period_s,
            sample_count=sample_counts[-1],
        )
        calibrations: list[CalibrationEstimate] = []
        for calibration_samples, calibration_seed in zip(
                calibration_samples_per_segment,
                child_seeds[:-1],
                strict=True,
        ):
            base_program = ferraris_style_reference_program(
                duration_s=calibration_samples * sample_period_s
            )
            program = tuple(
                type(segment)(
                    label=segment.label,
                    duration_s=calibration_samples * sample_period_s,
                    specific_force_body=segment.specific_force_body,
                    angular_rate_body=segment.angular_rate_body,
                    temperature_celsius=segment.temperature_celsius,
                    sample_count=calibration_samples,
                )
                for segment in base_program
            )
            calibrations.append(
                fit_known_reference_calibration(
                    simulate_reference_program(
                        ImuModel(config=config, rng=random.default_rng(calibration_seed)),
                        program,
                    )
                )
            )
        ####
        for duration_index, sample_count in enumerate(sample_counts):
            compass_batch = ImuMeasurementBatch(
                acceleration=compass_batch_full.acceleration[:sample_count],
                angular_rate=compass_batch_full.angular_rate[:sample_count],
                sample_period_s=sample_period_s,
                start_time=compass_batch_full.start_time,
            )
            for calibration_index, calibration in enumerate(calibrations):
                result = quality_aware_static_gyrocompass(
                    compass_batch,
                    calibration,
                    latitude_rad=latitude_rad,
                    covariance_max_lag=covariance_max_lag,
                )
                if not result.accepted or result.result is None or result.uncertainty is None:
                    continue
                ####
                key = (calibration_index, duration_index)
                error_deg = wrap_angle_rad(result.result.heading_rad - heading_rad) * 180.0 / pi
                half_width = result.uncertainty.confidence_95_half_width_deg
                cell_errors[key].append(error_deg)
                cell_sigmas[key].append(result.uncertainty.standard_deviation_deg)
                cell_half_widths[key].append(half_width)
                cell_measurement_fractions[key].append(result.uncertainty.measurement_fraction)
                cell_calibration_fractions[key].append(result.uncertainty.calibration_fraction)
                cell_covered[key].append(abs(error_deg) <= half_width)
            ####
        ####
    ####

    summaries: list[HeadingUncertaintyCoverageSummary] = []
    for calibration_index, calibration_samples in enumerate(calibration_samples_per_segment):
        for duration_index, duration_s in enumerate(compass_durations_s):
            key = (calibration_index, duration_index)
            errors = asarray(cell_errors[key], dtype=float)
            if errors.size < 2:
                raise RuntimeError("fewer than two paired trials passed the quality gates")
            ####
            summaries.append(
                HeadingUncertaintyCoverageSummary(
                    calibration_samples_per_segment=calibration_samples,
                    compass_duration_s=sample_counts[duration_index] * sample_period_s,
                    trial_count=trial_count,
                    accepted_trial_count=int(errors.size),
                    empirical_standard_deviation_deg=float(errors.std(ddof=1)),
                    empirical_p95_absolute_error_deg=float(percentile(abs(errors), 95.0)),
                    mean_predicted_standard_deviation_deg=float(asarray(cell_sigmas[key]).mean()),
                    mean_predicted_95_half_width_deg=float(asarray(cell_half_widths[key]).mean()),
                    coverage_95=float(asarray(cell_covered[key], dtype=float).mean()),
                    mean_measurement_variance_fraction=float(asarray(cell_measurement_fractions[key]).mean()),
                    mean_calibration_variance_fraction=float(asarray(cell_calibration_fractions[key]).mean()),
                )
            )
        ####
    ####
    return tuple(summaries)
####


def run_heading_uncertainty_coverage(
        *,
        config: ImuConfig,
        calibration_samples_per_segment: int,
        compass_duration_s: float,
        latitude_rad: float,
        heading_rad: float,
        sample_period_s: float = 0.02,
        trial_count: int = 100,
        covariance_max_lag: int | None = 0,
        seed: int = 0,
) -> HeadingUncertaintyCoverageSummary:
    """Validate first-order heading intervals with joint calibration/compass trials."""
    from .known_reference import ferraris_style_reference_program, fit_known_reference_calibration, simulate_reference_program
    from .quality import quality_aware_static_gyrocompass

    if not isinstance(config, ImuConfig):
        raise TypeError("config must be an ImuConfig")
    ####
    if (
            isinstance(calibration_samples_per_segment, bool)
            or not isinstance(calibration_samples_per_segment, int)
            or calibration_samples_per_segment < 2
    ):
        raise ValueError("calibration_samples_per_segment must be an integer >= 2")
    ####
    if not isfinite(compass_duration_s) or compass_duration_s <= 0.0:
        raise ValueError("compass_duration_s must be finite and positive")
    ####
    if not isfinite(sample_period_s) or sample_period_s <= 0.0:
        raise ValueError("sample_period_s must be finite and positive")
    ####
    if isinstance(trial_count, bool) or not isinstance(trial_count, int) or trial_count < 2:
        raise ValueError("trial_count must be an integer >= 2")
    ####
    compass_sample_count = max(2, round(compass_duration_s / sample_period_s))
    base_program = ferraris_style_reference_program(
        duration_s=calibration_samples_per_segment * sample_period_s
    )
    program = tuple(
        type(segment)(
            label=segment.label,
            duration_s=calibration_samples_per_segment * sample_period_s,
            specific_force_body=segment.specific_force_body,
            angular_rate_body=segment.angular_rate_body,
            temperature_celsius=segment.temperature_celsius,
            sample_count=calibration_samples_per_segment,
        )
        for segment in base_program
    )
    errors: list[float] = []
    predicted_sigma: list[float] = []
    predicted_half_width: list[float] = []
    measurement_fractions: list[float] = []
    calibration_fractions: list[float] = []
    covered: list[bool] = []
    for trial_seed in random.SeedSequence(seed).spawn(trial_count):
        calibration_seed, compass_seed = trial_seed.spawn(2)
        calibration_dataset = simulate_reference_program(
            ImuModel(config=config, rng=random.default_rng(calibration_seed)),
            program,
        )
        calibration = fit_known_reference_calibration(calibration_dataset)
        batch = simulate_static_measurements(
            ImuModel(config=config, rng=random.default_rng(compass_seed)),
            latitude_rad=latitude_rad,
            heading_rad=heading_rad,
            sample_period_s=sample_period_s,
            sample_count=compass_sample_count,
        )
        quality_result = quality_aware_static_gyrocompass(
            batch,
            calibration,
            latitude_rad=latitude_rad,
            covariance_max_lag=covariance_max_lag,
        )
        if not quality_result.accepted or quality_result.result is None or quality_result.uncertainty is None:
            continue
        ####
        error_deg = wrap_angle_rad(quality_result.result.heading_rad - heading_rad) * 180.0 / pi
        half_width = quality_result.uncertainty.confidence_95_half_width_deg
        errors.append(error_deg)
        predicted_sigma.append(quality_result.uncertainty.standard_deviation_deg)
        predicted_half_width.append(half_width)
        measurement_fractions.append(quality_result.uncertainty.measurement_fraction)
        calibration_fractions.append(quality_result.uncertainty.calibration_fraction)
        covered.append(abs(error_deg) <= half_width)
    ####
    if len(errors) < 2:
        raise RuntimeError("fewer than two gyrocompass trials passed the quality gates")
    ####
    error_values = asarray(errors, dtype=float)
    return HeadingUncertaintyCoverageSummary(
        calibration_samples_per_segment=calibration_samples_per_segment,
        compass_duration_s=compass_sample_count * sample_period_s,
        trial_count=trial_count,
        accepted_trial_count=len(errors),
        empirical_standard_deviation_deg=float(error_values.std(ddof=1)),
        empirical_p95_absolute_error_deg=float(percentile(abs(error_values), 95.0)),
        mean_predicted_standard_deviation_deg=float(asarray(predicted_sigma).mean()),
        mean_predicted_95_half_width_deg=float(asarray(predicted_half_width).mean()),
        coverage_95=float(asarray(covered, dtype=float).mean()),
        mean_measurement_variance_fraction=float(asarray(measurement_fractions).mean()),
        mean_calibration_variance_fraction=float(asarray(calibration_fractions).mean()),
    )
####


def white_noise_duration_for_p95_heading_error(
        gyro_white_noise_density_radps_sqrt_s: float,
        *,
        target_absolute_error_deg: float,
        latitude_rad: float,
) -> float:
    """First-order averaging time for a two-sided 95% white-noise heading target."""
    from math import cos

    from .earth_reference import WGS84_EARTH_ROTATION_RADPS

    if not isfinite(target_absolute_error_deg) or target_absolute_error_deg <= 0.0:
        raise ValueError("target_absolute_error_deg must be finite and positive")
    ####
    if not isfinite(gyro_white_noise_density_radps_sqrt_s) or gyro_white_noise_density_radps_sqrt_s < 0.0:
        raise ValueError("gyro_white_noise_density_radps_sqrt_s must be finite and non-negative")
    ####
    horizontal_rate = abs(WGS84_EARTH_ROTATION_RADPS * cos(latitude_rad))
    if horizontal_rate <= 1e-12:
        raise ValueError("horizontal Earth-rate component is too small for this approximation")
    ####
    target_rad = target_absolute_error_deg * pi / 180.0
    return (1.96 * gyro_white_noise_density_radps_sqrt_s / (horizontal_rate * target_rad)) ** 2
####


@dataclass(frozen=True, slots=True)
class PrefixConfidencePoint:
    """One quality-gated confidence estimate from a prefix of a measurement record."""

    duration_s: float
    sample_count: int
    accepted: bool
    stationarity_flags: tuple[str, ...]
    heading_deg: float | None
    predicted_standard_deviation_deg: float | None
    predicted_95_half_width_deg: float | None
    measurement_variance_fraction: float | None
    calibration_variance_fraction: float | None
    minimum_effective_sample_size: float | None
    covariance_method: str | None
    covariance_max_lag: int | None

    def meets_target(self, target_half_width_deg: float) -> bool:
        """Return whether this accepted prefix reaches a requested 95% half-width."""
        if not isfinite(target_half_width_deg) or target_half_width_deg <= 0.0:
            raise ValueError("target_half_width_deg must be finite and positive")
        ####
        return bool(
            self.accepted
            and self.predicted_95_half_width_deg is not None
            and self.predicted_95_half_width_deg <= target_half_width_deg
        )
    ####
####


@dataclass(frozen=True, slots=True)
class PrefixConfidenceTrace:
    """Predicted confidence and data sufficiency along one accumulating record."""

    target_half_width_deg: float
    sample_period_s: float
    points: tuple[PrefixConfidencePoint, ...]

    def __post_init__(self) -> None:
        if not isfinite(self.target_half_width_deg) or self.target_half_width_deg <= 0.0:
            raise ValueError("target_half_width_deg must be finite and positive")
        ####
        if not isfinite(self.sample_period_s) or self.sample_period_s <= 0.0:
            raise ValueError("sample_period_s must be finite and positive")
        ####
        if not self.points:
            raise ValueError("points must not be empty")
        ####
    ####

    @property
    def duration_to_target_s(self) -> float | None:
        """Return the first tested accepted duration that reaches the target."""
        for point in self.points:
            if point.meets_target(self.target_half_width_deg):
                return point.duration_s
            ####
        ####
        return None
    ####

    @property
    def first_accepted_duration_s(self) -> float | None:
        """Return the first tested duration that passes all quality gates."""
        for point in self.points:
            if point.accepted:
                return point.duration_s
            ####
        ####
        return None
    ####
####


def evaluate_prefix_confidence(
        batch: ImuMeasurementBatch,
        calibration: CalibrationEstimate,
        *,
        latitude_rad: float,
        durations_s: tuple[float, ...],
        target_half_width_deg: float,
        height_m: float = 0.0,
        thresholds: StationarityThresholds | None = None,
        covariance_max_lag: int | None = None,
) -> PrefixConfidenceTrace:
    """Evaluate quality and heading confidence for increasing prefixes of one record.

    Prefix reuse separates the effect of collecting more data from run-to-run
    stochastic variation. The reported duration-to-target is limited to the
    tested durations and does not extrapolate through rejected or unobserved
    intervals.
    """
    from .gyrocompass import ImuMeasurementBatch
    from .quality import StationarityThresholds, quality_aware_static_gyrocompass

    if not isinstance(batch, ImuMeasurementBatch):
        raise TypeError("batch must be an ImuMeasurementBatch")
    ####
    if not isinstance(calibration, CalibrationEstimate):
        raise TypeError("calibration must be a CalibrationEstimate")
    ####
    if not durations_s or any(not isfinite(value) or value <= 0.0 for value in durations_s):
        raise ValueError("durations_s must contain finite positive durations")
    ####
    if tuple(sorted(durations_s)) != durations_s or len(set(durations_s)) != len(durations_s):
        raise ValueError("durations_s must be unique and sorted in ascending order")
    ####
    if not isfinite(target_half_width_deg) or target_half_width_deg <= 0.0:
        raise ValueError("target_half_width_deg must be finite and positive")
    ####
    if thresholds is not None and not isinstance(thresholds, StationarityThresholds):
        raise TypeError("thresholds must be a StationarityThresholds instance")
    ####
    sample_counts = tuple(max(2, round(duration / batch.sample_period_s)) for duration in durations_s)
    if sample_counts[-1] > batch.sample_count:
        raise ValueError("the longest requested prefix exceeds the available measurement batch")
    ####
    if len(set(sample_counts)) != len(sample_counts):
        raise ValueError("durations_s collapse to duplicate sample counts at this sample period")
    ####
    points: list[PrefixConfidencePoint] = []
    for sample_count in sample_counts:
        prefix = ImuMeasurementBatch(
            acceleration=batch.acceleration[:sample_count],
            angular_rate=batch.angular_rate[:sample_count],
            sample_period_s=batch.sample_period_s,
            start_time=batch.start_time,
        )
        result = quality_aware_static_gyrocompass(
            prefix,
            calibration,
            latitude_rad=latitude_rad,
            height_m=height_m,
            thresholds=thresholds,
            covariance_max_lag=covariance_max_lag,
        )
        uncertainty = result.uncertainty
        effective_sample_size = result.measurement_effective_sample_size
        points.append(
            PrefixConfidencePoint(
                duration_s=prefix.duration_s,
                sample_count=prefix.sample_count,
                accepted=result.accepted,
                stationarity_flags=tuple(flag.value for flag in result.stationarity.flags),
                heading_deg=(None if result.result is None else result.result.heading_deg),
                predicted_standard_deviation_deg=(
                    None if uncertainty is None else uncertainty.standard_deviation_deg
                ),
                predicted_95_half_width_deg=(
                    None if uncertainty is None else uncertainty.confidence_95_half_width_deg
                ),
                measurement_variance_fraction=(
                    None if uncertainty is None else uncertainty.measurement_fraction
                ),
                calibration_variance_fraction=(
                    None if uncertainty is None else uncertainty.calibration_fraction
                ),
                minimum_effective_sample_size=(
                    None if effective_sample_size is None else float(effective_sample_size.min())
                ),
                covariance_method=result.covariance_method,
                covariance_max_lag=result.covariance_max_lag,
            )
        )
    ####
    return PrefixConfidenceTrace(
        target_half_width_deg=target_half_width_deg,
        sample_period_s=batch.sample_period_s,
        points=tuple(points),
    )
####
