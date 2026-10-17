"""Joint calibration and north-finding data-budget planning helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .data_confidence import HeadingUncertaintyCoverageSummary


@dataclass(frozen=True, slots=True)
class JointDataPlanPoint:
    """One tested allocation of calibration and gyrocompass acquisition time."""

    calibration_samples_per_segment: int
    compass_duration_s: float
    calibration_duration_s: float
    total_acquisition_duration_s: float
    predicted_95_half_width_deg: float
    empirical_p95_absolute_error_deg: float
    empirical_coverage_95: float
    accepted_fraction: float
    measurement_variance_fraction: float
    calibration_variance_fraction: float
    target_met: bool
    empirically_validated: bool
    pareto_optimal: bool = False

    @property
    def limiting_source(self) -> str:
        """Return the larger propagated uncertainty source for this allocation."""
        return (
            "measurement"
            if self.measurement_variance_fraction >= self.calibration_variance_fraction
            else "calibration"
        )
    ####
####


@dataclass(frozen=True, slots=True)
class JointDataPlan:
    """Evaluated data-budget grid and the least-time validated target solution."""

    target_95_half_width_deg: float
    sample_period_s: float
    calibration_segment_count: int
    per_transition_overhead_s: float
    minimum_empirical_coverage_95: float
    minimum_accepted_fraction: float
    points: tuple[JointDataPlanPoint, ...]

    @property
    def recommendation(self) -> JointDataPlanPoint | None:
        """Return the shortest validated allocation that meets the confidence target."""
        candidates = tuple(
            point
            for point in self.points
            if point.target_met and point.empirically_validated
        )
        if not candidates:
            return None
        ####
        return min(
            candidates,
            key=lambda point: (
                point.total_acquisition_duration_s,
                point.predicted_95_half_width_deg,
            ),
        )
    ####

    @property
    def pareto_front(self) -> tuple[JointDataPlanPoint, ...]:
        """Return allocations not dominated in time and predicted confidence."""
        return tuple(point for point in self.points if point.pareto_optimal)
    ####
####


def _validate_probability(value: float, field_name: str) -> None:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be finite and within [0, 1]")
    ####
####


def build_joint_data_plan(
        summaries: tuple[HeadingUncertaintyCoverageSummary, ...],
        *,
        sample_period_s: float,
        calibration_segment_count: int = 13,
        per_transition_overhead_s: float = 0.0,
        target_95_half_width_deg: float = 0.5,
        minimum_empirical_coverage_95: float = 0.8,
        minimum_accepted_fraction: float = 0.9,
) -> JointDataPlan:
    """Convert a tested uncertainty grid into an acquisition-time plan.

    The time accounting includes sampled calibration time, compass averaging,
    and a user-supplied overhead for each transition between calibration
    segments. It intentionally excludes unmodeled setup, warm-up, and operator
    time unless they are represented by ``per_transition_overhead_s``.
    """
    if not summaries:
        raise ValueError("summaries must not be empty")
    ####
    if not isfinite(sample_period_s) or sample_period_s <= 0.0:
        raise ValueError("sample_period_s must be finite and positive")
    ####
    if (
            isinstance(calibration_segment_count, bool)
            or not isinstance(calibration_segment_count, int)
            or calibration_segment_count < 1
    ):
        raise ValueError("calibration_segment_count must be a positive integer")
    ####
    if not isfinite(per_transition_overhead_s) or per_transition_overhead_s < 0.0:
        raise ValueError("per_transition_overhead_s must be finite and non-negative")
    ####
    if not isfinite(target_95_half_width_deg) or target_95_half_width_deg <= 0.0:
        raise ValueError("target_95_half_width_deg must be finite and positive")
    ####
    _validate_probability(minimum_empirical_coverage_95, "minimum_empirical_coverage_95")
    _validate_probability(minimum_accepted_fraction, "minimum_accepted_fraction")
    seen: set[tuple[int, float]] = set()
    provisional: list[JointDataPlanPoint] = []
    transition_count = max(0, calibration_segment_count - 1)
    for summary in summaries:
        if not isinstance(summary, HeadingUncertaintyCoverageSummary):
            raise TypeError("summaries must contain HeadingUncertaintyCoverageSummary values")
        ####
        key = (summary.calibration_samples_per_segment, summary.compass_duration_s)
        if key in seen:
            raise ValueError("summaries contain a duplicate calibration/compass allocation")
        ####
        seen.add(key)
        calibration_duration = (
            calibration_segment_count
            * summary.calibration_samples_per_segment
            * sample_period_s
            + transition_count * per_transition_overhead_s
        )
        accepted_fraction = summary.accepted_fraction
        empirically_validated = (
            summary.coverage_95 >= minimum_empirical_coverage_95
            and accepted_fraction >= minimum_accepted_fraction
        )
        provisional.append(
            JointDataPlanPoint(
                calibration_samples_per_segment=summary.calibration_samples_per_segment,
                compass_duration_s=summary.compass_duration_s,
                calibration_duration_s=calibration_duration,
                total_acquisition_duration_s=calibration_duration + summary.compass_duration_s,
                predicted_95_half_width_deg=summary.mean_predicted_95_half_width_deg,
                empirical_p95_absolute_error_deg=summary.empirical_p95_absolute_error_deg,
                empirical_coverage_95=summary.coverage_95,
                accepted_fraction=accepted_fraction,
                measurement_variance_fraction=summary.mean_measurement_variance_fraction,
                calibration_variance_fraction=summary.mean_calibration_variance_fraction,
                target_met=summary.mean_predicted_95_half_width_deg <= target_95_half_width_deg,
                empirically_validated=empirically_validated,
            )
        )
    ####
    points: list[JointDataPlanPoint] = []
    for candidate in provisional:
        dominated = any(
            other.total_acquisition_duration_s <= candidate.total_acquisition_duration_s
            and other.predicted_95_half_width_deg <= candidate.predicted_95_half_width_deg
            and (
                other.total_acquisition_duration_s < candidate.total_acquisition_duration_s
                or other.predicted_95_half_width_deg < candidate.predicted_95_half_width_deg
            )
            for other in provisional
        )
        points.append(
            JointDataPlanPoint(
                calibration_samples_per_segment=candidate.calibration_samples_per_segment,
                compass_duration_s=candidate.compass_duration_s,
                calibration_duration_s=candidate.calibration_duration_s,
                total_acquisition_duration_s=candidate.total_acquisition_duration_s,
                predicted_95_half_width_deg=candidate.predicted_95_half_width_deg,
                empirical_p95_absolute_error_deg=candidate.empirical_p95_absolute_error_deg,
                empirical_coverage_95=candidate.empirical_coverage_95,
                accepted_fraction=candidate.accepted_fraction,
                measurement_variance_fraction=candidate.measurement_variance_fraction,
                calibration_variance_fraction=candidate.calibration_variance_fraction,
                target_met=candidate.target_met,
                empirically_validated=candidate.empirically_validated,
                pareto_optimal=not dominated,
            )
        )
    ####
    return JointDataPlan(
        target_95_half_width_deg=target_95_half_width_deg,
        sample_period_s=sample_period_s,
        calibration_segment_count=calibration_segment_count,
        per_transition_overhead_s=per_transition_overhead_s,
        minimum_empirical_coverage_95=minimum_empirical_coverage_95,
        minimum_accepted_fraction=minimum_accepted_fraction,
        points=tuple(
            sorted(
                points,
                key=lambda point: (
                    point.total_acquisition_duration_s,
                    point.predicted_95_half_width_deg,
                ),
            )
        ),
    )
####
