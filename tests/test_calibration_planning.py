import pytest

from playground.calibration.data_confidence import HeadingUncertaintyCoverageSummary
from playground.calibration.planning import build_joint_data_plan


def _summary(
        calibration_samples: int,
        compass_duration_s: float,
        half_width_deg: float,
        *,
        coverage_95: float = 0.95,
        measurement_fraction: float = 0.5,
) -> HeadingUncertaintyCoverageSummary:
    return HeadingUncertaintyCoverageSummary(
        calibration_samples_per_segment=calibration_samples,
        compass_duration_s=compass_duration_s,
        trial_count=20,
        accepted_trial_count=20,
        empirical_standard_deviation_deg=half_width_deg / 1.96,
        empirical_p95_absolute_error_deg=half_width_deg,
        mean_predicted_standard_deviation_deg=half_width_deg / 1.96,
        mean_predicted_95_half_width_deg=half_width_deg,
        coverage_95=coverage_95,
        mean_measurement_variance_fraction=measurement_fraction,
        mean_calibration_variance_fraction=1.0 - measurement_fraction,
    )
####


def test_joint_plan_selects_shortest_validated_target_and_pareto_front() -> None:
    plan = build_joint_data_plan(
        (
            _summary(20, 30.0, 0.95, measurement_fraction=0.5),
            _summary(20, 120.0, 0.75, measurement_fraction=0.2),
            _summary(100, 30.0, 0.75, measurement_fraction=0.8),
            _summary(100, 120.0, 0.45, measurement_fraction=0.55),
            _summary(500, 30.0, 0.70, measurement_fraction=0.95),
            _summary(500, 120.0, 0.36, measurement_fraction=0.85),
        ),
        sample_period_s=0.2,
        calibration_segment_count=13,
        target_95_half_width_deg=0.5,
    )

    recommendation = plan.recommendation
    assert recommendation is not None
    assert recommendation.calibration_samples_per_segment == 100
    assert recommendation.compass_duration_s == 120.0
    assert recommendation.total_acquisition_duration_s == 380.0
    assert recommendation.limiting_source == "measurement"
    assert all(point.pareto_optimal for point in plan.pareto_front)
    assert not next(
        point
        for point in plan.points
        if point.calibration_samples_per_segment == 500 and point.compass_duration_s == 30.0
    ).pareto_optimal
####


def test_joint_plan_accounts_for_transition_overhead_and_validation_gate() -> None:
    plan = build_joint_data_plan(
        (
            _summary(20, 60.0, 0.4, coverage_95=0.5),
            _summary(100, 60.0, 0.45, coverage_95=0.95),
        ),
        sample_period_s=0.1,
        calibration_segment_count=13,
        per_transition_overhead_s=5.0,
        target_95_half_width_deg=0.5,
    )

    recommendation = plan.recommendation
    assert recommendation is not None
    assert recommendation.calibration_samples_per_segment == 100
    assert recommendation.calibration_duration_s == 190.0
    assert recommendation.total_acquisition_duration_s == 250.0
####


def test_joint_plan_rejects_duplicate_allocations() -> None:
    duplicate = _summary(20, 30.0, 1.0)
    with pytest.raises(ValueError, match="duplicate"):
        build_joint_data_plan(
            (duplicate, duplicate),
            sample_period_s=0.1,
        )
    ####
####
