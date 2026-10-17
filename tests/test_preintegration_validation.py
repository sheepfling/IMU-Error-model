from __future__ import annotations

from math import pi

import numpy as np
import pytest

from playground.calibration import preintegration
from playground.calibration.preintegration import (
    ConingTruth,
    ConvergencePoint,
    MinorImuIncrement,
    PreintegratedIncrement,
    ScullingTruth,
    apply_calibration_to_minor_increments,
    convergence_study,
    first_rate_meeting_error,
    naive_major_interval,
    preintegrate_piecewise_constant,
    recursive_coning_sculling,
    sample_minor_increments,
    so3_exp,
    so3_log,
)


def test_so3_small_angle_and_pi_paths() -> None:
    tiny = np.asarray((1e-10, -2e-10, 3e-10), dtype=float)
    assert np.allclose(so3_log(so3_exp(tiny)), tiny, atol=1e-12)
    half_turn = so3_log(so3_exp(np.asarray((pi, 0.0, 0.0), dtype=float)))
    assert np.linalg.norm(half_turn) == pytest.approx(pi, abs=1e-8)
####


def test_so3_and_rotation_integral_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="rotation_vector"):
        so3_exp(np.zeros(2))
    ####
    with pytest.raises(ValueError, match="rotation must"):
        so3_log(np.eye(2))
    ####
    with pytest.raises(ValueError, match="dt_s"):
        preintegration._rotation_integral(np.zeros(3), 0.0)
    ####
    assert np.allclose(preintegration._rotation_integral(np.zeros(3), 0.25), 0.25 * np.eye(3))
####


def test_increment_dataclasses_validate_shape_duration_and_metadata() -> None:
    with pytest.raises(ValueError, match="dt_s"):
        MinorImuIncrement(np.zeros(3), np.zeros(3), 0.0)
    ####
    with pytest.raises(ValueError, match="delta_theta_body"):
        MinorImuIncrement(np.zeros(2), np.zeros(3), 0.1)
    ####
    with pytest.raises(ValueError, match="rotation_start_from_end"):
        PreintegratedIncrement(np.zeros(3), np.zeros(3), np.eye(2), 1.0, 1, "test")
    ####
    with pytest.raises(ValueError, match="duration_s"):
        PreintegratedIncrement(np.zeros(3), np.zeros(3), np.eye(3), 0.0, 1, "test")
    ####
    with pytest.raises(ValueError, match="minor_count"):
        PreintegratedIncrement(np.zeros(3), np.zeros(3), np.eye(3), 1.0, 0, "test")
    ####
    with pytest.raises(ValueError, match="method"):
        PreintegratedIncrement(np.zeros(3), np.zeros(3), np.eye(3), 1.0, 1, "")
    ####
####


def test_aggregation_routes_reject_empty_samples() -> None:
    for function in (preintegrate_piecewise_constant, naive_major_interval, recursive_coning_sculling):
        with pytest.raises(ValueError, match="minor increment"):
            function(())
        ####
    ####
####


def test_truth_models_reject_nonphysical_configuration() -> None:
    with pytest.raises(ValueError, match="angular_rate_amplitude"):
        ConingTruth(angular_rate_amplitude_radps=0.0)
    ####
    with pytest.raises(ValueError, match="frequency_hz"):
        ConingTruth(frequency_hz=0.0)
    ####
    with pytest.raises(ValueError, match="angle_amplitude_rad"):
        ScullingTruth(angle_amplitude_rad=0.0)
    ####
    with pytest.raises(ValueError, match="force_amplitude_mps2"):
        ScullingTruth(force_amplitude_mps2=0.0)
    ####
    with pytest.raises(ValueError, match="frequency_hz"):
        ScullingTruth(frequency_hz=0.0)
    ####
####


def test_sampling_and_convergence_validate_schedule() -> None:
    truth = ConingTruth()
    with pytest.raises(ValueError, match="truth interval"):
        sample_minor_increments(truth, start_time_s=1.0, end_time_s=1.0, minor_count=1)
    ####
    with pytest.raises(ValueError, match="quadrature_substeps"):
        sample_minor_increments(truth, start_time_s=0.0, end_time_s=0.1, minor_count=1, quadrature_substeps=0)
    ####
    with pytest.raises(ValueError, match="minor_counts"):
        convergence_study(truth, start_time_s=0.0, end_time_s=0.1, minor_counts=(0,))
    ####
####


def test_rate_planner_validation_and_no_solution() -> None:
    point = ConvergencePoint(1, 10.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0)
    with pytest.raises(ValueError, match="non-negative"):
        first_rate_meeting_error((point,), attitude_error_rad=-1.0)
    ####
    with pytest.raises(ValueError, match="non-negative"):
        first_rate_meeting_error((point,), velocity_error_mps=-1.0)
    ####
    with pytest.raises(ValueError, match="method"):
        first_rate_meeting_error((point,), attitude_error_rad=1.0, method="unknown")
    ####
    assert first_rate_meeting_error((point,), attitude_error_rad=0.5) is None
    assert first_rate_meeting_error((point,), attitude_error_rad=1.0, velocity_error_mps=2.0, method="corrected") == 10.0
####


def test_calibration_minor_route_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="minor increment"):
        apply_calibration_to_minor_increments((), object())  # type: ignore[arg-type]
    ####
####
