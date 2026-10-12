"""Restricted, observable equipment fits and shared-reference uncertainty.

These are not a joint self-calibration of every sensor and table parameter.
Levers require calibrated accelerometer data and a surveyed orientation/rate
reference. Timing fits are conditional on an independent encoder timebase.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from .known_reference import CalibrationEstimate, calibration_coefficient_vectors
from .reference_equipment import AngleLaw, FloatArray, readonly, validate_intervals
from .reference_observations import ObservedCalibrationData, fit_observed_calibration


class UnobservableCalibrationError(ValueError):
    """Raised instead of fabricating certainty in a deficient design."""

    def __init__(self, rank: int, parameter_count: int, condition_number: float) -> None:
        self.rank = rank
        self.parameter_count = parameter_count
        self.condition_number = condition_number
        super().__init__(f"calibration is rank deficient or ill conditioned: rank {rank}/{parameter_count}, scaled condition {condition_number:g}")
    ####
####


def checked_design(design: FloatArray, *, maximum_condition: float = 1e8) -> tuple[FloatArray, FloatArray, int, float]:
    if not np.isfinite(maximum_condition) or maximum_condition < 1:
        raise ValueError("maximum_condition must be finite and at least one")
    ####
    x = np.asarray(design, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all() or x.shape[0] <= x.shape[1]:
        raise ValueError("design needs finite rows and more observations than parameters")
    ####
    scales = np.linalg.norm(x, axis=0)
    scales = np.where(scales > 0, scales, 1.0)
    scaled = x / scales
    singular = np.linalg.svd(scaled, compute_uv=False)
    rank = int(np.linalg.matrix_rank(scaled))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
    if rank < x.shape[1] or condition > maximum_condition:
        raise UnobservableCalibrationError(rank, x.shape[1], condition)
    ####
    return scaled, scales, rank, condition
####


@dataclass(frozen=True, slots=True)
class LeverArmEstimate:
    lever_arm_sensor_m: FloatArray
    residual_bias_mps2: FloatArray
    coefficient_covariance: FloatArray
    scaled_condition_number: float
    residual_rms_mps2: float
    covariance_assumption: str = "conditional_on_calibrated_accel_and_known_reference; iid_isotropic_rows"
####


def fit_lever_arm(calibrated_force: FloatArray, zero_lever_force: FloatArray, lever_operator: FloatArray, *, fit_residual_bias: bool = True) -> LeverArmEstimate:
    """Fit r (and optionally b) in f_measured-f_pivot = L r + b.

    L includes centrifugal, tangential, and retained Earth-Coriolis terms.
    A single fixed-axis spin cannot determine its axial lever component when
    Earth-Coriolis is absent. Weak Earth-rate information is not a practical
    substitute for multi-axis excitation; conditioning is checked as well.
    """
    f = np.asarray(calibrated_force, dtype=float)
    if f.ndim != 2 or f.shape[1] != 3 or len(f) < 3:
        raise ValueError("force observations must have shape (n, 3), n >= 3")
    ####
    n = len(f)
    measured = readonly(f, (n, 3), "calibrated force")
    origin = readonly(zero_lever_force, (n, 3), "zero lever force")
    lever = readonly(lever_operator, (n, 3, 3), "lever operator")
    x = lever.reshape(-1, 3)
    if fit_residual_bias:
        x = np.column_stack((x, np.tile(np.eye(3), (n, 1))))
    ####
    scaled, scales, _, condition = checked_design(x)
    y = (measured - origin).ravel()
    beta = np.linalg.lstsq(scaled, y, rcond=None)[0] / scales
    residual = y - x @ beta
    variance = float(residual @ residual / (len(y) - len(beta)))
    influence = np.linalg.pinv(scaled) / scales[:, None]
    covariance = variance * influence @ influence.T
    bias = beta[3:] if fit_residual_bias else np.zeros(3)
    return LeverArmEstimate(readonly(beta[:3], (3,), "lever"), readonly(bias, (3,), "bias"), readonly(covariance, covariance.shape, "covariance"), condition, float(np.sqrt(np.mean(residual ** 2))))
####


@dataclass(frozen=True, slots=True)
class EncoderTimingEstimate:
    gain: float
    bias_radps: float
    delay_s: float
    covariance_gain_bias_delay: FloatArray
    scaled_condition_number: float
    residual_rms_radps: float
    covariance_assumption: str = "conditional_on_independent_encoder_law; iid_sensor_rows"
####


def fit_encoder_timing(reference_law: AngleLaw, starts_s: FloatArray, ends_s: FloatArray, measured_rate: FloatArray, *, delay_bounds_s: tuple[float, float] = (-0.1, 0.1)) -> EncoderTimingEstimate:
    """Fit y = gain * mean(w_reference(t-delay)) + bias, for one channel.

    Positive delay means sensor output lags the aligned encoder. Gain must be
    positive. A bounded profile search is followed by a full Jacobian-rank test.
    The predeclared search interval must be shorter than half a harmonic period.
    Constant rate does NOT identify delay, even with unlimited samples.
    """
    starts, ends = validate_intervals(starts_s, ends_s)
    y = readonly(measured_rate, starts.shape, "measured rate")
    lo, hi = delay_bounds_s
    if not np.isfinite([lo, hi]).all() or hi <= lo or hi - lo >= 0.5 / reference_law.frequency_hz:
        raise ValueError("delay bounds must be finite, increasing, and narrower than half a period")
    ####
    # Fail early for flat references; profile fitting alone could return any delay.
    w0 = reference_law.mean_rate(starts, ends)
    d0 = (reference_law.evaluate(starts)[1] - reference_law.evaluate(ends)[1]) / (ends - starts)
    checked_design(np.column_stack((w0, np.ones(len(starts)), d0)))

    def profile(delay: float) -> tuple[float, FloatArray, FloatArray]:
        reference = reference_law.mean_rate(starts - delay, ends - delay)
        x = np.column_stack((reference, np.ones(len(starts))))
        coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
        residual = y - x @ coefficients
        cost = float(residual @ residual) if coefficients[0] > 0 else float("inf")
        return cost, coefficients, residual
    ####

    grid = np.linspace(lo, hi, 65)
    costs = np.array([profile(float(delay))[0] for delay in grid])
    best = int(np.argmin(costs))
    if best in (0, len(grid) - 1) or not np.isfinite(costs[best]):
        raise ValueError("timing optimum is outside/intersects the search boundary or has nonpositive gain")
    ####
    left, right = float(grid[best - 1]), float(grid[best + 1])
    ratio = (np.sqrt(5.0) - 1) / 2
    a, b = right - ratio * (right - left), left + ratio * (right - left)
    fa, fb = profile(a)[0], profile(b)[0]
    for _ in range(64):
        if fa < fb:
            right, b, fb = b, a, fa
            a = right - ratio * (right - left)
            fa = profile(a)[0]
        else:
            left, a, fa = a, b, fb
            b = left + ratio * (right - left)
            fb = profile(b)[0]
        ####
    ####
    delay = 0.5 * (left + right)
    _, coefficients, residual = profile(delay)
    gain, bias = map(float, coefficients)
    reference = reference_law.mean_rate(starts - delay, ends - delay)
    derivative = gain * (reference_law.evaluate(starts - delay)[1] - reference_law.evaluate(ends - delay)[1]) / (ends - starts)
    jacobian = np.column_stack((reference, np.ones(len(starts)), derivative))
    scaled, scales, _, condition = checked_design(jacobian)
    influence = np.linalg.pinv(scaled) / scales[:, None]
    covariance = float(residual @ residual / (len(y) - 3)) * influence @ influence.T
    return EncoderTimingEstimate(gain, bias, delay, readonly(covariance, (3, 3), "timing covariance"), condition, float(np.sqrt(np.mean(residual ** 2))))
####


@dataclass(frozen=True, slots=True)
class ReferenceUncertaintyResult:
    """Delta-method sensitivity to run-shared equipment calibration parameters."""

    calibration: CalibrationEstimate
    conditional_sensor_covariance: FloatArray
    shared_reference_covariance: FloatArray
    coefficient_reference_jacobian: FloatArray
    reference_parameter_names: tuple[str, ...]
####


def propagate_shared_reference_uncertainty(builder: Callable[[FloatArray], ObservedCalibrationData], parameter_mean: FloatArray, parameter_covariance: FloatArray, *, finite_difference_steps: FloatArray, parameter_names: tuple[str, ...]) -> ReferenceUncertaintyResult:
    """Add J_ref P_ref J_ref.T without treating a common error as N fresh noises.

    Builder reprocesses the SAME observed records with perturbed external survey/
    encoder-calibration estimates; it must not regenerate observations or access
    hidden truth. Inputs are independently established metrology means/covariance.
    Small-error linearization and independence from sensor noise are assumed.
    This is uncertainty propagation, NOT bias correction for general noisy
    regressors or a joint errors-in-variables/posterior estimator.
    """
    mean = np.asarray(parameter_mean, dtype=float)
    if mean.ndim != 1 or len(mean) == 0 or len(parameter_names) != len(mean) or len(set(parameter_names)) != len(mean) or any(not n for n in parameter_names):
        raise ValueError("unique nonempty names must match the reference parameter vector")
    ####
    m = len(mean)
    mean = readonly(mean, (m,), "reference mean")
    steps = readonly(finite_difference_steps, (m,), "finite-difference steps")
    covariance = readonly(parameter_covariance, (m, m), "reference covariance")
    if np.any(steps <= 0) or not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-15) or np.linalg.eigvalsh(covariance).min() < 0:
        raise ValueError("steps must be positive and reference covariance must be symmetric PSD")
    ####
    original = builder(mean.copy())
    base = fit_observed_calibration(original)

    def fit_candidate(parameters: FloatArray) -> CalibrationEstimate:
        candidate = builder(parameters)
        if not isinstance(candidate, ObservedCalibrationData):
            raise TypeError("reference builder must return observed data")
        ####
        if candidate.dataset.labels != original.dataset.labels:
            raise ValueError("shared-reference perturbations must reuse the same measured recording")
        ####
        for field in ("start_times", "end_times", "temperatures_celsius", "measured_acceleration", "measured_angular_rate"):
            if not np.array_equal(getattr(candidate.dataset, field), getattr(original.dataset, field), equal_nan=True):
                raise ValueError("shared-reference perturbations must reuse the same measured recording")
            ####
        ####
        return fit_observed_calibration(candidate)
    ####

    jacobian = np.empty((33, m))
    for i in range(m):
        plus, minus = mean.copy(), mean.copy()
        plus[i] += steps[i]
        minus[i] -= steps[i]
        p = np.concatenate(calibration_coefficient_vectors(fit_candidate(plus)))
        n = np.concatenate(calibration_coefficient_vectors(fit_candidate(minus)))
        jacobian[:, i] = (p - n) / (2 * steps[i])
    ####
    shared = jacobian @ covariance @ jacobian.T
    conditional = base.joint_coefficient_covariance
    total = conditional + shared
    total = 0.5 * (total + total.T)
    updated = replace(base, accelerometer_coefficient_covariance=total[:12, :12], gyroscope_coefficient_covariance=total[12:, 12:], accelerometer_gyroscope_coefficient_cross_covariance=total[:12, 12:], covariance_assumption="conditional_iid_sensor + independent_run_shared_reference_delta_method; no_EIV_bias_correction")
    return ReferenceUncertaintyResult(updated, readonly(conditional, (33, 33), "conditional covariance"), readonly(shared, (33, 33), "shared covariance"), readonly(jacobian, (33, m), "reference Jacobian"), parameter_names)
####
