"""Auditable mapping from IMU error mechanisms to calibration-playground excitation routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from imu_error_model import AxisConfig, ImuConfig, AcquisitionConfig, ClockConfig, FirFilterConfig, ThermalLagConfig


CoverageStatus = Literal["covered", "diagnostic", "not_calibration"]


@dataclass(frozen=True, slots=True)
class ExcitationCoverage:
    """One modeled effect and the playground route that exposes it."""

    scope: Literal["axis", "imu"]
    field_name: str
    status: CoverageStatus
    excitation_route: str
    estimator_or_diagnostic: str
####


AXIS_EXCITATION_COVERAGE: tuple[ExcitationCoverage, ...] = (
    ExcitationCoverage("axis", "white_noise_density", "covered", "stationary duration sweep", "Allan/Monte Carlo/heading confidence"),
    ExcitationCoverage("axis", "turn_on_bias_std", "covered", "reset ensemble", "between-run bias statistics"),
    ExcitationCoverage("axis", "bias_std", "covered", "long stationary record", "random-walk or Gauss-Markov characterization"),
    ExcitationCoverage("axis", "bias_correlation_time", "covered", "long stationary record", "autocorrelation/Allan characterization"),
    ExcitationCoverage("axis", "flicker_bias_std", "covered", "long stationary record", "finite-band Allan plateau"),
    ExcitationCoverage("axis", "flicker_min_correlation_time", "covered", "long stationary record", "Allan-band shape"),
    ExcitationCoverage("axis", "flicker_max_correlation_time", "covered", "long stationary record", "Allan-band shape"),
    ExcitationCoverage("axis", "flicker_components", "diagnostic", "long stationary record", "implementation-fidelity convergence"),
    ExcitationCoverage("axis", "fixed_bias", "covered", "symmetric static/known-rate references", "linear intrinsic fit"),
    ExcitationCoverage("axis", "scale_factor", "covered", "multi-amplitude known references", "linear intrinsic fit"),
    ExcitationCoverage("axis", "sensitivity_matrix", "covered", "multi-axis known references", "full-matrix intrinsic fit"),
    ExcitationCoverage("axis", "nonlinear_factor", "covered", "multi-amplitude sweep", "model-mismatch/nonlinear fit"),
    ExcitationCoverage("axis", "misalignment_std", "covered", "reset ensemble + multi-axis references", "realized transform distribution"),
    ExcitationCoverage("axis", "misalignment_covariance", "covered", "reset ensemble + multi-axis references", "realized transform covariance"),
    ExcitationCoverage("axis", "measurement_range", "diagnostic", "over-range amplitude sweep", "saturation onset"),
    ExcitationCoverage("axis", "thermal_bias_coefficient", "covered", "multi-temperature pose sweep", "thermal affine fit"),
    ExcitationCoverage("axis", "thermal_noise_coefficient", "covered", "multi-temperature stationary records", "noise-vs-temperature fit"),
    ExcitationCoverage("axis", "thermal_scale_factor_coefficient", "covered", "multi-temperature pose sweep", "thermal affine fit"),
    ExcitationCoverage("axis", "thermal_sensitivity_matrix_coefficient", "covered", "multi-temperature multi-axis pose sweep", "full thermal cross-axis matrix fit"),
    ExcitationCoverage("axis", "reference_temperature_celsius", "not_calibration", "temperature metadata", "reference convention"),
    ExcitationCoverage("axis", "noise_covariance", "covered", "stationary record", "sample covariance"),
    ExcitationCoverage("axis", "quantization_step", "diagnostic", "fine low-amplitude ramp", "quantization staircase"),
    ExcitationCoverage("axis", "apply_clipping", "diagnostic", "over-range amplitude sweep", "electronics policy"),
    ExcitationCoverage("axis", "output_scale", "covered", "known magnitude reference", "overall gain fit"),
)

IMU_EXCITATION_COVERAGE: tuple[ExcitationCoverage, ...] = (
    ExcitationCoverage("imu", "accelerometer", "not_calibration", "channel configuration", "container field"),
    ExcitationCoverage("imu", "gyroscope", "not_calibration", "channel configuration", "container field"),
    ExcitationCoverage("imu", "gyroscope_specific_force_sensitivity", "covered", "symmetric ±g poses", "gyro g-sensitivity fit"),
    ExcitationCoverage("imu", "gyroscope_specific_force_sensitivity_temperature_coefficient", "covered", "symmetric ±g poses across temperature", "thermal gyro g-sensitivity fit"),
    ExcitationCoverage("imu", "output_scale_accelerometer", "covered", "known acceleration magnitude", "overall gain fit"),
    ExcitationCoverage("imu", "output_scale_gyroscope", "covered", "known rotation magnitude", "overall gain fit"),
)


SYSTEM_EXCITATION_COVERAGE: tuple[ExcitationCoverage, ...] = (
    ExcitationCoverage("imu", "wgs84_earth_rotation", "covered", "dynamic local NED Earth reference with traceable rate override", "gyrocompass/slow-turntable fidelity"),
    ExcitationCoverage("imu", "wgs84_normal_gravity", "covered", "latitude/height local gravity", "pose and gyrocompass reference"),
    ExcitationCoverage("imu", "turntable_modulation", "covered", "time-varying body orientation on local-down table", "Ferraris scale/cross-axis sensitivity"),
    ExcitationCoverage("imu", "maytag_modulation", "covered", "1/2/4 static down-axis positions", "turn-on bias rejection"),
    ExcitationCoverage("imu", "data_confidence", "covered", "duration and Monte Carlo sweep", "empirical p95/RMS heading confidence"),
    ExcitationCoverage("imu", "calibration_covariance", "covered", "multivariate known-reference residuals and design geometry", "full coefficient covariance"),
    ExcitationCoverage("imu", "heading_covariance", "covered", "calibration covariance plus finite-record mean covariance", "first-order 95% heading interval and variance attribution"),
    ExcitationCoverage("imu", "stationarity_gate", "covered", "controlled harmonic force/rate contamination", "gravity, variation, and split-record drift gates"),
    ExcitationCoverage("imu", "continuous_carousel", "covered", "exact interval-average rotating-frame Earth reference", "sensor-fixed horizontal bias modulation and demodulation"),
    ExcitationCoverage("imu", "sample_aperture", "covered", "finite-angle turntable and carousel samples", "shared exact average frame transform rather than midpoint approximation"),
    ExcitationCoverage("imu", "serial_correlation_confidence", "covered", "long-run covariance with selectable lag window", "colored-noise mean covariance sensitivity"),
    ExcitationCoverage("imu", "calibration_serial_correlation", "covered", "label-wise moving-block resampling", "bootstrap calibration coefficient covariance"),
    ExcitationCoverage("imu", "effective_sample_size", "covered", "iid/HAC finite-record covariance", "per-channel effective independent sample count"),
    ExcitationCoverage("imu", "prefix_time_to_confidence", "covered", "quality-gated prefixes of one accumulating record", "first tested duration meeting a 95% heading target"),
    ExcitationCoverage("imu", "maytag_confidence", "covered", "1/2/4-position duration sweeps", "empirical modulation confidence versus total test time"),
    ExcitationCoverage("imu", "joint_data_plan", "covered", "paired calibration/compass grid", "Pareto and least-acquisition-time confidence target"),
    ExcitationCoverage("imu", "earth_rate_geometry_gate", "covered", "polar and near-polar local references", "pre-TRIAD horizontal Earth-rate observability rejection"),
)

def unclassified_config_fields() -> tuple[str, ...]:
    """Return configuration fields lacking an explicit excitation decision."""
    axis_fields = set(AxisConfig.model_fields)
    imu_fields = set(ImuConfig.model_fields)
    classified_axis = {item.field_name for item in AXIS_EXCITATION_COVERAGE}
    classified_imu = {item.field_name for item in IMU_EXCITATION_COVERAGE}
    missing = tuple(sorted({f"AxisConfig.{name}" for name in axis_fields - classified_axis} | {f"ImuConfig.{name}" for name in imu_fields - classified_imu}))
    return missing
####


@dataclass(frozen=True, slots=True)
class CalibrationRouteCoverage:
    """Named calibration or alignment route and its executable evidence path."""

    route_name: str
    status: CoverageStatus
    excitation_route: str
    estimator_or_diagnostic: str
    evidence: str
####


CALIBRATION_ROUTE_COVERAGE: tuple[CalibrationRouteCoverage, ...] = (
    CalibrationRouteCoverage(
        "known_reference_full_affine",
        "covered",
        "symmetric ±g poses and ±axis rates",
        "joint accelerometer/gyro affine least squares",
        "noiseless recovery, observability, covariance, and Monte Carlo tests",
    ),
    CalibrationRouteCoverage(
        "imucal_ferraris_hand_motion",
        "covered",
        "six static poses and three integrated rotations",
        "source-aligned Ferraris equations plus optional imucal adapter",
        "equation parity and optional external-package parity tests",
    ),
    CalibrationRouteCoverage(
        "imucal_ferraris_turntable",
        "covered",
        "WGS-84-aware constant-rate table rotations",
        "Ferraris turntable calibration representation",
        "Earth-free identity and Earth-rate contamination tests",
    ),
    CalibrationRouteCoverage(
        "thermal_affine_and_cross_axis",
        "covered",
        "multi-temperature multi-axis poses",
        "bias, scale, cross-axis, and gyro g-sensitivity temperature fits",
        "exact recovery and out-of-temperature model-mismatch tests",
    ),
    CalibrationRouteCoverage(
        "nonlinear_saturation_quantization",
        "diagnostic",
        "multi-amplitude ramps and over-range sweeps",
        "response-shape and failure-onset diagnostics",
        "excitation coverage audit; dedicated production estimator remains optional",
    ),
    CalibrationRouteCoverage(
        "stochastic_characterization",
        "covered",
        "long stationary records and reset ensembles",
        "Allan, autocorrelation, Monte Carlo, IID/HAC covariance",
        "white, Gauss-Markov, random-walk, flicker, and reset tests",
    ),
    CalibrationRouteCoverage(
        "static_gyrocompass",
        "covered",
        "stationary apparent gravity plus local Earth rate",
        "TRIAD alignment with stationarity gates and propagated covariance",
        "noiseless/noisy recovery, prefix-confidence, and coverage tests",
    ),
    CalibrationRouteCoverage(
        "maytag_gyrocompass",
        "covered",
        "one/two/four static down-axis positions",
        "sensor-fixed horizontal bias modulation",
        "single-realization bias preservation and empirical duration sweeps",
    ),
    CalibrationRouteCoverage(
        "continuous_carousel_gyrocompass",
        "covered",
        "known continuous down-axis rotation",
        "exact aperture demodulation and table-rate removal",
        "noiseless recovery, fixed-bias rejection, and rate/duration sweeps",
    ),
    CalibrationRouteCoverage(
        "correlated_calibration_uncertainty",
        "covered",
        "label-wise circular moving-block resampling",
        "bootstrap coefficient covariance",
        "PSD covariance and colored-bias inflation tests",
    ),
    CalibrationRouteCoverage(
        "joint_data_budget_planning",
        "covered",
        "paired calibration-size and compass-duration grid",
        "validated confidence target, Pareto front, and least-time allocation",
        "planner tests and tranche-four machine-readable campaign",
    ),
    CalibrationRouteCoverage(
        "matched_acquisition_reference",
        "covered",
        "independent readout/clock identification plus filtered/delayed calibration records",
        "forward-matched output-channel regressors; no FIR inversion",
        "matched intrinsic recovery, held-out frequency/reversal prediction, and support-covariance tests",
    ),
    CalibrationRouteCoverage(
        "high_rate_inertial_preintegration",
        "covered",
        "circular coning and orthogonal sculling cycles with high-rate raw increments",
        "independent SO(3) piecewise truth oracle plus recursive small-angle comparison",
        "coning/sculling zero-mean failure cases, sample-rate convergence, and calibrated-minor tests",
    ),
    CalibrationRouteCoverage(
        "multichannel_relative_clock",
        "covered",
        "multi-frequency six-channel known timing reference",
        "profiled gains/biases plus common offset/frequency map with full-rank Jacobian gate",
        "clock recovery, held-out frequency generalization, and constant-rate gauge rejection",
    ),
)


def routes_without_executable_coverage() -> tuple[str, ...]:
    """Return routes that are neither covered nor explicitly diagnostic-only."""
    return tuple(
        route.route_name
        for route in CALIBRATION_ROUTE_COVERAGE
        if route.status not in {"covered", "diagnostic"}
    )
####

# Acquisition is a distinct public configuration; diagnostics do not imply a fitted
# production estimator for every physical parameter.

ACQUISITION_FIELD_EVIDENCE: dict[str, tuple[str, str]] = {
    "AcquisitionConfig.schema_version": ("not_calibration", "tests/test_acquisition.py: invalid configuration tests"),
    "AcquisitionConfig.sample_period_s": ("covered", "tests/test_acquisition.py: identity, cadence, covariance tests"),
    "AcquisitionConfig.accelerometer_filter": ("covered", "tests/test_acquisition_calibration_routes.py: known-input FIR recovery"),
    "AcquisitionConfig.gyroscope_filter": ("covered", "tests/test_acquisition_calibration_routes.py: known-input FIR recovery"),
    "AcquisitionConfig.accelerometer_delay_samples": ("diagnostic", "tests/test_acquisition_calibration_routes.py: effective kernel only, delay/taps not separately identifiable"),
    "AcquisitionConfig.gyroscope_delay_samples": ("diagnostic", "tests/test_acquisition.py: delay response; effective-kernel recovery"),
    "AcquisitionConfig.thermal": ("diagnostic", "tests/test_acquisition.py: exact temperature step and deterministic mean"),
    "AcquisitionConfig.clock": ("covered", "tests/test_matched_acquisition_calibration.py: matched clock correction and multichannel relative-clock fit"),
    "FirFilterConfig.taps": ("covered", "tests/test_acquisition.py: independent convolution and frequency response"),
    "ThermalLagConfig.time_constant_s": ("diagnostic", "tests/test_acquisition.py: thermal step, subdivision, settling budget; no lag estimator yet"),
    "ThermalLagConfig.initial_die_temperature_celsius": ("diagnostic", "tests/test_acquisition.py: step and checkpoint replay"),
    "ClockConfig.epoch_s": ("not_calibration", "tests/test_acquisition.py: local-clock epoch law"),
    "ClockConfig.offset_s": ("covered", "tests/test_matched_acquisition_calibration.py: conditional multichannel relative-clock recovery"),
    "ClockConfig.frequency_error_ppm": ("covered", "tests/test_matched_acquisition_calibration.py: conditional multichannel relative-clock recovery and gauge rejection"),
    "ClockConfig.timestamp_jitter_bound_s": ("diagnostic", "tests/test_acquisition.py: monotonic/shared boundaries"),
    "ClockConfig.jitter_seed": ("not_calibration", "tests/test_acquisition.py: RNG isolation and replay"),
}


def unclassified_acquisition_fields() -> tuple[str, ...]:
    """Require an explicit evidence decision for every nested acquisition field."""
    actual = {
        f"{model.__name__}.{name}"
        for model in (AcquisitionConfig, FirFilterConfig, ThermalLagConfig, ClockConfig)
        for name in model.model_fields
    }
    return tuple(sorted(actual-set(ACQUISITION_FIELD_EVIDENCE)))
####
