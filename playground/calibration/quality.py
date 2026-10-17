"""Stationarity gates and quality-aware static gyrocompass execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from numpy import asarray, concatenate, isfinite as array_isfinite, ndarray
from numpy.linalg import norm

from .earth_reference import LocalEarthReference, WGS84_EARTH_ROTATION_RADPS
from .gyrocompass import GyrocompassResult, ImuMeasurementBatch, apply_calibration, static_gyrocompass
from .heading_uncertainty import HeadingUncertaintyEstimate, propagate_heading_uncertainty
from .time_series import estimate_mean_covariance
from .known_reference import CalibrationEstimate


class StationarityFlag(StrEnum):
    """Named reasons that a static alignment record is not admissible."""

    INSUFFICIENT_SAMPLES = "insufficient_samples"
    INSUFFICIENT_DURATION = "insufficient_duration"
    GRAVITY_MAGNITUDE_MISMATCH = "gravity_magnitude_mismatch"
    ACCELERATION_VARIATION = "acceleration_variation"
    ANGULAR_RATE_VARIATION = "angular_rate_variation"
    ACCELERATION_SPLIT_DRIFT = "acceleration_split_drift"
    ANGULAR_RATE_SPLIT_DRIFT = "angular_rate_split_drift"
    EARTH_RATE_GEOMETRY_ILL_CONDITIONED = "earth_rate_geometry_ill_conditioned"
####


@dataclass(frozen=True, slots=True)
class StationarityThresholds:
    """Engineering thresholds used to admit a nominally static record."""

    minimum_sample_count: int = 100
    minimum_duration_s: float = 2.0
    maximum_gravity_magnitude_error_mps2: float = 0.25
    maximum_acceleration_norm_std_mps2: float = 0.05
    maximum_angular_rate_axis_std_radps: float = 5e-4
    maximum_acceleration_split_mean_delta_mps2: float = 0.05
    maximum_angular_rate_split_mean_delta_radps: float = 2e-5
    minimum_horizontal_earth_rate_radps: float = 1e-8

    def __post_init__(self) -> None:
        if (
                isinstance(self.minimum_sample_count, bool)
                or not isinstance(self.minimum_sample_count, int)
                or self.minimum_sample_count < 2
        ):
            raise ValueError("minimum_sample_count must be an integer >= 2")
        ####
        for field_name in (
                "minimum_duration_s",
                "maximum_gravity_magnitude_error_mps2",
                "maximum_acceleration_norm_std_mps2",
                "maximum_angular_rate_axis_std_radps",
                "maximum_acceleration_split_mean_delta_mps2",
                "maximum_angular_rate_split_mean_delta_radps",
                "minimum_horizontal_earth_rate_radps",
        ):
            value = getattr(self, field_name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")
            ####
        ####
    ####
####


@dataclass(frozen=True, slots=True)
class StationarityReport:
    """Measured stationarity metrics and all failed gates."""

    sample_count: int
    duration_s: float
    gravity_magnitude_mps2: float
    gravity_magnitude_error_mps2: float
    acceleration_norm_std_mps2: float
    maximum_angular_rate_axis_std_radps: float
    acceleration_split_mean_delta_mps2: float
    angular_rate_split_mean_delta_radps: float
    horizontal_earth_rate_radps: float | None
    flags: tuple[StationarityFlag, ...]

    @property
    def accepted(self) -> bool:
        return not self.flags
    ####
####


@dataclass(frozen=True, slots=True)
class QualityAwareGyrocompassResult:
    """Static alignment result with admissibility and first-order confidence."""

    result: GyrocompassResult | None
    stationarity: StationarityReport
    uncertainty: HeadingUncertaintyEstimate | None
    calibrated_acceleration_mean: ndarray
    calibrated_angular_rate_mean: ndarray
    measurement_mean_covariance: ndarray | None
    measurement_effective_sample_size: ndarray | None = None
    covariance_method: str | None = None
    covariance_max_lag: int | None = None

    @property
    def accepted(self) -> bool:
        return self.result is not None and self.stationarity.accepted
    ####

    def meets_95_confidence(self, target_half_width_deg: float) -> bool:
        """Return whether the accepted result reaches a requested 95% half-width."""
        if not isfinite(target_half_width_deg) or target_half_width_deg <= 0.0:
            raise ValueError("target_half_width_deg must be finite and positive")
        ####
        return bool(
            self.accepted
            and self.uncertainty is not None
            and self.uncertainty.confidence_95_half_width_deg <= target_half_width_deg
        )
    ####
####


def _validated_samples(value: ndarray, field_name: str) -> ndarray:
    samples = asarray(value, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 3 or samples.shape[0] < 2 or not bool(array_isfinite(samples).all()):
        raise ValueError(f"{field_name} must be finite with shape (n, 3) and n >= 2")
    ####
    return samples
####


def evaluate_stationarity(
        acceleration: ndarray,
        angular_rate: ndarray,
        *,
        sample_period_s: float,
        expected_gravity_mps2: float,
        thresholds: StationarityThresholds | None = None,
        horizontal_earth_rate_radps: float | None = None,
) -> StationarityReport:
    """Evaluate static-record admissibility from calibrated rate samples."""
    accel = _validated_samples(acceleration, "acceleration")
    gyro = _validated_samples(angular_rate, "angular_rate")
    if accel.shape != gyro.shape:
        raise ValueError("acceleration and angular_rate must have the same shape")
    ####
    if not isfinite(sample_period_s) or sample_period_s <= 0.0:
        raise ValueError("sample_period_s must be finite and positive")
    ####
    if not isfinite(expected_gravity_mps2) or expected_gravity_mps2 <= 0.0:
        raise ValueError("expected_gravity_mps2 must be finite and positive")
    ####
    if horizontal_earth_rate_radps is not None and (
            not isfinite(horizontal_earth_rate_radps) or horizontal_earth_rate_radps < 0.0
    ):
        raise ValueError("horizontal_earth_rate_radps must be finite and non-negative when provided")
    ####
    active = thresholds or StationarityThresholds()
    sample_count = accel.shape[0]
    duration_s = sample_count * sample_period_s
    acceleration_mean = accel.mean(axis=0)
    gravity_magnitude = float(norm(acceleration_mean))
    gravity_error = abs(gravity_magnitude - expected_gravity_mps2)
    acceleration_norm_std = float(norm(accel, axis=1).std(ddof=1))
    angular_rate_axis_std = float(gyro.std(axis=0, ddof=1).max())
    split_index = sample_count // 2
    acceleration_split_delta = float(norm(accel[:split_index].mean(axis=0) - accel[split_index:].mean(axis=0)))
    angular_rate_split_delta = float(norm(gyro[:split_index].mean(axis=0) - gyro[split_index:].mean(axis=0)))

    flags: list[StationarityFlag] = []
    if sample_count < active.minimum_sample_count:
        flags.append(StationarityFlag.INSUFFICIENT_SAMPLES)
    ####
    if duration_s < active.minimum_duration_s:
        flags.append(StationarityFlag.INSUFFICIENT_DURATION)
    ####
    if gravity_error > active.maximum_gravity_magnitude_error_mps2:
        flags.append(StationarityFlag.GRAVITY_MAGNITUDE_MISMATCH)
    ####
    if acceleration_norm_std > active.maximum_acceleration_norm_std_mps2:
        flags.append(StationarityFlag.ACCELERATION_VARIATION)
    ####
    if angular_rate_axis_std > active.maximum_angular_rate_axis_std_radps:
        flags.append(StationarityFlag.ANGULAR_RATE_VARIATION)
    ####
    if acceleration_split_delta > active.maximum_acceleration_split_mean_delta_mps2:
        flags.append(StationarityFlag.ACCELERATION_SPLIT_DRIFT)
    ####
    if angular_rate_split_delta > active.maximum_angular_rate_split_mean_delta_radps:
        flags.append(StationarityFlag.ANGULAR_RATE_SPLIT_DRIFT)
    ####
    if (
            horizontal_earth_rate_radps is not None
            and horizontal_earth_rate_radps < active.minimum_horizontal_earth_rate_radps
    ):
        flags.append(StationarityFlag.EARTH_RATE_GEOMETRY_ILL_CONDITIONED)
    ####
    return StationarityReport(
        sample_count=sample_count,
        duration_s=duration_s,
        gravity_magnitude_mps2=gravity_magnitude,
        gravity_magnitude_error_mps2=gravity_error,
        acceleration_norm_std_mps2=acceleration_norm_std,
        maximum_angular_rate_axis_std_radps=angular_rate_axis_std,
        acceleration_split_mean_delta_mps2=acceleration_split_delta,
        angular_rate_split_mean_delta_radps=angular_rate_split_delta,
        horizontal_earth_rate_radps=horizontal_earth_rate_radps,
        flags=tuple(flags),
    )
####


def quality_aware_static_gyrocompass(
        batch: ImuMeasurementBatch,
        calibration: CalibrationEstimate,
        *,
        latitude_rad: float,
        height_m: float = 0.0,
        earth_rotation_radps: float = WGS84_EARTH_ROTATION_RADPS,
        thresholds: StationarityThresholds | None = None,
        covariance_max_lag: int | None = None,
) -> QualityAwareGyrocompassResult:
    """Calibrate, gate, solve, and propagate confidence for one static record."""
    calibrated_accel, calibrated_gyro = apply_calibration(
        batch.acceleration,
        batch.angular_rate,
        calibration,
    )
    earth = LocalEarthReference(
        latitude_rad=latitude_rad,
        height_m=height_m,
        earth_rotation_radps=earth_rotation_radps,
    )
    stationarity = evaluate_stationarity(
        calibrated_accel,
        calibrated_gyro,
        sample_period_s=batch.sample_period_s,
        expected_gravity_mps2=float(earth.gravity_ned_mps2[2]),
        thresholds=thresholds,
        horizontal_earth_rate_radps=earth.horizontal_earth_rate_radps,
    )
    accel_mean = calibrated_accel.mean(axis=0)
    gyro_mean = calibrated_gyro.mean(axis=0)
    if not stationarity.accepted:
        return QualityAwareGyrocompassResult(
            result=None,
            stationarity=stationarity,
            uncertainty=None,
            calibrated_acceleration_mean=accel_mean,
            calibrated_angular_rate_mean=gyro_mean,
            measurement_mean_covariance=None,
            measurement_effective_sample_size=None,
            covariance_method=None,
            covariance_max_lag=None,
        )
    ####
    result = static_gyrocompass(
        accel_mean,
        gyro_mean,
        latitude_rad=latitude_rad,
        earth_rotation_radps=earth_rotation_radps,
        gravity_mps2=float(earth.gravity_ned_mps2[2]),
    )
    raw_samples = concatenate((batch.acceleration, batch.angular_rate), axis=1)
    active_lag = (
        min(batch.sample_count - 1, max(0, round(batch.sample_count ** (1.0 / 3.0))))
        if covariance_max_lag is None
        else covariance_max_lag
    )
    mean_covariance = estimate_mean_covariance(raw_samples, max_lag=active_lag)
    uncertainty = propagate_heading_uncertainty(
        batch.acceleration.mean(axis=0),
        batch.angular_rate.mean(axis=0),
        mean_covariance.covariance,
        calibration,
        latitude_rad=latitude_rad,
        earth_rotation_radps=earth_rotation_radps,
        gravity_mps2=float(earth.gravity_ned_mps2[2]),
    )
    return QualityAwareGyrocompassResult(
        result=GyrocompassResult(
            heading_rad=result.heading_rad,
            attitude_nav_from_body=result.attitude_nav_from_body,
            earth_rate_horizontal_radps=result.earth_rate_horizontal_radps,
            earth_rate_residual_radps=result.earth_rate_residual_radps,
            sample_count=batch.sample_count,
        ),
        stationarity=stationarity,
        uncertainty=uncertainty,
        calibrated_acceleration_mean=accel_mean,
        calibrated_angular_rate_mean=gyro_mean,
        measurement_mean_covariance=mean_covariance.covariance,
        measurement_effective_sample_size=mean_covariance.effective_sample_size,
        covariance_method=mean_covariance.method,
        covariance_max_lag=mean_covariance.max_lag,
    )
####
