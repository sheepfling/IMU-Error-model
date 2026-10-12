"""Encoder observations and surveyed-reference reconstruction without truth access."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Annotated, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from imu_error_model.acquisition import AcquiredImuModelProtocol

from .earth_reference import LocalEarthReference
from .known_reference import CalibrationDataset, CalibrationEstimate, fit_known_reference_calibration
from .reference_equipment import AngleLaw, ApertureReference, Finite, FloatArray, RateTableReference, TableGeometry, aperture_reference, readonly, validate_intervals


NDArrayBool = NDArray[np.bool_]


class EncoderErrorConfig(BaseModel):
    """Simulation-only encoder errors; not attached to observed recordings.

    Encoder angles are continuous/unwrapped. Clock jitter changes tags only;
    physical encoder sampling jitter would change the supplied query times.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[1] = 1
    angle_offset_rad: Finite = 0.0
    scale_error_ppm: Annotated[float, Field(gt=-1e6, allow_inf_nan=False)] = 0.0
    cyclic_error_rad: Finite = 0.0
    cyclic_order: Annotated[int, Field(ge=1, le=32, strict=True)] = 1
    cyclic_phase_rad: Finite = 0.0
    angle_noise_std_rad: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] = 0.0
    quantization_rad: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] = 0.0
    clock_offset_s: Finite = 0.0
    clock_frequency_error_ppm: Annotated[float, Field(gt=-1e6, allow_inf_nan=False)] = 0.0
    timestamp_jitter_bound_s: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] = 0.0
####


class EncoderCorrection(BaseModel):
    """Externally supplied metrology estimates, not inferred from simulator config.

    Zero defaults are an explicit assumption, not a claim of a calibrated encoder.
    Clock mapping uses the lab zero epoch. Cyclic errors remain model mismatch.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    angle_offset_rad: Finite = 0.0
    scale_error_ppm: Annotated[float, Field(gt=-1e6, allow_inf_nan=False)] = 0.0
    clock_offset_s: Finite = 0.0
    clock_frequency_error_ppm: Annotated[float, Field(gt=-1e6, allow_inf_nan=False)] = 0.0
####


@dataclass(frozen=True, slots=True)
class EncoderObservations:
    """The encoder-visible record only. No command, truth, or error realization."""

    reported_times_s: FloatArray
    angles_rad: FloatArray
    source_id: str

    def __post_init__(self) -> None:
        t = np.asarray(self.reported_times_s, dtype=float)
        if t.ndim != 1 or len(t) < 5 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError("encoder needs at least five finite strictly increasing timestamps")
        ####
        if not self.source_id:
            raise ValueError("encoder source_id is required")
        ####
        object.__setattr__(self, "reported_times_s", readonly(t, t.shape, "encoder times"))
        object.__setattr__(self, "angles_rad", readonly(self.angles_rad, t.shape, "encoder angles"))
    ####
####


def simulate_encoder(law: AngleLaw, physical_times_s: FloatArray, error: EncoderErrorConfig, *, rng: np.random.Generator, source_id: str = "synthetic_encoder") -> EncoderObservations:
    t = np.asarray(physical_times_s, dtype=float)
    if t.ndim != 1 or len(t) < 5 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("encoder physical sampling times must be finite and strictly increasing")
    ####
    q = law.evaluate(t)[0]
    angles = (1 + error.scale_error_ppm * 1e-6) * q + error.angle_offset_rad
    angles += error.cyclic_error_rad * np.sin(error.cyclic_order * q + error.cyclic_phase_rad)
    angles += rng.normal(0, error.angle_noise_std_rad, len(t))
    if error.quantization_rad > 0:
        angles = np.rint(angles / error.quantization_rad) * error.quantization_rad
    ####
    times = t * (1 + error.clock_frequency_error_ppm * 1e-6) + error.clock_offset_s
    times += rng.uniform(-error.timestamp_jitter_bound_s, error.timestamp_jitter_bound_s, len(t))
    # Out-of-order timestamps are rejected, never sorted to conceal acquisition errors.
    return EncoderObservations(times, angles, source_id)
####


@dataclass(frozen=True, slots=True)
class EncoderFit:
    law: AngleLaw
    residual_rms_rad: float
    scaled_condition_number: float
    corrected_start_s: float
    corrected_end_s: float
    source_id: str
    law_coefficient_covariance: FloatArray
    covariance_assumption: str = "ols_iid_encoder_angle_rows"

    def __post_init__(self) -> None:
        covariance = readonly(self.law_coefficient_covariance, (4, 4), "encoder law covariance")
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12):
            raise ValueError("encoder law covariance must be symmetric")
        ####
        if np.linalg.eigvalsh(covariance).min() < -1e-12:
            raise ValueError("encoder law covariance must be positive semidefinite")
        ####
        if not self.source_id or not self.covariance_assumption:
            raise ValueError("encoder fit provenance and covariance assumption are required")
        ####
        object.__setattr__(self, "law_coefficient_covariance", covariance)
    ####
####


def fit_encoder_law(observations: EncoderObservations, *, frequency_hz: float, correction: EncoderCorrection = EncoderCorrection(), max_residual_rad: float | None = None) -> EncoderFit:
    """Fit declared harmonic basis from observations only, not commanded rates.

    The frequency is part of the experiment protocol. Insufficient excitation
    and excessive residuals are rejected; high-frequency encoder noise is not
    numerically differentiated. This is a model-based smoother, not a generic
    arbitrary-motion reconstructor and not a clock-calibration estimator.
    """
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("frequency_hz must be finite and positive")
    ####
    if max_residual_rad is not None and (not np.isfinite(max_residual_rad) or max_residual_rad < 0):
        raise ValueError("max_residual_rad must be finite and nonnegative")
    ####
    t = (observations.reported_times_s - correction.clock_offset_s) / (1 + correction.clock_frequency_error_ppm * 1e-6)
    y = (observations.angles_rad - correction.angle_offset_rad) / (1 + correction.scale_error_ppm * 1e-6)
    origin = float(np.mean(t))
    nu = 2 * np.pi * frequency_hz
    design = np.column_stack((np.ones(len(t)), t - origin, np.sin(nu * t), np.cos(nu * t)))
    scales = np.linalg.norm(design, axis=0)
    if np.any(scales == 0):
        raise ValueError("encoder design is rank deficient")
    ####
    scaled = design / scales
    beta_scaled, _, rank, singular = np.linalg.lstsq(scaled, y, rcond=None)
    condition = float(singular[0] / singular[-1]) if singular[-1] else float("inf")
    if rank != 4 or condition > 1e8:
        raise ValueError("encoder design is rank deficient or poorly conditioned")
    ####
    beta = beta_scaled / scales
    residual = y - design @ beta
    rms = float(np.sqrt(np.mean(residual ** 2)))
    if max_residual_rad is not None and rms > max_residual_rad:
        raise ValueError("encoder residual exceeds the declared model-mismatch limit")
    ####
    dof = len(y) - 4
    if dof <= 0:
        raise ValueError("encoder fit needs residual degrees of freedom")
    ####
    influence = np.linalg.pinv(design)
    beta_covariance = float(residual @ residual / dof) * influence @ influence.T
    transform = np.array(((1.0, -origin, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)))
    law_covariance = transform @ beta_covariance @ transform.T
    law = AngleLaw(offset_rad=float(beta[0] - origin * beta[1]), rate_radps=float(beta[1]), sine_rad=float(beta[2]), cosine_rad=float(beta[3]), frequency_hz=frequency_hz)
    return EncoderFit(law, rms, condition, float(t[0]), float(t[-1]), observations.source_id, law_covariance)
####


@dataclass(frozen=True, slots=True)
class MeasuredImuRecord:
    """Unmodified sensor-visible rates and validity with reported clock intervals."""

    starts_s: FloatArray
    ends_s: FloatArray
    rates_si: FloatArray
    valid_channels: NDArrayBool
    temperatures_celsius: FloatArray
    source_id: str

    def __post_init__(self) -> None:
        starts, ends = validate_intervals(self.starts_s, self.ends_s)
        n = len(starts)
        for name, value, shape in (("starts_s", starts, (n,)), ("ends_s", ends, (n,)), ("rates_si", self.rates_si, (n, 6)), ("temperatures_celsius", self.temperatures_celsius, (n,))):
            object.__setattr__(self, name, readonly(value, shape, name))
        ####
        validity = np.asarray(self.valid_channels)
        if validity.shape != (n, 6) or validity.dtype != bool:
            raise ValueError("valid_channels must be a boolean (n, 6) array")
        ####
        validity = validity.copy()
        validity.setflags(write=False)
        object.__setattr__(self, "valid_channels", validity)
        if not self.source_id:
            raise ValueError("IMU source_id is required")
        ####
    ####
####


def simulate_table_record(model: AcquiredImuModelProtocol, provider: RateTableReference, starts_s: FloatArray, ends_s: FloatArray, *, ambient_temperature_celsius: float = 25.0, quadrature_order: int = 8, reset: bool = True, source_id: str = "synthetic_imu") -> tuple[MeasuredImuRecord, ApertureReference]:
    """Return separate observed record and evaluator-only truth.

    With reset=False, callers preserve one device/run realization across
    contiguous setup segments. Re-indexing motion itself is not synthesized.
    The core models errors on aperture means; this is not analog integration of
    nonlinear intrinsics before the aperture or general coning/sculling.
    """
    starts, ends = validate_intervals(starts_s, ends_s)
    if not np.allclose(starts[1:], ends[:-1], rtol=0.0, atol=1e-12):
        raise ValueError("one sensor run must have contiguous physical intervals")
    ####
    if not np.isfinite(ambient_temperature_celsius):
        raise ValueError("ambient temperature must be finite")
    ####
    truth = aperture_reference(provider, starts, ends, order=quadrature_order)
    if reset:
        model.reset()
    ####
    outputs = [model.measure_ideal_rates(start_time=float(start), end_time=float(end), specific_force_body=f, angular_rate_body=w, ambient_temperature_celsius=ambient_temperature_celsius) for start, end, f, w in zip(truth.starts_s, truth.ends_s, truth.force, truth.rate, strict=True)]
    record = MeasuredImuRecord(np.array([s.start_time_s for s in outputs]), np.array([s.end_time_s for s in outputs]), np.array([s.rates_si for s in outputs]), np.array([s.valid_channels for s in outputs]), np.array([s.temperature_celsius for s in outputs]), source_id)
    return record, truth
####


@dataclass(frozen=True, slots=True)
class ObservedCalibrationData:
    """Fitting input with reconstructed references, never generator truth."""

    dataset: CalibrationDataset
    reference_source_ids: tuple[str, ...]
    reference_basis: Literal["observed_encoder_and_external_survey"] = "observed_encoder_and_external_survey"

    def __post_init__(self) -> None:
        if not self.reference_source_ids or any(not s for s in self.reference_source_ids):
            raise ValueError("observed reference provenance is required")
        ####
        if self.reference_basis != "observed_encoder_and_external_survey":
            raise ValueError("observed reference basis is required")
        ####
    ####
####


def reconstruct_reference(record: MeasuredImuRecord, encoder_fit: EncoderFit, *, surveyed_geometry: TableGeometry, earth: LocalEarthReference, survey_source_id: str, include_earth_rotation: bool = True, quadrature_order: int = 8) -> ObservedCalibrationData:
    """Use encoder observations and survey estimates on the IMU-reported clock.

    Caller must independently align both clocks. No hidden physical time or
    commanded angle can enter this API. Filtered data need a matched prediction
    operator or prior correction; removing invalid startup rows is not deblurring.
    """
    if not survey_source_id:
        raise ValueError("survey_source_id is required")
    ####
    valid = np.all(record.valid_channels, axis=1)
    if not np.any(valid):
        raise ValueError("no completely valid IMU samples")
    ####
    starts, ends = record.starts_s[valid], record.ends_s[valid]
    if starts[0] < encoder_fit.corrected_start_s - 1e-12 or ends[-1] > encoder_fit.corrected_end_s + 1e-12:
        raise ValueError("encoder observations do not cover the IMU intervals; extrapolation refused")
    ####
    provider = RateTableReference(earth, encoder_fit.law, surveyed_geometry, include_earth_rotation)
    ref = aperture_reference(provider, starts, ends, order=quadrature_order)
    data = CalibrationDataset(labels=(record.source_id,) * len(starts), start_times=starts, end_times=ends, temperatures_celsius=record.temperatures_celsius[valid], specific_force_body=ref.force, angular_rate_body=ref.rate, measured_acceleration=record.rates_si[valid, :3], measured_angular_rate=record.rates_si[valid, 3:])
    return ObservedCalibrationData(data, (encoder_fit.source_id, survey_source_id))
####


def combine_observed_records(records: tuple[ObservedCalibrationData, ...]) -> ObservedCalibrationData:
    if not records or any(not isinstance(r, ObservedCalibrationData) for r in records):
        raise TypeError("only nonempty observed-reference records can be combined")
    ####
    labels = tuple(label for r in records for label in r.dataset.labels)
    fields = ("start_times", "end_times", "temperatures_celsius", "specific_force_body", "angular_rate_body", "measured_acceleration", "measured_angular_rate")
    arrays = {field: np.concatenate([getattr(r.dataset, field) for r in records]) for field in fields}
    # Independent sessions can each start at zero; labels retain their boundaries.
    return ObservedCalibrationData(CalibrationDataset(labels=labels, **arrays), tuple(s for r in records for s in r.reference_source_ids))
####


def fit_observed_calibration(data: ObservedCalibrationData, *, maximum_condition: float = 1e8) -> CalibrationEstimate:
    if not isinstance(data, ObservedCalibrationData):
        raise TypeError("fit requires ObservedCalibrationData, not evaluator truth")
    ####
    if not np.isfinite(maximum_condition) or maximum_condition < 1:
        raise ValueError("maximum_condition must be finite and at least one")
    ####
    dataset = data.dataset
    n = len(dataset.labels)
    for design in (
        np.column_stack((dataset.specific_force_body, np.ones(n))),
        np.column_stack((dataset.angular_rate_body, dataset.specific_force_body, np.ones(n))),
    ):
        if n <= design.shape[1]:
            raise ValueError("observed fit needs more rows than parameters for uncertainty")
        ####
        scales = np.linalg.norm(design, axis=0)
        scaled = design / np.where(scales > 0, scales, 1.0)
        singular = np.linalg.svd(scaled, compute_uv=False)
        if np.linalg.matrix_rank(scaled) < design.shape[1] or singular[-1] <= 0 or singular[0] / singular[-1] > maximum_condition:
            raise ValueError("observed reference is rank deficient or poorly conditioned")
        ####
    ####
    estimate = fit_known_reference_calibration(dataset)
    return replace(estimate, covariance_assumption="conditional_on_observed_reference_iid_sensor_rows; reference_error_not_included")
####
