"""Forward-matched calibration for filtered/delayed acquisition records.

The central rule is deliberately asymmetric: reconstructed ideal references are
passed FORWARD through an independently identified readout operator. Sensor data
are never deconvolved by an arbitrary inverse FIR. This preserves phase/support,
keeps startup validity explicit, and lets intrinsic calibration operate on the
same effective measurement support as the observed IMU channels.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from .earth_reference import LocalEarthReference
from .known_reference import (
    CalibrationEstimate,
    calibration_coefficient_vectors,
)
from .readout_identification import ReadoutKernelEstimate
from .reference_equipment import (
    AngleLaw,
    ApertureReference,
    RateTableReference,
    TableGeometry,
    aperture_reference,
    readonly,
    validate_intervals,
)
from .reference_identification import checked_design
from .reference_observations import EncoderFit, MeasuredImuRecord


FloatArray = NDArray[np.float64]
Finite = Annotated[float, Field(allow_inf_nan=False)]


class RelativeClockCorrection(BaseModel):
    """Independent mapping from IMU-reported tags to the reference time base.

    If the IMU reports

        tag = epoch + offset + (t_ref - epoch) * (1 + ppm * 1e-6),

    this object applies the inverse mapping. It is a supplied/fitted metrology
    result, not inferred from hidden acquisition configuration.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    epoch_s: Finite = 0.0
    offset_s: Finite = 0.0
    frequency_error_ppm: Annotated[float, Field(gt=-1e6, allow_inf_nan=False)] = 0.0

    def to_reference_time(self, reported_time_s: FloatArray) -> FloatArray:
        tags = np.asarray(reported_time_s, dtype=float)
        if not np.isfinite(tags).all():
            raise ValueError("reported times must be finite")
        ####
        scale = 1.0 + self.frequency_error_ppm * 1e-6
        result = self.epoch_s + (tags - self.epoch_s - self.offset_s) / scale
        if not np.isfinite(result).all():
            raise ValueError("clock correction produced non-finite times")
        ####
        return result
    ####
####


@dataclass(frozen=True, slots=True)
class MatchedReferenceData:
    """Observed rows plus output-channel-specific matched reference regressors.

    ``accelerometer_force_by_output[:, j, :]`` is the reconstructed specific
    force after the readout kernel for accelerometer output channel ``j``.
    Gyroscope rate and force regressors are matched independently through each
    gyro output channel's effective kernel. Intercepts from the readout-ID fit
    are NOT applied: intrinsic sensor bias remains a calibration parameter.
    """

    labels: tuple[str, ...]
    starts_s: FloatArray
    ends_s: FloatArray
    temperatures_celsius: FloatArray
    accelerometer_force_by_output: FloatArray
    gyroscope_rate_by_output: FloatArray
    gyroscope_force_by_output: FloatArray
    measured_acceleration: FloatArray
    measured_angular_rate: FloatArray
    reference_source_ids: tuple[str, ...]
    readout_source_id: str
    clock_basis: str = "reported_imu_tags_corrected_to_reference_time_then_forward_readout_matched"

    def __post_init__(self) -> None:
        n = len(self.labels)
        if n == 0:
            raise ValueError("matched calibration data must not be empty")
        ####
        starts, ends = validate_intervals(self.starts_s, self.ends_s)
        if len(starts) != n:
            raise ValueError("labels and interval count must match")
        ####
        object.__setattr__(self, "starts_s", readonly(starts, (n,), "matched starts"))
        object.__setattr__(self, "ends_s", readonly(ends, (n,), "matched ends"))
        for name, shape in (
            ("temperatures_celsius", (n,)),
            ("accelerometer_force_by_output", (n, 3, 3)),
            ("gyroscope_rate_by_output", (n, 3, 3)),
            ("gyroscope_force_by_output", (n, 3, 3)),
            ("measured_acceleration", (n, 3)),
            ("measured_angular_rate", (n, 3)),
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
            ####
            copy = value.copy()
            copy.setflags(write=False)
            object.__setattr__(self, name, copy)
        ####
        if not self.reference_source_ids or any(not item for item in self.reference_source_ids):
            raise ValueError("matched reference provenance is required")
        ####
        if not self.readout_source_id or not self.clock_basis:
            raise ValueError("readout provenance and clock basis are required")
        ####
    ####
####


def _filter_columns(values: FloatArray, kernel: FloatArray) -> FloatArray:
    x = np.asarray(values, dtype=float)
    h = np.asarray(kernel, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError("reference channels must have finite shape (n, 3)")
    ####
    if h.ndim != 1 or len(h) < 1 or not np.isfinite(h).all():
        raise ValueError("readout kernel must be a finite nonempty vector")
    ####
    return np.column_stack([np.convolve(x[:, axis], h, mode="full")[:len(x)] for axis in range(3)])
####


def match_aperture_reference(
        record: MeasuredImuRecord,
        reference: ApertureReference,
        readout: ReadoutKernelEstimate,
        *, reference_source_ids: tuple[str, ...], readout_source_id: str,
) -> MatchedReferenceData:
    """Forward-match a reconstructed aperture reference to one observed record.

    Complete prehistory must be present in ``reference``. Startup rows are kept
    while filtering and discarded only AFTER the reference operator has run.
    This prevents the first retained row from being modeled as if prior samples
    were zero. A common all-six-channel validity mask is used so joint residual
    covariance retains synchronized rows.
    """
    n = len(record.starts_s)
    if len(reference.starts_s) != n:
        raise ValueError("reference and observed record lengths must match before support matching")
    ####
    if not np.allclose(reference.starts_s, record.starts_s, rtol=0.0, atol=1e-10) or not np.allclose(
            reference.ends_s, record.ends_s, rtol=0.0, atol=1e-10,
    ):
        raise ValueError("reference intervals must already be expressed on the corrected IMU support")
    ####
    length = readout.kernels.shape[1]
    if n < length:
        raise ValueError("record is shorter than the identified readout support")
    ####
    accel_force = np.zeros((n, 3, 3))
    gyro_rate = np.zeros((n, 3, 3))
    gyro_force = np.zeros((n, 3, 3))
    for output_axis in range(3):
        accel_force[:, output_axis, :] = _filter_columns(reference.force, readout.kernels[output_axis])
        gyro_rate[:, output_axis, :] = _filter_columns(reference.rate, readout.kernels[3 + output_axis])
        gyro_force[:, output_axis, :] = _filter_columns(reference.force, readout.kernels[3 + output_axis])
    ####
    complete_history = np.arange(n) >= length - 1
    valid = np.all(record.valid_channels, axis=1) & complete_history
    if not np.any(valid):
        raise ValueError("no rows have both observed validity and complete identified readout history")
    ####
    return MatchedReferenceData(
        labels=(record.source_id,) * int(np.count_nonzero(valid)),
        starts_s=reference.starts_s[valid],
        ends_s=reference.ends_s[valid],
        temperatures_celsius=record.temperatures_celsius[valid],
        accelerometer_force_by_output=accel_force[valid],
        gyroscope_rate_by_output=gyro_rate[valid],
        gyroscope_force_by_output=gyro_force[valid],
        measured_acceleration=record.rates_si[valid, :3],
        measured_angular_rate=record.rates_si[valid, 3:],
        reference_source_ids=reference_source_ids,
        readout_source_id=readout_source_id,
    )
####


def reconstruct_matched_reference(
        record: MeasuredImuRecord,
        encoder_fit: EncoderFit,
        *, surveyed_geometry: TableGeometry, earth: LocalEarthReference,
        readout: ReadoutKernelEstimate, clock_correction: RelativeClockCorrection,
        survey_source_id: str, readout_source_id: str,
        include_earth_rotation: bool = True, quadrature_order: int = 8,
) -> MatchedReferenceData:
    """Reconstruct ALL intervals on corrected time, then forward-match readout.

    No hidden physical timestamps, commanded angle, or injected readout config are
    available to this function. Encoder and clock estimates are external inputs.
    Reference extrapolation is refused, including the startup history needed by
    the FIR operator.
    """
    if not survey_source_id or not readout_source_id:
        raise ValueError("survey and readout source identifiers are required")
    ####
    starts = clock_correction.to_reference_time(record.starts_s)
    ends = clock_correction.to_reference_time(record.ends_s)
    starts, ends = validate_intervals(starts, ends)
    if starts[0] < encoder_fit.corrected_start_s - 1e-12 or ends[-1] > encoder_fit.corrected_end_s + 1e-12:
        raise ValueError("encoder observations do not cover matched IMU support; extrapolation refused")
    ####
    provider = RateTableReference(earth, encoder_fit.law, surveyed_geometry, include_earth_rotation)
    reference = aperture_reference(provider, starts, ends, order=quadrature_order)
    # Measured tags and corrected reference intervals intentionally differ. Build
    # an observation-view record on corrected support without changing measured values.
    corrected_record = MeasuredImuRecord(
        starts,
        ends,
        record.rates_si,
        record.valid_channels,
        record.temperatures_celsius,
        record.source_id,
    )
    return match_aperture_reference(
        corrected_record,
        reference,
        readout,
        reference_source_ids=(encoder_fit.source_id, survey_source_id),
        readout_source_id=readout_source_id,
    )
####


def combine_matched_records(records: tuple[MatchedReferenceData, ...]) -> MatchedReferenceData:
    """Combine independently reset/reindexed sessions after each was matched locally."""
    if not records or any(not isinstance(record, MatchedReferenceData) for record in records):
        raise TypeError("only a nonempty tuple of matched records can be combined")
    ####
    fields = (
        "starts_s",
        "ends_s",
        "temperatures_celsius",
        "accelerometer_force_by_output",
        "gyroscope_rate_by_output",
        "gyroscope_force_by_output",
        "measured_acceleration",
        "measured_angular_rate",
    )
    arrays = {name: np.concatenate([getattr(record, name) for record in records]) for name in fields}
    labels = tuple(label for record in records for label in record.labels)
    # Independent sessions may reuse local time epochs. CalibrationDataset-style
    # interval validation requires ordering, so shift only the bookkeeping times;
    # matched regressors are already computed and are not re-evaluated here.
    starts_parts: list[FloatArray] = []
    ends_parts: list[FloatArray] = []
    cursor = 0.0
    for record in records:
        local_start = record.starts_s[0]
        starts_parts.append(record.starts_s - local_start + cursor)
        ends_parts.append(record.ends_s - local_start + cursor)
        cursor = float(ends_parts[-1][-1] + 1e-9)
    ####
    arrays["starts_s"] = np.concatenate(starts_parts)
    arrays["ends_s"] = np.concatenate(ends_parts)
    sources = tuple(source for record in records for source in record.reference_source_ids)
    readout_sources = {record.readout_source_id for record in records}
    if len(readout_sources) != 1:
        raise ValueError("combined records must use one identified readout operator")
    ####
    return MatchedReferenceData(
        labels=labels,
        starts_s=arrays["starts_s"],
        ends_s=arrays["ends_s"],
        temperatures_celsius=arrays["temperatures_celsius"],
        accelerometer_force_by_output=arrays["accelerometer_force_by_output"],
        gyroscope_rate_by_output=arrays["gyroscope_rate_by_output"],
        gyroscope_force_by_output=arrays["gyroscope_force_by_output"],
        measured_acceleration=arrays["measured_acceleration"],
        measured_angular_rate=arrays["measured_angular_rate"],
        reference_source_ids=sources,
        readout_source_id=next(iter(readout_sources)),
    )
####


def _axis_fit(design: FloatArray, target: FloatArray, maximum_condition: float) -> tuple[FloatArray, FloatArray, int, float, FloatArray]:
    scaled, scales, rank, condition = checked_design(design, maximum_condition=maximum_condition)
    coefficients_scaled = np.linalg.lstsq(scaled, target, rcond=None)[0]
    coefficients = coefficients_scaled / scales
    residual = target - design @ coefficients
    influence = np.linalg.pinv(scaled) / scales[:, None]
    return coefficients, residual, rank, condition, influence
####


def fit_matched_calibration(data: MatchedReferenceData, *, maximum_condition: float = 1e8) -> CalibrationEstimate:
    """Fit affine intrinsics on output-channel-specific matched reference support."""
    if not isinstance(data, MatchedReferenceData):
        raise TypeError("matched calibration requires MatchedReferenceData")
    ####
    if not np.isfinite(maximum_condition) or maximum_condition < 1.0:
        raise ValueError("maximum_condition must be finite and at least one")
    ####
    n = len(data.labels)
    if n <= 7:
        raise ValueError("matched calibration needs residual degrees of freedom")
    ####
    accel_coefficients = np.zeros((4, 3))
    gyro_coefficients = np.zeros((7, 3))
    accel_residual = np.zeros((n, 3))
    gyro_residual = np.zeros((n, 3))
    accel_influence: list[FloatArray] = []
    gyro_influence: list[FloatArray] = []
    accel_ranks: list[int] = []
    gyro_ranks: list[int] = []
    accel_conditions: list[float] = []
    gyro_conditions: list[float] = []
    for axis in range(3):
        accel_design = np.column_stack((data.accelerometer_force_by_output[:, axis, :], np.ones(n)))
        a_beta, a_residual, a_rank, a_condition, a_influence = _axis_fit(
            accel_design, data.measured_acceleration[:, axis], maximum_condition,
        )
        gyro_design = np.column_stack((
            data.gyroscope_rate_by_output[:, axis, :],
            data.gyroscope_force_by_output[:, axis, :],
            np.ones(n),
        ))
        g_beta, g_residual, g_rank, g_condition, g_influence = _axis_fit(
            gyro_design, data.measured_angular_rate[:, axis], maximum_condition,
        )
        accel_coefficients[:, axis] = a_beta
        gyro_coefficients[:, axis] = g_beta
        accel_residual[:, axis] = a_residual
        gyro_residual[:, axis] = g_residual
        accel_influence.append(a_influence)
        gyro_influence.append(g_influence)
        accel_ranks.append(a_rank)
        gyro_ranks.append(g_rank)
        accel_conditions.append(a_condition)
        gyro_conditions.append(g_condition)
    ####
    residual = np.column_stack((accel_residual, gyro_residual))
    dof = n - 7
    residual_covariance = residual.T @ residual / dof
    accel_covariance = np.zeros((12, 12))
    gyro_covariance = np.zeros((21, 21))
    cross_covariance = np.zeros((12, 21))
    for left in range(3):
        a_left = slice(4 * left, 4 * (left + 1))
        g_left = slice(7 * left, 7 * (left + 1))
        for right in range(3):
            a_right = slice(4 * right, 4 * (right + 1))
            g_right = slice(7 * right, 7 * (right + 1))
            accel_covariance[a_left, a_right] = (
                residual_covariance[left, right] * accel_influence[left] @ accel_influence[right].T
            )
            gyro_covariance[g_left, g_right] = (
                residual_covariance[3 + left, 3 + right] * gyro_influence[left] @ gyro_influence[right].T
            )
            cross_covariance[a_left, g_right] = (
                residual_covariance[left, 3 + right] * accel_influence[left] @ gyro_influence[right].T
            )
        ####
    ####
    return CalibrationEstimate(
        accelerometer_sensitivity_matrix=accel_coefficients[:3, :].T,
        accelerometer_bias=accel_coefficients[3, :],
        gyroscope_sensitivity_matrix=gyro_coefficients[:3, :].T,
        gyroscope_specific_force_sensitivity=gyro_coefficients[3:6, :].T,
        gyroscope_bias=gyro_coefficients[6, :],
        accelerometer_design_rank=min(accel_ranks),
        gyroscope_design_rank=min(gyro_ranks),
        accelerometer_condition_number=max(accel_conditions),
        gyroscope_condition_number=max(gyro_conditions),
        accelerometer_residual_rms=float(np.sqrt(np.mean(accel_residual ** 2))),
        gyroscope_residual_rms=float(np.sqrt(np.mean(gyro_residual ** 2))),
        accelerometer_residual_covariance=residual_covariance[:3, :3],
        gyroscope_residual_covariance=residual_covariance[3:, 3:],
        accelerometer_coefficient_covariance=0.5 * (accel_covariance + accel_covariance.T),
        gyroscope_coefficient_covariance=0.5 * (gyro_covariance + gyro_covariance.T),
        accelerometer_gyroscope_coefficient_cross_covariance=cross_covariance,
        covariance_assumption="matched_axiswise_ols_iid_synchronized_residual_rows; reconstructed_reference_readout_clock_fixed",
    )
####


def predict_matched_measurement(data: MatchedReferenceData, calibration: CalibrationEstimate) -> FloatArray:
    """Predict sensor-visible matched rates on held-out reference support."""
    if not isinstance(data, MatchedReferenceData) or not isinstance(calibration, CalibrationEstimate):
        raise TypeError("matched prediction requires matched data and a calibration estimate")
    ####
    n = len(data.labels)
    prediction = np.zeros((n, 6))
    for axis in range(3):
        prediction[:, axis] = (
            data.accelerometer_force_by_output[:, axis, :] @ calibration.accelerometer_sensitivity_matrix[axis]
            + calibration.accelerometer_bias[axis]
        )
        prediction[:, 3 + axis] = (
            data.gyroscope_rate_by_output[:, axis, :] @ calibration.gyroscope_sensitivity_matrix[axis]
            + data.gyroscope_force_by_output[:, axis, :] @ calibration.gyroscope_specific_force_sensitivity[axis]
            + calibration.gyroscope_bias[axis]
        )
    ####
    return prediction
####


@dataclass(frozen=True, slots=True)
class MatchedSupportParameterization:
    """Joint uncertainty model for identified readout, encoder laws, and clock."""

    readout: ReadoutKernelEstimate
    encoder_fits: tuple[EncoderFit, ...]
    clock_correction: RelativeClockCorrection
    covariance: FloatArray
    parameter_names: tuple[str, ...]
    finite_difference_steps: FloatArray

    def __post_init__(self) -> None:
        mean = self.mean
        covariance = readonly(self.covariance, (len(mean), len(mean)), "matched support covariance")
        steps = readonly(self.finite_difference_steps, (len(mean),), "matched support finite-difference steps")
        if len(self.parameter_names) != len(mean) or len(set(self.parameter_names)) != len(mean) or any(not name for name in self.parameter_names):
            raise ValueError("matched support parameter names must be unique and complete")
        ####
        if np.any(steps <= 0.0) or not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12):
            raise ValueError("finite-difference steps must be positive and support covariance symmetric")
        ####
        if np.linalg.eigvalsh(covariance).min() < -1e-12:
            raise ValueError("matched support covariance must be positive semidefinite")
        ####
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "finite_difference_steps", steps)
    ####

    @property
    def mean(self) -> FloatArray:
        values: list[float] = list(self.readout.kernels.ravel())
        for fit in self.encoder_fits:
            values.extend((fit.law.offset_rad, fit.law.rate_radps, fit.law.sine_rad, fit.law.cosine_rad))
        ####
        values.extend((self.clock_correction.offset_s, self.clock_correction.frequency_error_ppm))
        result = np.asarray(values, dtype=float)
        result.setflags(write=False)
        return result
    ####

    def unpack(self, parameters: FloatArray) -> tuple[ReadoutKernelEstimate, tuple[EncoderFit, ...], RelativeClockCorrection]:
        values = np.asarray(parameters, dtype=float)
        if values.shape != self.mean.shape or not np.isfinite(values).all():
            raise ValueError("support parameter vector has wrong shape or non-finite values")
        ####
        length = self.readout.kernels.shape[1]
        cursor = 6 * length
        kernels = values[:cursor].reshape((6, length))
        readout = replace(self.readout, kernels=kernels)
        fits: list[EncoderFit] = []
        for fit in self.encoder_fits:
            offset, rate, sine, cosine = values[cursor:cursor + 4]
            cursor += 4
            law = AngleLaw(
                offset_rad=float(offset), rate_radps=float(rate), sine_rad=float(sine), cosine_rad=float(cosine),
                frequency_hz=fit.law.frequency_hz,
            )
            fits.append(replace(fit, law=law))
        ####
        clock = self.clock_correction.model_copy(update={
            "offset_s": float(values[cursor]),
            "frequency_error_ppm": float(values[cursor + 1]),
        })
        return readout, tuple(fits), clock
    ####
####


def build_matched_support_parameterization(
        readout: ReadoutKernelEstimate,
        encoder_fits: tuple[EncoderFit, ...],
        clock_correction: RelativeClockCorrection,
        *, clock_covariance_offset_ppm: FloatArray | None = None,
) -> MatchedSupportParameterization:
    """Assemble independent fitted-support covariance blocks without sample replication."""
    if not encoder_fits:
        raise ValueError("at least one encoder fit is required")
    ####
    clock_covariance = np.zeros((2, 2)) if clock_covariance_offset_ppm is None else np.asarray(
        clock_covariance_offset_ppm, dtype=float,
    )
    if clock_covariance.shape != (2, 2) or not np.isfinite(clock_covariance).all():
        raise ValueError("clock covariance must have finite shape (2, 2)")
    ####
    length = readout.kernels.shape[1]
    size = 6 * length + 4 * len(encoder_fits) + 2
    covariance = np.zeros((size, size))
    cursor = 0
    covariance[:6 * length, :6 * length] = readout.tap_covariance
    cursor += 6 * length
    names = [f"readout.channel_{axis}.tap_{tap}" for axis in range(6) for tap in range(length)]
    steps = [1e-6] * (6 * length)
    for index, fit in enumerate(encoder_fits):
        covariance[cursor:cursor + 4, cursor:cursor + 4] = fit.law_coefficient_covariance
        names.extend((
            f"encoder_{index}.offset_rad",
            f"encoder_{index}.rate_radps",
            f"encoder_{index}.sine_rad",
            f"encoder_{index}.cosine_rad",
        ))
        steps.extend((1e-7, 1e-7, 1e-7, 1e-7))
        cursor += 4
    ####
    covariance[cursor:cursor + 2, cursor:cursor + 2] = clock_covariance
    names.extend(("relative_clock.offset_s", "relative_clock.frequency_error_ppm"))
    steps.extend((1e-7, 0.1))
    return MatchedSupportParameterization(
        readout,
        encoder_fits,
        clock_correction,
        covariance,
        tuple(names),
        np.asarray(steps),
    )
####


@dataclass(frozen=True, slots=True)
class MatchedSupportUncertaintyResult:
    calibration: CalibrationEstimate
    conditional_calibration_covariance: FloatArray
    support_covariance: FloatArray
    coefficient_support_jacobian: FloatArray
    support_parameter_names: tuple[str, ...]
####


def _same_observations(left: MatchedReferenceData, right: MatchedReferenceData) -> bool:
    return (
        left.labels == right.labels
        and np.array_equal(left.measured_acceleration, right.measured_acceleration)
        and np.array_equal(left.measured_angular_rate, right.measured_angular_rate)
    )
####


def propagate_matched_support_uncertainty(
        builder: Callable[[ReadoutKernelEstimate, tuple[EncoderFit, ...], RelativeClockCorrection], MatchedReferenceData],
        parameterization: MatchedSupportParameterization,
) -> MatchedSupportUncertaintyResult:
    """Propagate fitted support-model covariance into intrinsic calibration.

    The builder must reprocess the SAME observed IMU recordings. Central finite
    differences perturb only identified FIR taps, fitted encoder-law coefficients,
    and independently established relative-clock correction. This is a delta-
    method uncertainty propagation; it is not a joint errors-in-variables fit.
    """
    readout, encoder_fits, clock = parameterization.unpack(parameterization.mean)
    base_data = builder(readout, encoder_fits, clock)
    base_fit = fit_matched_calibration(base_data)
    mean = parameterization.mean
    jacobian = np.zeros((33, len(mean)))
    for index, step in enumerate(parameterization.finite_difference_steps):
        plus = mean.copy()
        minus = mean.copy()
        plus[index] += step
        minus[index] -= step
        plus_models = parameterization.unpack(plus)
        minus_models = parameterization.unpack(minus)
        plus_data = builder(*plus_models)
        minus_data = builder(*minus_models)
        if not _same_observations(base_data, plus_data) or not _same_observations(base_data, minus_data):
            raise ValueError("support uncertainty builder must preserve the same measured recording")
        ####
        plus_fit = fit_matched_calibration(plus_data)
        minus_fit = fit_matched_calibration(minus_data)
        plus_vector = np.concatenate(calibration_coefficient_vectors(plus_fit))
        minus_vector = np.concatenate(calibration_coefficient_vectors(minus_fit))
        jacobian[:, index] = (plus_vector - minus_vector) / (2.0 * step)
    ####
    conditional = base_fit.joint_coefficient_covariance
    support = jacobian @ parameterization.covariance @ jacobian.T
    total = 0.5 * (conditional + support + (conditional + support).T)
    updated = replace(
        base_fit,
        accelerometer_coefficient_covariance=total[:12, :12],
        gyroscope_coefficient_covariance=total[12:, 12:],
        accelerometer_gyroscope_coefficient_cross_covariance=total[:12, 12:],
        covariance_assumption=(
            "matched_axiswise_ols_iid_sensor_rows_plus_delta_method_identified_readout_encoder_clock; "
            "support_estimates_independent_of_calibration_record"
        ),
    )
    return MatchedSupportUncertaintyResult(
        updated,
        readonly(conditional, (33, 33), "conditional calibration covariance"),
        readonly(support, (33, 33), "matched support covariance"),
        readonly(jacobian, jacobian.shape, "support Jacobian"),
        parameterization.parameter_names,
    )
####


@dataclass(frozen=True, slots=True)
class MultiChannelClockEstimate:
    """Common relative clock map with channel-wise affine calibrated response.

    The six gains/biases are nuisance parameters that prevent phase alignment
    from assuming perfect prior scale/bias. The common offset and clock-frequency
    error are accepted only when the full Jacobian is observable. This still
    cannot attribute a recovered relative clock error uniquely to the IMU versus
    the external reference without an independent clock anchor.
    """

    gains: FloatArray
    biases: FloatArray
    relative_offset_s: float
    clock_frequency_error_ppm: float
    covariance: FloatArray
    scaled_condition_number: float
    residual_rms: float
    covariance_assumption: str = "profiled_channel_gain_bias_plus_common_clock; iid_isotropic_rows"

    def __post_init__(self) -> None:
        object.__setattr__(self, "gains", readonly(self.gains, (6,), "clock-fit gains"))
        object.__setattr__(self, "biases", readonly(self.biases, (6,), "clock-fit biases"))
        covariance = readonly(self.covariance, (14, 14), "clock-fit covariance")
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12):
            raise ValueError("clock-fit covariance must be symmetric")
        ####
        if np.linalg.eigvalsh(covariance).min() < -1e-12:
            raise ValueError("clock-fit covariance must be positive semidefinite")
        ####
        if not np.isfinite((self.relative_offset_s, self.clock_frequency_error_ppm, self.scaled_condition_number, self.residual_rms)).all():
            raise ValueError("clock-fit scalar outputs must be finite")
        ####
        if np.any(self.gains <= 0.0) or self.scaled_condition_number < 0.0 or self.residual_rms < 0.0:
            raise ValueError("clock-fit gains must be positive and diagnostics nonnegative")
        ####
        if not self.covariance_assumption:
            raise ValueError("clock-fit covariance assumption is required")
        ####
        object.__setattr__(self, "covariance", covariance)
    ####
####


def _predictor_rates(
        predictor: Callable[[FloatArray, FloatArray], FloatArray], starts: FloatArray, ends: FloatArray,
) -> FloatArray:
    result = np.asarray(predictor(starts, ends), dtype=float)
    if result.shape != (len(starts), 6) or not np.isfinite(result).all():
        raise ValueError("reference predictor must return finite shape (n, 6)")
    ####
    return result
####


def _correct_clock_arrays(
        starts: FloatArray, ends: FloatArray, *, epoch_s: float, offset_s: float, frequency_error_ppm: float,
) -> tuple[FloatArray, FloatArray]:
    correction = RelativeClockCorrection(epoch_s=epoch_s, offset_s=offset_s, frequency_error_ppm=frequency_error_ppm)
    return correction.to_reference_time(starts), correction.to_reference_time(ends)
####


def _profile_multichannel_clock(
        predictor: Callable[[FloatArray, FloatArray], FloatArray],
        starts: FloatArray,
        ends: FloatArray,
        measured: FloatArray,
        *, epoch_s: float, offset_s: float, frequency_error_ppm: float,
) -> tuple[float, FloatArray, FloatArray, FloatArray]:
    corrected_starts, corrected_ends = _correct_clock_arrays(
        starts,
        ends,
        epoch_s=epoch_s,
        offset_s=offset_s,
        frequency_error_ppm=frequency_error_ppm,
    )
    reference = _predictor_rates(predictor, corrected_starts, corrected_ends)
    gains = np.zeros(6)
    biases = np.zeros(6)
    residual = np.zeros_like(measured)
    for axis in range(6):
        design = np.column_stack((reference[:, axis], np.ones(len(reference))))
        if np.linalg.matrix_rank(design) < 2:
            return float("inf"), gains, biases, residual
        ####
        coefficients = np.linalg.lstsq(design, measured[:, axis], rcond=None)[0]
        if coefficients[0] <= 0.0:
            return float("inf"), gains, biases, residual
        ####
        gains[axis], biases[axis] = coefficients
        residual[:, axis] = measured[:, axis] - design @ coefficients
    ####
    return float(np.sum(residual ** 2)), gains, biases, residual
####


def fit_multichannel_clock_alignment(
        predictor: Callable[[FloatArray, FloatArray], FloatArray],
        reported_starts_s: FloatArray,
        reported_ends_s: FloatArray,
        measured_rates: FloatArray,
        *, epoch_s: float = 0.0,
        offset_bounds_s: tuple[float, float] = (-0.1, 0.1),
        frequency_error_bounds_ppm: tuple[float, float] = (-2000.0, 2000.0),
        maximum_condition: float = 1e10,
) -> MultiChannelClockEstimate:
    """Fit common relative offset/frequency error with six gain/bias nuisances.

    A coarse two-dimensional profile search supplies a basin, then a scaled
    Gauss--Newton fit solves channel gains/biases and the common clock map
    jointly. The final 14-parameter Jacobian must be full rank; constant-rate
    or otherwise gauge-deficient experiments are rejected rather than returning
    a precise but arbitrary clock estimate.
    """
    starts, ends = validate_intervals(reported_starts_s, reported_ends_s)
    measured = np.asarray(measured_rates, dtype=float)
    if measured.shape != (len(starts), 6) or not np.isfinite(measured).all():
        raise ValueError("measured_rates must have finite shape (n, 6)")
    ####
    if len(starts) <= 14:
        raise ValueError("multi-channel clock fit needs more rows than parameters")
    ####
    if not np.isfinite(epoch_s):
        raise ValueError("epoch_s must be finite")
    ####
    offset_lo, offset_hi = map(float, offset_bounds_s)
    ppm_lo, ppm_hi = map(float, frequency_error_bounds_ppm)
    if not np.isfinite((offset_lo, offset_hi, ppm_lo, ppm_hi)).all() or offset_hi <= offset_lo or ppm_hi <= ppm_lo:
        raise ValueError("clock search bounds must be finite and increasing")
    ####
    if ppm_lo <= -1e6:
        raise ValueError("clock frequency search must keep a positive time scale")
    ####
    if not np.isfinite(maximum_condition) or maximum_condition < 1.0:
        raise ValueError("maximum_condition must be finite and at least one")
    ####

    def cost(offset: float, ppm: float) -> float:
        try:
            return _profile_multichannel_clock(
                predictor,
                starts,
                ends,
                measured,
                epoch_s=epoch_s,
                offset_s=offset,
                frequency_error_ppm=ppm,
            )[0]
        except ValueError:
            return float("inf")
        ####
    ####

    offset_grid = np.linspace(offset_lo, offset_hi, 13)
    ppm_grid = np.linspace(ppm_lo, ppm_hi, 13)
    costs = np.array([[cost(float(candidate_offset), float(candidate_ppm)) for candidate_ppm in ppm_grid] for candidate_offset in offset_grid])
    best = np.unravel_index(int(np.argmin(costs)), costs.shape)
    if best[0] in (0, len(offset_grid) - 1) or best[1] in (0, len(ppm_grid) - 1) or not np.isfinite(costs[best]):
        raise ValueError("clock optimum is outside/intersects the search boundary or profile is invalid")
    ####
    offset = float(offset_grid[best[0]])
    ppm = float(ppm_grid[best[1]])
    _, gains, biases, _ = _profile_multichannel_clock(
        predictor,
        starts,
        ends,
        measured,
        epoch_s=epoch_s,
        offset_s=offset,
        frequency_error_ppm=ppm,
    )
    offset_step = max(1e-7, (offset_hi - offset_lo) * 1e-6)
    ppm_step = max(1e-3, (ppm_hi - ppm_lo) * 1e-6)

    def state(
            active_gains: FloatArray, active_biases: FloatArray, active_offset: float, active_ppm: float,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        corrected_starts, corrected_ends = _correct_clock_arrays(
            starts,
            ends,
            epoch_s=epoch_s,
            offset_s=active_offset,
            frequency_error_ppm=active_ppm,
        )
        reference = _predictor_rates(predictor, corrected_starts, corrected_ends)
        prediction = reference * active_gains + active_biases
        residual = measured - prediction
        plus_starts, plus_ends = _correct_clock_arrays(
            starts, ends, epoch_s=epoch_s, offset_s=active_offset + offset_step, frequency_error_ppm=active_ppm,
        )
        minus_starts, minus_ends = _correct_clock_arrays(
            starts, ends, epoch_s=epoch_s, offset_s=active_offset - offset_step, frequency_error_ppm=active_ppm,
        )
        d_reference_offset = (
            _predictor_rates(predictor, plus_starts, plus_ends)
            - _predictor_rates(predictor, minus_starts, minus_ends)
        ) / (2.0 * offset_step)
        plus_starts, plus_ends = _correct_clock_arrays(
            starts, ends, epoch_s=epoch_s, offset_s=active_offset, frequency_error_ppm=active_ppm + ppm_step,
        )
        minus_starts, minus_ends = _correct_clock_arrays(
            starts, ends, epoch_s=epoch_s, offset_s=active_offset, frequency_error_ppm=active_ppm - ppm_step,
        )
        d_reference_ppm = (
            _predictor_rates(predictor, plus_starts, plus_ends)
            - _predictor_rates(predictor, minus_starts, minus_ends)
        ) / (2.0 * ppm_step)
        rows = len(starts) * 6
        jacobian = np.zeros((rows, 14))
        residual_vector = np.zeros(rows)
        for axis in range(6):
            block = slice(axis * len(starts), (axis + 1) * len(starts))
            jacobian[block, 2 * axis] = reference[:, axis]
            jacobian[block, 2 * axis + 1] = 1.0
            jacobian[block, 12] = active_gains[axis] * d_reference_offset[:, axis]
            jacobian[block, 13] = active_gains[axis] * d_reference_ppm[:, axis]
            residual_vector[block] = residual[:, axis]
        ####
        return reference, residual_vector, jacobian
    ####

    for _ in range(20):
        _, residual_vector, jacobian = state(gains, biases, offset, ppm)
        column_scales = np.linalg.norm(jacobian, axis=0)
        if np.any(column_scales == 0.0):
            checked_design(jacobian, maximum_condition=maximum_condition)
        ####
        scaled = jacobian / np.where(column_scales > 0.0, column_scales, 1.0)
        scaled_delta = np.linalg.lstsq(scaled, residual_vector, rcond=None)[0]
        delta = scaled_delta / np.where(column_scales > 0.0, column_scales, 1.0)
        parameters = np.empty(14)
        parameters[0:12:2] = gains
        parameters[1:12:2] = biases
        parameters[12] = offset
        parameters[13] = ppm
        baseline_cost = float(residual_vector @ residual_vector)
        accepted = False
        alpha = 1.0
        for _ in range(24):
            candidate = parameters + alpha * delta
            candidate_gains = candidate[0:12:2]
            candidate_biases = candidate[1:12:2]
            candidate_offset = float(candidate[12])
            candidate_ppm = float(candidate[13])
            if (
                    np.all(candidate_gains > 0.0)
                    and offset_lo < candidate_offset < offset_hi
                    and ppm_lo < candidate_ppm < ppm_hi
            ):
                _, candidate_residual, _ = state(
                    candidate_gains, candidate_biases, candidate_offset, candidate_ppm,
                )
                candidate_cost = float(candidate_residual @ candidate_residual)
                if candidate_cost < baseline_cost:
                    gains = candidate_gains
                    biases = candidate_biases
                    offset = candidate_offset
                    ppm = candidate_ppm
                    accepted = True
                    break
                ####
            ####
            alpha *= 0.5
        ####
        if not accepted:
            break
        ####
        applied = alpha * delta
        if (
                np.max(np.abs(applied[:12])) < 1e-11
                and abs(applied[12]) < 1e-11
                and abs(applied[13]) < 1e-6
        ):
            break
        ####
    ####
    _, residual_vector, jacobian = state(gains, biases, offset, ppm)
    scaled, scales, _, condition = checked_design(jacobian, maximum_condition=maximum_condition)
    influence = np.linalg.pinv(scaled) / scales[:, None]
    dof = len(residual_vector) - 14
    variance = float(residual_vector @ residual_vector / dof)
    covariance = variance * influence @ influence.T
    return MultiChannelClockEstimate(
        gains,
        biases,
        offset,
        ppm,
        covariance,
        condition,
        float(np.sqrt(np.mean(residual_vector ** 2))),
    )
####


def predict_multichannel_clock(
        estimate: MultiChannelClockEstimate,
        predictor: Callable[[FloatArray, FloatArray], FloatArray],
        reported_starts_s: FloatArray,
        reported_ends_s: FloatArray,
        *, epoch_s: float = 0.0,
) -> FloatArray:
    """Predict held-out six-channel rates using a previously fitted clock map."""
    starts, ends = validate_intervals(reported_starts_s, reported_ends_s)
    corrected_starts, corrected_ends = _correct_clock_arrays(
        starts,
        ends,
        epoch_s=epoch_s,
        offset_s=estimate.relative_offset_s,
        frequency_error_ppm=estimate.clock_frequency_error_ppm,
    )
    reference = _predictor_rates(predictor, corrected_starts, corrected_ends)
    return reference * estimate.gains + estimate.biases
####
