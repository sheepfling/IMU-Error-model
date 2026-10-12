"""Reproduce matched acquisition-aware calibration and held-out support tests.

All results are synthetic verification only. Readout and relative-clock models are
identified on independent records. Calibration receives observed IMU/encoder data
plus declared survey estimates; evaluator truth is used only for scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from imu_error_model import AxisConfig, ImuConfig
from imu_error_model.acquisition import AcquiredImuModel
from imu_error_model.acquisition_config import AcquisitionConfig, ClockConfig, FirFilterConfig
from imu_error_model.readout import FirReadout, effective_fir_kernels

from .earth_reference import LocalEarthReference
from .known_reference import calibration_parameter_error
from .matched_acquisition import (
    MatchedReferenceData,
    RelativeClockCorrection,
    build_matched_support_parameterization,
    combine_matched_records,
    fit_matched_calibration,
    fit_multichannel_clock_alignment,
    predict_matched_measurement,
    propagate_matched_support_uncertainty,
    reconstruct_matched_reference,
)
from .readout_identification import ReadoutKernelEstimate, fit_readout_kernel
from .reference_equipment import AngleLaw, FloatArray, RateTableReference, TableGeometry
from .reference_observations import (
    EncoderErrorConfig,
    EncoderFit,
    MeasuredImuRecord,
    ObservedCalibrationData,
    combine_observed_records,
    fit_encoder_law,
    fit_observed_calibration,
    reconstruct_reference,
    simulate_encoder,
    simulate_table_record,
)


MOUNTS = (
    (0.0, 0.0, 0.0),
    (np.pi, 0.0, 0.0),
    (0.0, np.pi / 2, 0.0),
    (0.0, -np.pi / 2, 0.0),
    (np.pi / 2, 0.0, 0.0),
    (-np.pi / 2, 0.0, 0.0),
)


def sensor_config() -> ImuConfig:
    return ImuConfig(
        accelerometer=AxisConfig(
            fixed_bias=(.02, -.03, .04),
            sensitivity_matrix=((1.01, .003, -.002), (.001, .99, .004), (-.002, .003, 1.02)),
            white_noise_density=8e-5,
        ),
        gyroscope=AxisConfig(
            fixed_bias=(2e-4, -3e-4, 1e-4),
            sensitivity_matrix=((1.02, .004, -.003), (.002, .99, .001), (.003, -.002, 1.01)),
            white_noise_density=1.5e-6,
        ),
        gyroscope_specific_force_sensitivity=((1e-4, 2e-5, 0.0), (0.0, -1e-4, 2e-5), (3e-5, 0.0, 1e-4)),
    )
####


def acquisition_config() -> AcquisitionConfig:
    return AcquisitionConfig(
        sample_period_s=.02,
        accelerometer_filter=FirFilterConfig(taps=(.12, .28, .36, .24)),
        gyroscope_filter=FirFilterConfig(taps=(.08, .19, .31, .27, .15)),
        accelerometer_delay_samples=(0, 1, 2),
        gyroscope_delay_samples=(2, 0, 1),
        clock=ClockConfig(offset_s=.021, frequency_error_ppm=360.0),
    )
####


def _rms(value: FloatArray) -> float:
    return float(np.sqrt(np.mean(np.square(value))))
####


def _mean_sine(starts: FloatArray, ends: FloatArray, frequency_hz: float, phase: float) -> FloatArray:
    omega = 2.0 * np.pi * frequency_hz
    return (np.cos(omega * starts + phase) - np.cos(omega * ends + phase)) / (omega * (ends - starts))
####


def _timing_predictor(starts: FloatArray, ends: FloatArray) -> FloatArray:
    frequencies = (.13, .19, .27, .37, .49, .61)
    return np.column_stack([
        .15 * (axis - 2) + _mean_sine(starts, ends, frequency, .14 * axis)
        for axis, frequency in enumerate(frequencies)
    ])
####


def identify_readout(config: AcquisitionConfig) -> tuple[ReadoutKernelEstimate, dict[str, object]]:
    rng = np.random.default_rng(7101)
    inputs = rng.normal(size=(3200, 6))
    readout = FirReadout(config)
    observed = np.array([readout.step(row)[0] for row in inputs])
    observed += rng.normal(0.0, 5e-4, observed.shape)
    estimate = fit_readout_kernel(inputs, observed, kernel_length=effective_fir_kernels(config).shape[1])
    heldout = rng.normal(size=(900, 6))
    test_readout = FirReadout(config)
    measured = np.array([test_readout.step(row)[0] for row in heldout])
    predicted = np.column_stack([
        np.convolve(heldout[:, axis], estimate.kernels[axis], mode="full")[:len(heldout)] + estimate.intercept[axis]
        for axis in range(6)
    ])
    first = estimate.kernels.shape[1] - 1
    return estimate, {
        "training_samples": len(inputs),
        "kernel_length": estimate.kernels.shape[1],
        "maximum_kernel_error": float(np.max(np.abs(estimate.kernels - effective_fir_kernels(config)))),
        "heldout_prediction_rms": _rms(predicted[first:] - measured[first:]),
        "mean_tap_sigma": float(np.mean(np.sqrt(np.maximum(np.diagonal(estimate.tap_covariance), 0.0)))),
    }
####


def identify_relative_clock(config: AcquisitionConfig) -> tuple[RelativeClockCorrection, FloatArray, dict[str, object]]:
    physical_edges = np.arange(1501, dtype=np.float64) * .02
    reported_edges = (
        config.clock.epoch_s
        + config.clock.offset_s
        + (physical_edges - config.clock.epoch_s) * (1.0 + config.clock.frequency_error_ppm * 1e-6)
    )
    gains = np.array((1.01, .995, 1.006, 1.012, .989, 1.004))
    biases = np.array((.01, -.02, .005, .001, -.0015, .0008))
    rng = np.random.default_rng(7102)
    measured = _timing_predictor(physical_edges[:-1], physical_edges[1:]) * gains + biases
    measured += rng.normal(0.0, 2e-4, measured.shape)
    fit = fit_multichannel_clock_alignment(
        _timing_predictor,
        reported_edges[:-1],
        reported_edges[1:],
        measured,
        offset_bounds_s=(-.05, .05),
        frequency_error_bounds_ppm=(-1000.0, 1000.0),
    )
    correction = RelativeClockCorrection(
        epoch_s=config.clock.epoch_s,
        offset_s=fit.relative_offset_s,
        frequency_error_ppm=fit.clock_frequency_error_ppm,
    )
    return correction, fit.covariance[12:, 12:], {
        "training_samples": len(measured),
        "injected_offset_s": config.clock.offset_s,
        "estimated_offset_s": fit.relative_offset_s,
        "offset_error_s": fit.relative_offset_s - config.clock.offset_s,
        "injected_frequency_error_ppm": config.clock.frequency_error_ppm,
        "estimated_frequency_error_ppm": fit.clock_frequency_error_ppm,
        "frequency_error_ppm_error": fit.clock_frequency_error_ppm - config.clock.frequency_error_ppm,
        "residual_rms": fit.residual_rms,
        "scaled_condition_number": fit.scaled_condition_number,
    }
####


def exact_readout(config: AcquisitionConfig) -> ReadoutKernelEstimate:
    kernels = effective_fir_kernels(config)
    length = kernels.shape[1]
    return ReadoutKernelEstimate(
        kernels,
        np.zeros(6),
        (length + 1,) * 6,
        (1.0,) * 6,
        (0.0,) * 6,
        np.zeros((6, length + 1, length + 1)),
        "evaluator_exact_readout",
    )
####


def _corrected_record(record: MeasuredImuRecord, clock: RelativeClockCorrection) -> MeasuredImuRecord:
    return MeasuredImuRecord(
        clock.to_reference_time(record.starts_s),
        clock.to_reference_time(record.ends_s),
        record.rates_si,
        record.valid_channels,
        record.temperatures_celsius,
        record.source_id,
    )
####


def calibration_sessions(
        *, earth: LocalEarthReference, sensor: ImuConfig, acquisition: AcquisitionConfig,
        readout: ReadoutKernelEstimate, clock: RelativeClockCorrection,
        prefix: str, frequencies: tuple[float, ...], reverse: bool,
) -> tuple[
    tuple[MatchedReferenceData, ...],
    tuple[ObservedCalibrationData, ...],
    tuple[MeasuredImuRecord, ...],
    tuple[EncoderFit, ...],
    tuple[TableGeometry, ...],
]:
    matched: list[MatchedReferenceData] = []
    naive: list[ObservedCalibrationData] = []
    records: list[MeasuredImuRecord] = []
    encoder_fits: list[EncoderFit] = []
    geometries: list[TableGeometry] = []
    for index, (mount, frequency) in enumerate(zip(MOUNTS, frequencies, strict=True)):
        sign = -1.0 if reverse and index % 2 == 0 else 1.0
        law = AngleLaw(
            offset_rad=.01 * index,
            rate_radps=sign * (.55 + .08 * index),
            sine_rad=.22,
            cosine_rad=.09,
            frequency_hz=frequency,
        )
        geometry = TableGeometry(
            platform_from_sensor_rotvec_rad=mount,
            tilt_rotvec_ned_rad=(.008, -.012, .002),
            wobble_x_rad=.0015,
            wobble_y_rad=.002,
            lever_arm_sensor_m=(.02, -.015, .008),
        )
        duration = 10.0
        samples = round(duration / acquisition.sample_period_s)
        edges = np.arange(samples + 1, dtype=np.float64) * acquisition.sample_period_s
        model = AcquiredImuModel(sensor, acquisition, sensor_rng=np.random.default_rng(7200 + index + (100 if reverse else 0)))
        record, _ = simulate_table_record(
            model,
            RateTableReference(earth, law, geometry),
            edges[:-1],
            edges[1:],
            source_id=f"{prefix}-imu-{index}",
        )
        encoder_times = np.linspace(-.6, duration + .6, 1501)
        encoder = simulate_encoder(
            law,
            encoder_times,
            EncoderErrorConfig(angle_noise_std_rad=6e-6, quantization_rad=1e-6),
            rng=np.random.default_rng(7300 + index + (100 if reverse else 0)),
            source_id=f"{prefix}-encoder-{index}",
        )
        encoder_fit = fit_encoder_law(encoder, frequency_hz=frequency)
        matched.append(reconstruct_matched_reference(
            record,
            encoder_fit,
            surveyed_geometry=geometry,
            earth=earth,
            readout=readout,
            clock_correction=clock,
            survey_source_id=f"{prefix}-survey",
            readout_source_id="independent-readout-id",
        ))
        naive.append(reconstruct_reference(
            _corrected_record(record, clock),
            encoder_fit,
            surveyed_geometry=geometry,
            earth=earth,
            survey_source_id=f"{prefix}-survey",
        ))
        records.append(record)
        encoder_fits.append(encoder_fit)
        geometries.append(geometry)
    ####
    return tuple(matched), tuple(naive), tuple(records), tuple(encoder_fits), tuple(geometries)
####


def run_campaign(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    earth = LocalEarthReference(np.deg2rad(34.7304))
    sensor = sensor_config()
    acquisition = acquisition_config()
    readout, readout_metrics = identify_readout(acquisition)
    clock, clock_covariance, clock_metrics = identify_relative_clock(acquisition)
    training_frequencies = (.17, .19, .21, .23, .25, .27)
    training, naive_records, raw_records, encoder_fits, geometries = calibration_sessions(
        earth=earth,
        sensor=sensor,
        acquisition=acquisition,
        readout=readout,
        clock=clock,
        prefix="training",
        frequencies=training_frequencies,
        reverse=False,
    )
    matched_training = combine_matched_records(training)
    matched_fit = fit_matched_calibration(matched_training)
    naive_fit = fit_observed_calibration(combine_observed_records(naive_records))

    oracle_parts: list[MatchedReferenceData] = []
    for record, encoder_fit, geometry in zip(raw_records, encoder_fits, geometries, strict=True):
        oracle_parts.append(reconstruct_matched_reference(
            record,
            encoder_fit,
            surveyed_geometry=geometry,
            earth=earth,
            readout=exact_readout(acquisition),
            clock_correction=RelativeClockCorrection(
                offset_s=acquisition.clock.offset_s,
                frequency_error_ppm=acquisition.clock.frequency_error_ppm,
            ),
            survey_source_id="training-survey",
            readout_source_id="evaluator-exact-readout",
        ))
    ####
    oracle_fit = fit_matched_calibration(combine_matched_records(tuple(oracle_parts)))

    heldout_frequencies = (.31, .37, .43, .49, .55, .61)
    heldout, _, _, _, _ = calibration_sessions(
        earth=earth,
        sensor=sensor,
        acquisition=acquisition,
        readout=readout,
        clock=clock,
        prefix="heldout",
        frequencies=heldout_frequencies,
        reverse=True,
    )
    heldout_data = combine_matched_records(heldout)
    observed = np.column_stack((heldout_data.measured_acceleration, heldout_data.measured_angular_rate))
    heldout_metrics = {}
    for name, fit in (("unmatched_intrinsic_fit", naive_fit), ("matched_identified_support", matched_fit), ("matched_oracle_support", oracle_fit)):
        prediction = predict_matched_measurement(heldout_data, fit)
        heldout_metrics[name] = {
            "parameter_max_abs_error": float(np.max(np.abs(calibration_parameter_error(fit, sensor)))),
            "heldout_sensor_prediction_rms": _rms(prediction - observed),
            "training_accelerometer_residual_rms": fit.accelerometer_residual_rms,
            "training_gyroscope_residual_rms": fit.gyroscope_residual_rms,
        }
    ####

    def support_builder(
            active_readout: ReadoutKernelEstimate,
            active_encoder_fits: tuple[EncoderFit, ...],
            active_clock: RelativeClockCorrection,
    ) -> MatchedReferenceData:
        parts = []
        for record, encoder_fit, geometry in zip(raw_records, active_encoder_fits, geometries, strict=True):
            parts.append(reconstruct_matched_reference(
                record,
                encoder_fit,
                surveyed_geometry=geometry,
                earth=earth,
                readout=active_readout,
                clock_correction=active_clock,
                survey_source_id="training-survey",
                readout_source_id="independent-readout-id",
            ))
        ####
        return combine_matched_records(tuple(parts))
    ####

    parameterization = build_matched_support_parameterization(
        readout,
        encoder_fits,
        clock,
        clock_covariance_offset_ppm=clock_covariance,
    )
    uncertainty = propagate_matched_support_uncertainty(support_builder, parameterization)
    np.savez_compressed(
        output / "matched-support-covariance.npz",
        conditional=uncertainty.conditional_calibration_covariance,
        support=uncertainty.support_covariance,
        total=uncertainty.calibration.joint_coefficient_covariance,
        jacobian=uncertainty.coefficient_support_jacobian,
    )
    results = {
        "scope": "synthetic verification only",
        "readout_identification": readout_metrics,
        "relative_clock_identification": clock_metrics,
        "training_retained_rows": len(matched_training.labels),
        "heldout_retained_rows": len(heldout_data.labels),
        "training_frequencies_hz": training_frequencies,
        "heldout_frequencies_hz": heldout_frequencies,
        "heldout_rate_reversal": True,
        "strategies": heldout_metrics,
        "uncertainty": {
            "conditional_covariance_trace": float(np.trace(uncertainty.conditional_calibration_covariance)),
            "identified_support_covariance_trace": float(np.trace(uncertainty.support_covariance)),
            "total_covariance_trace": float(np.trace(uncertainty.calibration.joint_coefficient_covariance)),
            "support_parameter_count": len(uncertainty.support_parameter_names),
            "support_covariance_assumption": uncertainty.calibration.covariance_assumption,
        },
        "claims": {
            "deconvolution_used": False,
            "hidden_acquisition_config_used_by_matched_fit": False,
            "encoder_law_frequency_is_declared_protocol": True,
            "relative_clock_attribution_to_specific_device": False,
            "general_errors_in_variables_bias_correction": False,
        },
    }
    (output / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps(results, indent=2, allow_nan=False))
    return results
####


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/tranche7"))
    args = parser.parse_args()
    run_campaign(args.output)
    return 0
####


if __name__ == "__main__":
    raise SystemExit(main())
####
