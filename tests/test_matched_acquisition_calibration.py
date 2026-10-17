import numpy as np
import pytest

from imu_error_model import AxisConfig, ImuConfig
from imu_error_model.acquisition import AcquiredImuModel
from imu_error_model.acquisition_config import AcquisitionConfig, ClockConfig, FirFilterConfig
from imu_error_model.readout import FirReadout, effective_fir_kernels
from playground.calibration.earth_reference import LocalEarthReference
from playground.calibration.known_reference import (
    CalibrationDataset,
    calibration_parameter_error,
    fit_known_reference_calibration,
)
from playground.calibration.matched_acquisition import (
    RelativeClockCorrection,
    build_matched_support_parameterization,
    fit_matched_calibration,
    fit_multichannel_clock_alignment,
    match_aperture_reference,
    predict_multichannel_clock,
    propagate_matched_support_uncertainty,
    reconstruct_matched_reference,
)
from playground.calibration.readout_identification import ReadoutKernelEstimate, fit_readout_kernel
from playground.calibration.reference_equipment import AngleLaw, ApertureReference, RateTableReference, TableGeometry
from playground.calibration.reference_identification import UnobservableCalibrationError
from playground.calibration.reference_observations import (
    EncoderErrorConfig,
    MeasuredImuRecord,
    fit_encoder_law,
    simulate_encoder,
    simulate_table_record,
)


def _sensor_config() -> ImuConfig:
    return ImuConfig(
        accelerometer=AxisConfig(
            fixed_bias=(.03, -.02, .015),
            sensitivity_matrix=((1.012, .006, -.004), (.003, .991, .005), (-.002, .004, 1.018)),
        ),
        gyroscope=AxisConfig(
            fixed_bias=(2e-3, -1e-3, 1.5e-3),
            sensitivity_matrix=((1.018, .007, -.003), (.002, .994, .004), (.005, -.002, 1.011)),
        ),
        gyroscope_specific_force_sensitivity=((2e-4, 3e-5, -1e-5), (1e-5, -1.5e-4, 4e-5), (2e-5, 1e-5, 1.2e-4)),
    )
####


def _acquisition_config() -> AcquisitionConfig:
    return AcquisitionConfig(
        sample_period_s=.02,
        accelerometer_filter=FirFilterConfig(taps=(.2, .5, .3)),
        gyroscope_filter=FirFilterConfig(taps=(.1, .2, .4, .3)),
        accelerometer_delay_samples=(0, 1, 2),
        gyroscope_delay_samples=(2, 0, 1),
    )
####


def _dynamic_inputs(samples: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    t = (np.arange(samples) + .5) * dt
    force = np.column_stack((
        4.0 * np.sin(.73 * t) + 1.2 * np.cos(.19 * t),
        3.0 * np.cos(.51 * t) - 1.5 * np.sin(.27 * t),
        -7.0 + 2.0 * np.sin(.37 * t) + .8 * np.cos(.91 * t),
    ))
    rate = np.column_stack((
        .8 * np.sin(.43 * t) + .3 * np.cos(.17 * t),
        -.7 * np.cos(.61 * t) + .25 * np.sin(.29 * t),
        .5 + .6 * np.sin(.31 * t) - .2 * np.cos(.83 * t),
    ))
    return force, rate
####


def _simulate_dynamic_record(
        sensor_config: ImuConfig, acquisition_config: AcquisitionConfig, *, samples: int = 800,
) -> tuple[MeasuredImuRecord, ApertureReference]:
    dt = acquisition_config.sample_period_s
    force, rate = _dynamic_inputs(samples, dt)
    model = AcquiredImuModel(sensor_config, acquisition_config, sensor_rng=np.random.default_rng(2))
    outputs = []
    for index in range(samples):
        outputs.append(model.measure_ideal_rates(
            start_time=index * dt,
            end_time=(index + 1) * dt,
            specific_force_body=force[index],
            angular_rate_body=rate[index],
            ambient_temperature_celsius=25.0,
        ))
    ####
    starts = np.array([sample.start_time_s for sample in outputs])
    ends = np.array([sample.end_time_s for sample in outputs])
    record = MeasuredImuRecord(
        starts,
        ends,
        np.array([sample.rates_si for sample in outputs]),
        np.array([sample.valid_channels for sample in outputs]),
        np.full(samples, 25.0),
        "dynamic-calibration",
    )
    reference = ApertureReference(
        np.arange(samples, dtype=np.float64) * dt,
        (np.arange(samples, dtype=np.float64) + 1) * dt,
        force,
        rate,
        np.zeros((samples, 3, 3)),
    )
    return record, reference
####


def _identify_readout(config: AcquisitionConfig, *, noise_std: float = 0.0) -> ReadoutKernelEstimate:
    rng = np.random.default_rng(11)
    reference = rng.normal(size=(1200, 6))
    readout = FirReadout(config)
    observed = np.array([readout.step(row)[0] for row in reference])
    if noise_std:
        observed += rng.normal(0.0, noise_std, observed.shape)
    ####
    return fit_readout_kernel(reference, observed, kernel_length=effective_fir_kernels(config).shape[1])
####


def test_forward_matched_reference_recovers_intrinsics_without_deconvolution() -> None:
    sensor_config = _sensor_config()
    acquisition_config = _acquisition_config()
    readout = _identify_readout(acquisition_config)
    record, reference = _simulate_dynamic_record(sensor_config, acquisition_config)
    matched = match_aperture_reference(
        record,
        reference,
        readout,
        reference_source_ids=("independent-known-input",),
        readout_source_id="identified-readout",
    )
    estimate = fit_matched_calibration(matched)
    assert np.max(np.abs(calibration_parameter_error(estimate, sensor_config))) < 2e-11

    valid = np.all(record.valid_channels, axis=1)
    naive = CalibrationDataset(
        labels=("naive",) * int(np.count_nonzero(valid)),
        start_times=reference.starts_s[valid],
        end_times=reference.ends_s[valid],
        temperatures_celsius=np.full(np.count_nonzero(valid), 25.0),
        specific_force_body=reference.force[valid],
        angular_rate_body=reference.rate[valid],
        measured_acceleration=record.rates_si[valid, :3],
        measured_angular_rate=record.rates_si[valid, 3:],
    )
    naive_fit = fit_known_reference_calibration(naive)
    assert np.max(np.abs(calibration_parameter_error(naive_fit, sensor_config))) > 2e-3
####


def test_readout_identification_reports_tap_uncertainty() -> None:
    fit = _identify_readout(_acquisition_config(), noise_std=.003)
    assert fit.coefficient_covariance_by_channel.shape == (6, fit.kernels.shape[1] + 1, fit.kernels.shape[1] + 1)
    assert fit.tap_covariance.shape == (fit.kernels.size, fit.kernels.size)
    assert np.trace(fit.tap_covariance) > 0.0
    assert np.linalg.eigvalsh(fit.tap_covariance).min() >= -1e-14
####


def _identity_readout() -> ReadoutKernelEstimate:
    covariance = np.zeros((6, 2, 2))
    return ReadoutKernelEstimate(
        np.ones((6, 1)),
        np.zeros(6),
        (2,) * 6,
        (1.0,) * 6,
        (0.0,) * 6,
        covariance,
        "known_identity_for_test",
    )
####


def test_reported_clock_is_corrected_before_reference_reconstruction() -> None:
    earth = LocalEarthReference(np.deg2rad(34.7304))
    law = AngleLaw(rate_radps=.8, sine_rad=.3, cosine_rad=.1, frequency_hz=.23)
    geometry = TableGeometry(tilt_rotvec_ned_rad=(.01, -.02, .003), wobble_x_rad=.002, wobble_y_rad=.003)
    acquisition = AcquisitionConfig(sample_period_s=.02, clock=ClockConfig(offset_s=.031, frequency_error_ppm=750.0))
    physical_edges = np.arange(501, dtype=np.float64) * acquisition.sample_period_s
    record, truth = simulate_table_record(
        AcquiredImuModel(ImuConfig(), acquisition),
        RateTableReference(earth, law, geometry),
        physical_edges[:-1],
        physical_edges[1:],
    )
    encoder = simulate_encoder(
        law,
        np.linspace(-.5, physical_edges[-1] + .5, 1501),
        EncoderErrorConfig(),
        rng=np.random.default_rng(4),
        source_id="encoder-clock-anchor",
    )
    encoder_fit = fit_encoder_law(encoder, frequency_hz=law.frequency_hz)
    matched = reconstruct_matched_reference(
        record,
        encoder_fit,
        surveyed_geometry=geometry,
        earth=earth,
        readout=_identity_readout(),
        clock_correction=RelativeClockCorrection(offset_s=.031, frequency_error_ppm=750.0),
        survey_source_id="survey",
        readout_source_id="identity-readout",
    )
    valid = np.all(record.valid_channels, axis=1)
    np.testing.assert_allclose(matched.accelerometer_force_by_output[:, 0, :], truth.force[valid], atol=2e-11)
    np.testing.assert_allclose(matched.gyroscope_rate_by_output[:, 2, :], truth.rate[valid], atol=2e-11)

    wrong = reconstruct_matched_reference(
        record,
        encoder_fit,
        surveyed_geometry=geometry,
        earth=earth,
        readout=_identity_readout(),
        clock_correction=RelativeClockCorrection(),
        survey_source_id="survey",
        readout_source_id="identity-readout",
    )
    assert np.max(np.abs(wrong.gyroscope_rate_by_output[:, 2, :] - truth.rate[valid])) > 1e-4
####


def _mean_sine(starts: np.ndarray, ends: np.ndarray, frequency_hz: float, phase: float = 0.0) -> np.ndarray:
    omega = 2.0 * np.pi * frequency_hz
    return (np.cos(omega * starts + phase) - np.cos(omega * ends + phase)) / (omega * (ends - starts))
####


def _clock_predictor(frequencies: tuple[float, ...], signs: tuple[float, ...] = (1, 1, 1, 1, 1, 1)):
    def predictor(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        columns = []
        for axis, frequency in enumerate(frequencies):
            phase = .17 * axis
            columns.append(signs[axis] * (0.2 * (axis - 2) + _mean_sine(starts, ends, frequency, phase)))
        ####
        return np.column_stack(columns)
    ####
    return predictor
####


def test_multichannel_clock_fit_recovers_offset_frequency_and_generalizes() -> None:
    dt = .02
    physical_edges = np.arange(1001, dtype=np.float64) * dt
    offset = .027
    ppm = 420.0
    reported_edges = offset + physical_edges * (1.0 + ppm * 1e-6)
    training_predictor = _clock_predictor((.17, .23, .31, .41, .53, .67))
    gains = np.array((1.01, .99, 1.02, 1.015, .985, 1.005))
    biases = np.array((.02, -.01, .015, .002, -.003, .001))
    training_reference = training_predictor(physical_edges[:-1], physical_edges[1:])
    measured = training_reference * gains + biases
    fit = fit_multichannel_clock_alignment(
        training_predictor,
        reported_edges[:-1],
        reported_edges[1:],
        measured,
        offset_bounds_s=(-.05, .05),
        frequency_error_bounds_ppm=(-1000.0, 1000.0),
    )
    assert fit.relative_offset_s == pytest.approx(offset, abs=2e-6)
    assert fit.clock_frequency_error_ppm == pytest.approx(ppm, abs=.1)
    np.testing.assert_allclose(fit.gains, gains, atol=2e-6)
    np.testing.assert_allclose(fit.biases, biases, atol=2e-6)

    held_edges = np.arange(501, dtype=np.float64) * .03 + 40.0
    held_reported = offset + held_edges * (1.0 + ppm * 1e-6)
    held_predictor = _clock_predictor((.29, .37, .47, .59, .71, .83), (-1, 1, -1, 1, -1, 1))
    expected = held_predictor(held_edges[:-1], held_edges[1:]) * gains + biases
    predicted = predict_multichannel_clock(fit, held_predictor, held_reported[:-1], held_reported[1:])
    np.testing.assert_allclose(predicted, expected, atol=5e-6)
####


def test_constant_reference_exposes_clock_frequency_sensor_scale_gauge() -> None:
    starts = np.arange(200, dtype=np.float64) * .02
    ends = starts + .02

    def constant_predictor(s: np.ndarray, e: np.ndarray) -> np.ndarray:
        del e
        return np.tile(np.arange(1.0, 7.0), (len(s), 1))
    ####

    with pytest.raises((ValueError, UnobservableCalibrationError)):
        fit_multichannel_clock_alignment(constant_predictor, starts, ends, constant_predictor(starts, ends))
    ####
####


def test_fitted_readout_uncertainty_propagates_to_intrinsic_covariance() -> None:
    acquisition = _acquisition_config()
    readout = _identify_readout(acquisition, noise_std=.004)
    record, reference = _simulate_dynamic_record(_sensor_config(), acquisition, samples=500)
    encoder_times = np.linspace(-1.0, 12.0, 900)
    law = AngleLaw(rate_radps=.5, sine_rad=.1, frequency_hz=.2)
    encoder_fit = fit_encoder_law(
        simulate_encoder(law, encoder_times, EncoderErrorConfig(angle_noise_std_rad=2e-5), rng=np.random.default_rng(90)),
        frequency_hz=.2,
    )
    parameterization = build_matched_support_parameterization(
        readout,
        (encoder_fit,),
        RelativeClockCorrection(),
        clock_covariance_offset_ppm=np.diag((2e-6 ** 2, .2 ** 2)),
    )

    def builder(
            active_readout: ReadoutKernelEstimate,
            encoder_fits: tuple,
            clock: RelativeClockCorrection,
    ):
        # The direct known-reference test has no encoder/clock dependence. They
        # remain in the joint parameterization and therefore contribute zero
        # sensitivity rather than being silently removed from the bookkeeping.
        assert len(encoder_fits) == 1
        assert isinstance(clock, RelativeClockCorrection)
        return match_aperture_reference(
            record,
            reference,
            active_readout,
            reference_source_ids=("known-reference",),
            readout_source_id="identified-readout",
        )
    ####

    result = propagate_matched_support_uncertainty(builder, parameterization)
    assert np.trace(result.support_covariance) > 0.0
    assert np.linalg.eigvalsh(result.calibration.joint_coefficient_covariance).min() >= -1e-10
    assert np.trace(result.calibration.joint_coefficient_covariance) > np.trace(result.conditional_calibration_covariance)
####
