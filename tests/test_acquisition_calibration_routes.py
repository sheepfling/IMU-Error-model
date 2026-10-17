"""Calibration-side tests for data sufficiency, FIR excitation and Earth-rate phase."""

from __future__ import annotations

from math import radians
import numpy as np
import pytest

from imu_error_model import AcquisitionConfig, ClockConfig, FirFilterConfig, effective_fir_kernels
from playground.calibration.acquisition_confidence import AcquisitionConfidencePoint, plan_filtered_heading
from playground.calibration.readout_identification import fit_readout_kernel, predict_readout
from playground.calibration.run_tranche5_campaign import (
    carousel_phase_experiment, filtered_noise_experiment, simulate_readout_record, thermal_step_experiment,
)


def test_known_excitation_recovers_effective_fir_and_axis_delays_on_heldout_data() -> None:
    cfg = AcquisitionConfig(
        accelerometer_filter=FirFilterConfig(taps=(.1,.2,.3,.4)),
        gyroscope_filter=FirFilterConfig(taps=(.4,.4,.2)),
        accelerometer_delay_samples=(0,1,2),gyroscope_delay_samples=(2,0,1),
    )
    x = np.random.default_rng(2).normal(size=(300,6))
    y,_,_ = simulate_readout_record(x,cfg)
    length = effective_fir_kernels(cfg).shape[1]
    fit = fit_readout_kernel(x,y,kernel_length=length)
    np.testing.assert_allclose(fit.kernels,effective_fir_kernels(cfg),atol=2e-14)
    heldout = np.random.default_rng(3).normal(size=(120,6))
    observed,_,_ = simulate_readout_record(heldout,cfg)
    predicted,first = predict_readout(fit,heldout)
    np.testing.assert_allclose(predicted[first:],observed[first:],atol=2e-14)
    assert all(x==length+1 for x in fit.rank_by_channel)
####


def test_noisy_known_input_identification_is_checked_out_of_sample() -> None:
    x = np.random.default_rng(22).normal(size=(800,6))
    cfg = AcquisitionConfig(gyroscope_filter=FirFilterConfig(taps=(.25,.5,.25)))
    y,_,_ = simulate_readout_record(x,cfg)
    y += np.random.default_rng(23).normal(0.,.005,y.shape)
    fitted = fit_readout_kernel(x,y,kernel_length=3)
    assert np.max(abs(fitted.kernels-effective_fir_kernels(cfg))) < .001
    assert min(fitted.residual_rms_by_channel) > .004
####


@pytest.mark.parametrize("mode",["static","short","mismatch","nan"])
def test_invalid_or_unobservable_readout_excitation_fails(mode: str) -> None:
    x = np.ones((100,6))
    y = x.copy()
    if mode == "short":
        x,y = x[:4],y[:4]
    elif mode == "mismatch":
        y = y[:-1]
    elif mode == "nan":
        y[10,0] = np.nan
    ####
    with pytest.raises(ValueError):
        fit_readout_kernel(x,y,kernel_length=4)
    ####
####


@pytest.mark.parametrize("revolutions",[1,2])
def test_carousel_phase_error_does_not_average_away_and_fitted_transfer_corrects(revolutions: int) -> None:
    cfg = AcquisitionConfig(sample_period_s=.1,gyroscope_filter=FirFilterConfig(taps=(.25,.5,.25)),clock=ClockConfig(offset_s=.025))
    x = np.random.default_rng(8).normal(size=(150,6))
    y,_,_ = simulate_readout_record(x,cfg)
    fitted = fit_readout_kernel(x,y,kernel_length=3)
    result = carousel_phase_experiment(cfg,fitted.kernels,estimated_clock_offset_s=.025,revolutions=revolutions)
    # Symmetric FIR group delay .1 s + offset .025 s, at 6 degrees/s.
    assert result["uncorrected_heading_error_deg"] == pytest.approx(-.75,abs=1e-10)
    assert abs(result["corrected_heading_error_deg"]) < 1e-10
####


def test_carousel_rejects_channel_mismatch_instead_of_claiming_scalar_phase_correction() -> None:
    cfg = AcquisitionConfig(gyroscope_delay_samples=(0,1,0))
    with pytest.raises(ValueError):
        carousel_phase_experiment(cfg,effective_fir_kernels(cfg),estimated_clock_offset_s=0.)
    ####
####


def test_thermal_step_exposes_room_vs_die_calibration_mismatch() -> None:
    results = thermal_step_experiment()
    assert abs(results[0]["ambient_compensated_heading_error_deg"]) > 20.
    assert abs(results[-1]["ambient_compensated_heading_error_deg"]) < .1
    assert max(abs(x["oracle_die_compensated_heading_error_deg"]) for x in results) < 1e-10
####


def test_exact_filter_mean_intervals_cover_and_iid_intervals_do_not() -> None:
    result = filtered_noise_experiment()
    assert .93 < result["exact_95_interval_coverage"] < .97
    assert result["naive_iid_95_interval_coverage"] < .4
    assert result["exact_to_naive_sigma_ratio"] > 4.8
####


def _plan(bound: float | None, *, count: int = 1000, looks: int = 1) -> AcquisitionConfidencePoint:
    return plan_filtered_heading(
        config=AcquisitionConfig(),retained_samples=count,gyro_noise_density_radps_sqrt_s=2e-6,
        latitude_rad=radians(34.7304),target_half_width_rad=radians(.25),
        systematic_heading_bound_rad=bound,planned_looks=looks,
    )
####


def test_unknown_or_excessive_systematics_cannot_pass_with_more_samples() -> None:
    assert _plan(None).status == "missing_systematic_bound"
    assert _plan(radians(1.),count=100000).status == "systematic_floor"
    assert _plan(radians(.03),count=120000).target_met
####


def test_predeclared_multiple_looks_widen_intervals() -> None:
    single = _plan(radians(.03),count=30000)
    multiple = _plan(radians(.03),count=30000,looks=7)
    assert single.target_met
    assert not multiple.target_met
    assert single.half_width_rad is not None
    assert multiple.half_width_rad is not None
    assert multiple.half_width_rad > single.half_width_rad
####


@pytest.mark.parametrize("bound",[-1.,float("nan")])
def test_invalid_systematic_bounds_are_not_silently_zero(bound: float) -> None:
    with pytest.raises(ValueError):
        _plan(bound)
    ####
####
