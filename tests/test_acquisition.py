"""Regression and independent law checks for the opt-in acquisition boundary."""

from __future__ import annotations

from math import exp

import numpy as np
import pytest
from pydantic import ValidationError

from imu_error_model import (
    AcquisitionCheckpoint, AcquisitionConfig, AcquiredImuModel, AcquiredImuModelProtocol, AcquiredImuSample,
    AxisConfig, ClockConfig, FirFilterConfig, ImuConfig, ImuModel, ImuModelProtocol,
    ThermalLagConfig, effective_fir_kernels, filtered_white_mean_covariance,
    fir_frequency_response, thermal_interval, thermal_settling_time_s,
)
from imu_error_model.readout import FirReadout, clock_tag
from imu_error_model.thermal_dynamics import remaining_thermal_error_celsius


def _step(model: AcquiredImuModel, i: int, *, temperature: float = 25.0) -> AcquiredImuSample:
    dt = model.acquisition_config.sample_period_s
    return model.measure_ideal_rates(
        start_time=i*dt, end_time=(i+1)*dt, specific_force_body=np.array([1., 2., 3.]),
        angular_rate_body=np.array([.1, .2, .3]), ambient_temperature_celsius=temperature,
    )
####


def test_acquisition_is_explicitly_not_the_legacy_contract() -> None:
    m = AcquiredImuModel()
    assert isinstance(m, AcquiredImuModelProtocol)
    assert not isinstance(m, ImuModelProtocol)
    assert not hasattr(_step(m, 0), "die_temperature_celsius")
    assert not hasattr(_step(m, 1), "physical_time_s")
####


def test_identity_acquisition_matches_legacy_sensor_seeded_noise() -> None:
    config = ImuConfig(
        accelerometer=AxisConfig(white_noise_density=.01, fixed_bias=(.1,.2,.3)),
        gyroscope=AxisConfig(white_noise_density=1e-5, turn_on_bias_std=1e-5),
    )
    core = ImuModel(config, rng=np.random.default_rng(82))
    acquired = AcquiredImuModel(config, sensor_rng=np.random.default_rng(82))
    for i in range(30):
        raw = core.measure_ideal_rates(
            start_time=i*.01, end_time=(i+1)*.01, specific_force_body=np.array([1.,2.,3.]),
            angular_rate_body=np.array([.1,.2,.3]), temperature_celsius=25.,
        )
        out = _step(acquired, i)
        np.testing.assert_array_equal(out.acceleration, raw.acceleration)
        np.testing.assert_array_equal(out.angular_rate, raw.angular_rate)
        assert out.is_valid
        assert out.start_time_s == raw.start_time
        assert out.end_time_s == raw.end_time
    ####
####


@pytest.mark.parametrize("taps", [(0.5,0.5), (1.,), (-.1,.2,.9)])
def test_fir_matches_independent_convolution(taps: tuple[float, ...]) -> None:
    config = AcquisitionConfig(
        accelerometer_filter=FirFilterConfig(taps=taps), gyroscope_filter=FirFilterConfig(taps=taps),
        accelerometer_delay_samples=(0,1,3), gyroscope_delay_samples=(2,0,4),
    )
    readout = FirReadout(config)
    x = np.random.default_rng(1).normal(size=(100,6))
    y = []
    valid = []
    for row in x:
        rates, flags = readout.step(row)
        y.append(rates)
        valid.append(flags)
    ####
    kernels = effective_fir_kernels(config)
    for j in range(6):
        expected = np.convolve(x[:,j], kernels[j])[:len(x)]
        np.testing.assert_allclose(np.asarray(y)[:,j], expected, atol=1e-15)
    ####
    assert not all(valid[0])
    assert all(valid[len(taps)+3])
####


def test_filter_is_causal_and_history_is_not_reset_at_pose_boundaries() -> None:
    filt = FirReadout(AcquisitionConfig(gyroscope_filter=FirFilterConfig(taps=(.5,.5))))
    first = filt.step(np.ones(6))
    second = filt.step(np.full(6, 3.))
    assert first[1][3] is False
    assert second[0][3] == 2.
    before = filt.history()
    with pytest.raises(ValueError):
        filt.step(np.full(6, np.nan))
    ####
    assert filt.history() == before
####


@pytest.mark.parametrize("x", [1e-10, 1e-5, .1, 10., 1e6])
def test_thermal_exact_end_and_subdivision_consistency(x: float) -> None:
    one = thermal_interval(start_celsius=20., ambient_celsius=60., dt_s=x, time_constant_s=1.)
    first = thermal_interval(start_celsius=20., ambient_celsius=60., dt_s=x/2, time_constant_s=1.)
    second = thermal_interval(start_celsius=first.end_celsius, ambient_celsius=60., dt_s=x/2, time_constant_s=1.)
    assert one.end_celsius == pytest.approx(60.-40.*exp(-x), abs=1e-13)
    assert one.mean_celsius == pytest.approx((first.mean_celsius+second.mean_celsius)/2, abs=1e-12)
    assert one.end_celsius == pytest.approx(second.end_celsius, abs=1e-12)
####


def test_thermal_linear_bias_is_integrated_at_die_mean_not_ambient() -> None:
    model = AcquiredImuModel(
        ImuConfig(gyroscope=AxisConfig(thermal_bias_coefficient=(0., 1e-4, 0.))),
        AcquisitionConfig(sample_period_s=1., thermal=ThermalLagConfig(time_constant_s=30.)),
    )
    sample = model.measure_ideal_rates(
        start_time=0., end_time=1., specific_force_body=np.zeros(3), angular_rate_body=np.zeros(3),
        ambient_temperature_celsius=55.,
    )
    expected_mean = 55. - 30.*30.*(1.-exp(-1./30.))
    assert sample.angular_rate[1] == pytest.approx((expected_mean-25.)*1e-4)
    assert sample.temperature_celsius == 55.  # observed ambient, not latent die
####


def test_thermal_settling_budget() -> None:
    t = thermal_settling_time_s(initial_difference_celsius=30., tolerance_celsius=.1, time_constant_s=30.)
    assert t == pytest.approx(171.113474, abs=1e-6)
    assert remaining_thermal_error_celsius(initial_difference_celsius=30., elapsed_s=t, time_constant_s=30.) == pytest.approx(.1)
    assert thermal_settling_time_s(initial_difference_celsius=.01, tolerance_celsius=.1, time_constant_s=30.) == 0.
####


def test_clock_drift_offsets_and_shared_jittered_boundaries() -> None:
    config = AcquisitionConfig(clock=ClockConfig(offset_s=.25, frequency_error_ppm=100., timestamp_jitter_bound_s=.001, jitter_seed=4))
    a = AcquiredImuModel(acquisition_config=config)
    samples = [_step(a, i) for i in range(100)]
    for left, right in zip(samples, samples[1:]):
        assert left.end_time_s == right.start_time_s
        assert right.reported_dt_s > 0.
    ####
    assert sum(s.reported_dt_s for s in samples) == pytest.approx(samples[-1].end_time_s-samples[0].start_time_s)
    # Timestamp errors must not modify the measured rates or physical excitation.
    np.testing.assert_allclose([s.angular_rate for s in samples], np.tile([.1,.2,.3], (100,1)))
####


def test_clock_rate_integrates_to_apparent_gain_error() -> None:
    clock = ClockConfig(frequency_error_ppm=100.)
    assert (clock_tag(clock, 10., 1000)-clock_tag(clock, 0., 0))/10.-1. == pytest.approx(100e-6)
####


def test_clock_seed_is_independent_of_sensor_randomness() -> None:
    cfg = ImuConfig(gyroscope=AxisConfig(white_noise_density=.02))
    a = AcquiredImuModel(cfg, AcquisitionConfig(clock=ClockConfig(jitter_seed=1, timestamp_jitter_bound_s=.001)), sensor_rng=np.random.default_rng(9))
    b = AcquiredImuModel(cfg, AcquisitionConfig(clock=ClockConfig(jitter_seed=2, timestamp_jitter_bound_s=.001)), sensor_rng=np.random.default_rng(9))
    for i in range(15):
        x, y = _step(a,i), _step(b,i)
        assert x.rates_si == y.rates_si
        assert x.end_time_s != y.end_time_s
    ####
####


def test_checkpoint_roundtrip_full_state_and_resume() -> None:
    cfg = AcquisitionConfig(
        gyroscope_filter=FirFilterConfig(taps=(.25,.5,.25)), gyroscope_delay_samples=(1,2,0),
        thermal=ThermalLagConfig(time_constant_s=8.), clock=ClockConfig(timestamp_jitter_bound_s=.001, jitter_seed=7),
    )
    m = AcquiredImuModel(ImuConfig(gyroscope=AxisConfig(white_noise_density=1e-5, thermal_bias_coefficient=1e-5)), cfg, sensor_rng=np.random.default_rng(55))
    for i in range(8):
        _step(m,i,temperature=50.)
    ####
    encoded = m.snapshot().model_dump_json()
    r = AcquiredImuModel.from_checkpoint(AcquisitionCheckpoint.model_validate_json(encoded))
    for i in range(8,30):
        assert _step(m,i,temperature=40.) == _step(r,i,temperature=40.)
    ####
    assert m.snapshot().model_dump_json() == r.snapshot().model_dump_json()
####


def test_restore_validation_is_atomic_and_reset_clears_all_history() -> None:
    m = AcquiredImuModel()
    _step(m,0)
    before = m.snapshot()
    bad = before.model_copy(update={"sample_count": 2})
    with pytest.raises(ValueError):
        m.restore(bad)
    ####
    assert m.snapshot() == before
    m.reset()
    assert m.snapshot().sample_count == 0
    assert m.snapshot().filter_history_newest_first == ()
    assert _step(m,0).is_valid
####


@pytest.mark.parametrize("kind", ["gap","wrong_period","nan_force","nan_temperature"])
def test_rejected_measurements_leave_owned_state_unchanged(kind: str) -> None:
    m = AcquiredImuModel()
    _step(m,0)
    before = m.snapshot().model_dump_json()
    start = .02 if kind == "gap" else .01
    end = start + (.02 if kind == "wrong_period" else .01)
    with pytest.raises(ValueError):
        m.measure_ideal_rates(
            start_time=start, end_time=end,
            specific_force_body=np.full(3, np.nan if kind == "nan_force" else 0.), angular_rate_body=np.zeros(3),
            ambient_temperature_celsius=np.nan if kind == "nan_temperature" else 25.,
        )
    ####
    assert m.snapshot().model_dump_json() == before
####


@pytest.mark.parametrize("payload", [
    {"taps":(0.,0.)}, {"taps":(float("nan"),)}, {"taps":()}, {"typo":1},
])
def test_fir_invalid_config_fails_closed(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        FirFilterConfig.model_validate(payload)
    ####
####


@pytest.mark.parametrize("payload", [
    {"sample_period_s":0.}, {"gyroscope_delay_samples":(-1,0,0)}, {"gyroscope_delay_samples":(True,0,0)},
    {"clock":{"timestamp_jitter_bound_s":.006}}, {"clock":{"frequency_error_ppm":-1e6}},
    {"thermal":{"time_constant_s":-1.}}, {"schema_version":2},
])
def test_acquisition_invalid_config_fails_closed(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AcquisitionConfig.model_validate(payload)
    ####
####


def test_frequency_response_including_axis_delay() -> None:
    cfg = AcquisitionConfig(sample_period_s=.01, gyroscope_filter=FirFilterConfig(taps=(.5,.5)), gyroscope_delay_samples=(0,1,2))
    frequency = 7.
    response = fir_frequency_response(cfg, frequency)
    z = np.exp(-2j*np.pi*frequency*.01)
    np.testing.assert_allclose(response[3:], [(.5+.5*z)*z**i for i in range(3)])
    with pytest.raises(ValueError):
        fir_frequency_response(cfg, 50.)
    ####
####


def test_filtered_mean_covariance_matches_independent_linear_operator() -> None:
    cfg = AcquisitionConfig(gyroscope_filter=FirFilterConfig(taps=(.25,.5,.25)), gyroscope_delay_samples=(0,1,2))
    matrix = np.random.default_rng(1).normal(size=(6,6))
    q = matrix@matrix.T
    n = 13
    kernels = effective_fir_kernels(cfg)
    length = kernels.shape[1]
    weights = np.zeros((6,n+length-1))
    for t in range(n):
        for lag in range(length):
            weights[:,t+lag] += kernels[:,lag]/n
        ####
    ####
    expected = np.zeros((6,6))
    for t in range(weights.shape[1]):
        expected += np.diag(weights[:,t])@q@np.diag(weights[:,t])
    ####
    np.testing.assert_allclose(filtered_white_mean_covariance(q,cfg,n),expected,atol=1e-15)
    assert np.linalg.eigvalsh(expected).min() >= -1e-12
####


def test_filtered_noise_is_not_an_iid_information_gain() -> None:
    cfg = AcquisitionConfig(gyroscope_filter=FirFilterConfig(taps=tuple([1/25]*25)))
    q = np.eye(6)
    n = 600
    correct = filtered_white_mean_covariance(q,cfg,n)[4,4]
    naive = sum(x*x for x in cfg.gyroscope_filter.taps)/n
    assert np.sqrt(correct/naive) > 4.9
    assert correct == pytest.approx(1/n, rel=.02)
####


@pytest.mark.parametrize("bad", [np.ones((3,3)), np.full((6,6),np.nan), -np.eye(6)])
def test_invalid_covariance_rejected(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        filtered_white_mean_covariance(bad, AcquisitionConfig(), 10)
    ####
####


def test_local_clock_epoch_keeps_elapsed_rate_error_separate_from_offset() -> None:
    cfg = ClockConfig(epoch_s=1000., offset_s=.25, frequency_error_ppm=200.)
    assert clock_tag(cfg,1000.,0) == 1000.25
    assert clock_tag(cfg,1010.,1) == pytest.approx(1010.252)
####


def test_constant_earth_rate_survives_unity_gain_filter_after_warmup() -> None:
    cfg = AcquisitionConfig(gyroscope_filter=FirFilterConfig(taps=tuple([.04]*25)))
    model = AcquiredImuModel(acquisition_config=cfg)
    omega = np.array([5.99e-5, -1.23e-5, -4.15e-5])
    for i in range(30):
        result = model.measure_ideal_rates(
            start_time=i*.01,end_time=(i+1)*.01,specific_force_body=np.array([0.,0.,9.81]),
            angular_rate_body=omega,ambient_temperature_celsius=25.,
        )
        if result.is_valid:
            np.testing.assert_allclose(result.angular_rate,omega,atol=1e-19,rtol=0.)
        ####
    ####
####


def test_checkpoint_schema_and_nonfinite_history_are_rejected_without_mutation() -> None:
    m = AcquiredImuModel()
    _step(m,0)
    before = m.snapshot()
    for updates in ({"schema_version":99},{"filter_history_newest_first":((float("nan"),)*6,)},{"previous_reported_time_s":9.}):
        with pytest.raises((ValueError,ValidationError)):
            m.restore(before.model_copy(update=updates))
        ####
        assert m.snapshot() == before
    ####
####


def test_acquisition_fields_have_explicit_forward_or_inverse_evidence() -> None:
    from playground.calibration.capability_audit import unclassified_acquisition_fields
    assert unclassified_acquisition_fields() == ()
####


@pytest.mark.parametrize("changes",[
    {"rates_si":(float("nan"),)*6}, {"valid_channels":(1,)*6}, {"temperature_celsius":float("inf")},
    {"end_time_s":0.}, {"sample_index":-1},
])
def test_sample_contract_rejects_malformed_payloads(changes: dict[str,object]) -> None:
    from dataclasses import replace
    sample = _step(AcquiredImuModel(),0)
    with pytest.raises(ValueError):
        replace(sample,**changes)
    ####
####


def test_empty_checkpoint_roundtrip_and_configuration_mismatch() -> None:
    m = AcquiredImuModel()
    before = m.snapshot()
    assert AcquiredImuModel.from_checkpoint(before).snapshot() == before
    other = AcquiredImuModel(acquisition_config=AcquisitionConfig(sample_period_s=.02))
    with pytest.raises(ValueError):
        other.restore(before)
    ####
    for update in ({"previous_physical_time_s":0.}, {"die_temperature_celsius":30.}):
        with pytest.raises(ValueError):
            m.restore(before.model_copy(update=update))
        ####
    ####
####


def test_active_checkpoint_rejects_missing_or_conflicting_sensor_boundaries() -> None:
    m = AcquiredImuModel()
    _step(m,0)
    before = m.snapshot()
    variants = [
        before.model_copy(update={"previous_physical_time_s":None}),
        before.model_copy(update={"sensor":before.sensor.model_copy(update={"previous_timestamp":.02})}),
    ]
    for variant in variants:
        with pytest.raises(ValueError):
            m.restore(variant)
        ####
        assert m.snapshot() == before
    ####
####


def test_thermal_configuration_failure_precedes_sensor_state_advance() -> None:
    m = AcquiredImuModel(ImuConfig(gyroscope=AxisConfig(thermal_noise_coefficient=-.1)))
    before = m.snapshot()
    with pytest.raises(ValueError):
        _step(m,0,temperature=55.)
    ####
    assert m.snapshot() == before
####


@pytest.mark.parametrize("dt,tau,temp",[(0.,1.,20.),(1.,-1.,20.),(1.,1.,float("nan"))])
def test_thermal_interval_rejects_invalid_inputs(dt: float,tau: float,temp: float) -> None:
    with pytest.raises(ValueError):
        thermal_interval(start_celsius=temp,ambient_celsius=50.,dt_s=dt,time_constant_s=tau)
    ####
####


def test_thermal_helpers_reject_invalid_envelopes_and_overflow() -> None:
    with pytest.raises(ValueError):
        thermal_interval(start_celsius=1e308,ambient_celsius=-1e308,dt_s=1.,time_constant_s=1.)
    ####
    for delta,tolerance,tau in [(float("nan"),.1,1.), (30.,0.,1.), (30.,.1,-1.)]:
        with pytest.raises(ValueError):
            thermal_settling_time_s(initial_difference_celsius=delta,tolerance_celsius=tolerance,time_constant_s=tau)
        ####
    ####
    for elapsed,tau in [(-1.,1.), (float("nan"),1.)]:
        with pytest.raises(ValueError):
            remaining_thermal_error_celsius(initial_difference_celsius=1.,elapsed_s=elapsed,time_constant_s=tau)
        ####
    ####
    assert remaining_thermal_error_celsius(initial_difference_celsius=30.,elapsed_s=0.,time_constant_s=0.) == 0.
####


def test_readout_statistics_and_restore_reject_malformed_inputs() -> None:
    cfg = AcquisitionConfig()
    with pytest.raises(ValueError):
        filtered_white_mean_covariance(np.eye(6),cfg,0)
    ####
    nonsymmetric = np.eye(6)
    nonsymmetric[0,1] = .1
    with pytest.raises(ValueError):
        filtered_white_mean_covariance(nonsymmetric,cfg,10)
    ####
    readout = FirReadout(cfg)
    for history,count in [((),-1),((),1)]:
        with pytest.raises(ValueError):
            readout.restore(history,count)
        ####
    ####
    for clock,time,index in [(ClockConfig(),float("nan"),0),(ClockConfig(),0.,-1),(ClockConfig(offset_s=1e308),1e308,0)]:
        with pytest.raises(ValueError):
            clock_tag(clock,time,index)
        ####
    ####
####
