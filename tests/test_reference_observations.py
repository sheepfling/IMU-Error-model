from dataclasses import fields

import numpy as np
import pytest

from imu_error_model import AxisConfig, ImuConfig
from imu_error_model.acquisition import AcquiredImuModel
from imu_error_model.acquisition_config import AcquisitionConfig, FirFilterConfig
from playground.calibration.earth_reference import LocalEarthReference
from playground.calibration.known_reference import calibration_parameter_error
from playground.calibration.reference_equipment import AngleLaw, RateTableReference, TableGeometry, aperture_reference
from playground.calibration.reference_observations import EncoderCorrection, EncoderErrorConfig, EncoderObservations, _validated_valid_channels, combine_observed_records, fit_encoder_law, fit_observed_calibration, reconstruct_reference, simulate_encoder, simulate_table_record


def _law() -> AngleLaw:
    return AngleLaw(offset_rad=.3, rate_radps=.7, sine_rad=.22, cosine_rad=.1, frequency_hz=.25)
####


def test_encoder_reports_only_observed_values_and_preserves_systematic_error() -> None:
    t = np.linspace(0, 8, 321)
    law = _law()
    e = EncoderErrorConfig(angle_offset_rad=.01,scale_error_ppm=1000,clock_offset_s=.03,clock_frequency_error_ppm=100)
    obs = simulate_encoder(law,t,e,rng=np.random.default_rng(0))
    assert {f.name for f in fields(obs)} == {"reported_times_s", "angles_rad", "source_id"}
    fit = fit_encoder_law(obs,frequency_hz=law.frequency_hz,correction=EncoderCorrection(angle_offset_rad=.01,scale_error_ppm=1000,clock_offset_s=.03,clock_frequency_error_ppm=100))
    np.testing.assert_allclose(fit.law.evaluate(t)[0],law.evaluate(t)[0],atol=1e-13)
    assert fit.residual_rms_rad < 1e-13
    assert not obs.angles_rad.flags.writeable
    assert np.max(np.abs(obs.angles_rad-law.evaluate(t)[0])) > .01
####


def test_encoder_noise_quantization_and_cyclic_error_are_seeded() -> None:
    t = np.linspace(0,8,321)
    e = EncoderErrorConfig(angle_noise_std_rad=1e-4,cyclic_error_rad=.001,quantization_rad=1e-4,timestamp_jitter_bound_s=1e-5)
    a=simulate_encoder(_law(),t,e,rng=np.random.default_rng(3))
    b=simulate_encoder(_law(),t,e,rng=np.random.default_rng(3))
    np.testing.assert_array_equal(a.angles_rad,b.angles_rad)
    np.testing.assert_allclose(a.angles_rad/1e-4,np.rint(a.angles_rad/1e-4),atol=1e-10)
    with pytest.raises(ValueError,match="residual"):
        fit_encoder_law(a,frequency_hz=.25,max_residual_rad=1e-5)
    ####
####


@pytest.mark.parametrize("times", ([0,1,1,2,3],[0,2,1,3,4],[0,1,2,3,float("nan")],[0,1,2]))
def test_encoder_rejects_bad_timing(times: list[float]) -> None:
    with pytest.raises(ValueError):
        EncoderObservations(np.array(times),np.zeros(len(times)),"encoder")
    ####
####


def test_excessive_encoder_tag_jitter_is_not_hidden_by_sorting() -> None:
    with pytest.raises(ValueError, match="increasing"):
        simulate_encoder(_law(),np.linspace(0,1,100),EncoderErrorConfig(timestamp_jitter_bound_s=.1),rng=np.random.default_rng(1))
    ####
####


def test_encoder_fit_rejects_short_uninformative_time_window() -> None:
    obs=EncoderObservations(np.linspace(0,1e-6,10),np.zeros(10),"encoder")
    with pytest.raises(ValueError,match="conditioned|deficient"):
        fit_encoder_law(obs,frequency_hz=.25)
    ####
####


def test_table_record_keeps_truth_separate_and_honors_acquisition_startup() -> None:
    model=AcquiredImuModel(acquisition_config=AcquisitionConfig(sample_period_s=.02,accelerometer_filter=FirFilterConfig(taps=(.5,.5))))
    edges=np.arange(201,dtype=np.float64)*.02
    p=RateTableReference(LocalEarthReference(.6),_law(),TableGeometry())
    record,truth=simulate_table_record(model,p,edges[:-1],edges[1:])
    assert {f.name for f in fields(record)} == {"starts_s","ends_s","rates_si","valid_channels","temperatures_celsius","source_id"}
    assert not record.valid_channels[0,:3].any()
    np.testing.assert_allclose(record.rates_si[1:,:3],.5*(truth.force[1:]+truth.force[:-1]),atol=1e-14)
    encoder=simulate_encoder(_law(),np.linspace(0,4,401),EncoderErrorConfig(),rng=np.random.default_rng(2))
    fit=fit_encoder_law(encoder,frequency_hz=.25)
    data=reconstruct_reference(record,fit,surveyed_geometry=TableGeometry(),earth=p.earth,survey_source_id="survey:unit-test")
    assert len(data.dataset.labels)==199
    assert "truth" not in str(fields(data))
####


def test_truth_cannot_be_passed_as_an_observed_fit() -> None:
    p=RateTableReference(LocalEarthReference(.6),AngleLaw(),TableGeometry())
    truth=aperture_reference(p,np.array([0.],dtype=np.float64),np.array([.1],dtype=np.float64))
    with pytest.raises(TypeError):
        fit_observed_calibration(truth)  # type: ignore[arg-type]
    ####
    with pytest.raises(TypeError):
        combine_observed_records((truth,))  # type: ignore[arg-type]
    ####
####


def test_reference_reconstruction_refuses_encoder_extrapolation() -> None:
    model=AcquiredImuModel(acquisition_config=AcquisitionConfig(sample_period_s=.02))
    edges=np.arange(101,dtype=np.float64)*.02
    p=RateTableReference(LocalEarthReference(.6),_law(),TableGeometry())
    record,_=simulate_table_record(model,p,edges[:-1],edges[1:])
    encoder=simulate_encoder(_law(),np.linspace(.1,1.9,101),EncoderErrorConfig(),rng=np.random.default_rng(2))
    with pytest.raises(ValueError,match="extrapolation"):
        reconstruct_reference(record,fit_encoder_law(encoder,frequency_hz=.25),surveyed_geometry=TableGeometry(),earth=p.earth,survey_source_id="survey")
    ####
####


def test_bad_sensor_record_is_rejected() -> None:
    with pytest.raises(ValueError,match="boolean"):
        _validated_valid_channels(np.ones((1,6)), 1)
    ####
####


def test_observed_encoder_and_survey_recover_full_intrinsics_without_truth_argument() -> None:
    config=ImuConfig(accelerometer=AxisConfig(fixed_bias=(.02,-.03,.04),sensitivity_matrix=((1.01,.003,-.002),(.001,.99,.004),(-.002,.003,1.02))),gyroscope=AxisConfig(fixed_bias=(.0002,-.0003,.0001),sensitivity_matrix=((1.02,.004,-.003),(.002,.99,.001),(.003,-.002,1.01))),gyroscope_specific_force_sensitivity=((1e-4,2e-5,0.),(0.,-1e-4,2e-5),(3e-5,0.,1e-4)))
    earth=LocalEarthReference(.6)
    records=[]
    mounts=((0.,0.,0.),(np.pi,0.,0.),(0.,np.pi/2,0.),(0.,-np.pi/2,0.),(np.pi/2,0.,0.),(-np.pi/2,0.,0.))
    for i,mount in enumerate(mounts):
        law=AngleLaw(rate_radps=.6+.1*i,sine_rad=.3,frequency_hz=.25)
        survey=TableGeometry(platform_from_sensor_rotvec_rad=mount,tilt_rotvec_ned_rad=(.01,-.02,.003),wobble_x_rad=.003,wobble_y_rad=.004,lever_arm_sensor_m=(.03,-.02,.01))
        provider=RateTableReference(earth,law,survey)
        edges=np.arange(201,dtype=np.float64)*.04
        record,_evaluator_only=simulate_table_record(AcquiredImuModel(config,AcquisitionConfig(sample_period_s=.04)),provider,edges[:-1],edges[1:],source_id=f"pose-{i}")
        encoder=simulate_encoder(law,np.linspace(0,8,401),EncoderErrorConfig(),rng=np.random.default_rng(i))
        fit=fit_encoder_law(encoder,frequency_hz=.25)
        records.append(reconstruct_reference(record,fit,surveyed_geometry=survey,earth=earth,survey_source_id="perfect-survey-control"))
    ####
    estimate=fit_observed_calibration(combine_observed_records(tuple(records)))
    assert np.max(np.abs(calibration_parameter_error(estimate,config)))<1e-10
    assert "reference_error_not_included" in estimate.covariance_assumption
####


def test_segmented_acquisition_preserves_one_run_realization() -> None:
    config=ImuConfig(gyroscope=AxisConfig(turn_on_bias_std=.003,white_noise_density=.0001))
    acquisition=AcquisitionConfig(sample_period_s=.02)
    p=RateTableReference(LocalEarthReference(.6),_law(),TableGeometry())
    edges=np.arange(101,dtype=np.float64)*.02
    full,_=simulate_table_record(AcquiredImuModel(config,acquisition,sensor_rng=np.random.default_rng(9)),p,edges[:-1],edges[1:])
    split_model=AcquiredImuModel(config,acquisition,sensor_rng=np.random.default_rng(9))
    a,_=simulate_table_record(split_model,p,edges[:50],edges[1:51])
    b,_=simulate_table_record(split_model,p,edges[50:-1],edges[51:],reset=False)
    np.testing.assert_array_equal(full.rates_si,np.concatenate((a.rates_si,b.rates_si)))
####


@pytest.mark.parametrize("kwargs", ({"scale_error_ppm":-1e6},{"angle_noise_std_rad":-1.},{"clock_offset_s":float("nan")},{"cyclic_order":True},{"quantization_rad":-1.}))
def test_encoder_config_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        EncoderErrorConfig.model_validate(kwargs)
    ####
####


def test_sensor_runner_rejects_physical_gaps_before_resetting() -> None:
    model=AcquiredImuModel(acquisition_config=AcquisitionConfig(sample_period_s=.02))
    checkpoint=model.snapshot()
    with pytest.raises(ValueError,match="contiguous"):
        simulate_table_record(
            model,
            RateTableReference(LocalEarthReference(.6),_law(),TableGeometry()),
            np.array([0.,.04],dtype=np.float64),
            np.array([.02,.06],dtype=np.float64),
        )
    ####
    assert model.snapshot()==checkpoint
####
