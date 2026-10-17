from dataclasses import replace

import numpy as np
import pytest

from imu_error_model import ImuModel
from playground.calibration.earth_reference import LocalEarthReference
from playground.calibration.known_reference import ferraris_style_reference_program, simulate_reference_program
from playground.calibration.reference_equipment import AngleLaw, RateTableReference, TableGeometry, aperture_reference
from playground.calibration.reference_identification import UnobservableCalibrationError, fit_encoder_timing, fit_lever_arm, propagate_shared_reference_uncertainty
from playground.calibration.reference_observations import ObservedCalibrationData


def test_variable_rate_recovers_gain_bias_and_encoder_delay() -> None:
    law=AngleLaw(rate_radps=.7,sine_rad=.3,cosine_rad=.1,frequency_hz=.25)
    edges=np.arange(401,dtype=np.float64)*.02
    starts,ends=edges[:-1],edges[1:]
    y=1.012*law.mean_rate(starts-.023,ends-.023)+.002
    fit=fit_encoder_timing(law,starts,ends,y,delay_bounds_s=(-.08,.08))
    assert fit.delay_s==pytest.approx(.023,abs=2e-10)
    assert fit.gain==pytest.approx(1.012,abs=1e-10)
    assert fit.bias_radps==pytest.approx(.002,abs=1e-10)
    assert fit.residual_rms_radps<1e-10
####


@pytest.mark.parametrize("samples", (50,500,2000))
def test_constant_rate_cannot_identify_delay_or_separate_gain_and_bias(samples: int) -> None:
    law=AngleLaw(rate_radps=.7)
    edges=np.linspace(0,10,samples+1,dtype=np.float64)
    with pytest.raises(UnobservableCalibrationError):
        fit_encoder_timing(law,edges[:-1],edges[1:],np.full(samples,.7))
    ####
####


def test_noisy_timing_fit_has_reasonable_covariance() -> None:
    law=AngleLaw(rate_radps=.7,sine_rad=.3,frequency_hz=.25)
    edges=np.arange(201,dtype=np.float64)*.04
    rng=np.random.default_rng(717)
    errors=[]
    sigmas=[]
    for _ in range(40):
        y=1.01*law.mean_rate(edges[:-1]-.025,edges[1:]-.025)+.002+rng.normal(0,.001,200)
        fit=fit_encoder_timing(law,edges[:-1],edges[1:],y)
        errors.append(fit.delay_s-.025)
        sigmas.append(np.sqrt(fit.covariance_gain_bias_delay[2,2]))
    ####
    assert np.std(errors)/np.mean(sigmas)==pytest.approx(1,rel=.35)
    assert np.mean(np.abs(errors)<1.96*np.array(sigmas))>=.85
####


def test_delay_outside_search_range_does_not_return_false_precision() -> None:
    law=AngleLaw(sine_rad=.3)
    edges=np.arange(201,dtype=np.float64)*.04
    y=law.mean_rate(edges[:-1]-.3,edges[1:]-.3)
    with pytest.raises(ValueError,match="boundary"):
        fit_encoder_timing(law,edges[:-1],edges[1:],y,delay_bounds_s=(-.05,.05))
    ####
####


def _lever_data() -> tuple[np.ndarray, np.ndarray]:
    earth=LocalEarthReference(.6)
    edges=np.arange(81,dtype=np.float64)*.05
    force=[]
    lever=[]
    for mount in ((0.,0.,0.),(np.pi/2,0.,0.),(0.,np.pi/2,0.)):
        ref=aperture_reference(RateTableReference(earth,AngleLaw(rate_radps=1.,sine_rad=.4),TableGeometry(platform_from_sensor_rotvec_rad=mount)),edges[:-1],edges[1:])
        force.append(ref.force)
        lever.append(ref.lever_operator)
    ####
    return np.concatenate(force),np.concatenate(lever)
####


@pytest.mark.parametrize("fit_bias", (True, False))
def test_multiaxis_variable_rate_recovers_lever_arm(fit_bias: bool) -> None:
    force, lever=_lever_data()
    r=np.array([.04,-.025,.015])
    b=np.array([.003,-.002,.001]) if fit_bias else np.zeros(3)
    measured=force+lever@r+b
    fit=fit_lever_arm(measured,force,lever,fit_residual_bias=fit_bias)
    np.testing.assert_allclose(fit.lever_arm_sensor_m,r,atol=1e-13)
    np.testing.assert_allclose(fit.residual_bias_mps2,b,atol=1e-13)
####


def test_single_axis_spin_does_not_identify_axial_lever() -> None:
    edges=np.arange(101,dtype=np.float64)*.02
    ref=aperture_reference(RateTableReference(LocalEarthReference(.6),AngleLaw(rate_radps=1.,sine_rad=.4),TableGeometry(),False),edges[:-1],edges[1:])
    with pytest.raises(UnobservableCalibrationError) as error:
        fit_lever_arm(ref.force,ref.force,ref.lever_operator,fit_residual_bias=False)
    ####
    assert error.value.rank==2
####


def test_common_reference_scale_uncertainty_survives_more_data() -> None:
    dataset=simulate_reference_program(ImuModel(),tuple(replace(s, sample_count=10) for s in ferraris_style_reference_program()))
    values=[]
    for copies in (1,10):
        def builder(p: np.ndarray) -> ObservedCalibrationData:
            # One external reference-rate scale estimate applied to all rows.
            arrays={name:np.tile(getattr(dataset,name),(copies,1)) for name in ("specific_force_body","angular_rate_body","measured_acceleration","measured_angular_rate")}
            arrays["angular_rate_body"]*=1+p[0]*1e-6
            data=replace(dataset,labels=dataset.labels*copies,start_times=np.tile(dataset.start_times,copies),end_times=np.tile(dataset.end_times,copies),temperatures_celsius=np.tile(dataset.temperatures_celsius,copies),**arrays)
            return ObservedCalibrationData(data,("observed-rate-reference","external-scale-certificate"))
        ####
        result=propagate_shared_reference_uncertainty(builder,np.array([0.]),np.array([[500.**2]]),finite_difference_steps=np.array([.2]),parameter_names=("reference_scale_ppm",))
        values.append(result.shared_reference_covariance[12,12])
        assert result.coefficient_reference_jacobian[12,0]==pytest.approx(-1e-6,rel=1e-6)
        assert result.calibration.joint_coefficient_covariance[12,12]>=result.shared_reference_covariance[12,12]
    ####
    np.testing.assert_allclose(values,[2.5e-7,2.5e-7],rtol=1e-6)
####


def test_reference_covariance_validation() -> None:
    def unused(p: np.ndarray) -> ObservedCalibrationData:
        raise AssertionError("must validate before invoking builder")
    ####
    with pytest.raises(ValueError,match="PSD"):
        propagate_shared_reference_uncertainty(unused,np.array([0.]),np.array([[-1.]]),finite_difference_steps=np.array([.1]),parameter_names=("bad",))
    ####
####


def test_table_scale_and_gyro_scale_have_an_exact_gauge_without_independent_reference() -> None:
    t=np.linspace(0,6,100)
    rate=np.sin(t)
    # y = sensor_gain * table_gain * commanded_rate. Two identical derivatives.
    jacobian=np.column_stack((rate,rate))
    assert np.linalg.matrix_rank(jacobian)==1
    a=1.02*1.003*rate
    b=(1.02*2)*(1.003/2)*rate
    np.testing.assert_array_equal(a,b)
####


def test_full_intrinsic_matrices_cannot_determine_mount_rotation_without_anchor() -> None:
    from playground.calibration.reference_equipment import rotation_vector
    rng=np.random.default_rng(90)
    f=rng.normal(size=(50,3))
    w=rng.normal(size=(50,3))
    ma=np.array([[1.01,.03,-.01],[.01,.99,.04],[-.02,0,1.02]])
    mg=np.array([[.99,.02,0],[.01,1.02,-.01],[.03,0,1.01]])
    force_sensitivity=.01*np.eye(3)
    q=rotation_vector((.02,-.03,.04))
    np.testing.assert_allclose(f@ma.T,(f@q)@(ma@q).T,atol=1e-14)
    np.testing.assert_allclose(w@mg.T+f@force_sensitivity.T,(w@q)@(mg@q).T+(f@q)@(force_sensitivity@q).T,atol=1e-14)
####


def test_reference_uncertainty_rejects_regenerating_measurements() -> None:
    data=simulate_reference_program(ImuModel(),ferraris_style_reference_program())
    def bad_builder(p: np.ndarray) -> ObservedCalibrationData:
        changed=replace(data,measured_angular_rate=data.measured_angular_rate+p[0])
        return ObservedCalibrationData(changed,("external-reference",))
    ####
    with pytest.raises(ValueError,match="same measured recording"):
        propagate_shared_reference_uncertainty(bad_builder,np.array([0.]),np.array([[1.]]),finite_difference_steps=np.array([.01]),parameter_names=("metrology",))
    ####
####
