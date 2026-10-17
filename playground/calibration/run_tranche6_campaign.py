"""Reproduce uncertain-equipment, held-out recovery, and confidence-floor studies.

Run from the source root with PYTHONPATH=src. All values are synthetic; observed
records and evaluator truth are written separately. No release/hardware claims.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from imu_error_model import AxisConfig, ImuConfig
from imu_error_model.acquisition import AcquiredImuModel
from imu_error_model.acquisition_config import AcquisitionConfig, ClockConfig

from .earth_reference import LocalEarthReference
from .gyrocompass import apply_calibration
from .known_reference import CalibrationDataset, fit_known_reference_calibration
from .reference_equipment import AngleLaw, FloatArray, RateTableReference, TableGeometry, aperture_reference
from .reference_identification import UnobservableCalibrationError, fit_encoder_timing, fit_lever_arm, propagate_shared_reference_uncertainty
from .reference_observations import EncoderCorrection, EncoderErrorConfig, ObservedCalibrationData, combine_observed_records, fit_encoder_law, fit_observed_calibration, reconstruct_reference, simulate_encoder, simulate_table_record


MOUNTS = ((0.,0.,0.),(np.pi,0.,0.),(0.,np.pi/2,0.),(0.,-np.pi/2,0.),(np.pi/2,0.,0.),(-np.pi/2,0.,0.))


def synthetic_sensor() -> ImuConfig:
    return ImuConfig(
        accelerometer=AxisConfig(fixed_bias=(.02,-.03,.04),sensitivity_matrix=((1.01,.003,-.002),(.001,.99,.004),(-.002,.003,1.02)),white_noise_density=2e-4),
        gyroscope=AxisConfig(fixed_bias=(2e-4,-3e-4,1e-4),sensitivity_matrix=((1.02,.004,-.003),(.002,.99,.001),(.003,-.002,1.01)),white_noise_density=3e-7),
        gyroscope_specific_force_sensitivity=((1e-4,2e-5,0.),(0.,-1e-4,2e-5),(3e-5,0.,1e-4)),
    )
####


def _rms(values: FloatArray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))
####


def _merge_legacy(records: list[CalibrationDataset]) -> CalibrationDataset:
    names=("start_times","end_times","temperatures_celsius","specific_force_body","angular_rate_body","measured_acceleration","measured_angular_rate")
    return CalibrationDataset(labels=tuple(label for record in records for label in record.labels),**{name:np.concatenate([getattr(record,name) for record in records]) for name in names})
####


def equipment_recovery(output: Path) -> dict[str, object]:
    earth=LocalEarthReference(np.deg2rad(34.7304))
    config=synthetic_sensor()
    fs,duration=50.,8.
    sensor=AcquiredImuModel(config,AcquisitionConfig(sample_period_s=1/fs),sensor_rng=np.random.default_rng(6101))
    observed=[]
    uncorrected=[]
    command_data=[]
    reprocess=[]
    truth_parameters=[]
    encoder_error=EncoderErrorConfig(angle_offset_rad=.01,scale_error_ppm=120.,clock_offset_s=.018,angle_noise_std_rad=5e-6,quantization_rad=1e-6,cyclic_error_rad=1e-5)
    # Declared independent metrology estimates deliberately differ from injection.
    correction=EncoderCorrection(angle_offset_rad=.01001,scale_error_ppm=130.,clock_offset_s=.01802)
    output.mkdir(parents=True,exist_ok=True)
    for i,mount in enumerate(MOUNTS):
        command=AngleLaw(rate_radps=.6+.1*i,sine_rad=.3,cosine_rad=.1,frequency_hz=.25)
        actual=command.model_copy(update={"rate_radps":command.rate_radps*1.004,"offset_rad":.007})
        true_geometry=TableGeometry(platform_from_sensor_rotvec_rad=mount,tilt_rotvec_ned_rad=(.01,-.02,.003),wobble_x_rad=.003,wobble_y_rad=.004,lever_arm_sensor_m=(.03,-.02,.01))
        surveyed=TableGeometry(platform_from_sensor_rotvec_rad=mount,tilt_rotvec_ned_rad=(.01001,-.02001,.003005),wobble_x_rad=.003003,wobble_y_rad=.004004,lever_arm_sensor_m=(.03002,-.02001,.01001))
        nominal=TableGeometry(platform_from_sensor_rotvec_rad=mount)
        edges=(i*round(fs*duration)+np.arange(round(fs*duration)+1))/fs
        provider=RateTableReference(earth,actual,true_geometry)
        record,truth=simulate_table_record(sensor,provider,edges[:-1],edges[1:],reset=i==0,source_id=f"setup-{i}")
        encoder=simulate_encoder(actual,np.linspace(edges[0]-.1,edges[-1]+.1,421),encoder_error,rng=np.random.default_rng(6200+i),source_id=f"encoder-{i}")
        fitted=fit_encoder_law(encoder,frequency_hz=.25,correction=correction)
        uncalibrated=fit_encoder_law(encoder,frequency_hz=.25)
        observed.append(reconstruct_reference(record,fitted,surveyed_geometry=surveyed,earth=earth,survey_source_id="synthetic-external-survey-estimate"))
        uncorrected.append(reconstruct_reference(record,uncalibrated,surveyed_geometry=nominal,earth=earth,survey_source_id="nominal-geometry-assumption"))
        nominal_ref=aperture_reference(RateTableReference(earth,command,nominal),edges[:-1],edges[1:])
        command_data.append(replace(observed[-1].dataset,specific_force_body=nominal_ref.force,angular_rate_body=nominal_ref.rate))
        reprocess.append((record,encoder,surveyed))
        # Consumer files never contain injected parameters or ideal sensor values.
        np.savetxt(output/f"imu-observed-{i}.csv",np.column_stack((record.starts_s,record.ends_s,record.rates_si)),delimiter=",",header="reported_start_s,reported_end_s,acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z",comments="")
        np.savetxt(output/f"encoder-observed-{i}.csv",np.column_stack((encoder.reported_times_s,encoder.angles_rad)),delimiter=",",header="reported_time_s,unwrapped_angle_rad",comments="")
        np.savetxt(output/f"evaluator-truth-{i}.csv",np.column_stack((truth.starts_s,truth.ends_s,truth.force,truth.rate)),delimiter=",",header="physical_start_s,physical_end_s,force_x,force_y,force_z,rate_x,rate_y,rate_z",comments="")
        truth_parameters.append({"command":command.model_dump(),"actual_law":actual.model_dump(),"actual_geometry":true_geometry.model_dump(),"survey_estimate":surveyed.model_dump()})
    ####
    fits={"nominal_commands":fit_known_reference_calibration(_merge_legacy(command_data)),"encoder_unsurveyed":fit_observed_calibration(combine_observed_records(tuple(uncorrected))),"encoder_and_survey":fit_observed_calibration(combine_observed_records(tuple(observed)))}
    heldout_force=[]
    heldout_rate=[]
    heldout_measurements=[]
    for i,mount in enumerate(MOUNTS[::2]):
        geometry=TableGeometry(platform_from_sensor_rotvec_rad=mount,base_rotvec_ned_rad=(.2,-.1,.4),tilt_rotvec_ned_rad=(.01,-.02,.003),wobble_x_rad=.003,wobble_y_rad=.004,lever_arm_sensor_m=(.03,-.02,.01))
        provider=RateTableReference(earth,AngleLaw(rate_radps=1.8+.1*i,sine_rad=.4,frequency_hz=.37),geometry)
        edges=np.arange(201)/fs
        record,truth=simulate_table_record(AcquiredImuModel(config,AcquisitionConfig(sample_period_s=1/fs),sensor_rng=np.random.default_rng(6300+i)),provider,edges[:-1],edges[1:])
        heldout_force.append(truth.force)
        heldout_rate.append(truth.rate)
        heldout_measurements.append(record.rates_si)
    ####
    f,w,y=np.concatenate(heldout_force),np.concatenate(heldout_rate),np.concatenate(heldout_measurements)
    metrics={}
    for name,fit in fits.items():
        recovered_f,recovered_w=apply_calibration(y[:,:3],y[:,3:],fit)
        metrics[name]={"heldout_force_rms_mps2":_rms(recovered_f-f),"heldout_rate_rms_radps":_rms(recovered_w-w),"training_accel_residual_rms_mps2":fit.accelerometer_residual_rms,"training_gyro_residual_rms_radps":fit.gyroscope_residual_rms,"accel_rank":fit.accelerometer_design_rank,"gyro_rank":fit.gyroscope_design_rank}
    ####

    def reference_builder(p: FloatArray) -> ObservedCalibrationData:
        parts=[]
        active=correction.model_copy(update={"scale_error_ppm":float(p[0]),"clock_offset_s":float(p[1])})
        for record,encoder,survey in reprocess:
            fit=fit_encoder_law(encoder,frequency_hz=.25,correction=active)
            parts.append(reconstruct_reference(record,fit,surveyed_geometry=survey,earth=earth,survey_source_id="synthetic-external-survey-estimate"))
        ####
        return combine_observed_records(tuple(parts))
    ####

    uncertainty=propagate_shared_reference_uncertainty(reference_builder,np.array([130.,.01802]),np.diag([500.**2,.0002**2]),finite_difference_steps=np.array([.2,1e-5]),parameter_names=("encoder_scale_error_ppm","encoder_clock_offset_s"))
    np.savez_compressed(output/"calibration-reference-covariance.npz",conditional=uncertainty.conditional_sensor_covariance,shared=uncertainty.shared_reference_covariance,total=uncertainty.calibration.joint_coefficient_covariance,jacobian=uncertainty.coefficient_reference_jacobian)
    (output/"evaluator-parameters.json").write_text(json.dumps({"sensor":config.model_dump(),"encoder_errors":encoder_error.model_dump(),"encoder_metrology_estimates":correction.model_dump(),"setups":truth_parameters},indent=2)+"\n")
    return {"training_samples":sum(len(r.dataset.labels) for r in observed),"training_sampled_duration_s":6*duration,"setup_handling_time_s":"not_modelled", "heldout_samples":len(y),"strategies":metrics,"gyro_xx_conditional_sigma_ppm":float(np.sqrt(uncertainty.conditional_sensor_covariance[12,12])*1e6),"gyro_xx_shared_reference_sigma_ppm":float(np.sqrt(uncertainty.shared_reference_covariance[12,12])*1e6),"uncertainty_assumptions":uncertainty.calibration.covariance_assumption}
####


def lever_recovery() -> dict[str, object]:
    earth=LocalEarthReference(.6)
    r=np.array([.04,-.025,.015])
    bias=np.array([.003,-.002,.001])
    f=[]
    lever_operators=[]
    measured=[]
    rng=np.random.default_rng(6401)
    for mount in (MOUNTS[0],MOUNTS[2],MOUNTS[4]):
        edges=np.arange(161,dtype=float)*.025
        ref=aperture_reference(RateTableReference(earth,AngleLaw(rate_radps=1.,sine_rad=.4),TableGeometry(platform_from_sensor_rotvec_rad=mount)),edges[:-1],edges[1:])
        f.append(ref.force)
        lever_operators.append(ref.lever_operator)
        measured.append(ref.force+ref.lever_operator@r+bias+rng.normal(0,1e-4,ref.force.shape))
    ####
    fit=fit_lever_arm(np.concatenate(measured),np.concatenate(f),np.concatenate(lever_operators))
    single=aperture_reference(RateTableReference(earth,AngleLaw(rate_radps=1.,sine_rad=.4),TableGeometry(),False),edges[:-1],edges[1:])
    rank=None
    try:
        fit_lever_arm(single.force,single.force,single.lever_operator,fit_residual_bias=False)
    except UnobservableCalibrationError as error:
        rank=error.rank
    ####
    return {"condition":"calibrated_accelerometer_and_surveyed_reference_control","injected_lever_m":r.tolist(),"estimated_lever_m":fit.lever_arm_sensor_m.tolist(),"maximum_lever_error_m":float(np.max(np.abs(fit.lever_arm_sensor_m-r))),"residual_rms_mps2":fit.residual_rms_mps2,"single_axis_rank_of_3":rank}
####


def timing_recovery() -> dict[str, object]:
    earth=LocalEarthReference(.6)
    law=AngleLaw(rate_radps=.7,sine_rad=.3,cosine_rad=.1,frequency_hz=.25)
    fs=50.
    config=ImuConfig(gyroscope=AxisConfig(scale_factor=.012,fixed_bias=(0.,0.,.002),white_noise_density=1e-5))
    acq=AcquisitionConfig(sample_period_s=1/fs,clock=ClockConfig(offset_s=.023))
    edges=np.arange(401)/fs
    record,_=simulate_table_record(AcquiredImuModel(config,acq,sensor_rng=np.random.default_rng(6501)),RateTableReference(earth,law,TableGeometry()),edges[:-1],edges[1:])
    encoder=simulate_encoder(law,np.linspace(-.2,8.2,841),EncoderErrorConfig(angle_noise_std_rad=1e-6),rng=np.random.default_rng(6502))
    observed_law=fit_encoder_law(encoder,frequency_hz=.25).law
    fit=fit_encoder_timing(observed_law,record.starts_s,record.ends_s,record.rates_si[:,5])
    noise_free=fit_encoder_timing(law,record.starts_s,record.ends_s,1.012*law.mean_rate(record.starts_s-.023,record.ends_s-.023)+.002)
    return {"injected_reported_clock_offset_s":.023,"estimated_relative_delay_s":fit.delay_s,"relative_delay_sigma_s":float(np.sqrt(fit.covariance_gain_bias_delay[2,2])),"estimated_gain":fit.gain,"gyro_bias_after_removing_fitted_gain_times_known_earth_vertical_radps":fit.bias_radps-fit.gain*float(earth.earth_rotation_ned_radps[2]),"noiseless_delay_error_s":noise_free.delay_s-.023,"constant_rate_delay_observable":False,"frequency_is_declared_protocol":True,"phase_and_amplitude_are_encoder_fitted":True}
####


def common_reference_confidence() -> list[dict[str, object]]:
    """Independent Gaussian sufficient-statistic MC, not the fitting implementation.

    For a balanced sinusoidal design, slope noise has sigma_y/sqrt(N/2).
    One unknown encoder gain is drawn per trial and held for the entire record.
    No claim is made about general EIV regression or adaptive stopping.
    """
    rng=np.random.default_rng(6601)
    trials=4000
    rows=[]
    for n in (200,800,3200):
        shared=rng.normal(0,500e-6,trials)
        sensor_sigma=.002/np.sqrt(n/2)
        slope=(1+rng.normal(0,sensor_sigma,trials))/(1+shared)
        conditional=sensor_sigma/(1+shared)
        reference_std=np.abs(slope)*500e-6/(1+shared)
        total=np.sqrt(conditional**2+reference_std**2)
        rows.append({"samples":n,"sampled_duration_s":n/20,"trials":trials,"reference_sigma_ppm":500.,"noise_only_halfwidth_95_ppm":float(np.mean(1.959963984540054*conditional)*1e6),"including_reference_halfwidth_95_ppm":float(np.mean(1.959963984540054*total)*1e6),"noise_only_coverage":float(np.mean(np.abs(slope-1)<=1.959963984540054*conditional)),"including_reference_coverage":float(np.mean(np.abs(slope-1)<=1.959963984540054*total)),"absolute_error_p95_ppm":float(np.quantile(np.abs(slope-1),.95)*1e6)})
    ####
    return rows
####


def quadrature_convergence() -> dict[str, object]:
    p=RateTableReference(LocalEarthReference(.6),AngleLaw(rate_radps=2.,sine_rad=.5,frequency_hz=.6),TableGeometry(wobble_x_rad=.04,wobble_y_rad=.03,wobble_order=3,lever_arm_sensor_m=(.08,-.03,.02)))
    starts,ends=np.array([0.,.4]),np.array([.4,.8])
    reference=aperture_reference(p,starts,ends,order=32)
    rows=[]
    for order in (2,4,8,16):
        result=aperture_reference(p,starts,ends,order=order)
        rows.append({"order":order,"maximum_force_difference_mps2":float(np.max(np.abs(result.force-reference.force))),"maximum_rate_difference_radps":float(np.max(np.abs(result.rate-reference.rate)))})
    ####
    return {"aperture_s":.4,"comparison_order":32,"rows":rows,"interpretation":"numerical_convergence_not_general_exactness"}
####


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("artifacts/tranche6"))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    results={"scope":"synthetic verification only", "equipment_recovery":equipment_recovery(args.output/"records"),"lever_recovery":lever_recovery(),"encoder_timing_recovery":timing_recovery(),"common_reference_confidence":common_reference_confidence(),"quadrature_convergence":quadrature_convergence()}
    (args.output/"results.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")
    print(json.dumps(results,indent=2,allow_nan=False))
    return 0
####


if __name__=="__main__":
    raise SystemExit(main())
####
