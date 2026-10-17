"""Reproduce readout identification, Earth-rate phase, thermal and confidence checks."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from math import degrees, pi, radians, sqrt
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from imu_error_model import (
    AcquisitionConfig, AcquiredImuModel, AxisConfig, ClockConfig, FirFilterConfig,
    ImuConfig, ThermalLagConfig, effective_fir_kernels, filtered_white_mean_covariance,
    thermal_interval, thermal_settling_time_s,
)
from .acquisition_confidence import plan_filtered_heading
from .earth_reference import LocalEarthReference
from .gyrocompass import static_gyrocompass, wrap_angle_rad
from .readout_identification import fit_readout_kernel, predict_readout
from .rotating_frame import average_sensor_from_base_rotation, rotation_z

FloatArray = NDArray[np.float64]


def simulate_readout_record(
        reference: FloatArray, config: AcquisitionConfig, *, seed: int = 91,
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    """Inject sample-level known rates; this is not a bandwidth-limited motion rig."""
    model = AcquiredImuModel(acquisition_config=config, sensor_rng=np.random.default_rng(seed))
    measured = np.zeros_like(reference)
    tags = np.zeros((len(reference),2))
    valid = np.zeros((len(reference),6),dtype=bool)
    dt = config.sample_period_s
    for i, row in enumerate(reference):
        sample = model.measure_ideal_rates(
            start_time=i*dt, end_time=(i+1)*dt, specific_force_body=row[:3],
            angular_rate_body=row[3:], ambient_temperature_celsius=25.,
        )
        measured[i] = sample.rates_si
        tags[i] = (sample.start_time_s,sample.end_time_s)
        valid[i] = sample.valid_channels
    ####
    return measured, tags, valid
####


def carousel_phase_experiment(
        config: AcquisitionConfig, estimated_kernels: FloatArray,
        *, estimated_clock_offset_s: float, revolutions: int = 1,
        rotation_rate_degps: float = 6., heading_deg: float = -28.,
) -> dict[str, float]:
    """Show phase masquerading as heading, then correct with a separately fitted FIR.

    Constant table rate, level body, unity deterministic sensor calibration,
    equal horizontal gyro filters, and a separately known clock offset only.
    Not a general inverse filter or a joint clock/intrinsic/heading estimator.
    """
    if config.clock.frequency_error_ppm != 0. or config.clock.timestamp_jitter_bound_s != 0.:
        raise ValueError("phase experiment currently requires offset-only clock error")
    ####
    if config.gyroscope_delay_samples[0] != config.gyroscope_delay_samples[1]:
        raise ValueError("phase experiment requires matched horizontal channel delays")
    ####
    if rotation_rate_degps <= 0.0 or not np.isfinite(rotation_rate_degps):
        raise ValueError("rotation rate must be positive and finite")
    ####
    if isinstance(revolutions,bool) or not isinstance(revolutions,int) or revolutions < 1:
        raise ValueError("revolutions must be a positive integer")
    ####
    if not np.isfinite(estimated_clock_offset_s):
        raise ValueError("clock offset estimate must be finite")
    ####
    dt = config.sample_period_s
    omega = radians(rotation_rate_degps)
    count = round(revolutions*2*pi/omega/dt)
    if not np.isclose(count*dt*omega,revolutions*2*pi,atol=1e-10,rtol=0.):
        raise ValueError("retained duration must contain complete revolutions")
    ####
    warmup = effective_fir_kernels(config).shape[1]-1
    earth = LocalEarthReference(latitude_rad=radians(34.7304))
    nav_from_base = rotation_z(radians(heading_deg))
    gravity = nav_from_base.T@earth.gravity_ned_mps2
    earth_rate = nav_from_base.T@earth.earth_rotation_ned_radps
    table = np.array([0.,0.,omega])
    model = AcquiredImuModel(acquisition_config=config, sensor_rng=np.random.default_rng(31))
    accel_sum = np.zeros(3)
    gyro_sum = np.zeros(3)
    for i in range(count+warmup):
        start, end = i*dt,(i+1)*dt
        aperture = average_sensor_from_base_rotation(omega*start,omega*end)
        sample = model.measure_ideal_rates(
            start_time=start, end_time=end, specific_force_body=aperture@gravity,
            angular_rate_body=aperture@earth_rate+table, ambient_temperature_celsius=25.,
        )
        if not sample.is_valid:
            continue
        ####
        reported_aperture = average_sensor_from_base_rotation(omega*sample.start_time_s,omega*sample.end_time_s)
        accel_sum += np.linalg.solve(reported_aperture,sample.acceleration)
        gyro_sum += np.linalg.solve(reported_aperture,sample.angular_rate-table)
    ####
    avg_accel, avg_gyro = accel_sum/count,gyro_sum/count
    uncorrected = static_gyrocompass(avg_accel,avg_gyro,latitude_rad=earth.latitude_rad)
    # The negative-frequency sensor-frame sinusoid has H(-spin_frequency).
    kernels = np.asarray(estimated_kernels,dtype=float)
    if kernels.ndim != 2 or kernels.shape[0] != 6 or not np.isfinite(kernels).all():
        raise ValueError("estimated kernels must be finite with shape (6, L)")
    ####
    harmonic = np.exp(1j*omega*dt*np.arange(kernels.shape[1]))
    response = complex(kernels[3]@harmonic)
    if not np.allclose(kernels[3],kernels[4],atol=1e-10,rtol=0.) or abs(response) < 1e-6:
        raise ValueError("horizontal transfer must be matched and invertible at carousel frequency")
    ####
    corrected = complex(avg_gyro[0],avg_gyro[1])/response*np.exp(-1j*omega*estimated_clock_offset_s)
    gyro_corrected = np.array([corrected.real,corrected.imag,avg_gyro[2]])
    result = static_gyrocompass(avg_accel,gyro_corrected,latitude_rad=earth.latitude_rad)
    return {
        "rotation_rate_degps": rotation_rate_degps,
        "retained_duration_s": count*dt,
        "warmup_duration_s": warmup*dt,
        "uncorrected_heading_error_deg": degrees(wrap_angle_rad(uncorrected.heading_rad-radians(heading_deg))),
        "corrected_heading_error_deg": degrees(wrap_angle_rad(result.heading_rad-radians(heading_deg))),
        "estimated_harmonic_gain": abs(response),
        "estimated_harmonic_phase_deg": degrees(np.angle(response)),
    }
####


def thermal_step_experiment() -> list[dict[str,float]]:
    """Apply ambient-based compensation to a lagging die in a static Earth record."""
    dt, tau, initial, ambient, beta = .1,30.,25.,55.,1e-6
    earth = LocalEarthReference(latitude_rad=radians(34.7304))
    model = AcquiredImuModel(
        ImuConfig(gyroscope=AxisConfig(thermal_bias_coefficient=(0.,beta,0.))),
        AcquisitionConfig(sample_period_s=dt,thermal=ThermalLagConfig(time_constant_s=tau)),
        sensor_rng=np.random.default_rng(31),
    )
    rows = np.zeros((2000,3))
    die_means = np.zeros(2000)
    temp = initial
    for i in range(len(rows)):
        sample = model.measure_ideal_rates(
            start_time=i*dt,end_time=(i+1)*dt,specific_force_body=earth.gravity_ned_mps2,
            angular_rate_body=earth.earth_rotation_ned_radps,ambient_temperature_celsius=ambient,
        )
        rows[i] = sample.angular_rate
        interval = thermal_interval(start_celsius=temp,ambient_celsius=ambient,dt_s=dt,time_constant_s=tau)
        die_means[i],temp = interval.mean_celsius,interval.end_celsius
    ####
    findings: list[dict[str,float]] = []
    for discard in (0.,30.,90.,180.):
        selection = slice(round(discard/dt),round((discard+10.)/dt))
        # Estimator is given the observed ambient thermometer, not latent die truth.
        naive = rows[selection].mean(axis=0)-np.array([0.,beta*(ambient-initial),0.])
        result = static_gyrocompass(earth.gravity_ned_mps2,naive,latitude_rad=earth.latitude_rad)
        # An oracle is reported separately as an algebraic check, NOT a fitted lag estimator.
        oracle = rows[selection].copy()
        oracle[:,1] -= beta*(die_means[selection]-initial)
        oracle_heading = static_gyrocompass(earth.gravity_ned_mps2,oracle.mean(axis=0),latitude_rad=earth.latitude_rad)
        findings.append({
            "settling_before_record_s":discard,"retained_duration_s":10.,
            "mean_die_minus_ambient_celsius":float(die_means[selection].mean()-ambient),
            "ambient_compensated_heading_error_deg":result.heading_deg,
            "oracle_die_compensated_heading_error_deg":oracle_heading.heading_deg,
        })
    ####
    return findings
####


def filtered_noise_experiment(*, trials: int = 2000, retained_samples: int = 600) -> dict[str,float|int]:
    """Independent Gaussian/convolution Monte Carlo for the exact FIR mean-covariance law."""
    taps = np.ones(25)/25.
    config = AcquisitionConfig(gyroscope_filter=FirFilterConfig(taps=tuple(taps)))
    rng = np.random.default_rng(2187)
    inputs = rng.normal(size=(trials,retained_samples+len(taps)-1))
    outputs = np.stack([np.convolve(x,taps,mode="valid") for x in inputs])
    means = outputs.mean(axis=1)
    exact_std = sqrt(filtered_white_mean_covariance(np.eye(6),config,retained_samples)[4,4])
    naive_std = outputs.std(axis=1,ddof=1)/sqrt(retained_samples)
    return {
        "trials":trials,"retained_samples":retained_samples,"fir_taps":25,
        "empirical_mean_std":float(means.std(ddof=1)),"exact_predicted_mean_std":exact_std,
        "mean_naive_iid_standard_error":float(naive_std.mean()),
        "exact_to_naive_sigma_ratio":float(exact_std/naive_std.mean()),
        "exact_95_interval_coverage":float(np.mean(abs(means)<=1.959963984540054*exact_std)),
        "naive_iid_95_interval_coverage":float(np.mean(abs(means)<=1.959963984540054*naive_std)),
    }
####


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("artifacts/calibration/tranche5"))
    args = parser.parse_args()
    output: Path = args.output
    output.mkdir(parents=True,exist_ok=True)
    config = AcquisitionConfig(
        gyroscope_filter=FirFilterConfig(taps=tuple([1/25]*25)),
        clock=ClockConfig(offset_s=.025),
    )
    rng = np.random.default_rng(761)
    reference = rng.normal(size=(900,6))
    measured,tags,valid = simulate_readout_record(reference,config)
    estimated = fit_readout_kernel(reference,measured,kernel_length=25)
    t = np.arange(1100)*config.sample_period_s
    heldout = np.column_stack([
        np.sin(2*pi*(.7+.21*j)*t)+.3*np.cos(2*pi*(3.+.13*j)*t) for j in range(6)
    ])
    heldout_measured,heldout_tags,heldout_valid = simulate_readout_record(heldout,config)
    predicted,start = predict_readout(estimated,heldout)
    identification = {
        "training_samples":len(reference),"heldout_samples":len(heldout),
        "maximum_kernel_error":float(np.max(abs(estimated.kernels-effective_fir_kernels(config)))),
        "maximum_heldout_prediction_error":float(np.max(abs(predicted[start:]-heldout_measured[start:]))),
        "rank_by_channel":estimated.rank_by_channel,
        "condition_by_channel":estimated.condition_by_channel,
        "reference_status":"known sample-level input, already synchronized; not a physical motion-rig validation",
    }
    # Separate the estimator's allowed observations/reference from evaluator-only injected truth.
    data_dir = output/"datasets"
    data_dir.mkdir(exist_ok=True)
    reference_bounds = np.column_stack((np.arange(len(reference))*config.sample_period_s,(np.arange(len(reference))+1)*config.sample_period_s))
    heldout_bounds = np.column_stack((np.arange(len(heldout))*config.sample_period_s,(np.arange(len(heldout))+1)*config.sample_period_s))
    np.savez_compressed(data_dir/"readout_training.npz",reference_bounds_s=reference_bounds,reference_rates=reference,measured_rates=measured,reported_bounds_s=tags,valid_channels=valid)
    np.savez_compressed(data_dir/"readout_heldout.npz",reference_bounds_s=heldout_bounds,reference_rates=heldout,measured_rates=heldout_measured,reported_bounds_s=heldout_tags,valid_channels=heldout_valid)
    np.savez_compressed(data_dir/"estimated_readout.npz",kernels=estimated.kernels,intercept=estimated.intercept)
    (output/"evaluator_only_truth.json").write_text(config.model_dump_json(indent=2)+"\n")
    metadata = {
        "schema_version":1, "sample_period_s":config.sample_period_s,
        "channel_order":["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"],
        "units":["m/s^2"]*3+["rad/s"]*3,
        "reference_bounds_s":"Allowed known reference-clock interval bounds; not injected filter/thermal parameters.",
        "reported_bounds_s":"Measurement clock interval bounds, not assumed synchronized.",
        "alignment":"This FIR experiment assumes corresponding rows are synchronized by the reference fixture.",
        "truth_separation":"Fitting does not read evaluator_only_truth.json.",
    }
    (data_dir/"recording_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    phase = [carousel_phase_experiment(config,estimated.kernels,estimated_clock_offset_s=.025,revolutions=n) for n in (1,2)]
    duration_grid = (10,30,60,120,300,600,1200)
    budgets: list[dict[str,object]] = []
    planner_config = AcquisitionConfig(gyroscope_filter=config.gyroscope_filter)
    for bound_deg,looks in ((.87,1),(.03,1),(.03,len(duration_grid))):
        for duration in duration_grid:
            point = plan_filtered_heading(
                config=planner_config,retained_samples=round(duration/config.sample_period_s),
                gyro_noise_density_radps_sqrt_s=2e-6,latitude_rad=radians(34.7304),
                target_half_width_rad=radians(.25),systematic_heading_bound_rad=radians(bound_deg),
                planned_looks=looks,
            )
            budgets.append({**asdict(point),"retained_duration_s":duration,"half_width_deg":None if point.half_width_rad is None else degrees(point.half_width_rad)})
        ####
    ####
    results = {
        "schema_version":1,"tranche":5,"evidence_status":"synthetic verification, not hardware validation",
        "readout_identification":identification,"carousel_phase":phase,
        "thermal_step":thermal_step_experiment(),
        "thermal_soak_to_point_one_celsius_s":thermal_settling_time_s(initial_difference_celsius=30.,tolerance_celsius=.1,time_constant_s=30.),
        "filtered_noise_confidence":filtered_noise_experiment(),"heading_data_budgets":budgets,
        "heading_budget_context":"Separate static, near-north planning illustration; systematic bounds are user assumptions, not validated residual bounds or carousel confidence.",
        "clock_gain_example":{"clock_frequency_error_ppm":100.,"apparent_integrated_angle_gain_error_ppm":100.},
        "assumptions":[
            "FIR identification has synchronized, noise-free known input in the numerical exact-recovery case.",
            "Carousel correction uses independently fitted FIR plus an assumed independent clock-offset estimate.",
            "Thermal oracle uses injected lag truth; no thermal time-constant estimator is claimed.",
            "Confidence planner assumes prefilter IID Gaussian noise and user-supplied systematic bounds.",
            "Seven-look adjustment is for predeclared finite tests, not arbitrary adaptive stopping.",
        ],
    }
    (output/"campaign_results.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"readout_identification":identification,"carousel_phase":phase,"filtered_noise_confidence":results["filtered_noise_confidence"]},indent=2))
    return 0
####


if __name__ == "__main__":
    raise SystemExit(main())
####
