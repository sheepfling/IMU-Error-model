# Calibration playground

This directory is an experimental verification harness, not part of the
runtime `imu_error_model` package contract.

## Matched acquisition-aware calibration — tranche 7

Read the [matched support contract](../../docs/matched_acquisition_calibration.md)
before fitting filtered/delayed records. This route reconstructs encoder/survey
references on an independently corrected relative clock and passes them FORWARD
through fitted per-output readout kernels. It never deconvolves the observed IMU.

```console
PYTHONPATH=src:. python -m playground.calibration.run_tranche7_campaign --output artifacts/tranche7
```

The campaign uses independent readout and timing-identification records, six
rate-table orientations for intrinsic calibration, and held-out table frequencies
with alternating rate reversals. Readout-tap, encoder-law and relative-clock
uncertainty can be propagated into the intrinsic covariance. Constant-rate clock
and scale gauges are rejected by the multi-channel timing fit.

## Uncertain reference equipment — tranche 6

Read the [frame and metrology contract](../../docs/reference_equipment_contract.md)
before combining these routes. Physical specific force has the opposite static
sign from the legacy apparent-gravity playground. Observed-reference covariance
is conditional until independent metrology uncertainty is supplied.

```console
PYTHONPATH=src:. python -m playground.calibration.run_tranche6_campaign --output artifacts/tranche6
```

The campaign produces 2,400 training samples and 600 independent holdout samples,
compares commanded versus encoder/surveyed references, fits relative timing and
lever arms in observable cases, and measures coverage with a run-shared reference
error. Unknown rig parameters are not silently treated as calibrated. See the
[updated backlog](BACKLOG.md) and [reference field audit](reference_capabilities.py).

## Known-reference recovery

The first path uses the model's explicit ideal-rate input to generate a
known-reference corpus with:

- six static `±g` orientations;
- symmetric known rotations about all three axes;
- full accelerometer and gyro sensitivity matrices;
- fixed accelerometer and gyro biases; and
- gyro specific-force sensitivity.

Run it from the repository root:

```console
PYTHONPATH=src:. python -m playground.calibration.run_known_reference
```

The script fits an independent NumPy least-squares calibration and writes the
synthetic raw/reference rows to
`artifacts/calibration/known_reference_noiseless.csv`. The fit reports design
rank, condition number, residual RMS, and truth-versus-estimate parameter
error.

## Ferraris equation parity

The nine-region corpus can be checked offline against a small, source-aligned
implementation of the equations used by `imucal` 2.6.0:

```console
PYTHONPATH=src:. python -m playground.calibration.run_ferraris_reference
```

This verifies the region layout, units, matrix orientation, bias convention,
gyro force-sensitivity convention, and integrated rotation angles without
claiming that the external package itself was imported.

## `imucal` compatibility path

Install the optional comparison dependency and run the external package on the
exact nine-region Ferraris path:

```console
python -m pip install -e ".[calibration]"
PYTHONPATH=src:. python -m playground.calibration.run_imucal
```

This path generates the six static regions `x_p`, `x_a`, `y_p`, `y_a`, `z_p`,
`z_a` and the three rotation regions `x_rot`, `y_rot`, `z_rot`. It exports the
region-indexed six-column CSV expected by `imucal`, runs
`FerrarisCalibration`, converts the returned `K`/`R` representation back into
canonical forward matrices, and compares every mutually representable
parameter with the injected truth.

The generated files are:

- `artifacts/calibration/imucal_ferraris_noiseless.csv`;
- `artifacts/calibration/imucal_ferraris_calibration.json`; and
- `artifacts/calibration/imucal_ferraris_validation.json`.

## Statistical, alignment, and data-confidence campaigns

The playground now separates five evidence layers instead of treating one
successful calibration as sufficient:

1. **Identifiability** — rank, singular values, condition number, and weakest
   design direction establish whether the requested parameters can be recovered
   from the chosen experiment.

2. **Parameter uncertainty** — multivariate least-squares covariance describes
   IID residual rows; a label-wise circular moving-block bootstrap is available
   when calibration residuals are serially correlated.

3. **Measurement uncertainty** — IID or Bartlett/Newey-West long-run covariance
   estimates the covariance of the accumulated static measurement mean and
   reports a per-channel effective sample size.

4. **Alignment quality** — gravity magnitude, sample count, duration, variation,
   split-record drift, and Earth-rate geometry gates must pass before a static
   heading and confidence interval are accepted.

5. **Empirical validation** — paired Monte Carlo trials compare propagated
   heading intervals with observed errors and expose undercoverage or model
   mismatch.

Run the complete tranche-four campaign from the repository root:

```console
PYTHONPATH=src:. python -m playground.calibration.run_tranche4_campaign
```

It writes machine-readable evidence under
`artifacts/calibration/tranche4/`, including:

- joint calibration-size/compass-duration uncertainty coverage;
- a Pareto and least-acquisition-time data plan for a requested 95% heading
  half-width;

- quality-gated confidence for increasing prefixes of one accumulating record;
- static, two-position, four-position, and continuous-carousel comparisons;
- carousel-rate sensitivity;
- IID versus colored-noise long-run covariance and effective sample size;
- moving-block-bootstrap calibration covariance inflation; and
- white-noise-only duration planning curves.

The data planner counts sampled calibration time, compass averaging time, and
an optional per-transition overhead. It does not silently include warm-up,
operator handling, or site setup; those must be supplied explicitly when a
physical campaign is planned.

## Earth rotation and rotating-table fidelity

`earth_reference.py` supplies WGS-84 nominal Earth rotation in local NED axes
and latitude/height-dependent normal gravity. The Earth rotation magnitude is
an explicit, validated input so a traceable reference value or sensitivity
study can replace the nominal constant.

Turntable and carousel paths use one shared exact finite-aperture rotation
integral. A sample therefore represents the interval-average projection of
Earth rate and gravity into the rotating sensor frame, not merely the table
orientation at the sample midpoint. This matters for coarse sample periods,
high table rates, and low-rate calibrations where Earth rotation is no longer
negligible relative to the commanded rotation.

The current Earth model intentionally remains a local engineering reference. It
does not yet ingest time-tagged Earth-orientation parameters, polar motion,
local deflection of the vertical, or a measured gravity anomaly. Those effects
belong in an external reference provider rather than as hidden terms inside the
sensor-error model.

## Gyrocompass modulation routes

- `gyrocompass.py` performs calibrated static TRIAD alignment.
- `quality.py` adds stationarity gates and first-order confidence propagation.
- `maytag.py` preserves one physical turn-on-bias realization across one, two,
  or four static down-axis positions.

- `carousel.py` removes known table rate and exactly demodulates continuously
  rotating samples back into a nonrotating base frame.

- `data_confidence.py` evaluates empirical duration sweeps and prefix stopping
  targets.

- `planning.py` chooses among tested calibration/compass allocations rather
  than assuming that longer compass averaging is always the best use of time.

These are verification and experiment-design tools, not a certified navigation
solution. Reported confidence is conditional on the configured error model,
reference assumptions, calibration covariance method, covariance lag window,
and stationarity gates.

## Executable completeness audit

`capability_audit.py` maintains two explicit mappings:

- every public `AxisConfig` and `ImuConfig` field has a named excitation or a
  documented non-calibration decision; and

- every supported calibration/alignment route has executable or diagnostic
  evidence.

The audit currently covers known-reference affine recovery, Ferraris hand and
turntable routes, thermal/cross-axis calibration, stochastic characterization,
static gyrocompassing, Maytagging, carouseling, correlated calibration
uncertainty, and joint data-budget planning.

Tranches 5--7 add digital FIR/readout-delay, reported-clock, one-node thermal
lag, uncertain rate-table references, and forward-matched acquisition-aware
calibration. Analog anti-aliasing, physical sampling jitter/fractional skew,
vibration rectification and general coning/sculling remain open. See the
[backlog](BACKLOG.md) for the next acceptance cases.

## Tranche 5: acquisition, reference phase and information sufficiency

```bash
PYTHONPATH=src python -m playground.calibration.run_tranche5_campaign
```

This writes `campaign_results.json`, synchronized known-reference training and
held-out NPZ records, estimated kernels, and a separately named evaluator-only
configuration. The fitter receives only the allowed measured/reference arrays.

The carousel diagnostic uses fitted FIR coefficients and an assumed independent
clock-offset estimate. It does not jointly estimate an unknown clock and heading.
The thermal oracle is explicitly labeled; no thermal time-constant estimator is
claimed. The confidence planner is a separate static near-north approximation
and does not supply carousel confidence. Its systematic bounds are illustrative
inputs, not bounds certified by the synthetic transfer-recovery case.

Read the [runtime contract](../../docs/acquisition_contract.md) for stage order,
noise placement, frame semantics, clock behavior and limitations.

## Tranche 9: high-rate coning/sculling preintegration

The high-rate playground now has an independent SO(3) preintegration oracle for
rotating body-frame angular rate and specific force:

```console
PYTHONPATH=src:. python -m playground.calibration.run_tranche9_campaign
```

The campaign includes a circular coning cycle whose component-wise mean gyro
rate is zero but whose net rotation is not, and an orthogonal sculling cycle
whose raw `delta_theta` and `delta_v` sums are both zero while the true
start-frame velocity increment is nonzero. It compares naive component sums,
a classical recursive small-angle correction, and exact piecewise-constant
SO(3) integration across tested minor-sample rates.

Calibration is applied to each high-rate minor sample before navigation
preintegration. A dedicated regression fits the existing full affine IMU
calibration, corrupts a sculling sequence with that sensor model, calibrates the
minor samples, and recovers the ideal preintegrated increment. Collapsing the
same measurements to one interval mean deliberately loses the sculling term.

See [the preintegration contract](../../docs/high_rate_preintegration.md).
