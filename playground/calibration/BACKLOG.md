# Calibration backlog after tranche 7

## Current boundary

The instrument-error core and opt-in acquisition interface remain independent
of the calibration playground. The new reference layer models fixed table tilt,
shaft-angle-synchronous wobble, a fixed sensor lever arm, Earth-rate projection,
encoder observations, and reconstructed surveyed references. See the
[reference contract](../../docs/reference_equipment_contract.md) and
[field evidence](reference_capabilities.py).

Completed identification is deliberately narrower than modeled physics:
observed encoder harmonic coefficients, conditional lever-arm fitting, and
single-channel gain/bias/relative-delay fitting. Full sensor matrices, fixed
biases, and gyro force sensitivity can be recovered against independently
surveyed references. Tilt/wobble and absolute encoder calibration are supplied
metrology, not jointly estimated from an arbitrary IMU trace.

## Completed: matched acquisition-aware reference calibration

The matched-support route now reconstructs all startup reference intervals on an
independently corrected clock, applies identified per-output FIR/delay kernels
FORWARD to every contributing force/rate regressor, and fits full deterministic
intrinsics only on synchronized valid rows. It never numerically deconvolves the
observed IMU. Separate readout and encoder-law fits retain covariance, and a
delta-method route carries readout/encoder/relative-clock uncertainty into the
full calibration covariance. A common six-channel clock fit profiles channel
gain/bias and rejects deficient constant-rate clock/scale gauges. Held-out tests
change frequencies and reverse rates. See `docs/matched_acquisition_calibration.md`.

## Completed: physical sampling and continuous-time sensor response

Physical sampling-clock frequency error and bounded boundary jitter now change the
truth-query schedule rather than only timestamp tags. A finite common aperture,
first-order analog triad bandwidth, and explicit post-transform fractional
channel skew are available through `ContinuousAcquiredImuModel`. Multi-frequency
phase excitation identifies the analog time constant, and alias campaigns expose
above-Nyquist contamination. See `docs/continuous_acquisition_contract.md`.

The current aperture still computes rate averages, not general coning/sculling-
corrected navigation increments. Fractional axis skew is deliberately a causal
post-transform approximation so full 3x3 intrinsic matrices keep coherent
semantics. Continuous sensor noise before the analog filter is not yet modeled.

## Further calibration and environmental fidelity

Add general errors-in-variables fitting for noisy observed references, including
correlated encoder errors and uncertainty from model-based trajectory smoothing.
Shared metrology covariance propagation is not a substitute for bias correction.

Add a constrained joint rig/IMU estimator only after its observable parameter
subspace and external anchors are explicit. Unknown reference scale and sensor
scale, and full intrinsic matrices versus mounting rotation, have exact gauges.
Add multi-axis fixtures, separate triad lever arms, flexible fixture response,
wobble metrology, and encoder cyclic-error calibration when supported by distinct
excitation and reference measurements.

Thermal time-constant identification, independently sampled thermometer dynamics,
multi-node temperature, hysteresis, vibration rectification, higher-order force
sensitivity, and shock recovery remain open. Their tests need identifiable
excitation and step-size convergence, not just fields that accept configuration.

## Confidence and promotion gates

Carry calibration, reference, readout, clock, and measurement uncertainty through
the appropriate heading route, retaining shared errors across samples. Validate
coverage over both random seeds and independent metrology realizations. Use an
appropriate confidence sequence for open-ended adaptive stopping; the existing
finite planned-look adjustment is not that guarantee.

Migrate physical-specific-force conventions through all quality/Maytag/carousel
helpers with backward-compatible explicit labels. Corrected X/Y legacy
turntable campaigns must be regenerated; old numeric reports are historical.

Add recorded hardware replay, independent equipment metrology, held-out
operating temperatures/rates, and a traceable empirical noise characterization
before claiming device accuracy. No synthetic profile certifies hardware.

External `imucal` parity remains dependency-gated. Run Ruff, Pyright, supported
Python 3.11/3.12 CI, packaging, and the optional calibration dependency before
publication. The current snapshot is local; no upstream push/tag is implied.

## Next: high-rate inertial preintegration and measured analog response

Add an independent high-rate truth oracle for rotating specific force and angular
rate, including coning/sculling reference algorithms and convergence studies. Add
higher-order or measured analog transfer functions, continuous-time noise before
the analog bandwidth, explicit ADC/decimation response, and spectral alias/noise
folding validation. Then connect those support models into matched calibration
and heading covariance rather than treating them as evaluator-only effects.

## Completed: high-rate inertial preintegration oracle

An independent SO(3) truth oracle now integrates rotating body-frame specific
force and angular rate over a major interval. It includes exact
piecewise-constant segment integration, raw high-rate minor-increment sampling,
a classical recursive small-angle coning/sculling comparison, and explicit
sample-rate convergence studies.

Two canonical zero-mean cases are executable: circular coning with zero
component-wise mean gyro rate but nonzero net rotation, and orthogonal sculling
with zero raw angle/velocity sums but nonzero cross-axis velocity increment.
Calibration is fitted and applied to minor sensor samples before preintegration,
which proves the intended ordering and demonstrates why major-interval mean
correction is insufficient.

## Next after high-rate preintegration

1. Higher-order or measured analog transfer functions plus explicit
   ADC/decimation response.

2. Continuous-time sensor noise before analog bandwidth and spectral
   noise-folding validation.

3. Propagate analog time-constant, physical-clock, and preintegration uncertainty
   into matched intrinsic calibration and heading covariance.

4. Evaluate nonlinear sensor terms inside the finite aperture/minor samples,
   instead of only on interval-mean rates.

5. Add six-channel asynchronous pre-transform analog sensing when hardware
   evidence supports distinct axis apertures.

6. Recorded hardware replay and independent metrology before device-level claims.
