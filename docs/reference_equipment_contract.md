# Reference-equipment calibration contract

Tranche 6, local development snapshot, 2026-09-08.

This is an experimental calibration/reference layer. It does not change the
runtime contracts of `ImuModel` or `AcquiredImuModel`, add a navigation estimator,
or claim that the entire IMU and calibration rig are jointly identifiable.
See the [playground](../playground/calibration/README.md),
[current backlog](../playground/calibration/BACKLOG.md), and
[field evidence map](../playground/calibration/reference_capabilities.py).

## Frames, signs, and the modeled physical system

NED is a local Earth-fixed frame: north, east, down. Sensor axes are right-handed.
`R` maps a sensor vector into NED. Angles are radians, angular rates rad/s,
specific force m/s², lever arms metres, and time seconds.

The new reference uses physical accelerometer specific force. A stationary
sensor measures `-R.T @ g_ned`, not `+R.T @ g_ned`. Earlier playground functions
use an apparent-gravity convention. The default remains compatible, but both
`static_gyrocompass` and `propagate_heading_uncertainty` now accept the explicit
argument `accelerometer_convention="specific_force"`. Intrinsic correction is
applied before this sign conversion. Existing quality/admission utilities have
not all been migrated to physical-force inputs; do not silently mix conventions.

The pivot is fixed in NED. One colocated accelerometer/gyro triad is at a fixed
sensor-coordinate lever arm `r` from that pivot. A table setup is

```text
R(t) = R_tilt R_base Rx(wx(q)) Ry(wy(q)) Rz(q) R_mount
q(t) = q0 + w0 t + a sin(2 pi f t) + b cos(2 pi f t)
wx(q) = wobble_x sin(order q + phase)
wy(q) = wobble_y cos(order q + phase)
```

Tilt and mount rotations use the exponential map. Wobble is synchronous with
shaft angle; derivatives include its contribution to angular velocity and
acceleration. Analytic first and second matrix derivatives use the product rule.
The tests compare them with finite differences and independent axis identities.

For Earth rotation `Omega` in NED, the generated ideal channels are

```text
omega_sensor = vee(R.T Rdot) + R.T Omega
L = R.T (Rddot + 2 skew(Omega) Rdot)
f_sensor = -R.T g_effective + L r
```

Thus `L r` retains tangential, centripetal, and Earth-Coriolis acceleration from
lever motion. WGS-84 normal gravity is treated as uniform effective gravity over
the small fixture. Spatial variations of the gravity/centrifugal field, transport
rate, pivot translation, tides, Earth-orientation updates, elastic deformation,
and separate within-IMU triad lever arms are not modeled. These limitations matter
before extrapolating to an ultrahigh-grade instrument or a large moving platform.

## Apertures and fidelity claims

`aperture_reference` integrates instantaneous sensor-channel components over
specified physical intervals using configurable Gauss-Legendre quadrature. It is
not midpoint substitution and not an exact formula for arbitrary wobble.

A channel average is not a navigation-preintegrated SO(3) increment. The simulator
feeds those means through the acquisition-facing interface and retains the
sensor-visible rates. It does not relabel the result as coning/sculling-corrected
navigation increments. Nonlinear sensor effects still act on the aperture mean
in the core; this is not analog integration of nonlinear intrinsics before the
aperture. Filtering retains the tranche-5 post-electronics ordering.

The campaign compares orders 2, 4, 8, and 16 with order 32 on a deliberately broad
0.4-second aperture. For that case order 16 agrees near floating-point precision;
this is a convergence check, not a universal error bound. Choose sample rate and
quadrature order against an absolute error budget below the Earth-rate signal.
Uniformly increasing quadrature order cannot compensate for a missing physical
error mechanism or a rank-deficient calibration experiment.

Setup changes are separate retained segments; handling/re-indexing trajectories
and handling time are not synthesized. `reset=False` preserves the same sensor
run realization across contiguous segments. The wrapper rejects physical gaps
before resetting or sampling. Each retained segment has a declared setup.

## Commands, observations, and truth are different objects

`AngleLaw` can express commands or the fitted encoder trajectory. Its source is
specified by the boundary using it. It is never automatically treated as truth.
`TableGeometry` can describe an injected rig or an externally surveyed estimate;
these are separate instances in the nonideal experiment.

`EncoderObservations` contains only reported timestamps, continuous unwrapped
angle samples, and a source identifier. `MeasuredImuRecord` contains reported
intervals, rates, temperature, validity, and a sensor identifier. Neither includes
latent IMU parameters, physical-time truth, or the injected encoder-error config.

`simulate_table_record` returns `(measured_record, evaluator_truth)`. The second
object is separate and is used only for scoring. Fitting accepts
`ObservedCalibrationData`, reconstructed from encoder observations, surveyed
geometry, a declared Earth reference, and independently supplied clock/encoder
correction estimates. Passing an `ApertureReference` directly to the observed-fit
API is rejected. These are structural/provenance guards, not a security boundary
against a caller deliberately fabricating a reference record.

The legacy `CalibrationDataset` names its predictor arrays `specific_force_body`
and `angular_rate_body`. In the observed wrapper these arrays contain estimated
references, not hidden truth. The ordinary least-squares covariance is explicitly
labeled conditional on these references.

The campaign compares three routes using the same IMU recordings: nominal
commands with nominal geometry; encoder observations with nominal geometry; and
encoder observations with nonideal, independently declared survey/correction
estimates. Ground truth is read only by the generator and holdout scorer. The
metrology estimates are synthetic assumptions, not measured certificates.

## Encoder model and reconstruction

The simulator adds angle zero error, scale error, angle-synchronous cyclic error,
white angle noise, quantization, clock offset, clock frequency error, and bounded
timestamp-tag jitter. Encoder noise uses an independent RNG. Reported timestamps
that become nonmonotone are rejected, not sorted to conceal the problem.

The reconstructor fits the declared harmonic basis to observed angles. It does
not differentiate quantized/noisy angles directly and does not substitute the
commanded amplitude or phase. The harmonic frequency is a declared experimental
input, not fitted. Centered/scaled least squares, rank/conditioning checks, an
optional residual limit, and observation-span admission protect this route.
Extrapolation beyond the encoder observation span is refused.

`EncoderCorrection` holds independent estimates of angle zero/scale and clock
offset/frequency. Defaults of zero are explicit assumptions, not evidence of
perfect metrology. Cyclic errors remain model mismatch; they are injected and
residual-tested but not automatically calibrated. Unknown physical sample-time
jitter is distinct from timestamp-tag jitter and remains open.

The reconstructed reference is queried on IMU-reported intervals. Both clocks
must already be aligned, or a separate relative-delay fit must provide the
alignment. For filtered IMU readings, removing invalid startup rows is not an
inverse filter: a matching reference prediction operator or independently
verified correction is still required. The held-out intrinsic-fit campaign uses
unity filters so it does not hide that limitation.

## Restricted equipment identification

`fit_lever_arm` solves `f_corrected - f_pivot = L r + b` with optional residual
bias. It assumes a previously calibrated accelerometer and an independently
surveyed reference. Variable rates and multiple body-axis orientations expose
both tangential and centripetal terms. A single-axis experiment rejects the
unobservable axial component instead of returning an arbitrary pseudoinverse
answer with zero uncertainty. Constant-rate levers and offsets can also be
confounded. The covariance is conditional on the reference and accelerometer
calibration, with IID isotropic residual-row assumptions.

`fit_encoder_timing` fits one channel to a positive gain, additive bias, and a
bounded relative delay using exact interval-average reference rates. It uses a
profile search and full Jacobian-rank check. Delay bounds must be shorter than
half the known harmonic period, and boundary optima are rejected. The fitted
quantity is relative time alignment, not a unique attribution to sensor latency
versus encoder-clock error. A constant rate cannot identify delay regardless of
record length. Its covariance is conditional on the independent reference law.

General simultaneous fitting of rig tilt, wobble, encoder scale, clock frequency,
IMU mounting, full intrinsic matrices, and lever arms is not implemented.
Two exact gauge tests show why unrestricted fitting would be dishonest: sensor
gain and an unknown reference-rate gain are inseparable from their product, and
an unconstrained mounting rotation can be absorbed into full sensor matrices.
Additional metrology or explicit structural constraints are needed.

## Shared-reference uncertainty and heading

`propagate_shared_reference_uncertainty` reprocesses the same observed recording
under small perturbations of externally estimated reference parameters. It adds
`J_ref P_ref J_ref.T` to the conditional 33-parameter calibration covariance.
The reference perturbation is shared across the entire run, not drawn afresh for
every sample. Raw measurements, timestamps, labels, and temperatures must remain
identical across perturbations; a builder that regenerates them is rejected.

The result retains cross-axis and accelerometer/gyro coefficient covariance.
It can be passed into the existing heading propagation, with the correct
accelerometer convention. First-order linearization and independence between
sensor noise and the metrology estimate are required. This is uncertainty
propagation, not general errors-in-variables bias correction, a posterior over
all rig parameters, or permission to claim IID behavior for colored residuals.

The heading covariance validator now rejects materially negative covariance
rather than silently clipping it into a plausible PSD matrix. Roundoff-scale
negative eigenvalues may still be projected to zero.

## Corrected earlier turntable behavior

The previous X/Y Ferraris rotation path applied a body-Z rotation to already
body-expressed Earth/gravity vectors. The actual table rotates about NED down.
The proper transform is `R_base.T @ mean(Rz.T) @ R_base` in base-body coordinates.
This is corrected, with independent numerical integration tests for X, Y, and Z.
Older X/Y dynamic-turntable campaign numbers must be regenerated; they are not
accepted as current physical validation. The earlier legacy apparent-gravity
sign convention is preserved and explicitly separated from the new physical
specific-force path.

## Reproduction and evidence

From the source root:

```console
PYTHONPATH=src:. python -m playground.calibration.run_tranche6_campaign --output artifacts/tranche6
PYTHONPATH=src:. python -m pytest -q tests/test_reference_*.py
PYTHONPATH=src:. python -m pytest -q
```

On PowerShell, set `$env:PYTHONPATH="src;."` before the Python commands. The
playground is delivered in the source snapshot, not installed by the core-only
wheel. There are no new runtime dependencies.

Primary research context and exact source URLs are retained in
[reference research provenance](../playground/calibration/reference_research.json).
Research motivates the error families and uncertainty treatment; it does not
validate the synthetic parameters chosen here.
