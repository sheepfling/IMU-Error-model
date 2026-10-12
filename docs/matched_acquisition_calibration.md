# Matched acquisition-aware calibration contract

## Purpose

A calibration reference and an IMU sample are not comparable merely because they
carry the same nominal timestamp. Digital filtering, channel readout delay,
finite measurement aperture, and independent clocks change the support on which
each reported sample is defined. This playground route therefore predicts the
reference **forward** onto the observed sensor support before fitting intrinsic
sensor coefficients.

The route is a calibration-research harness. It does not change the public
`ImuModel` contract and it does not claim to recover arbitrary analog transfer
functions, physical sampling jitter, or every source of latency.

## Boundary

The supported flow is

```text
encoder observations + declared survey
        |
        v
fitted smooth reference law
        |
IMU reported clock -- independent relative-clock correction
        |
        v
aperture reference on corrected interval boundaries
        |
identified FIR/readout operator
        |
        v
output-channel-specific matched regressors
        |
observed IMU rates
        v
matched intrinsic calibration
```

The estimator never receives injected acquisition configuration, physical-time
truth, commanded table motion, or evaluator-only sensor parameters.

## Forward matching, not deconvolution

Let `h_j[l]` be the independently identified effective readout kernel for output
channel `j`. For accelerometer output axis `j`, the deterministic sensor model is

\[
y^a_{j,k} = \sum_m M^a_{jm}\,\bar f^{(j)}_{m,k} + b^a_j + e^a_{j,k},
\]

where

\[
\bar f^{(j)}_{m,k} = \sum_l h_j[l] f_{m,k-l}.
\]

For gyro output axis `j`,

\[
y^g_{j,k} = \sum_m M^g_{jm}\,\bar\omega^{(j)}_{m,k}

            + \sum_m G_{jm}\,\bar f^{(j)}_{m,k}
            + b^g_j + e^g_{j,k}.
\]

The kernel is applied to every input regressor contributing to the same output
channel. This matters when the intrinsic matrices contain cross-axis terms.
Filtering only the nominal same-axis reference is not equivalent.

No numerical inverse of `h_j` is formed. An unstable/nonminimum-phase filter can
therefore be represented without creating an unstable calibration preprocessor.

The readout-identification intercept is also not applied to the reference.
Intrinsic sensor bias remains a calibration parameter. The intercept from an
independent readout-identification experiment is retained only as a diagnostic.

## Startup history and validity

Reference reconstruction retains all startup rows while the forward filter is
applied. Rows are discarded only after the complete identified filter history is
available and the observed IMU validity flags are true on all six channels.

Dropping invalid rows *before* filtering changes the first retained regressor and
is therefore incorrect.

Multiple separately reset/reindexed sessions are matched locally before they are
combined. Filter history is never carried across an unmodeled handling/reindexing
event.

## Relative clock mapping

The current reported-clock correction assumes

\[
t_{tag}=t_0+\delta_t+(t_{ref}-t_0)(1+\epsilon_c),
\]

with `epsilon_c` supplied in ppm. The matched route applies the inverse mapping
before reference aperture integration.

A separate multi-channel timing experiment can estimate a common relative clock
offset and frequency error while profiling six positive gains and six biases.
The final 14-parameter Jacobian must be full rank. Constant-rate excitation is
rejected because clock frequency, sensor scale, bias, and phase can otherwise
contain exact or near-exact gauges.

The result is **relative alignment**. Without an independently disciplined time
standard, it cannot identify whether a recovered error belongs physically to the
IMU clock or the external reference clock.

## Readout and encoder-law uncertainty

Known-input FIR identification now retains the OLS covariance of every fitted tap
and intercept. Encoder-law fitting retains the covariance of

```text
offset_rad, rate_radps, sine_rad, cosine_rad
```

for its declared harmonic basis.

The matched-support uncertainty route can assemble independent covariance blocks
for:

- all six identified FIR tap vectors;
- one or more independently fitted encoder laws; and
- an externally established relative-clock offset/frequency covariance.

It then refits the **same measured IMU recordings** under central perturbations of
those support parameters and forms

\[
P_{cal,total} \approx P_{cal\mid support} + J_s P_s J_s^T.
\]

This is a first-order delta-method propagation. It is not a general
errors-in-variables bias correction, and it assumes the support-identification
records are independent of the intrinsic-calibration record.

## Held-out promotion rule

A matched calibration must not be promoted merely because training residuals are
small. The current campaign holds out:

- different shaft-modulation frequencies;
- alternating rate reversals; and
- independent sensor/encoder noise realizations.

The fitted intrinsic model is scored by predicting the **sensor-visible matched
measurement**, not by comparing filtered measurements directly with unfiltered
instantaneous truth.

## Remaining limitations

This route still does not model or identify:

- physical sampling-clock jitter that changes truth-query times;
- fractional/inter-axis aperture skew;
- analog anti-aliasing or arbitrary continuous-time sensor response;
- filter coefficients that change with temperature, amplitude, or mode;
- general nonlinear intrinsics inside the finite aperture;
- coning/sculling-corrected navigation increments;
- general reference errors-in-variables bias correction;
- joint unrestricted rig geometry, encoder scale, clock, and full IMU matrices;
- flexible-fixture dynamics or vibration rectification.

Those require separate observable experiments and convergence references rather
than additional unconstrained configuration fields.
