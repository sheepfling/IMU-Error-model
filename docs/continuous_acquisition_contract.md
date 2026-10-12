# Continuous acquisition contract

## Purpose

`ContinuousAcquiredImuModel` is an opt-in simulation boundary for effects that
occur before or around discrete sensor sampling. It does not replace
`ImuModel` or `AcquiredImuModel`.

## Processing order

1. A physical oscillator creates sample boundaries. Its frequency error and
   bounded boundary jitter change the actual truth-query times.

2. Continuous ideal specific force, angular rate, and ambient temperature are
   evaluated on sub-intervals.

3. First-order analog accelerometer and gyro bandwidth states evolve over the
   full physical interval.

4. The configured finite aperture averages the analog output over the declared
   support.

5. The resulting common triad averages pass through `ImuModel`, preserving full
   3x3 intrinsic transforms, bias, thermal, and stochastic semantics.

6. A causal post-transform fractional channel delay approximates inter-axis
   aperture/readout skew without pretending that a full intrinsic matrix can be
   sampled independently by axis.

7. The existing digital FIR and integer readout delays run next.
8. Reported-clock errors are applied only to the output tags.

## Important distinction

Physical clock error is not reported-clock error. Physical frequency error and
jitter alter what truth is sampled. `ClockConfig` alters only the timestamp tags.
The two mechanisms may coexist and must be identified against an independent
reference if absolute timing matters.

## Analog model

The current analog path is a first-order low-pass per triad. It is intentionally
small and identifiable. A sinusoidal frequency sweep can estimate the time
constant from phase response. Higher-order measured transfer functions and
non-white continuous noise are future extensions.

## Aperture and skew limitations

The common aperture is evaluated before the full intrinsic sensor transform.
Fractional per-channel skew is then represented as a causal linear interpolation
between consecutive transformed samples. This is a declared approximation.
True asynchronous pre-transform sensing axes would require six continuous analog
channel models and a more detailed physical triad representation.

The aperture computes channel-rate averages. It is not a claim of general
coning/sculling-corrected navigation preintegration.

## Calibration implications

Calibration reference data should be placed on the same physical support as the
sensor. Ignoring analog phase, physical oscillator error, or aperture support can
move those errors into fitted scale/cross-axis parameters. More samples do not
remove a deterministic support mismatch.
