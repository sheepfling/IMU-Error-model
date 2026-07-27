# Changelog

## Unreleased

- Adds the pure `characterize_imu_error_process()` API for estimator-facing
  white-noise covariances, persistent-state transitions, elapsed-time priors,
  and conditional versus marginal increment covariances.

- Shares flicker-process calibration between the stateful sampler and the
  stateless characterization path.

## 0.1.2 — clean baseline

Focused baseline release of `imu-error-model`, a Python package for stochastic
and parametric IMU measurement-error models.

- Converts truth velocity without gravity and truth orientation into body-frame
  `delta_v` and `delta_theta` measurements.

- Includes white noise, correlated 3-axis noise, turn-on bias,
  Gauss–Markov/random-walk bias, finite-band flicker bias, thermal terms,
  misalignment, scale/nonlinearity, clipping, and quantization.

- Provides a runtime-checkable protocol for alternate IMU models.
- Includes JSON/JSONC configuration profiles, metadata-bearing JSON/JSONC/YAML
  profile documents, Allan-deviation analysis, and a dead-reckoning demonstration.

- Names temperature-bearing fields explicitly in Celsius, including
  `reference_temperature_celsius` and `ImuOutput.temperature_celsius`; Kelvin is
  reserved for future absolute-temperature models.

- Rejects non-finite numeric configuration values, including covariance entries
  and profile sample periods.

- All hardware-oriented profile values are notional, approximate, and
  non-official examples.
