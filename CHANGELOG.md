# Changelog

## 0.1.0 — initial release

Initial focused release of `imu-error-model`, a Python package for stochastic
and parametric IMU measurement-error models.

- Converts truth velocity without gravity and truth orientation into body-frame
  `delta_v` and `delta_theta` measurements.
- Includes white noise, correlated 3-axis noise, turn-on bias,
  Gauss–Markov/random-walk bias, finite-band flicker bias, thermal terms,
  misalignment, scale/nonlinearity, clipping, and quantization.
- Provides a protocol and base class for alternate IMU models.
- Includes JSON profiles, optional YAML example profiles, Allan-deviation
  analysis, and a dead-reckoning demonstration.
- All hardware-oriented profile values are notional, approximate, and
  non-official examples.
