# IMU Error Model

Composable stochastic and parametric error models for synthetic IMU increments.
The package receives truth orientation and truth velocity excluding gravity;
it returns imperfect accelerometer `delta_v` and gyroscope `delta_theta` over
each sample interval.

Install the initial release with:

```bash
python -m pip install imu-error-model
```

The canonical measurement frame is the body frame at the beginning of the
interval. The orientation input is `R_world_from_body`, a proper 3x3 rotation
matrix. The model does not calculate gravity, dynamics, reconstruction, or
state estimation.

## Extension contract

Alternate implementations can inherit `BaseImuModel` or satisfy
`ImuModelProtocol`. They implement `reset()` and:

```python
measure(timestamp, velocity_without_gravity, orientation_world_from_body,
        temperature=25.0) -> ImuOutput
```

The first call establishes the interval baseline and returns zero increments.
Subsequent calls require strictly increasing timestamps in seconds. Truth
velocity is a 3-vector in m/s, orientation is a proper `R_world_from_body`
rotation matrix, and temperature is in degrees Celsius. `AxisConfig` noise
densities are continuous-time rate densities: accelerometer values use
m/s²/sqrt(Hz), gyroscope values use rad/s/sqrt(Hz). If supplied,
`noise_covariance` is the 3x3 covariance of the corresponding continuous-time
rate noise and must be positive semidefinite.

`ImuOutput.delta_v` is in m/s and `ImuOutput.delta_theta` is in radians.
Their derived `acceleration` and `angular_rate` properties are rates over the
reported `dt`; they are convenience views, not the model's input contract.

Profiles are JSON-compatible and can be loaded with `load_profile()` or
created with `save_profile()`. Example profiles are in `profiles/`.

Hardware-oriented best-effort estimates are kept separately under
[`examples/imu_profiles/hardware-estimates`](examples/imu_profiles/hardware-estimates).
See the [profile examples guide](examples/imu_profiles/README.md) for loading
and interpreting those notional estimates.
They are documentation/examples only and are not package defaults or runtime
configuration. Every numeric value there is notional, approximate, and
non-official; even source-anchored values are modeling estimates, not vendor
specifications, certification results, or performance guarantees. The optional
`load_yaml_profile()` adapter maps their YAML
fields into the current model while retaining metadata, sample period, source
path, and reference links.

The built-in model supports independent white noise, correlated 3-axis noise
covariance, turn-on bias, random-walk/Gauss–Markov bias, a finite-band
flicker-bias approximation, fixed run-level misalignment, scale factor,
nonlinearity, thermal bias/noise coefficients, clipping, and quantization. More
elaborate thermal behavior can be supplied by implementing `ThermalModel`.

## Development and CI

Install the development tools and run the same task runner used by GitHub:

```bash
python3 -m pip install -e ".[dev]"
python3 scripts/ci.py test
python3 scripts/ci.py coverage
python3 scripts/ci.py lint
python3 scripts/ci.py build
python3 scripts/ci.py docs --build
```

`scripts/ci.py all` runs linting, coverage, and package distribution builds.
`scripts/ci.py markdown` checks that Markdown links are relative and resolve,
that nested Markdown files are linked, and that source-document references do
not enter the public documentation.
The GitHub workflow calls this runner on Python 3.11 and 3.12 and uploads the
wheel, source distribution, and generated documentation PDF as artifacts.

The `Publish to PyPI` workflow publishes a GitHub Release through PyPI Trusted
Publishing. Configure the repository's `pypi` environment and trusted
publisher on PyPI before publishing the first release; no long-lived PyPI
token is stored in the repository.

To regenerate the standard Allan-deviation analysis set, install the analysis
extra and run:

```bash
python3 -m pip install -e ".[analysis]"
python3 scripts/ci.py allan
```

This regenerates `artifacts/allan/` for all hardware profiles plus the two
short-correlation test fixtures, and regenerates the ADIS datasheet overlay in
`artifacts/adis-allan-overlay/`. Use `--duration`, `--points`, and
`--adis-duration` to change the record lengths or plot density.

For polished storefront-oriented charts, run:

```bash
python3 scripts/ci.py showcase
```

This writes paired PNG/SVG assets under `artifacts/showcase/`:

- `hero-allan-ladder`: representative navigation, tactical, and consumer
  Allan-deviation curves.
- `error-anatomy-allan`: white noise, Gauss–Markov bias, finite-band flicker,
  and combined Allan signatures.
- `reconstruction-drift`: long-term position and attitude drift from identical
  truth motion.
- `measurement-time-series`: truth rates compared with sampled imperfect rates
  and their residual errors.
- `thermal-trends`: thermal-only accelerometer and gyroscope error trends.

Use `--allan-duration`, `--reconstruction-duration`, and
`--temperature-points` to control generation time and density. The assets are
notional showcase renderings, not vendor performance claims.

To regenerate the complete analysis bundle in one step:

```bash
python3 scripts/ci.py analysis
```

This combines the standard Allan-deviation plots, the ADIS datasheet overlay,
and all showcase plots. Normal `pytest` intentionally does not generate files
or require Matplotlib; the explicit analysis task owns generated artifacts.

## Analysis example

To compare dead-reckoning drift across all hardware-estimate profiles:

```bash
python3 examples/dead_reckoning.py --duration 10
```

This reconstructs position, velocity, and attitude from the model's `delta_v`
and `delta_theta` outputs and writes a CSV summary. It is deliberately kept
outside the core package; filters such as an MEKF belong in a separate
estimation project.

For datasheet-oriented noise characterization, generate Allan-deviation data
and charts across the same estimate profiles:

```bash
python3 -m pip install -e ".[analysis]"
python3 scripts/ci.py allan
```

This writes `artifacts/allan/allan_deviation.csv` plus accelerometer and
gyroscope PNG charts. The τ range can be controlled explicitly:

```bash
python3 examples/allan_variance.py --duration 1000 \
  --tau-min-s 0.01 --tau-max-s 250 --points 32
```

To visualize the finite-band flicker fixture as well:

```bash
python3 examples/allan_variance.py --duration 100 \
  --profile profiles/test/flicker-band.json
```

The generated charts are analysis artifacts, not package runtime outputs. A
record should be several times longer than the largest τ of interest; a
120-second run cannot establish the long-term behavior of profiles with
250–3600-second bias correlation times. For the first-order Gauss–Markov model,
expect a short-time positive slope, a turnover near the correlation time, and
eventual averaging-down at long τ. A true bias-instability plateau would
require a different stochastic process. The model also supports a finite-band
flicker-bias approximation through `flicker_bias_std`,
`flicker_min_correlation_time`, `flicker_max_correlation_time`, and
`flicker_components`. It is calibrated to the requested Allan-deviation level
near the geometric-center τ of that band.

### Allan-variance meaning of the stochastic terms

The stochastic parameters are rate-domain parameters. With sample period
`dt`, `white_noise_density = N` produces independent rate samples with
standard deviation `N / sqrt(dt)`. Under the estimator used by the analysis
scripts, its short-time Allan deviation is therefore approximately

```text
sigma_A(tau) = N / sqrt(tau)
```

For a stationary Gauss–Markov bias with stationary standard deviation
`bias_std = sigma_b` and correlation time `T`, the continuous-time reference
curve is

```text
sigma_A²(tau) = sigma_b² [
    2 T / tau
    - (T / tau)² (3 - 4 exp(-tau/T) + exp(-2 tau/T))
]
```

It rises with an approximately `+1/2` log-log slope at short averaging times,
turns over near `T`, and falls with an approximately `-1/2` slope at long
averaging times. If `bias_correlation_time` is omitted, `bias_std` instead
specifies the random-walk driving scale; that process has the expected
longer-term `+1/2` Allan slope rather than a stationary knee.

The finite-band flicker model is a practical datasheet-matching approximation,
not an infinite-band 1/f generator. It sums `flicker_components` independent
Gauss–Markov processes whose correlation times are logarithmically spaced from
`flicker_min_correlation_time` to `flicker_max_correlation_time`. The
`flicker_bias_std` value is the target Allan deviation at the geometric-center
averaging time

```text
tau_center = sqrt(flicker_min_correlation_time
                  * flicker_max_correlation_time)
```

The result is a finite plateau-like region with roll-off outside the selected
band. More components make the approximation smoother; they do not extend the
physical validity range beyond the configured correlation-time bounds.
Independent process contributions add in Allan variance, while finite records
and random seeds introduce normal Monte Carlo variation around these reference
curves. Use `python3 scripts/ci.py analysis` to regenerate the parity fixtures,
Allan charts, and showcase plots together.
