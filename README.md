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

Alternate implementations satisfy `ImuModelProtocol` through structural typing.
They implement `reset()` and:

```python
measure(timestamp, velocity_without_gravity, orientation_world_from_body,
temperature_celsius=None) -> ImuOutput
```

`ImuModelProtocol` is also runtime-checkable with `isinstance`; that check only
confirms that `reset` and `measure` attributes exist. Use Pyright or another
static type checker to validate their signatures.

The first call establishes the interval baseline and returns zero increments.
Subsequent calls require strictly increasing timestamps in seconds. Truth
velocity is a 3-vector in m/s, orientation is a proper `R_world_from_body`
rotation matrix, and `temperature_celsius` is an optional temperature in
degrees Celsius (°C). `None` means temperature is unavailable and disables
thermal corrections for that sample. The default thermal reference is 25 °C,
exposed as `DEFAULT_REFERENCE_TEMPERATURE_C`. `AxisConfig` noise
densities are continuous-time rate densities: accelerometer values use
m/s²/sqrt(Hz), gyroscope values use rad/s/sqrt(Hz). If supplied,
`noise_covariance` is the 3x3 covariance of the corresponding continuous-time
rate noise and must be positive semidefinite.

`ImuOutput.delta_v` is a body-frame velocity increment in m/s, and
`ImuOutput.delta_theta` is a body-frame rotation vector in radians. Both use
the body axes at `start_time`; neither is expressed in world axes. To convert
an increment to world axes, multiply it by the supplied
`R_world_from_body` orientation at the interval start. Each output carries its
sampled interval as `start_time` and `end_time`; `dt` is derived from those
timestamps. The derived `acceleration` and `angular_rate` properties are also
body-frame quantities.

## Checkpointing and resume

`ImuModel` can pause and resume a stochastic run without changing the next
sample. `snapshot()` returns a validated `ImuModelCheckpoint` containing the
configuration, NumPy generator state, reset-level errors, persistent bias and
flicker states, misalignment, and the previous truth sample. Use
`save_checkpoint(path)` for an indented JSON file, or restore in memory with
`ImuModel.from_checkpoint(checkpoint)`:

```python
model.save_checkpoint("run-checkpoint.json")
resumed = ImuModel.load_checkpoint("run-checkpoint.json")
```

The checkpoint is taken between completed `measure()` calls and resumes the
same model implementation. `ImuModel` exposes a format-specific
`checkpoint_codec` whose byte boundary can be used for wire transport:

```python
payload = model.checkpoint_codec.encode(model.snapshot())
checkpoint = model.checkpoint_codec.decode(payload)
resumed = ImuModel.from_checkpoint(checkpoint)
```

The built-in codec uses validated UTF-8 JSON rather than pickle, with an
explicit schema version and model identifier. The generic
`CheckpointableImuModelProtocol` does not require JSON, Pydantic, or disk I/O;
another model can use a different checkpoint type and codec. Models that expose
the codec through `SerializableCheckpointableImuModelProtocol` provide a typed
byte-transport capability. The built-in checkpoint path supports
`LinearThermalModel`; custom thermal models need their own persistence contract.
Checkpoint files are written atomically. `save_checkpoint(..., codec=custom_codec)`
and `load_checkpoint(..., codec=custom_codec)` can use another typed byte codec
that produces and consumes the same `ImuModelCheckpoint` state.
Consumers that only need measurement can continue to depend on
`ImuModelProtocol`.

JSON profiles can be loaded with `load_profile()` or created with
`save_profile()`. The same `load_profile()` helper also loads JSONC, YAML, and YML
profiles based on their extension, returning a validated `ImuConfig` in all
cases. Use `load_profile_document()` when profile provenance metadata is needed.
Test-only JSON profile fixtures are in `tests/profiles/`.

Hardware-oriented best-effort estimates are bundled with the wheel as versioned
package resources. Use `list_example_profiles()` and `load_example_profile()`;
see the [profile examples guide](examples/imu_profiles/README.md) for loading
and interpreting those notional estimates. They are documentation/examples only
and are not package defaults or runtime configuration. Every numeric value is a notional, approximate estimate
derived from public datasheets or other cited source material; none is
authoritative. These profiles are not vendor specifications, certification
results, or performance guarantees. The
`load_example_profile()` maps their profile documents into the current model
while retaining metadata, sample period, a logical package-resource identifier,
and structured source metadata.
Use `list_example_profile_categories()` to discover categories and
`list_example_profiles(None)` to list every packaged profile using fully
qualified, directly loadable names. The default `list_example_profiles()` call
continues to list hardware estimates.
Each hardware profile also stores structured provenance in
`metadata.sources`, with a public URL and the date on which that source was
reviewed. `load_profile_document()` returns this metadata as typed
`ProfileMetadata` and `ProfileSource` models. These dates are provenance review
dates, not asserted publication dates. The links identify the evidence used to choose approximate parameters;
they do not turn the profiles into vendor specifications or certified models.

For deterministic tests and demonstrations, `load_noiseless_profile()` returns
the packaged zero-error baseline. It is intentionally separate from the
hardware-estimate profiles.

The built-in model supports independent white noise, correlated 3-axis noise
covariance, turn-on bias, random-walk/Gauss–Markov bias, a finite-band
flicker-bias approximation, fixed run-level misalignment, scale factor,
nonlinearity, thermal bias/noise coefficients, clipping, and quantization. Scale
factor, turn-on bias standard deviation, in-run bias standard deviation, and
quantization step may be scalar values broadcast to all three axes or explicit
three-axis values. Use `misalignment_covariance` for anisotropic reset
misalignment; `misalignment_std` remains the isotropic shortcut. More
elaborate thermal behavior can be supplied by implementing `ThermalModel`.
Thermal models bind their `AxisConfig` once when constructed and expose
`evaluate(temperature_celsius: float) -> ThermalState`; the per-sample
interface receives only the current temperature in °C. The built-in
`LinearThermalModel(config)` follows this contract.
At model construction, immutable configuration vectors and covariance matrices
are compiled into cached read-only NumPy arrays for reuse on the sampling hot
path; temperature-dependent states and timestep-dependent transitions remain
dynamic.

Configuration loading accepts plain mappings, streams, and extension-selected
files:

```python
config = config_from_mapping({"accelerometer": {"white_noise_density": 0.1}})
config = load_profile_stream(stream, format="yaml")  # text or binary stream
config = load_profile("profile.jsonc")  # .json, .jsonc, .yaml, .yml
```

JSON is the default stream format; YAML and JSONC require the explicit format
when a stream has no filename extension.

For estimator covariance work, `characterize_imu_error_process()` provides a
pure, deterministic description without sampling or mutating an `ImuModel`:

```python
noise = characterize_imu_error_process(
    imu_config,
    dt=imu_output.dt,
    process_age_at_interval_start_s=process_age_s,
    temperature_celsius=imu_output.temperature_celsius,
)
```

The result exposes continuous white-noise covariance, direct increment
covariance, exact persistent-state transitions and process covariance, reset,
start, and end priors, and conditional versus marginal increment covariance. It
also exposes the increment–state-process cross covariance needed by an exact
augmented filter. Use the direct covariance for fresh sensor noise, propagate
the full persistent state when modeling all configured blocks, and preserve the
returned cross covariance when injecting state process noise separately.
The marginal covariance includes the persistent prior and is a correlated-in-time
uncertainty envelope, not fresh independent noise for every sample. The six-dimensional ordering is
`[delta_v_x, delta_v_y, delta_v_z, delta_theta_x, delta_theta_y, delta_theta_z]`.

## Development and CI

Install the development tools and run the same task runner used by GitHub:

```bash
python3 -m pip install -e ".[dev]"
python3 scripts/ci.py test
python3 scripts/ci.py coverage
python3 scripts/ci.py lint
python3 scripts/ci.py format
python3 scripts/ci.py typecheck
python3 scripts/ci.py build
python3 scripts/ci.py package
python3 scripts/ci.py docs --build
```

The documentation task builds both `docs/imu_error_model.tex` and the visual
companion `docs/imu_profile_showcase.tex`. To regenerate the showcase figures
and then build both PDFs in one step, run:

```bash
python3 scripts/build_docs.py
```

The helper assumes the development dependencies and a local LaTeX installation
(`latexmk` or `pdflatex`) are available. Outputs are written under
`docs/artifacts/`.

`scripts/ci.py package` builds a wheel in a temporary directory, verifies that
all committed package data is present in the wheel, installs it into an isolated
target without the source tree on `PYTHONPATH`, and runs the packaged-profile
pytest suite, including the packaged-profile smoke test. `scripts/ci.py all`
includes this check after the regular distribution build.
`scripts/ci.py all` runs Ruff linting, Pyright type checking, Markdown checks,
regression tests, coverage, distribution builds, and the isolated wheel smoke
test.
`scripts/ci.py format` is the canonical auto-fix command: it runs Ruff's
autofixes and formatter, then inserts the repository's required `####` scope
markers. Ruff has no native extension point for project-specific marker rules,
so the marker fixer is intentionally chained after Ruff. It also normalizes
Markdown trailing whitespace, final newlines, heading/list spacing, and blank
lines without rewriting fenced code contents.
`scripts/ci.py markdown` checks that Markdown links are relative and resolve,
that nested Markdown files are linked, and that source-document references do
not enter the public documentation. Markdown and source comments must never
encode links or paths to gitignored source material; use committed, relative
references only. The check rejects local source-material paths and provenance
strings that could disclose excluded documents.
The GitHub workflow calls this runner on Python 3.11 and 3.12 and uploads the
wheel, source distribution, and generated documentation artifacts.

The `Publish to PyPI` workflow publishes a GitHub Release through PyPI Trusted
Publishing. Configure the repository's `pypi` environment and trusted
publisher on PyPI before publishing the first release; no long-lived PyPI
token is stored in the repository.

To regenerate the standard Allan-deviation analysis set, install the development
dependencies and run:

```bash
python3 -m pip install -e ".[dev]"
python3 scripts/ci.py allan
```

This regenerates `artifacts/allan/` for all hardware profiles plus the two
short-correlation test fixtures. Use `--duration` and `--points` to change the
record length or plot density.

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

This combines the standard Allan-deviation plots and all showcase plots. Normal
`pytest` intentionally does not generate files
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
python3 -m pip install -e ".[dev]"
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
  --profile tests/profiles/test/flicker-band.json
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
curve below is the scalar per-axis case. A three-axis `bias_std` applies the
same process independently with its corresponding per-axis variance.

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
