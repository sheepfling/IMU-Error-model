# Example IMU profiles

These profiles are illustrative best-effort estimates for representative
hardware systems. They are examples only: they are not imported by
`imu_error_model`, are not installed as package data, and are not treated as the
package's default sensor configuration.

Every number in these files is **notional, approximate, and non-official**.
Even when a value is anchored to a datasheet, it is a modeling estimate and
not a manufacturer configuration, certification result, performance guarantee,
or procurement/safety specification. The `baseline` tag means only a project
simulation baseline; it is not a vendor baseline.

The YAML source is intentionally preserved with its comments and links so that
parameter values remain anchored to their datasheet and measurement sources.

The compatibility adapter can execute one of these profiles without rewriting
it:

```python
from imu_error_model import ImuModel, load_yaml_profile

profile = load_yaml_profile("examples/imu_profiles/hardware-estimates/hg9900.yaml")
model = ImuModel(profile.config)
```

`profile.reference_links`, `profile.metadata`, `profile.model_name`, and
`profile.sample_period_s` retain the source anchors needed to trace an estimate
back to its datasheet or measurement record.

Included families include HG9900, HG5700, and HG1700 variants (including the
AG71 reference-backed example), plus ADIS16470,
ICM-42688-P, SBG PULSE 40, iPhone-like, and human vestibular reference
profiles.

The iPhone-like profile is a consumer-MEMS benchmark. The HumanVestibular
profile is a biological perception benchmark with adaptation and threshold
effects; it is not a directly comparable physical IMU and should not be used
as a vendor or procurement model.

Before using a profile in a model, verify its units, parameter interpretation,
datasheet revision, and whether a value is measured data or a modeling
assumption. In particular, `long_term_bias_std` should not be mapped blindly
to every future bias-process parameter.

The repository-level dead-reckoning comparison runs all of these estimates:

```bash
python3 examples/dead_reckoning.py --duration 10
```

It reports final position, velocity, and attitude drift in
`artifacts/dead_reckoning_summary.csv`. This is an analysis example and
validation harness, not a navigation or estimator implementation.

For noise characterization, use:

```bash
python3 scripts/ci.py allan
```

This produces accelerometer and gyroscope Allan-deviation charts and a CSV of
the plotted points for every hardware example and both synthetic test
fixtures. Longer records are needed to make claims about bias correlation
times; the profiles include correlation times from roughly 32 seconds to 3600
seconds. For direct control of the tau range, use
`python3 examples/allan_variance.py` with `--tau-min-s`, `--tau-max-s`, and
`--points`.

For fast Allan-model parity tests, the repository includes
`profiles/test/short-correlation.json`. Its 0.5-second bias correlation time
allows the white-noise region and Gauss–Markov knee to be checked in a
100-second synthetic record.

For the one profile with a published Allan plot, run:

```bash
PYTHONPATH=src python3 examples/adis_allan_overlay.py
```

This writes an overlay PNG and CSV under `artifacts/adis-allan-overlay/`.
The datasheet trace is an approximate digitization retained in
`research/data/`.
